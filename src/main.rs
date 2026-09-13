//! `local-search` CLI: ingest and query a local document store, and serve the
//! read-only search API.

use clap::{Parser, Subcommand};
use local_search::api::AppState;
use local_search::{Config, Pipeline, RetrievalMode};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "local-search",
    version,
    about = "Text + image search over a local document store"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create the database schema.
    InitDb,
    /// Ingest (or update) the source folder.
    Ingest {
        /// One or more folders to ingest; overrides LS_SOURCE_PATH.
        paths: Vec<String>,
        /// Remove stored documents whose source files no longer exist
        /// (scoped to the roots given on this command line).
        #[arg(long)]
        sync: bool,
        /// Reprocess every document, ignoring the content-hash dedup.
        #[arg(long)]
        force: bool,
    },
    /// Run a search query.
    Query {
        /// The query text.
        query: String,
        /// Retrieval mode: vector | bm25 | hybrid.
        #[arg(long, default_value = "hybrid")]
        mode: String,
        /// Number of results.
        #[arg(short, long, default_value_t = 5)]
        k: usize,
    },
    /// Show store statistics.
    Stats,
    /// Remove documents left with no chunks.
    Prune,
    /// Download the default local model set (docling assets + ONNX embedder).
    DownloadModels {
        /// Skip the local ONNX text embedder (bge-small-en-v1.5).
        #[arg(long)]
        no_embed: bool,
    },
    /// Serve the read-only search API.
    Serve,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let mut cfg = Config::from_env()?;
    // Fan LS_MODELS_DIR out to docling's own per-model variables before any
    // model is loaded (startup-time, single-threaded).
    cfg.apply_docling_env();
    // Report the ONNX execution provider (CUDA vs CPU fallback).
    local_search::runtime::log_execution_provider();

    match cli.command {
        Command::InitDb => {
            let _ = local_search::store::from_config(&cfg).await?;
            println!("initialized database: {}", cfg.db_path);
        }

        Command::Ingest { paths, sync, force } => {
            if !paths.is_empty() {
                cfg.source_paths = paths;
            }
            let pipeline = Pipeline::from_config(&cfg).await?;
            println!("source:     {}", cfg.source_paths.join(", "));
            println!(
                "extraction: {:?}  chunker: {:?}  embedder: {}",
                cfg.extract,
                cfg.chunker,
                pipeline.embedder().id()
            );

            let started = std::time::Instant::now();
            let report = pipeline.ingest_all(sync, force).await?;
            let wall = started.elapsed().as_secs_f64();

            println!(
                "\ningested {}  skipped {}  failed {}  deleted {}  chunks +{}",
                report.documents_ingested,
                report.documents_skipped,
                report.documents_failed,
                report.documents_deleted,
                report.chunks_added
            );

            println!("\nsummary by type:");
            println!(
                "  {:<9} {:>6} {:>7} {:>6} {:>6} {:>10}",
                "type", "files", "chunks", "skip", "fail", "time"
            );
            let (mut files, mut chunks, mut skip, mut fail, mut secs) =
                (0usize, 0usize, 0usize, 0usize, 0.0f64);
            for (ext, s) in &report.by_type {
                println!(
                    "  {:<9} {:>6} {:>7} {:>6} {:>6} {:>9.2}s",
                    ext, s.ingested, s.chunks, s.skipped, s.failed, s.secs
                );
                files += s.ingested;
                chunks += s.chunks;
                skip += s.skipped;
                fail += s.failed;
                secs += s.secs;
            }
            println!(
                "  {:<9} {:>6} {:>7} {:>6} {:>6} {:>9.2}s",
                "total", files, chunks, skip, fail, secs
            );
            println!("  wall time: {wall:.2}s");

            if report.documents_failed > 0 {
                std::process::exit(2);
            }
        }

        Command::Query { query, mode, k } => {
            let mode: RetrievalMode = mode.parse()?;
            let pipeline = Pipeline::from_config(&cfg).await?;
            let hits = pipeline.query(mode, &query, k).await?;
            if hits.is_empty() {
                println!("no results");
            }
            for (i, hit) in hits.iter().enumerate() {
                let snippet: String = hit.chunk.text.chars().take(240).collect();
                let page = hit
                    .chunk
                    .locator
                    .as_ref()
                    .and_then(|l| l.page)
                    .map(|p| format!(" p.{p}"))
                    .unwrap_or_default();
                println!(
                    "{:>2}. [{:.4}] {}{}  ({})",
                    i + 1,
                    hit.score,
                    hit.chunk.kind.as_str(),
                    page,
                    hit.chunk.doc_id
                );
                println!("    {snippet}");
            }
        }

        Command::Stats => {
            let pipeline = Pipeline::from_config(&cfg).await?;
            let store = pipeline.store();
            println!("database:       {}", cfg.db_path);
            println!("source (cfg):   {}", cfg.source_paths.join(", "));
            println!("embedder:       {}", pipeline.embedder().id());
            println!("embed dim:      {}", pipeline.embedder().dim());
            println!("extract:        {:?}", cfg.extract);
            println!("documents:      {}", store.count_documents().await?);
            println!("chunks:         {}", store.count_chunks().await?);
            println!("corpus version: {}", store.corpus_version().await?);
            println!("pipeline ver.:  {}", store.pipeline_version().await?);
        }

        Command::Prune => {
            let pipeline = Pipeline::from_config(&cfg).await?;
            let removed = pipeline.prune().await?;
            println!("removed {removed} document(s) with no chunks");
        }

        Command::DownloadModels { no_embed } => {
            local_search::download::download_models(&cfg, !no_embed).await?;
        }

        Command::Serve => {
            let pipeline = Arc::new(Pipeline::from_config(&cfg).await?);
            if !cfg.auth_enabled() {
                tracing::warn!("LS_API_KEYS is empty — API authentication is disabled");
            }
            let state = AppState {
                pipeline,
                api_keys: Arc::new(cfg.api_keys.clone()),
            };
            let app = local_search::api::router(state);
            let listener = tokio::net::TcpListener::bind(&cfg.addr).await?;
            println!("local-search listening on http://{}", cfg.addr);
            axum::serve(listener, app).await?;
        }
    }

    Ok(())
}
