//! `local-search` CLI: ingest and query a local document store, and serve the
//! read-only search API.

use clap::{Parser, Subcommand};
use local_search::api::AppState;
use local_search::pipeline::StageSet;
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
        /// Force the targeted stages to re-run, ignoring their fingerprints.
        #[arg(long)]
        force: bool,
        /// Comma-separated stages to run (default: all):
        /// extract | text_embed | image_embed.
        #[arg(long, value_name = "STAGES")]
        stages: Option<String>,
    },
    /// Run a search query.
    Query {
        /// The query text.
        query: String,
        /// Retrieval mode: hybrid | vector | bm25 | image.
        #[arg(long, default_value = "hybrid")]
        mode: String,
        /// Number of results (1-100).
        #[arg(short = 'k', long, default_value_t = 5)]
        limit: usize,
    },
    /// Score a gold set (JSONL) against the current store.
    Eval {
        /// Path to the gold-set JSONL.
        #[arg(long, value_name = "FILE")]
        queries: String,
        /// Comma-separated retrieval modes (default: hybrid,vector,bm25).
        #[arg(long, default_value = "hybrid,vector,bm25")]
        modes: String,
        /// Comma-separated cutoffs for recall/nDCG.
        #[arg(long, default_value = "1,5,10")]
        k: String,
        /// Emit the report as JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Mine a synthetic gold set from the corpus (see docs/EVAL.md).
    EvalGen {
        /// Where to write the JSONL.
        #[arg(long, value_name = "FILE", default_value = "eval/queries.jsonl")]
        out: String,
        /// Phrase queries per document.
        #[arg(long, default_value_t = 3)]
        per_doc: usize,
    },
    /// Show store statistics.
    Stats,
    /// Remove documents left with no chunks.
    Prune,
    /// Download the default model set (docling assets, text + image embedders,
    /// code/formula).
    DownloadModels {
        /// Skip the local text embedders (bge-small-en-v1.5 and the
        /// EmbeddingGemma 2 text graph).
        #[arg(long)]
        no_embed: bool,
        /// Skip the image embedders (SigLIP2 towers ~700 MB and the
        /// EmbeddingGemma 2 vision encoder).
        #[arg(long)]
        no_image: bool,
        /// Skip the CodeFormulaV2 code/formula models (~0.8 GB).
        #[arg(long)]
        no_enrich: bool,
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
    // On a GPU, CodeFormulaV2's fp32 models do not fit alongside layout / OCR /
    // TableFormer on a modest card (docling prefers fp32 on CUDA), so default
    // code/formula enrichment off unless the operator asked for it.
    if std::env::var_os("LS_EXTRACT_CODE_FORMULA").is_none()
        && local_search::runtime::gpu_selected()
    {
        cfg.code_formula = false;
        tracing::info!(
            "GPU execution selected; defaulting code/formula enrichment off \
             (set LS_EXTRACT_CODE_FORMULA=true to override)"
        );
    }

    match cli.command {
        Command::InitDb => {
            let _ = local_search::store::from_config(&cfg).await?;
            println!("initialized database: {}", cfg.db_path);
        }

        Command::Ingest {
            paths,
            sync,
            force,
            stages,
        } => {
            if !paths.is_empty() {
                cfg.source_paths = paths;
            }
            let stages = match stages {
                Some(spec) => StageSet::parse(&spec)?,
                None => StageSet::all(),
            };
            let pipeline = Pipeline::from_config(&cfg).await?;
            println!("source:     {}", cfg.source_paths.join(", "));
            println!(
                "extraction: {:?}  chunker: {:?}  embedder: {}",
                cfg.extract,
                cfg.chunker,
                pipeline.embedder().id()
            );

            let started = std::time::Instant::now();
            let report = pipeline.ingest_all(sync, force, stages).await?;
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

        Command::Query { query, mode, limit } => {
            let mode: RetrievalMode = mode.parse()?;
            let pipeline = Pipeline::from_config(&cfg).await?;
            let hits = pipeline.query(mode, &query, limit).await?;
            if hits.is_empty() {
                println!("no results");
            }
            for (i, hit) in hits.iter().enumerate() {
                let snippet: String = hit.chunk.text.chars().take(240).collect();
                println!(
                    "{:>2}. [{:.4}] {}  ({})",
                    i + 1,
                    hit.score,
                    hit.chunk.kind.as_str(),
                    hit.chunk.doc_id
                );
                if snippet.is_empty() && hit.chunk.kind == local_search::model::ChunkKind::Image {
                    // Image chunks have no text: name the source file instead.
                    let doc = pipeline.store().get_document(&hit.chunk.doc_id).await?;
                    let name = doc
                        .map(|d| {
                            d.source_uri
                                .rsplit('/')
                                .next()
                                .unwrap_or(&d.source_uri)
                                .to_string()
                        })
                        .unwrap_or_else(|| hit.chunk.id.chars().take(12).collect());
                    println!("    [image] {name}");
                } else {
                    println!("    {snippet}");
                }
            }
        }

        Command::Eval {
            queries,
            modes,
            k,
            json,
        } => {
            let gold = local_search::eval::load(&queries)?;
            let modes = local_search::eval::parse_modes(&modes)?;
            let cutoffs = local_search::eval::parse_cutoffs(&k)?;
            let pipeline = Pipeline::from_config(&cfg).await?;
            println!(
                "gold:   {} queries from {queries}\nmodes:  {modes:?}\ncutoff: {cutoffs:?}",
                gold.len()
            );
            let resolved = local_search::eval::resolve(&pipeline, &gold).await?;
            let reports = local_search::eval::run(&pipeline, &resolved, &modes, &cutoffs).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&reports)?);
            } else {
                local_search::eval::print_reports(&reports, &cutoffs);
            }
        }

        Command::EvalGen { out, per_doc } => {
            let pipeline = Pipeline::from_config(&cfg).await?;
            if let Some(parent) = std::path::Path::new(&out).parent() {
                std::fs::create_dir_all(parent)?;
            }
            let n = local_search::eval::generate(&pipeline, per_doc, &out).await?;
            println!("wrote {n} queries to {out}");
        }

        Command::Stats => {
            let pipeline = Pipeline::from_config(&cfg).await?;
            let store = pipeline.store();
            println!("database:       {}", cfg.db_path);
            println!("source (cfg):   {}", cfg.source_paths.join(", "));
            println!("embedder:       {}", pipeline.embedder().id());
            println!("embed dim:      {}", pipeline.embedder().dim());
            println!(
                "image space:    {}",
                match pipeline.store().image_space() {
                    local_search::store::ImageSpace::Off => "off".to_string(),
                    local_search::store::ImageSpace::Split(d) => format!("split ({d}-d)"),
                    local_search::store::ImageSpace::Unified =>
                        "unified (text + images in one index)".to_string(),
                }
            );
            match pipeline.image_embedder() {
                Some(image) => {
                    println!("image embedder: {}", image.id());
                    println!("image dim:      {}", image.dim());
                }
                None => println!("image embedder: off"),
            }
            println!("extract:        {:?}", cfg.extract);
            println!("documents:      {}", store.count_documents().await?);
            println!("chunks:         {}", store.count_chunks().await?);
            println!("corpus version: {}", store.corpus_version().await?);
            let fp = pipeline.fingerprints();
            println!(
                "fingerprints:   extract={} text_embed={} image_embed={}",
                fp.extract, fp.text_embed, fp.image_embed
            );
        }

        Command::Prune => {
            let pipeline = Pipeline::from_config(&cfg).await?;
            let removed = pipeline.prune().await?;
            println!("removed {removed} document(s) with no chunks");
        }

        Command::DownloadModels {
            no_embed,
            no_image,
            no_enrich,
        } => {
            local_search::download::download_models(&cfg, !no_embed, !no_image, !no_enrich).await?;
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
            let ingest_pipeline = state.pipeline.clone();
            let app = local_search::api::router(state);
            let listener = tokio::net::TcpListener::bind(&cfg.addr).await?;
            println!("local-search listening on http://{}", cfg.addr);
            // Periodic ingest (LS_INGEST_INTERVAL): starts after the listener
            // is bound, so the API is responsive while the first catch-up run
            // works through the corpus.
            if let Some(interval) = cfg.ingest_interval {
                tracing::info!(
                    secs = interval.as_secs(),
                    sync = cfg.ingest_sync,
                    sources = %cfg.source_paths.join(", "),
                    "scheduled ingest enabled (first run now, then every interval)"
                );
                local_search::schedule::spawn(ingest_pipeline, interval, cfg.ingest_sync);
            }
            axum::serve(listener, app).await?;
        }
    }

    Ok(())
}
