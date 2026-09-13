//! End-to-end orchestration: ingestion (source → convert → chunk → embed →
//! store) and querying (retrieve).

use crate::chunk::Chunker;
use crate::config::{ChunkerKind, Config, ExtractPipeline};
use crate::convert;
use crate::embed::{self, Embedder};
use crate::metrics::{ProcessingMetrics, Timings};
use crate::model::{Document, RetrievalMode, Scored, content_hash, stem};
use crate::retrieve::Retriever;
use crate::source;
use crate::store::{self, VectorStore};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Largest embedding batch sent at once.
const EMBED_BATCH: usize = 32;

/// Bump when the ingest pipeline changes what it produces (new extraction,
/// chunking, or embedding channels). Ingest reprocesses the whole corpus when
/// the stored version differs, so adding Phase 1's image/caption channels
/// upgrades an existing database without a reset.
pub const PIPELINE_VERSION: u32 = 1;

/// What happened to one document during ingestion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestOutcome {
    /// Ingested; carries the number of chunks stored.
    Ingested(usize),
    /// Skipped because identical content (same hash) is already stored.
    Skipped,
}

/// Per-file-type ingestion statistics.
#[derive(Debug, Clone, Default)]
pub struct TypeStats {
    pub ingested: usize,
    pub skipped: usize,
    pub failed: usize,
    pub chunks: usize,
    pub secs: f64,
}

/// Aggregate ingestion result over a whole source.
#[derive(Debug, Clone, Default)]
pub struct IngestReport {
    pub documents_ingested: usize,
    pub documents_skipped: usize,
    pub documents_failed: usize,
    pub chunks_added: usize,
    pub documents_deleted: usize,
    /// Counts and wall time grouped by file extension.
    pub by_type: BTreeMap<String, TypeStats>,
}

/// A fully-wired pipeline built from a [`Config`].
#[derive(Clone)]
pub struct Pipeline {
    cfg: Config,
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
    chunker: Chunker,
}

impl Pipeline {
    /// Build every component from config.
    pub async fn from_config(cfg: &Config) -> Result<Self> {
        let embedder = embed::from_config(cfg)?;
        let store = store::from_config(cfg).await?;
        let chunker = Chunker::from_config(cfg)?;
        Ok(Pipeline {
            cfg: cfg.clone(),
            embedder,
            store,
            chunker,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn store(&self) -> &Arc<dyn VectorStore> {
        &self.store
    }

    pub fn embedder(&self) -> &Arc<dyn Embedder> {
        &self.embedder
    }

    /// A retriever over this pipeline's store and embedder.
    pub fn retriever(&self) -> Retriever {
        Retriever::new(self.store.clone(), self.embedder.clone(), self.cfg.rrf_k)
    }

    /// Retrieve the top `k` chunks for a query under `mode`.
    pub async fn query(&self, mode: RetrievalMode, query: &str, k: usize) -> Result<Vec<Scored>> {
        self.retriever().retrieve(mode, query, k).await
    }

    /// Ingest every document under the configured source folder.
    ///
    /// With `sync`, stored documents whose source files no longer exist are
    /// removed afterwards (mark-and-sweep).
    pub async fn ingest_all(&self, sync: bool, force: bool) -> Result<IngestReport> {
        let src = source::from_config(&self.cfg);
        let roots = src.canonical_roots()?;
        let refs = src.list()?;
        let mut report = IngestReport::default();
        let mut current: Vec<String> = Vec::with_capacity(refs.len());

        // A pipeline change (e.g. Phase 1 adding image/caption channels) makes
        // every stored document stale, so ignore content-hash dedup for this run.
        let stored_version = self.store.pipeline_version().await?;
        if stored_version != PIPELINE_VERSION {
            tracing::info!(
                from = stored_version,
                to = PIPELINE_VERSION,
                "ingest pipeline changed; reprocessing all documents"
            );
        }
        let force = force || stored_version != PIPELINE_VERSION;

        for r in &refs {
            current.push(r.uri.clone());
            let kind = file_type(&r.name);
            let started = Instant::now();
            let bytes = match std::fs::read(&r.path) {
                Ok(b) => b,
                Err(e) => {
                    report.documents_failed += 1;
                    let stats = report.by_type.entry(kind).or_default();
                    stats.failed += 1;
                    stats.secs += started.elapsed().as_secs_f64();
                    tracing::warn!(uri = %r.uri, error = %e, "cannot read file");
                    continue;
                }
            };
            let outcome = self.ingest_bytes(&r.uri, &r.name, bytes, force).await;
            let secs = started.elapsed().as_secs_f64();
            let stats = report.by_type.entry(kind).or_default();
            stats.secs += secs;
            match outcome {
                Ok(IngestOutcome::Ingested(n)) => {
                    report.documents_ingested += 1;
                    report.chunks_added += n;
                    stats.ingested += 1;
                    stats.chunks += n;
                }
                Ok(IngestOutcome::Skipped) => {
                    report.documents_skipped += 1;
                    stats.skipped += 1;
                }
                Err(e) => {
                    report.documents_failed += 1;
                    stats.failed += 1;
                    tracing::warn!(uri = %r.uri, error = %e, "failed to ingest");
                }
            }
        }

        if sync {
            let stored = self.store.all_source_uris().await?;
            let keep: std::collections::HashSet<&String> = current.iter().collect();
            for uri in stored {
                // Only sweep documents inside the roots we just ingested, so
                // ingesting one root never deletes another root's documents.
                if !keep.contains(&uri) && source::uri_under_roots(&uri, &roots) {
                    self.store.delete_documents_by_source(&uri).await?;
                    report.documents_deleted += 1;
                    tracing::info!(uri = %uri, "removed (source file gone)");
                }
            }
            self.store.bump_corpus_version().await?;
        }

        self.store.set_pipeline_version(PIPELINE_VERSION).await?;
        Ok(report)
    }

    /// Ingest a single document from in-memory bytes.
    ///
    /// Conversion, chunking and embedding all run before any database mutation,
    /// so a failed conversion leaves the store untouched.
    pub async fn ingest_bytes(
        &self,
        source_uri: &str,
        name: &str,
        bytes: Vec<u8>,
        force: bool,
    ) -> Result<IngestOutcome> {
        let started = Instant::now();
        let method = self.method_label(name);
        let chunker = format!("{:?}", self.chunker.kind()).to_lowercase();
        tracing::info!(
            name = name,
            uri = source_uri,
            bytes = bytes.len(),
            method = %method,
            chunker = %chunker,
            "ingesting"
        );

        let hash = content_hash(&bytes);
        if !force && self.store.find_document_by_hash(&hash).await?.is_some() {
            tracing::info!(name = name, "unchanged; skipping");
            return Ok(IngestOutcome::Skipped);
        }
        let file_bytes = bytes.len() as u64;

        // The document row is written with a *pending* hash first and the real
        // hash only after every chunk is stored, so an interrupted ingest can
        // never satisfy the dedup check above.
        let mut doc = Document::new(source_uri, stem(name), format!("pending:{hash}"));

        // --- conversion
        let t = Instant::now();
        let converted = convert::convert(&self.cfg, name, &bytes)?;
        let parse_secs = t.elapsed().as_secs_f64();

        // --- chunking
        let t = Instant::now();
        let mut chunks = self.chunk_converted(&doc.id, &converted)?;
        let chunk_secs = t.elapsed().as_secs_f64();

        // --- embedding
        let t = Instant::now();
        let mut embedded_words = 0usize;
        for batch in chunks.chunks_mut(EMBED_BATCH) {
            let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
            let vectors = self.embedder.embed(&texts).await?;
            if vectors.len() != batch.len() {
                return Err(Error::embedding(format!(
                    "embedding count mismatch: sent {}, got {}",
                    batch.len(),
                    vectors.len()
                )));
            }
            for (chunk, vector) in batch.iter_mut().zip(vectors) {
                embedded_words += chunk.text.split_whitespace().count();
                chunk.embedding = Some(vector);
            }
        }
        let embed_secs = t.elapsed().as_secs_f64();

        // --- persist (replace any prior version of this source)
        let words = converted.markdown.split_whitespace().count();
        let title = first_heading(&converted.markdown).unwrap_or_else(|| stem(name));
        let metrics = ProcessingMetrics::compute(
            file_bytes,
            converted.pages,
            words,
            chunks.len(),
            embedded_words,
            Timings {
                parse_secs,
                chunk_secs,
                embed_secs,
            },
        );
        self.store.delete_documents_by_source(source_uri).await?;
        self.store.upsert_document(&doc).await?;
        let n = chunks.len();
        self.store.insert_chunks(&chunks).await?;

        doc.title = title;
        doc.hash = hash;
        doc.metadata = serde_json::json!({
            "source": source_uri,
            "metrics": metrics,
            "markdown": converted.markdown,
        });
        self.store.upsert_document(&doc).await?;
        self.store.bump_corpus_version().await?;

        tracing::info!(
            name = name,
            uri = source_uri,
            pages = ?converted.pages,
            words,
            chunks = n,
            parse_wps = metrics.parse_words_per_sec(),
            embed_wps = metrics.embed_words_per_sec(),
            secs = started.elapsed().as_secs_f64(),
            "ingested"
        );
        Ok(IngestOutcome::Ingested(n))
    }

    /// Human-readable extraction method for a file (text passthrough vs a
    /// docling pipeline).
    fn method_label(&self, name: &str) -> String {
        match file_type(name).as_str() {
            "md" | "markdown" | "txt" | "text" => "text".to_string(),
            _ => match self.cfg.extract {
                ExtractPipeline::Onnx => "docling-onnx".to_string(),
                ExtractPipeline::Vlm => "docling-vlm".to_string(),
            },
        }
    }

    /// Chunk the converted document, preferring docling's structure-aware
    /// chunkers when configured and the document tree is available.
    fn chunk_converted(
        &self,
        doc_id: &str,
        converted: &convert::Converted,
    ) -> Result<Vec<crate::model::Chunk>> {
        if self.chunker.kind() != ChunkerKind::Window {
            #[cfg(feature = "docling-extract")]
            if let Some(doc) = &converted.document {
                return self.chunker.chunk_document(doc_id, doc);
            }
        }
        self.chunker.chunk_markdown(doc_id, &converted.markdown)
    }

    /// Remove documents left with no chunks (interrupted ingests).
    pub async fn prune(&self) -> Result<usize> {
        let docs = self.store.list_documents().await?;
        let mut removed = 0;
        for d in docs {
            if self.store.count_chunks_for(&d.id).await? == 0 {
                self.store.delete_document(&d.id).await?;
                removed += 1;
            }
        }
        if removed > 0 {
            self.store.bump_corpus_version().await?;
        }
        Ok(removed)
    }
}

/// Lowercase file extension of a name, or `"other"`.
pub fn file_type(name: &str) -> String {
    std::path::Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| "other".to_string())
}

/// First `# `/`## ` heading text in a Markdown string.
pub fn first_heading(md: &str) -> Option<String> {
    for line in md.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix('#') {
            let heading = rest.trim_start_matches('#').trim();
            if !heading.is_empty() {
                return Some(heading.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_first_heading() {
        assert_eq!(
            first_heading("intro\n# Real Title\nbody"),
            Some("Real Title".into())
        );
        assert_eq!(first_heading("no headings here"), None);
    }
}
