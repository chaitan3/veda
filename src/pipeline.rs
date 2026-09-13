//! End-to-end orchestration: ingestion (source → convert → chunk → embed →
//! store) and querying (retrieve).

use crate::chunk::Chunker;
use crate::config::{ChunkerKind, Config, ExtractPipeline};
use crate::convert;
use crate::embed::image::ImageEmbedder;
use crate::embed::{self, Embedder};
use crate::fingerprint::{self, Fingerprints};
use crate::metrics::{ProcessingMetrics, Timings};
use crate::model::{Chunk, Document, RetrievalMode, Scored, content_hash, stem};
use crate::retrieve::Retriever;
use crate::source;
use crate::store::{self, VectorStore};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Largest embedding batch sent at once.
const EMBED_BATCH: usize = 32;

/// Which ingest stages a run targets. Reprocessing is scoped to the stages
/// whose fingerprint changed; `--force` forces only the targeted ones.
#[derive(Debug, Clone, Copy)]
pub struct StageSet {
    pub extract: bool,
    pub text_embed: bool,
    pub image_embed: bool,
}

impl StageSet {
    /// Every stage (the default when no `--stages` is given).
    pub fn all() -> Self {
        StageSet {
            extract: true,
            text_embed: true,
            image_embed: true,
        }
    }

    pub fn wants(&self, stage: &str) -> bool {
        match stage {
            "extract" => self.extract,
            "text_embed" => self.text_embed,
            "image_embed" => self.image_embed,
            _ => false,
        }
    }

    /// Parse a comma-separated stage list; empty or `all` means every stage.
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.is_empty() || spec.eq_ignore_ascii_case("all") {
            return Ok(Self::all());
        }
        let mut set = StageSet {
            extract: false,
            text_embed: false,
            image_embed: false,
        };
        for part in spec.split(',') {
            match part.trim() {
                "extract" => set.extract = true,
                "text_embed" => set.text_embed = true,
                "image_embed" => set.image_embed = true,
                other => {
                    return Err(Error::config(format!(
                        "unknown stage '{other}' (extract|text_embed|image_embed)"
                    )));
                }
            }
        }
        Ok(set)
    }
}

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
    /// Image→text embedder; `None` when the image channel is off.
    image: Option<Arc<dyn ImageEmbedder>>,
    store: Arc<dyn VectorStore>,
    chunker: Chunker,
    /// Current per-stage fingerprints (compared against each document's rows).
    fingerprints: Fingerprints,
}

impl Pipeline {
    /// Build every component from config.
    pub async fn from_config(cfg: &Config) -> Result<Self> {
        let embedder = embed::from_config(cfg)?;
        let image = embed::image::from_config(cfg)?;
        let store = store::from_config(cfg).await?;
        let chunker = Chunker::from_config(cfg)?;
        let fingerprints = fingerprint::compute(cfg);
        Ok(Pipeline {
            cfg: cfg.clone(),
            embedder,
            image,
            store,
            chunker,
            fingerprints,
        })
    }

    /// The current per-stage fingerprints.
    pub fn fingerprints(&self) -> &Fingerprints {
        &self.fingerprints
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

    /// The image→text embedder, when the image channel is enabled.
    pub fn image_embedder(&self) -> Option<&Arc<dyn ImageEmbedder>> {
        self.image.as_ref()
    }

    /// A retriever over this pipeline's store and embedder.
    pub fn retriever(&self) -> Retriever {
        Retriever::new(self.store.clone(), self.embedder.clone(), self.cfg.rrf_k)
            .with_image(self.image.clone())
    }

    /// Retrieve images most similar to a query image (image→image).
    pub async fn query_image(&self, image: &[u8], k: usize) -> Result<Vec<Scored>> {
        self.retriever().retrieve_image(image, k).await
    }

    /// Retrieve the top `k` chunks for a query under `mode`.
    pub async fn query(&self, mode: RetrievalMode, query: &str, k: usize) -> Result<Vec<Scored>> {
        self.retriever().retrieve(mode, query, k).await
    }

    /// Ingest every document under the configured source folder.
    ///
    /// With `sync`, stored documents whose source files no longer exist are
    /// removed afterwards (mark-and-sweep).
    pub async fn ingest_all(
        &self,
        sync: bool,
        force: bool,
        stages: StageSet,
    ) -> Result<IngestReport> {
        let src = source::from_config(&self.cfg);
        let roots = src.canonical_roots()?;
        let refs = src.list()?;
        let mut report = IngestReport::default();
        let mut current: Vec<String> = Vec::with_capacity(refs.len());

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
            let outcome = self
                .ingest_bytes(&r.uri, &r.name, bytes, force, stages)
                .await;
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

        Ok(report)
    }

    /// Ingest a single document, running only the stages that are stale (or
    /// forced). Unchanged content keeps its document id, so per-document stage
    /// rows and content-addressed image vectors persist across runs.
    pub async fn ingest_bytes(
        &self,
        source_uri: &str,
        name: &str,
        bytes: Vec<u8>,
        force: bool,
        stages: StageSet,
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
        let fp = &self.fingerprints;

        // Unchanged content reuses the existing document (its id, stage rows and
        // content-addressed image vectors); changed content replaces the prior
        // version.
        let doc_id = match self.store.find_document_by_hash(&hash).await? {
            Some(id) => id,
            None => {
                self.store.delete_documents_by_source(source_uri).await?;
                let doc = Document::new(source_uri, stem(name), format!("pending:{hash}"));
                self.store.upsert_document(&doc).await?;
                doc.id
            }
        };

        let force_extract = force && stages.wants("extract");
        let force_text = force && stages.wants("text_embed");
        let force_image = force && stages.wants("image_embed");
        let mut did_work = false;
        let mut n_chunks = 0usize;

        // --- extract: convert + chunk (coupled, since we cache no conversion)
        let has_text = self.store.count_text_chunks_for(&doc_id).await? > 0;
        let want_extract = stages.wants("extract") || (stages.wants("text_embed") && !has_text);
        let extract_fp = self.store.document_stage(&doc_id, "extract").await?;
        let mut extract_ran = false;
        if want_extract
            && (force_extract || extract_fp.as_deref() != Some(fp.extract.as_str()) || !has_text)
        {
            let t = Instant::now();
            let converted = convert::convert(&self.cfg, name, &bytes)?;
            let parse_secs = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let text_chunks = self.chunk_converted(&doc_id, &converted)?;
            let chunk_secs = t.elapsed().as_secs_f64();
            self.store
                .replace_text_chunks(&doc_id, &text_chunks)
                .await?;
            self.store
                .set_document_stage(&doc_id, "extract", &fp.extract)
                .await?;
            self.store
                .clear_document_stage(&doc_id, "text_embed")
                .await?;

            let words = converted.markdown.split_whitespace().count();
            let title = first_heading(&converted.markdown).unwrap_or_else(|| stem(name));
            let metrics = ProcessingMetrics::compute(
                bytes.len() as u64,
                converted.pages,
                words,
                text_chunks.len(),
                words,
                Timings {
                    parse_secs,
                    chunk_secs,
                    embed_secs: 0.0,
                },
            );
            let mut doc = self
                .store
                .get_document(&doc_id)
                .await?
                .ok_or_else(|| Error::store("document vanished mid-ingest"))?;
            doc.title = title;
            doc.hash = hash.clone();
            doc.metadata = serde_json::json!({
                "source": source_uri,
                "metrics": metrics,
                "markdown": converted.markdown,
            });
            self.store.upsert_document(&doc).await?;

            extract_ran = true;
            did_work = true;
            n_chunks += text_chunks.len();
        }

        // --- text_embed: (re)embed the stored text chunks in place
        let text_fp = self.store.document_stage(&doc_id, "text_embed").await?;
        let run_text = (stages.wants("text_embed") || extract_ran)
            && (extract_ran || force_text || text_fp.as_deref() != Some(fp.text_embed.as_str()));
        if run_text {
            let chunks = self.store.text_chunks(&doc_id).await?;
            let mut pairs: Vec<(String, Vec<f32>)> = Vec::with_capacity(chunks.len());
            for batch in chunks.chunks(EMBED_BATCH) {
                let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
                let vectors = self.embedder.embed(&texts).await?;
                if vectors.len() != batch.len() {
                    return Err(Error::embedding(format!(
                        "embedding count mismatch: sent {}, got {}",
                        batch.len(),
                        vectors.len()
                    )));
                }
                for (chunk, vector) in batch.iter().zip(vectors) {
                    pairs.push((chunk.id.clone(), vector));
                }
            }
            self.store.set_text_vectors(&pairs).await?;
            self.store
                .set_document_stage(&doc_id, "text_embed", &fp.text_embed)
                .await?;
            did_work = true;
        }

        // --- image_embed: one content-addressed image chunk for an image file
        if let Some(embedder) = &self.image
            && is_image_name(name)
        {
            let image_fp = self.store.document_stage(&doc_id, "image_embed").await?;
            if stages.wants("image_embed")
                && (force_image || image_fp.as_deref() != Some(fp.image_embed.as_str()))
            {
                let mut vectors = embedder.embed_images(std::slice::from_ref(&bytes)).await?;
                let vector = vectors
                    .pop()
                    .ok_or_else(|| Error::embedding("image embedder returned no vector"))?;
                // Content-addressed id: stable across reprocessing, so the
                // image vector is reused rather than recomputed on text changes.
                let mut chunk = Chunk::new(&doc_id, 0, "", 0);
                chunk.id = hash.clone();
                chunk.kind = crate::model::ChunkKind::Image;
                chunk.ordinal = self.store.count_text_chunks_for(&doc_id).await? as i64;
                chunk.metadata = serde_json::json!({ "mime": image_mime(name) });
                self.store.upsert_image_chunk(&chunk, &vector).await?;
                self.store
                    .set_document_stage(&doc_id, "image_embed", &fp.image_embed)
                    .await?;
                did_work = true;
                n_chunks += 1;
            }
        }

        if !did_work {
            tracing::info!(name = name, "up to date; skipping");
            return Ok(IngestOutcome::Skipped);
        }

        self.store.bump_corpus_version().await?;
        tracing::info!(
            name = name,
            uri = source_uri,
            chunks = n_chunks,
            secs = started.elapsed().as_secs_f64(),
            "ingested"
        );
        Ok(IngestOutcome::Ingested(n_chunks))
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

/// Whether a file name is a standalone image we index in the image space.
fn is_image_name(name: &str) -> bool {
    matches!(
        file_type(name).as_str(),
        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "webp" | "gif"
    )
}

/// MIME type for a standalone image by extension.
fn image_mime(name: &str) -> &'static str {
    match file_type(name).as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "tif" | "tiff" => "image/tiff",
        "bmp" => "image/bmp",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "application/octet-stream",
    }
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

    #[test]
    fn stage_set_parsing() {
        let all = StageSet::all();
        assert!(all.extract && all.text_embed && all.image_embed);

        let s = StageSet::parse("text_embed").unwrap();
        assert!(!s.extract && s.text_embed && !s.image_embed);

        let s = StageSet::parse("extract,image_embed").unwrap();
        assert!(s.extract && !s.text_embed && s.image_embed);

        // Empty and "all" both mean every stage.
        assert!(StageSet::parse("").unwrap().extract);
        assert!(StageSet::parse("all").unwrap().image_embed);

        assert!(StageSet::parse("nope").is_err());
    }
}
