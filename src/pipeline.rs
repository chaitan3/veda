//! End-to-end orchestration: ingestion (source → convert → chunk → embed →
//! store) and querying (retrieve).

use crate::chunk::Chunker;
use crate::config::{ChunkerKind, Config, EmbedProvider, ExtractPipeline};
use crate::convert;
use crate::embed::image::ImageEmbedder;
use crate::embed::{self, Embedder, idle};
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

/// `(size, mtime_ns)` for a path, or `None` if it cannot be stat'd. Used as the
/// cheap change gate: when both match the recorded values, the stored content
/// hash is trusted and the file is not read again.
fn file_stat(path: &std::path::Path) -> Option<(u64, i64)> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0);
    Some((meta.len(), mtime_ns))
}

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
    /// Documents that failed during *this* run (exit code 2 when > 0).
    pub documents_failed: usize,
    /// Documents whose previous failure was remembered and skipped (not retried).
    pub documents_failed_known: usize,
    pub chunks_added: usize,
    pub documents_deleted: usize,
    /// Counts and wall time grouped by file extension.
    pub by_type: BTreeMap<String, TypeStats>,
}

/// A fully-wired pipeline built from a [`Config`].
#[derive(Clone)]
pub struct Pipeline {
    cfg: Config,
    embedder: Arc<idle::IdleEmbedder>,
    /// Image→text embedder; `None` when the image channel is off.
    image: Option<Arc<idle::IdleImageEmbedder>>,
    store: Arc<dyn VectorStore>,
    chunker: Chunker,
    /// Current per-stage fingerprints (compared against each document's rows).
    fingerprints: Fingerprints,
    /// Identity of the conversion path (build + docling models + pdfium):
    /// gates whether a failed document is retried.
    extractor_id: String,
}

impl Pipeline {
    /// Build every component from config.
    pub async fn from_config(cfg: &Config) -> Result<Self> {
        // The `hybrid` budget (`LS_CHUNK_SIZE`) and the `window` word count are
        // both honored by the chunkers — but the ONNX embedder has a hard
        // position-embedding window (`LS_EMBED_MAX_TOKENS`, 512 for BGE-style
        // encoders). A chunk budget larger than that window is silently
        // truncated at embed time, so say so at startup instead.
        if matches!(cfg.embed.provider, EmbedProvider::Onnx | EmbedProvider::Eg2)
            && matches!(cfg.chunker, ChunkerKind::Hybrid | ChunkerKind::Window)
            && cfg.chunk_size + 2 > cfg.embed.max_tokens
        {
            if cfg.chunk_strict {
                return Err(crate::Error::config(format!(
                    "LS_CHUNK_SIZE={} exceeds the embedder's sequence window \
                     (LS_EMBED_MAX_TOKENS={}); lower it to {} or unset \
                     LS_CHUNK_STRICT to allow truncation",
                    cfg.chunk_size,
                    cfg.embed.max_tokens,
                    cfg.embed.max_tokens.saturating_sub(2)
                )));
            }
            tracing::warn!(
                chunk_size = cfg.chunk_size,
                max_tokens = cfg.embed.max_tokens,
                "LS_CHUNK_SIZE exceeds the embedder's sequence window; \
                 over-long chunks are truncated before embedding \
                 (LS_CHUNK_SIZE <= {} avoids truncation)",
                cfg.embed.max_tokens.saturating_sub(2)
            );
        }
        let embedder = embed::from_config(cfg)?;
        let image = embed::image::from_config(cfg)?;
        let store = store::from_config(cfg).await?;
        let chunker = Chunker::from_config(cfg)?;
        let fingerprints = fingerprint::compute(cfg);
        // `serve` releases the models after this idle window; the CLI does not
        // spawn a reaper, so it keeps them for its (short) lifetime.
        let idle = idle::IdleWindow::new(cfg.embed_idle);
        let embedder = Arc::new(idle::IdleEmbedder::new(cfg, embedder, idle.clone()));
        let image = image.map(|e| Arc::new(idle::IdleImageEmbedder::new(cfg, e, idle.clone())));
        Ok(Pipeline {
            cfg: cfg.clone(),
            embedder,
            image,
            store,
            chunker,
            fingerprints,
            extractor_id: fingerprint::extractor_identity(cfg),
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

    pub fn embedder(&self) -> &Arc<idle::IdleEmbedder> {
        &self.embedder
    }

    /// The image→text embedder, when the image channel is enabled.
    pub fn image_embedder(&self) -> Option<&Arc<idle::IdleImageEmbedder>> {
        self.image.as_ref()
    }

    /// Release the embedding models if the shared idle window has elapsed.
    /// Called periodically by `serve`; a no-op for the CLI.
    pub async fn reap_idle(&self) {
        self.embedder.reap().await;
        if let Some(image) = &self.image {
            image.reap().await;
        }
    }

    /// A retriever over this pipeline's store and embedder.
    pub fn retriever(&self) -> Retriever {
        let image: Option<Arc<dyn ImageEmbedder>> = self
            .image
            .as_ref()
            .map(|e| e.clone() as Arc<dyn ImageEmbedder>);
        Retriever::new(
            self.store.clone(),
            self.embedder.clone() as Arc<dyn Embedder>,
            self.cfg.rrf_k,
        )
        .with_image(image)
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
        // Folders whose image files skip the docling/OCR/text path entirely.
        let policy_roots = source::image_only_roots(&self.cfg);
        let mut report = IngestReport::default();
        let mut current: Vec<String> = Vec::with_capacity(refs.len());

        for r in &refs {
            current.push(r.uri.clone());
            let image_only = source::under_image_only(&policy_roots, &r.path);
            let kind = if image_only && is_image_name(&r.name) {
                format!("{}(image-only)", file_type(&r.name))
            } else {
                file_type(&r.name)
            };
            let started = Instant::now();
            // Cheap gate: a size+mtime match means the stored hash is trusted
            // and the file is not read (nor hashed) again.
            let stat = file_stat(&r.path);
            let known_hash = match stat {
                Some((size, mtime_ns)) => self
                    .store
                    .source_file(&r.uri)
                    .await?
                    .filter(|f| f.size == size && f.mtime_ns == mtime_ns)
                    .map(|f| f.hash),
                None => None,
            };
            let (size, mtime_ns) = stat.unwrap_or((0, 0));

            // A remembered failure is not retried unless the file changed or
            // the extractor identity changed (a rebuilt container with a fixed
            // pdfium or a better OCR model).
            let mut retry_extract = false;
            if let Some(f) = self.store.document_failure(&r.uri).await? {
                let unchanged = f.size == size && f.mtime_ns == mtime_ns;
                if !force && unchanged && f.extractor_id == self.extractor_id {
                    report.documents_skipped += 1;
                    report.documents_failed_known += 1;
                    let stats = report.by_type.entry(kind).or_default();
                    stats.skipped += 1;
                    stats.secs += started.elapsed().as_secs_f64();
                    tracing::debug!(uri = %r.uri, stage = %f.stage, "known failure; not retried");
                    continue;
                }
                retry_extract = f.stage == "extract";
                tracing::info!(
                    uri = %r.uri,
                    reason = if unchanged { "extractor changed" } else { "file changed" },
                    "retrying previously failed file"
                );
            }

            let outcome = self
                .ingest_bytes(
                    &r.uri,
                    &r.name,
                    &r.path,
                    size,
                    mtime_ns,
                    known_hash,
                    force,
                    retry_extract,
                    stages,
                    image_only,
                )
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
            let keep: std::collections::HashSet<&str> =
                current.iter().map(|s| s.as_str()).collect();
            for uri in stored {
                // Only sweep documents inside the roots we just ingested, so
                // ingesting one root never deletes another root's documents.
                if !keep.contains(uri.as_str()) && source::uri_under_roots(&uri, &roots) {
                    self.store.delete_documents_by_source(&uri).await?;
                    report.documents_deleted += 1;
                    tracing::info!(uri = %uri, "removed (source file gone)");
                }
            }
            // Also forget cached source-file rows for vanished paths.
            for f in self.store.all_source_files().await? {
                if !keep.contains(f.uri.as_str()) && source::uri_under_roots(&f.uri, &roots) {
                    self.store.delete_source_file(&f.uri).await?;
                }
            }
            // ... and remembered failures for paths that disappeared.
            for f in self.store.all_document_failures().await? {
                if !keep.contains(f.uri.as_str()) && source::uri_under_roots(&f.uri, &roots) {
                    self.store.clear_document_failure(&f.uri).await?;
                }
            }
            self.store.bump_corpus_version().await?;
        }

        Ok(report)
    }

    /// Ingest a single document, running only the stages that are stale (or
    /// forced). Unchanged content keeps its document id, so per-document stage
    /// rows and content-addressed image vectors persist across runs.
    ///
    /// `image_only` marks a file under an `LS_IMAGE_ONLY_PATHS` root: image
    /// files there get the image channel only — no conversion, no OCR, no
    /// text chunks. The policy is applied per run, and a document whose
    /// stored policy disagrees (the folder was added to or removed from the
    /// list) transitions on the next ingest.
    #[allow(clippy::too_many_arguments)] // cohesive per-file ingest inputs
    pub async fn ingest_bytes(
        &self,
        source_uri: &str,
        name: &str,
        path: &std::path::Path,
        size: u64,
        mtime_ns: i64,
        known_hash: Option<String>,
        force: bool,
        // A previously failed conversion is being retried (the extractor
        // identity changed), so extract must run even if its fingerprint
        // happens to match.
        retry_extract: bool,
        stages: StageSet,
        image_only_folder: bool,
    ) -> Result<IngestOutcome> {
        let started = Instant::now();
        // The policy means "image files in this folder are image-only"; other
        // file types there ingest normally.
        let image_only = image_only_folder && is_image_name(name);
        let method = if image_only {
            "image-only".to_string()
        } else {
            self.method_label(name)
        };
        let chunker = format!("{:?}", self.chunker.kind()).to_lowercase();

        // Resolve the content hash without reading the file when the cheap
        // (size, mtime) gate matched what we recorded last time. The bytes are
        // read lazily, only if a stage actually needs them.
        let mut bytes: Option<Vec<u8>> = None;
        let hash = match known_hash {
            Some(h) => h,
            None => {
                let b = std::fs::read(path).map_err(|e| Error::io(path, e))?;
                let h = content_hash(&b);
                bytes = Some(b);
                h
            }
        };
        let fp = &self.fingerprints;

        // Unchanged content reuses the existing document (its id, stage rows
        // and content-addressed image vectors). New content is *not* inserted
        // until it has been converted, so a failed conversion never replaces
        // the previous version; the old documents are removed once the new one
        // is committed.
        let existing = self.store.find_document_by_hash(&hash).await?;
        let created = existing.is_none();
        let old_ids = if created {
            self.store.document_ids_by_source(source_uri).await?
        } else {
            Vec::new()
        };
        let mut doc = match &existing {
            Some(id) => self
                .store
                .get_document(id)
                .await?
                .ok_or_else(|| Error::store("document vanished mid-ingest"))?,
            None => Document::new(source_uri, stem(name), hash.clone()),
        };
        let doc_id = doc.id.clone();
        // Whether the document row exists in the store yet (a created document
        // is inserted by the first stage that stores something).
        let mut doc_inserted = !created;
        let stored_image_only =
            doc.metadata.get("folder_policy").and_then(|v| v.as_str()) == Some("image-only");

        let force_extract = retry_extract || (force && stages.wants("extract"));
        let force_text = force && stages.wants("text_embed");
        let force_image = force && stages.wants("image_embed");
        let mut did_work = false;
        let mut n_chunks = 0usize;

        // Policy transition, image-only direction: drop the text channel (its
        // chunks, vectors and stage rows) — but only when there is one; a
        // brand-new image-only document has nothing to drop. The reverse
        // transition needs no code here — a promoted document has no `extract`
        // fingerprint, so the normal stage logic below converts and embeds it.
        let had_text_channel = image_only
            && !stored_image_only
            && (self
                .store
                .document_stage(&doc_id, "extract")
                .await?
                .is_some()
                || self.store.count_text_chunks_for(&doc_id).await? > 0);

        // --- plan the stages before doing any work, so an up-to-date file
        // stays silent and cheap (no "ingesting" line for a no-op).
        let has_text = self.store.count_text_chunks_for(&doc_id).await? > 0;
        let want_extract = !image_only
            && (stages.wants("extract")
                || retry_extract
                || (stages.wants("text_embed") && !has_text));
        let extract_fp = self.store.document_stage(&doc_id, "extract").await?;
        // Why extract would run — also the skip reason when empty. A matching
        // fingerprint always skips: a document with no text chunks (scanned or
        // blank pages) must not re-convert on every run, conversion being
        // deterministic it would produce no chunks again.
        let extract_reason = if retry_extract {
            "retry-failed"
        } else if force_extract {
            "forced"
        } else if extract_fp.as_deref() != Some(fp.extract.as_str()) {
            if extract_fp.is_some() {
                "fingerprint-changed"
            } else {
                "fingerprint-missing"
            }
        } else {
            ""
        };
        let extract_will_run = want_extract && !extract_reason.is_empty();

        let text_fp = self.store.document_stage(&doc_id, "text_embed").await?;
        let text_will_run = !image_only
            && (stages.wants("text_embed") || extract_will_run)
            && (extract_will_run
                || force_text
                || text_fp.as_deref() != Some(fp.text_embed.as_str()));

        let image_fp = self.store.document_stage(&doc_id, "image_embed").await?;
        let image_will_run = self.image.is_some()
            && is_image_name(name)
            && stages.wants("image_embed")
            && (force_image || image_fp.as_deref() != Some(fp.image_embed.as_str()));

        if had_text_channel || extract_will_run || text_will_run || image_will_run {
            tracing::info!(
                name = name,
                uri = source_uri,
                bytes = size,
                method = %method,
                policy = if image_only { "image-only" } else { "default" },
                chunker = %chunker,
                "processing"
            );
        }

        if had_text_channel {
            self.store.replace_text_chunks(&doc_id, &[]).await?;
            self.store.clear_document_stage(&doc_id, "extract").await?;
            self.store
                .clear_document_stage(&doc_id, "text_embed")
                .await?;
            did_work = true;
            tracing::info!(
                name = name,
                "folder policy image-only; text channel dropped"
            );
        }

        // --- extract: convert + chunk (coupled, since we cache no conversion)
        let mut extract_ran = false;
        if extract_will_run {
            if bytes.is_none() {
                bytes = Some(std::fs::read(path).map_err(|e| Error::io(path, e))?);
            }
            let raw = bytes.as_ref().expect("bytes loaded above");
            let t = Instant::now();
            let converted = match convert::convert(&self.cfg, name, raw) {
                Ok(c) => c,
                Err(e) => {
                    // Remember the failure: the next run skips this file until
                    // it changes or the extractor does. Nothing was inserted,
                    // so a previous version (if any) is left untouched.
                    self.record_failure(
                        source_uri,
                        size,
                        mtime_ns,
                        &hash,
                        "extract",
                        &e.to_string(),
                    )
                    .await?;
                    return Err(e);
                }
            };
            let parse_secs = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let text_chunks = self.chunk_converted(&doc_id, &converted)?;
            let chunk_secs = t.elapsed().as_secs_f64();

            if text_chunks.is_empty() {
                tracing::warn!(
                    name = name,
                    pages = converted.pages,
                    "converted document produced no text chunks; not searchable in the text channel"
                );
            }

            let words = converted.markdown.split_whitespace().count();
            let title = first_heading(&converted.markdown).unwrap_or_else(|| stem(name));
            let metrics = ProcessingMetrics::compute(
                size,
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
            doc.title = title;
            doc.hash = hash.clone();
            doc.metadata = serde_json::json!({
                "source": source_uri,
                "metrics": metrics,
                "markdown": converted.markdown,
            });
            // Insert before the chunks (they reference the document); a created
            // document is only written now that conversion has succeeded.
            self.store.upsert_document(&doc).await?;
            doc_inserted = true;
            self.store
                .replace_text_chunks(&doc_id, &text_chunks)
                .await?;
            // Order matters: drop the text-embed fingerprint *before* the new
            // extract fingerprint is recorded, so a crash in between leaves the
            // stage missing (re-embed the fresh chunks) rather than stale.
            self.store
                .clear_document_stage(&doc_id, "text_embed")
                .await?;
            self.store
                .set_document_stage(&doc_id, "extract", &fp.extract)
                .await?;
            // Conversion succeeded: any remembered failure for this path is
            // resolved.
            self.store.clear_document_failure(source_uri).await?;

            extract_ran = true;
            did_work = true;
            n_chunks += text_chunks.len();
        }

        // --- text_embed: (re)embed the stored text chunks in place
        if text_will_run {
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
        if image_will_run {
            let embedder = self
                .image
                .as_ref()
                .expect("image_will_run implies a provider");
            if bytes.is_none() {
                bytes = Some(std::fs::read(path).map_err(|e| Error::io(path, e))?);
            }
            // A created image-only document must exist before its chunk.
            if !doc_inserted {
                doc.title = stem(name);
                doc.hash = hash.clone();
                doc.metadata = serde_json::json!({
                    "source": source_uri,
                    "folder_policy": "image-only",
                });
                self.store.upsert_document(&doc).await?;
                doc_inserted = true;
            }
            let raw = bytes.as_ref().expect("bytes loaded above");
            let mut vectors = embedder.embed_images(std::slice::from_ref(raw)).await?;
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

        // Image-only documents never pass through extract, which is what
        // normally records title/hash/metadata — do it here instead.
        if image_only && did_work {
            doc.title = stem(name);
            doc.hash = hash.clone();
            doc.metadata = serde_json::json!({
                "source": source_uri,
                "folder_policy": "image-only",
            });
            self.store.upsert_document(&doc).await?;
        }

        if image_only && self.image.is_none() {
            tracing::warn!(
                name = name,
                "image-only folder but the image embedder is off; file not indexed"
            );
            self.record_source_file(source_uri, size, mtime_ns, &hash, None)
                .await?;
            return Ok(IngestOutcome::Skipped);
        }

        if !did_work {
            // Compute before logging: an `.await` inside the macro's argument
            // list makes the whole future non-`Send` (it would block running
            // ingest on a spawned task).
            let text_chunks = self.store.count_text_chunks_for(&doc_id).await?;
            tracing::debug!(name = name, text_chunks, "up to date; skipping");
            let stored_doc = if doc_inserted {
                Some(doc_id.as_str())
            } else {
                None
            };
            self.record_source_file(source_uri, size, mtime_ns, &hash, stored_doc)
                .await?;
            self.store.clear_document_failure(source_uri).await?;
            return Ok(IngestOutcome::Skipped);
        }

        // The new version is committed: drop the documents this path replaced
        // (they carry a different content hash, so this never touches the new
        // one).
        for id in &old_ids {
            if id != &doc_id {
                self.store.delete_document(id).await?;
            }
        }
        self.store.bump_corpus_version().await?;
        tracing::info!(
            name = name,
            uri = source_uri,
            chunks = n_chunks,
            secs = started.elapsed().as_secs_f64(),
            extract = if extract_ran {
                extract_reason
            } else {
                "unchanged"
            },
            "ingested"
        );
        self.record_source_file(source_uri, size, mtime_ns, &hash, Some(&doc_id))
            .await?;
        Ok(IngestOutcome::Ingested(n_chunks))
    }

    /// Remember a failed conversion so the next run does not retry it.
    async fn record_failure(
        &self,
        uri: &str,
        size: u64,
        mtime_ns: i64,
        hash: &str,
        stage: &str,
        error: &str,
    ) -> Result<()> {
        self.store
            .record_document_failure(&crate::store::DocumentFailure {
                uri: uri.to_string(),
                size,
                mtime_ns,
                content_hash: hash.to_string(),
                stage: stage.to_string(),
                extractor_id: self.extractor_id.clone(),
                error: error.to_string(),
                updated_at: crate::model::now_rfc3339(),
            })
            .await
    }

    /// Refresh the cheap-change cache for a source path.
    async fn record_source_file(
        &self,
        uri: &str,
        size: u64,
        mtime_ns: i64,
        hash: &str,
        doc_id: Option<&str>,
    ) -> Result<()> {
        self.store
            .upsert_source_file(&crate::store::SourceFile {
                uri: uri.to_string(),
                size,
                mtime_ns,
                hash: hash.to_string(),
                doc_id: doc_id.map(str::to_string),
            })
            .await
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
