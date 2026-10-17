//! Pluggable vector store.
//!
//! A store persists [`Document`]s (metadata) and their [`Chunk`]s (text or
//! image) and answers dense vector search in two spaces (text and image) plus
//! keyword (FTS5/BM25) search. The ingest pipeline drives it stage by stage:
//! `replace_text_chunks` + `set_text_vectors` for the text side,
//! `upsert_image_chunk` for the image side, and `document_stage` rows to record
//! which stage fingerprint each document was last built under.

pub mod sqlite;

use crate::Result;
use crate::config::Config;
use crate::model::{Chunk, Document, Scored};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// A document + chunk store with dense vector and keyword search.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Create tables/indexes if they do not exist. Safe to call repeatedly.
    async fn migrate(&self) -> Result<()>;

    /// Insert or replace a document row (keyed by `id`).
    async fn upsert_document(&self, doc: &Document) -> Result<()>;

    /// Return the id of an existing document with this content hash, if any.
    async fn find_document_by_hash(&self, hash: &str) -> Result<Option<String>>;

    /// Replace a document's **text** chunks: drop its existing text chunks
    /// (rows, FTS and text vectors) and insert the new ones (rows + FTS). The
    /// text vectors are written separately by [`Self::set_text_vectors`].
    async fn replace_text_chunks(&self, doc_id: &str, chunks: &[Chunk]) -> Result<()>;

    /// Insert or replace text-space vectors for the given chunk ids.
    async fn set_text_vectors(&self, pairs: &[(String, Vec<f32>)]) -> Result<()>;

    /// A document's text chunks, ordered by `ordinal` (no vectors).
    async fn text_chunks(&self, doc_id: &str) -> Result<Vec<Chunk>>;

    /// Number of text chunks stored for one document.
    async fn count_text_chunks_for(&self, doc_id: &str) -> Result<usize>;

    /// Insert or replace an **image** chunk and its image-space vector. Image
    /// chunk ids are content-addressed, so re-running this for the same image
    /// is idempotent.
    async fn upsert_image_chunk(&self, chunk: &Chunk, embedding: &[f32]) -> Result<()>;

    /// Dense search in the text space.
    async fn vector_search(&self, query: &[f32], k: usize) -> Result<Vec<Scored>>;

    /// Dense search in the image space.
    async fn image_search(&self, query: &[f32], k: usize) -> Result<Vec<Scored>>;

    /// Keyword search over the FTS5 index (BM25), best first.
    async fn keyword_search(&self, query: &str, k: usize) -> Result<Vec<Scored>>;

    /// Total number of stored chunks.
    async fn count_chunks(&self) -> Result<usize>;

    /// Number of chunks stored for one document (any kind).
    async fn count_chunks_for(&self, doc_id: &str) -> Result<usize>;

    /// Total number of stored documents.
    async fn count_documents(&self) -> Result<usize>;

    /// Every stored document with its metadata.
    async fn list_documents(&self) -> Result<Vec<Document>>;

    /// Load a single document by id.
    async fn get_document(&self, doc_id: &str) -> Result<Option<Document>>;

    /// Load several documents by id, keyed by id.
    async fn get_documents_by_ids(&self, ids: &[String]) -> Result<HashMap<String, Document>>;

    /// A contiguous slice of one document's chunks, ordered by `ordinal`.
    async fn get_chunks_range(
        &self,
        doc_id: &str,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<Chunk>>;

    /// Load a single chunk by id (used by the media route).
    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<Chunk>>;

    /// Every stored `source_uri` (for `--sync` deletion sweeps).
    async fn all_source_uris(&self) -> Result<Vec<String>>;

    /// Cached file identity for one source path (cheap change detection).
    async fn source_file(&self, uri: &str) -> Result<Option<SourceFile>>;

    /// Record (or refresh) a source file's size/mtime/hash and its document.
    async fn upsert_source_file(&self, file: &SourceFile) -> Result<()>;

    /// Forget a source path (the file disappeared, or failed a read).
    async fn delete_source_file(&self, uri: &str) -> Result<()>;

    /// Every cached source file (for `--sync` deletion sweeps).
    async fn all_source_files(&self) -> Result<Vec<SourceFile>>;

    /// Delete one document and all of its chunks, vectors and stage rows.
    async fn delete_document(&self, doc_id: &str) -> Result<()>;

    /// Delete every document (and chunks) originating from `source_uri`.
    async fn delete_documents_by_source(&self, source_uri: &str) -> Result<()>;

    /// Remove all documents, chunks and index rows.
    async fn clear(&self) -> Result<()>;

    /// Monotonic corpus version, bumped by every successful ingest transaction.
    async fn corpus_version(&self) -> Result<i64>;

    /// Increment and return the corpus version.
    async fn bump_corpus_version(&self) -> Result<i64>;

    /// Read a `meta` entry.
    async fn get_meta(&self, key: &str) -> Result<Option<String>>;

    /// Write a `meta` entry.
    async fn set_meta(&self, key: &str, value: &str) -> Result<()>;

    /// The fingerprint a document was last built under for `stage`.
    async fn document_stage(&self, doc_id: &str, stage: &str) -> Result<Option<String>>;

    /// Record a document's stage fingerprint.
    async fn set_document_stage(&self, doc_id: &str, stage: &str, fingerprint: &str) -> Result<()>;

    /// Forget a document's stage fingerprint (forces that stage to re-run).
    async fn clear_document_stage(&self, doc_id: &str, stage: &str) -> Result<()>;

    /// Forget a stage fingerprint for every document (e.g. after a dimension
    /// change recreated that space).
    async fn clear_stage_all(&self, stage: &str) -> Result<()>;

    /// How image vectors relate to the text space (drives retrieval wiring).
    fn image_space(&self) -> ImageSpace;
}

/// Cached file identity for cheap change detection.
///
/// `size` + `mtime_ns` gate the expensive path: when both match what was
/// recorded last time, the stored `hash` is trusted and the file is not read
/// again. `doc_id` links the path to the document its content resolved to
/// (dedup means several paths can share one document).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub uri: String,
    pub size: u64,
    pub mtime_ns: i64,
    pub hash: String,
    pub doc_id: Option<String>,
}

/// How image vectors relate to the text vector space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSpace {
    /// No image channel.
    Off,
    /// A separate image vector table (e.g. bge text + SigLIP images).
    Split(usize),
    /// Text and image chunks share one table and one dimension (EG2 both
    /// sides): one query vector serves both, and `mode=image` is a kind
    /// filter rather than a different index.
    Unified,
}

impl ImageSpace {
    /// Whether the two providers land in one space.
    ///
    /// True when both providers name the same space (EG2) *and* the
    /// dimensions agree; anything else keeps the two-table layout.
    pub fn for_config(cfg: &Config) -> Self {
        if !cfg.image.enabled() {
            return ImageSpace::Off;
        }
        let same_space = cfg.embed.space_id() == cfg.image.space_id();
        if same_space && cfg.embed.dim == cfg.image.dim {
            ImageSpace::Unified
        } else {
            ImageSpace::Split(cfg.image.dim)
        }
    }
}

/// Build and connect the store selected by `cfg`.
pub async fn from_config(cfg: &Config) -> Result<Arc<dyn VectorStore>> {
    let store: Arc<dyn VectorStore> = Arc::new(
        sqlite::SqliteStore::connect(&cfg.db_path, cfg.embed.dim, ImageSpace::for_config(cfg))
            .await?,
    );
    store.migrate().await?;
    Ok(store)
}
