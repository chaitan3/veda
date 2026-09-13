//! Pluggable vector store.
//!
//! A store persists [`Document`]s (metadata) and their [`Chunk`]s (text +
//! embedding) and answers dense vector search plus keyword (FTS5/BM25) search.
//!
//! Phase 0 ships the SQLite backend only. The trait is deliberately shaped like
//! docling-rag's so the Phase 1 image space (a second vector table with its own
//! dimensionality) can be added as additional methods rather than a redesign.

pub mod sqlite;

use crate::Result;
use crate::config::Config;
use crate::model::{Chunk, Document, Scored};
use async_trait::async_trait;
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

    /// Bulk-insert chunks (each must carry a populated `embedding`).
    async fn insert_chunks(&self, chunks: &[Chunk]) -> Result<()>;

    /// Dense search: the `k` chunks whose embeddings are most cosine-similar.
    async fn vector_search(&self, query: &[f32], k: usize) -> Result<Vec<Scored>>;

    /// Keyword search over the FTS5 index (BM25), best first.
    async fn keyword_search(&self, query: &str, k: usize) -> Result<Vec<Scored>>;

    /// Total number of stored chunks.
    async fn count_chunks(&self) -> Result<usize>;

    /// Number of chunks stored for one document.
    async fn count_chunks_for(&self, doc_id: &str) -> Result<usize>;

    /// Total number of stored documents.
    async fn count_documents(&self) -> Result<usize>;

    /// Every stored document with its metadata.
    async fn list_documents(&self) -> Result<Vec<Document>>;

    /// Every stored `source_uri` (for `--sync` deletion sweeps).
    async fn all_source_uris(&self) -> Result<Vec<String>>;

    /// Delete one document and all of its chunks.
    async fn delete_document(&self, doc_id: &str) -> Result<()>;

    /// Delete every document (and chunks) originating from `source_uri`.
    async fn delete_documents_by_source(&self, source_uri: &str) -> Result<()>;

    /// Remove all documents, chunks and index rows.
    async fn clear(&self) -> Result<()>;

    /// Monotonic corpus version, bumped by every successful ingest transaction.
    /// The service polls this to invalidate warm caches across processes.
    async fn corpus_version(&self) -> Result<i64>;

    /// Increment and return the corpus version.
    async fn bump_corpus_version(&self) -> Result<i64>;

    /// Version of the ingest pipeline that produced the current corpus.
    /// Ingest reprocesses everything when this differs from the binary's
    /// [`crate::pipeline::PIPELINE_VERSION`], so enabling new extraction /
    /// embedding channels upgrades an existing database without a reset.
    async fn pipeline_version(&self) -> Result<u32>;

    /// Record the current pipeline version.
    async fn set_pipeline_version(&self, version: u32) -> Result<()>;
}

/// Build and connect the store selected by `cfg`.
pub async fn from_config(cfg: &Config) -> Result<Arc<dyn VectorStore>> {
    let store: Arc<dyn VectorStore> =
        Arc::new(sqlite::SqliteStore::connect(&cfg.db_path, cfg.embed.dim).await?);
    store.migrate().await?;
    Ok(store)
}
