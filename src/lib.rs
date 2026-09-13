//! `local-search`: a text + image search service over a local document store.
//!
//! Phase 0 scope: documents (PDF/DOCX/text, OCR for scans) and images are
//! ingested by a CLI tool into a SQLite store, and a read-only HTTP service
//! answers search queries. Retrieval is hybrid (FTS5/BM25 + dense vectors,
//! fused with RRF). Image captioning and CLIP embeddings arrive in Phase 1;
//! the data model, store and retrieval seams are already shaped for them.

pub mod api;
pub mod chunk;
pub mod config;
pub mod convert;
pub mod download;
pub mod embed;
pub mod error;
pub mod math;
pub mod metrics;
pub mod model;
pub mod pipeline;
pub mod retrieve;
pub mod runtime;
pub mod source;
pub mod store;

pub use config::Config;
pub use error::{Error, Result};
pub use model::{Chunk, ChunkKind, Document, Locator, RetrievalMode, Scored};
pub use pipeline::Pipeline;
