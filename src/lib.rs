//! `local-search`: a text + image search service over a local document store.
//!
//! Documents (PDF/DOCX/text, OCR for scans) and standalone image files are
//! ingested by a CLI tool into a SQLite store; a read-only HTTP service answers
//! search queries. Retrieval is hybrid (FTS5/BM25 + dense vectors, fused with
//! RRF) across two spaces: text (bge-small) and image (SigLIP2). Ingest is
//! stage-scoped — `extract` / `text_embed` / `image_embed` — so reprocessing
//! touches only what changed.

pub mod api;
pub mod chunk;
pub mod config;
pub mod convert;
pub mod download;
pub mod embed;
pub mod error;
pub mod fingerprint;
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
pub use model::{Chunk, ChunkKind, Document, RetrievalMode, Scored};
pub use pipeline::Pipeline;
