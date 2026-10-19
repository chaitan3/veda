//! Error type for the veda service.

use std::path::PathBuf;

/// Crate-wide error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("config error: {0}")]
    Config(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("conversion error: {0}")]
    Conversion(String),

    #[error("retrieval error: {0}")]
    Retrieval(String),

    #[error("source error: {0}")]
    Source(String),

    #[error("feature not enabled: {0} (rebuild with --features {1})")]
    FeatureDisabled(String, String),

    #[error("unsupported input: {0}")]
    Unsupported(String),

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    IoPlain(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }

    pub fn store(msg: impl Into<String>) -> Self {
        Error::Store(msg.into())
    }

    pub fn conversion(msg: impl Into<String>) -> Self {
        Error::Conversion(msg.into())
    }

    pub fn embedding(msg: impl Into<String>) -> Self {
        Error::Embedding(msg.into())
    }

    pub fn source(msg: impl Into<String>) -> Self {
        Error::Source(msg.into())
    }

    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
