//! Pluggable embedding providers.
//!
//! All providers implement [`Embedder`]. The default is the deterministic
//! [`hash`] embedder (offline tests); [`llamacpp`] talks to an OpenAI-compatible
//! `/v1/embeddings` endpoint (your llama.cpp server), and `onnx` (feature
//! `onnx-embed`) runs a local sentence encoder in-process. There is no Ollama
//! or Gemini backend by design.

pub mod hash;
pub mod image;
pub mod llamacpp;

#[cfg(feature = "onnx-embed")]
pub mod eg2;
#[cfg(feature = "onnx-embed")]
pub mod onnx;

use crate::config::{Config, EmbedProvider};
use crate::{Error, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// How many sequences have been truncated to the embedder's window since this
/// process started. Truncation is silent quality loss (the tail of a chunk is
/// not embedded), so it is counted and surfaced in `stats` and the ingest
/// summary instead of only being logged.
static TRUNCATED: AtomicU64 = AtomicU64::new(0);

/// Record `n` truncated sequences (called by the ONNX batchers).
pub(crate) fn note_truncated(n: u64) {
    TRUNCATED.fetch_add(n, Ordering::Relaxed);
}

/// Truncated-sequence count since process start.
pub fn truncated_sequences() -> u64 {
    TRUNCATED.load(Ordering::Relaxed)
}

/// A source of embedding vectors.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a batch of **documents**, one vector per input, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embed a single **query**. Providers may apply a distinct query prefix.
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let mut v = self.embed(std::slice::from_ref(&query.to_string())).await?;
        v.pop()
            .ok_or_else(|| Error::embedding("embedder returned no vector"))
    }

    /// The dimensionality of every returned vector (0 if not yet known).
    fn dim(&self) -> usize;

    /// A short identifier (`"hash"`, `"llamacpp:bge-m3"`, …) for logs.
    fn id(&self) -> &str;
}

/// Build the embedder selected by `cfg.embed.provider`.
pub fn from_config(cfg: &Config) -> Result<Arc<dyn Embedder>> {
    match cfg.embed.provider {
        EmbedProvider::Hash => Ok(Arc::new(hash::HashEmbedder::new(cfg.embed.dim))),
        EmbedProvider::LlamaCpp => Ok(Arc::new(llamacpp::LlamaCppEmbedder::from_config(cfg)?)),
        EmbedProvider::Onnx => {
            #[cfg(feature = "onnx-embed")]
            {
                Ok(Arc::new(onnx::OnnxEmbedder::from_config(cfg)?))
            }
            #[cfg(not(feature = "onnx-embed"))]
            {
                Err(Error::FeatureDisabled(
                    "onnx embedder (LS_EMBED_PROVIDER=onnx)".into(),
                    "onnx-embed".into(),
                ))
            }
        }
        EmbedProvider::Eg2 => {
            #[cfg(feature = "onnx-embed")]
            {
                Ok(Arc::new(eg2::Eg2TextEmbedder::from_config(cfg)?))
            }
            #[cfg(not(feature = "onnx-embed"))]
            {
                Err(Error::FeatureDisabled(
                    "EG2 embedder (LS_EMBED_PROVIDER=eg2)".into(),
                    "onnx-embed".into(),
                ))
            }
        }
    }
}
