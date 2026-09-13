//! Pluggable embedding providers.
//!
//! All providers implement [`Embedder`]. The default is the deterministic
//! [`hash`] embedder (offline tests); [`llamacpp`] talks to an OpenAI-compatible
//! `/v1/embeddings` endpoint (your llama.cpp server), and `onnx` (feature
//! `onnx-embed`) runs a local sentence encoder in-process. There is no Ollama
//! or Gemini backend by design.

pub mod hash;
pub mod llamacpp;

#[cfg(feature = "onnx-embed")]
pub mod onnx;

use crate::config::{Config, EmbedProvider};
use crate::{Error, Result};
use async_trait::async_trait;
use std::sync::Arc;

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
    }
}
