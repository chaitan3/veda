//! OpenAI-compatible embeddings client — targets a llama.cpp server running in
//! embedding mode (`/v1/embeddings`).

use super::Embedder;
use crate::config::Config;
use crate::{Error, Result};
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Largest batch sent per HTTP request.
const MAX_BATCH: usize = 32;

/// Embedder backed by an OpenAI-compatible `/v1/embeddings` endpoint.
pub struct LlamaCppEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    query_prefix: String,
    doc_prefix: String,
    dim: AtomicUsize,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
}

impl LlamaCppEmbedder {
    pub fn from_config(cfg: &Config) -> Result<Self> {
        let model = cfg.embed.model.clone().ok_or_else(|| {
            Error::config("LS_EMBED_MODEL is required when LS_EMBED_PROVIDER=llamacpp")
        })?;
        Ok(LlamaCppEmbedder {
            client: reqwest::Client::new(),
            base_url: cfg.embed.base_url.trim_end_matches('/').to_string(),
            model,
            api_key: cfg.embed.api_key.clone(),
            query_prefix: cfg.embed.query_prefix.clone().unwrap_or_default(),
            doc_prefix: cfg.embed.doc_prefix.clone().unwrap_or_default(),
            dim: AtomicUsize::new(cfg.embed.dim),
        })
    }

    async fn call(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let body = serde_json::json!({ "model": self.model, "input": inputs });
        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::embedding(format!(
                "embeddings endpoint {} returned {}: {}",
                url, status, text
            )));
        }
        let mut parsed: EmbeddingsResponse = resp.json().await?;
        if parsed.data.is_empty() {
            return Err(Error::embedding("embeddings endpoint returned no data"));
        }
        parsed.data.sort_by_key(|d| d.index);
        let dim = parsed.data[0].embedding.len();
        self.dim.store(dim, Ordering::Relaxed);
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }

    async fn call_prefixed(&self, texts: &[String], prefix: &str) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH) {
            let inputs: Vec<String> = batch
                .iter()
                .map(|t| {
                    if prefix.is_empty() {
                        t.clone()
                    } else {
                        format!("{prefix}{t}")
                    }
                })
                .collect();
            let mut vectors = self.call(&inputs).await?;
            if vectors.len() != batch.len() {
                return Err(Error::embedding(format!(
                    "embeddings count mismatch: sent {}, got {}",
                    batch.len(),
                    vectors.len()
                )));
            }
            out.append(&mut vectors);
        }
        Ok(out)
    }
}

#[async_trait]
impl Embedder for LlamaCppEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.call_prefixed(texts, &self.doc_prefix).await
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let mut v = self
            .call_prefixed(std::slice::from_ref(&query.to_string()), &self.query_prefix)
            .await?;
        v.pop()
            .ok_or_else(|| Error::embedding("embeddings endpoint returned no vector"))
    }

    fn dim(&self) -> usize {
        self.dim.load(Ordering::Relaxed)
    }

    fn id(&self) -> &str {
        &self.model
    }
}
