//! EmbeddingGemma 2 text embedder (ONNX) — the `eg2` provider.
//!
//! EG2 maps text, images and audio into one 768-d space; this module is the
//! **text** half (the `eg2` image half lives in `embed::image::eg2`). It runs
//! the `model.onnx` graph exported by `onnx-community/embeddinggemma-2-ONNX`,
//! which outputs `sentence_embedding` (already mean-pooled, projected to 768
//! and normalised) alongside `last_hidden_state`.
//!
//! Two things are model-specific and matter for quality:
//!
//! * **Task prefixes.** The model is trained with them; `task: search result |
//!   query: …` for queries and `title: … | text: …` for documents. They come
//!   from `LS_EMBED_QUERY_PREFIX` / `LS_EMBED_DOC_PREFIX`, defaulted here.
//! * **fp32 only.** The model card is explicit that fp16 returns NaN or
//!   silently degraded embeddings (its activation range exceeds fp16), so this
//!   provider never asks for fp16 and runs fp32 on both CPU and CUDA.

use crate::config::Config;
use crate::embed::onnx::pad_batch;
use crate::embed::Embedder;
use crate::{Error, Result};
use async_trait::async_trait;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// EG2's default task prefixes (see the model card's prompt table).
pub const DEFAULT_QUERY_PREFIX: &str = "task: search result | query: ";
pub const DEFAULT_DOC_PREFIX: &str = "title: none | text: ";

/// Embedder backed by the EG2 text graph.
#[derive(Clone)]
pub struct Eg2TextEmbedder {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    dim: usize,
    id: String,
    query_prefix: String,
    doc_prefix: String,
    max_tokens: usize,
}

impl Eg2TextEmbedder {
    /// Load the graph and tokenizer from the paths in `cfg`.
    pub fn from_config(cfg: &Config) -> Result<Self> {
        let model = cfg.embed.eg2_model.clone().ok_or_else(|| {
            Error::config("LS_EG2_TEXT_MODEL is required when LS_EMBED_PROVIDER=eg2")
        })?;
        let tokenizer_path = cfg.embed.eg2_tokenizer.clone().ok_or_else(|| {
            Error::config("LS_EG2_TOKENIZER is required when LS_EMBED_PROVIDER=eg2")
        })?;

        let builder = Session::builder().map_err(|e| Error::embedding(format!("{e}")))?;
        let mut builder =
            docling_onnx::apply(builder).map_err(|e| Error::embedding(format!("eg2 {e}")))?;
        let session = builder
            .commit_from_file(docling_core::assets::resolve(&model))
            .map_err(|e| Error::embedding(format!("loading EG2 text graph '{model}': {e}")))?;

        let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
        if !outputs.contains(&"sentence_embedding") && !outputs.contains(&"last_hidden_state") {
            return Err(Error::embedding(format!(
                "EG2 graph '{model}' has no sentence_embedding/last_hidden_state output (found {outputs:?})"
            )));
        }

        let tokenizer = Tokenizer::from_file(docling_core::assets::resolve(&tokenizer_path))
            .map_err(|e| Error::embedding(format!("loading tokenizer '{tokenizer_path}': {e}")))?;

        Ok(Eg2TextEmbedder {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            dim: cfg.embed.dim,
            id: format!("eg2:{model}"),
            query_prefix: cfg
                .embed
                .query_prefix
                .clone()
                .unwrap_or_else(|| DEFAULT_QUERY_PREFIX.to_string()),
            doc_prefix: cfg
                .embed
                .doc_prefix
                .clone()
                .unwrap_or_else(|| DEFAULT_DOC_PREFIX.to_string()),
            max_tokens: cfg.embed.max_tokens,
        })
    }

    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::embedding(format!("tokenize: {e}")))?;
        let (batch, seq, ids, mask, _types) = pad_batch(&encodings, self.max_tokens);
        // Keep a copy: the pooling fallback needs the mask, and the graph
        // consumes the tensor.
        let mask_data = mask.clone();

        let (_input_names, dims, data) = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| Error::embedding("onnx session mutex poisoned"))?;
            let names: Vec<String> =
                session.inputs().iter().map(|i| i.name().to_string()).collect();
            let has = |n: &str| names.iter().any(|x| x == n);

            let id_tensor = Tensor::from_array(([batch, seq], ids))
                .map_err(|e| Error::embedding(e.to_string()))?;
            let mask_tensor = Tensor::from_array(([batch, seq], mask))
                .map_err(|e| Error::embedding(e.to_string()))?;
            // The text graph also takes pre-encoded modality features; empty
            // for pure-text inputs.
            let empty = || {
                Tensor::from_array(([0usize, 512], Vec::<f32>::new()))
                    .map_err(|e| Error::embedding(e.to_string()))
            };

            let mut feed: Vec<(&str, ort::value::Value)> = vec![("input_ids", id_tensor.into())];
            if has("attention_mask") {
                feed.push(("attention_mask", mask_tensor.into()));
            }
            for name in ["image_features", "video_features", "audio_features"] {
                if has(name) {
                    feed.push((name, empty()?.into()));
                }
            }

            let outputs = session
                .run_with_options(feed, docling_onnx::shrink_run_options())
                .map_err(|e| Error::embedding(format!("eg2 run: {e}")))?;
            // Prefer the pooled head; fall back to mean-pooling last_hidden_state.
            let value = if let Some(v) = outputs.get("sentence_embedding") {
                v
            } else {
                outputs
                    .get("last_hidden_state")
                    .ok_or_else(|| Error::embedding("EG2 run returned no usable output"))?
            };
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::embedding(e.to_string()))?;
            // Copy out: the values borrow the session's output buffers.
            (names, shape.to_vec(), data.to_vec())
        };
        let hidden = *dims.last().unwrap_or(&0) as usize;
        if hidden != self.dim {
            return Err(Error::embedding(format!(
                "EG2 output dim {hidden} does not match LS_EMBED_DIM {}",
                self.dim
            )));
        }
        let mut out = Vec::with_capacity(batch);
        if dims.len() == 2 {
            // [batch, dim] — already pooled (and normalised by the graph).
            for b in 0..batch {
                let mut v = data[b * hidden..(b + 1) * hidden].to_vec();
                crate::math::normalize(&mut v);
                out.push(v);
            }
        } else if dims.len() == 3 {
            // [batch, seq, hidden] — mean-pool over the attention mask.
            let seq_out = dims[1] as usize;
            for b in 0..batch {
                let mut v = vec![0f32; hidden];
                let mut count = 0.0f32;
                for t in 0..seq_out {
                    if mask_data[b * seq_out + t] == 0 {
                        continue;
                    }
                    let base = (b * seq_out + t) * hidden;
                    for (h, acc) in v.iter_mut().enumerate() {
                        *acc += data[base + h];
                    }
                    count += 1.0;
                }
                if count > 0.0 {
                    for acc in v.iter_mut() {
                        *acc /= count;
                    }
                }
                crate::math::normalize(&mut v);
                out.push(v);
            }
        } else {
            return Err(Error::embedding(format!(
                "unexpected EG2 output rank {} (shape {dims:?})",
                dims.len()
            )));
        }
        Ok(out)
    }
}

#[async_trait]
impl Embedder for Eg2TextEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.doc_prefix))
            .collect();
        let me = self.clone();
        // ONNX inference is blocking; keep it off the async runtime.
        let vectors = tokio::task::spawn_blocking(move || me.run(&prefixed))
            .await
            .map_err(|e| Error::embedding(format!("join: {e}")))??;
        Ok(vectors)
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let me = self.clone();
        let text = format!("{}{query}", self.query_prefix);
        let mut v = tokio::task::spawn_blocking(move || me.run(std::slice::from_ref(&text)))
            .await
            .map_err(|e| Error::embedding(format!("join: {e}")))??;
        v.pop()
            .ok_or_else(|| Error::embedding("embedder returned no vector"))
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        &self.id
    }
}
