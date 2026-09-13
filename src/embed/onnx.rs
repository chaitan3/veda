//! Local ONNX sentence embedder (`LS_EMBED_PROVIDER=onnx`).
//!
//! Reuses the same `ort` + Hugging Face `tokenizers` pattern as docling-rag.
//! The graph is adapted at load time: only the inputs it declares are fed
//! (`token_type_ids` is optional), and the output is either an already-pooled
//! sentence embedding (`dense_vecs [batch, dim]`) or a raw encoder's
//! `last_hidden_state [batch, seq, hidden]`, mean-pooled over the attention
//! mask. Either way the result is L2-normalized.
//!
//! Default model: bge-small-en-v1.5 (384-d). Override with `LS_ONNX_MODEL` and
//! `LS_ONNX_TOKENIZER`; fetch the defaults with `local-search download-models`.

use super::Embedder;
use crate::config::Config;
use crate::{Error, Result};
use async_trait::async_trait;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// Embedder backed by a local ONNX transformer encoder.
pub struct OnnxEmbedder {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    dim: usize,
    id: String,
    /// Feed `token_type_ids` only when the graph declares it.
    needs_token_type_ids: bool,
    /// The output to read; its rank decides pooled-vs-hidden at run time.
    output_name: String,
}

impl OnnxEmbedder {
    /// Load the ONNX model and tokenizer from the paths in `cfg`.
    pub fn from_config(cfg: &Config) -> Result<Self> {
        let model_path = cfg.embed.onnx_model.clone().ok_or_else(|| {
            Error::config("LS_ONNX_MODEL is required when LS_EMBED_PROVIDER=onnx")
        })?;
        let tokenizer_path = cfg.embed.onnx_tokenizer.clone().ok_or_else(|| {
            Error::config("LS_ONNX_TOKENIZER is required when LS_EMBED_PROVIDER=onnx")
        })?;

        let builder = Session::builder()
            .map_err(|e| Error::embedding(format!("ONNX session builder: {e}")))?;
        // Route through the shared execution-provider switch, so a GPU build
        // (DOCLING_RS_EP=cuda) embeds on the GPU with CPU fallback too.
        let mut builder =
            docling_onnx::apply(builder).map_err(|e| Error::embedding(format!("embedder {e}")))?;
        let session = builder
            .commit_from_file(docling_core::assets::resolve(&model_path))
            .map_err(|e| Error::embedding(format!("loading ONNX model '{model_path}': {e}")))?;

        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|o| o.name() == "token_type_ids");
        let output_names: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
        let output_name = ["dense_vecs", "last_hidden_state"]
            .into_iter()
            .find(|n| output_names.contains(n))
            .or(output_names.first().copied())
            .ok_or_else(|| Error::embedding("embedding model has no outputs"))?
            .to_string();

        let tokenizer = Tokenizer::from_file(docling_core::assets::resolve(&tokenizer_path))
            .map_err(|e| Error::embedding(format!("loading tokenizer '{tokenizer_path}': {e}")))?;

        Ok(OnnxEmbedder {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            dim: cfg.embed.dim,
            id: format!("onnx:{model_path}"),
            needs_token_type_ids,
            output_name,
        })
    }

    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::embedding(format!("tokenize: {e}")))?;
        let batch = encodings.len();
        let seq = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .max(1);

        // Right-pad ids / mask / type-ids to a uniform sequence length.
        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        let types = vec![0i64; batch * seq];
        for (b, enc) in encodings.iter().enumerate() {
            for (t, (&id, &m)) in enc
                .get_ids()
                .iter()
                .zip(enc.get_attention_mask())
                .enumerate()
            {
                ids[b * seq + t] = id as i64;
                mask[b * seq + t] = m as i64;
            }
        }

        let id_tensor =
            Tensor::from_array(([batch, seq], ids)).map_err(|e| Error::embedding(e.to_string()))?;
        let mask_tensor = Tensor::from_array(([batch, seq], mask.clone()))
            .map_err(|e| Error::embedding(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::embedding("onnx session mutex poisoned"))?;
        let outputs = if self.needs_token_type_ids {
            let type_tensor = Tensor::from_array(([batch, seq], types))
                .map_err(|e| Error::embedding(e.to_string()))?;
            session.run(ort::inputs![
                "input_ids" => id_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => type_tensor,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => id_tensor,
                "attention_mask" => mask_tensor,
            ])
        }
        .map_err(|e| Error::embedding(format!("onnx run: {e}")))?;

        let (shape, data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::embedding(e.to_string()))?;
        let hidden = *shape.last().unwrap_or(&0) as usize;

        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut pooled = vec![0.0f32; hidden];
            if shape.len() == 2 {
                // Already a sentence embedding ([batch, dim], e.g. bge-m3's
                // dense_vecs) — take the row as-is.
                pooled.copy_from_slice(&data[b * hidden..(b + 1) * hidden]);
            } else {
                // Raw encoder output [batch, seq, hidden]: mean-pool over the
                // attention mask.
                let mut count = 0.0f32;
                for t in 0..seq {
                    if mask[b * seq + t] == 0 {
                        continue;
                    }
                    let base = (b * seq + t) * hidden;
                    for (h, p) in pooled.iter_mut().enumerate() {
                        *p += data[base + h];
                    }
                    count += 1.0;
                }
                if count > 0.0 {
                    for p in pooled.iter_mut() {
                        *p /= count;
                    }
                }
            }
            crate::math::normalize(&mut pooled);
            out.push(pooled);
        }
        Ok(out)
    }
}

#[async_trait]
impl Embedder for OnnxEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let session = self.session.clone();
        let tokenizer = self.tokenizer.clone();
        let dim = self.dim;
        let id = self.id.clone();
        let needs_token_type_ids = self.needs_token_type_ids;
        let output_name = self.output_name.clone();
        let texts = texts.to_vec();

        // ONNX inference is blocking CPU work; keep it off the async runtime.
        tokio::task::spawn_blocking(move || {
            let embedder = OnnxEmbedder {
                session,
                tokenizer,
                dim,
                id,
                needs_token_type_ids,
                output_name,
            };
            embedder.run(&texts)
        })
        .await
        .map_err(|e| Error::embedding(format!("join: {e}")))?
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        &self.id
    }
}
