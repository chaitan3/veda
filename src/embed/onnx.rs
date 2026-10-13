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
    /// Hard cap on encoded sequence length. Position-embedding tables are
    /// fixed at export time (512 for BGE-style encoders) and the tokenizer
    /// ships no truncation config, so an over-long chunk would fail inside
    /// the graph (`/embeddings/Add_1` broadcast error) rather than at input.
    max_tokens: usize,
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
            max_tokens: cfg.embed.max_tokens,
        })
    }

    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::embedding(format!("tokenize: {e}")))?;
        let (batch, seq, ids, mask, types) = pad_batch(&encodings, self.max_tokens);

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
            session.run_with_options(
                ort::inputs![
                    "input_ids" => id_tensor,
                    "attention_mask" => mask_tensor,
                    "token_type_ids" => type_tensor,
                ],
                docling_onnx::shrink_run_options(),
            )
        } else {
            session.run_with_options(
                ort::inputs![
                    "input_ids" => id_tensor,
                    "attention_mask" => mask_tensor,
                ],
                docling_onnx::shrink_run_options(),
            )
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
        let max_tokens = self.max_tokens;
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
                max_tokens,
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

/// Right-pad a tokenized batch to a uniform sequence length, truncating every
/// sequence to `max_tokens`.
///
/// The cap is not optional: encoders carry a fixed position-embedding table
/// (`/embeddings/Add_1` in BGE graphs, 512 rows for bge-small), and the
/// tokenizer JSON ships neither truncation nor a model max length — so an
/// over-long chunk fails *inside the graph* as a broadcast error instead of
/// at the input. Chunk sizes are counted in words, and tables / OCR noise /
/// code tokenize to far more BPE tokens than words, so truncation is the
/// embedder's last line of defence.
///
/// Returns `(batch, seq, input_ids, attention_mask, token_type_ids)`; pads
/// with zeros (mask 0) so padded positions are ignored by the model.
pub(crate) fn pad_batch(
    encodings: &[tokenizers::Encoding],
    max_tokens: usize,
) -> (usize, usize, Vec<i64>, Vec<i64>, Vec<i64>) {
    let batch = encodings.len();
    if batch == 0 {
        return (0, 1, Vec::new(), Vec::new(), Vec::new());
    }
    let longest = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);
    let seq = longest.min(max_tokens).max(1);
    let truncated = encodings
        .iter()
        .filter(|e| e.get_ids().len() > max_tokens)
        .count() as u64;
    if truncated > 0 {
        crate::embed::note_truncated(truncated);
        tracing::warn!(
            longest_tokens = longest,
            max_tokens = max_tokens,
            truncated,
            "truncating over-long chunk(s) for embedding; consider a smaller LS_CHUNK_SIZE"
        );
    }

    let mut ids = vec![0i64; batch * seq];
    let mut mask = vec![0i64; batch * seq];
    let mut types = vec![0i64; batch * seq];
    for (b, enc) in encodings.iter().enumerate() {
        let eids = enc.get_ids();
        let emask = enc.get_attention_mask();
        let etypes = enc.get_type_ids();
        // Only real positions; the rest stays zero-padded.
        let n = seq.min(eids.len()).min(emask.len()).min(etypes.len());
        for t in 0..n {
            ids[b * seq + t] = eids[t] as i64;
            mask[b * seq + t] = emask[t] as i64;
            types[b * seq + t] = etypes[t] as i64;
        }
    }
    (batch, seq, ids, mask, types)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tokenizer whose sequences exceed the cap must be truncated, with the
    /// mask cut at exactly the cap. Uses the real bge tokenizer when present.
    #[test]
    fn pad_batch_truncates_long_sequences() {
        let Ok(tokenizer) = Tokenizer::from_file(".models/embed/tokenizer.json") else {
            return; // models not downloaded; nothing to assert here
        };
        // 300 words of token-heavy noise → comfortably over 512 BPE tokens.
        let long_text = "no. 12345678 ".repeat(600);
        let short_text = "hello world";
        let encodings = tokenizer
            .encode_batch(vec![long_text, short_text.to_string()], true)
            .unwrap();
        assert!(encodings[0].get_ids().len() > 512);
        assert!(encodings[1].get_ids().len() < 512);

        let before = crate::embed::truncated_sequences();
        let (batch, seq, ids, mask, types) = pad_batch(&encodings, 512);
        assert!(
            crate::embed::truncated_sequences() > before,
            "truncation must be counted"
        );
        assert_eq!(batch, 2);
        assert_eq!(seq, 512);
        assert_eq!(ids.len(), 2 * 512);
        assert_eq!(mask.len(), 2 * 512);
        // First sequence: full 512 real tokens; second: padded tail.
        assert_eq!(mask[0], 1);
        assert_eq!(mask[511], 1);
        assert_eq!(mask[512 + encodings[1].get_ids().len()], 0);
        assert_eq!(types[512], 0, "token_type_ids are per-token, not zeros");
    }
}
