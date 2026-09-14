//! SigLIP2 image + text embedder (in-process ONNX).
//!
//! SigLIP is trained with a **shared image+text space**, so the vision tower
//! (over image bytes) and the text tower (over a query string) live in one
//! vector space — text→image and image→image both reduce to a KNN.
//!
//! Default model: `onnx-community/siglip2-base-patch16-224-ONNX` — 768-d,
//! 224×224 input, mean/std 0.5, 64-token text. Fetch it with
//! `local-search download-models`.

use super::ImageEmbedder;
use crate::config::Config;
use crate::{Error, Result};
use async_trait::async_trait;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// SigLIP's text tower has a 64-token context; pad to it so a static graph works.
const MAX_TOKENS: usize = 64;

/// Image preprocessing derived from `preprocessor_config.json`.
#[derive(Debug, Clone, Copy)]
struct Preprocess {
    width: u32,
    height: u32,
    mean: [f32; 3],
    std: [f32; 3],
    rescale: f32,
}

impl Preprocess {
    fn from_file(path: &str) -> Self {
        let mut pre = Preprocess {
            width: 224,
            height: 224,
            mean: [0.5; 3],
            std: [0.5; 3],
            rescale: 1.0 / 255.0,
        };
        let Ok(text) = std::fs::read_to_string(docling_core::assets::resolve(path)) else {
            tracing::warn!(
                path,
                "preprocessor config unreadable; using SigLIP defaults"
            );
            return pre;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            tracing::warn!(path, "preprocessor config invalid JSON; using defaults");
            return pre;
        };
        if let Some(s) = v.get("size") {
            pre.width = s.get("width").and_then(|x| x.as_u64()).unwrap_or(224) as u32;
            pre.height = s.get("height").and_then(|x| x.as_u64()).unwrap_or(224) as u32;
        }
        if let Some(r) = v.get("rescale_factor").and_then(|x| x.as_f64()) {
            pre.rescale = r as f32;
        }
        if let Some(m) = triple(&v, "image_mean") {
            pre.mean = m;
        }
        if let Some(s) = triple(&v, "image_std") {
            pre.std = s;
        }
        pre
    }
}

fn triple(v: &serde_json::Value, key: &str) -> Option<[f32; 3]> {
    let arr = v.get(key)?.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    Some([
        arr[0].as_f64()? as f32,
        arr[1].as_f64()? as f32,
        arr[2].as_f64()? as f32,
    ])
}

/// SigLIP2 image+text embedder backed by two ONNX graphs.
#[derive(Clone)]
pub struct SigLipEmbedder {
    vision: Arc<Mutex<Session>>,
    text: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    pre: Preprocess,
    dim: usize,
    id: String,
    /// Input/output names the graphs actually declare.
    vision_input: String,
    vision_output: String,
    /// Input names the text graph declares (e.g. `input_ids`, `attention_mask`).
    text_inputs: Vec<String>,
    text_output: String,
}

impl SigLipEmbedder {
    pub fn from_config(cfg: &Config) -> Result<Self> {
        let vision = cfg.image.vision_model.clone().ok_or_else(|| {
            Error::config("LS_SIGLIP_VISION_MODEL is required for LS_IMAGE_EMBED_PROVIDER=siglip")
        })?;
        let text = cfg.image.text_model.clone().ok_or_else(|| {
            Error::config("LS_SIGLIP_TEXT_MODEL is required for LS_IMAGE_EMBED_PROVIDER=siglip")
        })?;
        let tokenizer = cfg.image.tokenizer.clone().ok_or_else(|| {
            Error::config("LS_SIGLIP_TOKENIZER is required for LS_IMAGE_EMBED_PROVIDER=siglip")
        })?;
        let preprocessor = cfg.image.preprocessor.clone().unwrap_or_default();
        Self::from_paths(&vision, &text, &tokenizer, &preprocessor, cfg.image.dim)
    }

    /// Load the two graphs and tokenizer from explicit paths.
    pub fn from_paths(
        vision_path: &str,
        text_path: &str,
        tokenizer_path: &str,
        preprocessor_path: &str,
        dim: usize,
    ) -> Result<Self> {
        let vision = load_session(vision_path)?;
        let text = load_session(text_path)?;

        let vision_input = vision
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| Error::embedding("SigLIP vision model has no inputs"))?;
        let vision_output = pick_output(
            &vision,
            &["image_embeds", "pooler_output", "image_features"],
        )?;

        let text_names: Vec<String> = text.inputs().iter().map(|i| i.name().to_string()).collect();
        let text_output = pick_output(&text, &["text_embeds", "pooler_output", "text_features"])?;
        let tokenizer = Tokenizer::from_file(docling_core::assets::resolve(tokenizer_path))
            .map_err(|e| Error::embedding(format!("loading tokenizer '{tokenizer_path}': {e}")))?;

        let pre = Preprocess::from_file(preprocessor_path);
        tracing::info!(
            vision = %vision_path,
            text = %text_path,
            input = %vision_input,
            output = %vision_output,
            text_inputs = ?text_names,
            %pre.width,
            %pre.height,
            "SigLIP image+text embedder loaded"
        );

        Ok(SigLipEmbedder {
            vision: Arc::new(Mutex::new(vision)),
            text: Arc::new(Mutex::new(text)),
            tokenizer: Arc::new(tokenizer),
            pre,
            dim,
            id: format!("siglip:{vision_path}"),
            vision_input,
            vision_output,
            text_inputs: text_names,
            text_output,
        })
    }

    fn run_images(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
        let (w, h) = (self.pre.width, self.pre.height);
        let batch = images.len();
        let mut pixels = vec![0f32; batch * 3 * h as usize * w as usize];
        for (b, bytes) in images.iter().enumerate() {
            let decoded = image::load_from_memory(bytes)
                .map_err(|e| Error::embedding(format!("decoding image: {e}")))?;
            let rgb = decoded
                .resize_exact(w, h, image::imageops::FilterType::CatmullRom)
                .to_rgb8();
            let raw = rgb.as_raw();
            let plane = h as usize * w as usize;
            for (i, px) in raw.as_chunks::<3>().0.iter().enumerate() {
                for c in 0..3 {
                    let v = px[c] as f32 * self.pre.rescale;
                    pixels[b * 3 * plane + c * plane + i] =
                        (v - self.pre.mean[c]) / self.pre.std[c];
                }
            }
        }

        let input = Tensor::from_array(([batch, 3, h as usize, w as usize], pixels))
            .map_err(|e| Error::embedding(e.to_string()))?;
        let mut session = self
            .vision
            .lock()
            .map_err(|_| Error::embedding("SigLIP vision mutex poisoned"))?;
        let outputs = session
            .run_with_options(
                vec![(self.vision_input.as_str(), SessionInputValue::from(input))],
                docling_onnx::shrink_run_options(),
            )
            .map_err(|e| Error::embedding(format!("SigLIP vision run: {e}")))?;
        let pooled = extract_pooled(&outputs[self.vision_output.as_str()], batch, None)?;
        Ok(pooled)
    }

    fn run_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::embedding(format!("tokenize: {e}")))?;
        let batch = encodings.len();
        let seq = MAX_TOKENS;
        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        for (b, enc) in encodings.iter().enumerate() {
            let toks = enc.get_ids();
            let attn = enc.get_attention_mask();
            let n = toks.len().min(seq);
            for t in 0..n {
                ids[b * seq + t] = toks[t] as i64;
                mask[b * seq + t] = attn[t] as i64;
            }
        }

        // Feed exactly the inputs the graph declares (SigLIP's text tower has no
        // attention_mask; some exports add position_ids).
        let mut input_values: Vec<(&str, SessionInputValue)> =
            Vec::with_capacity(self.text_inputs.len());
        for name in &self.text_inputs {
            let value = match name.as_str() {
                "input_ids" => Tensor::from_array(([batch, seq], ids.clone())),
                "attention_mask" => Tensor::from_array(([batch, seq], mask.clone())),
                "token_type_ids" => Tensor::from_array(([batch, seq], vec![0i64; batch * seq])),
                "position_ids" => {
                    let positions: Vec<i64> = (0..seq as i64).cycle().take(batch * seq).collect();
                    Tensor::from_array(([batch, seq], positions))
                }
                other => {
                    return Err(Error::embedding(format!(
                        "SigLIP text model needs unsupported input '{other}'"
                    )));
                }
            }
            .map_err(|e| Error::embedding(e.to_string()))?;
            input_values.push((name.as_str(), SessionInputValue::from(value)));
        }

        let mut session = self
            .text
            .lock()
            .map_err(|_| Error::embedding("SigLIP text mutex poisoned"))?;
        let outputs = session
            .run_with_options(input_values, docling_onnx::shrink_run_options())
            .map_err(|e| Error::embedding(format!("SigLIP text run: {e}")))?;
        let pooled = extract_pooled(&outputs[self.text_output.as_str()], batch, Some(&mask))?;
        Ok(pooled)
    }
}

fn load_session(path: &str) -> Result<Session> {
    let builder =
        Session::builder().map_err(|e| Error::embedding(format!("ONNX session builder: {e}")))?;
    let mut builder =
        docling_onnx::apply(builder).map_err(|e| Error::embedding(format!("embedder {e}")))?;
    builder
        .commit_from_file(docling_core::assets::resolve(path))
        .map_err(|e| Error::embedding(format!("loading ONNX model '{path}': {e}")))
}

fn pick_output(session: &Session, preferred: &[&str]) -> Result<String> {
    let names: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    preferred
        .iter()
        .find(|n| names.contains(n))
        .or_else(|| names.first())
        .map(|s| s.to_string())
        .ok_or_else(|| Error::embedding("SigLIP graph has no outputs"))
}

/// Read `[batch, dim]` (pooled) or `[batch, seq, dim]` (mean-pooled) and
/// L2-normalize each row. `mask` (when present) weights the mean.
fn extract_pooled(
    value: &ort::value::DynValue,
    batch: usize,
    mask: Option<&[i64]>,
) -> Result<Vec<Vec<f32>>> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::embedding(e.to_string()))?;
    let dims: &[i64] = shape;
    if dims.len() < 2 {
        return Err(Error::embedding(format!(
            "SigLIP output has unexpected rank {} (shape {dims:?})",
            dims.len()
        )));
    }
    let hidden = *dims.last().unwrap() as usize;
    let mut out = Vec::with_capacity(batch);
    if dims.len() == 2 {
        for b in 0..batch {
            let mut v = data[b * hidden..(b + 1) * hidden].to_vec();
            crate::math::normalize(&mut v);
            out.push(v);
        }
    } else {
        let seq = dims[1] as usize;
        for b in 0..batch {
            let mut v = vec![0f32; hidden];
            let mut count = 0.0f32;
            for t in 0..seq {
                if let Some(m) = mask
                    && m[b * seq + t] == 0
                {
                    continue;
                }
                let base = (b * seq + t) * hidden;
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
    }
    Ok(out)
}

#[async_trait]
impl ImageEmbedder for SigLipEmbedder {
    async fn embed_images(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let me = self.clone();
        let images = images.to_vec();
        tokio::task::spawn_blocking(move || me.run_images(&images))
            .await
            .map_err(|e| Error::embedding(format!("join: {e}")))?
    }

    async fn embed_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let me = self.clone();
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || me.run_text(&texts))
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
