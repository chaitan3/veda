//! EmbeddingGemma 2 image embedder — the `eg2` image provider.
//!
//! Two graphs run per image:
//!
//! 1. `vision_encoder.onnx` takes **patchified** pixels (`[P, 768]`, where a
//!    patch is 16×16×3 flattened in HWC order) plus `pixel_position_ids`
//!    (`[P, 2]`, `(x, y)` of each patch) and returns `[N, 512]` image features.
//! 2. `model.onnx` (the same text graph the `eg2` text provider uses) consumes
//!    those features alongside a placeholder token sequence
//!    `[bos, boi, <|image|> × N, eoi, eos]` and returns the pooled, normalised
//!    768-d embedding.
//!
//! The preprocessing mirrors `Gemma4ImageProcessor` exactly — verified against
//! the reference processor's tensors and published similarities
//! (`examples/eg2_parity.py`): aspect-ratio-preserving resize to at most
//! `max_soft_tokens × pooling²` patches with both sides a multiple of
//! `pooling × patch_size`, rescale to [0,1], patchify, pad to the patch budget
//! with position `-1`, and count `N = patches / pooling²`.
//!
//! fp32 only, like the text half: the model card is explicit that fp16
//! produces NaN or silently degraded embeddings.

use crate::config::Config;
use crate::embed::eg2::forward;
use crate::embed::image::ImageEmbedder;
use crate::{Error, Result};
use async_trait::async_trait;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::{Arc, Mutex, OnceLock};
use tokenizers::Tokenizer;

/// Token ids the EG2 text graph expects around the image placeholder.
const BOS: i64 = 2;
const BOI: i64 = 255999;
const IMAGE: i64 = 258880;
const EOI: i64 = 258882;
const EOS: i64 = 1;

/// Preprocessing parameters, read from `processor_config.json` with the
/// model's defaults as a fallback.
#[derive(Debug, Clone, Copy)]
pub struct Preprocess {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub max_soft_tokens: usize,
    pub rescale: f32,
}

impl Default for Preprocess {
    fn default() -> Self {
        Preprocess {
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: 280,
            rescale: 1.0 / 255.0,
        }
    }
}

impl Preprocess {
    /// Patch budget: soft tokens × pooling².
    pub fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// Read the image-processor settings from a `processor_config.json`.
    pub fn from_config_file(path: &str) -> Self {
        let mut p = Preprocess::default();
        let Ok(raw) = std::fs::read_to_string(docling_core::assets::resolve(path)) else {
            tracing::warn!(path, "EG2 processor config unreadable; using defaults");
            return p;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) else {
            tracing::warn!(path, "EG2 processor config is not JSON; using defaults");
            return p;
        };
        let ip = &json["image_processor"];
        if let Some(v) = ip["patch_size"].as_u64() {
            p.patch_size = v as usize;
        }
        if let Some(v) = ip["pooling_kernel_size"].as_u64() {
            p.pooling_kernel_size = v as usize;
        }
        if let Some(v) = ip["max_soft_tokens"].as_u64() {
            p.max_soft_tokens = v as usize;
        }
        if let Some(v) = ip["rescale_factor"].as_f64() {
            p.rescale = v as f32;
        }
        p
    }

    fn valid_soft_tokens(&self) -> bool {
        matches!(self.max_soft_tokens, 70 | 140 | 280 | 560 | 1120)
    }
}

/// One preprocessed image: padded patches, padded positions and the real
/// (unpadded) soft-token count.
#[derive(Debug, Clone)]
pub struct Preprocessed {
    pub patches: Vec<f32>,
    pub positions: Vec<i64>,
    pub padded_patches: usize,
    pub soft_tokens: usize,
    pub grid: (usize, usize),
}

/// Resize dimensions per `get_aspect_ratio_preserving_size` (transformers).
pub fn target_size(height: usize, width: usize, p: &Preprocess) -> Result<(usize, usize)> {
    let patch = p.patch_size as f64;
    let side = (p.pooling_kernel_size * p.patch_size) as f64;
    let max_patches = p.max_patches() as f64;
    let total_px = (height * width) as f64;
    let target_px = max_patches * patch * patch;
    let factor = (target_px / total_px).sqrt();
    let ideal_h = factor * height as f64;
    let ideal_w = factor * width as f64;
    let mut target_h = (ideal_h / side).floor() * side;
    let mut target_w = (ideal_w / side).floor() * side;
    if target_h == 0.0 && target_w == 0.0 {
        return Err(Error::embedding(
            "EG2 preprocessing: image resizes to 0×0 (too extreme an aspect ratio)",
        ));
    }
    let max_side_length =
        (max_patches / (p.pooling_kernel_size * p.pooling_kernel_size) as f64) * side;
    if target_h == 0.0 {
        target_h = side;
        target_w = ((width as f64 / height as f64).floor() * side).min(max_side_length);
    } else if target_w == 0.0 {
        target_w = side;
        target_h = ((height as f64 / width as f64).floor() * side).min(max_side_length);
    }
    if target_h * target_w > target_px {
        return Err(Error::embedding(format!(
            "EG2 preprocessing: [{height}x{width}] would resize to \
             [{target_h}x{target_w}], above the patch budget"
        )));
    }
    Ok((target_h as usize, target_w as usize))
}

/// Decode + resize + rescale + patchify + pad, as the reference processor does
/// for a single image.
pub fn preprocess_image(bytes: &[u8], p: &Preprocess) -> Result<Preprocessed> {
    let decoded = image::load_from_memory(bytes)
        .map_err(|e| Error::embedding(format!("decoding image: {e}")))?;
    let (h, w) = (decoded.height() as usize, decoded.width() as usize);
    let (target_h, target_w) = target_size(h, w, p)?;
    let rgb = if target_h == h && target_w == w {
        decoded.to_rgb8()
    } else {
        // Bicubic, matching torchvision's `resize(..., BICUBIC, antialias=True)`
        // as closely as the `image` crate's CatmullRom filter allows.
        decoded
            .resize_exact(
                target_w as u32,
                target_h as u32,
                image::imageops::FilterType::CatmullRom,
            )
            .to_rgb8()
    };
    let raw = rgb.as_raw();
    let (patch, pool) = (p.patch_size, p.pooling_kernel_size);
    let patch_h = target_h / patch;
    let patch_w = target_w / patch;
    let patches_real = patch_h * patch_w;
    if patches_real == 0 || !patches_real.is_multiple_of(pool * pool) {
        return Err(Error::embedding(format!(
            "EG2 preprocessing: {patches_real} patches is not a multiple of pooling²"
        )));
    }
    let soft_tokens = patches_real / (pool * pool);
    let padded_patches = p.max_patches();

    // Patch layout: (patch_row, patch_col) row-major, each patch flattened as
    // (row, col, channel) — the `permute(1,3,2,4,0)` of the reference.
    let patch_len = patch * patch * 3;
    let mut patches = vec![0f32; padded_patches * patch_len];
    for py in 0..patch_h {
        for px in 0..patch_w {
            let base = (py * patch_w + px) * patch_len;
            for r in 0..patch {
                for c in 0..patch {
                    let y = py * patch + r;
                    let x = px * patch + c;
                    let src = (y * target_w + x) * 3;
                    let dst = base + (r * patch + c) * 3;
                    for ch in 0..3 {
                        patches[dst + ch] = raw[src + ch] as f32 * p.rescale;
                    }
                }
            }
        }
    }
    // Positions are (x, y); padding uses -1 so the graph ignores it.
    let mut positions = vec![-1i64; padded_patches * 2];
    for py in 0..patch_h {
        for px in 0..patch_w {
            let idx = py * patch_w + px;
            positions[idx * 2] = px as i64;
            positions[idx * 2 + 1] = py as i64;
        }
    }
    Ok(Preprocessed {
        patches,
        positions,
        padded_patches,
        soft_tokens,
        grid: (patch_h, patch_w),
    })
}

struct Inner {
    vision: Mutex<Session>,
    /// The text graph is only needed for text→image queries; loaded on first
    /// use so a unified pipeline (which embeds the query once, with the text
    /// embedder) never pays for a second copy of it.
    text: OnceLock<Mutex<Session>>,
    text_model: String,
    tokenizer: Tokenizer,
    pre: Preprocess,
    dim: usize,
    id: String,
    query_prefix: String,
    max_tokens: usize,
}

/// Embedder backed by the EG2 vision + text graphs.
#[derive(Clone)]
pub struct Eg2ImageEmbedder {
    inner: Arc<Inner>,
}

impl Eg2ImageEmbedder {
    pub fn from_config(cfg: &Config) -> Result<Self> {
        let vision = cfg.image.eg2_vision_model.clone().ok_or_else(|| {
            Error::config("LS_EG2_VISION_MODEL is required when LS_IMAGE_EMBED_PROVIDER=eg2")
        })?;
        let processor = cfg
            .image
            .eg2_processor
            .clone()
            .unwrap_or_else(|| format!("{}/eg2/processor_config.json", cfg.models_dir));
        let mut pre = Preprocess::from_config_file(&processor);
        if let Some(soft) = cfg.image.soft_tokens {
            pre.max_soft_tokens = soft;
        }
        if !pre.valid_soft_tokens() {
            return Err(Error::config(format!(
                "EG2 max_soft_tokens must be one of 70|140|280|560|1120, got {}",
                pre.max_soft_tokens
            )));
        }

        let builder = Session::builder().map_err(|e| Error::embedding(format!("{e}")))?;
        let mut builder =
            docling_onnx::apply(builder).map_err(|e| Error::embedding(format!("eg2 {e}")))?;
        let session = builder
            .commit_from_file(docling_core::assets::resolve(&vision))
            .map_err(|e| Error::embedding(format!("loading EG2 vision graph '{vision}': {e}")))?;

        let text_model = cfg.embed.eg2_model.clone().ok_or_else(|| {
            Error::config(
                "LS_EG2_TEXT_MODEL is required with LS_IMAGE_EMBED_PROVIDER=eg2 \
                 (text→image queries use the shared text tower)",
            )
        })?;
        let tokenizer_path = cfg.embed.eg2_tokenizer.clone().ok_or_else(|| {
            Error::config("LS_EG2_TOKENIZER is required with LS_IMAGE_EMBED_PROVIDER=eg2")
        })?;
        let tokenizer = Tokenizer::from_file(docling_core::assets::resolve(&tokenizer_path))
            .map_err(|e| Error::embedding(format!("loading tokenizer '{tokenizer_path}': {e}")))?;

        tracing::info!(
            vision = %vision,
            soft_tokens = pre.max_soft_tokens,
            patch = pre.patch_size,
            pooling = pre.pooling_kernel_size,
            "EG2 vision encoder loaded (fp32)"
        );
        Ok(Eg2ImageEmbedder {
            inner: Arc::new(Inner {
                vision: Mutex::new(session),
                text: OnceLock::new(),
                text_model,
                tokenizer,
                pre,
                dim: cfg.image.dim,
                id: format!("eg2-vision:{vision}"),
                query_prefix: cfg
                    .embed
                    .query_prefix
                    .clone()
                    .unwrap_or_else(|| crate::embed::eg2::DEFAULT_QUERY_PREFIX.to_string()),
                max_tokens: cfg.embed.max_tokens,
            }),
        })
    }

    fn text_session(&self) -> Result<&Mutex<Session>> {
        if let Some(s) = self.inner.text.get() {
            return Ok(s);
        }
        let builder = Session::builder().map_err(|e| Error::embedding(format!("{e}")))?;
        let mut builder =
            docling_onnx::apply(builder).map_err(|e| Error::embedding(format!("eg2 {e}")))?;
        let session = builder
            .commit_from_file(docling_core::assets::resolve(&self.inner.text_model))
            .map_err(|e| Error::embedding(format!("loading EG2 text graph: {e}")))?;
        let _ = self.inner.text.set(Mutex::new(session));
        Ok(self.inner.text.get().expect("just set"))
    }

    fn run_images(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(images.len());
        for bytes in images {
            let pre = preprocess_image(bytes, &self.inner.pre)?;
            // 1. vision encoder
            let (rows, features) = {
                let mut vision = self
                    .inner
                    .vision
                    .lock()
                    .map_err(|_| Error::embedding("EG2 vision mutex poisoned"))?;
                let px = Tensor::from_array((
                    [
                        1usize,
                        pre.padded_patches,
                        self.inner.pre.patch_size * self.inner.pre.patch_size * 3,
                    ],
                    pre.patches.clone(),
                ))
                .map_err(|e| Error::embedding(e.to_string()))?;
                let pos =
                    Tensor::from_array(([1usize, pre.padded_patches, 2], pre.positions.clone()))
                        .map_err(|e| Error::embedding(e.to_string()))?;
                let outputs = vision
                    .run_with_options(
                        ort::inputs!["pixel_values" => px, "pixel_position_ids" => pos],
                        docling_onnx::shrink_run_options(),
                    )
                    .map_err(|e| Error::embedding(format!("EG2 vision run: {e}")))?;
                let value = outputs
                    .get("image_features")
                    .ok_or_else(|| Error::embedding("EG2 vision run returned no image_features"))?;
                let (shape, data) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::embedding(e.to_string()))?;
                let cols = *shape.last().unwrap_or(&0) as usize;
                if cols != 512 {
                    return Err(Error::embedding(format!(
                        "EG2 vision output has {cols} columns (expected 512)"
                    )));
                }
                (shape[0] as usize, data.to_vec())
            };
            // 2. text graph with the image placeholder tokens
            let mut ids = Vec::with_capacity(rows + 4);
            ids.push(BOS);
            ids.push(BOI);
            ids.extend(std::iter::repeat_n(IMAGE, rows));
            ids.push(EOI);
            ids.push(EOS);
            let seq = ids.len();
            let mask = vec![1i64; seq];
            let mut session = self
                .text_session()?
                .lock()
                .map_err(|_| Error::embedding("EG2 text mutex poisoned"))?;
            out.extend(forward(
                &mut session,
                1,
                seq,
                ids,
                mask,
                Some((rows, features)),
                self.inner.dim,
            )?);
        }
        Ok(out)
    }

    fn run_query_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .inner
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::embedding(format!("tokenize: {e}")))?;
        let (batch, seq, ids, mask, _types) =
            crate::embed::onnx::pad_batch(&encodings, self.inner.max_tokens);
        let mut session = self
            .text_session()?
            .lock()
            .map_err(|_| Error::embedding("EG2 text mutex poisoned"))?;
        forward(&mut session, batch, seq, ids, mask, None, self.inner.dim)
    }
}

#[async_trait]
impl ImageEmbedder for Eg2ImageEmbedder {
    /// The image embedder's text tower (used by `embed_query_text`).
    async fn embed_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let me = self.clone();
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || me.run_query_text(&texts))
            .await
            .map_err(|e| Error::embedding(format!("join: {e}")))?
    }

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

    async fn embed_query_text(&self, query: &str) -> Result<Vec<f32>> {
        let me = self.clone();
        let text = format!("{}{query}", self.inner.query_prefix);
        let mut v =
            tokio::task::spawn_blocking(move || me.run_query_text(std::slice::from_ref(&text)))
                .await
                .map_err(|e| Error::embedding(format!("join: {e}")))??;
        v.pop()
            .ok_or_else(|| Error::embedding("embedder returned no vector"))
    }

    fn dim(&self) -> usize {
        self.inner.dim
    }

    fn id(&self) -> &str {
        &self.inner.id
    }
}
