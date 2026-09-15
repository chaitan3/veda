//! Configuration, loaded from the process environment (and a `.env` file).
//!
//! Design rule: **docling's own settings are never wrapped or renamed.** Any
//! `DOCLING_*` / `DOCLING_RS_*` variable is read by the `docling` crate
//! directly (OCR language, model paths, execution provider, VLM endpoint, …).
//! Only RAG-layer settings use our `LS_` prefix.

use crate::error::{Error, Result};
use std::path::Path;

/// How text is embedded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedProvider {
    /// Deterministic hashing embedder — offline smoke tests only.
    Hash,
    /// OpenAI-compatible `/v1/embeddings` endpoint (llama.cpp server).
    LlamaCpp,
    /// In-process ONNX sentence encoder (feature `onnx-embed`).
    Onnx,
}

impl EmbedProvider {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hash" => Ok(EmbedProvider::Hash),
            "llamacpp" | "llama.cpp" | "llama-cpp" | "openai" => Ok(EmbedProvider::LlamaCpp),
            "onnx" => Ok(EmbedProvider::Onnx),
            other => Err(Error::config(format!(
                "LS_EMBED_PROVIDER must be hash|llamacpp|onnx, got '{other}'"
            ))),
        }
    }
}

/// Embedding backend settings.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub provider: EmbedProvider,
    /// Model name for the `llamacpp` provider (e.g. `bge-m3`).
    pub model: Option<String>,
    /// Vector dimensionality. Required for `hash`; auto-detected for `llamacpp`.
    pub dim: usize,
    /// OpenAI-compatible base URL, including `/v1`.
    pub base_url: String,
    pub api_key: Option<String>,
    /// Prefixes applied to queries / documents (for e5/bge-style models).
    pub query_prefix: Option<String>,
    pub doc_prefix: Option<String>,
    pub onnx_model: Option<String>,
    pub onnx_tokenizer: Option<String>,
    /// Cap on the encoded sequence length for the `onnx` provider. Must not
    /// exceed the model's position-embedding table (512 for BGE-style
    /// encoders); longer chunks are truncated with a warning.
    pub max_tokens: usize,
}

/// How images are embedded for cross-modal search (text→image, image→image).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageEmbedProvider {
    /// No image channel (default); the store has no image vector space.
    Off,
    /// Deterministic hashing — offline smoke tests only.
    Hash,
    /// In-process SigLIP ONNX (feature `image-embed`).
    Siglip,
}

impl ImageEmbedProvider {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "off" | "none" => Ok(ImageEmbedProvider::Off),
            "hash" => Ok(ImageEmbedProvider::Hash),
            "siglip" | "siglip2" | "siglip-2" | "onnx" => Ok(ImageEmbedProvider::Siglip),
            other => Err(Error::config(format!(
                "LS_IMAGE_EMBED_PROVIDER must be off|hash|siglip, got '{other}'"
            ))),
        }
    }
}

/// Image-embedding backend settings.
#[derive(Debug, Clone)]
pub struct ImageEmbedConfig {
    pub provider: ImageEmbedProvider,
    /// Image/text embedding dimensionality (SigLIP2 base = 768).
    pub dim: usize,
    /// Shared image+text embedding space, so a text query can match images.
    pub vision_model: Option<String>,
    pub text_model: Option<String>,
    pub tokenizer: Option<String>,
    pub preprocessor: Option<String>,
}

impl ImageEmbedConfig {
    /// Whether the image channel is active (any provider other than `Off`).
    pub fn enabled(&self) -> bool {
        self.provider != ImageEmbedProvider::Off
    }
}

/// How source documents are converted to Markdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractPipeline {
    /// Local ONNX layout/OCR/TableFormer stack (default).
    Onnx,
    /// Remote OpenAI-compatible VLM (llama.cpp / vLLM / LM Studio).
    Vlm,
}

impl ExtractPipeline {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "onnx" | "standard" | "local" => Ok(ExtractPipeline::Onnx),
            "vlm" | "remote" => Ok(ExtractPipeline::Vlm),
            other => Err(Error::config(format!(
                "LS_EXTRACT_PIPELINE must be onnx|vlm, got '{other}'"
            ))),
        }
    }
}

/// Chunking strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkerKind {
    /// Sliding window over Markdown (streaming, default).
    Window,
    /// docling's structure-aware chunker.
    Hierarchical,
    /// docling's structure-aware chunker with a token budget.
    Hybrid,
}

impl ChunkerKind {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "window" => Ok(ChunkerKind::Window),
            "hierarchical" => Ok(ChunkerKind::Hierarchical),
            "hybrid" => Ok(ChunkerKind::Hybrid),
            other => Err(Error::config(format!(
                "LS_CHUNKER must be window|hierarchical|hybrid, got '{other}'"
            ))),
        }
    }
}

/// All service configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// SQLite database path.
    pub db_path: String,
    /// Folders ingested by the CLI (from `LS_SOURCE_PATH`, a path-separated list).
    pub source_paths: Vec<String>,
    /// Folders whose image files are indexed by the image channel only — no
    /// docling conversion, no OCR, no text chunks (from `LS_IMAGE_ONLY_PATHS`).
    pub image_only_paths: Vec<String>,
    /// Root holding docling models (`.models`) and the embedder model.
    pub models_dir: String,
    pub embed: EmbedConfig,
    pub image: ImageEmbedConfig,
    pub extract: ExtractPipeline,
    /// Run docling's CodeFormulaV2 enrichment (code blocks + display formulas).
    pub code_formula: bool,
    pub chunker: ChunkerKind,
    /// Window chunk size, in words.
    pub chunk_size: usize,
    /// Window overlap fraction (0.0–0.9).
    pub chunk_overlap: f32,
    /// RRF constant used when fusing channels.
    pub rrf_k: f32,
    /// When non-empty, requests must present one of these keys.
    pub api_keys: Vec<String>,
    /// Bind address for `serve`.
    pub addr: String,
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn var_or(name: &str, default: &str) -> String {
    var(name).unwrap_or_else(|| default.to_string())
}

/// Parse a boolean env var (`1/true/yes/on` vs `0/false/no/off`), falling back
/// to `default` when unset or empty.
fn bool_or(name: &str, default: bool) -> bool {
    match var(name) {
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => default,
    }
}

/// Parse a path-separated path list (`LS_SOURCE_PATH`), keeping the platform
/// separator (`:` on Unix, `;` on Windows). Empty entries are dropped; an empty
/// or unset value falls back to `docs`.
fn parse_path_list(raw: &str) -> Vec<String> {
    let paths: Vec<String> = std::env::split_paths(raw)
        .map(|p| p.to_string_lossy().trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    if paths.is_empty() {
        vec!["docs".to_string()]
    } else {
        paths
    }
}

impl Config {
    /// Read configuration from the environment (call `dotenvy` first).
    pub fn from_env() -> Result<Self> {
        let provider = EmbedProvider::parse(&var_or("LS_EMBED_PROVIDER", "onnx"))?;
        let extract = ExtractPipeline::parse(&var_or("LS_EXTRACT_PIPELINE", "onnx"))?;
        let chunker = ChunkerKind::parse(&var_or("LS_CHUNKER", "hybrid"))?;

        let dim = var("LS_EMBED_DIM")
            .map(|v| v.parse::<usize>())
            .transpose()
            .map_err(|_| Error::config("LS_EMBED_DIM must be an integer"))?
            .unwrap_or(384);

        let chunk_overlap = var("LS_CHUNK_OVERLAP")
            .map(|v| v.parse::<f32>())
            .transpose()
            .map_err(|_| Error::config("LS_CHUNK_OVERLAP must be a number"))?
            .unwrap_or(0.05);
        if !(0.0..0.9).contains(&chunk_overlap) {
            return Err(Error::config("LS_CHUNK_OVERLAP must be in [0.0, 0.9)"));
        }

        let rrf_k = var("LS_RRF_K")
            .map(|v| v.parse::<f32>())
            .transpose()
            .map_err(|_| Error::config("LS_RRF_K must be a number"))?
            .unwrap_or(60.0);

        let models_dir = var("LS_MODELS_DIR").unwrap_or_else(|| ".models".to_string());
        let embed_dir = format!("{models_dir}/embed");
        let image_dir = format!("{models_dir}/image");

        let image = ImageEmbedConfig {
            provider: ImageEmbedProvider::parse(&var_or("LS_IMAGE_EMBED_PROVIDER", "siglip"))?,
            dim: var("LS_IMAGE_EMBED_DIM")
                .map(|v| v.parse::<usize>())
                .transpose()
                .map_err(|_| Error::config("LS_IMAGE_EMBED_DIM must be an integer"))?
                .unwrap_or(768),
            vision_model: var("LS_SIGLIP_VISION_MODEL")
                .or_else(|| Some(format!("{image_dir}/vision_model.onnx"))),
            text_model: var("LS_SIGLIP_TEXT_MODEL")
                .or_else(|| Some(format!("{image_dir}/text_model.onnx"))),
            tokenizer: var("LS_SIGLIP_TOKENIZER")
                .or_else(|| Some(format!("{image_dir}/tokenizer.json"))),
            preprocessor: var("LS_SIGLIP_PREPROCESSOR")
                .or_else(|| Some(format!("{image_dir}/preprocessor_config.json"))),
        };

        Ok(Config {
            db_path: var_or("LS_DB_PATH", "data/local-search.db"),
            source_paths: parse_path_list(&var_or("LS_SOURCE_PATH", "docs")),
            image_only_paths: parse_path_list(&var_or("LS_IMAGE_ONLY_PATHS", "")),
            models_dir,
            embed: EmbedConfig {
                provider,
                model: var("LS_EMBED_MODEL"),
                dim,
                base_url: var_or("LS_LLAMACPP_URL", "http://127.0.0.1:8080/v1"),
                api_key: var("LS_LLAMACPP_API_KEY"),
                query_prefix: var("LS_EMBED_QUERY_PREFIX"),
                doc_prefix: var("LS_EMBED_DOC_PREFIX"),
                onnx_model: var("LS_ONNX_MODEL")
                    .or_else(|| Some(format!("{embed_dir}/bge-small-en-v1.5.onnx"))),
                onnx_tokenizer: var("LS_ONNX_TOKENIZER")
                    .or_else(|| Some(format!("{embed_dir}/tokenizer.json"))),
                max_tokens: var("LS_EMBED_MAX_TOKENS")
                    .map(|v| v.parse::<usize>())
                    .transpose()
                    .map_err(|_| Error::config("LS_EMBED_MAX_TOKENS must be an integer"))?
                    .unwrap_or(512),
            },
            image,
            extract,
            code_formula: bool_or("LS_EXTRACT_CODE_FORMULA", true),
            chunker,
            chunk_size: var("LS_CHUNK_SIZE")
                .map(|v| v.parse::<usize>())
                .transpose()
                .map_err(|_| Error::config("LS_CHUNK_SIZE must be an integer"))?
                .unwrap_or(300),
            chunk_overlap,
            rrf_k,
            api_keys: var("LS_API_KEYS")
                .map(|v| {
                    v.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            addr: var_or("LS_ADDR", "127.0.0.1:8080"),
        })
    }

    /// Whether the service requires an API key.
    pub fn auth_enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }

    /// Push `LS_MODELS_DIR` into docling's own per-model environment variables.
    ///
    /// Docling resolves `.models` (and `.pdfium`) relative to the CWD by
    /// default; this only kicks in when the model tree lives elsewhere.
    /// Pre-existing `DOCLING_*` variables always win, so nothing is overridden.
    /// Call once during startup, before any worker threads are spawned.
    pub fn apply_docling_env(&self) {
        if self.models_dir == ".models" {
            return;
        }
        let dir = Path::new(&self.models_dir);
        let per_model = [
            ("DOCLING_LAYOUT_ONNX", "layout_heron.onnx"),
            ("DOCLING_TABLEFORMER_ENCODER", "tableformer/encoder.onnx"),
            ("DOCLING_TABLEFORMER_DECODER", "tableformer/decoder.onnx"),
            ("DOCLING_TABLEFORMER_BBOX", "tableformer/bbox.onnx"),
            ("DOCLING_OCR_REC_ONNX", "ocr_rec_en.onnx"),
            ("DOCLING_OCR_DICT", "en_dict.txt"),
            // The hybrid/hierarchical chunker reads this one directly
            // (chunk/mod.rs); without it a non-default model dir breaks
            // structure-aware chunking.
            ("DOCLING_CHUNK_TOKENIZER", "chunk/tokenizer.json"),
        ];
        for (key, rel) in per_model {
            let path = dir.join(rel);
            if std::env::var_os(key).is_none() && path.exists() {
                // SAFETY: startup-time, single-threaded; no other thread reads
                // the environment while this runs.
                unsafe { std::env::set_var(key, path) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_providers() {
        assert_eq!(
            EmbedProvider::parse("llama.cpp").unwrap(),
            EmbedProvider::LlamaCpp
        );
        assert_eq!(ExtractPipeline::parse("VLM").unwrap(), ExtractPipeline::Vlm);
        assert_eq!(ChunkerKind::parse("hybrid").unwrap(), ChunkerKind::Hybrid);
        assert!(EmbedProvider::parse("ollama").is_err());
    }
}
