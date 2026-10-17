//! Per-stage fingerprints: a hash of everything that determines a stage's
//! output (config + model identity), so reprocessing can be scoped to what
//! actually changed.
//!
//! Model identity prefers an entry in `<models_dir>/models.lock` — a JSON map
//! of model path → sha256, written by `download-models`. When absent it falls
//! back to `path:size`, which is cheap but misses a same-path/same-size swap.

use crate::config::Config;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

/// Fingerprints for the three ingest stages.
#[derive(Debug, Clone)]
pub struct Fingerprints {
    pub extract: String,
    pub text_embed: String,
    pub image_embed: String,
}

impl Fingerprints {
    /// The fingerprint for a stage name, if it is one.
    pub fn get(&self, stage: &str) -> Option<&str> {
        match stage {
            "extract" => Some(&self.extract),
            "text_embed" => Some(&self.text_embed),
            "image_embed" => Some(&self.image_embed),
            _ => None,
        }
    }
}

/// `DOCLING_*` variables whose values change the converted output.
const DOCLING_ENV: &[&str] = &[
    "DOCLING_RS_OCR_LANG",
    "DOCLING_RS_OCR_MODE",
    "DOCLING_RS_OCR_SCALE",
    "DOCLING_RS_OCR_ORIENTATION",
    "DOCLING_RS_OCR_SESSIONS",
    "DOCLING_RS_FP32",
    "DOCLING_RS_EP",
    "DOCLING_LAYOUT_ONNX",
    "DOCLING_TABLEFORMER_ENCODER",
    "DOCLING_TABLEFORMER_DECODER",
    "DOCLING_TABLEFORMER_BBOX",
    "DOCLING_OCR_REC_ONNX",
    "DOCLING_OCR_DICT",
    "DOCLING_CODE_FORMULA_DIR",
    "DOCLING_RS_VLM_MODEL",
    "DOCLING_RS_VLM_PROMPT",
    "DOCLING_RS_VLM_ENDPOINT",
];

/// Compute the fingerprints from the effective configuration.
pub fn compute(cfg: &Config) -> Fingerprints {
    let lock = load_lock(&cfg.models_dir);

    let mut extract = vec![
        format!("extract={:?}", cfg.extract),
        format!("code_formula={}", cfg.code_formula),
        format!("chunker={:?}", cfg.chunker),
        format!("chunk_size={}", cfg.chunk_size),
        format!("chunk_overlap={}", cfg.chunk_overlap),
    ];
    for name in DOCLING_ENV {
        if let Ok(value) = std::env::var(name) {
            extract.push(format!("{name}={value}"));
        }
    }

    Fingerprints {
        extract: fingerprint(&extract),
        text_embed: fingerprint(&[
            format!("provider={:?}", cfg.embed.provider),
            format!("model={:?}", cfg.embed.model),
            format!("dim={}", cfg.embed.dim),
            format!("query_prefix={:?}", cfg.embed.query_prefix),
            format!("doc_prefix={:?}", cfg.embed.doc_prefix),
            format!("base_url={}", cfg.embed.base_url),
            format!(
                "onnx_model={}",
                model_id(&lock, cfg.embed.onnx_model.as_deref())
            ),
            format!(
                "onnx_tokenizer={}",
                model_id(&lock, cfg.embed.onnx_tokenizer.as_deref())
            ),
        ]),
        image_embed: fingerprint(&[
            format!("provider={:?}", cfg.image.provider),
            format!("dim={}", cfg.image.dim),
            format!(
                "vision={}",
                model_id(&lock, cfg.image.vision_model.as_deref())
            ),
            format!("text={}", model_id(&lock, cfg.image.text_model.as_deref())),
            format!(
                "tokenizer={}",
                model_id(&lock, cfg.image.tokenizer.as_deref())
            ),
            format!(
                "preprocessor={}",
                model_id(&lock, cfg.image.preprocessor.as_deref())
            ),
        ]),
    }
}

/// Identity of everything that can change a *conversion*: the binary build,
/// the docling/pdfium assets and the `DOCLING_*` environment. A failed
/// document is retried only when the file changes or this identity changes, so
/// a rebuilt container (a pdfium fix, a better OCR model) re-attempts it
/// without touching successfully converted documents.
pub fn extractor_identity(cfg: &Config) -> String {
    let lock = load_lock(&cfg.models_dir);
    let mut parts = vec![
        format!("version={}", env!("CARGO_PKG_VERSION")),
        format!(
            "build={}",
            option_env!("BUILD_GIT_SHA").unwrap_or("unknown")
        ),
        format!("exe={}", exe_id()),
        format!("pdfium={}", pdfium_id()),
    ];
    for name in DOCLING_ENV {
        if let Ok(value) = std::env::var(name) {
            parts.push(format!("{name}={value}"));
        }
    }
    for (path, hash) in &lock {
        if is_extraction_model(path) {
            parts.push(format!("model:{path}={hash}"));
        }
    }
    parts.sort();
    fingerprint(&parts)
}

/// Model files under `<models_dir>` that affect conversion (docling layout,
/// OCR, TableFormer, the chunk tokenizer and the enrichment models) — the
/// embedder assets are deliberately excluded.
fn is_extraction_model(path: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "layout",
        "ocr_",
        "ppocr",
        "en_dict",
        "tableformer/",
        "chunk/",
        "code_formula",
    ];
    PREFIXES.iter().any(|p| path.starts_with(p))
}

/// Identity of the pdfium build. It lives at `.pdfium/` (CWD-relative, outside
/// the models tree and so not in `models.lock`), and a fix there is exactly the
/// kind of rebuild that should re-attempt failed PDFs. Hashed once per process.
fn pdfium_id() -> String {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let path = Path::new(".pdfium/lib/libpdfium.so");
        match std::fs::read(path) {
            Ok(bytes) => {
                let mut hasher = Sha256::new();
                hasher.update(&bytes);
                format!("sha256:{}", hex::encode(&hasher.finalize()[..16]))
            }
            Err(_) => match std::fs::metadata(path) {
                Ok(meta) => format!("size:{}", meta.len()),
                Err(_) => "missing".to_string(),
            },
        }
    })
    .clone()
}

/// Identity of the running binary (size + mtime). Unlike the git sha this also
/// moves when the image is rebuilt without `.git` present, and stays put across
/// a plain restart of the same image.
fn exe_id() -> String {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(
        || match std::env::current_exe().and_then(std::fs::metadata) {
            Ok(meta) => {
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0);
                format!("{}:{}", meta.len(), mtime)
            }
            Err(_) => "unknown".to_string(),
        },
    )
    .clone()
}

/// A short (128-bit) hex digest of the joined parts.
fn fingerprint(parts: &[String]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update(b"\n");
    }
    hex::encode(&hasher.finalize()[..16])
}

/// `<models_dir>/models.lock`: a JSON map of model path → sha256.
pub fn load_lock(models_dir: &str) -> BTreeMap<String, String> {
    let path = Path::new(models_dir).join("models.lock");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Model identity: the lock's sha256 when present, else `path:size`.
pub fn model_id(lock: &BTreeMap<String, String>, path: Option<&str>) -> String {
    let Some(path) = path else {
        return "none".to_string();
    };
    if let Some(hash) = lock.get(path) {
        return format!("sha256:{hash}");
    }
    match std::fs::metadata(path) {
        Ok(meta) => format!("{path}:{}", meta.len()),
        Err(_) => format!("{path}:missing"),
    }
}
