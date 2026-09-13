//! Fetch the default local model set.
//!
//! Implemented in Rust (no remote shell script): it downloads docling's native
//! assets (pdfium + layout/OCR/TableFormer ONNX) from the docling.rs releases,
//! and the default ONNX text embedder (bge-small-en-v1.5) from Hugging Face.
//! Mirrors are honoured via `DOCLING_RS_MODELS_URL` / `LS_ONNX_MODELS_URL`.
//!
//! Idempotent: files already on disk are skipped.

use crate::config::Config;
use crate::{Error, Result};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

const DEFAULT_MODELS_BASE: &str =
    "https://github.com/docling-project/docling.rs/releases/download/models-v1";
const DEFAULT_EMBED_BASE: &str = "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main";
const DEFAULT_IMAGE_BASE: &str =
    "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main";

struct Asset {
    dest: PathBuf,
    urls: Vec<String>,
    optional: bool,
}

/// Download the default model set. `with_embed` also fetches the local ONNX
/// text embedder (bge-small-en-v1.5, ~130 MB); `with_images` the SigLIP2
/// image+text towers (~700 MB) used by `LS_IMAGE_EMBED_PROVIDER=siglip`.
pub async fn download_models(cfg: &Config, with_embed: bool, with_images: bool) -> Result<()> {
    let base = env_or("DOCLING_RS_MODELS_URL", DEFAULT_MODELS_BASE);
    let embed_base = env_or("LS_ONNX_MODELS_URL", DEFAULT_EMBED_BASE);
    let image_base = env_or("LS_IMAGE_MODELS_URL", DEFAULT_IMAGE_BASE);
    let models = PathBuf::from(&cfg.models_dir);

    let docling = |rel: &str| models.join(rel);
    let release = |file: &str| format!("{base}/{file}");

    let mut assets = vec![
        // pdfium: docling's pinned Linux x64 build (PDF rasterization).
        Asset {
            dest: PathBuf::from(".pdfium/lib/libpdfium.so"),
            urls: vec![release("libpdfium.so")],
            optional: false,
        },
        asset(docling("layout_heron.onnx"), release("layout_heron.onnx")),
        asset(docling("ocr_rec.onnx"), release("ocr_rec.onnx")),
        asset(docling("ppocr_keys_v1.txt"), release("ppocr_keys_v1.txt")),
        Asset {
            dest: docling("ocr_rec_en.onnx"),
            urls: vec![
                release("ocr_rec_en.onnx"),
                "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx"
                    .to_string(),
            ],
            optional: false,
        },
        Asset {
            dest: docling("en_dict.txt"),
            urls: vec![
                release("en_dict.txt"),
                "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/en_dict.txt"
                    .to_string(),
            ],
            optional: false,
        },
        asset(docling("tableformer/encoder.onnx"), release("encoder.onnx")),
        asset(docling("tableformer/decoder.onnx"), release("decoder.onnx")),
        asset(docling("tableformer/bbox.onnx"), release("bbox.onnx")),
        // Optional: graph sidecars and the hybrid chunker's tokenizer.
        optional(docling("tableformer/encoder.onnx.data"), release("encoder.onnx.data")),
        optional(docling("tableformer/decoder.onnx.data"), release("decoder.onnx.data")),
        optional(docling("tableformer/bbox.onnx.data"), release("bbox.onnx.data")),
        Asset {
            dest: docling("chunk/tokenizer.json"),
            urls: vec![
                release("chunk_tokenizer.json"),
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
                    .to_string(),
            ],
            optional: true,
        },
        // Optional: INT8 CPU models (faster layout inference).
        optional(docling("layout_heron_int8.onnx"), release("layout_heron_int8.onnx")),
        optional(docling("tableformer/decoder_int8.onnx"), release("decoder_int8.onnx")),
    ];

    if with_embed {
        assets.push(Asset {
            dest: docling("embed/bge-small-en-v1.5.onnx"),
            urls: vec![format!("{embed_base}/onnx/model.onnx")],
            optional: false,
        });
        assets.push(Asset {
            dest: docling("embed/tokenizer.json"),
            urls: vec![format!("{embed_base}/tokenizer.json")],
            optional: false,
        });
    }

    if with_images {
        let image = |rel: &str| models.join("image").join(rel);
        assets.push(Asset {
            dest: image("vision_model.onnx"),
            urls: vec![format!("{image_base}/onnx/vision_model.onnx")],
            optional: false,
        });
        assets.push(Asset {
            dest: image("text_model.onnx"),
            urls: vec![format!("{image_base}/onnx/text_model.onnx")],
            optional: false,
        });
        assets.push(Asset {
            dest: image("tokenizer.json"),
            urls: vec![format!("{image_base}/tokenizer.json")],
            optional: false,
        });
        assets.push(Asset {
            dest: image("preprocessor_config.json"),
            urls: vec![format!("{image_base}/preprocessor_config.json")],
            optional: false,
        });
    }

    let client = reqwest::Client::builder()
        .user_agent("local-search/0.1")
        .build()
        .map_err(|e| Error::Other(format!("http client: {e}")))?;

    let total = assets.len();
    println!(
        "fetching {total} asset(s) into '{}' (pdfium under .pdfium/)",
        cfg.models_dir
    );
    let mut fetched = 0usize;
    for a in &assets {
        if fetch_asset(&client, a).await? {
            fetched += 1;
        }
    }
    println!(
        "done: {fetched} downloaded, {} already present",
        total - fetched
    );
    Ok(())
}

fn asset(dest: PathBuf, url: String) -> Asset {
    Asset {
        dest,
        urls: vec![url],
        optional: false,
    }
}

fn optional(dest: PathBuf, url: String) -> Asset {
    Asset {
        dest,
        urls: vec![url],
        optional: true,
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Returns `true` if it downloaded, `false` if it already existed.
async fn fetch_asset(client: &reqwest::Client, asset: &Asset) -> Result<bool> {
    if asset.dest.exists() {
        tracing::debug!(path = %asset.dest.display(), "already present");
        return Ok(false);
    }
    if let Some(parent) = asset.dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut last = String::new();
    for url in &asset.urls {
        match download(client, url, &asset.dest).await {
            Ok(()) => {
                println!("  > {}", asset.dest.display());
                return Ok(true);
            }
            Err(e) => {
                last = e.to_string();
                tracing::warn!(url = %url, error = %e, "download failed; trying next mirror");
            }
        }
    }

    if asset.optional {
        tracing::warn!(path = %asset.dest.display(), "optional asset unavailable: {last}");
        Ok(false)
    } else {
        Err(Error::Other(format!(
            "could not fetch {} from any mirror: {last}",
            asset.dest.display()
        )))
    }
}

async fn download(client: &reqwest::Client, url: &str, dest: &Path) -> Result<()> {
    let mut resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| Error::Other(format!("{url}: {e}")))?
        .error_for_status()
        .map_err(|e| Error::Other(format!("{url}: {e}")))?;

    let tmp = dest.with_extension("part");
    let mut file = tokio::fs::File::create(&tmp).await?;
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| Error::Other(format!("{url}: {e}")))?
    {
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    drop(file);
    tokio::fs::rename(&tmp, dest).await?;
    Ok(())
}
