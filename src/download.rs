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
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

const DEFAULT_MODELS_BASE: &str =
    "https://github.com/docling-project/docling.rs/releases/download/models-v1";
const DEFAULT_EMBED_BASE: &str = "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main";
const DEFAULT_IMAGE_BASE: &str =
    "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main";
/// EmbeddingGemma 2 (Apache-2.0) — the unified text+image space used by the
/// `eg2` providers. fp32 only: the model card is explicit that fp16 returns
/// NaN or silently degraded embeddings.
const DEFAULT_EG2_BASE: &str =
    "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main";

struct Asset {
    dest: PathBuf,
    urls: Vec<String>,
    optional: bool,
}

/// Download the default model set. `with_embed` fetches the local ONNX text
/// embedder (bge-small-en-v1.5, ~130 MB); `with_images` the SigLIP2 image+text
/// towers (~700 MB) used by `LS_IMAGE_EMBED_PROVIDER=siglip`; `with_enrich`
/// the CodeFormulaV2 code/formula VLM (~0.8 GB); `with_eg2` the
/// EmbeddingGemma 2 graphs (~1.8 GB fp32) used by the `eg2` providers — off by
/// default for callers that know EG2 is not configured, since it is the
/// largest single download.
pub async fn download_models(
    cfg: &Config,
    with_embed: bool,
    with_images: bool,
    with_enrich: bool,
    with_eg2: bool,
) -> Result<()> {
    let base = env_or("DOCLING_RS_MODELS_URL", DEFAULT_MODELS_BASE);
    let embed_base = env_or("LS_ONNX_MODELS_URL", DEFAULT_EMBED_BASE);
    let image_base = env_or("LS_IMAGE_MODELS_URL", DEFAULT_IMAGE_BASE);
    let eg2_base = env_or("LS_EG2_MODELS_URL", DEFAULT_EG2_BASE);
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
        // EmbeddingGemma 2 text graph (`LS_EMBED_PROVIDER=eg2`). The external
        // weights file must keep its upstream name: the graph references
        // `model.onnx_data` next to it.
        if with_eg2 {
            assets.push(Asset {
                dest: docling("eg2/text/model.onnx"),
                urls: vec![format!("{eg2_base}/onnx/model.onnx")],
                optional: false,
            });
            assets.push(Asset {
                dest: docling("eg2/text/model.onnx_data"),
                urls: vec![format!("{eg2_base}/onnx/model.onnx_data")],
                optional: false,
            });
            assets.push(Asset {
                dest: docling("eg2/tokenizer.json"),
                urls: vec![format!("{eg2_base}/tokenizer.json")],
                optional: false,
            });
        }
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
        // EmbeddingGemma 2 vision encoder (`LS_IMAGE_EMBED_PROVIDER=eg2`),
        // plus the processor config that defines the patch grid / budget.
        if with_eg2 {
            assets.push(Asset {
                dest: docling("eg2/vision/vision_encoder.onnx"),
                urls: vec![format!("{eg2_base}/onnx/vision_encoder.onnx")],
                optional: false,
            });
            assets.push(Asset {
                dest: docling("eg2/vision/vision_encoder.onnx_data"),
                urls: vec![format!("{eg2_base}/onnx/vision_encoder.onnx_data")],
                optional: false,
            });
            assets.push(Asset {
                dest: docling("eg2/processor_config.json"),
                urls: vec![format!("{eg2_base}/processor_config.json")],
                optional: false,
            });
        }
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

    if with_enrich {
        // CodeFormulaV2 (code/formula enrichment, ~0.8 GB with the int8 decoder).
        let cf = |rel: &str| models.join("code_formula").join(rel);
        assets.push(Asset {
            dest: cf("vision.onnx"),
            urls: vec![release("cf_vision.onnx")],
            optional: false,
        });
        assets.push(Asset {
            dest: cf("embed.onnx"),
            urls: vec![release("cf_embed.onnx")],
            optional: false,
        });
        assets.push(Asset {
            dest: cf("tokenizer.json"),
            urls: vec![release("cf_tokenizer.json")],
            optional: false,
        });
        // The int8 decoder (~165 MB) is preferred; the fp32 one (~655 MB) is
        // fetched after the loop only when the int8 graph is unavailable.
        assets.push(Asset {
            dest: cf("decoder_kv_int8.onnx"),
            urls: vec![release("cf_decoder_kv_int8.onnx")],
            optional: true,
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

    // CodeFormulaV2 needs a decoder; prefer the int8 graph, else the fp32 one.
    if with_enrich && !models.join("code_formula/decoder_kv_int8.onnx").exists() {
        let fp32 = Asset {
            dest: models.join("code_formula/decoder_kv.onnx"),
            urls: vec![release("cf_decoder_kv.onnx")],
            optional: false,
        };
        if fetch_asset(&client, &fp32).await? {
            fetched += 1;
        }
    }

    // Record sha256 digests for any model files not yet in the lock, so stage
    // fingerprints can detect a model swap without hashing gigabytes each run.
    update_models_lock(&models)?;

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

/// Ensure every file under `models` has a sha256 entry in `models.lock`.
/// Incremental: existing entries are kept, so only new files are hashed.
fn update_models_lock(models: &Path) -> Result<()> {
    if !models.exists() {
        return Ok(());
    }
    let lock_path = models.join("models.lock");
    let mut lock: BTreeMap<String, String> = std::fs::read_to_string(&lock_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let mut changed = false;
    for entry in walkdir::WalkDir::new(models).follow_links(false) {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) == Some("models.lock") {
            continue;
        }
        let key = path.to_string_lossy().to_string();
        if lock.contains_key(&key) {
            continue;
        }
        match sha256_file(path) {
            Ok(hash) => {
                lock.insert(key, hash);
                changed = true;
            }
            Err(e) => tracing::warn!(path = %key, error = %e, "models.lock: cannot hash"),
        }
    }
    if changed {
        let json = serde_json::to_string_pretty(&lock)?;
        std::fs::write(&lock_path, json).map_err(|e| Error::io(&lock_path, e))?;
        println!("  + models.lock ({} entries)", lock.len());
    }
    Ok(())
}

/// Streaming sha256 of a file.
fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| Error::io(path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|e| Error::io(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
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
    let mirrors = asset.urls.len();
    for (i, url) in asset.urls.iter().enumerate() {
        match download(client, url, &asset.dest).await {
            Ok(()) => {
                println!("  > {}", asset.dest.display());
                return Ok(true);
            }
            Err(e) => {
                last = e.to_string();
                // Only announce a retry when there is another mirror to try;
                // otherwise the outcome line below says what happened.
                if i + 1 < mirrors {
                    tracing::warn!(url = %url, error = %e, "download failed; trying next mirror");
                } else {
                    tracing::debug!(url = %url, error = %e, "download failed");
                }
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
