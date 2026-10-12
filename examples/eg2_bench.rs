//! Microbenchmark for the EmbeddingGemma 2 parity work: ONNX throughput on the
//! GPU for EG2 (fp32 and q8, text and vision) against the models we run today
//! (bge-small text, SigLIP2 vision/text). Inputs are realistic shapes; values
//! are synthetic for the vision graphs (throughput does not depend on pixels).
//!
//! Usage:
//!   cargo run --release --example eg2_bench -- <kind> [iters] [batch]
//! kinds: eg2-text-fp32 | eg2-text-q8 | eg2-vision-fp32 | eg2-vision-q8
//!        bge-text | siglip-text | siglip-vision
//!
//! VRAM is sampled externally (nvidia-smi) by the caller.
//!
//! Measured on the RTX 2080 Ti (11 GB, Turing; desktop baseline ~425 MiB),
//! ONNX Runtime CUDA EP. First block without run options, second block with
//! production's `docling_onnx::shrink_run_options()` (`--shrink`):
//!
//!   kind                     ms/iter   items/s   peak VRAM (delta)
//!   bge-text (32 x 244 tok)     56.2     569.6    788 MiB     <- current
//!   eg2-text-fp32 (32 x 217)   245.5     130.3   3776 MiB
//!   eg2-text-q8                257.6     124.2   3076 MiB
//!   siglip-text (32 x 64 tok)   45.8     699.2   1632 MiB     <- current
//!   siglip-vision (224px)        6.4     157.1    688 MiB     <- current
//!   eg2-vision-fp32 @280        176.2       5.7   2284 MiB
//!   eg2-vision-q8   @280        183.8       5.4   1760 MiB
//!   eg2-vision-fp32 @70          28.5      35.1   1090 MiB
//!   eg2-vision-q8   @70          31.3      31.9    566 MiB
//!
//!   with --shrink (production):
//!   bge-text (32)               59.6     536.5    580 MiB
//!   eg2-text-fp32 (32)         272.2     117.6   2532 MiB
//!   eg2-text-q8 (32)           286.1     111.9   1810 MiB
//!   siglip-vision (1)            6.5     154.2    688 MiB
//!   eg2-vision-fp32 @280 (1)   183.7       5.4   2286 MiB
//!   eg2-vision-fp32 @70 (1)     33.0      30.3   1070 MiB
//!   eg2-vision-q8   @70 (1)     33.8      29.6    544 MiB
//!
//! Batching (this is why the ingest pipeline's shape is already right):
//!   - text: EG2 at batch 64 = 117.8 items/s vs 117.6 at batch 32 -> no gain,
//!     more VRAM. bge likewise (551 vs 536). EMBED_BATCH=32 is the sweet spot.
//!   - vision: SigLIP2 batch 8 = 157 img/s (same as 1); EG2 @70 batch 8 =
//!     35.5 img/s (same as 30.3 single, +1844 MiB); EG2 @280 batch 2/4 = 5.3
//!     img/s (same as 1, VRAM +3.6/+6.2 GB) and batch 8 OOMs asking for a
//!     5.07 GB attention buffer. Image batching buys nothing for either
//!     model, so embedding one image per call (as the pipeline does) is fine.
//!
//! CPU (Ryzen 9 5950X, 16c/32t, DOCLING_RS_EP=cpu; idle-desktop frequency
//! scaling makes these vary ~±30%):
//!
//!   kind                     ms/iter   items/s   GPU items/s (shrink)
//!   bge-text (32 x 244)        962        33.3      536.5
//!   siglip-text (32 x 64)      551        58.1      -
//!   siglip-vision (224px)       60        16.7      154.2
//!   eg2-text-fp32 (32 x 217)  4445         7.2      117.6
//!   eg2-text-q8 (32 x 217)    5472         5.8      111.9
//!   eg2-vision-fp32 @70        278-516    1.9-3.6    30.3
//!   eg2-vision-fp32 @280      1596         0.6        5.4
//!
//! CPU is ~16x slower than the GPU for EG2 text, ~9-16x for vision. Thread
//! scaling is flat (8 threads = 32 threads = same throughput), so threads can
//! be capped without loss. q8 is slower than fp32 on CPU too (Zen 3 has no
//! VNNI). Bulk CPU ingest is impractical (100k chunks ~ 4 h); incremental is
//! fine (1k chunks ~ 2.5 min, 100 images @70 ~ 30-50 s).
//!
//! Takeaways: q8 is not faster on this GPU (slightly slower) but saves 20-40%
//! VRAM, with parity-level quality (see examples/eg2_parity.py); EG2 text is
//! ~4.6x slower than bge at equal (fp32) precision; EG2 vision is ~5x (budget
//! 70) to ~28x (budget 280) slower than SigLIP2.

use ort::session::Session;
use ort::value::Tensor;
use std::time::Instant;

const EG2_DIR: &str = "/tmp/opencode/eg2";

fn session(path: &str) -> anyhow::Result<Session> {
    let builder = Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut builder = docling_onnx::apply(builder).map_err(|e| anyhow::anyhow!("{e}"))?;
    builder
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Read a raw little-endian tensor dump produced by the parity setup script.
fn read_f32(path: &str) -> anyhow::Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn read_i64(path: &str) -> anyhow::Result<Vec<i64>> {
    let b = std::fs::read(path)?;
    Ok(b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect())
}

/// Default run options (production passes `docling_onnx::shrink_run_options`).
fn plain_run_options() -> &'static ort::session::RunOptions {
    static OPTS: std::sync::OnceLock<ort::session::RunOptions> = std::sync::OnceLock::new();
    OPTS.get_or_init(|| ort::session::RunOptions::new().expect("run options"))
}

fn feed_names(s: &Session) -> Vec<String> {
    s.inputs().iter().map(|i| i.name().to_string()).collect()
}

/// Tokenize `text` with the given tokenizer, right-padded, to `batch` rows.
fn batch_ids(tok_path: &str, text: &str, batch: usize, cap: usize) -> anyhow::Result<Vec<i64>> {
    let tok = tokenizers::Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids: Vec<u32> = enc.get_ids().iter().copied().take(cap).collect();
    let seq = ids.len();
    let mut out = vec![0i64; batch * seq];
    for b in 0..batch {
        for (t, &id) in ids.iter().enumerate() {
            out[b * seq + t] = id as i64;
        }
    }
    Ok(out)
}

fn bench<F>(label: &str, iters: usize, items: usize, mut f: F) -> anyhow::Result<()>
where
    F: FnMut() -> anyhow::Result<()>,
{
    // Warm up (session/graph init, cuDNN kernels).
    for _ in 0..2 {
        f()?;
    }
    let t = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    let secs = t.elapsed().as_secs_f64();
    println!(
        "{label:<18} {iters:>4} iters  {:.1} ms/iter  {:.1} items/s",
        secs * 1000.0 / iters as f64,
        (iters * items) as f64 / secs
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).cloned().unwrap_or_default();
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let batch: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let shrink = args.iter().any(|a| a == "--shrink");
    let opts: fn() -> &'static ort::session::RunOptions = if shrink {
        docling_onnx::shrink_run_options
    } else {
        plain_run_options
    };

    // A token-dense paragraph, similar in length to our 300-word chunks.
    let long_text = "The distillation column is used to separate a binary mixture. \
        Symbols x, y and z denote mole fractions of the more volatile component while \
        B, D, R and F represent molar flow rates. It is desired to control distillate \
        composition y despite disturbances in feed flow rate F. "
        .repeat(4);

    match kind.as_str() {
        "eg2-text-fp32" | "eg2-text-q8" => {
            let model = if kind.ends_with("q8") {
                format!("{EG2_DIR}/onnx/model_quantized.onnx")
            } else {
                format!("{EG2_DIR}/onnx/model.onnx")
            };
            let tok_path = format!("{EG2_DIR}/tokenizer.json");
            let ids = batch_ids(&tok_path, &long_text, batch, 2048)?;
            let seq = ids.len() / batch;
            let mut s = session(&model)?;
            let names = feed_names(&s);
            println!("[{kind}] inputs: {names:?}  batch={batch} seq={seq}");
            let mask = vec![1i64; ids.len()];
            bench(&kind, iters, batch, || {
                let ids_t = Tensor::from_array(([batch, seq], ids.clone()))?;
                let mask_t = Tensor::from_array(([batch, seq], mask.clone()))?;
                let _ = s.run_with_options(
                    ort::inputs![
                        "input_ids" => ids_t,
                        "attention_mask" => mask_t,
                        "image_features" => Tensor::from_array(([0usize, 512], Vec::<f32>::new()))?,
                        "video_features" => Tensor::from_array(([0usize, 512], Vec::<f32>::new()))?,
                        "audio_features" => Tensor::from_array(([0usize, 512], Vec::<f32>::new()))?,
                    ],
                    opts(),
                )?;
                Ok(())
            })?;
        }
        "eg2-vision-fp32" | "eg2-vision-q8" => {
            let model = if kind.ends_with("q8") {
                format!("{EG2_DIR}/onnx/vision_encoder_quantized.onnx")
            } else {
                format!("{EG2_DIR}/onnx/vision_encoder.onnx")
            };
            let mut s = session(&model)?;
            let names = feed_names(&s);
            // Soft-token budget (arg 4, default 280) × 3×3 pooling = patch count.
            let soft: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(280);
            let patches = soft * 9;
            println!("[{kind}] inputs: {names:?}  soft_tokens={soft}");
            let w = (patches as f64).sqrt().round() as i64;
            let pos_data: Vec<i64> = (0..patches)
                .flat_map(|i| [(i as i64 / w), (i as i64 % w)])
                .collect();
            bench(&kind, iters, 1, || {
                let px = Tensor::from_array(([1usize, patches, 768], vec![0.5f32; patches * 768]))?;
                let pos = Tensor::from_array(([1usize, patches, 2], pos_data.clone()))?;
                let _ = s.run(ort::inputs![
                    "pixel_values" => px,
                    "pixel_position_ids" => pos,
                ])?;
                Ok(())
            })?;
        }
        k if k.starts_with("eg2-vision280") || k.starts_with("eg2-vision70") => {
            let soft = if kind.contains("70") { 70 } else { 280 };
            let q8 = kind.contains("q8");
            let model = if q8 {
                format!("{EG2_DIR}/onnx/vision_encoder_quantized.onnx")
            } else {
                format!("{EG2_DIR}/onnx/vision_encoder.onnx")
            };
            let mut s = session(&model)?;
            let patches = soft * 9;
            let px_all = read_f32(&format!("{EG2_DIR}/vision_{soft}_px.bin"))?;
            let pos_all = read_i64(&format!("{EG2_DIR}/vision_{soft}_pos.bin"))?;
            let avail = px_all.len() / (patches * 768);
            let n = batch.min(avail);
            println!(
                "[{kind}] inputs: {:?}  batch={n} patches={patches} shrink={shrink}",
                feed_names(&s)
            );
            bench(&kind, iters, n, || {
                let px =
                    Tensor::from_array(([n, patches, 768], px_all[..n * patches * 768].to_vec()))?;
                let pos =
                    Tensor::from_array(([n, patches, 2], pos_all[..n * patches * 2].to_vec()))?;
                let _ = s.run_with_options(
                    ort::inputs!["pixel_values" => px, "pixel_position_ids" => pos],
                    opts(),
                )?;
                Ok(())
            })?;
        }
        "bge-text" => {
            let ids = batch_ids(".models/embed/tokenizer.json", &long_text, batch, 512)?;
            let seq = ids.len() / batch;
            let mut s = session(".models/embed/bge-small-en-v1.5.onnx")?;
            let names = feed_names(&s);
            println!("[bge-text] inputs: {names:?}  batch={batch} seq={seq}");
            let mask = vec![1i64; ids.len()];
            let needs_types = names.iter().any(|n| n == "token_type_ids");
            bench("bge-text", iters, batch, || {
                let ids_t = Tensor::from_array(([batch, seq], ids.clone()))?;
                let mask_t = Tensor::from_array(([batch, seq], mask.clone()))?;
                if needs_types {
                    let types = Tensor::from_array(([batch, seq], vec![0i64; ids.len()]))?;
                    let _ = s.run_with_options(
                        ort::inputs![
                            "input_ids" => ids_t,
                            "attention_mask" => mask_t,
                            "token_type_ids" => types,
                        ],
                        opts(),
                    )?;
                } else {
                    let _ = s.run_with_options(
                        ort::inputs![
                            "input_ids" => ids_t,
                            "attention_mask" => mask_t,
                        ],
                        opts(),
                    )?;
                }
                Ok(())
            })?;
        }
        "siglip-text" => {
            let ids = batch_ids(".models/image/tokenizer.json", &long_text, batch, 64)?;
            let seq = ids.len() / batch;
            let mut s = session(".models/image/text_model.onnx")?;
            let names = feed_names(&s);
            println!("[siglip-text] inputs: {names:?}  batch={batch} seq={seq}");
            let needs_mask = names.iter().any(|n| n == "attention_mask");
            bench("siglip-text", iters, batch, || {
                let ids_t = Tensor::from_array(([batch, seq], ids.clone()))?;
                if needs_mask {
                    let mask_t = Tensor::from_array(([batch, seq], vec![1i64; ids.len()]))?;
                    let _ = s.run_with_options(
                        ort::inputs![
                            "input_ids" => ids_t,
                            "attention_mask" => mask_t,
                        ],
                        opts(),
                    )?;
                } else {
                    let _ = s.run(ort::inputs!["input_ids" => ids_t])?;
                }
                Ok(())
            })?;
        }
        "siglip-vision" => {
            let mut s = session(".models/image/vision_model.onnx")?;
            let names = feed_names(&s);
            println!("[siglip-vision] inputs: {names:?}");
            bench("siglip-vision", iters, 1, || {
                let px = Tensor::from_array(([1usize, 3, 224, 224], vec![0.5f32; 3 * 224 * 224]))?;
                let _ = s.run(ort::inputs!["pixel_values" => px])?;
                Ok(())
            })?;
        }
        other => {
            eprintln!("unknown kind '{other}'");
            std::process::exit(2);
        }
    }
    Ok(())
}
