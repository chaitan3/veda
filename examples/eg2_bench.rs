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
//! ONNX Runtime CUDA EP, no arena shrinkage, fp32 unless stated:
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
//! Takeaways: q8 is not faster on this GPU (slightly slower) but saves 20-30%
//! VRAM, with parity-level quality (see examples/eg2_parity.py); EG2 text is
//! ~4.4x slower than bge at equal (fp32) precision; EG2 vision is ~4.5x
//! (budget 70) to ~27x (budget 280) slower than SigLIP2.

use ort::session::Session;
use ort::value::Tensor;
use std::time::Instant;

const EG2_DIR: &str = "/tmp/opencode/eg2";

fn session(path: &str) -> anyhow::Result<Session> {
    let builder = Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut builder = docling_onnx::apply(builder).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(builder
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("{e}"))?)
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
                let _ = s.run(ort::inputs![
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t,
                    "image_features" => Tensor::from_array(([0usize, 512], Vec::<f32>::new()))?,
                    "video_features" => Tensor::from_array(([0usize, 512], Vec::<f32>::new()))?,
                    "audio_features" => Tensor::from_array(([0usize, 512], Vec::<f32>::new()))?,
                ])?;
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
                    let _ = s.run(ort::inputs![
                        "input_ids" => ids_t,
                        "attention_mask" => mask_t,
                        "token_type_ids" => types,
                    ])?;
                } else {
                    let _ = s.run(ort::inputs![
                        "input_ids" => ids_t,
                        "attention_mask" => mask_t,
                    ])?;
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
                    let _ = s.run(ort::inputs![
                        "input_ids" => ids_t,
                        "attention_mask" => mask_t,
                    ])?;
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
