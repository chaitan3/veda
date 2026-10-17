//! EG2 vision provider: preprocessing and embedding parity.
//!
//! `tests/fixtures/eg2/reference.json` was produced by the reference stack
//! (`Gemma4ImageProcessor` + `vision_encoder.onnx` + `model.onnx`, fp32,
//! 280 soft tokens) for `tests/fixtures/eg2/test_image.png`. This test runs the
//! local Rust path on the same image and requires:
//!
//! * identical position ids (the patch grid and `-1` padding),
//! * the same soft-token count,
//! * a high cosine similarity to the reference embedding (pixel-level resize
//!   filters differ slightly between torchvision and the `image` crate, so the
//!   embedding is compared by direction, not bit-for-bit).
//!
//! Skipped when the EG2 weights are not downloaded.

use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::embed;
use serde_json::Value;

fn eg2_paths() -> Option<(String, String, String, String)> {
    let text = std::env::var("LS_EG2_TEXT_MODEL")
        .unwrap_or_else(|_| ".models/eg2/text/model.onnx".to_string());
    let tokenizer = std::env::var("LS_EG2_TOKENIZER")
        .unwrap_or_else(|_| ".models/eg2/tokenizer.json".to_string());
    let vision = std::env::var("LS_EG2_VISION_MODEL")
        .unwrap_or_else(|_| ".models/eg2/vision/vision_encoder.onnx".to_string());
    let processor = std::env::var("LS_EG2_PROCESSOR")
        .unwrap_or_else(|_| ".models/eg2/processor_config.json".to_string());
    [&text, &tokenizer, &vision]
        .iter()
        .all(|p| std::path::Path::new(p).exists())
        .then_some((text, tokenizer, vision, processor))
}

fn config(text: String, tokenizer: String, vision: String, processor: String) -> Config {
    Config {
        db_path: ":memory:".to_string(),
        source_paths: Vec::new(),
        image_only_paths: Vec::new(),
        models_dir: ".models".to_string(),
        embed: EmbedConfig {
            provider: EmbedProvider::Eg2,
            model: None,
            dim: 768,
            base_url: String::new(),
            api_key: None,
            query_prefix: Some(embed::eg2::DEFAULT_QUERY_PREFIX.to_string()),
            doc_prefix: Some(embed::eg2::DEFAULT_DOC_PREFIX.to_string()),
            onnx_model: None,
            onnx_tokenizer: None,
            eg2_model: Some(text),
            eg2_tokenizer: Some(tokenizer),
            max_tokens: 512,
        },
        image: ImageEmbedConfig {
            provider: ImageEmbedProvider::Eg2,
            dim: 768,
            vision_model: None,
            text_model: None,
            tokenizer: None,
            preprocessor: None,
            eg2_vision_model: Some(vision),
            eg2_processor: Some(processor),
            soft_tokens: None,
        },
        extract: ExtractPipeline::Onnx,
        code_formula: false,
        chunker: ChunkerKind::Window,
        chunk_size: 100,
        chunk_overlap: 0.0,
        rrf_k: 60.0,
        api_keys: Vec::new(),
        addr: "127.0.0.1:0".to_string(),
        ingest_interval: None,
        ingest_sync: false,
        chunk_strict: false,
        embed_idle: None,
    }
}

#[tokio::test]
async fn eg2_vision_matches_the_reference_embedding() {
    let Some((text, tokenizer, vision, processor)) = eg2_paths() else {
        eprintln!("skipping: EG2 vision model not present");
        return;
    };
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string("tests/fixtures/eg2/reference.json").unwrap(),
    )
    .unwrap();
    let bytes = std::fs::read("tests/fixtures/eg2/test_image.png").unwrap();

    // 1. preprocessing parity (positions are exact; patches may differ by the
    //    resize filter, so they are only compared via the embedding below).
    let pre = embed::image::eg2::Preprocess::default();
    let preprocessed = embed::image::eg2::preprocess_image(&bytes, &pre).unwrap();
    assert_eq!(
        preprocessed.soft_tokens,
        reference["soft_tokens"].as_u64().unwrap() as usize,
        "soft-token count differs from the reference"
    );
    assert_eq!(
        preprocessed.padded_patches,
        reference["padded_patches"].as_u64().unwrap() as usize
    );
    let positions_sha = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for v in &preprocessed.positions {
            h.update(v.to_le_bytes());
        }
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    assert_eq!(
        positions_sha,
        reference["positions_sha256"].as_str().unwrap(),
        "patch grid / padding positions differ from the reference"
    );

    // 2. embedding parity (cosine, since resize filters are not bit-identical).
    let cfg = config(text, tokenizer, vision, processor);
    let image = embed::image::from_config(&cfg).unwrap().unwrap();
    let vectors = image.embed_images(&[bytes]).await.unwrap();
    let got = &vectors[0];
    let want: Vec<f32> = reference["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let cos: f32 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
    assert!(
        cos > 0.99,
        "EG2 vision embedding diverges from the reference (cosine {cos:.4})"
    );
    let norm: f32 = got.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "not normalised: {norm}");
}
