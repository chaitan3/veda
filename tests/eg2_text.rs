//! EG2 text provider: parity against the reference implementation.
//!
//! The expected similarities come from the official `google/embeddinggemma-2`
//! (sentence-transformers, fp32) on the model card's planet example with the
//! documented task prefixes — see `examples/eg2_parity.py`. Matching them here
//! proves the whole local path (tokenizer, prefixes, graph, pooled output,
//! normalisation) is wired correctly, not just that the model loads.
//!
//! Skipped when the EG2 weights are not downloaded.

use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::embed;

fn eg2_paths() -> Option<(String, String)> {
    let model = std::env::var("LS_EG2_TEXT_MODEL")
        .unwrap_or_else(|_| ".models/eg2/text/model.onnx".to_string());
    let tokenizer = std::env::var("LS_EG2_TOKENIZER")
        .unwrap_or_else(|_| ".models/eg2/tokenizer.json".to_string());
    (std::path::Path::new(&model).exists() && std::path::Path::new(&tokenizer).exists())
        .then_some((model, tokenizer))
}

fn config(model: String, tokenizer: String) -> Config {
    Config {
        db_path: ":memory:".to_string(),
        source_paths: Vec::new(),
        image_only_paths: Vec::new(),
        models_dir: ".models".to_string(),
        embed: EmbedConfig {
            provider: EmbedProvider::Eg2,
            model: None,
            dim: EmbedProvider::Eg2.natural_dim(),
            base_url: String::new(),
            api_key: None,
            // Defaults, as `Config::from_env` would set for provider=eg2.
            query_prefix: Some(embed::eg2::DEFAULT_QUERY_PREFIX.to_string()),
            doc_prefix: Some(embed::eg2::DEFAULT_DOC_PREFIX.to_string()),
            onnx_model: None,
            onnx_tokenizer: None,
            eg2_model: Some(model),
            eg2_tokenizer: Some(tokenizer),
            max_tokens: 512,
        },
        image: ImageEmbedConfig {
            provider: ImageEmbedProvider::Off,
            dim: 768,
            vision_model: None,
            text_model: None,
            tokenizer: None,
            preprocessor: None,
            eg2_vision_model: None,
            eg2_processor: None,
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
    }
}

#[tokio::test]
async fn eg2_text_matches_the_reference_similarities() {
    let Some((model, tokenizer)) = eg2_paths() else {
        eprintln!("skipping: EG2 text model not present");
        return;
    };
    let cfg = config(model, tokenizer);
    let embedder = embed::from_config(&cfg).unwrap();
    assert_eq!(embedder.dim(), 768);

    let query = embedder
        .embed_query("Which planet is known as the Red Planet?")
        .await
        .unwrap();
    let docs = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]
    .map(String::from);
    let vectors = embedder.embed(&docs).await.unwrap();

    // Reference (official sentence-transformers, fp32).
    let expected = [0.6755f32, 0.8532, 0.7468, 0.7783];
    let sims: Vec<f32> = vectors
        .iter()
        .map(|v| v.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>())
        .collect();
    for (i, (got, want)) in sims.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "doc {i}: similarity {got:.4} != reference {want:.4} (all: {sims:?})"
        );
    }
    // Embeddings are unit length.
    for v in &vectors {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "not normalised: {norm}");
    }
}

#[test]
fn provider_dimension_is_auto_derived() {
    assert_eq!(EmbedProvider::Eg2.natural_dim(), 768);
    assert_eq!(EmbedProvider::Onnx.natural_dim(), 384);
}
