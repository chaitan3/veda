//! `LS_CHUNK_STRICT`: warn (default) vs refuse to start when the chunk budget
//! exceeds the embedder's sequence window.

use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::pipeline::Pipeline;
use std::path::PathBuf;

fn temp_base(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ls-strict-{tag}-{}-{nanos}", std::process::id()))
}

fn config(db: &str, chunk_size: usize, strict: bool) -> Config {
    Config {
        db_path: db.to_string(),
        source_paths: Vec::new(),
        image_only_paths: Vec::new(),
        models_dir: ".models".to_string(),
        embed: EmbedConfig {
            // The check only applies to providers that truncate (ONNX/EG2).
            provider: EmbedProvider::Onnx,
            model: None,
            dim: 32,
            base_url: String::new(),
            api_key: None,
            query_prefix: None,
            doc_prefix: None,
            onnx_model: Some(".models/embed/bge-small-en-v1.5.onnx".to_string()),
            onnx_tokenizer: Some(".models/embed/tokenizer.json".to_string()),
            eg2_model: None,
            eg2_tokenizer: None,
            max_tokens: 64,
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
        chunker: ChunkerKind::Hybrid,
        chunk_size,
        chunk_overlap: 0.0,
        rrf_k: 60.0,
        api_keys: Vec::new(),
        addr: "127.0.0.1:0".to_string(),
        ingest_interval: None,
        ingest_sync: false,
        chunk_strict: strict,
    }
}

#[tokio::test]
async fn strict_mode_refuses_an_oversized_chunk_budget() {
    let base = temp_base("strict");
    std::fs::create_dir_all(&base).unwrap();
    let db = base.join("a.db").to_string_lossy().to_string();
    // 64-token window, 200-token budget -> error in strict mode. The check
    // runs before any model is loaded, so no weights are needed here.
    let err = Pipeline::from_config(&config(&db, 200, true)).await;
    assert!(err.is_err(), "strict mode must refuse to start");

    // The same config is only a warning by default (needs the bge weights,
    // since the non-strict path continues into the embedder).
    if std::path::Path::new(".models/embed/bge-small-en-v1.5.onnx").exists() {
        let db = base.join("b.db").to_string_lossy().to_string();
        assert!(Pipeline::from_config(&config(&db, 200, false)).await.is_ok());
    } else {
        eprintln!("skipping the non-strict half: bge model not present");
    }

    std::fs::remove_dir_all(&base).ok();
}
