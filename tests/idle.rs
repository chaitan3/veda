//! The idle window releases the embedding models and reloads them on demand.

use local_search::RetrievalMode;
use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::pipeline::Pipeline;
use std::path::PathBuf;
use std::time::Duration;

fn temp_base(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ls-idle-{tag}-{}-{nanos}", std::process::id()))
}

fn config(db: &str, idle: Option<Duration>) -> Config {
    Config {
        db_path: db.to_string(),
        source_paths: Vec::new(),
        image_only_paths: Vec::new(),
        models_dir: ".models".to_string(),
        embed: EmbedConfig {
            provider: EmbedProvider::Hash,
            model: None,
            dim: 32,
            base_url: String::new(),
            api_key: None,
            query_prefix: None,
            doc_prefix: None,
            onnx_model: None,
            onnx_tokenizer: None,
            eg2_model: None,
            eg2_tokenizer: None,
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
        ingest_interval: None,
        ingest_sync: false,
        embed_idle: idle,
        chunk_strict: false,
    }
}

#[tokio::test]
async fn idle_window_releases_and_reloads_models() {
    let base = temp_base("reap");
    let db = base.join("app.db").to_string_lossy().to_string();
    let cfg = config(&db, Some(Duration::from_millis(60)));
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();

    assert!(pipeline.embedder().is_loaded().await, "loaded at startup");

    // A query touches the shared window.
    pipeline
        .query(RetrievalMode::Vector, "hello", 1)
        .await
        .unwrap();
    pipeline.reap_idle().await;
    assert!(
        pipeline.embedder().is_loaded().await,
        "kept while the window is fresh"
    );

    // After the window it is released...
    tokio::time::sleep(Duration::from_millis(120)).await;
    pipeline.reap_idle().await;
    assert!(
        !pipeline.embedder().is_loaded().await,
        "released after the idle window"
    );

    // ...and the next query reloads it.
    pipeline
        .query(RetrievalMode::Vector, "hello", 1)
        .await
        .unwrap();
    assert!(
        pipeline.embedder().is_loaded().await,
        "reloaded on the next query"
    );

    // `off` (None) never releases.
    let cfg = config(&db, None);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    tokio::time::sleep(Duration::from_millis(80)).await;
    pipeline.reap_idle().await;
    assert!(
        pipeline.embedder().is_loaded().await,
        "an unset idle window keeps the model"
    );

    std::fs::remove_dir_all(&base).ok();
}
