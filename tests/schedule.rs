//! Scheduled ingest while serving (`LS_INGEST_INTERVAL`).
//!
//! The supervisor runs each ingest in a child process (the same binary). This
//! test drives that end to end: it points `LS_INGEST_BIN` at the built binary,
//! passes the test's config through the environment, and checks the child's
//! commits land in the shared database.

use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::pipeline::Pipeline;
use local_search::schedule;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

fn temp_base(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ls-sched-{tag}-{}-{nanos}", std::process::id()))
}

fn config(db: &str, root: PathBuf) -> Config {
    Config {
        db_path: db.to_string(),
        source_paths: vec![root.to_string_lossy().to_string()],
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
        ingest_interval: Some(Duration::from_millis(50)),
        ingest_sync: false,
        embed_idle: None,
        chunk_strict: false,
    }
}

#[tokio::test]
async fn scheduled_ingest_child_picks_up_new_files() {
    let base = temp_base("run");
    let docs = base.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("one.md"), "# One\nfirst document body").unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    // The child re-derives its config from the environment; pass this test's
    // settings so it writes the same database with the offline hash embedder.
    // Safe to set process-wide: this test binary holds a single test.
    unsafe {
        std::env::set_var("LS_INGEST_BIN", env!("CARGO_BIN_EXE_local-search"));
        std::env::set_var("LS_DB_PATH", &db);
        std::env::set_var("LS_MODELS_DIR", ".models");
        std::env::set_var("LS_EMBED_PROVIDER", "hash");
        std::env::set_var("LS_EMBED_DIM", "32");
        std::env::set_var("LS_IMAGE_EMBED_PROVIDER", "off");
        std::env::set_var("LS_CHUNKER", "window");
        std::env::set_var("LS_CHUNK_SIZE", "100");
    }

    let cfg = config(&db, docs.clone());
    let pipeline = Arc::new(Pipeline::from_config(&cfg).await.unwrap());

    let (tx, rx) = tokio::sync::watch::channel(false);
    // First run happens immediately (catch-up), then every 50 ms.
    let handle = schedule::spawn(cfg.clone(), Duration::from_millis(50), false, rx);

    // Wait for the catch-up run to land the first document.
    let mut waited = 0u64;
    while pipeline.store().count_documents().await.unwrap() < 1 && waited < 200 {
        tokio::time::sleep(Duration::from_millis(20)).await;
        waited += 1;
    }
    assert_eq!(pipeline.store().count_documents().await.unwrap(), 1);

    // A new file is picked up by a later tick.
    std::fs::write(docs.join("two.md"), "# Two\nsecond document body").unwrap();
    waited = 0;
    while pipeline.store().count_documents().await.unwrap() < 2 && waited < 200 {
        tokio::time::sleep(Duration::from_millis(20)).await;
        waited += 1;
    }
    assert_eq!(
        pipeline.store().count_documents().await.unwrap(),
        2,
        "scheduled ingest did not pick up the new file"
    );

    // Shutdown stops the supervisor.
    let _ = tx.send(true);
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;

    std::fs::remove_dir_all(&base).ok();
}
