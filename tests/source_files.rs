//! Cheap change detection: the `source_files` cache (size + mtime gate) and
//! its `--sync` cleanup.

use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::pipeline::{Pipeline, StageSet};
use std::path::PathBuf;

fn temp_base(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ls-src-{tag}-{}-{nanos}", std::process::id()))
}

fn config(db: &str, roots: Vec<PathBuf>) -> Config {
    Config {
        db_path: db.to_string(),
        source_paths: roots
            .into_iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
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
        extract: ExtractPipeline::Onnx,
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
    }
}

#[tokio::test]
async fn source_files_cache_tracks_content() {
    let base = temp_base("cache");
    let a = base.join("a");
    std::fs::create_dir_all(&a).unwrap();
    let file = a.join("one.md");
    std::fs::write(&file, "# One\nunique wombat content alpha.").unwrap();
    let uri = file.to_string_lossy().to_string();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, vec![a.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let first = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(first.documents_ingested, 1);

    // The cache records the on-disk size and a hash.
    let meta = std::fs::metadata(&file).unwrap();
    let cached = pipeline
        .store()
        .source_file(&uri)
        .await
        .unwrap()
        .expect("source_files row written");
    assert_eq!(cached.size, meta.len());
    assert!(!cached.hash.is_empty());
    assert!(cached.doc_id.is_some());

    // Unchanged: skipped, and the cache is stable.
    let second = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(second.documents_skipped, 1);
    assert_eq!(second.documents_ingested, 0);
    let cached_again = pipeline.store().source_file(&uri).await.unwrap().unwrap();
    assert_eq!(cached.hash, cached_again.hash);

    // Content change: re-ingested, the cache hash moves, one document remains.
    std::fs::write(&file, "# One\nentirely different pangolin content beta.").unwrap();
    let third = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(third.documents_ingested, 1);
    let changed = pipeline.store().source_file(&uri).await.unwrap().unwrap();
    assert_ne!(cached.hash, changed.hash, "cache tracks the new content");
    assert_eq!(pipeline.store().count_documents().await.unwrap(), 1);

    std::fs::remove_dir_all(&base).ok();
}

#[tokio::test]
async fn sync_drops_source_files_for_vanished_paths() {
    let base = temp_base("sync");
    let a = base.join("a");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::write(a.join("one.md"), "# One\nstable content.").unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, vec![a.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(pipeline.store().all_source_files().await.unwrap().len(), 1);

    std::fs::remove_file(a.join("one.md")).unwrap();
    let report = pipeline
        .ingest_all(true, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_deleted, 1);
    assert!(
        pipeline
            .store()
            .all_source_files()
            .await
            .unwrap()
            .is_empty(),
        "the vanished path's cache row is swept"
    );

    std::fs::remove_dir_all(&base).ok();
}
