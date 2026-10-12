//! Unified space (EG2 on both sides): text and image chunks live in one
//! vector table, so a text query matches both and `mode=image` is a kind
//! filter rather than a second index.

use local_search::RetrievalMode;
use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::model::ChunkKind;
use local_search::pipeline::{Pipeline, StageSet};
use local_search::store::ImageSpace;
use std::path::PathBuf;

fn temp_base(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ls-unified-{tag}-{}-{nanos}", std::process::id()))
}

/// Both providers hash at the *same* dimension, which is what makes the store
/// choose the unified layout (the `eg2` providers do this in production).
fn config(db: &str, root: PathBuf, image_only: Vec<PathBuf>) -> Config {
    Config {
        db_path: db.to_string(),
        source_paths: vec![root.to_string_lossy().to_string()],
        image_only_paths: image_only
            .into_iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
        models_dir: ".models".to_string(),
        embed: EmbedConfig {
            provider: EmbedProvider::Hash,
            model: None,
            dim: 64,
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
            provider: ImageEmbedProvider::Hash,
            dim: 64,
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
    }
}

fn png_bytes() -> Vec<u8> {
    vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ]
}

#[tokio::test]
async fn unified_space_indexes_text_and_images_together() {
    let base = temp_base("space");
    let docs = base.join("docs");
    let photos = base.join("photos");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::create_dir_all(&photos).unwrap();
    std::fs::write(
        docs.join("notes.md"),
        "# Notes\nprocess control loops and drum level measurements",
    )
    .unwrap();
    std::fs::write(photos.join("shot.png"), png_bytes()).unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, base.clone(), vec![photos.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    assert_eq!(
        pipeline.store().image_space(),
        ImageSpace::Unified,
        "equal dims with hash providers must select the unified layout"
    );
    pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();

    // Dense search over the one table returns both kinds.
    let dense = pipeline
        .query(RetrievalMode::Vector, "process control", 10)
        .await
        .unwrap();
    assert!(
        dense.iter().any(|h| h.chunk.kind == ChunkKind::Text),
        "text chunks must be in the unified index"
    );
    assert!(
        dense.iter().any(|h| h.chunk.kind == ChunkKind::Image),
        "image chunks must be in the unified index"
    );

    // `mode=image` filters the same table to image chunks only.
    let images = pipeline
        .query(RetrievalMode::Image, "process control", 10)
        .await
        .unwrap();
    assert!(!images.is_empty(), "image mode returned nothing");
    assert!(
        images.iter().all(|h| h.chunk.kind == ChunkKind::Image),
        "image mode must not return text chunks"
    );

    // Hybrid fuses dense (both kinds) with bm25.
    let hybrid = pipeline
        .query(RetrievalMode::Hybrid, "process control", 10)
        .await
        .unwrap();
    assert!(!hybrid.is_empty());

    std::fs::remove_dir_all(&base).ok();
}

#[tokio::test]
async fn switching_layout_rebuilds_the_vector_tables() {
    let base = temp_base("switch");
    let docs = base.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("a.md"), "# A\nsome words about pumps").unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    // 1. split layout (different dims).
    let mut cfg = config(&db, base.clone(), Vec::new());
    cfg.image.dim = 32;
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    assert_eq!(pipeline.store().image_space(), ImageSpace::Split(32));
    pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    let before = pipeline.store().count_chunks().await.unwrap();

    // 2. unified layout (same dims): the vector tables are rebuilt and the
    //    embed stages forgotten, so a plain re-ingest re-embeds.
    let cfg = config(&db, base.clone(), Vec::new());
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    assert_eq!(pipeline.store().image_space(), ImageSpace::Unified);
    let report = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_ingested, 1, "layout change re-embeds");
    assert_eq!(pipeline.store().count_chunks().await.unwrap(), before);

    std::fs::remove_dir_all(&base).ok();
}
