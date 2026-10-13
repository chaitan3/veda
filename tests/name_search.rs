//! Name matching inside `/api/search` (`hybrid`): a query that matches a
//! document's title or source path contributes that document's chunks, so
//! "find the file called X" works through search — including for image-only
//! documents, whose only chunk is the image.

use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::model::ChunkKind;
use local_search::pipeline::{Pipeline, StageSet};
use local_search::RetrievalMode;
use std::path::PathBuf;

fn temp_base(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ls-name-{tag}-{}-{nanos}", std::process::id()))
}

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
            // Same dim as the text embedder -> one unified space, so the
            // image-only document is indexed and can be found by name.
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
        chunk_strict: false,
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
async fn hybrid_matches_document_names() {
    let base = temp_base("docs");
    let docs = base.join("docs");
    let photos = base.join("photos");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::create_dir_all(&photos).unwrap();
    // The file name shares no words with the content, so only the name
    // channel can find it.
    std::fs::write(
        docs.join("quarterly_summary.md"),
        "# Untitled\nsome unrelated body text about pumps and valves",
    )
    .unwrap();
    std::fs::write(photos.join("holiday_snapshot.png"), png_bytes()).unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, base.clone(), vec![photos.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();

    // Document ids are per-ingest; map them to their source paths.
    let docs = pipeline.store().list_documents().await.unwrap();
    let id_for = |needle: &str| {
        docs.iter()
            .find(|d| d.source_uri.contains(needle))
            .map(|d| d.id.clone())
            .expect("document present")
    };
    let text_doc = id_for("quarterly_summary.md");
    let image_doc = id_for("holiday_snapshot.png");

    // Text document found by file-name stem. The name channel puts its chunks
    // at rank 1 of its own list, so with hash embeddings (where the dense list
    // is arbitrary) the target must still come out on top.
    let hits = pipeline
        .query(RetrievalMode::Hybrid, "quarterly_summary", 5)
        .await
        .unwrap();
    assert!(
        hits.first().map(|h| h.chunk.doc_id.as_str()) == Some(text_doc.as_str()),
        "name channel did not surface the document at rank 1"
    );

    // Image-only document found by file-name stem (the original failure).
    let hits = pipeline
        .query(RetrievalMode::Hybrid, "holiday_snapshot", 5)
        .await
        .unwrap();
    assert!(
        hits.first()
            .map(|h| (h.chunk.doc_id.as_str(), h.chunk.kind))
            == Some((image_doc.as_str(), ChunkKind::Image)),
        "image-only document not found by name"
    );

    std::fs::remove_dir_all(&base).ok();
}
