//! Image-only folder policy: `LS_IMAGE_ONLY_PATHS` folders index their image
//! files through the image channel only — no docling conversion, no OCR, no
//! text chunks — and a document follows the policy when the folder changes.

use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::model::{Chunk, ChunkKind, Document};
use local_search::pipeline::{Pipeline, StageSet};
use local_search::store::VectorStore;
use std::path::PathBuf;

fn temp_base(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ls-io-{tag}-{}-{nanos}", std::process::id()))
}

fn config(db: &str, roots: Vec<PathBuf>, image_only: Vec<PathBuf>) -> Config {
    Config {
        db_path: db.to_string(),
        source_paths: roots
            .into_iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
        image_only_paths: image_only
            .into_iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
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
            // The offline image embedder: exercise the policy without models.
            provider: ImageEmbedProvider::Hash,
            dim: 64,
            vision_model: None,
            text_model: None,
            tokenizer: None,
            preprocessor: None,
        },
        code_formula: false,
        chunker: ChunkerKind::Window,
        chunk_size: 100,
        chunk_overlap: 0.0,
        rrf_k: 60.0,
        api_keys: Vec::new(),
        addr: "127.0.0.1:0".to_string(),
    }
}

/// Smallest valid PNG (1×1 transparent pixel).
fn png_bytes() -> Vec<u8> {
    vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ]
}

async fn chunk_text_kind(pipeline: &Pipeline, doc_id: &str) -> usize {
    pipeline
        .store()
        .get_chunks_range(doc_id, 0, 100)
        .await
        .unwrap()
        .iter()
        .filter(|c| c.kind == ChunkKind::Text)
        .count()
}

async fn chunk_image_kind(pipeline: &Pipeline, doc_id: &str) -> usize {
    pipeline
        .store()
        .get_chunks_range(doc_id, 0, 100)
        .await
        .unwrap()
        .iter()
        .filter(|c| c.kind == ChunkKind::Image)
        .count()
}

async fn find_doc(pipeline: &Pipeline, needle: &str) -> Document {
    pipeline
        .store()
        .list_documents()
        .await
        .unwrap()
        .into_iter()
        .find(|d| d.source_uri.contains(needle))
        .expect("document present")
}

#[tokio::test]
async fn image_only_folder_indexes_images_without_text() {
    let base = temp_base("policy");
    let a = base.join("text");
    let b = base.join("photos");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    std::fs::write(a.join("one.md"), "# One\nunique wombat content alpha.").unwrap();
    std::fs::write(b.join("pic.png"), png_bytes()).unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, vec![a.clone(), b.clone()], vec![b.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let report = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_ingested, 2);
    assert!(
        report.by_type.keys().any(|k| k.contains("image-only")),
        "the image-only file is visible in the summary: {:?}",
        report.by_type.keys().collect::<Vec<_>>()
    );

    // The image file: one image chunk, no text chunks, policy recorded.
    let doc = find_doc(&pipeline, "pic.png").await;
    assert_eq!(chunk_text_kind(&pipeline, &doc.id).await, 0);
    assert_eq!(chunk_image_kind(&pipeline, &doc.id).await, 1);
    assert!(doc.metadata["folder_policy"] == "image-only");
    // The md file: normal full path.
    let md = find_doc(&pipeline, "one.md").await;
    assert!(chunk_text_kind(&pipeline, &md.id).await > 0);
    assert_eq!(chunk_image_kind(&pipeline, &md.id).await, 0);

    // Second run: everything is up to date.
    let again = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(again.documents_skipped, 2);

    std::fs::remove_dir_all(&base).ok();
}

#[tokio::test]
async fn removing_policy_promotes_the_document() {
    let base = temp_base("promote");
    let b = base.join("scans");
    std::fs::create_dir_all(&b).unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    // Pre-seed a document as if it had been ingested under the image-only
    // policy: policy recorded, no `extract`/`text_embed` stage rows, no text
    // chunks. (A real image-only file has exactly this shape; seeding keeps
    // the promotion test offline — no docling models needed.)
    let text = "# Scan\nparaphrase target text.".to_string();
    std::fs::write(b.join("scan.md"), &text).unwrap();
    let bytes = text.clone().into_bytes();
    let cfg = config(&db, vec![b.clone()], vec![b.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let store = pipeline.store();
    let mut doc = Document::new(
        format!("{}/scan.md", b.display()),
        "scan",
        local_search::model::content_hash(&bytes),
    );
    doc.metadata = serde_json::json!({"folder_policy": "image-only"});
    store.upsert_document(&doc).await.unwrap();

    // Policy removed: the next plain ingest promotes the document — the
    // missing `extract` fingerprint makes it stale again.
    let cfg = config(&db, vec![b.clone()], vec![]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let report = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_ingested, 1, "promotion backfills extract");
    let doc = find_doc(&pipeline, "scan.md").await;
    assert!(doc.metadata["folder_policy"] != "image-only");
    assert!(
        chunk_text_kind(&pipeline, &doc.id).await > 0,
        "text backfilled"
    );

    std::fs::remove_dir_all(&base).ok();
}

#[tokio::test]
async fn adding_policy_drops_the_text_channel() {
    let base = temp_base("demote");
    let b = base.join("scans");
    std::fs::create_dir_all(&b).unwrap();
    let png = png_bytes();
    let path = b.join("scan.png");
    std::fs::write(&path, &png).unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, vec![b.clone()], vec![]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    let doc = find_doc(&pipeline, "scan.png").await;
    assert_ne!(
        doc.metadata.get("folder_policy"),
        Some(&"image-only".into())
    );

    // Add the folder to the image-only policy: the text channel is dropped on
    // the next run, without touching the image vector.
    let cfg = config(&db, vec![b.clone()], vec![b.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let report = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_ingested, 1, "the demotion is work");
    let doc = find_doc(&pipeline, "scan.png").await;
    assert!(doc.metadata["folder_policy"] == "image-only");
    assert_eq!(chunk_text_kind(&pipeline, &doc.id).await, 0);
    assert_eq!(
        chunk_image_kind(&pipeline, &doc.id).await,
        1,
        "image vector kept"
    );

    // Third run: settled, skipped.
    let again = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(again.documents_skipped, 1);

    std::fs::remove_dir_all(&base).ok();
}

/// Sanity for the demotion helper: seeding a document with text chunks and
/// clearing them leaves the image chunk untouched.
#[tokio::test]
async fn replace_text_chunks_with_empty_clears_only_text() {
    let base = temp_base("seed");
    let db = base.join("app.db").to_string_lossy().to_string();
    std::fs::create_dir_all(&base).unwrap();
    let cfg = config(&db, vec![base.clone()], vec![]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let store = pipeline.store();

    let png = png_bytes();
    let mut doc = Document::new(
        "x/scan.png",
        "scan",
        local_search::model::content_hash(&png),
    );
    doc.metadata = serde_json::json!({"source": "x/scan.png"});
    store.upsert_document(&doc).await.unwrap();
    let mut text = Chunk::new(&doc.id, 0, "some words", 2);
    text.kind = ChunkKind::Text;
    store
        .replace_text_chunks(&doc.id, &[text.clone()])
        .await
        .unwrap();
    let mut image = Chunk::new(&doc.id, 1, "", 0);
    image.id = "img-hash".to_string();
    image.kind = ChunkKind::Image;
    store
        .upsert_image_chunk(&image, &vec![0.0; 64])
        .await
        .unwrap();

    store.replace_text_chunks(&doc.id, &[]).await.unwrap();
    assert_eq!(count_text_chunks(store, &doc.id).await, 0);
    assert_eq!(count_image_chunks(store, &doc.id).await, 1);

    std::fs::remove_dir_all(&base).ok();
}

async fn count_text_chunks(store: &std::sync::Arc<dyn VectorStore>, doc_id: &str) -> usize {
    store.count_text_chunks_for(doc_id).await.unwrap()
}

async fn count_image_chunks(store: &std::sync::Arc<dyn VectorStore>, doc_id: &str) -> usize {
    store
        .get_chunks_range(doc_id, 0, 100)
        .await
        .unwrap()
        .iter()
        .filter(|c| c.kind == ChunkKind::Image)
        .count()
}
