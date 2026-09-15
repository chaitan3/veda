//! Stage-skip regression: a document whose `extract` fingerprint matches must
//! never re-convert, even when it has zero text chunks (scanned/blank PDFs used
//! to re-extract on every run).

use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::model::{Document, content_hash};
use local_search::pipeline::{Pipeline, StageSet};
use std::path::PathBuf;

fn temp_base(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ls-skip-{tag}-{}-{nanos}", std::process::id()))
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
        },
        extract: ExtractPipeline::Onnx,
        image: ImageEmbedConfig {
            provider: ImageEmbedProvider::Off,
            dim: 768,
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

#[tokio::test]
async fn matching_fingerprint_skips_even_without_text() {
    let base = temp_base("notext");
    std::fs::create_dir_all(&base).unwrap();
    let text = b"# Blank\nunchanged body text.".to_vec();
    std::fs::write(base.join("doc.md"), &text).unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, base.clone());
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let fp = pipeline.fingerprints();

    // Seed the exact state a scanned/blank document ends in after one
    // successful extract: stage fingerprints recorded, zero text chunks.
    let mut doc = Document::new(
        format!("{}/doc.md", base.display()),
        "doc",
        content_hash(&text),
    );
    doc.metadata = serde_json::json!({"source": format!("{}/doc.md", base.display())});
    pipeline.store().upsert_document(&doc).await.unwrap();
    pipeline
        .store()
        .set_document_stage(&doc.id, "extract", &fp.extract)
        .await
        .unwrap();
    pipeline
        .store()
        .set_document_stage(&doc.id, "text_embed", &fp.text_embed)
        .await
        .unwrap();

    // Without the fix, the `!has_text` clause re-converted this on every run.
    let report = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_ingested, 0, "must skip, not re-extract");
    assert_eq!(report.documents_skipped, 1);

    // `--force --stages extract` still re-extracts on demand.
    let forced = pipeline
        .ingest_all(false, true, StageSet::parse("extract").unwrap())
        .await
        .unwrap();
    assert_eq!(forced.documents_ingested, 1);

    std::fs::remove_dir_all(&base).ok();
}

#[tokio::test]
async fn missing_fingerprint_still_backfills() {
    let base = temp_base("recover");
    std::fs::create_dir_all(&base).unwrap();
    let text = "# Recovery\nsome words to chunk.".to_string();
    std::fs::write(base.join("doc.md"), &text).unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, base.clone());
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();

    // A document row with no stage rows at all (interrupted earlier ingest):
    // the missing fingerprint must trigger extract.
    let mut doc = Document::new(
        format!("{}/doc.md", base.display()),
        "doc",
        content_hash(text.as_bytes()),
    );
    doc.metadata = serde_json::json!({"source": format!("{}/doc.md", base.display())});
    pipeline.store().upsert_document(&doc).await.unwrap();

    let report = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_ingested, 1);
    assert_eq!(
        pipeline
            .store()
            .count_text_chunks_for(&doc.id)
            .await
            .unwrap(),
        1
    );

    // …and the second run is a clean skip.
    let again = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(again.documents_skipped, 1);

    std::fs::remove_dir_all(&base).ok();
}
