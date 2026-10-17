//! Remembered conversion failures: a failed file is not retried until it
//! changes or the extractor identity changes.

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
    std::env::temp_dir().join(format!("ls-fail-{tag}-{}-{nanos}", std::process::id()))
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
        embed_idle: None,
    }
}

#[tokio::test]
async fn failed_conversion_is_remembered_and_not_retried() {
    let base = temp_base("remember");
    let a = base.join("a");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::write(a.join("good.md"), "# Good\nstable content.").unwrap();
    // A PDF with no valid structure: docling/pdfium cannot convert it.
    let bad = a.join("bad.pdf");
    std::fs::write(&bad, b"%PDF-1.4\nthis is not a real pdf body\n").unwrap();
    let bad_uri = bad.to_string_lossy().to_string();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, vec![a.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();

    // First run: the good file ingests, the bad one fails and is remembered.
    let first = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(first.documents_ingested, 1);
    assert_eq!(first.documents_failed, 1, "the bad pdf fails");
    assert!(
        pipeline
            .store()
            .document_failure(&bad_uri)
            .await
            .unwrap()
            .is_some(),
        "the failure is remembered"
    );
    assert_eq!(
        pipeline.store().count_documents().await.unwrap(),
        1,
        "a failed conversion leaves no placeholder document"
    );

    // Second run: the failure is remembered and not retried.
    let second = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(second.documents_failed, 0, "not retried");
    assert_eq!(second.documents_failed_known, 1);

    // Extractor change (simulated by editing the remembered identity): retried.
    let mut failure = pipeline
        .store()
        .document_failure(&bad_uri)
        .await
        .unwrap()
        .unwrap();
    failure.extractor_id = "older-build".to_string();
    pipeline
        .store()
        .record_document_failure(&failure)
        .await
        .unwrap();
    let third = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(third.documents_failed, 1, "extractor change retries it");

    // File change (different bytes/size): retried.
    std::fs::write(&bad, b"%PDF-1.4\na completely different invalid body").unwrap();
    let fourth = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(fourth.documents_failed, 1, "a changed file is retried");

    // `--force` retries a remembered failure without any change.
    let fifth = pipeline
        .ingest_all(false, true, StageSet::all())
        .await
        .unwrap();
    assert_eq!(fifth.documents_failed, 1, "--force retries");

    // Deleting the file with `--sync` clears the failure.
    std::fs::remove_file(&bad).unwrap();
    pipeline
        .ingest_all(true, false, StageSet::all())
        .await
        .unwrap();
    assert!(
        pipeline
            .store()
            .document_failure(&bad_uri)
            .await
            .unwrap()
            .is_none(),
        "a vanished path's failure is cleared"
    );

    std::fs::remove_dir_all(&base).ok();
}
