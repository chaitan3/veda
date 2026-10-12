//! Multi-root ingestion and `--sync` scoping.

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
    std::env::temp_dir().join(format!("ls-mp-{tag}-{}-{nanos}", std::process::id()))
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
    }
}

async fn doc_count(pipeline: &Pipeline) -> usize {
    pipeline.store().count_documents().await.unwrap()
}

#[tokio::test]
async fn multi_root_ingest_and_scoped_sync() {
    let base = temp_base("root");
    let a = base.join("a");
    let b = base.join("b");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    std::fs::write(a.join("one.md"), "# One\nunique wombat content alpha.").unwrap();
    std::fs::write(b.join("two.md"), "# Two\nunique pangolin content beta.").unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    // 1. Ingest root A only.
    let cfg = config(&db, vec![a.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let report = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_ingested, 1);
    assert_eq!(doc_count(&pipeline).await, 1);

    // 2. Add root B with sync: A is kept, B is added.
    let cfg = config(&db, vec![a.clone(), b.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let report = pipeline
        .ingest_all(true, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_ingested, 1, "B's document is new");
    assert_eq!(report.documents_deleted, 0, "A must not be swept");
    assert_eq!(doc_count(&pipeline).await, 2);

    // 3. Ingest root B alone with sync: A is untouched (outside the roots).
    let cfg = config(&db, vec![b.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let report = pipeline
        .ingest_all(true, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(
        report.documents_deleted, 0,
        "sync is scoped to ingested roots"
    );
    assert_eq!(doc_count(&pipeline).await, 2);

    // 4. Remove A's file, then ingest both roots with sync: A is swept, B stays.
    std::fs::remove_file(a.join("one.md")).unwrap();
    let cfg = config(&db, vec![a.clone(), b.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    let report = pipeline
        .ingest_all(true, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_deleted, 1, "the missing file is swept");
    assert_eq!(doc_count(&pipeline).await, 1, "B's document remains");

    // 5. `--force` reprocesses, ignoring the content hash.
    let report = pipeline
        .ingest_all(false, true, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_ingested, 1);
    assert_eq!(report.documents_skipped, 0);

    std::fs::remove_dir_all(&base).ok();
}

#[tokio::test]
async fn unchanged_document_is_skipped() {
    let base = temp_base("skip");
    let a = base.join("a");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::write(a.join("one.md"), "# One\nstable content.").unwrap();
    let db = base.join("app.db").to_string_lossy().to_string();

    let cfg = config(&db, vec![a.clone()]);
    let pipeline = Pipeline::from_config(&cfg).await.unwrap();
    // First run sets the pipeline version and processes.
    let first = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(first.documents_ingested, 1);
    let id_first = pipeline.store().list_documents().await.unwrap()[0]
        .id
        .clone();
    // Second run: unchanged, so skipped — and the document id is reused.
    let report = pipeline
        .ingest_all(false, false, StageSet::all())
        .await
        .unwrap();
    assert_eq!(report.documents_skipped, 1);
    assert_eq!(report.documents_ingested, 0);
    let id_second = pipeline.store().list_documents().await.unwrap()[0]
        .id
        .clone();
    assert_eq!(
        id_first, id_second,
        "unchanged content keeps its document id"
    );

    std::fs::remove_dir_all(&base).ok();
}
