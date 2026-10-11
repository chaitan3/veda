//! HTTP API contract: search hits are self-contained (chunk + ordinal +
//! hydrated document), and one bounded, pageable item endpoint expands a
//! document.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use local_search::api::{AppState, router};
use local_search::config::{
    ChunkerKind, Config, EmbedConfig, EmbedProvider, ExtractPipeline, ImageEmbedConfig,
    ImageEmbedProvider,
};
use local_search::embed::{Embedder, hash::HashEmbedder};
use local_search::model::{Chunk, Document};
use local_search::pipeline::Pipeline;
use std::sync::Arc;
use tower::ServiceExt;

fn config(db: &str) -> Config {
    Config {
        db_path: db.to_string(),
        source_paths: vec!["docs".to_string()],
        image_only_paths: Vec::new(),
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

/// Build a router over a temp store holding one document with three chunks.
async fn setup() -> (axum::Router, String) {
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("ls-api-{}-{nanos}.db", std::process::id()));
    let cfg = config(path.to_str().unwrap());
    let pipeline = Arc::new(Pipeline::from_config(&cfg).await.unwrap());

    let embedder = HashEmbedder::new(cfg.embed.dim);
    let doc = Document::new("file:///sample.txt", "Sample", "deadbeef");
    pipeline.store().upsert_document(&doc).await.unwrap();
    let texts = [
        "the quick brown fox",
        "a mountain lake at sunset",
        "lazy dog sleeps",
    ];
    let mut chunks = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let mut chunk = Chunk::new(
            &doc.id,
            i as i64,
            *text,
            text.split_whitespace().count() as i64,
        );
        chunk.embedding = Some(
            embedder
                .embed(&[text.to_string()])
                .await
                .unwrap()
                .pop()
                .unwrap(),
        );
        chunks.push(chunk);
    }
    pipeline
        .store()
        .replace_text_chunks(&doc.id, &chunks)
        .await
        .unwrap();
    let pairs: Vec<(String, Vec<f32>)> = chunks
        .iter()
        .map(|c| (c.id.clone(), c.embedding.clone().unwrap()))
        .collect();
    pipeline.store().set_text_vectors(&pairs).await.unwrap();
    pipeline.store().bump_corpus_version().await.unwrap();

    let state = AppState {
        pipeline,
        api_keys: Arc::new(Vec::new()),
    };
    (router(state), doc.id)
}

fn get(uri: &str) -> Request<Body> {
    Request::builder().uri(uri).body(Body::empty()).unwrap()
}

async fn json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn search_hits_are_self_contained() {
    let (app, doc_id) = setup().await;
    let resp = app
        .oneshot(get("/api/search?q=mountain%20lake&mode=vector&k=5"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json(resp).await;
    assert_eq!(body["mode"], "vector");
    assert!(body["corpus_version"].as_i64().unwrap() >= 1);

    let results = body["results"].as_array().unwrap();
    assert!(!results.is_empty());
    let hit = &results[0];
    // The ordinal is present so callers can compute a context window.
    assert!(hit["chunk"]["ordinal"].is_i64());
    // The parent document is hydrated, so no second lookup is needed.
    assert_eq!(hit["document"]["id"].as_str().unwrap(), doc_id);
    assert_eq!(hit["document"]["source_uri"], "file:///sample.txt");
    assert_eq!(hit["document"]["title"], "Sample");
    // The reference is compact: no metadata blob (which can hold full text).
    assert!(hit["document"].get("metadata").is_none());
    // Embeddings never leave the service.
    assert!(hit["chunk"]["embedding"].is_null());
}

#[tokio::test]
async fn image_mode_reports_a_disabled_image_channel() {
    let (app, _doc_id) = setup().await;
    // The test config has LS_IMAGE_EMBED_PROVIDER off: mode=image must say
    // so, rather than returning text hits (this corpus has no image space).
    let resp = app
        .oneshot(get("/api/search?q=mountain%20lake&mode=image"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = json(resp).await;
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("LS_IMAGE_EMBED_PROVIDER"),
        "unexpected error body: {body}"
    );
}

#[tokio::test]
async fn unknown_mode_falls_back_to_hybrid() {
    let (app, _doc_id) = setup().await;
    let resp = app
        .oneshot(get("/api/search?q=mountain&mode=banana"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(json(resp).await["mode"], "hybrid");
}

#[tokio::test]
async fn item_endpoint_pages_chunks() {
    let (app, doc_id) = setup().await;
    let resp = app
        .clone()
        .oneshot(get(&format!("/api/documents/{doc_id}?limit=2")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json(resp).await;
    assert_eq!(body["total"].as_u64(), Some(3));
    assert_eq!(body["chunks"].as_array().unwrap().len(), 2);
    assert_eq!(body["next"].as_u64(), Some(2));
    assert_eq!(body["document"]["source_uri"], "file:///sample.txt");

    // Second page: the remainder, then no next cursor.
    let resp = app
        .oneshot(get(&format!("/api/documents/{doc_id}?offset=2&limit=2")))
        .await
        .unwrap();
    let body = json(resp).await;
    assert_eq!(body["chunks"].as_array().unwrap().len(), 1);
    assert!(body["next"].is_null());
}

#[tokio::test]
async fn offset_past_end_is_empty_not_an_error() {
    let (app, doc_id) = setup().await;
    let resp = app
        .oneshot(get(&format!("/api/documents/{doc_id}?offset=99")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json(resp).await;
    assert!(body["chunks"].as_array().unwrap().is_empty());
    assert_eq!(body["total"].as_u64(), Some(3));
    assert!(body["next"].is_null());
}

#[tokio::test]
async fn unknown_document_is_404() {
    let (app, _) = setup().await;
    let resp = app
        .oneshot(get("/api/documents/does-not-exist"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn collection_is_paginated() {
    let (app, _) = setup().await;
    let resp = app.oneshot(get("/api/documents?limit=1")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json(resp).await;
    assert_eq!(body["total"].as_u64(), Some(1));
    assert_eq!(body["count"].as_u64(), Some(1));
    assert!(body["documents"][0]["source_uri"].is_string());
    assert!(body["documents"][0].get("metadata").is_none());
}

#[tokio::test]
async fn documents_query_matches_title_and_path() {
    let (app, doc_id) = setup().await;

    // Path fragment, title, and case-insensitive forms all match.
    for q in ["sample.txt", "Sample", "SAMPLE"] {
        let resp = app
            .clone()
            .oneshot(get(&format!("/api/documents?query={q}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json(resp).await;
        assert_eq!(body["total"].as_u64(), Some(1), "query={q}");
        assert_eq!(body["documents"][0]["id"].as_str().unwrap(), doc_id);
    }

    // No match: an empty page, not an error.
    let resp = app
        .oneshot(get("/api/documents?query=nothing-matches-this"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json(resp).await;
    assert_eq!(body["count"].as_u64(), Some(0));
    assert_eq!(body["total"].as_u64(), Some(0));
    assert!(body["next"].is_null());
}
