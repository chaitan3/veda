//! Read-only HTTP API over a [`Pipeline`].
//!
//! Ingest is deliberately **not** exposed: it runs as a CLI command. The
//! service only searches and reports, which keeps it side-effect free and safe
//! to expose on loopback. Authentication is optional and disabled when
//! `LS_API_KEYS` is empty.

use crate::model::{Document, RetrievalMode, Scored};
use crate::pipeline::Pipeline;
use axum::{
    Json, Router,
    extract::{Query, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Shared service state.
#[derive(Clone)]
pub struct AppState {
    pub pipeline: Arc<Pipeline>,
    pub api_keys: Arc<Vec<String>>,
}

/// Build the router.
pub fn router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/api/search", get(search_get).post(search_post))
        .route("/api/stats", get(stats))
        .route("/api/documents", get(documents))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth));

    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------

async fn auth(State(state): State<AppState>, req: Request, next: Next) -> Response {
    if state.api_keys.is_empty() {
        return next.run(req).await;
    }
    if let Some(key) = presented_key(req.headers(), req.uri().query())
        && state.api_keys.iter().any(|k| k == &key)
    {
        return next.run(req).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({ "error": "missing or invalid API key" })),
    )
        .into_response()
}

fn presented_key(headers: &HeaderMap, query: Option<&str>) -> Option<String> {
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v.to_string());
    }
    if let Some(token) = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    {
        return Some(token.to_string());
    }
    let query = query?;
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=')
            && k == "api_key"
        {
            return Some(v.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    mode: String,
    count: usize,
    results: Vec<Scored>,
}

async fn run_search(state: &AppState, query: &str, mode: RetrievalMode, k: usize) -> Response {
    if query.trim().is_empty() {
        return err(StatusCode::BAD_REQUEST, "query must not be empty");
    }
    match state.pipeline.query(mode, query, k).await {
        Ok(mut results) => {
            // Embeddings are an internal detail; don't ship them to clients.
            for hit in results.iter_mut() {
                hit.chunk.embedding = None;
            }
            Json(SearchResponse {
                query: query.to_string(),
                mode: mode.to_string(),
                count: results.len(),
                results,
            })
            .into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize)]
struct SearchParams {
    q: Option<String>,
    mode: Option<String>,
    k: Option<usize>,
}

async fn search_get(State(state): State<AppState>, Query(p): Query<SearchParams>) -> Response {
    let query = p.q.unwrap_or_default();
    let mode = p
        .mode
        .and_then(|m| m.parse().ok())
        .unwrap_or(RetrievalMode::Hybrid);
    let k = p.k.unwrap_or(5).clamp(1, 100);
    run_search(&state, &query, mode, k).await
}

#[derive(Deserialize)]
struct SearchBody {
    query: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    k: Option<usize>,
}

async fn search_post(State(state): State<AppState>, Json(body): Json<SearchBody>) -> Response {
    let mode = body
        .mode
        .and_then(|m| m.parse().ok())
        .unwrap_or(RetrievalMode::Hybrid);
    let k = body.top_k.or(body.k).unwrap_or(5).clamp(1, 100);
    run_search(&state, &body.query, mode, k).await
}

async fn health(State(state): State<AppState>) -> Response {
    match stats_json(&state).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn stats(State(state): State<AppState>) -> Response {
    match stats_json(&state).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn stats_json(state: &AppState) -> std::result::Result<serde_json::Value, String> {
    let pipeline = &state.pipeline;
    let store = pipeline.store();
    let documents = store.count_documents().await.map_err(|e| e.to_string())?;
    let chunks = store.count_chunks().await.map_err(|e| e.to_string())?;
    let corpus_version = store.corpus_version().await.map_err(|e| e.to_string())?;
    let pipeline_version = store.pipeline_version().await.map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "status": "ok",
        "embedder": pipeline.embedder().id(),
        "embed_dim": pipeline.embedder().dim(),
        "extract_pipeline": format!("{:?}", pipeline.config().extract).to_lowercase(),
        "documents": documents,
        "chunks": chunks,
        "corpus_version": corpus_version,
        "pipeline_version": pipeline_version,
    }))
}

#[derive(Serialize)]
struct DocumentsResponse {
    count: usize,
    documents: Vec<Document>,
}

async fn documents(State(state): State<AppState>) -> Response {
    match state.pipeline.store().list_documents().await {
        Ok(documents) => Json(DocumentsResponse {
            count: documents.len(),
            documents,
        })
        .into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": msg.into() }))).into_response()
}
