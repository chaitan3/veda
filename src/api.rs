//! Read-only HTTP API over a [`Pipeline`].
//!
//! Ingest is deliberately **not** exposed: it runs as a CLI command. The
//! service only searches and reports, which keeps it side-effect free and safe
//! to expose on loopback. Authentication is optional and disabled when
//! `LS_API_KEYS` is empty.

use crate::model::{Chunk, Document, RetrievalMode};
use crate::pipeline::Pipeline;
use axum::{
    Json, Router,
    extract::{Path, Query, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Chunks returned by the item endpoint when `limit` is not given; also its cap.
const DEFAULT_CHUNK_WINDOW: usize = 20;
const MAX_CHUNK_WINDOW: usize = 100;
/// Documents returned by the collection endpoint when `limit` is not given.
const DEFAULT_DOC_PAGE: usize = 100;
const MAX_DOC_PAGE: usize = 1000;

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
        .route("/api/documents/{id}", get(document_by_id))
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

/// A compact document reference exposed by the API.
///
/// Deliberately omits `metadata` — which can hold the full converted text and
/// per-file metrics — and the internal content hash, so search hits stay
/// token-bounded. Full metadata lives in the store, never on the wire.
#[derive(Serialize)]
struct DocumentRef {
    id: String,
    title: String,
    source_uri: String,
    created_at: String,
}

impl From<&Document> for DocumentRef {
    fn from(doc: &Document) -> Self {
        DocumentRef {
            id: doc.id.clone(),
            title: doc.title.clone(),
            source_uri: doc.source_uri.clone(),
            created_at: doc.created_at.clone(),
        }
    }
}

/// One search hit: the chunk, its score, and its source document, so the result
/// is self-contained and directly citable.
#[derive(Serialize)]
struct SearchResult {
    chunk: Chunk,
    score: f32,
    document: DocumentRef,
}

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    mode: String,
    count: usize,
    results: Vec<SearchResult>,
    corpus_version: i64,
}

async fn run_search(state: &AppState, query: &str, mode: RetrievalMode, k: usize) -> Response {
    if query.trim().is_empty() {
        return err(StatusCode::BAD_REQUEST, "query must not be empty");
    }
    match state.pipeline.query(mode, query, k).await {
        Ok(mut hits) => {
            // Embeddings are an internal detail; don't ship them to clients.
            for hit in hits.iter_mut() {
                hit.chunk.embedding = None;
            }
            let store = state.pipeline.store();
            // Hydrate each hit's parent document with a single batched lookup.
            let mut ids: Vec<String> = hits.iter().map(|h| h.chunk.doc_id.clone()).collect();
            ids.sort();
            ids.dedup();
            let documents = match store.get_documents_by_ids(&ids).await {
                Ok(d) => d,
                Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            };
            let corpus_version = store.corpus_version().await.unwrap_or(0);
            let results: Vec<SearchResult> = hits
                .into_iter()
                .filter_map(|hit| {
                    documents
                        .get(&hit.chunk.doc_id)
                        .map(|document| SearchResult {
                            chunk: hit.chunk,
                            score: hit.score,
                            document: DocumentRef::from(document),
                        })
                })
                .collect();
            Json(SearchResponse {
                query: query.to_string(),
                mode: mode.to_string(),
                count: results.len(),
                results,
                corpus_version,
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

#[derive(Deserialize)]
struct PageParams {
    offset: Option<usize>,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct DocumentsResponse {
    count: usize,
    offset: usize,
    limit: usize,
    total: usize,
    next: Option<usize>,
    documents: Vec<DocumentRef>,
}

async fn documents(State(state): State<AppState>, Query(p): Query<PageParams>) -> Response {
    let store = state.pipeline.store();
    let total = match store.count_documents().await {
        Ok(n) => n,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let limit = p.limit.unwrap_or(DEFAULT_DOC_PAGE).clamp(1, MAX_DOC_PAGE);
    let offset = p.offset.unwrap_or(0);
    let documents: Vec<DocumentRef> = match store.list_documents().await {
        Ok(all) => all
            .iter()
            .skip(offset)
            .take(limit)
            .map(DocumentRef::from)
            .collect(),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let next = (offset + documents.len() < total).then_some(offset + documents.len());
    Json(DocumentsResponse {
        count: documents.len(),
        offset,
        limit,
        total,
        next,
        documents,
    })
    .into_response()
}

/// Fetch one document plus a bounded, pageable slice of its chunks.
///
/// This is the single expansion call: with no params it returns the first
/// `limit` chunks; callers page with `offset`/`next`. Chunk `ordinal`s come from
/// search hits, so a caller wanting context around a hit passes
/// `offset = ordinal - radius`.
async fn document_by_id(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(p): Query<PageParams>,
) -> Response {
    let store = state.pipeline.store();
    let document = match store.get_document(&id).await {
        Ok(Some(doc)) => DocumentRef::from(&doc),
        Ok(None) => return err(StatusCode::NOT_FOUND, format!("no document with id '{id}'")),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let total = match store.count_chunks_for(&id).await {
        Ok(n) => n,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let limit = p
        .limit
        .unwrap_or(DEFAULT_CHUNK_WINDOW)
        .clamp(1, MAX_CHUNK_WINDOW);
    let offset = p.offset.unwrap_or(0);
    if offset >= total {
        return Json(serde_json::json!({
            "document": document,
            "offset": offset,
            "limit": limit,
            "total": total,
            "chunks": [],
            "next": null,
            "corpus_version": store.corpus_version().await.unwrap_or(0),
        }))
        .into_response();
    }
    let chunks = match store.get_chunks_range(&id, offset, limit).await {
        Ok(c) => c,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let next = (offset + chunks.len() < total).then_some(offset + chunks.len());
    Json(serde_json::json!({
        "document": document,
        "offset": offset,
        "limit": limit,
        "total": total,
        "chunks": chunks,
        "next": next,
        "corpus_version": store.corpus_version().await.unwrap_or(0),
    }))
    .into_response()
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": msg.into() }))).into_response()
}
