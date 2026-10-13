//! Read-only HTTP API over a [`Pipeline`].
//!
//! Ingest is deliberately **not** exposed: it runs as a CLI command. The
//! service only searches and reports, which keeps it side-effect free and safe
//! to expose on loopback. Authentication is optional and disabled when
//! `LS_API_KEYS` is empty.

use crate::model::{Chunk, ChunkKind, Document, RetrievalMode, Scored};
use crate::pipeline::Pipeline;
use crate::store::ImageSpace;
use axum::{
    Json, Router,
    extract::{Path, Query, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine as _;
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
        .route("/api/search/image", post(search_image))
        .route("/api/stats", get(stats))
        .route("/api/documents", get(documents))
        .route("/api/documents/{id}", get(document_by_id))
        .route("/api/media/{chunk_id}", get(media))
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

/// One search hit: the chunk, its score, its source document, and (for image
/// chunks) a URL to the bytes, so the result is self-contained and citable.
#[derive(Serialize)]
struct SearchResult {
    chunk: Chunk,
    score: f32,
    document: DocumentRef,
    #[serde(skip_serializing_if = "Option::is_none")]
    media_url: Option<String>,
}

/// Where an image chunk's bytes are served from. Nothing is stored: the route
/// resolves the source document (standalone image files, for now).
fn media_url(chunk: &Chunk) -> Option<String> {
    (chunk.kind == ChunkKind::Image).then(|| format!("/api/media/{}", chunk.id))
}

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    mode: String,
    count: usize,
    results: Vec<SearchResult>,
    corpus_version: i64,
}

/// Strip embeddings and hydrate each hit's parent document (one batched lookup).
async fn hydrate(
    state: &AppState,
    mut hits: Vec<Scored>,
) -> std::result::Result<Vec<SearchResult>, (StatusCode, String)> {
    // Embeddings are an internal detail; don't ship them to clients.
    for hit in hits.iter_mut() {
        hit.chunk.embedding = None;
    }
    let store = state.pipeline.store();
    let mut ids: Vec<String> = hits.iter().map(|h| h.chunk.doc_id.clone()).collect();
    ids.sort();
    ids.dedup();
    let documents = store
        .get_documents_by_ids(&ids)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(hits
        .into_iter()
        .filter_map(|hit| {
            documents
                .get(&hit.chunk.doc_id)
                .map(|document| SearchResult {
                    media_url: media_url(&hit.chunk),
                    chunk: hit.chunk,
                    score: hit.score,
                    document: DocumentRef::from(document),
                })
        })
        .collect())
}

async fn respond(state: &AppState, query: String, mode: String, hits: Vec<Scored>) -> Response {
    let corpus_version = state.pipeline.store().corpus_version().await.unwrap_or(0);
    match hydrate(state, hits).await {
        Ok(results) => Json(SearchResponse {
            query,
            mode,
            count: results.len(),
            results,
            corpus_version,
        })
        .into_response(),
        Err((status, msg)) => err(status, msg),
    }
}

async fn run_search(state: &AppState, query: &str, mode: RetrievalMode, limit: usize) -> Response {
    if query.trim().is_empty() {
        return err(StatusCode::BAD_REQUEST, "query must not be empty");
    }
    match state.pipeline.query(mode, query, limit).await {
        Ok(hits) => respond(state, query.to_string(), mode.to_string(), hits).await,
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchParams {
    query: Option<String>,
    mode: Option<String>,
    limit: Option<usize>,
}

async fn search_get(State(state): State<AppState>, Query(p): Query<SearchParams>) -> Response {
    let query = p.query.unwrap_or_default();
    let mode = p
        .mode
        .and_then(|m| m.parse().ok())
        .unwrap_or(RetrievalMode::Hybrid);
    let limit = p.limit.unwrap_or(5).clamp(1, 100);
    run_search(&state, &query, mode, limit).await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchBody {
    query: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

async fn search_post(State(state): State<AppState>, Json(body): Json<SearchBody>) -> Response {
    let mode = body
        .mode
        .and_then(|m| m.parse().ok())
        .unwrap_or(RetrievalMode::Hybrid);
    let limit = body.limit.unwrap_or(5).clamp(1, 100);
    run_search(&state, &body.query, mode, limit).await
}

/// Liveness probe. Deliberately public and minimal: any authenticated caller
/// wanting corpus details uses `GET /api/stats` (same handler, protected).
/// Keeping the body free of counts/embedder paths means an unauthenticated
/// probe cannot enumerate the corpus.
async fn health() -> Response {
    Json(serde_json::json!({"status": "ok"})).into_response()
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
    let fp = pipeline.fingerprints();
    Ok(serde_json::json!({
        "status": "ok",
        "embedder": pipeline.embedder().id(),
        "embed_dim": pipeline.embedder().dim(),
        "image_space": match pipeline.store().image_space() {
            ImageSpace::Off => "off",
            ImageSpace::Split(_) => "split",
            ImageSpace::Unified => "unified",
        },
        "image_embedder": pipeline.image_embedder().map(|e| e.id()).unwrap_or("off"),
        // Sequences cut to the embedder's window since startup (silent quality
        // loss otherwise; raise LS_EMBED_MAX_TOKENS or lower LS_CHUNK_SIZE).
        "truncated_sequences": crate::embed::truncated_sequences(),
        "image_embed_dim": pipeline.image_embedder().map(|e| e.dim()).unwrap_or(0),
        "extract_pipeline": format!("{:?}", pipeline.config().extract).to_lowercase(),
        "documents": documents,
        "chunks": chunks,
        "corpus_version": corpus_version,
        "fingerprints": {
            "extract": fp.extract,
            "text_embed": fp.text_embed,
            "image_embed": fp.image_embed,
        },
    }))
}

/// Paging for the document listing, plus the file-name/title filter.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentListParams {
    offset: Option<usize>,
    limit: Option<usize>,
    /// Case-insensitive substring match over the document title / source path.
    query: Option<String>,
}

/// Paging for one document's chunk slice — no filter: a single document is
/// already the scope. Unknown parameters are rejected, so a `query` sent here
/// by mistake fails loudly instead of paging the whole document unfiltered.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChunkPageParams {
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

async fn documents(State(state): State<AppState>, Query(p): Query<DocumentListParams>) -> Response {
    let store = state.pipeline.store();
    let limit = p.limit.unwrap_or(DEFAULT_DOC_PAGE).clamp(1, MAX_DOC_PAGE);
    let offset = p.offset.unwrap_or(0);
    // `query` is a case-insensitive substring match over the title and the
    // source path — the way to find a document by file name or folder. The
    // whole documents table is loaded for paging anyway, so filtering here
    // costs no extra query; move it into SQL if a corpus ever outgrows memory.
    let needle = p
        .query
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase);
    let all = match store.list_documents().await {
        Ok(all) => all,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let matched: Vec<&Document> = match &needle {
        Some(needle) => all
            .iter()
            .filter(|d| {
                d.title.to_lowercase().contains(needle)
                    || d.source_uri.to_lowercase().contains(needle)
            })
            .collect(),
        None => all.iter().collect(),
    };
    let total = matched.len();
    let documents: Vec<DocumentRef> = matched
        .iter()
        .skip(offset)
        .take(limit)
        .map(|d| DocumentRef::from(*d))
        .collect();
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
    Query(p): Query<ChunkPageParams>,
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageSearchBody {
    /// Base64-encoded query image bytes (PNG/JPEG/…).
    image_base64: String,
    #[serde(default)]
    limit: Option<usize>,
}

/// Query-by-image: embed the posted image and search the image space.
async fn search_image(
    State(state): State<AppState>,
    Json(body): Json<ImageSearchBody>,
) -> Response {
    let limit = body.limit.unwrap_or(5).clamp(1, 100);
    let bytes = match base64::engine::general_purpose::STANDARD.decode(body.image_base64.as_bytes())
    {
        Ok(b) => b,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                format!("image_base64 is not valid base64: {e}"),
            );
        }
    };
    if bytes.is_empty() {
        return err(StatusCode::BAD_REQUEST, "image is empty");
    }
    match state.pipeline.query_image(&bytes, limit).await {
        Ok(hits) => respond(&state, String::new(), "image".to_string(), hits).await,
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// Serve an image chunk's bytes from its source document (standalone image
/// files only; nothing is stored).
async fn media(State(state): State<AppState>, Path(chunk_id): Path<String>) -> Response {
    let store = state.pipeline.store();
    let chunk = match store.get_chunk(&chunk_id).await {
        Ok(Some(c)) => c,
        Ok(None) => return err(StatusCode::NOT_FOUND, "no such chunk"),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    if chunk.kind != ChunkKind::Image {
        return err(StatusCode::NOT_FOUND, "chunk is not an image");
    }
    let document = match store.get_document(&chunk.doc_id).await {
        Ok(Some(d)) => d,
        Ok(None) => return err(StatusCode::NOT_FOUND, "document is missing"),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let bytes = match tokio::fs::read(&document.source_uri).await {
        Ok(b) => b,
        Err(e) => {
            return err(
                StatusCode::NOT_FOUND,
                format!("source image unreadable: {e}"),
            );
        }
    };
    let mime = chunk
        .metadata
        .get("mime")
        .and_then(|m| m.as_str())
        .unwrap_or("application/octet-stream")
        .to_string();
    ([(axum::http::header::CONTENT_TYPE, mime)], bytes).into_response()
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": msg.into() }))).into_response()
}
