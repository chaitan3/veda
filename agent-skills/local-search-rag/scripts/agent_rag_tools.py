#!/usr/bin/env python3
"""Tool wrappers over the local-search HTTP API, shaped for agent tool-calling.

Every function maps 1:1 to an endpoint of the read-only service (see
docs/ARCHITECTURE.md). Nothing here mutates the corpus: ingest stays CLI-side,
so these tools are safe to hand to an LLM agent.

Configuration (environment):
    LS_API_URL   base URL of the service   (default http://127.0.0.1:8080)
    LS_API_KEY   value of one entry of the server's LS_API_KEYS
                 (auth is disabled server-side when LS_API_KEYS is empty)

Integration: give TOOL_SCHEMAS to your LLM client as the tool definitions,
route tool calls through dispatch(name, **args), and feed the returned
dicts back into the model. The functions are also directly importable, and
the module works standalone for smoke-testing:

    python3 agent_rag_tools.py search "hybrid retrieval sqlite"
    python3 agent_rag_tools.py stats
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = os.environ.get("LS_API_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.environ.get("LS_API_KEY", "").strip()
VALID_MODES = ("hybrid", "vector", "bm25", "image")


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {"x-api-key": API_KEY} if API_KEY else {}


def _request(method: str, path: str, body: dict | None = None, timeout: float = 60.0):
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=_headers())
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
        if payload and payload[:1] in (b"{", b"["):
            return json.loads(payload)
        return payload
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"local-search {method} {path} -> HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"local-search unreachable at {BASE_URL} (is `serve` running?): {e.reason}"
        ) from e


def _compact_hit(hit: dict) -> dict:
    """One search result, trimmed to what an LLM needs and can cite."""
    chunk, doc = hit["chunk"], hit["document"]
    out = {
        "source": doc["title"] or doc["source_uri"],
        "source_uri": doc["source_uri"],
        "doc_id": doc["id"],
        "chunk_id": chunk["id"],
        "ordinal": chunk["ordinal"],
        "score": round(hit["score"], 4),
        "text": chunk["text"],
    }
    if chunk.get("metadata"):
        # heading path etc. — small, and helps the model name the section
        out["metadata"] = chunk["metadata"]
    if hit.get("media_url"):
        out["media_url"] = f"{BASE_URL}{hit['media_url']}"
    return out


def _hits_from_search(payload: dict) -> dict:
    return {
        "mode": payload["mode"],
        "count": payload["count"],
        "corpus_version": payload.get("corpus_version"),
        "results": [_compact_hit(h) for h in payload["results"]],
        "_hint": (
            "Cite `source` in answers. Need more context around a hit? Call "
            "get_context_around(doc_id, ordinal). Full doc: get_document(doc_id)."
        ),
    }


# ---------------------------------------------------------------------------
# tools (one function per agent-facing capability)
# ---------------------------------------------------------------------------

def search_documents(query: str, mode: str = "hybrid", limit: int = 5) -> dict:
    """Rank text chunks for a query. GET /api/search."""
    if not query or not query.strip():
        return {"error": "query must not be empty"}
    if mode not in VALID_MODES:
        return {"error": f"mode must be one of {VALID_MODES}"}
    limit = max(1, min(int(limit), 100))
    payload = _request(
        "GET",
        f"/api/search?query={urllib.parse.quote(query)}&mode={mode}&limit={limit}",
    )
    return _hits_from_search(payload)


def search_image(image: str, limit: int = 5) -> dict:
    """Find indexed images similar to `image` (local path or http(s) URL).
    POST /api/search/image with base64-encoded bytes."""
    if image.startswith(("http://", "https://")):
        with urllib.request.urlopen(image, timeout=30) as resp:
            data = resp.read()
    else:
        with open(os.path.expanduser(image), "rb") as f:
            data = f.read()
    limit = max(1, min(int(limit), 100))
    payload = _request(
        "POST",
        "/api/search/image",
        {"image_base64": base64.b64encode(data).decode(), "limit": limit},
    )
    return _hits_from_search(payload)


def list_documents(offset: int = 0, limit: int = 100, query: str | None = None) -> dict:
    """Browse the corpus, or find a document by file name / title / folder.

    GET /api/documents — `query` is a case-insensitive substring over the
    title and source path; use `next` from the response to page. Don't pull
    the whole index into context."""
    url = f"/api/documents?offset={int(offset)}&limit={max(1, min(int(limit), 1000))}"
    if query and query.strip():
        url += "&query=" + urllib.parse.quote(query.strip())
    payload = _request("GET", url)
    return {
        "total": payload["total"],
        "next": payload["next"],
        "documents": [
            {
                "id": d["id"],
                "title": d["title"],
                "source_uri": d["source_uri"],
                "created_at": d["created_at"],
            }
            for d in payload["documents"]
        ],
    }


def get_document(doc_id: str, offset: int = 0, limit: int = 20) -> dict:
    """One document + a bounded slice of its chunks. GET /api/documents/{id}.
    The expansion call when a search hit isn't enough context."""
    payload = _request(
        "GET",
        f"/api/documents/{urllib.parse.quote(doc_id)}"
        f"?offset={int(offset)}&limit={max(1, min(int(limit), 100))}",
    )
    return {
        "document": payload["document"],
        "total_chunks": payload["total"],
        "next": payload["next"],
        "chunks": [
            {
                "chunk_id": c["id"],
                "ordinal": c["ordinal"],
                "kind": c["kind"],
                "text": c["text"],
            }
            for c in payload["chunks"]
        ],
    }


def get_context_around(doc_id: str, ordinal: int, radius: int = 2) -> dict:
    """The chunks surrounding a hit, so the model can read the passage in
    context instead of the isolated slice search returned."""
    ordinal = max(0, int(ordinal) - max(0, int(radius)))
    return get_document(doc_id, offset=ordinal, limit=2 * max(0, int(radius)) + 1)


def fetch_image_chunk(chunk_id: str) -> dict:
    """Download an image chunk's bytes to a local file and return the path —
    hand this path to a vision-capable model to actually *look* at the image.
    GET /api/media/{chunk_id}."""
    data = _request("GET", f"/api/media/{urllib.parse.quote(chunk_id)}")
    if not isinstance(data, bytes):
        return {"error": "chunk has no image bytes (text chunks have no media)"}
    path = os.path.join(tempfile.gettempdir(), f"ls-chunk-{chunk_id[:24]}.png")
    with open(path, "wb") as f:
        f.write(data)
    return {"path": path, "chunk_id": chunk_id}


def corpus_stats() -> dict:
    """Corpus health/size; `corpus_version` bumps whenever ingest changes
    anything — key any agent-side cache on it."""
    return _request("GET", "/api/stats")


def is_healthy() -> bool:
    """Liveness probe — `GET /health`, which is public and needs no API key
    (unlike `corpus_stats`, which is authenticated)."""
    try:
        return _request("GET", "/health").get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# tool schemas (OpenAI/Anthropic function-calling shape)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the user's local document corpus (PDFs, OCR'd scans, "
                "images, text files). Returns ranked text chunks with their "
                "source document. Call this before answering questions that "
                "might be answered by the user's own files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "mode": {
                        "type": "string",
                        "enum": list(VALID_MODES),
                        "description": (
                            "hybrid (default): semantic+keyword fused; bm25: "
                            "exact keywords/identifiers/quotes; vector: "
                            "paraphrases and meaning-based queries; image: "
                            "text query, image hits only (needs the image "
                            "embedder on)"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Hits to return (1-100, default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_image",
            "description": (
                "Find indexed images visually similar to a query image "
                "(local file path or URL). Use for 'find images like this'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Local path or http(s) URL of the query image"},
                    "limit": {
                        "type": "integer",
                        "description": "Hits to return (1-100, default 5)",
                    },
                },
                "required": ["image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List indexed documents (id, title, source path), optionally "
                "filtered by file name / title / folder substring. Use to "
                "discover what the corpus contains; page via the returned `next`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "offset": {"type": "integer", "description": "Skip N documents (default 0)"},
                    "limit": {"type": "integer", "description": "Docs per page (default 100)"},
                    "query": {
                        "type": "string",
                        "description": (
                            "Case-insensitive substring to match against the "
                            "title and source path — use for 'find the file "
                            "named X' or 'docs under folder Y'"
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": (
                "Fetch one document with a page of its text chunks. Use when a "
                "search hit points at the right document but you need more of it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document id from a search hit or list_documents"},
                    "offset": {"type": "integer", "description": "First chunk ordinal to return (default 0)"},
                    "limit": {"type": "integer", "description": "Chunks to return (default 20)"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_context_around",
            "description": (
                "Read the chunks surrounding a specific search hit (its "
                "ordinal) so the passage has its surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document id of the hit"},
                    "ordinal": {"type": "integer", "description": "Chunk ordinal of the hit"},
                    "radius": {"type": "integer", "description": "Chunks of context on each side (default 2)"},
                },
                "required": ["doc_id", "ordinal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_image_chunk",
            "description": (
                "Download an image hit's file to a local path so a vision "
                "model can view it. Only image chunks have bytes."
            ),
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string", "description": "chunk_id of an image search hit"}},
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "corpus_stats",
            "description": "Corpus size, embedder info, and corpus_version.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_HANDLERS = {
    "search_documents": search_documents,
    "search_image": search_image,
    "list_documents": list_documents,
    "get_document": get_document,
    "get_context_around": get_context_around,
    "fetch_image_chunk": fetch_image_chunk,
    "corpus_stats": corpus_stats,
}


def dispatch(name: str, **args) -> dict:
    """Execute a tool call by name — plug this into your tool-call loop."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return handler(**args)
    except Exception as e:  # surface failures to the model, not the traceback
        return {"error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    name, *args = sys.argv[1:]
    if name == "search" and args:
        result = dispatch("search_documents", query=args[0])
    elif name == "image" and args:
        result = dispatch("search_image", image=args[0])
    elif name == "doc" and args:
        result = dispatch("get_document", doc_id=args[0])
    elif name == "docs":
        result = dispatch("list_documents", query=args[0] if args else None)
    elif name == "stats":
        result = dispatch("corpus_stats")
    else:
        print(
            f"usage: {sys.argv[0]} "
            "search <q> | image <path> | doc <id> | docs [query] | stats"
        )
        sys.exit(2)
    print(json.dumps(result, indent=2, ensure_ascii=False))
