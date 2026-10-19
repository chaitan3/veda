---
name: veda
description: Search the user's local document corpus (veda service) as an agent tool.
version: 1.0.0
metadata:
  hermes:
    tags: [rag, search, documents, local]
    category: productivity
    config:
      - key: veda.url
        description: Base URL of the veda HTTP service
        default: "http://127.0.0.1:8080"
        prompt: "veda service URL"
      - key: veda.api_key
        description: API key if the service has LS_API_KEYS set (leave empty otherwise)
        default: ""
        prompt: "veda API key (optional)"
---

# Veda RAG

A retrieval layer over the user's own documents — PDFs (including OCR'd
scans), standalone images, and text files — indexed with hybrid search
(semantic embeddings + SQLite FTS5/BM25, fused with reciprocal rank fusion).
Read-only: nothing indexed here can be modified through these tools.

## When to Use

- The question might be answered by the user's files — **always search before
  answering from model knowledge** when such a corpus is expected to contain
  the answer.
- "What do my documents say about…", "find the file where…", quoting/verifying
  something the user says they wrote.
- "Find images like this one" → image similarity search.
- Not for general web knowledge — this index only holds the user's own corpus
  (check its size first with corpus_stats if unsure it's non-empty).

## Procedure

1. **Check availability once per session** (skip if you already know it's up):
   call `is_healthy()` (public `GET /health`, no key needed), then
   `corpus_stats()` for the `documents`/`chunks` counts. If unreachable, tell
   the user the `serve` process isn't running; do not retry silently in a loop.
2. **Search** with `search_documents(query=…)`:
   - default to `mode="hybrid"` (semantic + keyword fused);
   - use `mode="bm25"` for exact identifiers, error codes, quoted phrases;
   - use `mode="vector"` for paraphrases ("what's the deadline for X" when the
     document says "must be submitted by…");
   - use `mode="image"` to find images with a text query (image hits only —
     the hybrid image channel without text results competing for slots);
   - start with `limit=5`; raise only if the first page misses.
3. **Expand when needed**: a hit is a ~paragraph slice. If it's plausibly part
   of a larger passage, call `get_context_around(doc_id, ordinal)`. If the
   whole document matters, call `get_document(doc_id)` and page with `next`.
4. **Images**: image hits include `media_url`. To actually *look* at one, call
   `fetch_image_chunk(chunk_id)` and open the returned path with vision
   capabilities. To search by an image the user provides, call
   `search_image(path_or_url)`.
5. **Find a file by name/title/folder**: `list_documents(query="…")` is the
   precise lookup — a case-insensitive substring over titles and source paths.
   `search_documents(query=…)` in `mode="hybrid"` also matches names (it adds
   the chunks of any document whose name contains the query), so a name-shaped
   query can just be searched; `bm25`/`vector` alone search content only.
   Never use `search_image` for text — it takes an image.
6. **Answer with citations**: cite the `source` (document title) for every
   factual claim drawn from a hit. If search returns nothing relevant, say so
   plainly — do not pad with model-prior knowledge and imply it came from the
   corpus.

Tool implementations live in `scripts/agent_rag_tools.py` next to this
SKILL.md — endpoint wrappers + `TOOL_SCHEMAS` function-calling definitions +
`dispatch(name, **args)` for a tool-call loop, plus a CLI for smoke-testing:
`python3 scripts/agent_rag_tools.py stats`.

## Pitfalls

- **Search is retrieval, not QA**: hits are evidence, not answers. Read the
  chunk text before asserting anything; the score is a relevance rank, not
  truth.
- **bm25-style verification**: after drafting an answer, one extra
  `search_documents` call with `mode="bm25"` on a distinctive phrase from the
  draft catches hallucinations cheaply — exact-string matching is exactly what
  embeddings are worst at.
- **Queries during an ingest are fine but slower** (shared GPU), and the
  corpus grows mid-ingest: results reflect the last committed document.
  `corpus_version` in every response changes when anything is added — key any
  cache on it.
- **Empty/garbage results usually mean**: the corpus wasn't ingested yet
  (stats will show 0 documents), or the query mode fights the query shape
  (quoted strings in vector mode, prose in bm25).
- **Don't fetch whole documents unless needed** — chunk text is sized for
  context windows; `limit` on `get_document` caps at 100.
- **Parameters are strict**: unknown or renamed options are rejected (400/422),
  never ignored — e.g. `get_document` takes only `offset`/`limit` (use
  `list_documents(query=…)` to find a document; it has no per-document filter).
  If a call errors with "unknown field", fix the parameter name rather than
  retrying — don't invent option names.

## Verification

- `python3 agent_rag_tools.py stats` → `"status": "ok"` and non-zero counts.
- `python3 agent_rag_tools.py search "<term you know is in the corpus>"`
  returns a hit whose `text` visibly contains the term — proves the end-to-end
  path (embed → store → hydrate) works, not just that the API responds.
