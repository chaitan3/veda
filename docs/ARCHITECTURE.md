# Architecture

How `local-search` is put together: the pipeline, the data model, the two
possible vector layouts, and the retrieval channels. For using it, see
[../README.md](../README.md); for measuring it, [EVAL.md](EVAL.md); for
diagnosing it, [DEBUGGING.md](DEBUGGING.md).

## Pipeline

```
source folder ──ingest CLI──▶ docling convert → chunk → embed → SQLite (docs/chunks/FTS5/vec0)
        ▲                                                             ▲
        └── serve can schedule the same ingest (LS_INGEST_INTERVAL)   │
                                             serve ──▶ /api/search (dense + bm25 → RRF)
```

- **Ingest is a CLI, the service is read-only.** `local-search ingest` walks
  the source roots, converts each file, chunks it, embeds the chunks and commits
  per document. `serve` only reads — unless you set `LS_INGEST_INTERVAL`, in
  which case it runs the same ingest in one background task (one run at a time,
  catch-up at startup, then every interval).
- **Everything is content-addressed.** A file's SHA-256 is its identity: an
  unchanged file keeps its document id, its stage fingerprints and (for images)
  its vector, so a re-ingest is cheap and idempotent.
- **Every successful ingest bumps `corpus_version`** — the seam for
  cross-process cache invalidation (it is returned by every search response and
  by `/api/stats`).

## Data model

| table | holds |
|---|---|
| `documents` | one row per asset: `id` (UUID), `source_uri`, `title` (first heading, or file stem), `hash`, `metadata` (JSON: source, metrics, converted Markdown, folder policy), `created_at` |
| `chunks` | one row per fragment: `id` (UUID for text, content hash for images), `doc_id`, `ordinal`, `text`, `token_count`, `kind` (`text` \| `image`), `metadata` (headings, docling item refs, mime) |
| `chunks_fts` | FTS5 index over text chunks (BM25) |
| `chunk_vectors` | `sqlite-vec` `vec0`, cosine distance — text vectors, plus image vectors in the unified layout |
| `image_vectors` | `vec0` — image vectors in the two-space layout only |
| `document_stages` | per document, per stage (`extract`, `text_embed`, `image_embed`): the fingerprint that produced it |
| `meta` | `corpus_version`, the configured dimensions and the active layout marker |

A **document** is an asset; a **chunk** is a retrievable fragment. Text chunks
come from the converter; an image chunk is a standalone image file
(content-addressed, so its vector survives a text-only reprocess). PDF *pages*
are not image-embedded (no figure extraction yet).

## Two vector layouts

The store picks the layout from the configured providers
(`ImageSpace::for_config`): two providers that name the same space *and* the
same dimension unify, anything else stays split.

| | two-space (default) | unified (`LS_*_PROVIDER=eg2`) |
|---|---|---|
| text | bge-small, 384-d → `chunk_vectors` | EG2 text, 768-d |
| images | SigLIP2, 768-d → `image_vectors` | EG2 vision, 768-d → `chunk_vectors` |
| a query is embedded | twice (text model; image model's text tower for the image channel) | once (EG2 text graph) |
| `mode=vector`/`hybrid` | text chunks; images only via the image channel | text **and** images |
| `mode=image` | a second index | a kind filter over the one index |
| `/api/stats` | `image_space: split` | `image_space: unified` |

Changing layout (or a dimension) reconciles only what is affected: image
vectors are dropped or removed from the shared table and the `image_embed` stage
is forgotten; **text vectors survive** an image-side change. See
[Upgrades](#upgrades-and-the-database).

## Conversion

`docling.rs` does the heavy lifting, locally by default:

- PDF: text layer + layout analysis + TableFormer tables + optional
  CodeFormulaV2 code/formula enrichment, with OCR for scans.
- DOCX/XLSX/PPTX/HTML: parsed into the same document tree.
- `.md`/`.txt`: passthrough (no ML, no tree) — chunked as plain Markdown.
- Standalone images: OCR text (unless the folder is image-only) plus a vision
  embedding.
- Optional remote **VLM** pipeline (`LS_EXTRACT_PIPELINE=vlm`) sends rendered
  pages to an OpenAI-compatible endpoint instead.

`DOCLING_*` / `DOCLING_RS_*` variables are read by docling directly and are
never wrapped; only RAG-layer settings use `LS_*`.

## Chunking

| `LS_CHUNKER` | what it walks | `LS_CHUNK_SIZE` counts |
|---|---|---|
| `hybrid` (default) | the docling tree, splitting/merging against a token budget | chunk-tokenizer tokens |
| `hierarchical` | the tree's structure | nothing — **no size budget** |
| `window` | plain Markdown (also the fallback when there is no tree) | **words** |

Because the budget is measured with the *chunk* tokenizer and the embedder
counts its own, the two can disagree (EG2's 262k SentencePiece vocab produces
~0.9× the chunk tokenizer's count on prose, more on digit-heavy text). Anything
past `LS_EMBED_MAX_TOKENS` is truncated with a warning; see
[DEBUGGING.md](DEBUGGING.md#chunk-size-and-truncation).

## Retrieval

One query, up to four ranked lists, fused with Reciprocal Rank Fusion:

| channel | source | active |
|---|---|---|
| dense | `vec0` cosine KNN over the query embedding | always |
| keyword | FTS5 / BM25 over chunk text | always |
| image | the image model's text tower against image chunks | two-space layout only (in the unified layout the dense list already contains image chunks) |
| name | chunks of documents whose **title or source path** contains the query (≥3 chars) | `hybrid` |

`mode` selects which of these run: `vector` (dense), `bm25` (keyword), `hybrid`
(all applicable, RRF-fused, default), `image` (text query → image chunks only).
`POST /api/search/image` embeds an uploaded image instead of text.

Scores are comparable only within a mode: `vector`/`image` report cosine
similarity, `bm25` a negated BM25 score, `hybrid` an RRF score (rank-based, so
its magnitude is not meaningful).

## Stage-scoped ingest

Each document records a fingerprint per stage, and `ingest` re-runs only the
stages whose fingerprint changed (or that are missing):

- `extract` — converter settings, chunker, chunk size/overlap, docling env
  vars, and the *code path* (conversion + chunking are coupled).
- `text_embed` — provider, model, dimension, prefixes, truncation cap.
- `image_embed` — provider, model, dimension, vision budget.

`--stages extract,text_embed,image_embed` narrows the run; `--force` forces just
the targeted stages. Every ingest log line says why extract ran
(`extract=fingerprint-missing|fingerprint-changed|forced|unchanged`), and skips
report their chunk count.

**Folder policy.** Folders in `LS_IMAGE_ONLY_PATHS` index their image files
through the image channel only — no conversion, no OCR, no text chunks — and
the document records `"folder_policy": "image_only"`. The policy is re-derived
from the path on every run, so adding a folder to the list *demotes* its
documents (text chunks and stages dropped, image vectors kept) and removing it
*promotes* them (the next plain ingest backfills conversion and text).

## API surface

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness only, public |
| GET/POST | `/api/search` | `query`, `mode`, `limit` (1–100, default 5) |
| POST | `/api/search/image` | `image_base64`, `limit` |
| GET | `/api/stats` | counts, embedders, `image_space`, fingerprints, `corpus_version`, `truncated_sequences` |
| GET | `/api/documents` | paged; `query` = case-insensitive title/path substring |
| GET | `/api/documents/{id}` | one document + a pageable chunk slice (the expansion call) |
| GET | `/api/media/{chunk_id}` | image bytes, resolved from the source file (nothing is stored) |

Hits are self-contained: `chunk` (with `ordinal`), `score`, a compact
`document` reference (`id`, `title`, `source_uri`, `created_at`) and, for image
chunks, `media_url`. Unknown or stale request options are rejected (400 for
query parameters, 422 for JSON bodies) rather than ignored.

Auth is disabled unless `LS_API_KEYS` is set; then `X-Api-Key`,
`Authorization: Bearer` or `?api_key=` are accepted.

## Design decisions and limits

- **Single user, single writer.** SQLite in WAL mode: `serve` reads while an
  ingest writes, but two ingests at once are not supported (a second writer
  waits on the lock).
- **No folder watcher.** Ingest is explicit (`ingest`), scheduled
  (`LS_INGEST_INTERVAL`) or external (cron/timer).
- **Static API keys only** — no per-user keys or rotation.
- **PDF figure extraction is not implemented**; images are indexed only as
  standalone files. No image captioning, no reranker/Multi-Query/HyDE.
- **`hash` embedders are for smoke tests** — deterministic but not semantic.
- **The model set is not baked into the image**; `download-models` fetches it
  into a volume (see the README's Docker section).
