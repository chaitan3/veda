# local-search

A local, single-user **text + image search service** over a store of documents
and images. Ingestion runs as a CLI tool; a read-only HTTP service answers
search queries. Results fuse keyword (SQLite FTS5 / BM25) and dense vector
search with Reciprocal Rank Fusion.

The design mirrors [docling-rag](https://github.com/docling-project/docling.rs)
closely — same module layout, same `Embedder` / `VectorStore` / `Retriever`
seams — while the data model is shaped for the image channel from day one.

## Status

**Working (Phase 0):**

- **Extraction** via docling.rs: PDF (text layer + layout + TableFormer
  tables), DOCX, and PNG/JPEG/… via **OCR**, using the local ONNX pipeline by
  default or the optional remote **VLM** pipeline.
- **Execution provider**: CUDA when usable, with an automatic CPU fallback that
  is reported in the log (see below).
- **Embeddings**: in-process ONNX sentence encoder (default, bge-small-en-v1.5),
  an OpenAI-compatible llama.cpp endpoint, or a deterministic `hash` embedder.
- **Store**: SQLite (documents + chunks + FTS5), WAL, corpus versioning.
- **Retrieval**: vector (`sqlite-vec` `vec0` cosine KNN), BM25 (FTS5), and
  Hybrid via RRF.
- **Chunking**: `hybrid` (default) and `hierarchical` via docling's
  structure-aware chunkers, plus `window` (and the plain-text fallback).
- **CLI**: `init-db`, `ingest` (`--sync`), `query`, `stats`, `prune`,
  `download-models`, `serve`.
- **Service**: read-only `/api/search`, `/api/stats`, `/api/documents`,
  `/health`; optional API-key auth (off by default); media/embedding internals
  stripped from responses.

**Phase 1 (planned):** image captioning (VLM via llama.cpp), CLIP image
embeddings, the image retrieval channel, media-serving routes.

## Quick start

```bash
cp .env.example .env            # optional; every key has a default

# 1. fetch the local model set (docling assets + bge-small embedder, ~0.5 GB)
cargo run --release -- download-models

# 2. create the schema and ingest a folder
cargo run --release -- init-db
cargo run --release -- ingest ./docs ./more-docs
cargo run --release -- ingest ./docs ./more-docs --sync   # also drop deleted files
cargo run --release -- prune                  # drop docs with no chunks

# 3. query
cargo run --release -- query "mountain lake" --mode hybrid -k 5

# 4. serve the read-only API
cargo run --release -- serve
curl 'localhost:8080/api/search?q=mountain+lake&k=5'
```

Default end-to-end is fully local/ONNX: docling's ONNX extraction plus the
in-process bge-small embedder. No network or extra services needed after
`download-models`.

`ingest` logs each file with its extraction method and chunker, then prints a
per-type summary of files, chunks and elapsed time:

```text
INFO ingesting name="report.pdf" method=docling-onnx chunker=hybrid bytes=868
INFO ingested  name="report.pdf" chunks=1 secs=1.99

summary by type:
  type       files  chunks   skip   fail       time
  docx           1       1      0      0      0.10s
  md             1       1      0      0      0.01s
  pdf            1       1      0      0      1.99s
  png            1       1      0      0      1.33s
  total          4       4      0      0      3.43s
```

## Architecture

```
source folder ──ingest CLI──▶ docling convert → chunk → embed → SQLite (docs/chunks/FTS5)
                                                                      ▲
                                                    serve ──▶ /api/search (vector + bm25 → RRF)
```

- **Documents vs images.** One `Document` (asset) fans out to many `Chunk`
  fragments. A chunk carries a `kind` (`text` | `image`), an optional media
  reference and a locator (page/bbox). Phase 0 stores text; Phase 1 adds image
  fragments without a schema change.
- **Two vector spaces, kept apart.** Text and CLIP have different
  dimensionality, so Phase 1 adds a second vector table rather than mixing.
- **Out-of-process ingest.** The service never ingests, so it is
  side-effect-free. Every successful ingest bumps a `corpus_version`, the seam
  for cross-process cache invalidation.
- **Docling config passthrough.** `DOCLING_*` / `DOCLING_RS_*` are read by
  docling directly; only RAG-layer settings use `LS_*`.

## Configuration

See `.env.example` for the full list. Key RAG-layer settings:

| Variable | Default | Purpose |
|---|---|---|
| `LS_DB_PATH` | `data/local-search.db` | SQLite database |
| `LS_SOURCE_PATH` | `docs` | folder(s) the ingest CLI walks; path-separated |
| `LS_MODELS_DIR` | `.models` | model tree; fans out to docling's `DOCLING_*` vars |
| `LS_EXTRACT_PIPELINE` | `onnx` | `onnx` (local) or `vlm` (remote, llama.cpp) |
| `LS_EMBED_PROVIDER` | `onnx` | `onnx` \| `llamacpp` \| `hash` |
| `LS_EMBED_DIM` | `384` | must match the embedder; sizes the `vec0` index |
| `LS_ONNX_MODEL` | `.models/embed/bge-small-en-v1.5.onnx` | ONNX embedder graph |
| `LS_ONNX_TOKENIZER` | `.models/embed/tokenizer.json` | ONNX embedder tokenizer |
| `LS_LLAMACPP_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible embeddings |
| `LS_EMBED_MODEL` | — | model name for `llamacpp` |
| `LS_CHUNKER` | `hybrid` | `hybrid` \| `hierarchical` \| `window` |
| `LS_RRF_K` | `60` | RRF constant |
| `LS_ADDR` | `127.0.0.1:8080` | bind address |
| `LS_API_KEYS` | — | comma-separated keys; empty disables auth |

### Models

`download-models` fetches docling's native assets (pdfium + layout / PP-OCR /
TableFormer ONNX) from the docling.rs releases, and the ONNX embedder from
Hugging Face, into `.models/` and `.pdfium/`. Mirrors: `DOCLING_RS_MODELS_URL`
and `LS_ONNX_MODELS_URL`. Use `--no-embed` to skip the embedder.

Docling honours its own per-model overrides — `DOCLING_LAYOUT_ONNX`,
`DOCLING_OCR_REC_ONNX` / `DOCLING_OCR_DICT`, `DOCLING_TABLEFORMER_*`,
`DOCLING_RS_OCR_LANG`, `DOCLING_RS_EP`, `DOCLING_RS_FP32`,
`DOCLING_CHUNK_TOKENIZER` — all unchanged.

### Execution provider (CUDA)

The default build compiles the **CUDA** execution provider and uses docling's
`auto` selection, so ONNX inference prefers the GPU. It needs a CUDA runtime and
**cuDNN 9** that match the ONNX Runtime build `ort` downloads (on this machine:
CUDA 13 + cuDNN 9). If `libcudnn.so` is not loadable, the provider is pinned to
CPU before any inference and the fallback is logged:

```text
WARN CUDA is compiled in but the CUDA runtime (libcudnn) is not loadable —
     CUDA 12 + cuDNN 9 are required. Falling back to CPU for ONNX inference.
INFO ONNX execution provider (CPU fallback) resolved=Cpu
```

Setting `DOCLING_RS_EP=cpu` (or `cuda`) explicitly overrides the automatic
choice; an explicit value is always respected.

### VLM extraction

```bash
LS_EXTRACT_PIPELINE=vlm
DOCLING_RS_VLM_ENDPOINT=http://127.0.0.1:8080/v1
DOCLING_RS_VLM_MODEL=<your-vlm>
```

Docling renders each page and sends it to the OpenAI-compatible endpoint; the
returned markup is parsed back into a document. Pages that fail fail the
document (no silent drops).

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness + counts + corpus version |
| GET | `/api/search?q=&mode=&k=` | search (`vector` \| `bm25` \| `hybrid`) |
| POST | `/api/search` | `{"query","mode","top_k"}` |
| GET | `/api/stats` | counts, embedder, extraction pipeline, corpus version |
| GET | `/api/documents` | stored documents with metadata |

Auth is disabled unless `LS_API_KEYS` is set; then send `X-Api-Key`,
`Authorization: Bearer`, or `?api_key=`.

## Deployment

See [`docs/PRODUCTION.md`](docs/PRODUCTION.md) for the package contents, config,
systemd unit, backups and upgrade path. Package a release build with:

```bash
cargo build --release
scripts/package.sh dist/local-search
```

`deploy/local-search.service` is a systemd unit template. Upgrades need no
database reset: the DB records a `pipeline_version`, and `ingest` automatically
reprocesses the corpus when the binary's pipeline version differs.

## Notes

- `LS_EMBED_PROVIDER=hash` is deterministic but **not semantic** — for smoke
  tests only.
- BM25 scores are zero on very small corpora (FTS5's IDF collapses); ranking
  still works and the vector channel compensates.
- The vector index is a `sqlite-vec` `vec0` table (`distance_metric=cosine`),
  sized by `LS_EMBED_DIM`. It must match the embedder's output; a mismatch is
  reported at ingest/search. Changing embedder or dimension requires
  re-ingesting (opening an older database rebuilds the index from the stored
  embeddings automatically).
- The llama.cpp server must run in embedding mode for `/v1/embeddings`;
  query/document prefixes are applied client-side.
- `examples/rasterize.rs` and `examples/docling_dump.rs` are test helpers, not
  product surface.
