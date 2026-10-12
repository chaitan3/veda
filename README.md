# local-search

A local, single-user **text + image search service** over a store of documents
and images. Ingestion runs as a CLI tool; a read-only HTTP service answers
search queries. Results fuse keyword (SQLite FTS5 / BM25) and dense vector
search with Reciprocal Rank Fusion.

The design mirrors [docling-rag](https://github.com/docling-project/docling.rs)
closely — same module layout, same `Embedder` / `VectorStore` / `Retriever`
seams — while the data model is shaped for the image channel from day one.

## Status

**Working:**

- **Extraction** via docling.rs: PDF (text layer + layout + TableFormer tables
  + CodeFormulaV2 code/formula enrichment), DOCX, and PNG/JPEG/… via **OCR**,
  using the local ONNX pipeline by default or the optional remote **VLM**
  pipeline.
- **Execution provider**: CUDA when usable, with an automatic CPU fallback that
  is reported in the log (see below).
- **Embeddings**: in-process ONNX sentence encoder (default, bge-small-en-v1.5),
  an OpenAI-compatible llama.cpp endpoint, or a deterministic `hash` embedder.
  Standalone images additionally get a **SigLIP2** embedding in a shared
  image+text space.
- **Store**: SQLite (documents + chunks + FTS5 + two `vec0` spaces), WAL,
  corpus versioning, per-document stage fingerprints.
- **Retrieval**: text vector (`vec0` cosine KNN), BM25 (FTS5), Hybrid via RRF
  with an image channel; query-by-image.
- **Chunking**: `hybrid` (default) and `hierarchical` via docling's
  structure-aware chunkers, plus `window` (and the plain-text fallback).
- **CLI**: `init-db`, `ingest` (`--sync`, `--force`, `--stages`), `query`,
  `stats`, `prune`, `download-models`, `serve`.
- **Service**: read-only `/api/search`, `/api/search/image`, `/api/stats`,
  `/api/documents`, `/api/documents/{id}`, `/api/media/{chunk_id}`, `/health`;
  optional API-key auth.

**Not yet:** PDF figure extraction, image captioning, Multi-Query/HyDE.

## Quick start

```bash
cp .env.example .env            # optional; every key has a default

# 1. fetch the local model set (docling assets + bge-small + SigLIP2, ~2 GB;
#    add EmbeddingGemma 2 with LS_EMBED_PROVIDER=eg2 / LS_IMAGE_EMBED_PROVIDER=eg2
#    — `download-models` fetches those too)
cargo run --release -- download-models

# 2. create the schema and ingest a folder
cargo run --release -- init-db
cargo run --release -- ingest ./docs ./more-docs
cargo run --release -- ingest ./docs ./more-docs --sync   # also drop deleted files
cargo run --release -- ingest --stages text_embed         # re-embed text only
cargo run --release -- prune                  # drop docs with no chunks

# 3. query
cargo run --release -- query "mountain lake" --mode hybrid --limit 5

# 3b. measure retrieval (gold set mined from the corpus; see docs/EVAL.md)
cargo run --release -- eval-gen --out eval/queries.jsonl
cargo run --release -- eval --queries eval/queries.jsonl --k 1,5,10

# 4. serve the read-only API
cargo run --release -- serve
# ...optionally re-ingesting the source folders on a schedule (catch-up run
# at startup, then every interval; LS_INGEST_SYNC=true also prunes removed
# files):
# LS_INGEST_INTERVAL=24h cargo run --release -- serve
curl 'localhost:8080/api/search?query=mountain+lake&limit=5'
```

Default end-to-end is fully local/ONNX: docling's ONNX extraction plus the
in-process bge-small embedder. No network or extra services needed after
`download-models`.

### Docker

The image compiles the binary inside NVIDIA's CUDA 13 + cuDNN 9 image (so the
glibc matches) and downloads the models into a volume on first run. GPU runs
need [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/).

```bash
docker build -t local-search .

# ingest (one-shot; shares the data + model volumes with the server)
docker run --rm --gpus all \
  -v local-search-data:/data -v local-search-models:/models \
  -v /host/documents:/documents:ro local-search ingest

# serve
docker run -d --name local-search --gpus all -p 8080:8080 \
  -v local-search-data:/data -v local-search-models:/models \
  -v /host/documents:/documents:ro local-search serve
```

Volumes: `/data` (SQLite DB), `/models` (downloaded model tree), `/documents`
(ingest source), `/config` (optional `.env`). Drop `--gpus all` for a CPU-only
host. Build with `--build-arg PUID=$(id -u) --build-arg PGID=$(id -g)` to match
bind-mounted host directories. See `docker/.env.example` and
[`docs/PRODUCTION.md`](docs/PRODUCTION.md).


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
  fragments. A chunk is `text` or `image`; an image chunk is a standalone image
  file (content-addressed), served from its source on demand.
- **Two vector spaces, kept apart.** Text and image embeddings have different
  dimensionality and live in two `vec0` tables, so a text query reaches images
  through SigLIP's shared space without mixing the vectors.
- **Stage-scoped ingest.** Each document records a fingerprint per stage
  (`extract` / `text_embed` / `image_embed`); `ingest` re-runs only the stages
  whose configuration or models changed. `--stages` targets specific stages and
  `--force` forces just those.
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
| `LS_EXTRACT_CODE_FORMULA` | `true` on CPU / `false` on CUDA | CodeFormulaV2 code/formula enrichment (skipped if models absent) |
| `LS_EMBED_PROVIDER` | `onnx` | `onnx` \| `llamacpp` \| `hash` |
| `LS_IMAGE_EMBED_PROVIDER` | `siglip` | `siglip` \| `off` \| `hash` — image channel (SigLIP2 shared space) |
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
| GET | `/health` | liveness only (`{"status":"ok"}`), public — corpus details are at `/api/stats` |
| GET | `/api/search?query=&mode=&limit=` | search (`vector` \| `bm25` \| `hybrid` \| `image`) |
| POST | `/api/search` | `{"query","mode","limit"}` |
| GET | `/api/stats` | counts, embedder, extraction pipeline, corpus version |
| GET | `/api/documents?offset=&limit=&query=` | stored documents (paged; `query` = case-insensitive title/path substring) |
| GET | `/api/documents/{id}?offset=&limit=` | one document + a bounded, pageable chunk slice |
| POST | `/api/search/image` | query-by-image: `{"image_base64","limit"}` (needs `LS_IMAGE_EMBED_PROVIDER`) |
| GET | `/api/media/{chunk_id}` | bytes of an image chunk (from its source document) |

Each search hit carries its `chunk` (including the `ordinal`), its `score`, and
a hydrated `document` reference (`id`, `title`, `source_uri`, `created_at`), so
results are self-contained and citable in one call. The reference is
deliberately compact: `metadata` — which can embed the full converted text — and
the internal content hash are never sent.

Unknown or stale request options are **rejected** (`400` for query parameters,
`422` for JSON bodies) rather than ignored, so a typo or a renamed option is
visible instead of silently changing behaviour.

`/api/documents/{id}` is the **single expansion call**: with no params it returns
the first 20 chunks, and callers page with `offset`/`next` (capped at 100). An
offset past the end returns an empty `chunks` list, not an error — so a caller
can always pass `offset = ordinal - radius` for context around a hit.

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
database reset: each document records a per-stage fingerprint, so `ingest`
re-runs only the stages whose configuration or models changed.

## Notes

- **Unified space.** With `LS_EMBED_PROVIDER=eg2` **and**
  `LS_IMAGE_EMBED_PROVIDER=eg2`, text chunks and image chunks are embedded by
  one model into one 768-d space and stored in a **single** vector index: a
  query is embedded once and matches both, `mode=vector`/`hybrid` return text
  *and* images, and `mode=image` is a kind filter rather than a second index.
  Any other provider pair keeps the two-space layout (below). `/api/stats`
  reports which layout is active (`image_space`).
- **Image search** (SigLIP2, images only for now): **on by default**
  (`LS_IMAGE_EMBED_PROVIDER=siglip`; set `off` to disable). Standalone
  PNG/JPG/… get an image chunk in a shared image+text space, so `mode=hybrid`
  text queries also match images, `mode=image` returns image hits only, and
  `POST /api/search/image` does query-by-image. Image chunks carry a
  `media_url` served by `GET /api/media/{chunk_id}` (from the source file;
  nothing is stored). PDF figure extraction is not implemented yet.
- **Code/formula enrichment** (CodeFormulaV2): on by default on **CPU**, off
  when the provider resolves to **CUDA** (`LS_EXTRACT_CODE_FORMULA`). Requires
  the models fetched by `download-models`; if they're missing the pass is
  skipped with a warning. Set `LS_EXTRACT_CODE_FORMULA=true` to force it on.
- **Runtime defaults** (all overridable): `DOCLING_RS_OCR_ORIENTATION=off`
  (the orientation pass re-runs the recognizer over rotated hypotheses and is a
  large share of OCR time), and on CUDA `DOCLING_RS_PDF_WORKERS=1` (parallel
  page workers each build their own CUDA sessions and exhaust GPU memory). Set
  either explicitly to opt back in. A vendored `docling-onnx` patch additionally
  applies memory-friendly CUDA defaults (HEURISTIC conv search, capped
  workspace, `SameAsRequested` arenas) — see `vendor/docling-onnx` and
  `docs/PRODUCTION.md`.
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
