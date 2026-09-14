# local-search — production notes

Deployment notes: a single CLI/service binary over a SQLite database. Ingest
runs as a CLI; the service only reads.

## Package

```bash
cargo build --release
scripts/package.sh dist/local-search          # binary + libs + models + templates
scripts/package.sh dist/cpu --no-models       # fetch models on the host instead
```

The packaged directory is the service's **working directory** — docling resolves
`.models/` and `.pdfium/` relative to the CWD.

| Artifact | Required | Notes |
|---|---|---|
| `local-search` | yes | the product binary |
| `libonnxruntime_providers_*.so` | GPU only | next to the binary; CPU runs fine without them (logs a fallback) |
| `.models/` | yes | ~2 GB: docling assets, ONNX text embedder, SigLIP2 towers, CodeFormulaV2 |
| `.pdfium/lib/libpdfium.so` | yes (PDF) | ~7.5 MB |
| SQLite DB (`*.db`, `-wal`, `-shm`) | yes | all state |
| CUDA runtime + cuDNN 9 | GPU only | must match the ONNX Runtime build |

For a GPU-less host, build without CUDA:

```bash
cargo build --release --no-default-features --features docling-extract,onnx-embed
```

## Models

```bash
./local-search download-models              # docling + bge-small + SigLIP2 + CodeFormulaV2
./local-search download-models --no-embed   # skip the 133 MB text embedder
./local-search download-models --no-image   # skip the ~700 MB SigLIP2 towers
./local-search download-models --no-enrich  # skip the ~0.8 GB CodeFormulaV2 models
```

Idempotent. Mirrors: `DOCLING_RS_MODELS_URL`, `LS_ONNX_MODELS_URL`, `LS_IMAGE_MODELS_URL`.

## Configuration

Provide via the environment (systemd `EnvironmentFile` or container env).
`dotenvy` also loads `.env` from the working directory. See `.env.example`.

| Variable | Recommended | Notes |
|---|---|---|
| `LS_DB_PATH` | `/var/lib/local-search/app.db` | durable storage |
| `LS_SOURCE_PATH` | `/srv/documents` | one or more folders, path-separated |
| `LS_ADDR` | `127.0.0.1:8080` | keep loopback; proxy if public |
| `LS_API_KEYS` | a random key | required if exposed; empty disables auth |
| `LS_CHUNKER` | `hybrid` | `hybrid` \| `hierarchical` \| `window` |
| `LS_EMBED_PROVIDER` | `onnx` | `onnx` \| `llamacpp` \| `hash` |
| `LS_EMBED_DIM` | `384` | must match the embedder; sizes the `vec0` index |
| `LS_EXTRACT_PIPELINE` | `onnx` | `vlm` requires `DOCLING_RS_VLM_*` |
| `LS_RRF_K` | `60` | fusion constant |

Docling passthrough (names unchanged): `DOCLING_RS_OCR_LANG`, `DOCLING_RS_EP`
(`auto` by default), `DOCLING_RS_FP32`, `DOCLING_CHUNK_TOKENIZER`,
`DOCLING_RS_MODELS_URL`, `DOCLING_RS_VLM_*`.

## Ingest

```bash
./local-search init-db
./local-search ingest /srv/documents /srv/archive   # one or more roots
./local-search ingest                 # or use LS_SOURCE_PATH (a path list)
./local-search ingest --sync          # also delete removed files
./local-search ingest --force         # reprocess everything
./local-search prune                  # drop documents with no chunks
```

- **Multiple roots:** pass several paths, or set `LS_SOURCE_PATH` to a
  path-separated list (`:` on Linux). A root that does not exist is an error,
  so a missing mount never silently shrinks the corpus.
- **`--sync` is scoped to the roots given on that command line** — ingesting one
  root never deletes another root's documents.
- **Single writer:** never run two ingests at once. WAL lets `serve` read
  concurrently.
- Content-hash dedup skips unchanged files; exit code **2** if any file failed.
- Schedule it (systemd timer / cron); there is no folder watcher.

## Docker

The image is a two-stage build: `local-search` is compiled inside
`nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` so it links glibc 2.39, matching the
Ubuntu 24.04 runtime image (`nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04`).
Do **not** copy a host-built binary into the image: one built on a rolling
distro requires `GLIBC_2.43` (re-versioned libm functions) and will not load.

```bash
docker build -t local-search .

# one-off ingest (same volumes as the server)
docker run --rm --gpus all \
  -v local-search-data:/data -v local-search-models:/models \
  -v /srv/documents:/documents:ro local-search ingest

# server
docker run -d --name local-search --restart unless-stopped --gpus all \
  -p 8080:8080 \
  -v local-search-data:/data -v local-search-models:/models \
  -v /srv/documents:/documents:ro \
  local-search serve
```

| Mount | Holds | Notes |
|---|---|---|
| `/data` | SQLite DB | named volume; the only mutable state |
| `/models` | docling assets + embedder | downloaded on first run (~650 MB); must be writable |
| `/documents` | ingest source | mount read-only |
| `/config` | optional `.env` | Docker env vars take precedence |

- **Models** are not baked in; the entrypoint runs `download-models` on every
  start (idempotent — files already present are skipped without network, so it
  also pulls models added by newer versions). The first run pulls ~2 GB. Set
  `LS_SKIP_MODEL_DOWNLOAD=1` for a pre-provisioned volume, a remote-VLM
  (`LS_EXTRACT_PIPELINE=vlm`) or `LS_EMBED_PROVIDER=llamacpp` setup.
- **Non-root**: the entrypoint runs as root only to prepare volumes, then drops
  to `PUID:PGID` (default `1000:1000`). Match a bind-mounted host directory with
  `docker build --build-arg PUID=$(id -u) --build-arg PGID=$(id -g) .`
- **GPU**: `--gpus all` requires nvidia-container-toolkit. Without it, drop the
  flag and the ONNX runtime falls back to CPU (logged).
- **Single writer**: SQLite is single-writer. Run the ingest container to
  completion before/around serving; do not run two ingests at once.
- Health: the image healthcheck polls `GET /health` on `${LS_ADDR}`.

## Service

Install `deploy/local-search.service`, set `LS_DB_PATH` to a path under
`ReadWritePaths`, then `systemctl enable --now local-search`. Health:
`GET /health` (public), `GET /api/stats` (auth). Logs to stderr/journald
(`RUST_LOG=info` includes the execution provider, per-file ingest lines, and a
per-type timing summary).

## Backups

All state is the database:

```bash
sqlite3 /var/lib/local-search/app.db ".backup '/backup/app-$(date +%F).db'"
```

## GPU

The log line `CUDA execution provider registered; GPU inference enabled`
confirms GPU use. If `libcudnn` is missing, the provider is pinned to CPU and
the fallback is logged. Force either side with `DOCLING_RS_EP=cuda|cpu`.

The binary applies three defaults at startup unless you set them:
`DOCLING_RS_OCR_ORIENTATION=off` (the orientation pass is a large share of OCR
time), `DOCLING_RS_PDF_WORKERS=1` when the provider resolves to CUDA (parallel
page workers exhaust GPU memory and fail with `cuBLAS` allocation errors), and
`LS_EXTRACT_CODE_FORMULA=false` on CUDA (CodeFormulaV2's fp32 models do not fit
alongside the rest on a modest card — it stays on for CPU). Override any of them
to change the behaviour; raise `PDF_WORKERS` only if the GPU has headroom.

On top of that, a **vendored `docling-onnx` patch** (see `vendor/docling-onnx`,
wired via `[patch.crates-io]`) registers CUDA with memory-friendly defaults:
HEURISTIC cuDNN conv search with a 32 MB workspace cap and
`SameAsRequested` arena growth, instead of ORT's EXHAUSTIVE search with an
unbounded workspace and power-of-two arenas. That is what keeps a 15-page,
OCR-heavy PDF stable on an 11 GB card (peak ~8.1 GB instead of ~10.7 GB and
intermittent OOM). Override with `DOCLING_RS_CUDA_CONV_ALGO`,
`DOCLING_RS_CUDA_CONV_MAX_WORKSPACE`, `DOCLING_RS_CUDA_ARENA_EXTEND`, and
`DOCLING_RS_CUDA_MEM_LIMIT_MB`.

## Upgrades and the database

There is **no need to rebuild the database from scratch** for a new pipeline:

- Schema changes are additive (`CREATE TABLE IF NOT EXISTS`), applied
  automatically on startup.
- Each document records a fingerprint per ingest stage (`extract`,
  `text_embed`, `image_embed`) in `document_stages`; the current fingerprints are
  shown by `local-search stats` and `/api/stats`. `ingest` re-runs **only the
  stages whose fingerprint changed**, per document. So switching extraction
  (e.g. onnx → vlm) re-extracts text but leaves image embeddings untouched, and
  changing the text embedder re-embeds text only.
- `ingest --stages extract|text_embed|image_embed` targets specific stages;
  `ingest --force` forces just the targeted ones.

Changing an embedding **dimension** is structural: the affected `vec0` table is
dropped and recreated, and that stage re-runs for every document. The text and
image spaces are independent, so a text-dimension change does not disturb image
vectors (or vice versa).

Models are identified via `<models_dir>/models.lock` (sha256), written by
`download-models`, with a `path:size` fallback — so swapping a model file is
detected as a change.

## Not yet

No PDF figure extraction, no image captioning, no metrics endpoint, no
graceful-shutdown hook, no folder watcher, static API keys only, no
reranker/Multi-Query/HyDE, no eval harness.
