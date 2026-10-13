# local-search

A local, single-user **text + image search service** over your own documents and
images. Ingest is a CLI tool; a read-only HTTP API answers queries. Results fuse
keyword search (SQLite FTS5 / BM25) with dense vector search using Reciprocal
Rank Fusion, and standalone images are searchable by text or by example image.

Everything runs on your machine: the default pipeline is local ONNX (docling for
conversion, bge-small/SigLIP2 for embeddings, optionally EmbeddingGemma 2),
with no external services required.

- **Ingest** PDFs, Office files, HTML, Markdown, text and standalone images.
- **Search** with four modes — `hybrid` (default), `vector`, `bm25`, `image` —
  plus query-by-image and file-name matching.
- **Measure** retrieval quality on your own corpus (`eval` / `eval-gen`).
- **Serve** a small read-only JSON API with optional API keys and an optional
  scheduled re-ingest.

Where to go next: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (how it works),
[docs/EVAL.md](docs/EVAL.md) (measuring quality),
[docs/DEBUGGING.md](docs/DEBUGGING.md) (logs, performance, troubleshooting).

## Requirements

- **Rust** (edition 2024 toolchain) to build.
- **Disk**: ~2 GB for the default model set (docling assets + bge-small +
  SigLIP2); ~4 GB if you also fetch EmbeddingGemma 2.
- **GPU** (optional): a CUDA 13 + cuDNN 9 runtime for GPU inference. Without it
  the binary falls back to CPU automatically (slower — see
  [DEBUGGING.md](docs/DEBUGGING.md#performance)).
- Linux is the supported platform; PDF rasterization uses a bundled pdfium
  build fetched with the models.

## Build

```bash
cargo build --release
# GPU-less build (no CUDA provider):
cargo build --release --no-default-features --features docling-extract,onnx-embed
```

## Run

```bash
cp .env.example .env      # optional — every setting has a default

# 1. fetch the local model set into .models/ (idempotent; re-run after upgrades)
cargo run --release -- download-models

# 2. create the schema, then ingest one or more folders
cargo run --release -- init-db
cargo run --release -- ingest /path/to/documents

# 3. search from the CLI
cargo run --release -- query "boiler feedwater drum level" --mode hybrid --limit 5
cargo run --release -- query-image ./photos/pump.jpg --limit 5

# 4. serve the read-only API
cargo run --release -- serve
curl 'localhost:8080/api/search?query=boiler+feedwater&limit=5'
```

`ingest` prints a per-file line and a per-type summary:

```text
INFO ingesting name="report.pdf" method=docling-onnx chunker=hybrid
INFO ingested  name="report.pdf" chunks=12 secs=7.31 extract="fingerprint-missing"

summary by type:
  type       files  chunks   skip   fail       time
  pdf            1      12      0      0      7.31s
  total          1      12      0      0      7.31s
```

Re-running `ingest` is cheap: unchanged files are skipped by content hash, and
only the stages whose configuration changed re-run (see
[ARCHITECTURE.md](docs/ARCHITECTURE.md#stage-scoped-ingest)). The service and the
CLI can run at the same time — the database is in WAL mode and each document is
committed as it finishes, so queries see the corpus grow.

### Common commands and flags

| command | what it does |
|---|---|
| `ingest <paths…>` | ingest one or more folders (or use `LS_SOURCE_PATH`) |
| `ingest --sync` | also delete documents whose files disappeared — **scoped to the roots given on the command line** |
| `ingest --force` | re-run the targeted stages even if their fingerprint matches |
| `ingest --stages extract,text_embed,image_embed` | run only some stages (e.g. re-embed text after switching the text model) |
| `query "<text>" --mode hybrid\|vector\|bm25\|image --limit N` | search text |
| `query-image <file> --limit N` | search by an example image |
| `eval --queries gold.jsonl --modes hybrid,vector,bm25 --k 1,5,10` | score retrieval quality |
| `eval-gen --out eval/queries.jsonl` | mine a gold set from the corpus |
| `stats` | corpus counts, embedders, active vector layout, fingerprints |
| `prune` | drop documents left with no chunks (interrupted ingests) |
| `download-models [--no-embed\|--no-image\|--no-enrich\|--no-eg2]` | fetch the model set |
| `serve` | run the read-only API (see below for the optional scheduler) |

Ingest exits with code **2** if any file failed (the errors are in the log).

## Configuration

Settings are environment variables (all prefixed `LS_`, except docling's own
`DOCLING_*` / `DOCLING_RS_*`, which are passed through unchanged). `dotenvy`
also loads `.env` from the working directory; **real environment variables win
over the file**, which matters in containers (see the Docker section).

[`.env.example`](.env.example) documents every option with defaults — it is the
reference. The settings you are most likely to touch:

| variable | default | purpose |
|---|---|---|
| `LS_SOURCE_PATH` | `docs` | folders the ingest CLI walks (path-separated) |
| `LS_DB_PATH` | `data/local-search.db` | SQLite database (all state) |
| `LS_ADDR` | `127.0.0.1:8080` | API bind address |
| `LS_API_KEYS` | — | comma-separated keys; **empty disables authentication** |
| `LS_MODELS_DIR` | `.models` | where `download-models` puts the model tree |
| `LS_EMBED_PROVIDER` | `onnx` | text embeddings: `onnx` (bge-small, 384-d), `eg2` (EmbeddingGemma 2, 768-d, multilingual), `llamacpp`, `hash` |
| `LS_IMAGE_EMBED_PROVIDER` | `siglip` | image embeddings: `siglip` (SigLIP2), `eg2`, `off`, `hash` |
| `LS_IMAGE_ONLY_PATHS` | — | folders whose images are indexed **without** OCR/text chunks (photo dumps) |
| `LS_CHUNKER` / `LS_CHUNK_SIZE` | `hybrid` / `300` | how text is split (see below) |
| `LS_INGEST_INTERVAL` | — | `serve` re-ingests every interval (e.g. `24h`); unset = manual |
| `LS_INGEST_SYNC` | `false` | scheduled runs also prune documents whose files disappeared |

**Choosing an embedder.** The default pair (bge-small text + SigLIP2 images) is
the cheap, well-tested option. Setting **both** to `eg2` puts text and images in
one shared 768-d space and one vector index — a query is embedded once and
matches both, `mode=image` becomes a filter, and cross-modal quality is markedly
better; the cost is throughput (EmbeddingGemma 2 is ~4.6× slower for text and
~5–28× for images than the defaults) and ~1.8 GB of extra models. Measured
numbers: [DEBUGGING.md](docs/DEBUGGING.md#performance). Switching embedders only
re-embeds — it never re-extracts.

**Chunk size.** `LS_CHUNK_SIZE` means different things per chunker: `hybrid`
counts *tokens*, `window` counts *words*, `hierarchical` has no budget at all.
Anything longer than `LS_EMBED_MAX_TOKENS` (default 512) is truncated, so 300 is
a safe default for both; dense text (logs, numbers) needs less. Check your own
files with `cargo run --release --example chunk_audit -- file.pdf`, and set
`LS_CHUNK_STRICT=true` to make an over-budget configuration a startup error
instead of a warning.

## Docker

The image compiles the binary inside `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04`
so it links the same glibc as the `…-runtime-ubuntu24.04` base — do **not** copy
a host-built binary into it.

### Build

```bash
docker build -f docker/Dockerfile -t local-search .
# match a bind-mounted host directory's ownership:
docker build -f docker/Dockerfile --build-arg PUID=$(id -u) --build-arg PGID=$(id -g) -t local-search .
```

### Run

The wrapper script creates the mount folders and runs the service with the
standard layout (from the repository root, any working directory):

```bash
./docker/run-container.sh
```

It is equivalent to:

```bash
# server
docker run -d --name local-search --restart unless-stopped --gpus all \
  -p 8080:8080 \
  -v "$PWD/data:/data" -v "$PWD/models:/models" \
  -v "$PWD/documents:/documents:ro" -v "$PWD/photos:/photos:ro" \
  -e LS_SOURCE_PATH=/documents:/photos -e LS_IMAGE_ONLY_PATHS=/photos \
  -v "$PWD/config:/config" \
  local-search serve

# one-off ingest, sharing the same volumes
docker run --rm --gpus all \
  -v "$PWD/data:/data" -v "$PWD/models:/models" \
  -v "$PWD/documents:/documents:ro" local-search ingest
```

| mount | holds | notes |
|---|---|---|
| `/data` | SQLite database | the only mutable state; back this up |
| `/models` | model tree | downloaded on first run; must be writable |
| `/documents` | ingest source | mount read-only |
| `/photos` (or any folder) | image-only source | optional; see `LS_IMAGE_ONLY_PATHS` |
| `/config` | optional `.env` | container env vars take precedence over it |

Notes for container use:

- **Models**: the entrypoint runs `download-models` on every start (idempotent —
  present files are skipped, so upgrades fetch only what's new). Set
  `LS_SKIP_MODEL_DOWNLOAD=1` for a pre-provisioned volume or a
  remote-VLM/llama.cpp setup. It passes `--no-eg2` unless a provider is `eg2`,
  so the ~1.8 GB EmbeddingGemma 2 graphs are fetched only when used.
- **GPU**: `--gpus all` requires nvidia-container-toolkit. Without it the
  container has no GPU access and logs `CUDA execution provider unavailable …
  libcuda.so.1` — verify with
  `docker run --rm --gpus all local-search nvidia-smi`. Drop the flag on a
  GPU-less host (CPU inference works, just slower).
- **Non-root**: the entrypoint prepares volumes as root, then drops to
  `PUID:PGID` (default `1000:1000`).
- **Configuration**: copy [`docker/.env.example`](docker/.env.example) to the
  mounted config directory as `.env`. Values passed with `-e` win over the file.
- **Health**: the image healthcheck polls `GET /health`; `/api/stats` (behind
  auth) has the details.
- **Single writer**: SQLite allows one writer — don't run two ingests at once
  (a scheduled ingest inside `serve` counts as one).

## Deployment

```bash
cargo build --release
scripts/package.sh dist/local-search      # binary + native libs + models + templates
```

The packaged directory is the service's working directory (docling resolves
`.models/` and `.pdfium/` relative to the CWD). `scripts/local-search.service`
is a systemd unit template; set `LS_DB_PATH` under `ReadWritePaths` and
`systemctl enable --now local-search`. With `LS_INGEST_INTERVAL` set the service
also ingests on its own (one run at a time, catch-up at startup).

All state is the database, so a backup is one file:

```bash
sqlite3 /var/lib/local-search/app.db ".backup '/backup/app-$(date +%F).db'"
```

Upgrades need no database reset: schema changes are additive and per-stage
fingerprints mean only affected stages re-run.

## Status

Working end to end: ingestion (docling ONNX or remote VLM), text and image
embeddings (bge/SigLIP2 or EmbeddingGemma 2), hybrid retrieval with file-name
matching, the read-only API with optional auth and scheduled ingest, and the
evaluation harness. Not implemented: PDF figure extraction, image captioning,
rerankers/Multi-Query/HyDE, a folder watcher, and metrics/graceful-shutdown
hooks — see [ARCHITECTURE.md](docs/ARCHITECTURE.md#design-decisions-and-limits).
