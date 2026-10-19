# Debugging and performance

What the logs mean, how to tell where time and memory go, and the knobs for
both. For the design behind it, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Reading the logs

`RUST_LOG=info` (the default) prints one line per file that actually does work,
plus a summary. Up-to-date files are silent (a `debug` line at most):

```text
INFO processing name="report.pdf" uri="…/report.pdf" bytes=8680 method=docling-onnx policy="default" chunker=hybrid
INFO ingested   name="report.pdf" chunks=12 secs=7.31 extract="fingerprint-missing"

summary by type:
  type       files  chunks   skip   fail       time
  pdf            1      12      0      0      7.31s
  total          1      12      0      0      7.31s
```

Useful fields:

| field | meaning |
|---|---|
| `method` | `docling-onnx`, `vlm`, or a passthrough label |
| `policy` | `image-only` for folders in `LS_IMAGE_ONLY_PATHS` |
| `extract=` | why extract ran: `fingerprint-missing`, `fingerprint-changed`, `retry-failed`, `forced`, or `unchanged` (didn't run) |
| `chunks` | chunks produced or re-embedded; `0` with a warning means the file converted to no text |

A skipped file is only logged at `debug` (`up to date; skipping`), so the
`info` stream shows exactly the documents that were processed.

Other lines worth knowing:

- `CUDA execution provider registered; GPU inference enabled` — GPU is live.
- `CUDA execution provider unavailable; … libcuda.so.1: cannot open shared
  object file` — **no GPU access**. In a container that means it was started
  without `--gpus all` (libcuda is the *host* driver) or
  nvidia-container-toolkit is missing; check with
  `docker run --rm --gpus all <image> nvidia-smi`. CPU inference is ~9–16×
  slower (numbers below).
- `truncating over-long chunk(s) for embedding … truncated=N` — see
  [Chunk size and truncation](#chunk-size-and-truncation).
- `image vector layout changed; image embeddings will be recomputed` — a
  provider/dimension change; text vectors are kept.
- `retrying previously failed file … reason="file changed"|"extractor changed"`
  — a remembered conversion failure is being re-attempted (see
  [Failed files](#failed-files)).
- `embedding model released (idle)` / `embedding model loaded` — the
  `LS_EMBED_IDLE_TIMEOUT` window (see [Idle models](#idle-models)).
- `scheduled ingest finished …` / `scheduled ingest failed; will retry next
  interval` — the `LS_INGEST_INTERVAL` child process ran.
- `scheduled ingest stopped (shutdown)` — `serve` is stopping and killed the
  child mid-run.
- `LS_API_KEYS is empty — API authentication is disabled` — expected unless you
  set keys.

## Diagnosing "why did this file re-process?"

`ingest` only re-runs stale stages, so a re-processed file tells you which
fingerprint moved:

| `extract=` value | cause |
|---|---|
| `fingerprint-missing` | first ingest, or a previous run was interrupted before it completed |
| `fingerprint-changed` | converter/chunker settings or a `DOCLING_*` variable differs from the recorded one — including running ingest with a *different environment* than last time (host vs container) |
| `retry-failed` | a remembered conversion failure is being retried after a file/extractor change |
| `forced` | `--force --stages extract` |
| *(absent / `unchanged`)* | extract didn't run |

Two cheaper gates sit in front of the fingerprint check, so most unchanged
files never reach it:

- A **size + mtime** match against the `source_files` cache means the content
  hash is trusted and the file is not read (or hashed) again. A mere `touch`
  with the same content still skips (the hash is unchanged).
- A **remembered failure** (`document_failures`) skips a file whose conversion
  already failed, without reading it (see [Failed files](#failed-files)).

Text re-embeds are reported the same way: switching the text embedder (or its
prefixes, dimension or truncation cap) re-embeds text only — no conversion. A
layout or dimension change drops the affected vectors and forgets that stage
(see ARCHITECTURE.md, *Upgrades*). `veda stats` and `/api/stats` show
the current fingerprints, so you can compare them across environments.

## Failed files

A file that fails to **convert** (a corrupt PDF, a pdfium error) is recorded in
`document_failures` with the error and the *extractor identity* that failed it.
The next run skips it without reading or converting it again — otherwise every
run would re-pay the full docling cost for the same bad file. The CLI reports
these as `known failure(s) skipped`; `/api/stats` has `failed_documents`.

It is retried when either changes:

- **the file** — size/mtime, then content hash;
- **the extractor identity** — the binary build (git sha + executable
  size/mtime), the docling model hashes (`models.lock`), the pdfium build hash,
  and the `DOCLING_*` environment. So rebuilding the container with a fixed
  pdfium or a better OCR model re-attempts every failed file, while a plain
  restart does not.

`ingest --force` (with the extract stage) retries them without any change;
`ingest --sync` forgets the failures of files that disappeared. A failed
conversion never replaces a previously good version: new content is only
inserted once conversion succeeds, and the old document is dropped only after
the new one is committed. `prune` removes any document left with no chunks.

To clear the whole memory (e.g. after changing the identity by hand):

```bash
sqlite3 data/veda.db "DELETE FROM document_failures;"
```

## Idle models

`serve` keeps the embedding models resident while queries arrive and releases
them `LS_EMBED_IDLE_TIMEOUT` (default `5m`) after the last embedding call. Text
and image share one window, so either kind of query refreshes it; `off` keeps
them for the process lifetime. In-flight queries hold a reference, so a model is
never released mid-query — the next call after eviction just reloads it (a
one-off latency cost). This is why a quiet `serve` can show much lower GPU/CPU
memory than during ingest; see [GPU and memory](#gpu-and-memory).

Scheduled ingest does **not** keep serve's models warm: it runs in a child
process, which loads its own docling + embedder models and releases them on
exit.

## Chunk size and truncation

A chunk longer than the embedder's window has its tail dropped (the head is
embedded, BM25 still indexes the whole text). Symptoms: the warning above, the
ingest summary's `note: N sequence(s) truncated`, or `truncated_sequences` in
`/api/stats`.

Measure before guessing — the chunker and the embedder count different
tokenizers:

```bash
# per-chunk token counts with the *configured* embedder's tokenizer
LS_EMBED_PROVIDER=eg2 cargo run --release --example chunk_audit -- file.pdf
#   chunks=32 max_bpe_tokens=289 over_512=0
# compare tokenizers explicitly (2nd argument):
… -- file.pdf .models/embed/tokenizer.json
```

Measured guidance (details in the README's configuration section):

| chunker | budget | EG2 tokens |
|---|---|---|
| `hybrid` | `LS_CHUNK_SIZE=300` | ~270–400 → fits the 512 window |
| `window` | 300 **words** | ~370 on prose; **1652** on a token-dense file (truncated) |

Fix it by lowering `LS_CHUNK_SIZE`, or raising `LS_EMBED_MAX_TOKENS` (EG2
supports 8k) — raise both together, since longer chunks dilute the embedding.
`LS_CHUNK_STRICT=true` makes an over-budget configuration a startup error
instead of a warning.

## GPU and memory

Where the memory goes depends on the process:

- **`ingest`** (and the scheduled-ingest child) loads docling plus the
  embedders; its footprint is the model weights plus ORT's arena high-water
  mark. It releases everything on exit.
- **`serve`** loads only the embedders (never docling), and releases them
  `LS_EMBED_IDLE_TIMEOUT` after the last query. A quiet server can therefore
  sit near zero, and a query after eviction pays a one-off reload. Use
  `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
  to attribute VRAM to the server vs a running ingest child.

The binary applies three defaults at startup unless you set them:

- `DOCLING_RS_OCR_ORIENTATION=off` — the orientation pass re-runs the recognizer
  over rotated hypotheses and is a large share of OCR time.
- `DOCLING_RS_PDF_WORKERS=1` on CUDA — parallel page workers each build their
  own CUDA sessions and exhaust GPU memory.
- `LS_EXTRACT_CODE_FORMULA=false` on CUDA — CodeFormulaV2's fp32 models do not
  fit alongside the rest on a modest card (it stays on for CPU).

Force an execution provider with `DOCLING_RS_EP=cuda|cpu`.

### The vendored docling patch

`vendor/docling-onnx` (wired through `[patch.crates-io]`) registers CUDA with
memory-friendly defaults: HEURISTIC cuDNN convolution search with a 32 MB
workspace cap and `SameAsRequested` arena growth, instead of ORT's EXHAUSTIVE
search with an unbounded workspace and power-of-two arenas. That is what keeps
a 15-page OCR-heavy PDF stable on an 11 GB card (peak ~8.1 GB instead of
~10.7 GB). Override with `DOCLING_RS_CUDA_CONV_ALGO`,
`DOCLING_RS_CUDA_CONV_MAX_WORKSPACE`, `DOCLING_RS_CUDA_ARENA_EXTEND`,
`DOCLING_RS_CUDA_MEM_LIMIT_MB`.

The same patch fixes **VRAM creep across a batch ingest**: ORT's BFC arenas
never return free chunks to the driver, and the pipeline reuses its sessions for
every document, so mixed page sizes ratchet the reservation upward until an
ordinary mid-run allocation fails. Every inference call therefore runs with
shared `RunOptions` enabling `memory.enable_memory_arena_shrinkage=gpu:0`
(each Run returns its unused arena chunks). Measured: batch peak 9.3 GB → 4.1 GB
with no accumulation, at ~10–20% slower inference. It engages only after a
one-time probe proves the CUDA EP really initialized (a CUDA build without
drivers falls back per session, where the key would fail every Run);
`DOCLING_RS_CUDA_ARENA_SHRINK=0` opts out, `DOCLING_RS_DEBUG=1` logs the probe
and the decision.

### Other docling knobs

| variable | effect |
|---|---|
| `DOCLING_RS_DEBUG=1` | verbose docling/ONNX logging (probe decisions, run options) |
| `DOCLING_RS_TIMING=1` | per-stage timing inside docling |
| `DOCLING_RS_GRAPH_CACHE_DIR`, `DOCLING_RS_NO_GRAPH_CACHE=1` | where ONNX graphs are cached / disable caching (first-run speed vs disk) |
| `DOCLING_RS_NO_ARENA=1` | disable ORT arenas entirely (diagnostic; slow) |
| `DOCLING_RS_XNNPACK_THREADS=<n>` | CPU EP thread count |
| `DOCLING_RS_FP32=1` | force fp32 inference |

## Performance

Measured on an RTX 2080 Ti (11 GB) + Ryzen 9 5950X. Embedding is a small share
of a document's ingest cost — docling conversion dominates (~7.3 s for a
15-page PDF), so changing the embedder moves end-to-end time far less than the
per-embedding numbers suggest.

| workload | GPU | CPU |
|---|---|---|
| bge text (32 × 244 tok) | 536 chunks/s | 33 chunks/s |
| EG2 text fp32 (32 × 217 tok) | 118 chunks/s | 7 chunks/s |
| EG2 text q8 | 112 chunks/s | 6 chunks/s |
| SigLIP2 vision (224 px) | 154 img/s | 17 img/s |
| EG2 vision fp32 @280 / @70 | 5.4 / 30 img/s | 0.6 / ~3 img/s |
| EG2 vision q8 @70 | 30 img/s | 3 img/s |

- **Batching buys nothing for images** (SigLIP2 and EG2 alike) and nothing for
  text past batch 32: EG2 vision attention is quadratic in total patches, so
  batches cost VRAM without adding throughput — at budget 280 a batch of 8 asks
  for a 5 GB buffer and fails on an 11 GB card. The pipeline's one-image-per-call
  shape is already right.
- **q8 is not faster** on this GPU or CPU (Zen 3 has no VNNI), but saves 20–40%
  VRAM at parity-level quality: choose it for memory, not speed.
- **CPU**: bulk ingest is impractical (~4 h per 100k chunks), incremental is
  fine (~2.5 min per 1k chunks). Thread scaling is flat between 8 and 32
  threads, so capping (`OMP_NUM_THREADS=8`) costs nothing.
- Knobs that matter most: `LS_IMAGE_SOFT_TOKENS` (70 is ~5× faster than 280),
  `LS_CHUNK_SIZE`, and for text-only corpora `LS_IMAGE_EMBED_PROVIDER=off`.

The full tables, including the with/without arena-shrinkage comparison and the
batching sweep, live in the header of `examples/eg2_bench.rs`; reproduce with:

```bash
cargo run --release --example eg2_bench -- eg2-text-fp32 20 32 --shrink
cargo run --release --example eg2_bench -- eg2-vision280 5 1 --shrink
```

## Common symptoms

| symptom | likely cause / fix |
|---|---|
| `libcuda.so.1: cannot open shared object file` | container without `--gpus all` (or missing nvidia-container-toolkit); else CPU-only host |
| `CUDA is compiled in but the CUDA runtime is not loadable` | driver (`libcuda.so.1`) or cuDNN 9 missing; the provider is pinned to CPU before inference |
| intermittent OOM mid-batch | arena creep — the vendored shrinkage handles it; check `DOCLING_RS_DEBUG=1` shows it engaged, and that you are not running two ingests |
| `database is locked` | two writers (a manual `ingest` plus the scheduled child, or two containers). Writers queue for 5 s (rusqlite's default busy timeout) then fail |
| vector/`image` mode returns nothing after a config change | the layout/dimension changed: vectors were dropped and the stage forgotten — run `ingest` to re-embed |
| a file-name query returns nothing | use `mode=hybrid` (name channel) or `/api/documents?query=`; `bm25`/`vector` search content only |
| `no results` for a term you know exists | wrong mode for the query shape (quoted phrases in `vector`, prose in `bm25`), or the corpus was never ingested (`/api/stats` shows counts) |
| a bad file fails on every run | it shouldn't now — failures are remembered; if it repeats, the file or the extractor identity is changing (see [Failed files](#failed-files)) |
| ingest exit code 2 | at least one *new* failure this run; remembered failures don't set it. Per-file errors are in the log |
| first query after a quiet period is slow | the models were idle-evicted and are reloading — expected; raise `LS_EMBED_IDLE_TIMEOUT` or set `off` to keep them |

## Diagnostic tools

| tool | use |
|---|---|
| `examples/chunk_audit.rs` | per-chunk token counts with the configured (or a given) tokenizer |
| `examples/eg2_bench.rs` | embedding throughput/VRAM across providers, budgets, batching, CPU/GPU |
| `examples/eg2_parity.py` | EG2 ONNX parity against the official reference (needs the Python venv described in its header) |
| `examples/docling_dump.rs` | dump docling Markdown for one file under different options |
| `examples/rasterize.rs` | rasterize a PDF's pages (to exercise OCR on a text-less image) |
| `examples/siglip_probe.rs` | load the SigLIP towers and print dims/cosines |
| `veda eval` / `eval-gen` | retrieval quality on your corpus — see [EVAL.md](EVAL.md) |
