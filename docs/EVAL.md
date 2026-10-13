# Retrieval evaluation

`local-search eval` scores a **gold set** against the current store, and
`local-search eval-gen` mines a synthetic one from the corpus. Together they
are the harness that decides whether an embedding change (EG2 text/vision, a
different chunker, a different budget) actually improves retrieval — benchmark
numbers from model cards are a prior, not a measurement of *this* corpus.

## Quick start

```bash
# 1. Build a gold set from whatever is already ingested.
LS_DB_PATH=data/local-search.db local-search eval-gen \
    --out eval/queries.jsonl --per-doc 3

# 2. Score every retrieval mode.
LS_DB_PATH=data/local-search.db local-search eval \
    --queries eval/queries.jsonl --modes hybrid,vector,bm25 --k 1,5,10
```

```
mode       queries       MRR  recall@k    ndcg@k
hybrid          33     0.936 1:0.909 5:1.000 10:1.000 1:0.909 5:0.952 10:0.952
  filename      13     0.838  recall 0.769 1.000 1.000
  phrase        10     1.000  recall 1.000 1.000 1.000
  title         10     1.000  recall 1.000 1.000 1.000
```

`--json` emits the same report as JSON (for diffing two runs).

## Gold-set format (JSONL)

```json
{"query": "depropanizer distillate composition analyzer",
 "kind": "phrase",
 "relevant_chunk_ids": ["3f1c…"]}
{"query": "boiler_notes", "kind": "filename",
 "relevant_doc_ids": ["51f6…"]}
```

- A result is a hit when its `chunk.id` is in `relevant_chunk_ids` **or** its
  `chunk.doc_id` is in `relevant_doc_ids` — chunk-level for precision,
  document-level for "find this document" queries.
- `kind` is a free-form label; the report groups by it.
- Hand-written queries can be appended to the same file at any time; they are
  the only ones that measure *real* retrieval quality (see caveats).

## Metrics

| metric | definition |
|---|---|
| **recall@k** | fraction of the query's relevant targets found in the top `k`, normalised by `min(relevant, k)` |
| **MRR** | reciprocal rank of the first hit (`1/rank`), averaged over queries |
| **nDCG@k** | binary-relevance DCG over the top `k`, divided by the ideal (all hits at the top) |

## What `eval-gen` produces

For each document, up to `--per-doc` **phrase** queries (an 8-word window
scored for rare/long/digit-bearing tokens), plus one **title** query and one
**filename** query. Documents with no text chunks (image-only folders) still
get title/filename queries.

## Caveats — read before drawing conclusions

- **Phrase queries are lexically identical to the corpus**, which flatters
  bm25. They are useful for regression detection (a drop means something
  broke) but not for ranking semantic quality.
- **`filename` queries**: `mode=hybrid` matches document names directly (its
  name channel contributes the chunks of any document whose title or source
  path contains the query), so these score high on hybrid even for
  non-descriptive names like `frame_00000`. `mode=bm25`/`vector` search
  content only and stay near zero for those — the gap between the two is
  itself the measurement. `GET /api/documents?query=…` remains the precise
  name lookup.
- **No image→image evaluation yet.** Query images need labels; `mode=image`
  is exercised indirectly (text→image) only through hybrid.
- **Small corpora give noisy numbers.** A handful of documents makes MRR jump
  between 0.9 and 1.0; treat differences under ~0.05 as noise.
- **Chunk-level relevance is strict**: a hit must be the exact chunk. Long
  documents with many near-duplicate chunks will under-report.

## Comparing embedders (the reason this exists)

Ingest the same corpus into separate DBs per configuration, generate **one**
gold set, and score each DB against it:

```bash
# baseline: bge text + SigLIP2 image
LS_DB_PATH=/tmp/base.db  LS_MODELS_DIR=.models local-search ingest docs photos
# candidate: EG2 text + EG2 vision (phases 1–3)
LS_DB_PATH=/tmp/eg2.db   LS_MODELS_DIR=.models LS_EMBED_PROVIDER=eg2 \
    LS_IMAGE_EMBED_PROVIDER=eg2 local-search ingest docs photos

for db in /tmp/base.db /tmp/eg2.db; do
  echo "== $db"
  LS_DB_PATH=$db LS_MODELS_DIR=.models local-search eval \
      --queries eval/queries.jsonl --k 1,5,10
done
```

Costs (throughput, VRAM, ingest wall time) come from `examples/eg2_bench.rs`;
quality comes from here. A model switch is justified when quality moves
meaningfully *and* the cost is acceptable — never on model-card numbers alone.
