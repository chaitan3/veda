#!/usr/bin/env bash
# Package local-search for deployment: the binary, the native runtime libraries,
# the model tree, and config/docs templates.
#
# Usage:
#   scripts/package.sh [OUT_DIR] [--no-models] [--debug]
#
#   OUT_DIR       destination directory (default: dist/local-search)
#   --no-models   skip .models/ and .pdfium/ (~650 MB); fetch them on the host
#                 with `local-search download-models` instead
#   --debug       package the debug binary instead of the release one
#
# The packaged directory is the service's working directory: docling resolves
# .models/ and .pdfium/ relative to the CWD.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="dist/local-search"
PROFILE="release"
WITH_MODELS=1

for arg in "$@"; do
  case "$arg" in
    --no-models) WITH_MODELS=0 ;;
    --debug) PROFILE="debug" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) OUT="$arg" ;;
  esac
done

case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac

BINDIR="$ROOT/target/$PROFILE"
BIN="$BINDIR/local-search"
if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found." >&2
  echo "       build it first:  cargo build$( [ "$PROFILE" = release ] && echo ' --release' )" >&2
  exit 1
fi

echo "packaging '$PROFILE' build -> $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"

# 1. the product binary
cp -a "$BIN" "$OUT/local-search"

# 2. ONNX Runtime libraries, next to the binary. Only the execution-provider
#    .so files are shared (the base runtime is linked in statically); the GPU
#    providers are optional — a CPU-only host runs without them.
#    NB: ort materializes these as symlinks into its download cache, so they
#    must be dereferenced (-L) or the package breaks on a fresh host.
shopt -s nullglob
libs=("$BINDIR"/libonnxruntime.so* "$BINDIR"/libonnxruntime_providers_*.so)
shopt -u nullglob
if [ ${#libs[@]} -gt 0 ]; then
  cp -aL "${libs[@]}" "$OUT/"
  echo "  + ${#libs[@]} ONNX Runtime library/libraries"
else
  echo "  ! no ONNX Runtime libs found (fine for CPU-only)"
fi

# 3. model tree (docling assets + the ONNX text embedder) and pdfium.
if [ "$WITH_MODELS" = 1 ]; then
  for dir in .models .pdfium; do
    if [ -e "$ROOT/$dir" ]; then
      cp -aL "$ROOT/$dir" "$OUT/"
      echo "  + $dir"
    else
      echo "  ! $dir is missing — run: local-search download-models" >&2
    fi
  done
else
  echo "  - skipping models (--no-models); run 'local-search download-models' on the host"
fi

# 4. config + docs templates.
[ -f "$ROOT/.env.example" ] && cp -a "$ROOT/.env.example" "$OUT/.env.example"
[ -f "$ROOT/README.md" ] && cp -a "$ROOT/README.md" "$OUT/README.md"
if [ -f "$ROOT/docs/PRODUCTION.md" ]; then
  mkdir -p "$OUT/docs"
  cp -a "$ROOT/docs/PRODUCTION.md" "$OUT/docs/PRODUCTION.md"
fi
if [ -f "$ROOT/deploy/local-search.service" ]; then
  mkdir -p "$OUT/deploy"
  cp -a "$ROOT/deploy/local-search.service" "$OUT/deploy/local-search.service"
fi

echo
echo "contents of $OUT:"
( cd "$OUT" && ls -la )
echo
echo "package size: $(du -sh "$OUT" | cut -f1)"

cat <<'EOF'

Next steps on the target host:
  1. Put the package somewhere, e.g. /opt/local-search, and cd into it.
  2. cp .env.example .env    # set LS_DB_PATH, LS_SOURCE_PATH, LS_API_KEYS
  3. ./local-search download-models   # only if you packaged with --no-models
  4. ./local-search init-db
  5. ./local-search ingest /path/to/documents
  6. ./local-search serve             # see docs/PRODUCTION.md
EOF
