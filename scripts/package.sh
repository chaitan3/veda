#!/usr/bin/env bash
# Package veda for deployment: the binary, the native runtime libraries,
# the model tree, and config/docs templates.
#
# Usage:
#   scripts/package.sh [OUT_DIR] [--no-models] [--debug]
#
#   OUT_DIR       destination directory (default: dist/veda)
#   --no-models   skip .models/ and .pdfium/ (~650 MB); fetch them on the host
#                 with `veda download-models` instead
#   --debug       package the debug binary instead of the release one
#
# The packaged directory is the service's working directory: docling resolves
# .models/ and .pdfium/ relative to the CWD.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="dist/veda"
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
BIN="$BINDIR/veda"
if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found." >&2
  echo "       build it first:  cargo build$( [ "$PROFILE" = release ] && echo ' --release' )" >&2
  exit 1
fi

echo "packaging '$PROFILE' build -> $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"

# 1. the product binary
cp -a "$BIN" "$OUT/veda"

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
      echo "  ! $dir is missing — run: veda download-models" >&2
    fi
  done
else
  echo "  - skipping models (--no-models); run 'veda download-models' on the host"
fi

# 4. config + docs templates.
[ -f "$ROOT/.env.example" ] && cp -a "$ROOT/.env.example" "$OUT/.env.example"
[ -f "$ROOT/README.md" ] && cp -a "$ROOT/README.md" "$OUT/README.md"
if [ -d "$ROOT/docs" ]; then
  mkdir -p "$OUT/docs"
  cp -a "$ROOT"/docs/*.md "$OUT/docs/"
fi
[ -f "$ROOT/scripts/veda.service" ] && \
  cp -a "$ROOT/scripts/veda.service" "$OUT/veda.service"

echo
echo "contents of $OUT:"
( cd "$OUT" && ls -la )
echo
echo "package size: $(du -sh "$OUT" | cut -f1)"

cat <<'EOF'

Next steps on the target host:
  1. Put the package somewhere, e.g. /opt/veda, and cd into it.
  2. cp .env.example .env    # set LS_DB_PATH, LS_SOURCE_PATH, LS_API_KEYS
  3. ./veda download-models   # only if you packaged with --no-models
  4. ./veda init-db
  5. ./veda ingest /path/to/documents
  6. ./veda serve             # see docs/ for architecture, eval, debugging
  7. systemd: copy ./veda.service to /etc/systemd/system/ and adjust it
EOF
