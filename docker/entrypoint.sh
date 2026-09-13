#!/bin/sh
# local-search container entrypoint.
#
#   1. Links an optional /config/.env so the binary's dotenvy loader reads it
#      (dotenvy does not override variables Docker already set, so container
#      defaults and `-e` win over the file).
#   2. Bridges docling's CWD-relative `.pdfium` to the model volume, since
#      `download-models` always writes `./.pdfium` regardless of LS_MODELS_DIR.
#   3. Downloads the model tree on first run when the volume is unprovisioned.
#   4. Drops from root to PUID:PGID before running the binary.
set -eu

PUID="${PUID:-1000}"
PGID="${PGID:-1000}"
MODELS_DIR="${LS_MODELS_DIR:-/models}"

# Optional configuration. The binary loads /app/.env via dotenvy.
if [ -f /config/.env ]; then
    ln -sf /config/.env /app/.env 2>/dev/null || true
fi

mkdir -p /data /documents "$MODELS_DIR/.pdfium"
if [ ! -e /app/.pdfium ]; then
    ln -s "$MODELS_DIR/.pdfium" /app/.pdfium
fi

# First-run bootstrap of the docling assets + bge-small embedder (~650 MB).
# Skip for a pre-provisioned volume or a remote-VLM / llama.cpp setup.
downloaded=0
if [ ! -f "$MODELS_DIR/embed/bge-small-en-v1.5.onnx" ] \
    && [ "${LS_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    echo "[entrypoint] no model tree in $MODELS_DIR; downloading (one-time, ~650 MB)" >&2
    /app/local-search download-models
    downloaded=1
fi

if [ "$(id -u)" = "0" ]; then
    # Files created above are root-owned; hand them to the service user. Only
    # walk the tree when we actually wrote it, to keep restarts fast.
    chown "$PUID:$PGID" /data "$MODELS_DIR" 2>/dev/null || true
    [ "$downloaded" = "1" ] && chown -R "$PUID:$PGID" "$MODELS_DIR" 2>/dev/null || true
    exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups \
        /app/local-search "$@"
fi

exec /app/local-search "$@"
