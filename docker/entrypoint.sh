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

# Ensure the model tree: docling assets + bge-small + SigLIP2 + CodeFormulaV2.
# Idempotent — files already present are skipped without any network use, so an
# existing volume only fetches models added by a newer version. Skip entirely
# for a pre-provisioned volume or a remote-VLM / llama.cpp setup.
if [ "${LS_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    echo "[entrypoint] ensuring model tree in $MODELS_DIR (idempotent)" >&2
    /app/local-search download-models
fi

if [ "$(id -u)" = "0" ]; then
    # The model tree is a few dozen files; hand them to the service user.
    chown -R "$PUID:$PGID" /data "$MODELS_DIR" 2>/dev/null || true
    exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups \
        /app/local-search "$@"
fi

exec /app/local-search "$@"
