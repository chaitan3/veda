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

# Ensure the model tree: docling assets + bge-small + SigLIP2 + EmbeddingGemma 2
# + CodeFormulaV2.
# Idempotent — files already present are skipped without any network use, so an
# existing volume only fetches models added by a newer version. Skip entirely
# for a pre-provisioned volume or a remote-VLM / llama.cpp setup.
if [ "${LS_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    # EmbeddingGemma 2 is the largest download (~1.8 GB) and most deployments
    # run bge + SigLIP2, so only fetch it when a provider asks for it -- from
    # the environment (docker -e) or the mounted /config/.env.
    eg2_wanted=0
    for v in "${LS_EMBED_PROVIDER:-}" "${LS_IMAGE_EMBED_PROVIDER:-}"; do
        [ "$v" = "eg2" ] && eg2_wanted=1
    done
    if [ -f /config/.env ] && grep -Eq '^[[:space:]]*(LS_EMBED_PROVIDER|LS_IMAGE_EMBED_PROVIDER)[[:space:]]*=[[:space:]]*eg2' /config/.env; then
        eg2_wanted=1
    fi
    eg2_flag="--no-eg2"
    if [ "$eg2_wanted" = "1" ]; then
        eg2_flag=""
        echo "[entrypoint] eg2 provider configured; fetching EmbeddingGemma 2 (~1.8 GB)" >&2
    fi
    echo "[entrypoint] ensuring model tree in $MODELS_DIR (idempotent)" >&2
    # shellcheck disable=SC2086
    /app/local-search download-models $eg2_flag
fi

if [ "$(id -u)" = "0" ]; then
    # The model tree is a few dozen files; hand them to the service user.
    chown -R "$PUID:$PGID" /data "$MODELS_DIR" 2>/dev/null || true
    exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups \
        /app/local-search "$@"
fi

exec /app/local-search "$@"
