#!/usr/bin/env bash
# Run local-search in Docker (image built from ./Dockerfile).
#
# Mounts (point the host side wherever you want):
#   ./documents -> /documents  text corpus, full pipeline (read-only)
#   ./photos    -> /photos     image-only folder: image channel only, no
#                              docling/OCR/text chunks (read-only)
#   ./data      -> /data       SQLite DB + blobs
#   ./models    -> /models     model tree (downloaded on first run)
#   ./config    -> /config     optional .env (extra settings)
#
# NOTE on precedence: the two -e variables below are passed as real
# environment variables, so they win over /config/.env (dotenvy does not
# override variables Docker already set). To change the corpus layout, either
# edit the defaults here, export LS_SOURCE_PATH / LS_IMAGE_ONLY_PATHS before
# invoking this script, or remove those two -e lines and manage them in
# /config/.env.

# Source roots: everything under both folders is ingested; path-separated.
LS_SOURCE_PATH="${LS_SOURCE_PATH:-/documents:/photos}"

# Image-only policy root: image files under /photos are indexed by the image
# channel only (one SigLIP vector per image, no OCR, no text chunks).
# Non-image files in that folder still ingest normally.
LS_IMAGE_ONLY_PATHS="${LS_IMAGE_ONLY_PATHS:-/photos}"

mkdir -p documents photos data models config

docker run --rm --gpus all \
  --name local-search \
  -p 8080:8080 \
  -e PUID="$(id -u)" \
  -e PGID="$(id -g)" \
  -e LS_SOURCE_PATH="$LS_SOURCE_PATH" \
  -e LS_IMAGE_ONLY_PATHS="$LS_IMAGE_ONLY_PATHS" \
  -v "$PWD/documents:/documents:ro" \
  -v "$PWD/photos:/photos:ro" \
  -v "$PWD/data:/data" \
  -v "$PWD/models:/models" \
  -v "$PWD/config:/config" \
  local-search serve
