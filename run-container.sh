#!/usr/bin/env bash
# Run local-search in Docker (image built from ./Dockerfile).
#
# Mounts (point the host side wherever you want):
#   ./documents -> /documents  ingest source (read-only)
#   ./data      -> /data       SQLite DB + blobs
#   ./models    -> /models     model tree (downloaded on first run)
#   ./config    -> /config     optional .env

docker run --rm --gpus all \
  --name local-search \
  -p 8080:8080 \
  -e PUID="$(id -u)" \
  -e PGID="$(id -g)" \
  -v "$PWD/documents:/documents:ro" \
  -v "$PWD/data:/data" \
  -v "$PWD/models:/models" \
  -v "$PWD/config:/config" \
  local-search serve
