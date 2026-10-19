# syntax=docker/dockerfile:1.7

# veda, container image.
#
# Build from the repository root (the build context must include src/, vendor/,
# Cargo.toml and docker/):
#   docker build -f docker/Dockerfile -t veda .
#
# Two stages, because the binary must be compiled against the same glibc the
# image ships: NVIDIA's CUDA 13 / cuDNN 9 images are Ubuntu 24.04 (glibc 2.39),
# and a binary built on a rolling distro (glibc >= 2.43) fails to load with
# "version `GLIBC_2.43' not found" (glibc 2.43 re-versioned several libm fns).
# The model tree is NOT baked in: it is downloaded on first run into a volume.

# ---- builder ---------------------------------------------------------------
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        pkg-config \
        cmake \
        clang \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /build
COPY Cargo.toml Cargo.lock build.rs ./
# Vendored docling-onnx (see `[patch.crates-io]` in Cargo.toml); the build fails
# with "failed to read /build/vendor/docling-onnx/Cargo.toml" without it.
COPY vendor ./vendor
COPY src ./src
COPY examples ./examples
COPY tests ./tests

# Build with the CUDA execution provider (default features). `ort` fetches the
# prebuilt ONNX Runtime CUDA provider `.so` files into target/release. Cargo
# registry/target cache mounts keep rebuilds incremental.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    set -eux; \
    cargo build --release --locked; \
    mkdir -p /out; \
    cp -a target/release/veda /out/; \
    cp -aL target/release/libonnxruntime_providers_*.so /out/ 2>/dev/null || true

# ---- runtime ---------------------------------------------------------------
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
        util-linux \
        libstdc++6 \
        libgomp1 \
        libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Container-first defaults. The host defaults (127.0.0.1, data/, .models) do not
# work here: bind the API on all interfaces and keep state on volumes.
ENV LS_ADDR=0.0.0.0:8080 \
    LS_DB_PATH=/data/veda.db \
    LS_SOURCE_PATH=/documents \
    LS_MODELS_DIR=/models \
    LS_EXTRACT_PIPELINE=onnx \
    LS_EMBED_PROVIDER=onnx \
    LS_EMBED_DIM=384 \
    LS_CHUNKER=hybrid

WORKDIR /app
COPY --from=builder /out/veda /app/veda
COPY --from=builder /out/libonnxruntime_providers_*.so /app/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/.env.example /app/.env.example
COPY README.md /app/README.md
COPY docs /app/docs

# Numeric UID/GID the service drops to. Override to match a bind-mounted host
# directory: docker build --build-arg PUID=$(id -u) --build-arg PGID=$(id -g) .
ARG PUID=1000
ARG PGID=1000
ENV PUID=$PUID PGID=$PGID

RUN chmod +x /app/veda /usr/local/bin/entrypoint.sh \
    && mkdir -p /data /documents /models /config

VOLUME ["/data", "/documents", "/models", "/config"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://${LS_ADDR}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
