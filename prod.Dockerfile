# syntax=docker/dockerfile:1
#
# Production image: builds the Svelte frontend and bakes it into the server
# image, which serves it as static files alongside the API on a single port.
#
# Build from the repository root:
#   docker build -f prod.Dockerfile -t omnivoice-tts:prod .
# Run (needs the NVIDIA Container Toolkit):
#   docker run --gpus all -p 9001:9001 omnivoice-tts:prod
# Then open http://<host>:9001 — UI and API are served from the same origin.

# ---- Stage 1: build the frontend with Deno ----
FROM denoland/deno:2.8.1 AS frontend
WORKDIR /frontend
# Install deps first for layer caching.
COPY frontend/deno.json frontend/package.json ./
RUN deno install
COPY frontend/ ./
# The bundle calls the API at its own origin (this same server), so there is no
# API base to bake in. Outputs to /frontend/dist.
RUN deno task build

# ---- Stage 2: assemble the server image ----
FROM ghcr.io/astral-sh/uv@sha256:929560df6a6231d0509739319bc214aaa1e78838ce1f779b1faeb877c23a50d8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        ffmpeg \
        libsndfile1 \
        build-essential \
        wget \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=managed \
    UV_COMPILE_BYTECODE=1 \
    UV_CONCURRENT_INSTALLS=8 \
    HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf/hub \
    TORCH_HOME=/cache/torch \
    STATIC_DIR=/app/static

WORKDIR /app

# Lockfile-first install keeps this (huge) layer byte-identical across builds
# as long as dependencies don't change, maximizing remote cache hits.
COPY server/pyproject.toml server/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY server/main.py ./main.py
COPY server/src ./src
COPY --from=frontend /frontend/dist ./static

EXPOSE 9001

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD wget --no-verbose --tries=1 http://localhost:9001/health || exit 1

CMD ["uv", "run", "--no-dev", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001"]
