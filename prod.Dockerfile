# syntax=docker/dockerfile:1
#
# Production image: the Svelte frontend is compiled and baked into the API image,
# which serves the UI and the API from a single port.
#
#   docker build -f prod.Dockerfile -t omnivoice-tts:prod .
#   docker run --gpus '"device=0"' -p 9001:9001 \
#       -e ASR_URL=http://host.docker.internal:9000 omnivoice-tts:prod
#
# Three stages, because the official `uv` image is a full build environment
# (gcc, git, LLVM, dev headers, its own Python 3.13 - ~700 MB) and none of that
# belongs in a runtime image:
#
#   frontend  deno + vite -> static/
#   pydeps    uv + python:3.12-slim -> /app/.venv, then pruned
#   runtime   python:3.12-slim + ffmpeg/libsndfile -> the shipped image

# ---- Stage 1: build the frontend with Deno ----
FROM denoland/deno:2.8.1 AS frontend
WORKDIR /frontend
# Dependencies first (from the committed lockfile) for layer caching.
COPY frontend/deno.json frontend/package.json frontend/deno.lock ./
RUN deno install --frozen
COPY frontend/ ./
# The bundle calls the API at its own origin (this same server), so there is no
# API base to bake in. Outputs to /frontend/dist.
RUN deno task build \
    && find /frontend/dist -type f \( -name '*.js' -o -name '*.css' -o -name '*.wasm' \) -size +256k \
        -exec gzip -9 -k -f {} +

# ---- Stage 2: Python dependencies ----
FROM python:3.12-slim-bookworm AS pydeps
# uv itself is only needed here; the runtime image runs the venv directly.
COPY --from=ghcr.io/astral-sh/uv@sha256:929560df6a6231d0509739319bc214aaa1e78838ce1f779b1faeb877c23a50d8 /usr/local/bin/uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_COMPILE_BYTECODE=1 \
    UV_CONCURRENT_INSTALLS=8 \
    HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf/hub \
    TORCH_HOME=/cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor

WORKDIR /app
# Every dependency ships a manylinux wheel, so no compiler is needed.
COPY server/pyproject.toml server/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# omnivoice pulls a Gradio demo UI, pandas and scikit-learn (its eval tooling).
# Nothing on the serving path imports them - `app.main` and
# `omnivoice.models.omnivoice` only touch `cli/demo.py`-level modules - so drop
# them and the 164 MB they occupy.
RUN uv pip uninstall --python /app/.venv/bin/python \
        gradio gradio-client pandas scikit-learn

# Prune what a wheel ships but inference never touches (~380 MB):
#   torch/include            C++ headers, only needed to build extensions
#   libnvrtc.alt.so           alternate-arch nvrtc; torch links libnvrtc.so
#   libcusolverMg.so          multi-GPU solver; nothing links it
#   libnvperf_*               profiler API; torch links libcupti, not these
#   triton .../lib/cupti      static profiler archives bundled for triton's build
# (triton/backends/amd stays: `import triton` touches every backend package.)
RUN SP=/app/.venv/lib/python3.12/site-packages \
    && rm -rf "$SP/torch/include" "$SP"/triton/backends/*/lib/cupti \
    && rm -f "$SP"/nvidia/cu13/lib/libnvrtc.alt.so.13 \
              "$SP"/nvidia/cu13/lib/libcusolverMg.so.12 \
              "$SP"/nvidia/cu13/lib/libnvperf_*.so* \
    && du -sh /app/.venv

# ---- Stage 3: runtime ----
FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libsndfile decodes WAV/FLAC/OGG; ffmpeg covers MP3/M4A/MP4/WebM and encodes
# mp3/opus/aac; wget runs the healthcheck; gcc/libc6-dev are required at *run*
# time by torch.compile (triton compiles a small launcher stub per kernel);
# pip/setuptools are not needed because the venv is prebuilt.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        gcc \
        libc6-dev \
        libsndfile1 \
        wget \
    && rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man /usr/share/locale/* \
    && rm -rf /usr/local/lib/python3.12/site-packages/pip* \
              /usr/local/lib/python3.12/site-packages/setuptools* \
              /usr/local/lib/python3.12/site-packages/wheel* \
              /usr/local/bin/pip*

ENV HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf/hub \
    TORCH_HOME=/cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRANSFORMERS_VERBOSITY=error \
    STATIC_DIR=/app/static \
    VOICE_SAMPLES_DIR=/app/voices \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# The CUDA libraries come from pip wheels; ldconfig is what lets the dynamic
# linker find libcudart in the venv at import time.
COPY --from=pydeps /app/.venv /app/.venv
RUN find /app/.venv/lib -path '*/nvidia/*/lib' -type d \
        | tee /etc/ld.so.conf.d/nvidia-pip-pkgs.conf \
    && ldconfig

COPY server/app ./app
COPY server/voices ./voices
COPY --from=frontend /frontend/dist ./static

EXPOSE 9001

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD wget --no-verbose --tries=1 http://localhost:9001/readyz || exit 1

# granian (Rust/Hyper) instead of uvicorn: one worker, because the scheduler,
# session counters and metrics are per process. See docs/deployment.md.
CMD ["/app/.venv/bin/granian", "--interface", "asgi", "--host", "0.0.0.0", \
     "--port", "9001", "--workers", "1", "--log-level", "info", \
     "--ws", "--backpressure", "64", "app.main:app"]
