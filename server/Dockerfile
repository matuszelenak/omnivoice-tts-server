# syntax=docker/dockerfile:1
#
# Development image: hot-reloading API only (the Vite dev server serves the UI).
#   docker compose up --watch
FROM ghcr.io/astral-sh/uv@sha256:929560df6a6231d0509739319bc214aaa1e78838ce1f779b1faeb877c23a50d8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libsndfile1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=managed \
    UV_PYTHON=3.12 \
    UV_CONCURRENT_INSTALLS=8 \
    HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf/hub \
    TORCH_HOME=/cache/torch

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev && \
    find /app/.venv/lib -path '*/nvidia/*/lib' -type d \
        | tee /etc/ld.so.conf.d/nvidia-pip-pkgs.conf \
    && ldconfig

COPY app app
COPY tests tests

EXPOSE 9001

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD wget --no-verbose --tries=1 http://localhost:9001/readyz || exit 1

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9001", "--reload"]
