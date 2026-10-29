"""Prometheus metrics.

All names are prefixed ``tts_`` so they can share a scrape config with the ASR
service. Import is cheap and side-effect free; the HTTP route lives in
``main``.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

# ── HTTP ────────────────────────────────────────────────────────────────────
http_requests_total = Counter(
    "tts_http_requests_total", "HTTP requests", ["route", "method", "status"], registry=REGISTRY
)
http_request_duration_seconds = Histogram(
    "tts_http_request_duration_seconds", "HTTP request duration",
    ["route"], buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)

# ── Model / scheduler ───────────────────────────────────────────────────────
model_ready = Gauge("tts_model_ready", "1 when the model is loaded and serving", registry=REGISTRY)
gpu_memory_bytes = Gauge(
    "tts_gpu_memory_bytes", "CUDA memory in use by this process", ["kind"], registry=REGISTRY
)
queue_depth = Gauge("tts_queue_depth", "Jobs waiting for the GPU", registry=REGISTRY)
queue_capacity = Gauge("tts_queue_capacity", "Admission limit in jobs", registry=REGISTRY)
inflight_jobs = Gauge("tts_inflight_jobs", "Jobs currently running on the GPU", registry=REGISTRY)
queue_wait_seconds = Histogram(
    "tts_queue_wait_seconds", "Time a job spent waiting before inference",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)

# ── Synthesis ───────────────────────────────────────────────────────────────
inference_seconds = Histogram(
    "tts_inference_seconds", "One model call", ["kind", "language"],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 20, 40),
    registry=REGISTRY,
)
realtime_factor = Histogram(
    "tts_realtime_factor", "Audio seconds generated per wall second", ["kind"],
    buckets=(0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 100), registry=REGISTRY,
)
audio_seconds_total = Counter(
    "tts_audio_seconds_total", "Audio seconds generated", ["language"], registry=REGISTRY
)
chunks_total = Counter("tts_chunks_total", "Chunks that produced audio", registry=REGISTRY)
batches_total = Counter("tts_batches_total", "Model calls (a batch of chunks counts once)", registry=REGISTRY)
batch_size = Histogram(
    "tts_batch_size", "Chunks per model call",
    buckets=(1, 2, 3, 4, 6, 8, 12, 16), registry=REGISTRY,
)
input_chars = Histogram(
    "tts_input_chars", "Characters per synthesis request",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000), registry=REGISTRY,
)
requests_total = Counter(
    "tts_synthesis_requests_total", "Synthesis requests", ["kind", "outcome"], registry=REGISTRY
)
first_chunk_seconds = Histogram(
    "tts_first_chunk_seconds", "Request start to first audio chunk",
    ["kind"], buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8),
    registry=REGISTRY,
)
encode_seconds = Histogram(
    "tts_encode_seconds", "Audio encoding time", ["format"],
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2), registry=REGISTRY,
)
inference_errors_total = Counter(
    "tts_inference_errors_total", "Model calls that raised", registry=REGISTRY
)
split_batches_total = Counter(
    "tts_split_batches_total", "Batches retried item-by-item after a failure", registry=REGISTRY
)
rejected_total = Counter(
    "tts_rejected_total", "Requests rejected by admission control", ["reason"], registry=REGISTRY
)

# ── Streaming sessions ──────────────────────────────────────────────────────
stream_sessions_active = Gauge(
    "tts_stream_sessions_active", "Open WebSocket synthesis sessions", registry=REGISTRY
)
stream_sessions_total = Counter(
    "tts_stream_sessions_total", "WebSocket sessions by outcome", ["outcome"], registry=REGISTRY
)
stream_session_seconds = Histogram(
    "tts_stream_session_seconds", "WebSocket session duration",
    buckets=(1, 5, 15, 30, 60, 120, 300, 900, 3600), registry=REGISTRY,
)
pending_audio_seconds = Gauge(
    "tts_pending_audio_seconds", "Estimated seconds of audio waiting in the queue", registry=REGISTRY
)

# ── Voice store ─────────────────────────────────────────────────────────────
prompt_cache_hits = Counter("tts_prompt_cache_hits_total", "Voice prompt cache hits", registry=REGISTRY)
prompt_cache_misses = Counter(
    "tts_prompt_cache_misses_total", "Voice prompt cache misses", registry=REGISTRY
)
sanitize_total = Counter(
    "tts_sanitize_total", "Text-sanitisation calls by outcome", ["outcome"], registry=REGISTRY
)
sanitize_seconds = Histogram(
    "tts_sanitize_seconds", "Time spent in the sanitisation LLM",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10), registry=REGISTRY,
)
voice_operations_total = Counter(
    "tts_voice_operations_total", "Voice store operations", ["operation", "outcome"], registry=REGISTRY
)
