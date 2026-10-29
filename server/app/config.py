"""Runtime configuration, sourced from environment variables.

Everything the server can be tuned with lives here; see ``docs/configuration.md``
for the annotated reference. Values are read once at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str) -> list[str] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- Model -------------------------------------------------------------
    model_name: str = _str("MODEL_NAME", "k2-fsa/OmniVoice")
    # Id reported by ``/v1/models``; incoming ``model`` fields are informational.
    served_model_name: str = _str("SERVED_MODEL_NAME", "omnivoice")
    # ``DEVICE`` (new) with ``DEVICE_MAP`` kept for backwards compatibility.
    device: str = _str("DEVICE", "") or _str("DEVICE_MAP", "cuda:0")
    dtype: str = _str("DTYPE", "auto").lower()

    # --- Decoding ----------------------------------------------------------
    # Diffusion steps per chunk. Cost is exactly linear in this number.
    num_steps: int = _int("NUM_STEPS", 16)
    # Fewer steps for the first chunk of a stream: it is what the listener waits
    # for, and a single sentence survives the quality drop (see docs/benchmarks).
    # 0 = use NUM_STEPS for every chunk (one shape: better for CUDA graphs and
    # torch.compile, and pointless to differentiate on a GPU where a step is cheap).
    first_chunk_steps: int = _int("FIRST_CHUNK_STEPS", 12)
    # Classifier-free guidance scale. 0 is ~1.6x faster on a patched decoder and
    # measurably worse for Slovak; upstream still pays for the uncond branch.
    guidance_scale: float = _float("GUIDANCE_SCALE", 2.0)
    t_shift: float = _float("T_SHIFT", 0.1)
    postprocess_output: bool = _bool("POSTPROCESS_OUTPUT", True)
    # Upper bound accepted from a per-request ``num_step`` extension.
    max_num_steps: int = _int("MAX_NUM_STEPS", 64)

    # --- Batching ----------------------------------------------------------
    # Sentences that are queued at the same time are decoded in one model call
    # (1=off). Measured 1.22-1.29x per item at batch 8 with equal lengths; the
    # gain shrinks with padding, so batches are also grouped by length.
    batch_max_size: int = _int("BATCH_MAX_SIZE", 8)
    # Wait this long for a partner job when only one is queued. Under load the
    # queue is already deep and the wait is skipped.
    batch_window_ms: int = _int("BATCH_WINDOW_MS", 8)
    # Bound on the summed sequence length of one batch (tokens) -> VRAM bound.
    batch_max_tokens: int = _int("BATCH_MAX_TOKENS", 4096)
    # Longest/shortest estimated length ratio allowed inside a batch: padding a
    # short sentence up to a long one costs more than it saves.
    batch_length_ratio: float = _float("BATCH_LENGTH_RATIO", 1.6)
    # Batch the first chunk of a request too. Only ever happens when the chunk is
    # already queued behind others, i.e. it would have waited anyway.
    batch_first_chunks: bool = _bool("BATCH_FIRST_CHUNKS", True)

    # --- torch.compile -----------------------------------------------------
    # Compile the Qwen3 backbone with inductor. Measured 1.65x end-to-end
    # (225.8 -> 136.6 ms for a 16-step sentence) on the reference 3090, with
    # logits within 0.25 of eager. Costs ~40 s of compilation on first start
    # (cached afterwards in TORCHINDUCTOR_CACHE_DIR).
    torch_compile: bool = _bool("TORCH_COMPILE", True)
    compile_mode: str = _str("COMPILE_MODE", "default")

    # --- CUDA graphs -------------------------------------------------------
    # Capture the denoising forward pass into a CUDA graph and replay it.
    # Measured 1.13-1.16x on streamed requests (1.04x non-streaming) on a 3090,
    # with bit-identical logits; ~160 MB and ~40 ms per captured length bucket.
    # Set CUDA_GRAPHS=false to compare, or if a driver update breaks capture.
    # Graph replay adds ~6 % on top of an uncompiled model but nothing once
    # inductor has fused the kernels, so it defaults off when compile is on.
    cuda_graphs: bool = _bool("CUDA_GRAPHS", False)
    # Sequence lengths are rounded up to this multiple before capture, so one
    # graph serves a range of sentences (32 wastes ~6 % of tokens on average).
    graph_bucket: int = _int("GRAPH_BUCKET", 32)
    graph_max: int = _int("GRAPH_MAX", 4)
    # Do not capture when free VRAM is below this (a graph pins its intermediates).
    graph_min_free_mb: int = _int("GRAPH_MIN_FREE_MB", 4096)

    # --- Text --------------------------------------------------------------
    # Largest request body we synthesise (characters).
    max_input_chars: int = _int("MAX_INPUT_CHARS", 4000)
    # Hard cap on one model call; longer sentences are split at clause breaks.
    max_chunk_chars: int = _int("MAX_CHUNK_CHARS", 320)
    # Wait for the first sentence in streaming mode up to this many chars before
    # synthesising a fragment (keeps TTFA low without chopping words).
    min_chunk_chars: int = _int("MIN_CHUNK_CHARS", 8)

    # --- Scheduler / admission control -------------------------------------
    # Jobs queued (not started) before new work is rejected with 503.
    max_queue_jobs: int = _int("MAX_QUEUE_JOBS", 48)
    # Estimated seconds of audio queued before rejection: the real capacity bound.
    # At ~17 audio-seconds/s per GPU this bounds an admitted request's wait to
    # roughly a dozen seconds; anything beyond is answered with 503 + Retry-After
    # instead of an unbounded queue.
    max_pending_audio_s: float = _float("MAX_PENDING_AUDIO_S", 180.0)
    # How long an already-queued job may wait before it is failed with 503.
    queue_timeout_s: float = _float("QUEUE_TIMEOUT_S", 120.0)
    # ``Retry-After`` (seconds) sent with 503.
    retry_after_s: int = _int("RETRY_AFTER_S", 2)
    # Sentences of one request kept in flight at once (bounds head-of-line delay
    # for other users and memory for very long inputs).
    max_inflight_per_session: int = _int("MAX_INFLIGHT_PER_SESSION", 3)
    # First chunks served back-to-back before a queued follow-up chunk gets a
    # turn. Keeps time-to-first-audio short without starving sessions that are
    # already playing. 0 = strictly first-chunks-first.
    first_chunk_burst: int = _int("FIRST_CHUNK_BURST", 4)

    # --- WebSocket sessions ------------------------------------------------
    max_stream_sessions: int = _int("MAX_STREAM_SESSIONS", 64)
    # Unprocessed (queued, not yet synthesised) audio a client may accumulate.
    stream_max_pending_s: float = _float("STREAM_MAX_PENDING_S", 180.0)
    stream_max_session_s: float = _float("STREAM_MAX_SESSION_S", 3600.0)

    # --- Voice store -------------------------------------------------------
    voice_samples_dir: Path = field(
        default_factory=lambda: Path(_str("VOICE_SAMPLES_DIR", "voices"))
    )
    # References are cut to at most this long (the model degrades beyond ~10 s).
    max_ref_seconds: float = _float("MAX_REF_SECONDS", 8.0)
    min_ref_seconds: float = _float("MIN_REF_SECONDS", 2.0)
    # Largest reference upload accepted.
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 25)
    # ASR service used to transcribe reference clips (the sibling
    # parakeet-asr-server). Empty string disables it, in which case every voice
    # registration must supply ref_text and clips are cut on silence.
    asr_url: str = _str("ASR_URL", "http://localhost:9000")
    asr_timeout_s: float = _float("ASR_TIMEOUT_S", 60.0)
    # Precomputed voice-clone prompts kept in memory (one per distinct voice).
    prompt_cache_size: int = _int("PROMPT_CACHE_SIZE", 64)
    # Precompute prompts for the built-in voices during startup.
    prewarm_voices: bool = _bool("PREWARM_VOICES", True)

    # --- Text sanitisation (optional) --------------------------------------
    # OpenAI-compatible LLM that rewrites digits/abbreviations/symbols into
    # spoken words before synthesis. Empty base URL disables it.
    sanitize_llm_base_url: str = _str("SANITIZE_LLM_BASE_URL", "")
    sanitize_llm_api_key: str = _str("SANITIZE_LLM_API_KEY", "none")
    sanitize_llm_model: str = _str("SANITIZE_LLM_MODEL", "gpt-4o-mini")
    sanitize_llm_timeout_s: float = _float("SANITIZE_LLM_TIMEOUT_S", 10.0)

    # --- Encoding ----------------------------------------------------------
    # ffmpeg forks for mp3/opus/aac (~50-95 ms per chunk). On a fast GPU the
    # encoder, not the model, becomes the bottleneck; 0 = min(8, cpu_count).
    encode_workers: int = _int("ENCODE_WORKERS", 0)

    # --- HTTP --------------------------------------------------------------
    host: str = _str("HOST", "0.0.0.0")
    port: int = _int("PORT", 9001)
    cors_origins: list[str] = field(default_factory=lambda: _csv("CORS_ORIGINS") or ["*"])
    # Built frontend to serve; empty = no static serving (Vite handles the UI).
    static_dir: str = _str("STATIC_DIR", "")

    # --- Observability -----------------------------------------------------
    metrics_enabled: bool = _bool("METRICS_ENABLED", True)
    log_level: str = _str("LOG_LEVEL", "info")
    logfire_token: str = _str("LOGFIRE_TOKEN", "")

    # --- Startup -----------------------------------------------------------
    # Run a few inferences after loading so the first real request is not slow.
    warmup: bool = _bool("WARMUP", True)

    @property
    def resolved_first_chunk_steps(self) -> int:
        """Steps for the first streamed chunk (falls back to ``num_steps``)."""
        if self.first_chunk_steps <= 0:
            return self.num_steps
        return min(self.first_chunk_steps, self.num_steps)

    @property
    def device_short(self) -> str:
        return self.device.split(":")[0] if self.device else "cpu"


settings = Settings()
