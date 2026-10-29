"""The model: loading, voice prompts, synthesis and reference transcription.

One :class:`TTSEngine` per process. Every CUDA call goes through a single-worker
executor (``engine.executor``): a 3090 is already saturated by one request (see
docs/benchmarks.md), so extra threads only add context switching, and serialising
calls keeps the model's mutable state (dtype, ``eval()``) consistent.

The expensive, request-independent part of voice cloning — encoding the
reference clip into audio tokens — is cached per voice in :class:`PromptCache`,
which removes ~60 ms and one whisper call from every cloned sentence.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import metrics
from .audio import SAMPLE_RATE
from .config import settings

log = logging.getLogger(__name__)

# ``omnivoice.models.omnivoice.VoiceClonePrompt`` — aliased to avoid importing the
# model package (and torch's CUDA context) at module import time.
VoicePrompt = Any

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "auto": None,
}


class EngineError(RuntimeError):
    """Model-level failure (as opposed to a bad request)."""


class NotReadyError(EngineError):
    """The model is still loading or shutting down."""


@dataclass(frozen=True)
class SynthParams:
    """Everything that influences one model call, besides the text itself."""

    language: str
    num_step: int
    guidance_scale: float
    speed: float | None = None
    instruct: str | None = None
    # Cache key and the resolved object: holding the object for the lifetime of
    # the request keeps it from being evicted between scheduling and execution.
    prompt_key: str | None = None
    prompt: Any = None


@dataclass
class PromptEntry:
    prompt: Any
    language_hint: str | None
    # Reference length in audio frames, used to size a batch before running it.
    tokens: int = 0


class PromptCache:
    """Bounded LRU of ``VoiceClonePrompt`` objects keyed by voice id/content hash."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._items: OrderedDict[str, PromptEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> PromptEntry | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                metrics.prompt_cache_misses.inc()
                return None
            self._items.move_to_end(key)
            metrics.prompt_cache_hits.inc()
            return entry

    def peek(self, key: str) -> PromptEntry | None:
        """Look up without touching hit/miss counters (used for batch sizing)."""
        with self._lock:
            return self._items.get(key)

    def put(self, key: str, entry: PromptEntry) -> None:
        with self._lock:
            self._items[key] = entry
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._items)


class TTSEngine:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-gpu")
        self.model: Any = None
        self.device = settings.device
        self.dtype: torch.dtype | None = None
        self.sample_rate = SAMPLE_RATE
        self.prompts = PromptCache(settings.prompt_cache_size)
        self._ready = threading.Event()
        self._load_seconds = 0.0
        # Set when CUDA graphs are enabled (see app/graphs.py).
        self.graphs: Any = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self.model is not None

    def _resolve_dtype(self) -> torch.dtype | None:
        """``auto``: fp16 on CUDA, fp32 on CPU.

        Measured on the reference 3090 (both stock and gpu-unlock): fp16 is ~7 %
        faster end-to-end than bf16 even though its forward pass is marginally
        slower, because the decode loop casts logits to fp32 every step and
        bf16->fp32 casts cost more. Upstream's own CLI also uses fp16. Set
        ``DTYPE=bfloat16`` explicitly if you hit fp16 overflow on a fine-tune.
        """
        if settings.dtype not in _DTYPES:
            log.warning("unknown DTYPE=%r; falling back to float16", settings.dtype)
            return torch.float16
        dtype = _DTYPES[settings.dtype]
        if dtype is not None:
            return dtype
        return torch.float16 if self.device.startswith("cuda") else torch.float32

    def load(self) -> None:
        """Load the model. Must run on the GPU worker thread."""
        from omnivoice import OmniVoice

        started = time.perf_counter()
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise EngineError("CUDA is not available but DEVICE requests it")
            torch.cuda.set_device(torch.device(self.device))
        torch.set_float32_matmul_precision("high")
        dtype = self._resolve_dtype()
        log.info("loading %s on %s (dtype=%s)", settings.model_name, self.device, dtype)
        self.model = OmniVoice.from_pretrained(
            settings.model_name, device_map=self.device, dtype=dtype
        )
        self.model.eval()
        self.dtype = next(self.model.parameters()).dtype
        self.sample_rate = int(getattr(self.model, "sampling_rate", SAMPLE_RATE) or SAMPLE_RATE)
        self._load_seconds = time.perf_counter() - started
        self.compile_seconds = 0.0
        self.compile_mode: str | None = None
        if settings.torch_compile and self.device.startswith("cuda"):
            self._compile_backbone()
        if settings.cuda_graphs and self.device.startswith("cuda"):
            self._install_graphs()
        self._ready.set()
        metrics.model_ready.set(1)
        self._report_memory("loaded")
        log.info(
            "model ready in %.1fs (dtype=%s, sample_rate=%d)",
            self._load_seconds, self.dtype, self.sample_rate,
        )

    # Modes measured slower than "default" on OmniVoice (docs/benchmarks.md):
    # "max-autotune" 1.00x (its CUDA-graph trees fall back), "reduce-overhead"
    # 0.08x on the forward pass. Autotuning buys ~0 % for ~4x the compile time.
    SLOW_COMPILE_MODES = {"max-autotune", "reduce-overhead"}

    def _compile_backbone(self) -> None:
        """torch.compile the LLM backbone; keep serving eagerly if it fails."""
        started = time.perf_counter()
        if settings.compile_mode in self.SLOW_COMPILE_MODES:
            log.warning(
                "COMPILE_MODE=%s measured slower than 'default' on this model "
                "(see docs/benchmarks.md#torchcompile-on-the-backbone-the-biggest-single-win)",
                settings.compile_mode,
            )
        try:
            self.model.llm = torch.compile(
                self.model.llm, mode=settings.compile_mode, dynamic=True, fullgraph=False
            )
            self.compile_mode = settings.compile_mode
            log.info(
                "torch.compile(%s) attached to the backbone; first inference compiles",
                settings.compile_mode,
            )
        except Exception as exc:  # pragma: no cover - depends on the torch build
            log.warning("torch.compile unavailable (%s); serving eagerly", exc)
            self.compile_mode = None
        self.compile_seconds = time.perf_counter() - started

    def _install_graphs(self) -> None:
        """Wrap the model's forward pass in a CUDA-graph replayer."""
        from .graphs import GraphedForward

        self.graphs = GraphedForward(
            self.model,
            bucket=settings.graph_bucket,
            max_graphs=settings.graph_max,
            min_free_mb=settings.graph_min_free_mb,
        )
        self.model.forward = self.graphs
        log.info(
            "CUDA graphs enabled (bucket=%d, max=%d, min_free_mb=%d)",
            settings.graph_bucket, settings.graph_max, settings.graph_min_free_mb,
        )

    def shutdown(self) -> None:
        self._ready.clear()
        metrics.model_ready.set(0)
        self.executor.shutdown(wait=False, cancel_futures=True)

    def warmup(self) -> None:
        """Run representative calls so the first user does not pay for them."""
        started = time.perf_counter()
        try:
            warmups = (
                ("Hello, this is a warm up.", SynthParams(
                    language="en", num_step=settings.first_chunk_steps,
                    guidance_scale=settings.guidance_scale)),
                ("Hello, this is a longer warm up sentence so the compiled graph "
                 "sees a second sequence length as well.", SynthParams(
                     language="en", num_step=settings.num_steps,
                     guidance_scale=settings.guidance_scale)),
                ("Warm up in a designed voice.", SynthParams(
                    language="en", num_step=settings.num_steps,
                    guidance_scale=settings.guidance_scale, instruct="female")),
            )
            for text, params in warmups:
                self._generate(text, params)
            log.info("warm-up complete in %.1fs", time.perf_counter() - started)
        except Exception as exc:  # pragma: no cover - warm-up must never kill startup
            log.warning("warm-up failed: %s", exc)
        if self.graphs is not None:
            # Typical sequence lengths: ~110 tokens of reference + text + 50-150
            # target frames, i.e. 190-420 tokens. Capturing up front keeps the
            # first request per length from paying ~40 ms.
            ladder = range(
                settings.graph_bucket * 6, settings.graph_bucket * 16, settings.graph_bucket
            )
            captured = self.graphs.prewarm(ladder)
            log.info(
                "CUDA graphs ready: %d pre-captured, buckets %s",
                captured, self.graphs.stats()["buckets"],
            )
        self._report_memory("warm")

    # -- prompts -----------------------------------------------------------
    def create_prompt(self, samples: np.ndarray, ref_text: str | None) -> Any:
        """Encode a reference clip into a reusable voice-clone prompt."""
        if self.model is None:
            raise NotReadyError("model is not loaded")
        return self.model.create_voice_clone_prompt(
            ref_audio=(np.asarray(samples, dtype=np.float32), self.sample_rate),
            ref_text=ref_text,
            preprocess_prompt=True,
        )

    # -- synthesis ---------------------------------------------------------
    def synthesize_batch(self, items: list[tuple[str, SynthParams]]) -> list[np.ndarray]:
        """Synthesise several chunks in one model call. Runs on the GPU thread.

        Items must share ``num_step`` and be homogeneous in "has a clone prompt"
        (the model raises on a mixed call). Everything else — language, voice
        prompt, instruct string, speed — is per item.
        """
        if not self.ready:
            raise NotReadyError("model is not ready")
        if len(items) == 1:
            return [self.synthesize(*items[0])]
        if len({params.num_step for _text, params in items}) != 1:
            raise EngineError("cannot batch chunks with different num_step")

        started = time.perf_counter()
        audio, elapsed = self._generate_batch(items)
        per_item = elapsed / len(items)
        kind = "first" if items[0][1].num_step == settings.first_chunk_steps else "body"
        for samples, (_text, params) in zip(audio, items, strict=True):
            metrics.inference_seconds.labels(kind=kind, language=params.language).observe(per_item)
            metrics.chunks_total.inc()
            if len(samples):
                seconds = len(samples) / self.sample_rate
                metrics.audio_seconds_total.labels(language=params.language).inc(seconds)
                if per_item > 0:
                    metrics.realtime_factor.labels(kind=kind).observe(seconds / per_item)
        metrics.batch_size.observe(len(items))
        metrics.batches_total.inc()
        log.debug(
            "batch of %d chunks in %.0f ms (%.0f ms/item)",
            len(items), (time.perf_counter() - started) * 1000, per_item * 1000,
        )
        return audio

    def _generate_batch(self, items: list[tuple[str, SynthParams]]) -> tuple[list[np.ndarray], float]:
        first = items[0][1]
        kwargs: dict[str, Any] = {
            "text": [text for text, _p in items],
            "language": [params.language for _t, params in items],
            "num_step": first.num_step,
            "guidance_scale": first.guidance_scale,
            "t_shift": settings.t_shift,
            "postprocess_output": settings.postprocess_output,
        }
        prompts = [params.prompt for _t, params in items]
        if any(p is not None for p in prompts):
            kwargs["voice_clone_prompt"] = prompts
        instructs = [params.instruct or None for _t, params in items]
        if any(instructs):
            kwargs["instruct"] = instructs
        speeds = [params.speed for _t, params in items]
        if any(s is not None and s != 1.0 for s in speeds):
            kwargs["speed"] = speeds
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                out = self.model.generate(**kwargs)
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - defensive
            torch.cuda.empty_cache()
            raise EngineError("GPU out of memory while synthesising this batch") from exc
        elapsed = time.perf_counter() - started
        return [np.asarray(a, dtype=np.float32) for a in out], elapsed

    def synthesize(self, text: str, params: SynthParams) -> np.ndarray:
        """Synthesise one chunk. Runs on the GPU worker thread."""
        if not self.ready:
            raise NotReadyError("model is not ready")
        started = time.perf_counter()
        audio = self._generate(text, params)
        elapsed = time.perf_counter() - started
        kind = "first" if params.num_step == settings.first_chunk_steps else "body"
        metrics.inference_seconds.labels(kind=kind, language=params.language).observe(elapsed)
        metrics.chunks_total.inc()
        if len(audio):
            seconds = len(audio) / self.sample_rate
            metrics.audio_seconds_total.labels(language=params.language).inc(seconds)
            if elapsed > 0:
                metrics.realtime_factor.labels(kind=kind).observe(seconds / elapsed)
        return audio

    def _generate(self, text: str, params: SynthParams) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "language": params.language,
            "num_step": params.num_step,
            "guidance_scale": params.guidance_scale,
            "t_shift": settings.t_shift,
            "postprocess_output": settings.postprocess_output,
        }
        if params.speed is not None and params.speed != 1.0:
            kwargs["speed"] = params.speed
        if params.instruct:
            kwargs["instruct"] = params.instruct
        prompt = params.prompt
        if prompt is None and params.prompt_key:
            entry = self.prompts.get(params.prompt_key)
            prompt = entry.prompt if entry is not None else None
        if params.prompt_key and prompt is None:
            raise EngineError("voice prompt is no longer cached; retry the request")
        if prompt is not None:
            kwargs["voice_clone_prompt"] = prompt
        try:
            with torch.inference_mode():
                audio = self.model.generate(text=text, **kwargs)[0]
        except ValueError:
            # The model validates ``instruct`` and raises on unknown descriptors.
            raise
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - defensive
            torch.cuda.empty_cache()
            raise EngineError("GPU out of memory while synthesising this chunk") from exc
        return np.asarray(audio, dtype=np.float32)

    # -- introspection -----------------------------------------------------
    def _report_memory(self, kind: str) -> None:
        if not self.device.startswith("cuda"):
            return
        with contextlib.suppress(Exception):  # metrics must never break synthesis
            metrics.gpu_memory_bytes.labels(kind=kind).set(torch.cuda.memory_allocated())

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", "") if self.dtype else None,
            "model": settings.model_name,
            "sample_rate": self.sample_rate,
            "load_seconds": round(self._load_seconds, 2),
            "prompts_cached": len(self.prompts),
            "cuda_graphs": self.graphs.stats() if self.graphs is not None else None,
            "torch_compile": self.compile_mode,
        }
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                out["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
                out["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 1)
            except Exception:  # pragma: no cover
                pass
        return out


engine = TTSEngine()
