"""Request pipeline: text → scheduled chunks → encoded audio frames.

Shared by the HTTP and WebSocket entry points so both get the same chunking,
step selection and encoding behaviour. Encoding of ``mp3``/``opus``/``aac``
forks ffmpeg (~60 ms for a sentence), so it runs on a small CPU pool instead of
the event loop: while the encoder works, the GPU is already producing the next
chunk.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import audio, metrics
from .config import settings
from .engine import SynthParams
from .sanitize import sanitizer
from .scheduler import Scheduler, SchedulerError, Session

log = logging.getLogger(__name__)

# ffmpeg work runs here so it never blocks the event loop. Sized from CPU count:
# at a few hundred audio-seconds/s the encoders, not the GPU, are the limit.
_ENCODE_WORKERS = settings.encode_workers or min(8, (os.cpu_count() or 4))
CPU_POOL = ThreadPoolExecutor(max_workers=_ENCODE_WORKERS, thread_name_prefix="tts-encode")


@dataclass(frozen=True)
class RequestParams:
    """Resolved synthesis options for one client request."""

    language: str
    prompt_key: str | None = None
    prompt: object | None = None
    instruct: str | None = None
    speed: float | None = None
    num_step: int = settings.num_steps
    first_step: int = settings.resolved_first_chunk_steps
    guidance_scale: float = settings.guidance_scale
    # Rewrite digits/abbreviations into spoken words first (needs SANITIZE_LLM_*).
    sanitize: bool = True

    def synth_params(self, *, first: bool, streaming: bool) -> SynthParams:
        return SynthParams(
            language=self.language,
            num_step=self.first_step if (streaming and first) else self.num_step,
            guidance_scale=self.guidance_scale,
            speed=self.speed,
            instruct=self.instruct,
            prompt_key=self.prompt_key,
            prompt=self.prompt,
        )


async def encode(samples: np.ndarray, fmt: str) -> bytes:
    """Encode one chunk, keeping heavy codecs off the event loop."""
    started = time.perf_counter()
    if audio.needs_thread(fmt):
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(CPU_POOL, audio.encode, samples, fmt)
    else:
        data = audio.encode(samples, fmt)
    metrics.encode_seconds.labels(format=fmt).observe(time.perf_counter() - started)
    return data


async def clean(text: str, params: RequestParams) -> str:
    """Sanitise one chunk, if the request asked for it and a model is configured."""
    if not params.sanitize or not sanitizer.enabled:
        return text
    return await sanitizer.sanitize(text, language=params.language)


async def iter_chunks(
    scheduler: Scheduler,
    session: Session,
    texts: list[str],
    params: RequestParams,
    *,
    streaming: bool,
) -> AsyncIterator[np.ndarray]:
    """Yield synthesised chunks in text order, keeping a small pipeline full.

    Up to ``MAX_INFLIGHT_PER_SESSION`` chunks of this request may be queued at
    once. More would let one long request crowd the queue (hurting other users'
    time-to-first-audio) and hold more audio in RAM than is useful.
    """
    if not texts:
        return
    window = max(1, settings.max_inflight_per_session)
    pending: asyncio.Queue[asyncio.Future] = asyncio.Queue()
    iterator = iter(texts)
    first = True
    try:
        for _ in range(window):
            text = next(iterator, None)
            if text is None:
                break
            pending.put_nowait(
                scheduler.submit(
                    session, await clean(text, params),
                    params.synth_params(first=first, streaming=streaming),
                    first=first, speed=params.speed or 1.0,
                )
            )
            first = False
        while not pending.empty():
            future = await pending.get()
            samples = await future
            yield samples
            text = next(iterator, None)
            if text is not None:
                pending.put_nowait(
                    scheduler.submit(
                        session, await clean(text, params),
                        params.synth_params(first=False, streaming=streaming),
                        first=False, speed=params.speed or 1.0,
                    )
                )
    finally:
        while not pending.empty():
            future = pending.get_nowait()
            if not future.done():
                future.cancel()


def concat(chunks: list[np.ndarray]) -> np.ndarray:
    """Join chunk waveforms, with a short gap so sentence edges do not collide."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    gap = np.zeros(int(audio.SAMPLE_RATE * 0.04), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        if i:
            out.append(gap)
        out.append(chunk)
    return np.concatenate(out)


def sse(event: dict) -> bytes:
    import json

    return f"data: {json.dumps(event)}\n\n".encode()


__all__ = [
    "CPU_POOL", "RequestParams", "concat", "encode", "iter_chunks", "sse", "SchedulerError",
]
