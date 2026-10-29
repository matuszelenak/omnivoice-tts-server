"""GPU job scheduler: one worker, fair queue, admission control.

Why not batching? Measured on an RTX 3090 the model is already FLOP-bound at
batch size 1: eight requests in one ``generate()`` call take 8x as long as one
(0.60 s vs 0.67 s per item), so batching buys ~10 % throughput and costs the
first request in the batch a full batch window of latency plus a padded (and
therefore slower) forward pass. See docs/benchmarks.md.

So the scheduling unit is a **sentence**, not a request:

* A long request is split into one job per sentence, and jobs from different
  requests are interleaved round-robin. Nobody waits for somebody else's 40
  sentences, and the GPU never idles between requests.
* The *first* job of each request is dispatched before any queued follow-up job
  (FIFO across requests). That is what keeps time-to-first-audio flat as load
  grows — the number new users wait behind is bounded, not the number of
  sentences already accepted.
* Admission control bounds both the number of queued jobs and the estimated
  seconds of audio queued. Beyond that the server answers 503 + ``Retry-After``
  instead of letting latency grow without bound.
* A cancelled request drops its queued jobs, so a client that navigates away
  does not keep the GPU busy.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from . import metrics
from .config import settings
from .engine import NotReadyError, SynthParams, TTSEngine
from .text import estimate_audio_seconds

log = logging.getLogger(__name__)

# Cost normalisation: one second of audio at NUM_STEPS costs ~1 unit.
_BASE_STEPS = 16


class SchedulerError(RuntimeError):
    """Base class for admission failures."""


class QueueFullError(SchedulerError):
    """Too much work queued; the caller should retry later."""


class QueueTimeoutError(SchedulerError):
    """The job waited longer than ``QUEUE_TIMEOUT_S``."""


class NotReady(SchedulerError):
    """The engine is loading or shutting down."""


@dataclass
class Session:
    """One client request (HTTP) or one live WebSocket connection."""

    id: int
    kind: str
    created_at: float = field(default_factory=time.monotonic)
    queue: deque[Job] = field(default_factory=deque)
    pending_cost: float = 0.0
    cancelled: bool = False
    first_pending: bool = False
    in_first: bool = False
    in_ring: bool = False

    @property
    def pending(self) -> int:
        return len(self.queue)


@dataclass
class Job:
    text: str
    params: SynthParams
    session: Session
    future: asyncio.Future
    cost: float
    first: bool
    # Estimated sequence length (reference + text + target frames): the padding
    # cost of batching two items is driven by the larger of the two.
    tokens: int = 1
    enqueued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None

    @property
    def deadline(self) -> float:
        return self.enqueued_at + settings.queue_timeout_s


class Scheduler:
    def __init__(self, engine: TTSEngine) -> None:
        self.engine = engine
        self._sessions: dict[int, Session] = {}
        self._first: deque[Session] = deque()
        self._ring: deque[Session] = deque()
        self._next_id = 0
        self._pending = 0
        self._pending_cost = 0.0
        self._inflight = 0
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closing = False
        self._started_at = time.monotonic()
        self._firsts_since_body = 0

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        self._closing = False
        self._wakeup = asyncio.Event()
        self._task = asyncio.create_task(self._dispatch_loop(), name="tts-dispatch")
        metrics.queue_capacity.set(settings.max_queue_jobs)

    async def stop(self) -> None:
        self._closing = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        for session in list(self._sessions.values()):
            self.cancel_session(session, reason="shutdown")
        metrics.queue_depth.set(0)
        metrics.inflight_jobs.set(0)

    # -- admission ---------------------------------------------------------
    def new_session(self, kind: str) -> Session:
        self._next_id += 1
        session = Session(id=self._next_id, kind=kind)
        self._sessions[session.id] = session
        return session

    def end_session(self, session: Session) -> None:
        self.cancel_session(session, reason="closed")
        self._sessions.pop(session.id, None)

    def cancel_session(self, session: Session, reason: str = "cancelled") -> int:
        """Drop every queued job of *session*; returns how many were dropped."""
        session.cancelled = True
        dropped = 0
        while session.queue:
            job = session.queue.popleft()
            self._pending = max(0, self._pending - 1)
            self._pending_cost = max(0.0, self._pending_cost - job.cost)
            session.pending_cost = max(0.0, session.pending_cost - job.cost)
            if not job.future.done():
                job.future.cancel()
            dropped += 1
        session.first_pending = False
        if dropped:
            log.debug("session %d: dropped %d queued job(s) (%s)", session.id, dropped, reason)
            metrics.rejected_total.labels(reason="cancelled").inc(dropped)
        metrics.queue_depth.set(self._pending)
        metrics.pending_audio_seconds.set(self._pending_cost)
        return dropped

    def _check_admission(self, cost: float) -> None:
        if self._closing:
            raise NotReady("server is shutting down")
        if not self.engine.ready:
            raise NotReady("model is not ready")
        if settings.max_queue_jobs and self._pending >= settings.max_queue_jobs:
            metrics.rejected_total.labels(reason="queue_full").inc()
            raise QueueFullError(
                f"queue is full ({self._pending} chunks waiting); retry shortly"
            )
        if (
            settings.max_pending_audio_s
            and self._pending_cost + cost > settings.max_pending_audio_s
        ):
            metrics.rejected_total.labels(reason="queue_full").inc()
            raise QueueFullError(
                "server is at capacity (queued audio limit); retry shortly"
            )

    # -- submission --------------------------------------------------------
    def check_admission(self, cost: float) -> None:
        """Raise :class:`QueueFullError`/``NotReady`` if *cost* would overbook."""
        self._check_admission(cost)

    def submit(
        self,
        session: Session,
        text: str,
        params: SynthParams,
        *,
        first: bool = False,
        speed: float = 1.0,
    ) -> asyncio.Future:
        """Queue one chunk. Returns a future resolved with float32 samples."""
        if session.cancelled:
            raise asyncio.CancelledError
        seconds = estimate_audio_seconds(text, speed)
        cost = seconds * (params.num_step / _BASE_STEPS)
        self._check_admission(cost)
        loop = asyncio.get_running_loop()
        job = Job(
            text=text, params=params, session=session,
            future=loop.create_future(), cost=cost, first=first,
            tokens=self._estimate_tokens(text, seconds, params),
        )
        session.queue.append(job)
        session.pending_cost += cost
        self._pending += 1
        self._pending_cost += cost
        if first:
            session.first_pending = True
            if not session.in_first:
                session.in_first = True
                self._first.append(session)
        elif not session.in_ring:
            session.in_ring = True
            self._ring.append(session)
        metrics.queue_depth.set(self._pending)
        self._wakeup.set()
        return job.future

    async def run(
        self,
        session: Session,
        text: str,
        params: SynthParams,
        *,
        first: bool = False,
        speed: float = 1.0,
    ):
        """Submit and await a chunk, translating admission failures."""
        future = self.submit(session, text, params, first=first, speed=speed)
        return await asyncio.shield(future)

    # -- batching ----------------------------------------------------------
    def _estimate_tokens(self, text: str, seconds: float, params: SynthParams) -> int:
        """Rough sequence length of one chunk, for batch padding/size decisions."""
        reference = 0
        cache = getattr(self.engine, "prompts", None)
        if params.prompt_key and cache is not None:
            entry = cache.peek(params.prompt_key)
            if entry is not None:
                reference = entry.tokens or 0
        text_tokens = max(1, len(text) // 4)
        target_frames = int(seconds * 25)  # the tokenizer's frame rate
        return reference + text_tokens + target_frames

    def _batchable(self, head: Job, job: Job) -> bool:
        """Can *job* share a model call with *head*?"""
        if job.params.num_step != head.params.num_step:
            return False
        # The model raises when some items have a clone prompt and others do not.
        if (job.params.prompt is None) != (head.params.prompt is None):
            return False
        shorter, longer = sorted((head.tokens, job.tokens))
        return not (shorter > 0 and longer / shorter > settings.batch_length_ratio)

    def _refresh(self, session: Session) -> None:
        """Keep the deques consistent after a job was taken out of a queue."""
        if session.cancelled or not session.queue:
            return
        if session.first_pending and not session.in_first:
            session.in_first = True
            self._first.append(session)
        pending_body = len(session.queue) - (1 if session.first_pending else 0)
        if pending_body > 0 and not session.in_ring:
            session.in_ring = True
            self._ring.append(session)

    def _take_head(self, session: Session) -> Job:
        job = session.queue.popleft()
        self._account_removed(session, job)
        self._refresh(session)
        return job

    def _candidates(self) -> list[Session]:
        """Sessions with a queued chunk, first-chunk sessions first, de-duplicated."""
        seen: set[int] = set()
        out: list[Session] = []
        for session in (*self._first, *self._ring):
            if id(session) in seen:
                continue
            seen.add(id(session))
            if session.cancelled or not session.queue:
                continue
            out.append(session)
        return out

    def _pop_batch(self) -> list[Job]:
        """Take the next fair job, plus any compatible jobs already queued.

        Batching only ever merges work that is *already waiting*: the alternative
        is running those chunks back to back on the same GPU, so merging them is
        never worse for the queue as a whole, and it is 1.2-1.3x cheaper per chunk
        because the tensor cores are fed a wider matrix.
        """
        head = self._pop_next()
        if head is None:
            return []
        limit = settings.batch_max_size
        if limit <= 1:
            return [head]

        batch = [head]
        widest = head.tokens
        for session in self._candidates():
            if len(batch) >= limit:
                break
            job = session.queue[0]
            if job.future.cancelled():
                self._take_head(session)
                continue
            if job.first and not settings.batch_first_chunks:
                continue
            widest_after = max(widest, job.tokens)
            if not self._batchable(head, job) or (len(batch) + 1) * widest_after > settings.batch_max_tokens:
                continue
            self._take_head(session)
            batch.append(job)
            widest = widest_after
        if len(batch) > 1:
            log.debug("batching %d chunks (tokens=%s)", len(batch), [j.tokens for j in batch])
        return batch

    # -- dispatch ----------------------------------------------------------
    def _pop_next(self) -> Job | None:
        # Fairness guard: after a burst of first chunks, let follow-up chunks
        # through so a flood of new requests cannot freeze sessions that are
        # already playing.
        burst = settings.first_chunk_burst
        if burst and self._firsts_since_body >= burst and self._ring:
            job = self._pop_body()
            if job is not None:
                self._firsts_since_body = 0
                return job
        # 1. First chunks of new requests, FIFO, so TTFA does not grow with load.
        while self._first:
            session = self._first.popleft()
            session.in_first = False
            if session.cancelled or not session.queue or not session.first_pending:
                continue
            job = session.queue.popleft()
            self._account_removed(session, job)
            if session.queue and not session.in_ring:
                session.in_ring = True
                self._ring.append(session)
            self._firsts_since_body += 1
            return job
        # 2. Round-robin over sessions with follow-up chunks.
        return self._pop_body()

    def _pop_body(self) -> Job | None:
        while self._ring:
            session = self._ring.popleft()
            session.in_ring = False
            if session.cancelled or not session.queue:
                continue
            if session.first_pending:
                # This session's first chunk has not run yet; it belongs to the
                # first-chunk queue, which is serviced above.
                if not session.in_first:
                    session.in_first = True
                    self._first.append(session)
                continue
            job = session.queue.popleft()
            self._account_removed(session, job)
            if session.queue:
                session.in_ring = True
                self._ring.append(session)
            return job
        return None

    def _account_removed(self, session: Session, job: Job) -> None:
        self._pending = max(0, self._pending - 1)
        self._pending_cost = max(0.0, self._pending_cost - job.cost)
        session.pending_cost = max(0.0, session.pending_cost - job.cost)
        metrics.queue_depth.set(self._pending)
        metrics.pending_audio_seconds.set(self._pending_cost)
        if job.first:
            session.first_pending = False

    def _expire(self) -> None:
        now = time.monotonic()
        for session in list(self._sessions.values()):
            if not session.queue:
                continue
            expired = [job for job in session.queue if job.deadline <= now]
            for job in expired:
                session.queue.remove(job)
                self._account_removed(session, job)
                metrics.rejected_total.labels(reason="queue_timeout").inc()
                if not job.future.done():
                    job.future.set_exception(
                        QueueTimeoutError(
                            f"request waited more than {settings.queue_timeout_s:.0f}s"
                        )
                    )

    async def _dispatch_loop(self) -> None:
        while not self._closing:
            try:
                self._expire()
                if self._pending == 0:
                    self._wakeup.clear()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._wakeup.wait(), timeout=0.5)
                    continue
                # Give a partner job a chance to arrive before decoding alone.
                # Under load the queue is already deep, so nothing is waited for.
                if settings.batch_max_size > 1 and settings.batch_window_ms > 0 and self._pending < 2:
                    await asyncio.sleep(settings.batch_window_ms / 1000.0)
                batch = [job for job in self._pop_batch() if not job.future.cancelled()]
                if not batch:
                    continue  # everything was cancelled while it queued
                await self._execute_batch(batch)
            except asyncio.CancelledError:
                return
            except Exception:  # pragma: no cover - the loop must never die
                log.exception("scheduler loop error")
                await asyncio.sleep(0.05)

    async def _execute_batch(self, batch: list[Job]) -> None:
        loop = asyncio.get_running_loop()
        now = time.monotonic()
        for job in batch:
            job.started_at = now
            metrics.queue_wait_seconds.observe(now - job.enqueued_at)
        self._inflight += len(batch)
        metrics.inflight_jobs.set(self._inflight)
        try:
            items = [(job.text, job.params) for job in batch]
            results = await loop.run_in_executor(
                self.engine.executor, self.engine.synthesize_batch, items
            )
            if len(results) != len(batch):
                raise RuntimeError(
                    f"model returned {len(results)} results for {len(batch)} chunks"
                )
            for job, samples in zip(batch, results, strict=True):
                if not job.future.done():
                    job.future.set_result(samples)
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            for job in batch:
                if not job.future.done():
                    job.future.cancel()
            raise
        except Exception as exc:
            if len(batch) == 1:
                metrics.inference_errors_total.inc()
                log.warning("synthesis failed (%d chars): %s", len(batch[0].text), exc)
                if not batch[0].future.done():
                    batch[0].future.set_exception(exc)
            else:
                # Isolate the failure: one bad chunk must not fail its neighbours.
                metrics.split_batches_total.inc()
                log.warning(
                    "batched synthesis failed (%s); retrying %d chunk(s) individually",
                    exc, len(batch),
                )
                await self._execute_individually(batch)
        finally:
            self._inflight = max(0, self._inflight - len(batch))
            metrics.inflight_jobs.set(self._inflight)

    async def _execute_individually(self, batch: list[Job]) -> None:
        loop = asyncio.get_running_loop()
        for job in batch:
            try:
                samples = await loop.run_in_executor(
                    self.engine.executor, self.engine.synthesize, job.text, job.params
                )
                if not job.future.done():
                    job.future.set_result(samples)
            except Exception as exc:
                metrics.inference_errors_total.inc()
                log.warning("synthesis failed (%d chars): %s", len(job.text), exc)
                if not job.future.done():
                    job.future.set_exception(exc)

    # -- introspection -----------------------------------------------------
    @property
    def pending(self) -> int:
        return self._pending

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def pending_audio_seconds(self) -> float:
        return self._pending_cost

    def stats(self) -> dict[str, Any]:
        return {
            "sessions": len(self._sessions),
            "pending_chunks": self._pending,
            "pending_audio_seconds": round(self._pending_cost, 1),
            "inflight": self._inflight,
            "capacity_chunks": settings.max_queue_jobs,
            "capacity_audio_seconds": settings.max_pending_audio_s,
            "max_batch_size": settings.batch_max_size,
            "batch_window_ms": settings.batch_window_ms,
            "uptime_seconds": round(time.monotonic() - self._started_at, 1),
        }


def is_not_ready_error(exc: BaseException) -> bool:
    return isinstance(exc, (NotReady, NotReadyError))
