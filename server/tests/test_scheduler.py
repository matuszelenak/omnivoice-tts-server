"""Scheduler behaviour: fairness, first-chunk priority, admission, cancellation."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from app import scheduler as scheduler_mod
from app.config import settings
from app.engine import SynthParams
from app.scheduler import Job, NotReady, QueueFullError, Scheduler


class FakeEngine:
    """Records which chunks ran, in order, with a controllable duration."""

    def __init__(self, delay: float = 0.01, fail_on: set[str] | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on or set()
        self.ready = True
        self.executor = None  # replaced by the scheduler test helper
        self.order: list[str] = []
        self.batches: list[list[str]] = []

    def synthesize(self, text: str, params: SynthParams) -> np.ndarray:
        import time

        time.sleep(self.delay)
        if text in self.fail_on:
            raise RuntimeError(f"boom: {text}")
        self.order.append(text)
        return np.zeros(2400, dtype=np.float32)

    def synthesize_batch(self, items):
        self.batches.append([text for text, _params in items])
        return [self.synthesize(text, params) for text, params in items]


def params(language: str = "en", num_step: int | None = None) -> SynthParams:
    return SynthParams(
        language=language,
        num_step=num_step or settings.num_steps,
        guidance_scale=2.0,
    )


@pytest.fixture
async def sched(monkeypatch):
    """Scheduler with batching off, so ordering tests are about fairness only."""
    from concurrent.futures import ThreadPoolExecutor

    engine = FakeEngine()
    engine.executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(
        scheduler_mod, "settings", replace(settings, batch_max_size=1, batch_window_ms=0)
    )
    s = Scheduler(engine)  # type: ignore[arg-type]
    await s.start()
    yield s, engine
    await s.stop()
    engine.executor.shutdown(wait=True)


@pytest.fixture
async def sched_batching(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    engine = FakeEngine()
    engine.executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(
        scheduler_mod,
        "settings",
        replace(settings, batch_max_size=8, batch_window_ms=0, batch_max_tokens=8192),
    )
    s = Scheduler(engine)  # type: ignore[arg-type]
    await s.start()
    yield s, engine
    await s.stop()
    engine.executor.shutdown(wait=True)


async def test_round_robin_interleaves_sessions(sched):
    s, engine = sched
    a = s.new_session("http")
    b = s.new_session("http")
    futures = []
    for i in range(3):
        futures.append(("a", s.submit(a, f"a{i}", params(), first=(i == 0))))
        futures.append(("b", s.submit(b, f"b{i}", params(), first=(i == 0))))
    await asyncio.gather(*(f for _, f in futures))
    # Both first chunks run before either session gets its second chunk, and the
    # follow-ups alternate.
    assert set(engine.order[:2]) == {"a0", "b0"}
    assert engine.order[2:4] in (["a1", "b1"], ["b1", "a1"])
    assert s.pending == 0


async def test_first_chunks_jump_ahead_of_a_long_queue(sched):
    s, engine = sched
    long_session = s.new_session("http")
    for i in range(5):
        s.submit(long_session, f"long{i}", params(), first=(i == 0))
    newcomer = s.new_session("http")
    s.submit(newcomer, "newcomer0", params(), first=True)
    while s.pending or s.inflight:
        await asyncio.sleep(0.01)
    # The new request's first chunk is served before the long session's tail.
    assert engine.order.index("newcomer0") < engine.order.index("long4")


async def test_result_is_returned_to_the_right_caller(sched):
    s, _ = sched
    a = s.new_session("http")
    b = s.new_session("http")
    fa = s.submit(a, "alpha", params(), first=True)
    fb = s.submit(b, "beta", params(), first=True)
    ra, rb = await asyncio.gather(fa, fb)
    assert ra.shape == rb.shape == (2400,)


async def test_admission_rejects_when_queue_is_full(sched, monkeypatch):
    s, engine = sched
    engine.delay = 0.2
    monkeypatch.setattr(scheduler_mod, "settings", replace(settings, max_queue_jobs=2))
    session = s.new_session("http")
    s.submit(session, "one", params(), first=True)
    s.submit(session, "two", params())
    with pytest.raises(QueueFullError):
        s.submit(session, "three", params())
    s.cancel_session(session)


async def test_admission_rejects_when_pending_audio_is_full(sched, monkeypatch):
    s, engine = sched
    engine.delay = 0.2
    monkeypatch.setattr(scheduler_mod, "settings", replace(settings, max_pending_audio_s=10.0))
    session = s.new_session("http")
    long_text = "this is a moderately long sentence " * 3
    s.submit(session, long_text, params(), first=True)
    with pytest.raises(QueueFullError):
        s.submit(session, long_text, params())
    s.cancel_session(session)


async def test_cancel_drops_queued_jobs(sched):
    s, engine = sched
    engine.delay = 0.05
    session = s.new_session("http")
    futures = [s.submit(session, f"job{i}", params(), first=(i == 0)) for i in range(6)]
    dropped = s.cancel_session(session, reason="test")
    assert dropped >= 1
    assert s.pending < 6
    await asyncio.sleep(0.2)
    assert sum(1 for f in futures if f.cancelled()) >= 1


async def test_failed_chunk_raises_to_caller(sched):
    s, engine = sched
    engine.fail_on = {"bad"}
    session = s.new_session("http")
    with pytest.raises(RuntimeError):
        await s.submit(session, "bad", params(), first=True)


async def test_not_ready_is_reported(sched):
    s, engine = sched
    engine.ready = False
    session = s.new_session("http")
    with pytest.raises(NotReady):
        s.submit(session, "hello", params(), first=True)


async def test_scheduler_stats(sched):
    s, _ = sched
    stats = s.stats()
    assert stats["capacity_chunks"] == scheduler_mod.settings.max_queue_jobs
    assert stats["pending_chunks"] == 0
    assert stats["max_batch_size"] == 1  # batching disabled by the fixture


async def test_burst_limit_lets_followups_through(sched, monkeypatch):
    """A flood of new requests must not freeze a session that is already playing."""
    s, engine = sched
    monkeypatch.setattr(
        scheduler_mod,
        "settings",
        replace(settings, first_chunk_burst=2, batch_max_size=1, batch_window_ms=0),
    )
    playing = s.new_session("http")
    s.submit(playing, "play0", params(), first=True)
    for i in range(4):
        s.submit(playing, f"play{i + 1}", params())
    for i in range(10):
        newcomer = s.new_session("http")
        s.submit(newcomer, f"new{i}", params(), first=True)
    while s.pending or s.inflight:
        await asyncio.sleep(0.01)
    # The playing session's follow-ups appear well before the last first chunk.
    last_play = max(engine.order.index(f"play{i}") for i in range(1, 5))
    assert last_play < len(engine.order) - 1


# ── batching ────────────────────────────────────────────────────────────────


async def test_batching_puts_queued_chunks_in_one_call(sched_batching):
    s, engine = sched_batching
    sessions = [s.new_session("http") for _ in range(4)]
    futures = [s.submit(session, f"text {i}", params(), first=True) for i, session in enumerate(sessions)]
    await asyncio.gather(*futures)
    assert len(engine.batches) == 1, engine.batches
    assert len(engine.batches[0]) == 4


async def test_batching_respects_max_size(sched_batching, monkeypatch):
    s, engine = sched_batching
    monkeypatch.setattr(
        scheduler_mod, "settings", replace(scheduler_mod.settings, batch_max_size=2)
    )
    sessions = [s.new_session("http") for _ in range(6)]
    futures = [s.submit(session, f"t{i}", params(), first=True) for i, session in enumerate(sessions)]
    await asyncio.gather(*futures)
    assert all(len(batch) <= 2 for batch in engine.batches)
    assert len(engine.batches) >= 3


async def test_batching_never_mixes_step_counts_or_voice_modes(sched_batching):
    s, engine = sched_batching
    a = s.new_session("http")
    b = s.new_session("http")
    c = s.new_session("http")
    with_prompt = replace_params(prompt=object())
    f1 = s.submit(a, "first", params(num_step=16), first=True)
    f2 = s.submit(b, "second", params(num_step=12), first=True)
    f3 = s.submit(c, "third", with_prompt, first=True)
    await asyncio.gather(f1, f2, f3)
    for batch in engine.batches:
        assert len(batch) == 1, f"incompatible chunks were batched: {batch}"


async def test_batching_isolates_a_failing_chunk(sched_batching):
    s, engine = sched_batching
    engine.fail_on = {"bad"}
    good_session = s.new_session("http")
    bad_session = s.new_session("http")
    f_good = s.submit(good_session, "good", params(), first=True)
    f_bad = s.submit(bad_session, "bad", params(), first=True)
    await asyncio.gather(f_good, f_bad, return_exceptions=True)
    assert (await f_good).shape == (2400,)
    with pytest.raises(RuntimeError):
        await f_bad


async def test_batching_keeps_per_session_order(sched_batching):
    s, engine = sched_batching
    session = s.new_session("http")
    futures = [s.submit(session, f"s{i}", params(), first=(i == 0)) for i in range(4)]
    await asyncio.gather(*futures)
    own = [t for batch in engine.batches for t in batch if t.startswith("s") and t[1:].isdigit()]
    assert own == ["s0", "s1", "s2", "s3"]


def test_batchable_rejects_mismatched_lengths(sched_batching):
    s, _ = sched_batching
    head = Job(text="x", params=params(), session=s.new_session("http"),
               future=asyncio.get_event_loop().create_future(), cost=1.0, first=True, tokens=100)
    same = Job(text="y", params=params(), session=s.new_session("http"),
               future=asyncio.get_event_loop().create_future(), cost=1.0, first=False, tokens=150)
    long = Job(text="z", params=params(), session=s.new_session("http"),
               future=asyncio.get_event_loop().create_future(), cost=1.0, first=False, tokens=400)
    assert s._batchable(head, same)
    assert not s._batchable(head, long)


def replace_params(**kwargs) -> SynthParams:
    return SynthParams(**{**params().__dict__, **kwargs})
