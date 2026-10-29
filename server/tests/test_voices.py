"""Reference preparation and the voice store."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app import voices as voices_mod
from app.asr import Word
from app.audio import SAMPLE_RATE
from app.config import settings
from app.voices import (
    VoiceError,
    VoiceRegistry,
    VoiceStore,
    _pick_reference_window,
    _words_per_second,
)


def words(*spec: tuple[str, float, float]) -> list[Word]:
    return [Word(text=t, start=s, end=e) for t, s, e in spec]


class FakeASR:
    """Deterministic ASR stub: text is provided per test, timings are synthetic."""

    def __init__(self, words_list=None, text="", available=True) -> None:
        self._words = words_list or []
        self._text = text
        self.enabled = True
        self.available = available
        self.calls = 0

    def check(self) -> bool:
        return self.available

    def transcribe_words(self, samples):
        self.calls += 1
        return list(self._words)

    def transcribe_text(self, samples):
        self.calls += 1
        return self._text

    def close(self):
        pass


class FakeEngine:
    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prompts = {}
        self.created = []

    def create_prompt(self, samples, ref_text):
        self.created.append((len(samples), ref_text))
        return object()

    def synthesize(self, text, params):  # pragma: no cover - not used here
        raise NotImplementedError

    def shutdown(self):
        self.executor.shutdown(wait=True)


@pytest.fixture
def store(tmp_path):
    return VoiceStore(tmp_path)


@pytest.fixture
def registry(store):
    from app.engine import PromptCache

    engine = FakeEngine()
    engine.prompts = PromptCache(4)
    reg = VoiceRegistry(store, engine, FakeASR())
    yield reg
    engine.shutdown()


# ── window selection ────────────────────────────────────────────────────────


def test_window_uses_the_last_sentence_end_that_fits():
    ws = words(("Hello", 0.0, 0.4), ("there.", 0.4, 0.9), ("Second", 1.2, 1.6), ("one.", 1.6, 2.1))
    start, end, text = _pick_reference_window(ws, max_seconds=8, min_seconds=0.5)
    assert text == "Hello there. Second one."
    assert end == round((2.1 + 0.15) * SAMPLE_RATE)


def test_window_never_includes_a_word_past_max_seconds():
    ws = words(("Hello", 0.0, 0.4), ("there.", 0.4, 0.9), ("Much", 6.0, 6.5), ("later.", 6.5, 9.0))
    _, end, text = _pick_reference_window(ws, max_seconds=8, min_seconds=0.5)
    assert text == "Hello there."  # "much" has no terminal mark, "later." is past 8 s
    assert end == round((0.9 + 0.15) * SAMPLE_RATE)


def test_window_falls_back_to_last_word_when_no_punctuation():
    ws = words(("one", 0.0, 1.0), ("two", 1.0, 2.0), ("three", 2.0, 3.0))
    _, end, text = _pick_reference_window(ws, max_seconds=8, min_seconds=1)
    assert text == "one two three."  # punctuation added so the model stops cleanly
    assert end == round(3.15 * SAMPLE_RATE)


def test_window_respects_max_seconds():
    ws = words(("a", 0.0, 2.0), ("b.", 2.0, 4.0), ("c.", 4.0, 12.0))
    start, end, text = _pick_reference_window(ws, max_seconds=5, min_seconds=1)
    assert text == "a b."
    assert start == 0


def test_window_starts_at_first_word():
    # A clipped recording: the ASR starts at 0.6 s, the file should too.
    ws = words(("sa", 0.6, 1.0), ("nás", 1.0, 1.4), ("volá.", 1.4, 2.0))
    start, _, _ = _pick_reference_window(ws, max_seconds=8, min_seconds=1)
    assert start == round(0.55 * SAMPLE_RATE)


def test_window_none_when_nothing_fits():
    ws = words(("later", 9.0, 10.0))
    assert _pick_reference_window(ws, max_seconds=8, min_seconds=1) is None


def test_words_per_second():
    assert _words_per_second("one two three four", 2.0) == 2.0
    assert _words_per_second("", 0.0) == 0.0


# ── preparation ─────────────────────────────────────────────────────────────


def test_prepare_rejects_short_reference(registry):
    with pytest.raises(VoiceError):
        registry.prepare_reference(np.zeros(int(0.5 * SAMPLE_RATE), np.float32), "en")


def test_prepare_uses_provided_transcript(registry):
    samples = np.ones(int(3 * SAMPLE_RATE), np.float32) * 0.1
    prepared = registry.prepare_reference(samples, "en", "one two three four five")
    assert prepared.text == "one two three four five"
    assert prepared.provenance == "provided"


def test_prepare_repairs_misaligned_transcript(registry, monkeypatch):
    samples = np.ones(int(5 * SAMPLE_RATE), np.float32) * 0.1
    registry.asr = FakeASR(text="a proper transcript of the clip")
    prepared = registry.prepare_reference(samples, "en", "one")
    assert prepared.provenance == "asr-repaired"
    assert prepared.text.startswith("a proper transcript")


def test_prepare_uses_asr_when_no_transcript(registry):
    samples = np.ones(int(3 * SAMPLE_RATE), np.float32) * 0.1
    registry.asr = FakeASR(
        words_list=words(("Hello", 0.0, 0.5), ("world.", 0.5, 1.2)), text="Hello world."
    )
    prepared = registry.prepare_reference(samples, "en", None)
    assert prepared.provenance == "asr"
    assert prepared.text == "Hello world."


def test_prepare_requires_transcript_without_asr(registry):
    samples = np.ones(int(9 * SAMPLE_RATE), np.float32) * 0.1
    registry.asr = FakeASR(available=False)
    with pytest.raises(VoiceError):
        registry.prepare_reference(samples, "en", None)


# ── store ───────────────────────────────────────────────────────────────────


def test_store_save_list_get_delete(store):
    samples = np.ones(int(3 * SAMPLE_RATE), np.float32) * 0.1
    voice = store.save("Test Voice", samples, "hello there.", "en")
    assert voice.id == "Test_Voice"
    assert voice.language == "en"
    assert voice.ref_text == "hello there."
    assert [v.id for v in store.list()] == ["Test_Voice"]

    meta = json.loads((store.root / "Test_Voice" / "meta.json").read_text())
    assert meta["duration"] == pytest.approx(3.0, abs=0.05)

    store.delete("Test_Voice")
    assert store.list() == []
    with pytest.raises(VoiceError):
        store.get("Test_Voice")


def test_store_rejects_bad_ids(store):
    with pytest.raises(VoiceError):
        store.save("../evil", np.zeros(10, np.float32), "x", "en")
    with pytest.raises(VoiceError):
        store.get("../../etc/passwd")


def test_prompt_key_changes_with_content(store):
    samples = np.ones(int(2 * SAMPLE_RATE), np.float32) * 0.1
    voice = store.save("v1", samples, "hello.", "en")
    key1 = store.prompt_key(voice)
    store.save("v1", samples * 0.5, "hello.", "en")
    assert store.prompt_key(store.get("v1")) != key1


def test_language_change_removes_stale_transcript(store):
    samples = np.ones(int(2 * SAMPLE_RATE), np.float32) * 0.1
    store.save("v2", samples, "ahoj.", "sk")
    store.save("v2", samples, "hello.", "en")
    files = {p.suffix for p in (store.root / "v2").iterdir() if p.suffix == ".txt"}
    assert files == {".txt"}
    assert (store.root / "v2" / "en.txt").is_file()
    assert not (store.root / "v2" / "sk.txt").exists()
    assert store.get("v2").language == "en"


async def test_registry_add_and_resolve(store, registry):
    samples = np.ones(int(3 * SAMPLE_RATE), np.float32) * 0.1
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV")
    registry.asr = FakeASR(words_list=words(("one", 0.0, 0.5), ("two.", 0.5, 1.0)), text="one two.")
    voice = await registry.add_voice("Speaker", buf.getvalue(), "ref.wav", None, "en")
    assert voice.id == "Speaker"
    key, prompt = await registry.resolve_voice("Speaker")
    assert key.startswith("voice:Speaker:")
    assert prompt is not None
    # Second resolve hits the cache instead of rebuilding the prompt.
    assert (await registry.resolve_voice("Speaker"))[0] == key
    registry.delete_voice("Speaker")
    assert store.list() == []


def test_settings_defaults_are_sane():
    assert settings.max_ref_seconds >= settings.min_ref_seconds
    assert settings.first_chunk_steps <= settings.num_steps
    assert voices_mod.SAMPLE_RATE == SAMPLE_RATE
