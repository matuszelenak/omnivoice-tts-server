"""HTTP surface, with the model replaced by a stub (no GPU required)."""
from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as main_mod
from app.audio import SAMPLE_RATE
from app.engine import PromptCache


class StubEngine:
    """Stands in for TTSEngine: returns a deterministic beep per chunk."""

    def __init__(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.model = object()
        self.dtype = None
        self.sample_rate = SAMPLE_RATE
        self.prompts = PromptCache(8)
        self.calls: list[str] = []
        from threading import Event

        self._ready = Event()
        self._ready.set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def load(self) -> None:
        self._ready.set()

    def warmup(self) -> None:
        return None

    def shutdown(self) -> None:
        self._ready.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def synthesize_batch(self, items):
        return [self.synthesize(text, params) for text, params in items]

    def synthesize(self, text, params):
        self.calls.append(text)
        t = np.arange(int(0.4 * SAMPLE_RATE)) / SAMPLE_RATE
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def create_prompt(self, samples, ref_text):
        return object()

    def stats(self) -> dict:
        return {"device": "cpu", "dtype": "float16", "sample_rate": SAMPLE_RATE}


@pytest.fixture
def stub(monkeypatch):
    engine = StubEngine()
    monkeypatch.setattr(main_mod, "engine", engine)
    monkeypatch.setattr(main_mod.scheduler, "engine", engine)
    monkeypatch.setattr(main_mod.voices, "engine", engine)
    main_mod.asr._available = True
    yield engine
    engine.shutdown()


@pytest.fixture
def client(stub):
    with TestClient(main_mod.app) as test_client:
        yield test_client


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as handle:
        return handle.getnframes() / handle.getframerate()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["engine"]["sample_rate"] == SAMPLE_RATE


def test_models_listing_is_openai_shaped(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "omnivoice"
    assert client.get("/v1/models/omnivoice").status_code == 200
    assert client.get("/v1/models/nope").status_code == 404


def test_speech_full(client, stub):
    response = client.post(
        "/v1/audio/speech",
        json={"input": "Hello there. Second sentence.", "language": "en", "response_format": "wav"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["cache-control"] == "no-store"
    # One model call per sentence, with the auto voice (no voice, no instruct).
    assert len(stub.calls) == 2
    assert wav_seconds(response.content) == pytest.approx(0.4 * 2 + 0.04, abs=0.05)


def test_speech_rejects_empty_input(client):
    response = client.post("/v1/audio/speech", json={"input": "   ", "language": "en"})
    assert response.status_code == 422
    assert "error" in response.json()


def test_speech_rejects_unknown_language(client):
    response = client.post(
        "/v1/audio/speech", json={"input": "hi", "language": "klingon"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_speech_rejects_long_input(client, monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(main_mod, "settings", replace(main_mod.settings, max_input_chars=20))
    response = client.post(
        "/v1/audio/speech", json={"input": "x" * 50, "language": "en"}
    )
    assert response.status_code == 413


def test_speech_rejects_unknown_voice(client):
    response = client.post(
        "/v1/audio/speech", json={"input": "hi", "language": "en", "voice": "nope"}
    )
    assert response.status_code == 404


def test_speech_sse_stream(client):
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "input": "First sentence here. Second sentence here. Third sentence here.",
            "language": "en",
            "response_format": "wav",
            "stream_format": "sse",
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[6:])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    deltas = [e for e in events if e.get("type") == "speech.audio.delta"]
    assert len(deltas) == 3
    assert events[-1]["type"] == "speech.audio.done"
    for delta in deltas:
        assert wav_seconds(base64.b64decode(delta["audio"])) > 0.3


def test_speech_pcm_and_multiple_formats(client):
    for fmt, content_type in (("pcm", "audio/pcm"), ("wav", "audio/wav")):
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello.", "language": "en", "response_format": fmt},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == content_type


def test_languages(client):
    body = client.get("/v1/languages").json()
    assert any(item["id"] == "sk" for item in body)


def test_voices_empty_then_created(client, tmp_path):
    from app.voices import VoiceStore

    main_mod.store.root = tmp_path
    assert client.get("/v1/voices").json() == []

    # Register without ASR: an exact transcript is required.
    samples = (np.zeros(int(3 * SAMPLE_RATE), np.float32) + 0.1).astype(np.float32)
    buf = io.BytesIO()
    import soundfile as sf

    sf.write(buf, samples, SAMPLE_RATE, format="WAV")
    response = client.post(
        "/v1/voices",
        files={"ref_audio": ("ref.wav", buf.getvalue(), "audio/wav")},
        data={"name": "Test Voice", "language": "en", "ref_text": "one two three four five six"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "Test_Voice"
    assert body["language"] == "en"
    assert body["duration"] == pytest.approx(3.0, abs=0.1)

    listed = client.get("/v1/voices").json()
    assert [v["id"] for v in listed] == ["Test_Voice"]
    assert client.get("/v1/voices/Test_Voice/preview").status_code == 200
    assert client.delete("/v1/voices/Test_Voice").status_code == 200
    assert client.get("/v1/voices").json() == []
    assert VoiceStore(tmp_path).list() == []


def test_voice_registration_rejects_bad_name(client, tmp_path):
    main_mod.store.root = tmp_path
    samples = (np.zeros(int(3 * SAMPLE_RATE), np.float32) + 0.1).astype(np.float32)
    buf = io.BytesIO()
    import soundfile as sf

    sf.write(buf, samples, SAMPLE_RATE, format="WAV")
    response = client.post(
        "/v1/voices",
        files={"ref_audio": ("ref.wav", buf.getvalue(), "audio/wav")},
        data={"name": "../evil", "language": "en", "ref_text": "one two three four"},
    )
    assert response.status_code == 400


def test_websocket_stream(client):
    with client.websocket_connect("/v1/ws/synthesize?language=en&response_format=wav") as ws:
        ws.send_text("Hello there, ")
        ws.send_text("this is a test. ")
        ws.send_text("")
        frames = []
        while True:
            try:
                frames.append(ws.receive_bytes())
            except Exception:
                break
    assert frames
    assert all(wav_seconds(frame) > 0.3 for frame in frames)


def test_websocket_rejects_bad_language(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v1/ws/synthesize?language=klingon") as ws:
        ws.receive_bytes()


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"tts_model_ready" in response.content
