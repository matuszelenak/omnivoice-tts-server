"""Client for the Parakeet ASR service.

Reference-clip transcription runs on the sibling ASR service
(https://github.com/…/parakeet-asr-server, default ``http://localhost:9000``)
instead of loading a second speech model next to the TTS one:

* no extra ~1.6 GB of VRAM competing with the TTS model for the same GPU,
* no Whisper weights to download or version,
* better Slovak (Parakeet Ultra beats whisper-large-v3-turbo on the same
  material — see docs/benchmarks.md), and word timestamps are what reference
  preparation needs to cut a clip between words.

The service is only used when a voice reference is registered (a few times per
lifetime), never on the synthesis path, so it is called synchronously from a
worker thread.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from .audio import SAMPLE_RATE

log = logging.getLogger(__name__)


class ASRError(RuntimeError):
    """The ASR service could not transcribe the clip."""


@dataclass
class Word:
    text: str
    start: float
    end: float


class ASRClient:
    def __init__(self, base_url: str, timeout_s: float = 60.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_s = timeout_s
        self._client: httpx.Client | None = None
        self._available: bool | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    @property
    def available(self) -> bool:
        """Last probe result (``check()`` refreshes it)."""
        return bool(self._available)

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=3.0),
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- availability ------------------------------------------------------
    def check(self) -> bool:
        """Best-effort readiness probe, cached after the first success."""
        if not self.enabled:
            self._available = False
            return False
        if self._available:
            return True
        try:
            response = self._http().get("/health", timeout=3.0)
            self._available = response.status_code == 200
        except Exception:
            self._available = False
        return bool(self._available)

    # -- transcription -----------------------------------------------------
    def _post(self, path: str, wav: bytes, **data: Any) -> dict[str, Any]:
        try:
            response = self._http().post(
                path, files={"file": ("reference.wav", wav, "audio/wav")}, data=data
            )
        except Exception as exc:
            self._available = False
            raise ASRError(f"ASR service unreachable at {self.base_url}: {exc}") from exc
        if response.status_code != 200:
            raise ASRError(
                f"ASR service returned {response.status_code}: {response.text[:200]}"
            )
        return response.json()

    def transcribe_words(self, samples: np.ndarray) -> list[Word]:
        """Transcribe with word timestamps (``POST /v1/transcribe/timestamps``)."""
        payload = _wav_bytes(samples)
        started = time.perf_counter()
        body = self._post("/v1/transcribe/timestamps", payload)
        words = [
            Word(text=str(item.get("word", "")).strip(), start=float(item["start"]), end=float(item["end"]))
            for item in body.get("words", [])
            if item.get("word")
        ]
        log.info(
            "transcribed reference with ASR in %.2fs (%d words)",
            time.perf_counter() - started, len(words),
        )
        return words

    def transcribe_text(self, samples: np.ndarray) -> str:
        """Plain transcription (``POST /v1/transcribe``)."""
        body = self._post("/v1/transcribe", _wav_bytes(samples))
        return str(body.get("text", "")).strip()


def _wav_bytes(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()
