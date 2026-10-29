"""Audio decoding, encoding and the DSP helpers the voice store needs.

The model emits 24 kHz mono float samples. Everything on the wire is produced
here: ``wav``/``flac`` through libsndfile, ``pcm`` as raw 16-bit LE (OpenAI's
definition: 24 kHz, 16-bit, mono) and ``mp3``/``opus``/``aac`` by piping a WAV
through ffmpeg. Each of those containers is framed independently per chunk, so
the deltas of a stream can be decoded one by one *and* concatenated into a
single valid file.
"""
from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000

RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")

_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

_FFMPEG_ARGS = {
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-qscale:a", "2"],
    "opus": ["-f", "ogg", "-codec:a", "libopus", "-b:a", "48k"],
    "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "96k"],
}

# Decodable upload formats for reference audio.
_UPLOAD_SUFFIXES = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".mp4", ".webm")


class AudioError(ValueError):
    """Raised when an upload cannot be decoded."""


def content_type(fmt: str) -> str:
    return _CONTENT_TYPES.get(fmt, "application/octet-stream")


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg is required for this audio format but is not on PATH")
    return exe


def _resample(x: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return x
    try:
        import torch
        import torchaudio

        return (
            torchaudio.functional.resample(torch.from_numpy(x), sr, target).numpy().astype(np.float32)
        )
    except Exception:  # pragma: no cover - torchaudio always present with the model
        # Linear interpolation is fine for a 20 ms-accurate reference clip.
        n = int(round(len(x) * target / sr))
        return np.interp(
            np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x
        ).astype(np.float32)


# ── decoding ─────────────────────────────────────────────────────────────────


def decode_audio(raw: bytes, filename: str = "ref.wav") -> np.ndarray:
    """Decode an uploaded audio file into mono float32 at :data:`SAMPLE_RATE`."""
    if not raw:
        raise AudioError("empty upload")
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ".wav"
    if suffix not in _UPLOAD_SUFFIXES:
        raise AudioError(f"unsupported audio format '{suffix}'")
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        return _resample(mono, sr)
    except Exception:
        pass
    # libsndfile cannot read M4A/MP4/WEBM; let ffmpeg decode those.
    try:
        proc = subprocess.run(
            [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
            input=raw, capture_output=True, check=False,
        )
    except RuntimeError as exc:
        raise AudioError(str(exc)) from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
        raise AudioError(f"could not decode audio: {detail or 'unknown format'}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


# ── encoding ────────────────────────────────────────────────────────────────


def _soundfile_bytes(samples: np.ndarray, fmt: str, subtype: str | None = None) -> bytes:
    buf = io.BytesIO()
    with sf.SoundFile(
        buf, mode="w", samplerate=SAMPLE_RATE, channels=1, format=fmt,
        subtype=subtype, closefd=False,
    ) as handle:
        handle.write(samples)
    buf.seek(0)
    return buf.read()


def wav_bytes(samples: np.ndarray) -> bytes:
    return _soundfile_bytes(samples, "WAV", "PCM_16")


def _pcm_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _ffmpeg_bytes(samples: np.ndarray, fmt: str) -> bytes:
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         *_FFMPEG_ARGS[fmt], "pipe:1"],
        input=wav_bytes(samples), capture_output=True, check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
        raise RuntimeError(f"ffmpeg failed encoding '{fmt}': {detail}")
    return proc.stdout


# Encoders cheap enough to run inline on the event loop (sub-millisecond for a
# sentence) vs. ones that must go to a thread (ffmpeg fork, ~60 ms per chunk).
_FFMPEG_FORMATS = frozenset(_FFMPEG_ARGS)


def needs_thread(fmt: str) -> bool:
    return fmt in _FFMPEG_FORMATS


def encode(samples: np.ndarray, fmt: str) -> bytes:
    if fmt == "wav":
        return wav_bytes(samples)
    if fmt == "pcm":
        return _pcm_bytes(samples)
    if fmt == "flac":
        return _soundfile_bytes(samples, "FLAC")
    if fmt in _FFMPEG_ARGS:
        return _ffmpeg_bytes(samples, fmt)
    raise AudioError(f"unsupported response_format '{fmt}'")


# ── DSP helpers (reference preparation) ─────────────────────────────────────


def trim_silence(x: np.ndarray, win_ms: int = 20, threshold_ratio: float = 0.08) -> np.ndarray:
    """Drop leading/trailing silence, judged on 20 ms frame RMS."""
    win = max(1, SAMPLE_RATE * win_ms // 1000)
    n = len(x) // win
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
    voiced = np.flatnonzero(rms > max(rms.max() * threshold_ratio, 1e-4))
    if len(voiced) == 0:
        return x
    return x[voiced[0] * win : (voiced[-1] + 1) * win]


def fade(x: np.ndarray, ms: int = 10) -> np.ndarray:
    """Short cosine fade in/out: avoids clicks at chunk boundaries."""
    n = min(len(x) // 2, max(1, SAMPLE_RATE * ms // 1000))
    if n < 2:
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = x.astype(np.float32, copy=True)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def normalize_rms(x: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Scale *x* to a target RMS (no-op for silence)."""
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    if rms < 1e-6:
        return x
    target = 10 ** (target_dbfs / 20.0)
    return (x * (target / rms)).astype(np.float32)


def cap_duration(x: np.ndarray, max_s: float, min_s: float = 0.0) -> np.ndarray:
    """Cut *x* to at most *max_s* seconds on a quiet frame.

    Cutting mid-word makes the model *continue* the half-spoken word into the
    target text (see docs/pitfalls.md), so the cut is placed inside the longest
    quiet run near the end rather than at an arbitrary sample offset.
    """
    if len(x) / SAMPLE_RATE <= max_s:
        return x
    win = SAMPLE_RATE // 50
    limit = int(max_s * SAMPLE_RATE)
    rms = np.sqrt(np.mean(x[:limit][: (limit // win) * win].reshape(-1, win) ** 2, axis=1))
    quiet = rms < max(rms.max() * 0.06, 1e-4)
    runs: list[tuple[int, int]] = []
    start = None
    for i, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = i
        elif not is_quiet and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(quiet)))
    floor = int(min_s * SAMPLE_RATE) if min_s else int(0.55 * limit)
    best = -1
    for a, b in runs:
        if b - a < 4 or a * win < floor:
            continue
        best = max(best, (a + (b - a) // 2) * win)
    return x[: best if best > 0 else limit]
