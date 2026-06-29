"""Encode synthesized audio into the formats the OpenAI speech API exposes.

The model emits 24 kHz mono float samples. ``wav``/``flac`` go through
libsndfile (soundfile), ``pcm`` is the raw 16-bit little-endian stream OpenAI
documents, and the compressed formats (``mp3``/``opus``/``aac``) are produced
by piping a WAV through ffmpeg (present in the production image).
"""
from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
import soundfile as sf

# OmniVoice always renders at this rate; OpenAI's ``pcm`` format is defined as
# 24 kHz 16-bit signed little-endian mono, so the two line up exactly.
SAMPLE_RATE = 24000

# The exact set accepted by OpenAI's ``response_format`` field.
RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")

_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# ffmpeg muxer + codec args for the formats libsndfile cannot write portably.
_FFMPEG_ARGS: dict[str, list[str]] = {
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-qscale:a", "2"],
    "opus": ["-f", "ogg", "-codec:a", "libopus"],
    "aac": ["-f", "adts", "-codec:a", "aac"],
}


def content_type(fmt: str) -> str:
    """MIME type for a ``response_format`` value."""
    return _CONTENT_TYPES.get(fmt, "application/octet-stream")


def _soundfile_bytes(audio: np.ndarray, fmt: str, subtype: str | None = None) -> bytes:
    buf = io.BytesIO()
    with sf.SoundFile(
        buf, mode="w", samplerate=SAMPLE_RATE, channels=1,
        format=fmt, subtype=subtype, closefd=False,
    ) as f:
        f.write(audio)
    buf.seek(0)
    return buf.read()


def wav_bytes(audio: np.ndarray) -> bytes:
    """16-bit PCM WAV — used for SSE deltas and the WebSocket stream."""
    return _soundfile_bytes(audio, "WAV", "PCM_16")


def _pcm_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _ffmpeg_bytes(audio: np.ndarray, fmt: str) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(f"ffmpeg is required to encode '{fmt}' audio but was not found on PATH")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *_FFMPEG_ARGS[fmt], "pipe:1"],
        input=wav_bytes(audio),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed encoding '{fmt}': {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout


def encode(audio: np.ndarray, fmt: str) -> bytes:
    """Encode a float sample array into *fmt* (one of :data:`RESPONSE_FORMATS`)."""
    if fmt == "wav":
        return wav_bytes(audio)
    if fmt == "flac":
        return _soundfile_bytes(audio, "FLAC")
    if fmt == "pcm":
        return _pcm_bytes(audio)
    if fmt in _FFMPEG_ARGS:
        return _ffmpeg_bytes(audio, fmt)
    raise ValueError(f"Unsupported response_format '{fmt}'")
