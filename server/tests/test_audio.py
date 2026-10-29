"""Audio decoding/encoding and the reference-preparation DSP helpers."""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app import audio


def tone(seconds: float = 0.5, freq: float = 440.0, rate: int = audio.SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("fmt", ["wav", "pcm", "flac"])
def test_encode_roundtrip(fmt):
    samples = tone()
    data = audio.encode(samples, fmt)
    assert data
    if fmt == "pcm":
        decoded = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0
        assert len(decoded) == len(samples)
    else:
        back, rate = sf.read(io.BytesIO(data), dtype="float32")
        assert rate == audio.SAMPLE_RATE
        assert abs(len(back) - len(samples)) <= 1


def test_encode_pcm_is_clipped():
    loud = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    pcm = np.frombuffer(audio.encode(loud, "pcm"), dtype="<i2")
    assert pcm.max() == 32767 and pcm.min() == -32767


def test_unknown_format_is_rejected():
    with pytest.raises(audio.AudioError):
        audio.encode(tone(), "ogg-vorbis")


def test_needs_thread_flags_ffmpeg_formats():
    assert audio.needs_thread("mp3") and audio.needs_thread("opus") and audio.needs_thread("aac")
    assert not audio.needs_thread("wav") and not audio.needs_thread("pcm")


@pytest.mark.parametrize("fmt,content_type", [("wav", "audio/wav"), ("pcm", "audio/pcm"), ("mp3", "audio/mpeg")])
def test_content_types(fmt, content_type):
    assert audio.content_type(fmt) == content_type


def test_decode_wav_upload():
    samples = tone(1.0)
    buf = io.BytesIO()
    sf.write(buf, samples, audio.SAMPLE_RATE, format="WAV")
    decoded = audio.decode_audio(buf.getvalue(), "clip.wav")
    assert decoded.ndim == 1
    assert abs(len(decoded) - len(samples)) <= 2


def test_decode_stereo_is_downmixed():
    left, right = tone(0.5, 440), tone(0.5, 880)
    stereo = np.stack([left, right], axis=1)
    buf = io.BytesIO()
    sf.write(buf, stereo, audio.SAMPLE_RATE, format="WAV")
    decoded = audio.decode_audio(buf.getvalue(), "clip.wav")
    assert decoded.ndim == 1
    assert abs(len(decoded) - len(left)) <= 2


def test_decode_rejects_unknown_extension():
    with pytest.raises(audio.AudioError):
        audio.decode_audio(b"not audio", "clip.xyz")


def test_decode_rejects_garbage():
    with pytest.raises(audio.AudioError):
        audio.decode_audio(b"\x00" * 100, "clip.wav")


def test_trim_silence_removes_edges():
    quiet = np.zeros(audio.SAMPLE_RATE, dtype=np.float32)
    x = np.concatenate([quiet, tone(0.5), quiet])
    trimmed = audio.trim_silence(x)
    assert 0.4 < len(trimmed) / audio.SAMPLE_RATE < 0.7


def test_cap_duration_respects_limit():
    x = tone(4.0)
    capped = audio.cap_duration(x, max_s=2.0, min_s=1.0)
    assert len(capped) / audio.SAMPLE_RATE <= 2.0 + 0.01


def test_cap_duration_is_noop_when_short_enough():
    x = tone(1.0)
    assert len(audio.cap_duration(x, max_s=2.0)) == len(x)


def test_normalize_rms_targets_level():
    x = tone(0.5) * 0.01
    out = audio.normalize_rms(x, target_dbfs=-20.0)
    rms_db = 20 * np.log10(np.sqrt(np.mean(out**2)))
    assert abs(rms_db + 20.0) < 0.5


def test_fade_edges():
    x = np.ones(1000, dtype=np.float32)
    faded = audio.fade(x, ms=10)
    assert faded[0] < 0.1 and faded[-1] < 0.1 and faded[500] == 1.0
