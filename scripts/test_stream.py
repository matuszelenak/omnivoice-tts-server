#!/usr/bin/env python3
"""
Measure streaming TTS latency from the /v1/synthesize endpoint.

Reports time-to-first-chunk and total synthesis time, then saves the
concatenated WAV output.

Usage:
    python scripts/test_stream.py [options]

Examples:
    # auto-pick first available voice
    python scripts/test_stream.py

    # specific voice and longer text
    python scripts/test_stream.py --voice F1 --text "Hello world. How are you today?"

    # adjust quality steps to match what you use in the UI
    python scripts/test_stream.py --steps 30
"""

import argparse
import io
import struct
import sys
import time
import wave

import requests

DEFAULT_URL = "http://localhost:9001"
DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore. "
    "How much wood would a woodchuck chuck?"
)


def fetch_voices(base_url: str) -> list[dict]:
    r = requests.get(f"{base_url}/v1/voices", timeout=10)
    r.raise_for_status()
    return r.json()


def check_health(base_url: str) -> None:
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        data = r.json()
        loaded = data.get("model_loaded", False)
        status = "ready" if loaded else "NOT LOADED"
        print(f"server: {status}")
        if not loaded:
            print("warning: model not loaded — synthesis will fail", file=sys.stderr)
    except Exception as e:
        print(f"health check failed: {e}", file=sys.stderr)
        sys.exit(1)


def stream_synthesize(
    base_url: str,
    text: str,
    voice_id: str,
    language: str,
    speed: float | None,
    total_steps: int | None,
) -> tuple[float, float, list[bytes]]:
    """
    Returns (ttfc_seconds, total_seconds, wav_chunks).
    ttfc = time to first chunk (from request start).
    """
    form: dict[str, str] = {
        "text": text,
        "voice_id": voice_id,
        "language": language,
        "stream": "true",
    }
    if speed is not None:
        form["speed"] = str(speed)
    if total_steps is not None:
        form["total_steps"] = str(total_steps)

    t_start = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/synthesize",
        data=form,
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    chunks: list[bytes] = []
    t_first: float | None = None
    pending = b""

    for raw in resp.iter_content(chunk_size=4096):
        pending += raw
        while len(pending) >= 4:
            size = struct.unpack(">I", pending[:4])[0]
            if len(pending) < 4 + size:
                break
            wav_bytes = pending[4 : 4 + size]
            pending = pending[4 + size :]
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(wav_bytes)

    t_end = time.perf_counter()
    ttfc = (t_first - t_start) if t_first is not None else float("nan")
    total = t_end - t_start
    return ttfc, total, chunks


def merge_wav_chunks(chunks: list[bytes]) -> bytes:
    """Concatenate WAV audio frames from multiple single-channel WAV buffers."""
    if not chunks:
        return b""
    frames = b""
    params = None
    for chunk in chunks:
        with wave.open(io.BytesIO(chunk)) as wf:
            if params is None:
                params = wf.getparams()
            frames += wf.readframes(wf.getnframes())

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(frames)
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure streaming TTS latency")
    parser.add_argument("--url", default=DEFAULT_URL, help="server base URL")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default=None, metavar="VOICE_ID",
                        help="voice id (default: first available)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None, metavar="N",
                        help="total_steps (default: server default)")
    parser.add_argument("--output", default="stream_output.wav",
                        help="path to save concatenated WAV (default: stream_output.wav)")
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()

    if not args.skip_health:
        check_health(args.url)

    voice_id = args.voice
    if voice_id is None:
        voices = fetch_voices(args.url)
        if not voices:
            print("error: no voices available", file=sys.stderr)
            sys.exit(1)
        voice_id = voices[0]["id"]
        print(f"voices: {[v['id'] for v in voices]}")
        print(f"using:  {voice_id}")

    print(f"\ntext ({len(args.text)} chars):")
    preview = args.text[:120] + ("..." if len(args.text) > 120 else "")
    print(f"  {preview}")
    params_line = f"voice={voice_id}  lang={args.language}"
    if args.speed is not None:
        params_line += f"  speed={args.speed}"
    if args.steps is not None:
        params_line += f"  steps={args.steps}"
    print(f"params: {params_line}\n")

    ttfc, total, chunks = stream_synthesize(
        base_url=args.url,
        text=args.text,
        voice_id=voice_id,
        language=args.language,
        speed=args.speed,
        total_steps=args.steps,
    )

    total_bytes = sum(len(c) for c in chunks)
    print(f"chunks:          {len(chunks)}")
    print(f"total audio:     {total_bytes / 1024:.1f} KB")
    print(f"time to first:   {ttfc * 1000:.0f} ms  ← first audio chunk received")
    print(f"total time:      {total:.2f} s")

    if chunks:
        merged = merge_wav_chunks(chunks)
        with open(args.output, "wb") as f:
            f.write(merged)
        print(f"saved:           {args.output} ({len(merged) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
