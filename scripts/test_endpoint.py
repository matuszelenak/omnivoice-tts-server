#!/usr/bin/env python3
"""End-to-end smoke test for a running OmniVoice server.

    python scripts/test_endpoint.py --text "Hello there." --voice chloe_sample
    python scripts/test_endpoint.py --stream --text "..."   # SSE, first-chunk latency
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import wave
from pathlib import Path

import httpx


def check_health(base: str) -> dict:
    r = httpx.get(f"{base}/health", timeout=30)
    r.raise_for_status()
    return r.json()


def synthesize(base: str, args) -> bytes:
    body = {
        "input": args.text,
        "language": args.language,
        "response_format": args.format,
        "speed": args.speed,
        "stream_format": "sse" if args.stream else "audio",
    }
    if args.voice:
        body["voice"] = args.voice
    if args.instruct:
        body["instructions"] = args.instruct
    if args.num_step:
        body["num_step"] = args.num_step

    started = time.perf_counter()
    if args.stream:
        deltas: list[bytes] = []
        first = None
        with httpx.stream("POST", f"{base}/v1/audio/speech", json=body, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("type") == "speech.audio.delta":
                    if first is None:
                        first = time.perf_counter() - started
                    deltas.append(base64.b64decode(event["audio"]))
                elif event.get("type") == "speech.audio.done":
                    break
                elif "error" in event:
                    raise SystemExit(f"server error: {event['error']}")
        total = time.perf_counter() - started
        payload = _join_deltas(deltas, args.format)
        print(
            f"first chunk: {first:.2f}s  total: {total:.2f}s  "
            f"chunks: {len(deltas)}  bytes: {len(payload)}"
        )
        return payload

    r = httpx.post(f"{base}/v1/audio/speech", json=body, timeout=300)
    r.raise_for_status()
    print(f"total: {time.perf_counter() - started:.2f}s  bytes: {len(r.content)}")
    return r.content


def _join_deltas(deltas: list[bytes], audio_format: str) -> bytes:
    """Join stream deltas into one playable file.

    Each delta of a streamed response is independently decodable. ``pcm``,
    ``mp3`` and ``opus`` concatenate byte-for-byte; ``wav`` does *not* — every
    delta carries its own header, so the frames are re-wrapped here instead.
    """
    if not deltas:
        return b""
    if audio_format != "wav":
        return b"".join(deltas)
    frames = bytearray()
    rate = channels = width = None
    for delta in deltas:
        with wave.open(io.BytesIO(delta)) as handle:
            rate = rate or handle.getframerate()
            channels = channels or handle.getnchannels()
            width = width or handle.getsampwidth()
            frames += handle.readframes(handle.getnframes())
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels or 1)
        out.setsampwidth(width or 2)
        out.setframerate(rate or 24000)
        out.writeframes(bytes(frames))
    return buffer.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9001")
    ap.add_argument("--text", default="Hello, this is a test of the speech server.")
    ap.add_argument("--language", default="en")
    ap.add_argument("--voice", default="chloe_sample")
    ap.add_argument("--instruct", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--num-step", type=int, default=None)
    ap.add_argument("--format", default="wav")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--output", default="out.wav")
    ap.add_argument("--skip-health", action="store_true")
    args = ap.parse_args()

    if not args.skip_health:
        health = check_health(args.url)
        print(json.dumps({k: health[k] for k in ("status", "model", "ready")}, indent=2))
        if not health.get("ready"):
            print("warning: model not ready yet", file=sys.stderr)

    data = synthesize(args.url, args)
    path = Path(args.output)
    path.write_bytes(data)
    print(f"wrote {path} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
