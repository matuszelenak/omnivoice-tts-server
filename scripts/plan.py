#!/usr/bin/env python3
"""Measure this machine and print the environment values it should run with.

The admission limits are not universal constants: they are derived from how many
audio-seconds per wall-second the GPU can render. Run this once on the target
hardware (after `make run`), then paste the output into `.env`.

    server/.venv/bin/python scripts/plan.py --url http://localhost:9001

It also prints the quality/latency trade-off per `NUM_STEPS`, so you can pick the
step count for your latency budget instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import httpx

DEFAULT_TEXT = (
    "Hello, this is a test of the speech server. "
    "The quick brown fox jumps over the lazy dog. "
    "Could you please send me the report by Friday afternoon?"
)


def health(base: str) -> dict:
    response = httpx.get(f"{base}/health", timeout=30)
    response.raise_for_status()
    return response.json()


def full_request(base: str, text: str, lang: str, voice: str, steps: int | None) -> tuple[float, float]:
    """(audio seconds, wall seconds) for one non-streaming request."""
    body = {"input": text, "language": lang, "voice": voice, "response_format": "pcm"}
    if steps:
        body["num_step"] = steps
    started = time.perf_counter()
    response = httpx.post(f"{base}/v1/audio/speech", json=body, timeout=300)
    wall = time.perf_counter() - started
    response.raise_for_status()
    return len(response.content) / (24000 * 2), wall


def sse_ttfa(base: str, text: str, lang: str, voice: str, steps: int | None) -> float:
    """Time from request start to the first audio delta."""
    body = {
        "input": text, "language": lang, "voice": voice,
        "response_format": "pcm", "stream_format": "sse",
    }
    if steps:
        body["num_step"] = steps
    started = time.perf_counter()
    with httpx.stream("POST", f"{base}/v1/audio/speech", json=body, timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event.get("type") == "speech.audio.delta":
                    return time.perf_counter() - started
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9001")
    ap.add_argument("--language", default="en")
    ap.add_argument("--voice", default="chloe_sample")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--steps", type=int, nargs="*", default=[8, 12, 16, 24, 32])
    ap.add_argument("--target-queue-wait", type=float, default=10.0,
                    help="seconds a request may wait in the queue before we call it overload")
    args = ap.parse_args()

    info = health(args.url)
    engine = info.get("engine", {})
    print(f"device={engine.get('device')} dtype={engine.get('dtype')} "
          f"model={engine.get('model')} vram={engine.get('gpu_memory_reserved_mb')} MB\n")

    print(f"{'NUM_STEPS':>9} {'latency p50':>12} {'chunk of audio':>15} {'audio-s/s':>10} {'TTFA p50':>9}")
    table = []
    for steps in args.steps:
        audio = []
        wall = []
        ttfa = []
        for _ in range(args.repeats):
            a, w = full_request(args.url, args.text, args.language, args.voice, steps)
            audio.append(a)
            wall.append(w)
            ttfa.append(sse_ttfa(args.url, args.text, args.language, args.voice, steps))
        rate = sum(audio) / sum(wall)
        table.append((steps, statistics.median(wall), statistics.mean(audio), rate, statistics.median(ttfa)))
        print(f"{steps:>9} {statistics.median(wall):>11.2f}s {statistics.mean(audio):>14.2f}s "
              f"{rate:>10.1f} {statistics.median(ttfa):>8.2f}s")

    # Recommended sizing from the throughput of the current default step count.
    current = info.get("config", {}).get("num_steps", 16)
    best = next((row for row in table if row[0] == current), table[len(table) // 2])
    rate = best[3]
    mean_request_audio = best[2]
    chunks_per_request = max(1, round(mean_request_audio / 4.0))  # ~4 s per sentence
    per_chunk = mean_request_audio / chunks_per_request

    pending = max(60, round(rate * args.target_queue_wait, -1))
    jobs = max(8, round(pending / per_chunk))
    streams = max(4, int(rate * 0.8))

    print(f"\n# measured at NUM_STEPS={best[0]}: {rate:.1f} audio-seconds per wall-second")
    print(f"#   ~{per_chunk:.1f}s of audio per queued chunk, {chunks_per_request} chunks per request")
    print(f"#   a full queue of MAX_PENDING_AUDIO_S costs ~{args.target_queue_wait:.0f}s of queue wait")
    print("\n# --- pasted into .env / docker-compose environment ---")
    print(f"NUM_STEPS={best[0]}")
    print(f"MAX_PENDING_AUDIO_S={pending}")
    print(f"MAX_QUEUE_JOBS={jobs}")
    print(f"MAX_STREAM_SESSIONS={streams}")
    print(f"ENCODE_WORKERS={min(8, len(os.sched_getaffinity(0)) or 4)}")
    print("\n# sanity: concurrency that keeps up with real time ~=",
          f"{rate:.0f} streams at {best[0]} steps")
    if rate > 40:
        print("# This GPU is far from saturated at batch size 1 (see bench/batch_probe.py):\n"
              "# consider MAX_BATCH_SIZE>1 / CUDA graphs before raising the queue limits further.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
