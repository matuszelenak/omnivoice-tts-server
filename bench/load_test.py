#!/usr/bin/env python3
"""Load generator for the TTS server.

    python bench/load_test.py --mode sse --concurrency 8 --requests 200
    python bench/load_test.py --mode full --concurrency 16 --duration 60
    python bench/load_test.py --mode ws --concurrency 4 --text corpus/sk_long

Reports latency percentiles, time-to-first-audio, throughput and error codes,
and can save one response per mode for ASR verification (``--save-dir``).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import ssl
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx

EN = [
    "Hello, this is a test of the speech server.",
    "The quick brown fox jumps over the lazy dog.",
    "Could you please send me the report by Friday afternoon?",
    "Our flight to Vienna was delayed by almost three hours, which was annoying.",
    "I have been looking forward to this concert for months.",
    "The machine learning model produces remarkably natural sounding speech.",
    "Please remember to water the plants before you leave the house.",
    "We should probably order dinner now, because the kitchen closes soon.",
]

SK = [
    "Dobrý deň, ako sa máte?",
    "Prosím, pošlite mi ten dokument do piatku popoludní.",
    "Náš let do Viedne meškal takmer tri hodiny a bolo to veľmi nepríjemné.",
    "Na tento koncert sa teším už niekoľko mesiacov.",
    "Tento model strojového učenia vytvára pozoruhodne prirodzene znejúci hlas.",
    "Nezabudni pred odchodom z domu poliať kvety na balkóne.",
    "Mali by sme si objednať večeru, lebo kuchyňa čoskoro zatvára.",
    "Pekný sobotný večer, vážení priatelia, vitajte v našej relácii.",
]

LONG_EN = (
    "Welcome to the show. Today we are talking about speech synthesis, and why "
    "latency matters more than throughput for interactive applications. A short "
    "pause is fine; a long silence feels broken. We will look at how the model "
    "splits text into sentences, how the scheduler keeps several users busy at "
    "once, and what happens when the queue fills up. Stay with us."
)


@dataclass
class Sample:
    started: float
    ttfa: float | None = None
    finished: float | None = None
    audio_bytes: int = 0
    status: int = 0
    error: str | None = None
    bytes: bytes = field(default=b"", repr=False)


def corpus(name: str) -> list[str]:
    if name == "en":
        return EN
    if name == "sk":
        return SK
    if name == "long":
        return [LONG_EN]
    if name == "mixed":
        return EN + SK + [LONG_EN]
    path = Path(name)
    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise SystemExit(f"unknown corpus {name!r}")


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[idx]


async def one_full(client: httpx.AsyncClient, url: str, text: str, args, lang: str) -> Sample:
    body = {
        "input": text,
        "language": lang,
        "response_format": "wav",
        "voice": args.voice,
    }
    if args.instruct:
        body.pop("voice", None)
        body["instructions"] = args.instruct
    if args.num_step:
        body["num_step"] = args.num_step
    sample = Sample(started=time.perf_counter())
    try:
        response = await client.post(f"{url}/v1/audio/speech", json=body, timeout=args.timeout)
        sample.status = response.status_code
        if response.status_code == 200:
            sample.bytes = response.content
            sample.audio_bytes = len(response.content)
            sample.ttfa = time.perf_counter() - sample.started
        else:
            sample.error = response.text[:160]
    except Exception as exc:
        sample.error = f"{type(exc).__name__}: {exc}"
    sample.finished = time.perf_counter()
    return sample


async def one_sse(client: httpx.AsyncClient, url: str, text: str, args, lang: str) -> Sample:
    body = {
        "input": text,
        "language": lang,
        "response_format": "wav",
        "voice": args.voice,
        "stream_format": "sse",
    }
    if args.instruct:
        body.pop("voice", None)
        body["instructions"] = args.instruct
    if args.num_step:
        body["num_step"] = args.num_step
    sample = Sample(started=time.perf_counter())
    audio = bytearray()
    try:
        async with client.stream(
            "POST", f"{url}/v1/audio/speech", json=body, timeout=args.timeout
        ) as response:
            sample.status = response.status_code
            if response.status_code != 200:
                sample.error = (await response.aread()).decode()[:160]
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event.get("type") == "speech.audio.delta":
                        if sample.ttfa is None:
                            sample.ttfa = time.perf_counter() - sample.started
                        audio += base64.b64decode(event["audio"])
                    elif "error" in event:
                        sample.error = str(event["error"])[:160]
                        break
                    elif event.get("type") == "speech.audio.done":
                        break
    except Exception as exc:
        sample.error = f"{type(exc).__name__}: {exc}"
    sample.bytes = bytes(audio)
    sample.audio_bytes = len(audio)
    sample.finished = time.perf_counter()
    return sample


async def one_ws(url: str, text: str, args, lang: str) -> Sample:
    import websockets

    sample = Sample(started=time.perf_counter())
    ws_url = url.replace("http", "ws")
    query = f"?language={lang}&voice_id={args.voice}&response_format=wav"
    if args.instruct:
        query = f"?language={lang}&instruct={httpx.QueryParams({'i': args.instruct})['i']}&response_format=wav"
    if args.num_step:
        query += f"&num_step={args.num_step}"
    audio = bytearray()
    try:
        async with websockets.connect(f"{ws_url}/v1/ws/synthesize{query}", open_timeout=30) as ws:
            words = text.split()
            step = args.token_words
            sender_done = asyncio.Event()

            async def send() -> None:
                try:
                    for i in range(0, len(words), step):
                        await ws.send(" ".join(words[i:i + step]) + " ")
                        if args.token_delay:
                            await asyncio.sleep(args.token_delay)
                    await ws.send("")
                finally:
                    sender_done.set()

            async def receive() -> None:
                try:
                    async for message in ws:
                        if isinstance(message, bytes):
                            if sample.ttfa is None:
                                sample.ttfa = time.perf_counter() - sample.started
                            audio.extend(message)
                except Exception:
                    pass

            await asyncio.gather(send(), receive())
            sample.status = 200
    except Exception as exc:
        sample.error = f"{type(exc).__name__}: {exc}"
    sample.bytes = bytes(audio)
    sample.audio_bytes = len(audio)
    sample.finished = time.perf_counter()
    return sample


def wav_seconds(data: bytes) -> float:
    try:
        import io

        with wave.open(io.BytesIO(data)) as handle:
            return handle.getnframes() / handle.getframerate()
    except Exception:
        return 0.0


async def run(args) -> int:
    texts = corpus(args.corpus)
    langs = ["en", "sk"] if args.corpus == "mixed" else [args.lang]
    samples: list[Sample] = []
    lock = asyncio.Lock()
    stop_at = time.perf_counter() + args.duration if args.duration else None
    issued = 0

    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=args.timeout) as client:

        async def worker(index: int) -> None:
            nonlocal issued
            while True:
                async with lock:
                    if stop_at is not None:
                        if time.perf_counter() >= stop_at:
                            return
                    elif issued >= args.requests:
                        return
                    issued += 1
                    n = issued
                text = texts[(n + index) % len(texts)]
                lang = langs[(n + index) % len(langs)]
                if args.mode == "full":
                    sample = await one_full(client, args.url, text, args, lang)
                elif args.mode == "sse":
                    sample = await one_sse(client, args.url, text, args, lang)
                else:
                    sample = await one_ws(args.url, text, args, lang)
                async with lock:
                    samples.append(sample)
                    if args.save_dir and len(samples) <= args.save:
                        path = Path(args.save_dir)
                        path.mkdir(parents=True, exist_ok=True)
                        (path / f"{args.mode}-{len(samples):03d}.wav").write_bytes(sample.bytes)

        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))

    ok = [s for s in samples if s.status == 200 and s.error is None and s.audio_bytes]
    errors = [s for s in samples if s.error]
    rejected = [s for s in samples if s.status in (429, 503)]
    wall = max(s.finished for s in samples) - min(s.started for s in samples) if samples else 0.0
    latencies = [(s.finished - s.started) for s in ok]
    ttf = [s.ttfa for s in ok if s.ttfa is not None]
    audio_s = sum(wav_seconds(s.bytes) for s in ok)

    print(f"\nmode={args.mode} concurrency={args.concurrency} corpus={args.corpus} "
          f"requests={len(samples)} wall={wall:.1f}s")
    if errors:
        print(f"errors={len(errors)} rejected={len(rejected)}")
        for sample in errors[:5]:
            print(f"  {sample.status} {sample.error}")
    if ok:
        print(f"throughput: {len(ok)/wall:6.2f} req/s   {audio_s/wall:6.1f} audio-s/s   "
              f"RTF={audio_s/sum(latencies):5.1f}x")
        print(f"latency  p50={pct(latencies,.5):6.2f}s p90={pct(latencies,.9):6.2f}s "
              f"p99={pct(latencies,.99):6.2f}s max={max(latencies):6.2f}s")
        if ttf:
            print(f"first chunk p50={pct(ttf,.5):6.2f}s p90={pct(ttf,.9):6.2f}s "
                  f"p99={pct(ttf,.99):6.2f}s max={max(ttf):6.2f}s")
        print(f"audio per request: mean={audio_s/len(ok):5.2f}s")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9001")
    ap.add_argument("--mode", choices=["full", "sse", "ws"], default="full")
    ap.add_argument("--corpus", default="en", help="en|sk|long|mixed|path/to/file")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--voice", default="chloe_sample")
    ap.add_argument("--instruct", default=None)
    ap.add_argument("--num-step", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--duration", type=float, default=None, help="run for N seconds instead of N requests")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--token-words", type=int, default=3, help="words per WS text frame")
    ap.add_argument("--token-delay", type=float, default=0.0, help="seconds between WS text frames")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--save", type=int, default=3)
    args = ap.parse_args()
    ssl._create_default_https_context = ssl._create_unverified_context
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
