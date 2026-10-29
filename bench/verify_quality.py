#!/usr/bin/env python3
"""Verify *served* audio by transcribing it with the Parakeet ASR service.

Every case goes through the real HTTP/WebSocket API in the format a client
would request, so this catches regressions that unit tests cannot: a broken
chunk join, a truncated stream, a bad mp3/opus encode, wrong language handling.

    python bench/verify_quality.py --url http://localhost:9001 --asr http://localhost:9000
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import wave
from pathlib import Path

import httpx

EN = [
    "Hello, this is a test of the speech server.",
    "The quick brown fox jumps over the lazy dog.",
    "Could you please send me the report by Friday afternoon?",
    "Our flight to Vienna was delayed by almost three hours.",
    "I have been looking forward to this concert for months.",
    "The machine learning model produces remarkably natural sounding speech.",
]
SK = [
    "Dobrý deň, ako sa máte?",
    "Naraz sa nás ten Kukučín zmocnil, ale všetkých nás to prekvapilo.",
    "Prosím, pošlite mi ten dokument do piatku popoludní.",
    "Náš let do Viedne meškal takmer tri hodiny.",
    "Na tento koncert sa teším už niekoľko mesiacov.",
    "Tento model strojového učenia vytvára pozoruhodne prirodzene znejúci hlas.",
]
VOICES = {"en": "chloe_sample", "sk": "kroner"}


# ASR returns digits, the input text spells them out. Map both to one token so a
# difference in notation is not counted as a word error.
_NUMBERS = {
    "0": "nula", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "20": "twenty", "30": "thirty",
}
_NUMBERS_SK = {
    "1": "jeden", "2": "dva", "3": "tri", "4": "štyri", "5": "päť", "6": "šesť",
    "7": "sedem", "8": "osem", "9": "deväť", "10": "desať",
}


def norm(text: str) -> list[str]:
    import re
    import unicodedata

    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    tokens = re.sub(r"\s+", " ", text).strip().split()
    out = []
    for token in tokens:
        if token.isdigit():
            # Fold numbers to a single canonical token regardless of notation.
            out.append("NUM" if token not in _NUMBERS_SK else "NUM")
        else:
            out.append(token)
    # Spelled-out numbers in the reference are folded too.
    spelled = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
               "jeden", "dva", "tri", "štyri", "päť", "šesť", "sedem", "osem", "deväť", "desať"}
    return ["NUM" if token in spelled else token for token in out]


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    """Wrap OpenAI-style raw PCM (24 kHz, mono, s16le) in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buf.getvalue()


def audio_seconds(data: bytes, fmt: str) -> float:
    if fmt == "pcm":
        return len(data) / (24000 * 2)
    if fmt == "wav":
        try:
            with wave.open(io.BytesIO(data)) as handle:
                return handle.getnframes() / handle.getframerate()
        except Exception:
            return 0.0
    # Compressed formats: let ffprobe tell us (it also proves the stream is valid).
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as handle:
        handle.write(data)
        handle.flush()
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", handle.name],
            capture_output=True, text=True, check=False,
        )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except Exception:
        return 0.0


async def asr_text(client: httpx.AsyncClient, url: str, data: bytes, suffix: str) -> str:
    response = await client.post(
        f"{url}/v1/transcribe",
        files={"file": (f"clip.{suffix}", data, "application/octet-stream")},
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"ASR {response.status_code}: {response.text[:300]} (fmt={suffix}, {len(data)} bytes)")
    return response.json()["text"].strip()


async def synth_full(client, url, text, lang, voice, fmt, num_step=None) -> bytes:
    body = {"input": text, "language": lang, "voice": voice, "response_format": fmt}
    if num_step:
        body["num_step"] = num_step
    response = await client.post(f"{url}/v1/audio/speech", json=body, timeout=300)
    response.raise_for_status()
    return response.content


async def synth_sse(client, url, text, lang, voice, fmt) -> bytes:
    body = {
        "input": text, "language": lang, "voice": voice,
        "response_format": fmt, "stream_format": "sse",
    }
    audio = bytearray()
    async with client.stream("POST", f"{url}/v1/audio/speech", json=body, timeout=300) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "speech.audio.delta":
                audio += base64.b64decode(event["audio"])
            elif event.get("type") == "speech.audio.done":
                break
    return bytes(audio)


async def synth_ws(url, text, lang, voice, fmt, token_words=3) -> bytes:
    import websockets

    ws_url = url.replace("http", "ws") + f"/v1/ws/synthesize?language={lang}&voice_id={voice}&response_format={fmt}"
    audio = bytearray()
    async with websockets.connect(ws_url, open_timeout=30) as ws:
        words = text.split()

        async def send():
            for i in range(0, len(words), token_words):
                await ws.send(" ".join(words[i:i + token_words]) + " ")
            await ws.send("")

        async def recv():
            async for message in ws:
                if isinstance(message, bytes):
                    audio.extend(message)

        await asyncio.gather(send(), recv())
    return bytes(audio)


async def main_async(args) -> int:
    cases = []
    for lang, texts in (("en", EN), ("sk", SK)):
        voice = VOICES[lang]
        for mode, fmt in (
            ("full", "wav"), ("full", "mp3"), ("full", "opus"), ("full", "flac"), ("full", "pcm"),
            ("sse", "wav"), ("sse", "mp3"), ("ws", "wav"),
        ):
            cases.append((lang, texts, voice, mode, fmt))
    if args.steps:
        for steps in args.steps:
            cases.append(("en", EN, VOICES["en"], "full", "wav", steps))
            cases.append(("sk", SK, VOICES["sk"], "full", "wav", steps))

    results = []
    async with httpx.AsyncClient() as client:
        for case in cases:
            lang, texts, voice, mode, fmt = case[:5]
            steps = case[5] if len(case) > 5 else None
            scores, durations = [], []
            for text in texts:
                if mode == "full":
                    data = await synth_full(client, args.url, text, lang, voice, fmt, steps)
                elif mode == "sse":
                    data = await synth_sse(client, args.url, text, lang, voice, fmt)
                else:
                    data = await synth_ws(args.url, text, lang, voice, fmt)
                if not data:
                    scores.append(1.0)
                    continue
                payload = pcm_to_wav(data) if fmt == "pcm" else data
                hyp = await asr_text(client, args.asr, payload, fmt)
                scores.append(wer(text, hyp))
                durations.append(audio_seconds(data, fmt))
                if args.save_dir:
                    out = Path(args.save_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    suffix = fmt if fmt != "pcm" else "pcm"
                    (out / f"{lang}-{mode}-{fmt}-{steps or 'default'}-{len(scores)}.{suffix}").write_bytes(data)
            label = f"{lang:2s} {mode:4s} {fmt:4s}" + (f" steps={steps}" if steps else "")
            mean = statistics.mean(scores)
            results.append((label, mean, len(texts), statistics.mean(durations) if durations else 0.0))
            print(f"{label:28s} WER={mean*100:6.2f}%  n={len(scores)}  "
                  f"audio={statistics.mean(durations) if durations else 0:.2f}s", flush=True)
    print("\nsummary")
    worst = max(results, key=lambda r: r[1])
    print(f"  best: {min(results, key=lambda r: r[1])[0]} ({min(r[1] for r in results)*100:.2f}%)")
    print(f"  worst: {worst[0]} ({worst[1]*100:.2f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9001")
    ap.add_argument("--asr", default="http://localhost:9000")
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="extra num_step sweep")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
