#!/usr/bin/env python3
"""
Test script for the OmniVoice TTS endpoint.

Usage:
    python scripts/test_endpoint.py [--url URL] --ref-audio PATH --ref-text TEXT

Defaults to http://localhost:9001 and the sample files in the project root.
"""

import argparse
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_URL = "http://localhost:9001"


def check_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        data = r.json()
        print(f"Health: {data}")
        return data.get("model_loaded", False)
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def synthesize(
    base_url: str,
    text: str,
    language: str,
    speed: float | None,
    ref_audio_path: str | None,
    ref_text: str | None,
    output_path: str,
) -> None:
    data: dict = {"text": text, "language": language}
    if speed is not None:
        data["speed"] = str(speed)
    if ref_text:
        data["ref_text"] = ref_text

    files = {}
    ref_fh = None
    if ref_audio_path and os.path.exists(ref_audio_path):
        ref_fh = open(ref_audio_path, "rb")
        files["ref_audio"] = (os.path.basename(ref_audio_path), ref_fh, "audio/wav")

    try:
        print(f"Sending synthesis request to {base_url}/v1/synthesize ...")
        print(f"  text     : {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  language : {language}")
        print(f"  speed    : {speed}")
        print(f"  ref_audio: {ref_audio_path}")

        t0 = time.perf_counter()
        response = requests.post(
            f"{base_url}/v1/synthesize",
            data=data,
            files=files if files else None,
            timeout=300,
        )
        elapsed = time.perf_counter() - t0

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
            sys.exit(1)

        with open(output_path, "wb") as f:
            f.write(response.content)

        size_kb = len(response.content) / 1024
        print(f"Done in {elapsed:.1f}s — saved {size_kb:.1f} KB to {output_path}")
    finally:
        if ref_fh:
            ref_fh.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the OmniVoice TTS endpoint")
    parser.add_argument("--url", default=DEFAULT_URL)
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument("--text", default=None)
    text_group.add_argument("--text-file", metavar="PATH")
    parser.add_argument("--language", default="sk")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--output", default=os.path.join(ROOT, "test_output.wav"))
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = args.text

    if not args.skip_health:
        if not check_health(args.url):
            print("Warning: model not yet loaded or server unreachable", file=sys.stderr)

    synthesize(
        base_url=args.url,
        text=text,
        language=args.language,
        speed=args.speed,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
