#!/usr/bin/env python3
"""Micro-benchmarks direct against the model (no HTTP): what actually costs time.

Answers the questions the server design depends on:

* how cost scales with ``num_step`` (linear) and with text length,
* whether batching concurrent requests helps (it does not: the GPU is already
  saturated at batch size 1),
* what a cached voice prompt saves versus re-encoding the reference,
* what audio encoding costs per format.

    HF_HOME=/mnt/nvme/huggingface CUDA_VISIBLE_DEVICES=0 \
        server/.venv/bin/python bench/raw_model.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch
from omnivoice import OmniVoice

DEVICE = os.environ.get("BENCH_DEVICE", "cuda:0")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE = os.path.join(ROOT, "server/voices/chloe_sample/voice.wav")
REF_TEXT = (Path(ROOT) / "server/voices/chloe_sample/en.txt").read_text(encoding="utf-8").strip()


def timed(fn, n: int = 3) -> tuple[object, list[float]]:
    times = []
    out = None
    for _ in range(n):
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return out, times


def fmt(times: list[float]) -> str:
    return f"mean={statistics.mean(times)*1000:7.1f}ms min={min(times)*1000:7.1f}ms"


def main() -> None:
    started = time.perf_counter()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=DEVICE, dtype=torch.float16)
    print(f"load: {time.perf_counter()-started:.1f}s vram={torch.cuda.memory_allocated()/2**30:.2f}GiB")
    print(f"model: {sum(p.numel() for p in model.parameters())/1e9:.3f}B params, "
          f"frame_rate={model.audio_tokenizer.config.frame_rate}")

    prompt = model.create_voice_clone_prompt(VOICE, REF_TEXT)
    print(f"prompt tokens: {tuple(prompt.ref_audio_tokens.shape)}")
    short = "Hello, this is a test of the speech server."
    mid = "The quick brown fox jumps over the lazy dog, and then it runs away into the forest."
    model.generate(text=short, language="en", voice_clone_prompt=prompt, num_step=8)  # warm up

    print("\n--- num_step (cost is linear) ---")
    for steps in (4, 8, 12, 16, 24, 32):
        out, times = timed(lambda steps=steps: model.generate(
            text=short, language="en", voice_clone_prompt=prompt, num_step=steps))
        audio_s = len(out[0]) / model.sampling_rate
        print(f"steps={steps:3d} {fmt(times)} audio={audio_s:5.2f}s rtf={audio_s/statistics.mean(times):5.1f}x")

    print("\n--- text length at 16 steps ---")
    for multiplier in (1, 2, 4, 8):
        text = " ".join([short] * multiplier)
        out, times = timed(lambda text=text: model.generate(
            text=text, language="en", voice_clone_prompt=prompt, num_step=16), n=2)
        audio_s = len(out[0]) / model.sampling_rate
        print(f"chars={len(text):4d} {fmt(times)} audio={audio_s:6.2f}s rtf={audio_s/statistics.mean(times):5.1f}x")

    print("\n--- batching identical requests (16 steps) ---")
    for batch in (1, 2, 4, 8):
        texts = [short] * batch
        out, times = timed(lambda texts=texts: model.generate(
            text=texts, language="en", voice_clone_prompt=prompt, num_step=16), n=2)
        wall = statistics.mean(times)
        audio_s = sum(len(a) for a in out) / model.sampling_rate
        print(f"batch={batch:2d} wall={wall*1000:6.0f}ms per-item={wall/batch*1000:6.0f}ms "
              f"{batch/wall:5.2f} sent/s audio={audio_s:5.1f}s "
              f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")

    print("\n--- prompt: re-encode reference vs cached ---")
    _, with_path = timed(lambda: model.generate(
        text=short, language="en", ref_audio=VOICE, ref_text=REF_TEXT, num_step=16))
    _, cached = timed(lambda: model.generate(
        text=short, language="en", voice_clone_prompt=prompt, num_step=16))
    print(f"ref_audio path : {fmt(with_path)}")
    print(f"cached prompt  : {fmt(cached)}")
    _, create = timed(lambda: model.create_voice_clone_prompt(VOICE, REF_TEXT), n=5)
    print(f"create_voice_clone_prompt: {fmt(create)}")

    print("\n--- encoding (one 8 s sentence) ---")
    sys.path.insert(0, os.path.join(ROOT, "server"))
    from app import audio as encoder

    samples = model.generate(text=mid, language="en", voice_clone_prompt=prompt, num_step=16)[0]
    for audio_format in ("wav", "pcm", "flac", "mp3", "opus", "aac"):
        _, times = timed(lambda audio_format=audio_format: encoder.encode(samples, audio_format), n=5)
        print(f"{audio_format:5s} {fmt(times)}")


if __name__ == "__main__":
    main()
