#!/usr/bin/env python3
"""Does torch.compile help the OmniVoice backbone? (measured, not assumed)

Variants, all on the same inputs:

* ``eager``      — reference (already covered by CUDA graphs elsewhere)
* ``graph``      — our captured forward (app/graphs.py)
* ``compiled``   — ``torch.compile(model.llm)`` on top of eager
* ``compiled+graph`` — compiled backbone, then captured into our graph
* ``maxautotune`` — inductor's own CUDA graphs (``reduce-overhead``)

Compilation itself takes minutes; the numbers below are steady-state.

    HF_HOME=/mnt/nvme/huggingface CUDA_VISIBLE_DEVICES=0 \\
        server/.venv/bin/python bench/torch_compile.py [--modes eager,compiled,compiled+graph]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from omnivoice import OmniVoice

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))
from app.graphs import GraphedForward  # noqa: E402

DEV = "cuda:0"
TEXT = "Hello, this is a test of the speech server."
REF = "server/voices/chloe_sample/voice.wav"
REF_TEXT = "I was wondering, are we friends? I hope you're okay today."


def bench(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def inputs(model, prompt, target=78):
    inp = model._prepare_inference_inputs(
        TEXT, target, prompt.ref_text, prompt.ref_audio_tokens, "en", None, True
    )
    length = inp["input_ids"].size(2)
    ids = inp["input_ids"].repeat(2, 1, 1)
    am = inp["audio_mask"].repeat(2, 1)
    attn = torch.ones((2, 1, length, length), dtype=torch.bool, device=DEV)
    return ids, am, attn, length


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="eager,graph,compiled,compiled+graph",
                    help="eager, graph, compiled, compiled+graph")
    ap.add_argument("--compile-modes", default="default",
                    help="comma-separated torch.compile modes to try, e.g. "
                         "default,max-autotune-no-cudagraphs,max-autotune,reduce-overhead")
    ap.add_argument("--dynamic", default="true", choices=["true", "false"],
                    help="torch.compile(dynamic=...): false specializes per shape")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    modes = args.modes.split(",")
    compile_modes = [m for m in args.compile_modes.split(",") if m]

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=DEV, dtype=torch.float16)
    prompt = model.create_voice_clone_prompt(REF, REF_TEXT)
    model.generate(text=TEXT, language="en", voice_clone_prompt=prompt, num_step=8)  # warm up
    ids, am, attn, length = inputs(model, prompt)
    eager_forward = model.forward

    with torch.inference_mode():
        reference = eager_forward(input_ids=ids, audio_mask=am, attention_mask=attn).logits.float()
    results = {}

    def record(name, forward_callable):
        """Install *forward_callable* as model.forward for both measurements."""
        model.forward = forward_callable
        with torch.inference_mode():
            out = forward_callable(input_ids=ids, audio_mask=am, attention_mask=attn)
        diff = (out.logits.float() - reference).abs().max().item()
        forward = bench(
            lambda: forward_callable(input_ids=ids, audio_mask=am, attention_mask=attn),
        )
        generate = bench(
            lambda: model.generate(
                text=TEXT, language="en", voice_clone_prompt=prompt, num_step=16
            ),
            n=3,
        )
        model.forward = eager_forward
        results[name] = (forward, generate, diff)
        print(
            f"{name:16s} forward {forward*1000:7.2f} ms   generate(16) {generate*1000:7.2f} ms   "
            f"max|diff| {diff:.3g}",
            flush=True,
        )

    if "eager" in modes:
        record("eager", eager_forward)

    if "graph" in modes:
        graphed = GraphedForward(model, bucket=32)
        graphed(input_ids=ids, audio_mask=am, attention_mask=attn)  # capture
        record("graph", graphed)

    compiled_any = "compiled" in modes or "compiled+graph" in modes
    if compiled_any:
        # Restore the eager backbone between attempts.
        backbone = model.llm
        for compile_mode in compile_modes:
            started = time.perf_counter()
            dynamic = args.dynamic == "true"
            model.llm = torch.compile(
                backbone, mode=compile_mode, dynamic=dynamic, fullgraph=False
            )
            try:
                with torch.inference_mode():
                    for _ in range(2):  # trigger compilation on the real shapes
                        eager_forward(input_ids=ids, audio_mask=am, attention_mask=attn)
            except Exception as exc:
                print(f"compile mode {compile_mode!r} failed: {type(exc).__name__}: {exc}", flush=True)
                model.llm = backbone
                continue
            compile_s = time.perf_counter() - started
            print(
                f"compiled backbone (mode={compile_mode}, dynamic={dynamic}) in {compile_s:.0f}s",
                flush=True,
            )
            if "compiled" in modes:
                record(f"compiled[{compile_mode}]", eager_forward)
        model.llm = backbone

    if results:
        base = results.get("eager", (1.0, 1.0, 0))[1]
        print("\nspeed-up of generate() vs eager:")
        for name, (_f, gen, _d) in sorted(results.items(), key=lambda kv: kv[1][1]):
            print(f"  {name:16s} {base/gen:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
