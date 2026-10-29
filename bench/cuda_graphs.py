#!/usr/bin/env python3
"""Experiment: does a CUDA graph around the denoising forward pass pay off?

OmniVoice runs `num_step` identical-shaped forward passes per sentence (2 x batch
because of CFG, same sequence length), which is exactly what CUDA graphs are for:
2800 kernel launches per sentence become 16 graph replays. Two complications:

* the sequence length varies per sentence -> capture one graph per *padded*
  length bucket (pad to a multiple of 64); padded positions are masked so real
  outputs are unchanged (verified here),
* the sampling loop between steps has a data-dependent `topk(k)` per step, so it
  stays eager — only the forward pass is captured.

Run with the server stopped:

    HF_HOME=/mnt/nvme/huggingface CUDA_VISIBLE_DEVICES=0 \\
        server/.venv/bin/python bench/cuda_graphs.py
"""
from __future__ import annotations

import statistics
import time

import torch
from omnivoice import OmniVoice

DEV = "cuda:0"
TEXT = "Hello, this is a test of the speech server."
REF = "server/voices/chloe_sample/voice.wav"
REF_TEXT = "I was wondering, are we friends? I hope you're okay today."
BUCKET = 64


def bench(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


class GraphedForward:
    """Drop-in replacement for ``model.forward`` that replays a captured graph.

    Inputs are copied into static buffers (a few KB), the graph is replayed and
    the fp32 logits are read from the static output buffer. Shapes are padded up
    to a multiple of ``bucket``; padded query positions attend only to themselves
    so the rows we read are numerically unchanged.
    """

    def __init__(self, model, cfgbatch: int, bucket: int = 0, max_graphs: int = 6):
        self.model = model
        self.cfgbatch = cfgbatch  # 2 (cond + uncond) for a single request
        self.bucket = bucket      # 0 = capture at the exact length, >0 = pad to a bucket
        self.max_graphs = max_graphs
        self._graphs: dict[int, tuple] = {}
        self._eager = model.forward
        self.captures = 0
        self.capture_seconds = 0.0

    # -- shape handling ----------------------------------------------------
    def _bucket_for(self, length: int) -> int:
        if self.bucket <= 0:
            return length
        return max(self.bucket, -(-length // self.bucket) * self.bucket)  # ceil

    def _build(self, length: int):
        """Allocate static buffers and capture the graph for padded length."""
        b = self.cfgbatch
        codebooks = self.model.config.num_audio_codebook
        device = next(self.model.parameters()).device
        ids = torch.zeros((b, codebooks, length), dtype=torch.long, device=device)
        audio_mask = torch.zeros((b, length), dtype=torch.bool, device=device)
        attn = torch.zeros((b, 1, length, length), dtype=torch.bool, device=device)
        # Warm up eagerly on the same shapes (allocator + kernel selection).
        for _ in range(2):
            out = self._eager(input_ids=ids, audio_mask=audio_mask, attention_mask=attn)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        with torch.cuda.graph(graph):
            out = self._eager(input_ids=ids, audio_mask=audio_mask, attention_mask=attn)
            logits = out.logits.to(torch.float32)
        self.capture_seconds += time.perf_counter() - started
        self.captures += 1
        return graph, ids, audio_mask, attn, logits

    # -- call --------------------------------------------------------------
    def __call__(self, input_ids=None, audio_mask=None, attention_mask=None, **kwargs):
        if kwargs or input_ids is None or attention_mask is None:
            return self._eager(
                input_ids=input_ids, audio_mask=audio_mask, attention_mask=attention_mask, **kwargs
            )
        length = input_ids.size(2)
        padded = self._bucket_for(length)
        if padded != length:
            # Exact-shape callers cannot use a padded graph: fall back.
            return self._eager(
                input_ids=input_ids, audio_mask=audio_mask, attention_mask=attention_mask
            )
        entry = self._graphs.get(length)
        if entry is None:
            if len(self._graphs) >= self.max_graphs:
                return self._eager(
                    input_ids=input_ids, audio_mask=audio_mask, attention_mask=attention_mask
                )
            entry = self._build(length)
            self._graphs[length] = entry
        graph, ids, am, attn, logits = entry
        ids.copy_(input_ids)
        am.copy_(audio_mask)
        attn.copy_(attention_mask)
        graph.replay()
        return type("Out", (), {"logits": logits})

    def replay_padded(self, input_ids, audio_mask, attention_mask):
        """Replay a padded graph with exact-shape inputs (measures padding cost)."""
        length = input_ids.size(2)
        padded = self._bucket_for(length)
        p_ids, p_am, p_attn = padded_inputs(
            {"input_ids": input_ids, "audio_mask": audio_mask, "attention_mask": attention_mask},
            padded,
        )
        entry = self._graphs.get(padded)
        if entry is None:
            entry = self._build(padded)
            self._graphs[padded] = entry
        graph, ids, am, attn, logits = entry
        ids.copy_(p_ids)
        am.copy_(p_am)
        attn.copy_(p_attn)
        graph.replay()
        return logits


def padded_inputs(inp, length: int):
    """Pad a (2, C, L) / (2, L) / (2, 1, L, L) input set to *length*.

    Padded positions are self-attention-only, so rows [0, c_len) are unchanged.
    """
    ids, am, attn = inp["input_ids"], inp["audio_mask"], inp["attention_mask"]
    b, c, n = ids.shape
    padded_ids = ids.new_zeros((b, c, length))
    padded_am = am.new_zeros((b, length))
    padded_attn = attn.new_zeros((b, 1, length, length))
    padded_ids[:, :, :n] = ids
    padded_am[:, :n] = am
    padded_attn[:, :, :n, :n] = attn
    if length > n:
        diag = torch.arange(n, length, device=ids.device)
        padded_attn[:, :, diag, diag] = True
    return padded_ids, padded_am, padded_attn


def main() -> None:
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=DEV, dtype=torch.float16)
    prompt = model.create_voice_clone_prompt(REF, REF_TEXT)
    generated = model.generate(text=TEXT, language="en", voice_clone_prompt=prompt, num_step=8)

    # Reproduce exactly what _generate_iterative builds for one item (B=1 -> 2 rows).
    inp = model._prepare_inference_inputs(
        TEXT, 78, prompt.ref_text, prompt.ref_audio_tokens, "en", None, True
    )
    c_len = inp["input_ids"].size(2)
    ids = inp["input_ids"].repeat(2, 1, 1)
    am = inp["audio_mask"].repeat(2, 1)
    attn = torch.ones((2, 1, c_len, c_len), dtype=torch.bool, device=DEV)
    print(f"sequence length (cond) = {c_len}; CFG batch = 2; {len(generated)} clip(s)")

    eager = bench(lambda: model(input_ids=ids, audio_mask=am, attention_mask=attn), n=10)
    print(f"eager forward            {eager*1000:7.2f} ms")

    # (a) exact-length graph: no padding waste, one capture per distinct length
    exact = GraphedForward(model, cfgbatch=2, bucket=0)
    exact(input_ids=ids, audio_mask=am, attention_mask=attn)  # captures
    torch.cuda.synchronize()
    exact_fwd = bench(lambda: exact(input_ids=ids, audio_mask=am, attention_mask=attn), n=10)
    print(f"graph, exact L={c_len}     {exact_fwd*1000:7.2f} ms  ({eager/exact_fwd:.2f}x)   "
          f"capture {exact.capture_seconds*1000:.0f} ms")

    # (b) padded to a bucket: reusable across lengths, but pays for padding
    bucketed = GraphedForward(model, cfgbatch=2, bucket=BUCKET)
    padded = bucketed._bucket_for(c_len)
    bucketed.replay_padded(ids, am, attn)  # captures at the padded length
    torch.cuda.synchronize()
    bucketed_fwd = bench(lambda: bucketed.replay_padded(ids, am, attn), n=10)
    print(f"graph, padded L={padded}   {bucketed_fwd*1000:7.2f} ms  ({eager/bucketed_fwd:.2f}x)   "
          f"capture {bucketed.capture_seconds*1000:.0f} ms")

    graphed = exact
    model.forward = graphed  # _generate_iterative calls self(...) -> forward

    # Numerical check: graph output vs eager on random inputs, real rows only.
    torch.manual_seed(0)
    rnd = torch.randint(0, 1000, ids.shape, device=DEV)
    rnd_am = torch.zeros_like(am)
    rnd_am[:, c_len - 78 :] = True  # target region is audio
    eager_logits = model.__class__.forward(
        model, input_ids=rnd, audio_mask=rnd_am, attention_mask=attn
    ).logits.to(torch.float32)
    graph_logits = graphed(input_ids=rnd, audio_mask=rnd_am, attention_mask=attn).logits
    diff = (eager_logits - graph_logits).abs().max().item()
    print(f"max |eager - graph| on logits: {diff:.4g}")

    # End to end: same generate() call through the graphed forward.
    orig_forward = model.__class__.forward.__get__(model, type(model))
    model.forward = graphed
    gen_graphed = bench(
        lambda: model.generate(text=TEXT, language="en", voice_clone_prompt=prompt, num_step=16),
        n=5,
    )
    model.forward = orig_forward
    gen_eager = bench(
        lambda: model.generate(text=TEXT, language="en", voice_clone_prompt=prompt, num_step=16),
        n=5,
    )
    print(f"generate 16 steps eager   {gen_eager*1000:7.2f} ms")
    print(f"generate 16 steps graphed {gen_graphed*1000:7.2f} ms  ({gen_eager/gen_graphed:.2f}x)")
    print(f"graphs captured: {graphed.captures} (total capture {graphed.capture_seconds*1000:.0f} ms)")


if __name__ == "__main__":
    main()
