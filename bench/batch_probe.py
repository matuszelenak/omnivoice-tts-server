"""Batching probe: is OmniVoice compute-bound at batch 1, or is batching being
lost to per-item host overhead in the decode loop?

Run against GPU 0 with the server stopped:

    HF_HOME=/mnt/nvme/huggingface CUDA_VISIBLE_DEVICES=0 \
        server/.venv/bin/python bench/batch_probe.py

Reports, per batch size: one denoising step's forward pass (pure GPU), the whole
16-step generate() (forward + per-item sampling), SM utilisation and power.

Splits the measurement into:
  * forward(2B x L)  - pure GPU work of one denoising step
  * generate()       - the whole loop (forward + per-item sampling in Python)
  * SM util / power  - is the GPU actually busy?
"""
import statistics
import subprocess
import threading
import time

import torch
from omnivoice import OmniVoice

DEV = "cuda:0"
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=DEV, dtype=torch.float16)
prompt = model.create_voice_clone_prompt(
    "server/voices/chloe_sample/voice.wav",
    "I was wondering, are we friends? I hope you're okay today.")
TEXT = "Hello, this is a test of the speech server."
model.generate(text=TEXT, language="en", voice_clone_prompt=prompt, num_step=8)


def bench(fn, n=3):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.mean(ts)


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.util = []
        self.power = []
        self.stop = False

    def run(self):
        while not self.stop:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                                  "--format=csv,noheader,nounits", "-i", "0"],
                                 capture_output=True, text=True).stdout.strip().split(",")
            try:
                self.util.append(float(out[0]))
                self.power.append(float(out[1]))
            except Exception:
                pass
            time.sleep(0.05)

# ---- pure forward scaling -------------------------------------------------
inp = model._prepare_inference_inputs(
    TEXT, 78, prompt.ref_text, prompt.ref_audio_tokens, "en", None, True)
L = inp["input_ids"].size(2)
ids1, am1 = inp["input_ids"], inp["audio_mask"]
mask1 = torch.ones((1, 1, L, L), dtype=torch.bool, device=DEV)
print(f"one sentence: {L} tokens/sequence (style+text+ref+target), 2x because of CFG\n")
print(f"{'B':>3} {'forward(2B)ms':>14} {'per-item':>9} {'vs B=1':>7} | {'generate ms':>12} {'per-item':>9} {'vs B=1':>7} | {'SM%':>5} {'W':>5}")
base_f = base_g = None
for B in (1, 2, 4, 8):
    ids = ids1.repeat(2 * B, 1, 1)
    am = am1.repeat(2 * B, 1)
    mask = torch.ones((2 * B, 1, L, L), dtype=torch.bool, device=DEV)
    with torch.inference_mode():
        f = bench(lambda ids=ids, am=am, mask=mask: model(
            input_ids=ids, audio_mask=am, attention_mask=mask), n=5)
        texts = [TEXT] * B
        sampler = GpuSampler()
        sampler.start()
        g = bench(lambda texts=texts: model.generate(
            text=texts, language="en", voice_clone_prompt=prompt, num_step=16), n=3)
        sampler.stop = True
        sampler.join()
    if B == 1:
        base_f, base_g = f, g
    print(f"{B:>3} {f*1000:>14.0f} {f/B*1000:>9.0f} {f/base_f:>7.2f} | {g*1000:>12.0f} {g/B*1000:>9.0f} {g/base_g:>7.2f} | "
          f"{statistics.mean(sampler.util):>5.0f} {statistics.mean(sampler.power):>5.0f}")

# ---- practical peak: a big fp16 GEMM, i.e. the denominator for "fraction of peak"
def gemm_peak(dtype, n=8192, iters=10):
    a = torch.randn(n, n, device=DEV, dtype=dtype)
    b = torch.randn(n, n, device=DEV, dtype=dtype)
    with torch.inference_mode():
        for _ in range(3):
            _ = a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    return 2 * n**3 * iters / seconds / 1e12


print(f"\npractical peak (8192^3 GEMM): fp16 {gemm_peak(torch.float16):.0f} TFLOPS, "
      f"bf16 {gemm_peak(torch.bfloat16):.0f} TFLOPS")

# TFLOPS achieved on the forward pass alone (2 x params x tokens), fp16
params = sum(p.numel() for p in model.parameters())
flops = 2 * params * 2 * 1 * L
achieved = flops / (base_f or 1) / 1e12
print(f"\nforward FLOPs at B=1: {flops/1e12:.3f} TFLOP -> {achieved:.1f} TFLOPS "
      f"({achieved / max(gemm_peak(torch.float16, iters=3), 1e-9) * 100:.0f}% of this card's "
      f"measured GEMM peak)")
print("note: published table peaks for consumer Ampere are often the FP32-accumulate half-rate "
      "figure; compare against the measured GEMM peak above, not the datasheet.")
print("timesteps:", model.config.num_audio_codebook, "codebooks;", L, "tokens")
