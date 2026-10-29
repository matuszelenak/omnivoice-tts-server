# Benchmarks

Every number here was measured on the reference machine with the scripts in
`bench/`, against the **production image** (`omnivoice-tts:prod`, granian,
fp16, `NUM_STEPS=16` unless stated). They are properties of *this* hardware and
*this* model revision, not model-card claims.

## Test setup

| | |
| --- | --- |
| GPU | 2 × RTX 3090 24 GB (TTS on GPU 0 at 100 % utilisation, ASR on GPU 1), **TTS card runs the [gpu-unlock](https://github.com/Zagitta/gpu-unlock) fix at 1800 MHz** |
| CPU / RAM | 16 threads, 94 GB |
| Model | `k2-fsa/OmniVoice`, 0.81 B params (Qwen3 28×1024 backbone), 24 kHz output, 25 fps audio tokens |
| Precision | fp16, `torch.set_float32_matmul_precision("high")` |
| Server | one granian worker, `NUM_STEPS=16`, `FIRST_CHUNK_STEPS=12`, guidance 2.0 |
| Corpus (EN) | 8 sentences, 2.5–5.5 s of audio each (~4.4 s mean) |
| Corpus (SK) | 8 sentences, ~4 s each, `kroner` voice |
| Corpus (long) | one 6-sentence paragraph, 21 model calls, ~28 s of audio |
| Quality reference | transcripts of the served audio by the parakeet ASR service |
| Load generator | `bench/load_test.py` (asyncio + httpx; WebSocket via `websockets`) |
| Acceleration | shipped defaults: `TORCH_COMPILE=true`, `BATCH_MAX_SIZE=8`, `CUDA_GRAPHS=false`. Each section below names the setting it isolates |

Reproduce:

```bash
make build && make run            # production image on GPU 0
bench/run_all.sh                  # writes bench/results/run-<stamp>.log
bench/verify_quality.py           # WER of served audio, per format/mode
server/.venv/bin/python bench/raw_model.py   # direct model micro-benchmarks
```

## Hardware unlock (gpu-unlock) — what it changed

GeForce cards run FP16/BF16/TF32 tensor math **with FP32 accumulation at half
rate**; professional cards on the same GA102 die run it at full rate. NVIDIA's own
whitepaper lists the RTX 3090 at 71 TFLOPS for FP16+FP32-accumulate versus 142 for
FP16+FP16-accumulate (`RTX A6000` gets the full rate). [gpu-unlock](https://github.com/Zagitta/gpu-unlock)
writes the speed-selector registers before the driver binds and restores the full
issue rate.

Measured on this box with `bench/batch_probe.py`, same script before and after:

| | stock | unlocked | change |
| --- | --- | --- | --- |
| one 16-step sentence (`generate`) | 289 ms | **227 ms** | 1.27× faster |
| one denoising step (forward, 2×224 tokens) | 17 ms | **13 ms** | 1.31× |
| achieved TFLOPS at batch 1 | 42.4 | **54.4** | 1.28× |
| SM utilisation at batch 1 | 96 % | **76 %** | host gaps now visible |
| sustained SM clock / power | ~1400 MHz / 330 W | **1800 MHz / ~300 W** | — |
| dense 8192³ GEMM (measured, fp16 / bf16) | — | **114 / 117 TFLOPS** | 90 % of the 142 full-rate spec |

End to end the same suite got **1.2–1.3×** faster: single-request latency
0.34 → 0.27 s, first chunk 0.26 → 0.20 s, long-text render 1.85 → 1.52 s,
throughput 13.3 → 15.2 audio-s/s at 16 steps.

Why not 2×: the unlock only lifts tensor ops that accumulate in FP32. The rest of
the step — attention softmax, RMSNorm, SwiGLU, embedding gathers, the fp32 logit
cast, the guidance/sampling arithmetic and the audio decoder — is untouched, and
the short 224-token sequence only reaches ~40 % of the GEMM peak anyway (small
matrices). Amdahl caps the gain; the SM-utilisation drop from 96 % to 76 % is the
same effect seen from the other side: the model is now waiting on non-tensor work
and host-side step overhead rather than on FLOPs.

Note the unlock does **not** exist on datacenter parts (H100/H200 always run full
rate), so it is a development-box property, not something to plan production
capacity around.

### torch.compile on the backbone: the biggest single win

Compiling the Qwen3 backbone with inductor fuses RMSNorm/SwiGLU/residual work into
far fewer, wider kernels. `bench/torch_compile.py`, 224-token sequence:

| variant | forward pass | `generate` (16 steps) | vs eager |
| --- | --- | --- | --- |
| eager | 16.27 ms | 225.8 ms | 1.00× |
| CUDA graph only | 11.23 ms | 219.7 ms | 1.03× |
| **`torch.compile`** | **8.11 ms** | **136.6 ms** | **1.65×** |
| compile + CUDA graph | 6.93 ms | 143.0 ms | 1.58× |

Through the HTTP API (same session, defaults otherwise), compile off → on:

| scenario | `TORCH_COMPILE=false` | `true` | |
| --- | --- | --- | --- |
| SSE, 1 client (first chunk) | 0.23 s / 18.3 audio-s/s | **0.15 s / 24.3** | 1.53× |
| SSE, 8 clients | 1.43 s / 24.2 audio-s/s | **0.84 s / 39.8** | 1.70× |
| non-streaming, 8 clients | 1.88 s / 19.0 | **1.14 s / 33.3** | 1.65× |
| 28 s text, 1 client | 1.46 s / 18.7 | **0.90 s / 30.4** | 1.62× |

That matters beyond the wall clock: at 0.73 TFLOP per step in 8.11 ms the forward
now runs at **~90 TFLOPS, about 78 % of this card's measured GEMM peak** — up from
46 % eager. What is left is FP8 (theoretical 2×) or FlashAttention-3, not
scheduling.

#### Autotuning does not help (measured, not assumed)

`torch.compile` modes, same shape, fresh inductor cache per variant
(`bench/torch_compile.py --compile-modes ...`):

| mode | forward | `generate` (16) | speed-up | compile | inductor cache |
| --- | --- | --- | --- | --- | --- |
| eager | 13.20 ms | 225.9 ms | 1.00× | — | — |
| **`default`** (dynamic=True) | 8.12 ms | **135.1 ms** | **1.67×** | 41 s | 1.2 GB |
| `default` (dynamic=False) | 7.75 ms | 132.8 ms | 1.70× | 15 s | per shape |
| `max-autotune-no-cudagraphs` | 8.45 ms | 136.2 ms | 1.66× | 156 s | 1.9 GB |
| `max-autotune` | 9.14 ms | 226.5 ms | **1.00×** | 156 s | 1.9 GB |
| `reduce-overhead` (dynamic=True) | 172.4 ms | 226.0 ms | **0.08×** | 48 s | — |
| `reduce-overhead` (dynamic=False) | 15.7 ms | 225.9 ms | 1.00× | 0 s | — |

* Triton autotuning finds nothing worth having (0 %) and costs 4× the compile
  time. Its block-size search log is full of 0.03 ms kernels at 80 % of peak —
  the shapes are already well served by cuBLAS.
* The two modes that also enable inductor's own CUDA graphs are **broken for this
  workload**: with symbolic shapes the graph trees fall back every call, and
  `reduce-overhead` added 160 ms to a forward pass. This is the same reason
  `max-autotune` matched eager exactly.
* `dynamic=False` is ~2 % faster and compiles 3× quicker, but it specialises per
  sequence length: with variable-length sentences a new length triggers a 15 s
  inline compile. Only use it if all your chunks are pre-warmed fixed lengths.

Notes:

* Output changes slightly (max |logit diff| 0.25 vs eager) but **not audibly**:
  English 0.00 % WER and Slovak 3.4–7.9 % WER, the same as eager.
* Compilation costs ~40 s on the first start (three warm-up passes covering two
  sequence lengths and both voice modes). With `TORCHINDUCTOR_CACHE_DIR` on a
  volume, a restart takes ~20 s instead of ~48 s.
* CUDA graphs become redundant once inductor has fused the kernels — hence
  `CUDA_GRAPHS=false` by default and `compile + graph` slightly *slower* than
  compile alone (extra buffer copies).

### Batching already-queued chunks

Batching only merges work that is *already waiting*, so it never trades
time-to-first-audio for throughput: the alternative is running those chunks back
to back on the same GPU. Batches are grouped by `num_step`, by "has a clone
prompt" (the model raises on a mixed call) and by length (ratio 1.6, so a short
sentence is not padded up to a long one).

`BATCH_MAX_SIZE` 1 → 8, with `TORCH_COMPILE=true`, average batch 4.9 chunks:

| scenario | batch off | batch on | |
| --- | --- | --- | --- |
| SSE, 8 clients (first chunk) | 1.08 s, 36.7 audio-s/s | **0.89 s, 37.4** | 1.21× |
| non-streaming, 8 clients | 1.26 s, 27.2 audio-s/s | **1.10 s, 30.4** | 1.15× |
| SSE, 1 client | 0.13 s, 33.7 audio-s/s | 0.14 s, 30.8 | unchanged |

The throughput gain is smaller than the per-item micro-benchmark (1.29× at batch
8 with *identical* lengths) because real batches mix lengths and pay padding;
what it buys reliably is **lower latency at concurrency**, i.e. fewer, larger
GPU submissions.

### CUDA graphs for the denoising loop

One sentence is 16 forward passes over identical shapes (`bench/cuda_graphs.py`),
which is exactly what a CUDA graph removes: ~170 kernel launches per pass become
one replay. Measured on the unlocked 3090, fp16, 224-token sequence:

| | eager | CUDA graph | |
| --- | --- | --- | --- |
| one forward pass (`inference_mode`) | 13.24 ms | **11.27 ms** | 1.18× |
| one 16-step sentence, end to end | 225.9 ms | **198.8 ms** | 1.14× |
| max abs diff of logits | — | **0** | bit-identical |

End to end through the HTTP API, same session, graphs on vs off:

| scenario | graphs off | graphs on | |
| --- | --- | --- | --- |
| non-streaming, 1 client | 0.27 s, 16.7 audio-s/s | **0.25 s, 17.4** | 1.08× |
| non-streaming, 8 clients | 2.14 s p50, 16.9 audio-s/s | **1.91 s, 18.5** | 1.12× |
| SSE, 1 client (first chunk) | 0.22 s, 18.3 audio-s/s | **0.19 s, 22.6** | 1.16× |
| SSE, 8 clients (first chunk) | 1.67 s, 22.0 audio-s/s | **1.46 s, 23.9** | 1.14× |
| 28 s text, 1 client | 1.46 s, 19.2 audio-s/s | **1.33 s, 20.8** | 1.10× |

Cost: 11 pre-captured length buckets use **+1.2 GB** of VRAM (≈110 MB each) and
0.21 s of startup; sequences longer than the captured ladder fall back to eager
for free. Quality is unchanged (English 0.00 % / Slovak 1.9–7.5 % WER, same as
eager).

Two implementation notes that are easy to get wrong:

* **Length bucketing is mandatory** — capturing per exact length would thrash.
  Lengths are padded to a multiple of 32 and padded query positions attend only
  to themselves, so real rows are numerically untouched (verified: max |diff| = 0).
* **Only the forward pass is captured.** The sampling between steps has a
  data-dependent `topk(k)`, so it stays eager (~1 ms/step, ~7 %).

On Hopper the launch overhead is a larger share of each step, so expect more than
the 1.14× measured here — upstream reports up to 1.95× from graphs + kernel fusion
on newer parts.

### Precision: fp16 beats bf16 by ~7 % (counter-intuitively)

Direct model calls on the unlocked card, 5-sample medians:

| dtype | `generate` (16 steps) | forward (2×224) | TFLOPS in forward |
| --- | --- | --- | --- |
| **fp16** | **226 ms** | 15.7 ms | 46.3 |
| bf16 | 244 ms | **14.4 ms** | 50.5 |
| fp32 | 566 ms | 32.8 ms | 22.2 |

bf16 has the faster *forward* pass, but a slower *end to end* call: the decode
loop casts logits to fp32 every step (`logits.to(torch.float32)` plus the
guidance math), and bf16→fp32 conversion costs more than fp16→fp32. That is the
same effect the H100 roofline post measured as "sampling 24→78 ms, norms
45→127 ms" after a naive bf16 switch. So `DTYPE=auto` resolves to **fp16**, which
is also what upstream's CLI uses. Quality is equivalent (English 0.00 % WER,
Slovak within run-to-run noise for both).

## Latency and throughput vs diffusion steps

`num_step` is the dominant cost knob; cost is exactly linear in it. One client,
non-streaming, English, ~4.4 s of audio per request:

| `NUM_STEPS` | p50 latency | requests/s | audio-s/s |
| --- | --- | --- | --- |
| 8 | **0.11 s** | 9.02 | 38.9 |
| 12 | 0.14 s | 7.33 | 31.9 |
| **16 (default)** | 0.18 s | 5.80 | 24.5 |
| 24 | 0.25 s | 4.10 | 17.1 |
| 32 | 0.33 s | 3.17 | 13.6 |

Direct model calls (one 3.1 s sentence, prepared 4.5 s reference) show the same
slope with less HTTP overhead: 150 / 226 / 564 ms at 8 / 16 / 32 steps
(RTF 16.5× / 13.7× / 5.5×).

## Concurrency: non-streaming

`c` concurrent clients, each sending its own sentences, `c × 6` requests:

| clients | requests/s | audio-s/s | latency p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| 1 | 5.73 | 25.1 | 0.17 s | 0.19 s | 0.19 s |
| 2 | 7.01 | 29.8 | 0.30 s | 0.31 s | 0.31 s |
| 4 | 7.05 | 34.4 | 0.57 s | 0.58 s | 0.58 s |
| 8 | 7.25 | 31.8 | 1.09 s | 1.10 s | 1.19 s |
| 16 | 7.40 | 31.7 | 2.16 s | 2.18 s | 2.19 s |

Throughput is **flat** and latency grows **linearly** in concurrency: the GPU is
the bottleneck (this 3090 renders ~25 audio-seconds/s at 16 steps for one client,
~32 with batching) and the scheduler divides it evenly. That is the intended shape — no client is starved, none runs
ahead, and the p99/p50 spread is < 10 %.

## Concurrency: streaming (SSE)

Same corpus, `stream_format=sse`; "first chunk" is request start → first audio
delta, i.e. what a listener actually experiences:

| clients | requests/s | audio-s/s | first chunk p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| 1 | 7.10 | 33.1 | **0.14 s** | 0.15 s | 0.15 s |
| 2 | 8.34 | 38.3 | 0.25 s | 0.25 s | 0.25 s |
| 4 | 8.67 | 36.8 | 0.45 s | 0.50 s | 1.07 s |
| 8 | 8.91 | 37.5 | 0.91 s | 0.92 s | 0.94 s |
| 16 | 9.10 | 41.6 | 1.74 s | 1.79 s | 1.80 s |

Streaming is ~30 % cheaper than non-streaming per request (the first chunk uses
12 steps, and there is no whole-body buffering), and the first-chunk latency stays
well under the duration of the audio itself up to 16 concurrent listeners — each
listener's playback does not run dry (0.91 s to first audio for 4.5 s of speech at
8 clients).

**Fairness check.** With 8 long streams already running, a newcomer's first
chunk arrived in **2.11 s**, while the eight long requests' own first chunks were
spread over 0.22–2.28 s. The newcomer waits behind *first chunks* (and one
follow-up chunk per `FIRST_CHUNK_BURST`), not behind 8 × 28 s of queued audio —
without the burst limit it was 1.35 s, but then a sustained flood of new requests
could freeze sessions that are already playing.

## WebSocket (LLM-style input)

One word per text frame, 8 words/s (a realistic LLM token rate). Latency includes
waiting for the client to finish the first sentence:

| clients | first chunk p50 | p99 | total p50 | audio-s/s |
| --- | --- | --- | --- | --- |
| 1 | 1.26 s | 1.27 s | 1.27 s | 2.9 |
| 4 | 1.25 s | 1.51 s | 1.25 s | 11.4 |
| 8 | 1.33 s | 1.87 s | 1.33 s | 20.9 |

The 1.4 s floor is the client's own speech rate: synthesis cannot start before a
clause exists. In the bundled UI's "simulate LLM" mode at 60 tok/s the first
audio arrives in ~0.4 s.

## Slovak

Same hardware, `kroner` (male, Slovak) reference:

| clients | mode | requests/s | audio-s/s | p50 |
| --- | --- | --- | --- | --- |
| 1 | full | 4.10 | 19.4 | 0.24 s |
| 1 | SSE | 4.57 | 22.8 | 0.18 s |
| 4 | full | 4.87 | 19.9 | 0.83 s |
| 4 | SSE | 6.42 | 28.0 | 0.61 s |

Slovak renders slightly *fewer* audio-seconds per wall second than English at the
same step count (~20-28 vs ~25-38) because its sentences are longer; the scheduling
behaviour is identical.

## Long inputs

One request = 21 chunks = ~28 s of audio:

| clients | audio-s/s | p50 total | p99 |
| --- | --- | --- | --- |
| 1 | 31.0 | 0.91 s | 0.95 s |
| 4 | 37.3 | 2.95 s | 3.20 s |
| 8 | 37.0 | 6.15 s | 6.24 s |

Long inputs do not reduce throughput (31-37 audio-s/s): they are
split into sentences like everything else, and `MAX_INFLIGHT_PER_SESSION=3` keeps a single
long request from occupying the whole queue.

## Overload and admission control

64 concurrent clients, long corpus (≈1800 s of audio of work):

| | |
| --- | --- |
| admitted | 11 requests |
| rejected | 53 × `503` + `Retry-After`, returned in milliseconds |
| admitted p50 | 8.8 s (bounded by the 180 s pending-audio cap ≈ queue wait + render) |
| wall clock | 14.0 s |

Without admission control the same test produced a 50 s p50 latency and a queue
that kept growing (measured before `MAX_PENDING_AUDIO_S` was tuned to 180).

## Soak

60 s at 8 concurrent SSE clients over a mixed EN/SK/long corpus:

| | |
| --- | --- |
| requests | 300+ (no errors, no `503`) |
| throughput | 5.95 req/s, 24.8 audio-s/s |
| first chunk | p50 0.98 s, p90 1.37 s, p99 1.90 s |
| total | p50 1.00 s, p99 6.0 s (the long text dominates) |
| VRAM after | 8.4 GB of 24 GB, GPU 77 % |

## Quality of the served audio

`bench/verify_quality.py` synthesises each sentence through the real API, then
transcribes the result with the Parakeet ASR service and scores WER (numbers
folded, punctuation/case ignored).

English, 6 sentences per case:

| mode | wav | mp3 | opus | flac | pcm |
| --- | --- | --- | --- | --- | --- |
| full | **0.0 %** | 0.0 % | 1.9 % | 0.0 % | 0.0 % |
| SSE | **0.0 %** | 0.0 % | – | – | – |
| WebSocket | 0.0–5.0 % | – | – | – | – |

Slovak, 6 sentences per case:

| mode | wav | mp3 | opus | flac | pcm |
| --- | --- | --- | --- | --- | --- |
| full | 7.9 % | 13.9 % | 3.9 % | 10.2 % | 8.1 % |
| SSE | 3.7 % | 1.9 % | – | – | – |
| WebSocket | 6.0 % | – | – | – | – |

Notes:

- English is word-perfect in every format; the codecs are transparent at these
  bitrates.
- Slovak varies 2–14 % between runs with *identical* settings. Generation is
  stochastic (`position_temperature=5.0`), and six sentences make a noisy
  sample: one wrong word in eight is 12 %. The differences between formats are
  noise, not codec damage.
- Steps barely move WER in this range (16 steps: 7.9 %, 32 steps: 11.5 %, 8
  steps: 8.1 % on the same Slovak set) — intelligibility saturates long before
  naturalness does. Use more steps for prosody, not for accuracy.
- Voice choice matters more than steps: the same Slovak sentences through
  `fico` scored 12.7 % vs 5.8 % through `kroner`. Reference quality (clean
  single speaker, transcript that matches the clip) dominates.

## Micro-benchmarks (direct model calls)

`bench/raw_model.py`, 3 s English sentence, prepared 4.5 s reference:

| question | result |
| --- | --- |
| cost vs steps | 150 / 218 / 287 / 425 / 563 ms at 8 / 12 / 16 / 24 / 32 steps — perfectly linear |
| cost vs text length | RTF *improves* with length (10.5× at 43 chars → 23.0× at 351 chars): the reference prompt is a fixed per-call cost |
| batching 8 identical sentences | 288 → 224 ms per item (−22 %); SM utilisation 96 % → 99 %, 42 → 55 TFLOPS, i.e. a ~1.3× ceiling, not 8× |
| batching mixed lengths | pad to the longest member; a 2-item batch of short+long cost 8.9 s vs 1.3 s + 7.6 s separately (no gain, more latency) |
| cached voice prompt vs re-encoding | 287 vs 309 ms per call; `create_voice_clone_prompt` alone is 21.7 ms |
| guidance_scale = 0 | no speed-up upstream (the uncond branch is always computed); a patched decoder that skips it is 1.63× faster but ~2 WER points worse for Slovak |
| encoding | wav 0.6 ms, pcm 0.1 ms, flac 1.6 ms, mp3 48 ms, opus 51 ms, aac 94 ms per sentence |
| model load / warm-up | 1.1 s from a warm HF cache, 0.7 s warm-up (2 inferences) |
| VRAM | 1.95 GB allocated after load, 6.4 GB reserved under load |

## Can the model batch? (measured, and what upstream does)

`generate()` accepts a list and runs a genuine batched forward —
`(2B, 8, L_max)` per denoising step — so batching is supported by the
architecture. The question is how much it can buy. `bench/batch_probe.py`
separates the GPU work from the host-side loop:

| batch | one denoising step (forward) | per item | 16-step `generate()` | per item | SM util | power | effective TFLOPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 ms | 13 ms | 227 ms | 227 ms | 76 % | 149 W | 54.4 |
| 2 | 25 ms | 12 ms | 400 ms | 200 ms | 100 % | 276 W | — |
| 4 | 43 ms | 11 ms | 747 ms | 187 ms | 95 % | 322 W | — |
| 8 | 82 ms | 10 ms | 1411 ms | 176 ms | 99 % | 327 W | 71.0 |

Two conclusions:

* **Batch 1 leaves the tensor cores half idle, but batching barely helps**: 54
  TFLOPS of this card's measured ~117 TFLOPS GEMM peak (46 %). Batching to 8
  reaches 71 TFLOPS (61 %), i.e. the ceiling is **~1.3×**, and that is what the
  end-to-end number shows (6.12× wall for 8× work). The residual gap is the short
  sequence (224 tokens → small matrices) plus the non-tensor part of each step.
* **Forward and `generate()` scale identically** (6.12× vs 6.22× at batch 8), so
  the per-item Python sampling loop is *not* eating a batching win — the decode
  loop is bound by the same per-token work.

Why the ceiling is so low: this is a diffusion LM. Each request costs
`num_step` full passes over the whole `style + text + reference + target`
sequence, the mask is bidirectional so nothing can be KV-cached across steps,
classifier-free guidance doubles the batch unconditionally, and every item runs
exactly `num_step` steps (no early exit). Batching only improves tensor-core
utilisation on top of that.

Upstream reaches the same conclusion in practice:

* `omnivoice-infer-batch` (the official CLI) **clusters samples by estimated
  duration** before forming fixed-size batches, and separates voice-clone from
  auto-voice samples (mixing them raises inside `generate()`). It parallelises
  across GPUs with one *process* per device rather than one big forward.
* `vllm-omni` (their production engine) does continuous batching
  (`BATCH_SIZE=16`, 10 ms batch wait, `duration_bucket` strategy) — and the
  single biggest win in
  [this roofline analysis](https://www.pavankatta.com/diffusion-tts-roofline/)
  was not batching at all: the baseline was silently running **FP32**; BF16 made
  the audio head 8× faster. The remaining bottleneck there was kernel launch
  overhead, addressed with CUDA graphs and kernel fusion (up to 1.95× on newer
  cards). We already run fp16.

So for *this* server batching is a deliberate non-goal: a duration-bucketed
batcher over follow-up chunks would add ~1.2–1.3× throughput at the cost of
time-to-first-audio, which is the metric an interactive service is judged on.
The levers that do move the needle are `num_step` (linear), the first-chunk step
count, and — untested here — CUDA graphs / `torch.compile` around the step loop,
which is where upstream found its largest remaining gain.

## What these numbers mean for sizing

- **Interactive chat (streaming)**: one 3090 serves 16 concurrent listeners with
  1.7 s to first audio (0.14 s at one client) and no playback underrun at 16
  steps; ~40 audio-seconds/s at that concurrency.
- **Batch/offline use**: ~25 audio-seconds/s per GPU at 16 steps for one client,
  ~39 at 8 steps, ~14 at 32 steps. A 10-minute audiobook (600 s of audio) takes
  ~24 s at 16 steps.
- **Latency floor**: ~0.11–0.18 s per sentence, of which ~20 ms is text
  splitting, ~0.6–50 ms encoding (format-dependent) and the rest is GPU compute.
- **Scaling out**: add a GPU, not a worker. The model saturates one card at batch
  size 1, so a second instance on the same GPU only splits the same FLOPs.
