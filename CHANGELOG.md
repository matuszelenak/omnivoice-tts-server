# Changelog

## 0.1.0 — production TTS engine

The server was rewritten around the reference PyTorch model instead of
`audio.cpp`, and measured end to end on an RTX 3090.

### Throughput and latency (one 3090, fp16, 16 diffusion steps)

| | before | 0.1.0 |
| --- | --- | --- |
| first audio, one client | 0.34 s | **0.15 s** |
| first audio, 8 streaming clients | 2.03 s | **0.86 s** |
| audio rendered per wall second | 13.4 | **42** (8 clients) |
| long text (28 s of audio), one client | 1.85 s | **0.91 s** |
| overload | unbounded queueing | **`503` + `Retry-After` in ms** |

### Server

- **Fair scheduler**: the unit of work is a sentence. First chunks are dispatched
  ahead of other requests' follow-ups (in bounded bursts so nobody starves),
  follow-ups round-robin across sessions, admissions are bounded by queued jobs
  *and* queued audio seconds, and a client that disconnects drops its work.
- **Batching** (`BATCH_MAX_SIZE=8`): already-queued chunks share one model call,
  grouped by step count, clone/auto mode and length (1.15–1.2× latency at
  concurrency, no effect for a single client).
- **`torch.compile` on the backbone** (`TORCH_COMPILE=true`): 1.65× end to end,
  logits within 0.25 of eager and identical WER. Autotuned modes measured
  *slower* — see `docs/benchmarks.md`.
- **CUDA-graph replay** available (`CUDA_GRAPHS=true`): +14 % on an uncompiled
  model, redundant once inductor has fused the kernels.
- **OpenAI-compatible API**: `POST /v1/audio/speech` (whole body or SSE audio
  deltas), `WS /v1/ws/synthesize`, `/v1/models`, `/v1/languages`, voice CRUD,
  `/health`, `/readyz`, `/metrics`. Extension fields: `language`,
  `stream_format`, `num_step`, `sanitize`.
- **Voice store**: references are trimmed, cut on an ASR word boundary and their
  transcript is verified/repair-transcribed, which fixed cloned voices reading
  the *reference* text first (WER 102 % → 0 %). Prompts are precomputed and
  LRU-cached.
- **Text sanitisation** (ported from `main`): an optional OpenAI-compatible LLM
  rewrites digits, abbreviations and symbols into spoken words, per chunk so
  streaming stays fast, falling back to the raw text on any failure.
- **Observability**: Prometheus metrics (queue depth, queued audio, batch sizes,
  per-chunk latency, realtime factor, first-chunk latency, rejections,
  sanitisation), structured logs, and a `/health` payload with GPU, queue, voice
  and ASR state.
- **Tests**: 96 unit/API tests that stub the model, so they run without a GPU in CI.

### Operations

- Development and production images (granian, one worker), compose stacks with
  `env_file`, `Makefile` (`build/run/status/plan/probe/bench/test/lint`), CI
  lint+test gate, and the `v*` tag release pipeline.
- `make plan` measures a machine and prints the admission limits it should run
  with; the shipped defaults are calibrated for one RTX 3090.
- Docs: API, configuration, deployment (including H200 notes), benchmarks
  (methodology + every measurement), and pitfalls (each failure and its fix).
