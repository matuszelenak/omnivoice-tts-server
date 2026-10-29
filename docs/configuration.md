# Configuration

Everything is an environment variable, read once at startup (see
`server/app/config.py`). `docker-compose.prod.yml` forwards the ones you normally
touch; anything else can be added there or passed with `docker run -e`.

## Model and device

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_NAME` | `k2-fsa/OmniVoice` | Hugging Face model id or local path |
| `SERVED_MODEL_NAME` | `omnivoice` | Id reported by `/v1/models` |
| `DEVICE` | `cuda:0` | Torch device (`DEVICE_MAP` is accepted as a legacy alias) |
| `DTYPE` | `auto` | `auto` = fp16 on CUDA, fp32 on CPU. fp16 measured ~7 % faster end-to-end than bf16 on a 3090 (the sampler casts logits to fp32 each step, and bf16->fp32 casts are slower); upstream's CLI also defaults to fp16. Set `bfloat16` if a fine-tune overflows in fp16 |
| `WARMUP` | `true` | Run two throwaway inferences at startup (avoids a slow first request) |
| `HF_HOME` | `/cache/hf` (image) | Hugging Face cache; mount it to avoid re-downloading |
| `HF_TOKEN` | — | Token for gated models |
| `STATIC_DIR` | `/app/static` (image) | Built frontend to serve; empty disables static serving |
| `LOG_LEVEL` | `info` | `debug` also logs per-chunk dispatch decisions |

**One GPU per instance.** The model saturates a 3090 at batch size 1, so a second
replica on the same GPU adds VRAM and context switching but no throughput.
Run additional instances only on additional GPUs (`DEVICE=cuda:1`, separate
ports, separate `container_name`). The sibling parakeet ASR container should own
the other GPU.

## Decoding: the quality/latency dial

| Variable | Default | Meaning |
| --- | --- | --- |
| `NUM_STEPS` | `16` | Diffusion steps per chunk. Cost is linear: 8 steps ≈ 2× faster than 16, 32 ≈ 2× slower |
| `FIRST_CHUNK_STEPS` | `12` | Steps for the first chunk of a *streamed* request (faster first audio). `0` = same as `NUM_STEPS` (uniform first-chunk and follow-up latency; CUDA-graph captures are keyed by sequence length, so this does not matter for them) |
| `MAX_NUM_STEPS` | `64` | Upper bound for the per-request `num_step` override |
| `GUIDANCE_SCALE` | `2.0` | Classifier-free guidance. `0` measured 1.6× faster with a patched decoder but ~2 WER points worse for Slovak |
| `TORCH_COMPILE` | `true` | Compile the Qwen3 backbone with inductor: **1.6–1.75× end-to-end** (measured), logits within 0.25 of eager, identical WER. Costs ~40 s on the first start, cached afterwards |
| `COMPILE_MODE` | `default` | Passed to `torch.compile`. Keep `default`: `max-autotune` measured **1.00×** (its CUDA-graph trees fall back) and `reduce-overhead` **0.08×** on the forward pass; `max-autotune-no-cudagraphs` matched `default` at 4× the compile time. See [benchmarks](benchmarks.md#torchcompile-on-the-backbone-the-biggest-single-win) |
| `CUDA_GRAPHS` | `false` | Replay the forward pass as a captured graph. +14 % on an *uncompiled* model, but nothing once inductor has fused the kernels, so off by default; graphs only apply to single-item calls |
| `GRAPH_BUCKET` | `32` | Lengths are padded to this multiple so one graph serves a range of sentences |
| `GRAPH_MAX` | `12` | Maximum captured lengths (~110 MB each) |
| `GRAPH_MIN_FREE_MB` | `4096` | Do not capture when free VRAM is below this |
| `T_SHIFT` | `0.1` | Time-step shift; smaller emphasises low-SNR steps |
| `POSTPROCESS_OUTPUT` | `true` | Silence removal and edge fades on each chunk |

Measured (RTX 3090, fp16, English, 4.5 s of audio per request, one client):

| `NUM_STEPS` | latency | throughput | audio-s/s |
| --- | --- | --- | --- |
| 8 | 0.19 s | 5.5 req/s | 23 |
| 12 | 0.26 s | 3.9 req/s | 17 |
| 16 (default) | 0.34 s | 3.0 req/s | 12.5 |
| 24 | 0.51 s | 2.0 req/s | 8.9 |
| 32 | 0.67 s | 1.5 req/s | 6.4 |

Use 16–24 for interactive use, 8–12 when throughput matters more than the last
bit of naturalness, 32 for offline rendering.

## Text handling

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_INPUT_CHARS` | `4000` | Larger requests get `413`; split long documents client-side |
| `MAX_CHUNK_CHARS` | `320` | Hard cap per model call; longer sentences are cut at clause breaks |
| `MIN_CHUNK_CHARS` | `8` | WebSocket fragments shorter than this wait for more text (avoids "Yes." costing a full diffusion pass) |

## Admission control and scheduling

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_QUEUE_JOBS` | `48` | Queued chunks before `503` |
| `MAX_PENDING_AUDIO_S` | `180` | Queued *audio seconds* before `503` — the real capacity bound |
| `QUEUE_TIMEOUT_S` | `120` | A chunk that waits longer fails with `503` |
| `RETRY_AFTER_S` | `2` | `Retry-After` value on `503` |
| `MAX_INFLIGHT_PER_SESSION` | `3` | Chunks of one HTTP request kept in the queue at once |
| `BATCH_MAX_SIZE` | `8` | Chunks decoded in one model call (`1` = off). Only work that is *already queued* is merged; measured 1.15–1.2× latency under load, no effect at one client |
| `BATCH_WINDOW_MS` | `8` | How long a lone queued chunk waits for a partner before running alone |
| `BATCH_MAX_TOKENS` | `4096` | Summed sequence length per batch (VRAM bound) |
| `BATCH_LENGTH_RATIO` | `1.6` | Longest/shortest length allowed in one batch (padding cost) |
| `BATCH_FIRST_CHUNKS` | `true` | Let first chunks join a batch — only ever when they are already queued behind others |
| `FIRST_CHUNK_BURST` | `4` | First chunks served back-to-back before a queued follow-up chunk gets a turn (keeps TTFA low without starving sessions that are already playing; `0` = strictly first-chunks-first) |

`MAX_PENDING_AUDIO_S` is the knob that keeps latency honest. The GPU renders
~14 audio-seconds/s, so 180 s of queued audio means an admitted request waits at
most ~12 s before its first chunk is rendered. Raise it to admit more requests at
the cost of longer waits; lower it to shed load earlier.

## WebSocket sessions

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_STREAM_SESSIONS` | `64` | Concurrent sessions; further upgrades are closed with `1013` |
| `STREAM_MAX_PENDING_S` | `180` | Unsynthesised text a client may buffer before `1013` |
| `STREAM_MAX_SESSION_S` | `3600` | Hard session lifetime |

## Text sanitisation (optional)

OmniVoice mispronounces digits, abbreviations and symbols. An optional
OpenAI-compatible LLM rewrites each chunk into spoken language first — in the
input language (the prompt forbids translation) — before it reaches the model.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SANITIZE_LLM_BASE_URL` | _(empty = off)_ | OpenAI-compatible endpoint, e.g. `http://llm:4000/v1` |
| `SANITIZE_LLM_API_KEY` | `none` | Key for that endpoint |
| `SANITIZE_LLM_MODEL` | `gpt-4o-mini` | Model id to call |
| `SANITIZE_LLM_TIMEOUT_S` | `10` | Per-chunk timeout; on expiry the raw text is used |

Sanitisation runs **per chunk, not per request**, so a long text does not delay
the first sentence, and it is pipelined with synthesis (the next chunk is cleaned
while the GPU renders the previous one). Any failure — timeout, HTTP error, empty
answer — falls back to the raw text, so it can never break a request. Costs one
LLM round trip (~100–400 ms) per chunk; clients can opt out per request with
`sanitize=false`. Counters: `tts_sanitize_total{outcome}`, `tts_sanitize_seconds`.

## Voices

| Variable | Default | Meaning |
| --- | --- | --- |
| `VOICE_SAMPLES_DIR` | `/app/voices` | Voice catalogue (must be writable to register voices) |
| `MAX_REF_SECONDS` | `8` | References are cut to this length (the model degrades beyond ~10 s) |
| `MIN_REF_SECONDS` | `2` | Shorter clips are rejected |
| `MAX_UPLOAD_MB` | `25` | Reference upload limit |
| `ASR_URL` | `http://localhost:9000` | Parakeet ASR service used to transcribe/repair reference clips; empty disables it (then `ref_text` is required) |
| `ASR_TIMEOUT_S` | `60` | ASR request timeout |
| `PROMPT_CACHE_SIZE` | `64` | Voice-clone prompts kept in memory (each ~1 MB) |
| `PREWARM_VOICES` | `true` | Build prompts for all built-in voices at startup (~40 ms each) |

## HTTP / observability

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` / `PORT` | `0.0.0.0` / `9001` | Bind address (the image overrides the port in its CMD) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `ENCODE_WORKERS` | `0` (auto) | Threads for mp3/opus/aac encoding (`0` = min(8, CPU count)). At high GPU throughput the encoders, not the model, become the CPU bottleneck |
| `TORCHINDUCTOR_CACHE_DIR` | `/cache/inductor` (image) | Where inductor caches compiled kernels — mount it, or every restart pays ~40 s |
| `METRICS_ENABLED` | `true` | Serve `/metrics` and record HTTP metrics |
| `LOGFIRE_TOKEN` | — | Send structured traces to Logfire when set |

## Frontend (build time)

| Variable | Meaning |
| --- | --- |
| `VITE_API_BASE` | API origin if the UI is served from a different host (default: same origin) |
| `API_PROXY_TARGET` | Dev only: where Vite proxies `/v1` (default `http://server:9001`) |
