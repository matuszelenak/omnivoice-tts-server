# Deployment

## Images

| Image | Dockerfile | Contents | Use |
| --- | --- | --- | --- |
| `omnivoice-tts:prod` | `prod.Dockerfile` | API + model deps + baked Svelte UI, served by **granian** on `:9001` | production |
| dev image | `server/Dockerfile` | API only, `uvicorn --reload` | development |

The production image is **~6.0 GB** (was 7.5 GB). Model weights are **not** baked
in; they are downloaded on first start into `HF_HOME` (`/cache/hf`).

Where the size goes, and what is deliberately absent:

| | size |
| --- | --- |
| `torch` (CUDA build) | 1.1 GB |
| `nvidia-*` CUDA libraries (cuBLAS/cuDNN/cuFFT/cuSPARSE/NCCL/…) | 2.5 GB |
| `triton` (needed by `torch.compile`) | 0.6 GB |
| everything else (transformers, numpy, scipy, blingfire, …) | ~1.0 GB |
| OS + ffmpeg + Python | ~0.8 GB |

The build is three stages. The official `uv` image is a **full build environment**
(gcc, git, LLVM, dev headers, its own Python 3.13), so it is used only to resolve
and install wheels into `/app/.venv`; the shipped image is `python:3.12-slim`
plus that venv. On top of that the build prunes what inference never touches:
`torch/include` (C++ headers), `libnvrtc.alt` / `libcusolverMg` / `libnvperf`
(archives nothing links against), triton's static profiler archives, pip and
setuptools, `/usr/share/{doc,man,locale}`, and the Gradio/pandas/scikit-learn
stack that `omnivoice` drags in for its demo and eval tooling.

Two consequences worth knowing:

* **`gcc` stays in the runtime image on purpose.** `torch.compile` compiles a
  small C launcher per kernel through Triton, so a compiler is required at run
  time (`Failed to find C compiler` otherwise). If you build with
  `TORCH_COMPILE=false` you can drop `gcc`/`libc6-dev` and save another ~190 MB.
* The CUDA libraries come from pip wheels, so `ldconfig` is run at build time to
  register `/app/.venv/lib/**/nvidia/*/lib` — otherwise `libcudart` is not found
  at import.

```bash
make build                       # docker build -f prod.Dockerfile -t omnivoice-tts:prod .
docker run --gpus '"device=0"' -p 9001:9001 \
  -v /mnt/nvme/huggingface:/cache/hf \
  -v $(pwd)/server/voices:/app/voices \
  -e ASR_URL=http://host.docker.internal:9000 \
  --add-host host.docker.internal:host-gateway \
  omnivoice-tts:prod
```

## GPU assignment

One instance per GPU. OmniVoice saturates a 3090 at batch size 1 (see
[benchmarks](benchmarks.md)), so a second process on the same card only splits
the same FLOPs.

| GPU | Owner | Port |
| --- | --- | --- |
| 0 | OmniVoice TTS | 9001 |
| 1 | parakeet-asr-server (ASR) | 9000 |

`docker-compose.prod.yml` selects the card with `GPU` (default `0`). The TTS
container and the ASR container are separate compose projects, so the TTS
container reaches ASR through `host.docker.internal` (`extra_hosts` in the
compose file). If you run both in one compose project, use the service name
instead: `ASR_URL=http://parakeet:9000`.

## Startup sequence

1. `engine.load()` — model + tokenizers onto the GPU (≈1–2 s with a warm cache,
   plus the HF download on a cold one), then `torch.compile` is attached to the
   backbone.
2. `engine.warmup()` — three throwaway inferences; this is where inductor
   compiles the backbone (~40 s cold, ~20 s with a warm `TORCHINDUCTOR_CACHE_DIR`)
   and where CUDA graph buckets would be captured if enabled.
3. `scheduler.start()` — the dispatch loop.
4. `voices.prewarm()` — one voice-clone prompt per built-in voice (≈40 ms each).
5. `GET /readyz` flips to `200`. The container healthcheck waits for it
   (`start_period: 600s` covers a cold download).

`GET /health` stays available the whole time and reports `"status": "loading"`
until step 5.

Shutdown (SIGTERM) stops the dispatcher, cancels queued chunks (they resolve as
`503`/cancelled), shuts the GPU thread down and closes the ASR client.

## Capacity planning

Measured on one RTX 3090, fp16, `NUM_STEPS=16`, English, ~4.5 s of audio per
request:

| | value |
| --- | --- |
| audio rendered per wall second | **≈25 audio-s/s** (≈39 at 8 steps, ≈14 at 32) |
| requests/s (4.5 s of audio) | ≈5.7 (≈7.3 with batching) |
| concurrent streams that keep up with real time | ≈25 (≈39 at 8 steps) |
| first-audio latency, one client | 0.11–0.18 s |
| first-audio latency, 8 concurrent clients | ≈0.9 s (≈0.11 s each, round-robin) |
| VRAM | ≈8.4 GB (compile + 11 CUDA-graph buckets) / 2 GB weights |

A "stream that keeps up" means sustained synthesis at 1× real time or faster.
Above that the queue grows; admission control turns the excess into fast `503`s.

Run `make plan` on the target hardware to print these three values for the
measured throughput instead of guessing — the shipped numbers are calibrated for
one RTX 3090 and are deliberately conservative.

Tune for your product:

* interactive chat, few users → `NUM_STEPS=16..24`, `FIRST_CHUNK_STEPS=12`,
  `MAX_PENDING_AUDIO_S=120`
* many concurrent short utterances → `NUM_STEPS=8..12` (≈2× capacity)
* offline rendering → `NUM_STEPS=32`, high `MAX_PENDING_AUDIO_S`, large batches of
  requests from one client

## Target hardware: H200 (and what changes from a consumer GPU)

The development box is a 2 x RTX 3090 node; the deployment target is an H200.
The server needs no code changes, but several defaults stop being right. An H200 is
~989 TFLOPS dense fp16/bf16 with 4.8 TB/s of bandwidth and 141 GB; the reference
3090 measures ~117 TFLOPS in a real GEMM, so this is roughly **7-8x** the matrix
throughput (careful with datasheet numbers: GeForce specs often quote the
FP32-accumulate half rate — see [benchmarks](benchmarks.md#hardware-unlock-gpu-unlock--what-it-changed)).
One instance uses ~6.4 GB, so VRAM is a non-issue.

**The model will be nowhere near compute-bound at batch size 1 on Hopper.** On the
3090 a single sentence reaches 54 of ~117 TFLOPS (46 %) and only 76 % SM
utilisation (`bench/batch_probe.py`), so batching already recovers just 1.3x here;
on Hopper the per-token work shrinks further while kernel launches do not. On Hopper the
same 224-token step is limited by kernel launch overhead and small-matrix
efficiency instead — the published H100 roofline finds attention at 57 % of
runtime and *launch-overhead bound*. So expect to revisit these decisions, in
this order:

1. **`torch.compile` is already on** (`TORCH_COMPILE=true`) — the single biggest
   win measured on this workload (1.65×). Keep `TORCHINDUCTOR_CACHE_DIR` on a
   persistent volume or every restart pays ~40 s; watch the warm-up time in the
   logs to confirm compilation is happening there.
2. **Batching is already on** (`BATCH_MAX_SIZE=8`, average batch ~5 under load).
   On a GPU where per-item efficiency matters more this has more headroom: watch
   the `tts_batch_size` histogram and raise `BATCH_MAX_SIZE` /
   `BATCH_MAX_TOKENS`, then re-measure with `bench/run_all.sh`.
3. **Then measure** (`make probe`, stops the container for a minute). If step-1
   throughput is still far from the card's GEMM peak, the remaining levers are
   kernel-level (FlashAttention-3, FP8), not scheduling.
4. **Raise quality for free.** On a card this fast, `NUM_STEPS=24..32` costs less
   in absolute terms than 16 steps does on the 3090 (3090: 4.5 audio-s/s at 32
   steps — expect roughly an order of magnitude more on H200). Verify with
   `bench/verify_quality.py`.
5. **Keep the encoder fed.** At 100+ audio-seconds/s, mp3/opus encoding
   (~50-95 ms per chunk) reaches ~1.5 CPU cores per 100 streams. `ENCODE_WORKERS`
   defaults to `min(8, cpus)`; ask clients for `pcm`/`wav` when you control them.
6. **Re-derive the admission limits** — do not copy them from this repository:

   ```bash
   make run && make plan      # measures audio-s/s and prints the env block to set
   ```

   `MAX_PENDING_AUDIO_S`, `MAX_QUEUE_JOBS` and `MAX_STREAM_SESSIONS` are all
   "how much work should be allowed to queue"; the right numbers scale linearly
   with measured throughput.

7. **Multi-GPU nodes.** One instance per GPU (`GPU=0..7`, `HOST_PORT=9001..`), all
   pointing at the same ASR service and voice catalogue, behind a load balancer.
   `/v1/audio/speech` (including SSE) needs no affinity; WebSocket sessions do.

Other H200-specific notes:

- `DTYPE=auto` resolves to **fp16**; on Ampere, fp16 measured ~7 % faster than
  bf16 end-to-end because the sampler casts logits to fp32 every step. Re-measure
  on the H200 with `make plan` for each dtype before locking it in.
- **FP8** (E4M3, ~2x fp16) is not implemented. The sampler casts logits to fp32
  and diffusion decoding is precision-sensitive, so it needs a WER evaluation
  before it is worth the risk — `bench/verify_quality.py` is the harness.
- Nothing in the server is hard-coded to a GPU count or model family:
  `DEVICE`, `DTYPE`, `NUM_STEPS`, admission limits and the encode pool are all
  environment-driven.

## Health and readiness

* `GET /readyz` → `200 ok` / `503 loading` (use for Kubernetes readiness and
  container healthchecks).
* `GET /health` → JSON with `ready`, `engine` (device, dtype, VRAM, prompt cache),
  `scheduler` (queued chunks, queued audio seconds, inflight), `voices`, `asr`,
  `stream_sessions`, and the active decoding settings.

Example Kubernetes probes:

```yaml
readinessProbe:
  httpGet: {path: /readyz, port: 9001}
  periodSeconds: 5
  failureThreshold: 3
livenessProbe:
  httpGet: {path: /readyz, port: 9001}
  initialDelaySeconds: 60
  periodSeconds: 30
  failureThreshold: 5
resources:
  limits: {nvidia.com/gpu: 1}
```

## Metrics

Prometheus exposition on `/metrics` (`METRICS_ENABLED=true`), all prefixed
`tts_`:

| Metric | Type | Use |
| --- | --- | --- |
| `tts_queue_depth` | gauge | chunks waiting for the GPU |
| `tts_pending_audio_seconds` | gauge | queued audio — the saturation signal |
| `tts_inflight_jobs` | gauge | 0/1 for this single-worker engine |
| `tts_queue_wait_seconds` | histogram | how long a chunk waited |
| `tts_inference_seconds{kind,language}` | histogram | per-chunk GPU time |
| `tts_realtime_factor{kind}` | histogram | audio-s produced per wall-s |
| `tts_first_chunk_seconds{kind}` | histogram | request start → first audio |
| `tts_rejected_total{reason}` | counter | `queue_full`, `queue_timeout`, `stream_limit`, `stream_backpressure`, `cancelled` |
| `tts_stream_sessions_active` | gauge | live WebSocket sessions |
| `tts_prompt_cache_hits_total` / `_misses_total` | counter | voice prompt cache |
| `tts_gpu_memory_bytes{kind}` | gauge | `loaded`, `warm` |
| `tts_http_requests_total{route,method,status}` | counter | API traffic |

Useful alerts:

```yaml
- alert: TTSQueueSaturated
  expr: tts_pending_audio_seconds > 120
  for: 2m
- alert: TTSRejectingRequests
  expr: rate(tts_rejected_total[5m]) > 1
  for: 5m
- alert: TTSFirstChunkSlow
  expr: histogram_quantile(0.9, rate(tts_first_chunk_seconds_bucket[5m])) > 3
  for: 5m
- alert: TTSModelNotReady
  expr: tts_model_ready == 0
  for: 5m
```

## Multiple instances / load balancing

* State (scheduler, session counters, prompt cache, metrics) is **per process**.
  Run one granian worker per instance; scale by adding instances on other GPUs.
* Any HTTP load balancer works for `/v1/audio/speech` (whole body or SSE). Disable
  proxy buffering for SSE (`X-Accel-Buffering: no` is already set) and keep
  read timeouts ≥ 120 s.
* WebSocket sessions must reach the same instance for their lifetime: use sticky
  sessions or a load balancer that pins upgraded connections.
* Point every instance at the same ASR service and the same voice catalogue if
  you want identical voices everywhere.

## Voice catalogue

`VOICE_SAMPLES_DIR` is the source of truth (default `/app/voices`). Mount it as a
volume so registrations survive redeploys:

```yaml
volumes:
  - /srv/omnivoice/voices:/app/voices
```

Contents per voice: `voice.wav`, `<lang>.txt`, `meta.json`, plus `source.wav`
(the untouched upload). Back up the directory; there is nothing else to persist.

`scripts/prepare_voices.py` re-normalises every voice (trim, word-boundary cut,
transcript via ASR) — useful after adding clips by hand or upgrading the ASR
service.

## Security

The server has no authentication. Put it behind your gateway (LiteLLM,
Traefik, nginx, an API gateway) and:

* restrict `CORS_ORIGINS` to your UI's origin,
* keep `MAX_INPUT_CHARS`, `MAX_UPLOAD_MB` and `MAX_STREAM_SESSIONS` at values
  your hardware can absorb — they are the abuse limits,
* do not expose `VOICE_SAMPLES_DIR` writes to untrusted users (voice
  registration writes files as the container user).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `/readyz` 503 forever | model download failed / no GPU | check `docker logs`, `nvidia-smi`, `HF_TOKEN` |
| `503 queue is full` under light load | `MAX_PENDING_AUDIO_S` too low for long texts | raise it, or send fewer/short texts |
| First audio slow (>2 s) at low load | `FIRST_CHUNK_STEPS`/`NUM_STEPS` too high, or cold prompt cache | lower steps; `PREWARM_VOICES=true` |
| Generated voice reads the reference text first | reference transcript does not match the clip | re-run `make prepare-voices`, see [pitfalls](pitfalls.md) |
| `Could not prepare voice: ASR ... unreachable` | ASR container down / wrong `ASR_URL` | `curl $ASR_URL/health`, fix networking |
| SSE arrives all at once | a proxy is buffering | disable response buffering, or call the instance directly |
| OOM after a long uptime | very long inputs with high `MAX_CHUNK_CHARS` | lower `MAX_CHUNK_CHARS`, watch `tts_pending_audio_seconds` |
