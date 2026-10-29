# OmniVoice TTS Server

A production-grade, self-hosted text-to-speech service for the
[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) model. It speaks **OpenAI's
`/v1/audio/speech` protocol**, so any OpenAI SDK, LiteLLM, or OpenAI-compatible
client works unchanged, and it adds what interactive use actually needs:

- **Streaming in, streaming out** — SSE audio deltas, or a WebSocket that turns a
  token stream from an LLM into audio while the LLM is still writing.
- **Voice cloning** from a short reference clip, and **voice design** from a
  natural-language descriptor ("female, young adult, british accent").
- **Accelerated decoding** — the backbone is `torch.compile`d with inductor
  (1.67× measured, same WER; `max-autotune` and `reduce-overhead` both measured
  *slower*, see docs) and queued sentences are batched into one model call (1.2×
  lower latency under load); CUDA-graph replay is available as an extra 14 % on
  an uncompiled model.
- **Concurrency without latency cliffs** — sentence-level scheduling with
  round-robin fairness, admission control (`503` + `Retry-After`), and cancellation
  when a client hangs up.
- **English and Slovak** (and the ~600 other languages the model supports), each
  verified end-to-end by transcribing the served audio with the sibling Parakeet
  ASR service (0 % word error on the English benchmark set, low single digits on
  Slovak — see [docs/benchmarks.md](docs/benchmarks.md)).
- **Operational visibility** — Prometheus metrics, structured logs, readiness
  probes and a `/health` payload with queue, GPU and voice state.

```
.
├── server/                # FastAPI + OmniVoice (uv)
│   ├── app/               # config, text, audio, engine, scheduler, voices, api
│   ├── tests/             # unit + API tests, no GPU required
│   └── voices/            # reference clips (voice cloning)
├── frontend/              # Svelte 5 UI (Deno + Vite)
├── bench/                 # load generator, quality verification, micro-benchmarks
├── docs/                  # API, configuration, deployment, benchmarks, pitfalls
├── scripts/               # voice preparation, endpoint smoke test
├── Makefile               # build/run/status/bench/dev
├── docker-compose.yml     # development stack (hot reload)
├── docker-compose.prod.yml
└── prod.Dockerfile        # frontend baked into the API image
```

## Why this exists

LiteLLM + `audio.cpp` works for one user at a time. `audio.cpp` has no request
scheduler, no admission control, no prompt reuse, and no per-chunk streaming, so
under real traffic latency grows without bound and the GPU is idle between
requests. This server replaces that layer with a Python engine around the
reference PyTorch implementation:

| | audio.cpp behind LiteLLM | this server |
| --- | --- | --- |
| concurrent clients | one backend, serialised | sentence-level fair queue + batching, ~40 audio-seconds/s at 16 clients |
| overload behaviour | latency grows with the queue | measured `503` + `Retry-After` in milliseconds |
| time to first audio | waits for the whole text | **0.11–0.18 s** for the first sentence (SSE, measured) |
| voice prompt | re-encoded per call | precomputed, LRU-cached (saves ~60 ms per chunk) |
| speaker quality | relies on the caller's transcript | reference is trimmed, cut on a word boundary, transcript repaired via ASR |
| observability | none | Prometheus + structured logs + `/health` |

## Quick start

Requires Docker with the NVIDIA Container Toolkit, and one GPU dedicated to TTS
(the sibling ASR service should own the other one).

```bash
cp .env.example .env          # GPU, HF cache path, ASR URL
make build                    # production image (UI baked in)
make run                      # serve on GPU 0 -> http://localhost:9001
make status                   # health, queue, GPU
```

`make run` prints `/health` once the model is loaded (about 6 s with a warm HF
cache). Open <http://localhost:9001> for the UI, `/docs` for the API.

Without Docker:

```bash
cd server
uv sync
CUDA_VISIBLE_DEVICES=0 uv run uvicorn app.main:app --port 9001
```

## Usage

```bash
# voice cloning, MP3 (the OpenAI default)
curl -X POST http://localhost:9001/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"Hello there. This is a cloned voice.","voice":"chloe_sample","language":"en"}' \
  --output hello.mp3

# streaming: first audio delta in ~0.2 s
curl -N -X POST http://localhost:9001/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"Hello there. This is streamed.","voice":"kroner","language":"sk",
       "response_format":"wav","stream_format":"sse"}'

# voice design, no reference clip
curl -X POST http://localhost:9001/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"Hello there.","instructions":"female, young adult, british accent",
       "language":"en","response_format":"wav"}' --output design.wav

# register a voice from a clip (transcript optional — ASR fills it in)
curl -X POST http://localhost:9001/v1/voices \
  -F name=narrator -F language=en -F ref_audio=@my_voice.wav
```

Python (any OpenAI SDK, `base_url` pointed here):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9001/v1", api_key="unused")
audio = client.audio.speech.create(
    model="omnivoice", voice="chloe_sample", input="Hello there.",
    response_format="wav", extra_body={"language": "en"},
)
audio.write_to_file("hello.wav")
```

Full API reference: [docs/api.md](docs/api.md). Configuration: every knob is an
environment variable — [docs/configuration.md](docs/configuration.md).

## How it works

```
client ──HTTP/WS──▶ FastAPI ──▶ text splitter ──▶ scheduler ──▶ GPU worker ──▶ encoder ──▶ client
                                   (BlingFire,      (fair queue,    (OmniVoice,    (wav/pcm/
                                    clause cap)      admission)      fp16)          mp3/…)
```

* **One sentence per model call.** OmniVoice is a diffusion LM: it renders a
  masked target region in a fixed number of bidirectional passes, so cost is
  proportional to text length and steps and nothing can be incrementally decoded
  autoregressively (the vllm-omni serving path goes further with *blockwise*
  denoising — a fixed 1.3 s block at a time — at a throughput cost). Splitting at
  sentence boundaries is the sanctioned way to stream, and it bounds
  time-to-first-audio.
* **No request batching, on purpose.** A 3090 is already FLOP-bound at batch
  size 1: batching eight identical sentences saves 22 % per item, and nothing at
  all once lengths differ — while the batch's first item waits for the whole
  forward pass (measured, [docs/benchmarks.md](docs/benchmarks.md)). The
  scheduler therefore optimises *fairness and latency*, not batch efficiency.
* **First chunk priority.** Each request's first sentence is queued ahead of
  other requests' follow-up sentences, so time-to-first-audio depends on how
  many users are waiting, not on how much text they sent.
* **Voice prompts are cached.** Encoding a reference clip costs ~60 ms and an
  ASR call; the resulting `VoiceClonePrompt` is cached per voice (and per
  uploaded clip hash) and reused for every sentence.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/api.md](docs/api.md) | Endpoints, request fields, SSE format, WebSocket protocol, LiteLLM, errors |
| [docs/configuration.md](docs/configuration.md) | Every environment variable, with defaults and tuning advice |
| [docs/deployment.md](docs/deployment.md) | Images, GPU assignment, **H200 deployment notes**, capacity planning, metrics, upgrades, health checks |
| [docs/benchmarks.md](docs/benchmarks.md) | Latency, throughput, concurrency, steps/quality trade-offs, methodology |
| [docs/pitfalls.md](docs/pitfalls.md) | Reference/transcript mismatch, word-boundary cuts, batching, SSE through proxies |

## Development

```bash
make dev-up        # API :9001 (hot reload) + Vite :5173
make test          # 72 unit/API tests, no GPU needed
make lint          # ruff
make bench         # quick latency check against a running server
make bench-load    # full concurrency/overload/soak suite -> bench/results/
make plan          # measure this machine, print the admission limits to use
make probe         # is the GPU saturated at batch size 1? (batching decision)
```

The UI in `frontend/` is a Svelte 5 app that exercises every mode: voice cloning,
voice design, full audio, per-sentence streaming, and an "LLM simulation" that
streams tokens over the WebSocket to show real-time synthesis latency.

## Voices

Drop a reference clip into `server/voices/<id>/` and normalise it:

```bash
make prepare-voices      # or: scripts/prepare_voices.py --language myvoice=sk
```

Each voice is a directory with `voice.wav` (mono 24 kHz, 3–8 s) and a transcript
(`<lang>.txt`); the untouched original is kept as `source.wav` so preparation can
be repeated. See [docs/pitfalls.md](docs/pitfalls.md#a-reference-transcript-that-does-not-match-the-audio)
for why the transcript must match the clip *exactly*.

## License

MIT — see [LICENSE](LICENSE).

> This project was originally prototyped with heavy AI assistance; the
> architecture and code were then rewritten and benchmarked by hand. Treat the
> numbers in `docs/benchmarks.md` as measurements of *this* repository on *that*
> hardware, not as model-card claims.
