# OmniVoice TTS Server

A self-hosted text-to-speech service built on the [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) model. Supports multilingual synthesis, voice cloning from a reference audio clip, and a built-in web UI.

```
┌─────────────────────────┐      ┌─────────────────────────┐
│   Svelte frontend        │ ───► │   FastAPI server         │
│   (port 5173 / dev)      │      │   (port 9001)            │
│   (port 9001 / prod)     │      │   k2-fsa/OmniVoice       │
└─────────────────────────┘      └─────────────────────────┘
```

In **production**, the frontend is compiled and served as static files by the same server process on a single port. In **development**, the Vite dev server proxies API calls to the Python backend.

## Repository layout

```
.
├── server/          # FastAPI + OmniVoice Python backend
│   ├── main.py
│   ├── src/
│   │   ├── config.py
│   │   └── chunker.py
│   ├── voices/     # Built-in voice samples (.wav + .txt pairs)
│   └── Dockerfile
├── frontend/        # Svelte 5 web UI
│   └── src/
├── scripts/
│   └── test_endpoint.py   # CLI smoke-test for the API
├── docker-compose.yml      # Development stack
├── docker-compose.prod.yml # Production stack (single port)
├── prod.Dockerfile         # Multi-stage build (frontend baked in)
└── .env.example
```

## Prerequisites

- Docker + Docker Compose with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- A CUDA-capable GPU

The server downloads model weights from Hugging Face on first run (~several GB). Mount a persistent cache volume (see below) to avoid re-downloading.

## Development

Copy the example env file and adjust as needed:

```bash
cp .env.example .env
```

Start the full stack with live-reload:

```bash
docker compose up --watch
```

- Frontend: http://localhost:5173
- API: http://localhost:9001
- API docs: http://localhost:9001/docs

`--watch` enables file sync — edits to `server/main.py`, `server/src/`, and `frontend/src/` are reflected immediately without a full rebuild.

To persist the model cache between runs:

```bash
HF_HOME=$HOME/.cache/huggingface docker compose up --watch
```

## Production

The production build compiles the frontend and bakes it into the server image, serving everything from a single port:

```bash
docker compose -f docker-compose.prod.yml up -d
```

- UI + API: http://localhost:9001

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `NUM_WORKERS` | `1` | Number of inference workers |
| `DEVICES` | _(auto)_ | Comma-separated CUDA device list, e.g. `cuda:0,cuda:1` |
| `VOICE_SAMPLES_DIR` | `/app/voices` | Path to built-in voice samples inside the container |
| `STATIC_DIR` | _(unset)_ | Path to compiled frontend static files (set automatically in prod) |
| `HF_HOME` | `./server/models` | Host path for the Hugging Face model cache |
| `HF_TOKEN` | _(unset)_ | Hugging Face token for gated models |
| `LOGFIRE_TOKEN` | _(unset)_ | [Logfire](https://logfire.pydantic.dev) token; telemetry is disabled when absent |

## Web UI

The frontend provides:

- Language selection (ISO 639-3 codes or full names)
- Built-in voice picker (loaded from `server/voices/`)
- Voice cloning from an uploaded audio file (WAV, MP3, FLAC, OGG)
- Optional reference transcript to improve cloning quality
- Speech speed control (0.5× – 2.0×)
- In-browser playback and WAV download

## Adding built-in voices

Place a `.wav` file in `server/voices/`. Optionally add a `.txt` file with the same stem containing the transcript of the recording — this improves cloning quality.

```
server/voices/
  alice.wav
  alice.txt    # "Hi, this is Alice speaking."
```

The voice ID is the filename stem (`alice`). Voices appear in the UI and via `GET /v1/voices` immediately without a restart (the directory is read on each request).

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/synthesize` | Synthesize speech; returns a WAV file |
| `GET` | `/v1/languages` | List supported languages |
| `GET` | `/v1/voices` | List built-in voices |
| `GET` | `/v1/voices/{id}/preview` | Stream the reference WAV for a voice |
| `DELETE` | `/v1/voices/{id}` | Delete a built-in voice |
| `GET` | `/health` | `{"status": "ok", "model_loaded": true/false}` |

Full interactive docs at `/docs` when the server is running.

### Quick example

```bash
curl -X POST http://localhost:9001/v1/synthesize \
  -F "text=Hello, world!" \
  -F "language=en" \
  --output output.wav
```

## Testing

```bash
python scripts/test_endpoint.py \
  --ref-audio server/voices/alice.wav \
  --ref-text "Hi, this is Alice speaking." \
  --text "The quick brown fox jumps over the lazy dog." \
  --output out.wav
```
