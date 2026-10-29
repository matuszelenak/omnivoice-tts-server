# API

Base URL: `http://<host>:9001`. Interactive OpenAPI docs at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/audio/speech` | Synthesise (OpenAI "create speech"), whole body or SSE |
| `WS` | `/v1/ws/synthesize` | Live synthesis from streamed text (LLM output) |
| `GET` | `/v1/voices` | List cloneable voices |
| `POST` | `/v1/voices` | Register a voice from a reference clip |
| `GET` | `/v1/voices/{id}/preview` | Reference clip of a voice (WAV) |
| `DELETE` | `/v1/voices/{id}` | Delete a voice |
| `GET` | `/v1/languages` | Languages the model knows (ISO 639-3 + display name) |
| `GET` | `/v1/models` | OpenAI model list (for LiteLLM routing) |
| `GET` | `/v1/models/{id}` | OpenAI model object |
| `GET` | `/health` | Readiness, queue, GPU, voice and config state |
| `GET` | `/metrics` | Prometheus exposition |
| `GET` | `/readyz` | `200 ok` once the model is loaded, `503` while loading |

## `POST /v1/audio/speech`

OpenAI-compatible body plus three OmniVoice extensions (`language`,
`stream_format`, `num_step`).

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `input` | string | — | Text to synthesise (≤ `MAX_INPUT_CHARS`, default 4000) |
| `model` | string | `omnivoice` | Accepted for compatibility; one model is served |
| `voice` | string | `null` | Voice id from `/v1/voices`, or `auto`/`null` for the model's own voice |
| `instructions` | string | `null` | Voice-design descriptor: `gender, age, pitch, style, accent` |
| `response_format` | string | `mp3` | `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | float | `1.0` | 0.25–4.0 |
| `language` | string | `en` | ISO 639-3 code or full name (`sk`, `Slovak`) |
| `stream_format` | string | `audio` | `audio` = one body; `sse` = server-sent audio deltas |
| `num_step` | int | `NUM_STEPS` | Diffusion steps override (1–64): lower is faster and rougher |
| `sanitize` | bool | `true` | Rewrite digits/abbreviations into spoken words with the configured LLM (no-op when `SANITIZE_LLM_BASE_URL` is unset) |

Rules:

- `voice` wins over `instructions`. With neither, the model picks a voice itself
  (`auto` mode).
- `voice` and `instructions` are mutually exclusive *in practice*: the request is
  rejected if the voice id is unknown.
- Input is split into sentences internally; the response is the concatenation in
  order, with a 40 ms gap between sentences.

### Whole body

`200` with `Content-Type` for the format, `Content-Disposition: attachment`, and
`Cache-Control: no-store`.

### SSE (`"stream_format": "sse"`)

`Content-Type: text/event-stream`, one event per sentence:

```
data: {"type":"speech.audio.delta","audio":"<base64>","format":"wav"}

data: {"type":"speech.audio.delta","audio":"<base64>","format":"wav"}

data: {"type":"speech.audio.done"}
```

Each delta is a complete, independently decodable audio chunk. For `mp3`,
`opus` and `pcm` the deltas also concatenate into one valid stream byte-for-byte.
For `wav` **each delta carries its own 44-byte header**, so naively concatenating
the bytes produces a file that most decoders read only up to the first data
chunk — decode the deltas individually (as the bundled UI does with the Web Audio
API), or re-wrap the frames if you need a single file
(`scripts/test_endpoint.py` shows both). Errors that happen after the stream started are
sent as `data: {"error": {"message": "...", "type": "server_error"}}`.

Admission is checked before the stream starts, so an overloaded node returns a
plain `503` instead of an empty stream.

## `WS /v1/ws/synthesize`

Query parameters: `language`, `voice_id` (or `voice`), `speed`, `instruct`,
`response_format` (default `wav`), `num_step`, `sanitize` (default `true`).

Client → server:

- text frames: text fragments, any size. The server detects sentence boundaries
  and starts synthesising as soon as the first fragment is long enough.
- `""` (empty text frame) or `{"type": "end"}`: end of input; the remaining
  partial sentence is flushed and then the socket is closed.
- `{"type": "flush"}`: flush the current fragment as a chunk immediately (useful
  after a heading or a list item).
- `{"type": "configure", ...}` as the first frame: override `language`,
  `voice_id`, `speed`, `instruct` (same keys as query parameters).

Server → client: **binary** frames only, one per synthesised chunk, in order,
encoded as `response_format`. Clients that cannot play chunks immediately should
buffer them.

Admission and backpressure:

- more than `MAX_STREAM_SESSIONS` concurrent sessions → close code `1013`;
- a client that feeds text faster than the GPU can render it, accumulating more
  than `STREAM_MAX_PENDING_S` seconds of unsynthesised text → close code `1013`
  (send slower, or use `POST /v1/audio/speech` for bulk text);
- `STREAM_MAX_SESSION_S` (default 1 h) closes long-lived sessions.

## Voices

### `POST /v1/voices` (multipart)

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `name` | string | yes | Voice id (letters, digits, `-`, `_`; spaces become `_`) |
| `ref_audio` | file | yes | Reference clip (WAV/FLAC/OGG/MP3/M4A/MP4/WebM) |
| `language` | string | no (`en`) | Language of the clip |
| `ref_text` | string | no | Exact transcript of the clip; omitted → transcribed by the ASR service |

The server normalises the clip (mono 24 kHz, edge silence removed, level
normalised, cut to `MAX_REF_SECONDS` **on a word boundary**) and guarantees the
stored transcript matches the stored audio — a mismatched transcript makes the
model *speak the reference text* before the requested text (see
[docs/pitfalls.md](pitfalls.md)).

Response `201`:

```json
{"id":"narrator","name":"narrator","filename":"voice.wav",
 "ref_text":"Hello, this is the narrator.","language":"en","duration":4.47,"created":true}
```

Errors: `400` invalid name, `413` upload too large, `422` clip unusable (too
short, no speech, transcript cannot be aligned).

### `GET /v1/voices`

```json
[{"id":"chloe_sample","name":"Chloe Sample","filename":"voice.wav",
  "ref_text":"I was wondering, are we friends? ...","language":"en","duration":4.47}]
```

### `GET /v1/voices/{id}/preview`

The prepared reference clip (`voice.wav`) — handy for a UI "listen to this voice"
button.

### `DELETE /v1/voices/{id}`

Removes the directory and invalidates the cached prompt.

## Languages

`GET /v1/languages` returns ~650 entries: `{"id": "sk", "name": "Slovak"}`. Both
forms are accepted everywhere a language is expected. An unknown language is a
`422` with the list of alternatives in the message.

## Errors

OpenAI-style envelope on `/v1/audio/*` and `/v1/models`, FastAPI's `detail` on
the rest:

```json
{"error": {"message": "queue is full (48 chunks waiting); retry shortly",
           "type": "server_error", "param": null, "code": 503}}
```

| Status | When | Client action |
| --- | --- | --- |
| `400` | malformed body / invalid voice name | fix the request |
| `404` | unknown voice or model | fix the request |
| `413` | input or upload too large | split the request |
| `422` | empty text, unsupported language, unusable reference clip, invalid `instruct` | fix the request |
| `503` | queue full / at capacity / model loading | retry after `Retry-After` seconds |
| `500` | model failure (e.g. CUDA OOM) | retry, then report |

`503` responses always carry `Retry-After` (default 2 s, `RETRY_AFTER_S`).

Every response carries `X-Request-ID` (echoed if the client sent one).

## LiteLLM

LiteLLM proxies TTS through its `openai/` provider; point it at this server and
the `voice`/`response_format`/`speed` fields are forwarded as-is:

```yaml
# litellm/config.yaml
model_list:
  - model_name: omnivoice
    litellm_params:
      model: openai/omnivoice
      api_base: http://omnivoice-tts-server:9001/v1
      api_key: unused

  # Slovak alias that pins the language via extra body fields
  - model_name: omnivoice-sk
    litellm_params:
      model: openai/omnivoice
      api_base: http://omnivoice-tts-server:9001/v1
      api_key: unused
      extra_body:
        language: sk
```

```bash
curl -X POST http://litellm:4000/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"omnivoice","voice":"chloe_sample","input":"Hello there.","response_format":"wav"}' \
  --output hello.wav
```

Notes:

- LiteLLM passes `stream_format`/`language` only via `extra_body`; the plain
  OpenAI schema has no language field.
- LiteLLM buffers SSE responses in some versions. For real progressive playback
  call the server directly (or through a proxy that does not buffer — the
  server sets `X-Accel-Buffering: no`).
- The WebSocket endpoint is not proxied by LiteLLM.
