"""FastAPI application: OpenAI-compatible speech synthesis for OmniVoice.

Endpoints
---------
``POST /v1/audio/speech``      OpenAI "create speech" (JSON in, audio or SSE out)
``WS   /v1/ws/synthesize``     streaming: text fragments in, audio frames out
``GET  /v1/models``            OpenAI model listing (LiteLLM routing)
``GET  /v1/languages``         supported languages
``GET  /v1/voices``            registered voices (clone references)
``POST /v1/voices``            register a voice from a clip + transcript
``GET  /v1/voices/{id}/preview``  play a voice's reference clip
``DELETE /v1/voices/{id}``     remove a voice
``GET  /health``               readiness + queue/model/GPU state
``GET  /metrics``              Prometheus exposition
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

# Silence the ML stack *before* it is imported: transformers and huggingface_hub
# log a deprecation notice per model class at import time.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _name in ("transformers", "huggingface_hub", "omnivoice"):
    logging.getLogger(_name).setLevel(logging.ERROR)
for _name in ("httpcore", "httpx", "filelock"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import logfire
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import audio, metrics, pipeline
from .asr import ASRClient
from .config import settings
from .engine import EngineError, NotReadyError, engine
from .pipeline import RequestParams, clean, concat, encode, iter_chunks, sse
from .sanitize import sanitizer
from .scheduler import (
    NotReady,
    QueueFullError,
    QueueTimeoutError,
    Scheduler,
    SchedulerError,
)
from .text import SentenceStreamer, normalize_space, split_text
from .voices import VoiceError, VoiceRegistry, VoiceStore, audio_media_type

try:  # Language catalogue ships with the model package.
    from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID, lang_display_name
except Exception:  # pragma: no cover - only if the model package is missing
    LANG_IDS, LANG_NAME_TO_ID = set(), {}

    def lang_display_name(name: str) -> str:  # type: ignore[misc]
        return name.title()


# Re-apply after the imports above: importing transformers resets its own level.
for noisy in ("transformers", "huggingface_hub", "omnivoice"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

if settings.logfire_token:
    os.environ.setdefault("LOGFIRE_TOKEN", settings.logfire_token)
logfire.configure(service_name="omnivoice-tts-server", send_to_logfire="if-token-present")
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("omnivoice")

scheduler = Scheduler(engine)
store = VoiceStore(settings.voice_samples_dir)
asr = ASRClient(settings.asr_url, settings.asr_timeout_s)
voices = VoiceRegistry(store, engine, asr)

# Live WebSocket sessions, for admission control.
_active_streams = 0

_LANGUAGES = sorted(
    [{"id": code, "name": lang_display_name(name)} for name, code in LANG_NAME_TO_ID.items()],
    key=lambda item: item["name"],
)
_ALIASES = {code: code for code in LANG_IDS}


# ── validation helpers ───────────────────────────────────────────────────────


def normalize_language(value: str | None, default: str = "en") -> str:
    """Map an ISO 639-3 code or a full language name onto a model language id."""
    if value is None or not str(value).strip():
        return default
    raw = str(value).strip()
    if raw in _ALIASES:
        return raw
    key = raw.lower()
    if key in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[key]
    raise HTTPException(
        status_code=422,
        detail=(
            f"unsupported language '{value}'; use an ISO 639-3 code (e.g. 'en', 'sk') "
            "or a full name (e.g. 'English', 'Slovak'). See GET /v1/languages."
        ),
    )


def _clamp_steps(value: int | None) -> int:
    if value is None:
        return settings.num_steps
    return max(1, min(int(value), settings.max_num_steps))


def _check_text(text: str) -> str:
    text = normalize_space(text)
    if not text:
        raise HTTPException(status_code=422, detail="input text is empty")
    if len(text) > settings.max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"input is {len(text)} characters, the limit is "
                f"{settings.max_input_chars}; split long texts into requests"
            ),
        )
    return text


def _scheduler_http_error(exc: SchedulerError) -> HTTPException:
    if isinstance(exc, (NotReady, NotReadyError)):
        return HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "5"})
    if isinstance(exc, QueueTimeoutError):
        return HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": str(settings.retry_after_s)})
    return HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": str(settings.retry_after_s)})


# ── lifespan ────────────────────────────────────────────────────────────────


async def _gpu_metrics_loop() -> None:
    """Publish CUDA memory every 15 s; torch only reports it on demand."""
    while True:
        try:
            stats = engine.stats()
            for key, value in (
                ("used", stats.get("gpu_memory_allocated_mb")),
                ("reserved", stats.get("gpu_memory_reserved_mb")),
            ):
                if value is not None:
                    metrics.gpu_memory_bytes.labels(kind=f"cuda_{key}").set(value * 2**20)
        except Exception:  # pragma: no cover - metrics must never break serving
            pass
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(engine.executor, engine.load)
        if settings.warmup:
            await loop.run_in_executor(engine.executor, engine.warmup)
        await scheduler.start()
        if settings.prewarm_voices:
            await voices.prewarm()
        asr_up = await asyncio.to_thread(asr.check) if asr.enabled else False
        if asr.enabled and not asr_up:
            log.warning(
                "ASR service at %s is unreachable; voice registration will require "
                "an exact ref_text", settings.asr_url,
            )
        log.info(
            "ready: device=%s dtype=%s voices=%d asr=%s",
            settings.device, engine.stats().get("dtype"), len(store.list()),
            settings.asr_url if asr_up else "disabled",
        )
    except Exception:
        log.exception("startup failed")
        raise
    metrics_task = asyncio.create_task(_gpu_metrics_loop(), name="tts-gpu-metrics")
    try:
        yield
    finally:
        log.info("shutting down")
        metrics_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await metrics_task
        await scheduler.stop()
        engine.shutdown()
        asr.close()


app = FastAPI(
    title="OmniVoice TTS Server",
    version="0.1.0",
    description="OpenAI-compatible text-to-speech on OmniVoice (voice cloning, voice design, streaming)",
    lifespan=lifespan,
)

logfire.instrument_fastapi(app, capture_headers=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Retry-After"],
)


# ── middleware & errors ─────────────────────────────────────────────────────


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


@app.middleware("http")
async def _observability(request: Request, call_next):
    if request.url.path == "/metrics" or not settings.metrics_enabled:
        return await call_next(request)
    request_id = request.headers.get("x-request-id") or os.urandom(8).hex()
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metrics.http_requests_total.labels(
            route=_route_label(request), method=request.method, status="500"
        ).inc()
        raise
    route = _route_label(request)
    metrics.http_requests_total.labels(
        route=route, method=request.method, status=str(response.status_code)
    ).inc()
    metrics.http_request_duration_seconds.labels(route=route).observe(
        time.perf_counter() - started
    )
    response.headers["X-Request-ID"] = request_id
    return response


_OPENAI_PREFIXES = ("/v1/audio/", "/v1/models")


def _openai_error(status: int, message: str, *, param: str | None = None,
                  headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "param": param,
                "code": status,
            }
        },
        status_code=status,
        headers=headers,
    )


def _is_openai_path(request: Request) -> bool:
    return request.url.path.startswith(_OPENAI_PREFIXES)


@app.exception_handler(StarletteHTTPException)
async def _http_exception(request: Request, exc: StarletteHTTPException) -> Response:
    if _is_openai_path(request):
        return _openai_error(exc.status_code, str(exc.detail), headers=exc.headers)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _validation_exception(request: Request, exc: RequestValidationError) -> Response:
    if _is_openai_path(request):
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = ".".join(str(part) for part in first.get("loc", ())[1:])
        return _openai_error(400, first.get("msg", "invalid request"), param=loc or None)
    return await request_validation_exception_handler(request, exc)


if settings.metrics_enabled:

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(content=generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ── request models ──────────────────────────────────────────────────────────


class SpeechRequest(BaseModel):
    """OpenAI "create speech" body, plus OmniVoice extensions.

    ``language``        ISO 639-3 code or language name (OmniVoice needs it).
    ``stream_format``   ``audio`` (single body) or ``sse`` (audio deltas).
    ``num_step``        diffusion steps override; lower = faster, less precise.
    """

    model: str = Field(default=settings.served_model_name)
    input: str
    voice: str | None = None
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = None
    language: str = "en"
    stream_format: Literal["audio", "sse"] = "audio"
    num_step: int | None = Field(default=None, ge=1, le=64)
    # Rewrite digits/abbreviations into spoken words with the sanitisation LLM
    # (no-op when SANITIZE_LLM_BASE_URL is unset).
    sanitize: bool = True


async def _resolve_params(
    *,
    language: str,
    voice: str | None,
    instructions: str | None,
    speed: float,
    num_step: int | None,
    sanitize: bool = True,
) -> RequestParams:
    """Map an OpenAI-style request onto OmniVoice inputs."""
    lang = normalize_language(language)
    instruct = (instructions or "").strip() or None
    voice_id = (voice or "").strip()
    if voice_id.lower() in {"", "auto", "none", "default"}:
        voice_id = ""

    prompt_key = None
    prompt = None
    if voice_id:
        prompt_key, prompt = await voices.resolve_voice(voice_id)
    elif instruct is None:
        log.debug("no voice and no instructions: falling back to the model's auto voice")
    steps = _clamp_steps(num_step)
    return RequestParams(
        language=lang,
        prompt_key=prompt_key,
        prompt=prompt,
        instruct=instruct,
        speed=None if abs(speed - 1.0) < 1e-3 else speed,
        num_step=steps,
        first_step=min(steps, settings.resolved_first_chunk_steps),
        guidance_scale=settings.guidance_scale,
        sanitize=sanitize,
    )


# ── OpenAI speech ───────────────────────────────────────────────────────────


def _audio_response(data: bytes, fmt: str) -> Response:
    return Response(
        content=data,
        media_type=audio.content_type(fmt),
        headers={
            "Content-Disposition": f'attachment; filename="speech.{fmt}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, request: Request):
    """Synthesise speech. Returns the whole clip, or SSE audio deltas."""
    started = time.perf_counter()
    text = _check_text(req.input)
    try:
        params = await _resolve_params(
            language=req.language, voice=req.voice, instructions=req.instructions,
            speed=req.speed, num_step=req.num_step, sanitize=req.sanitize,
        )
    except VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    metrics.input_chars.observe(len(text))
    texts = split_text(text, settings.max_chunk_chars)
    kind = "stream" if req.stream_format == "sse" else "full"
    if not texts:
        raise HTTPException(status_code=422, detail="input text is empty")

    if req.stream_format == "sse":
        return await _sse_response(
            session_kind="http-sse", texts=texts, params=params, fmt=req.response_format,
            started=started,
        )

    session = scheduler.new_session("http")
    try:
        chunks = [chunk async for chunk in iter_chunks(scheduler, session, texts, params, streaming=False)]
        samples = concat(chunks)
        data = await encode(samples, req.response_format)
    except SchedulerError as exc:
        metrics.requests_total.labels(kind=kind, outcome="rejected").inc()
        raise _scheduler_http_error(exc) from exc
    except ValueError as exc:
        metrics.requests_total.labels(kind=kind, outcome="rejected").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EngineError as exc:
        metrics.requests_total.labels(kind=kind, outcome="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        scheduler.end_session(session)

    metrics.requests_total.labels(kind=kind, outcome="ok").inc()
    metrics.first_chunk_seconds.labels(kind=kind).observe(time.perf_counter() - started)
    log.info(
        "synth full: %d chars, %d chunks, %.2fs audio, %.2fs wall",
        len(text), len(texts), len(samples) / audio.SAMPLE_RATE,
        time.perf_counter() - started,
    )
    return _audio_response(data, req.response_format)


async def _sse_response(
    *,
    session_kind: str,
    texts: list[str],
    params: RequestParams,
    fmt: str,
    started: float,
) -> StreamingResponse:
    """Server-sent events: one base64 audio delta per chunk, then a done event."""
    session = scheduler.new_session(session_kind)
    # Fail fast (before headers are sent) when the node is saturated.
    try:
        scheduler.check_admission(estimate_cost(texts, params))
    except SchedulerError as exc:
        scheduler.end_session(session)
        raise _scheduler_http_error(exc) from exc

    async def events() -> AsyncGenerator[bytes, None]:
        first = True
        outcome = "ok"
        chunks_sent = 0
        audio_bytes = 0
        try:
            async for samples in iter_chunks(scheduler, session, texts, params, streaming=True):
                data = await encode(samples, fmt)
                if first:
                    metrics.first_chunk_seconds.labels(kind=session_kind).observe(
                        time.perf_counter() - started
                    )
                    first = False
                chunks_sent += 1
                audio_bytes += len(data)
                yield sse({
                    "type": "speech.audio.delta",
                    "audio": base64.b64encode(data).decode("ascii"),
                    "format": fmt,
                })
        except (SchedulerError, EngineError, ValueError) as exc:
            outcome = "error"
            log.warning("sse stream failed: %s", exc)
            yield sse({"error": {"message": str(exc), "type": "server_error"}})
            return
        except asyncio.CancelledError:
            outcome = "client_disconnect"
            raise
        finally:
            metrics.requests_total.labels(kind=session_kind, outcome=outcome).inc()
            scheduler.end_session(session)
            log.info(
                "synth %s: %d chunks, %d bytes, %.2fs wall, outcome=%s",
                session_kind, chunks_sent, audio_bytes, time.perf_counter() - started, outcome,
            )
        yield sse({"type": "speech.audio.done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def estimate_cost(texts: list[str], params: RequestParams) -> float:
    return sum(estimate_seconds(t, params.speed or 1.0) for t in texts) * (
        params.num_step / 16
    )


def estimate_seconds(text: str, speed: float = 1.0) -> float:
    from .text import estimate_audio_seconds

    return estimate_audio_seconds(text, speed)


# ── WebSocket streaming ─────────────────────────────────────────────────────


@app.websocket("/v1/ws/synthesize")
async def ws_synthesize(
    ws: WebSocket,
    language: str = "en",
    voice_id: str | None = None,
    voice: str | None = None,
    speed: float | None = None,
    instruct: str | None = None,
    response_format: str = "wav",
    num_step: int | None = None,
    sanitize: bool = True,
) -> None:
    """Live synthesis: send text fragments, receive one WAV/WEBM frame per chunk.

    Options come from the query string. ``""`` (or ``{"type": "end"}``) ends the
    stream; the server then flushes the remaining partial sentence and closes.
    ``{"type": "flush"}`` forces the current fragment to be synthesised now.
    """
    global _active_streams

    await ws.accept()
    if not engine.ready:
        await ws.close(code=1013, reason="model not ready")
        return
    if _active_streams >= settings.max_stream_sessions:
        metrics.rejected_total.labels(reason="stream_limit").inc()
        metrics.stream_sessions_total.labels(outcome="rejected").inc()
        await ws.close(code=1013, reason="server at capacity; retry shortly")
        return
    if response_format not in audio.RESPONSE_FORMATS:
        await ws.close(code=1008, reason=f"unsupported response_format '{response_format}'")
        return

    try:
        language = normalize_language(language)
    except HTTPException as exc:
        await ws.close(code=1008, reason=str(exc.detail))
        return

    try:
        params = await _resolve_params(
            language=language, voice=voice_id or voice, instructions=instruct,
            speed=speed or 1.0, num_step=num_step, sanitize=sanitize,
        )
    except (HTTPException, VoiceError) as exc:
        detail = getattr(exc, "detail", None) or str(exc)
        await ws.close(code=1008, reason=str(detail))
        return

    _active_streams += 1
    metrics.stream_sessions_active.set(_active_streams)
    started = time.monotonic()
    outcome = "error"
    session = scheduler.new_session("ws")
    in_q: asyncio.Queue[str | None] = asyncio.Queue()
    out_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    streamer = SentenceStreamer(
        max_chunk_chars=settings.max_chunk_chars,
        min_chunk_chars=settings.min_chunk_chars,
    )
    first_sent = False
    queued_text_s = 0.0

    async def _reader() -> None:
        nonlocal outcome, queued_text_s
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    outcome = "client_disconnect"
                    break
                payload = message.get("text")
                if payload is None:
                    continue
                if payload == "":
                    outcome = "ended_by_client"
                    break
                try:
                    ctrl = json.loads(payload)
                except ValueError:
                    await in_q.put(payload)
                    continue
                kind = ctrl.get("type")
                if kind == "end":
                    outcome = "ended_by_client"
                    break
                if kind == "flush":
                    for sentence in streamer.flush():
                        await in_q.put(sentence)
                        queued_text_s += estimate_seconds(sentence)
                    continue
                if kind in {"configure", "ping"}:
                    continue  # accepted for forward compatibility, no effect
                await in_q.put(payload)
                queued_text_s += estimate_seconds(payload)
                if queued_text_s > settings.stream_max_pending_s:
                    # The client is feeding text faster than the GPU renders it;
                    # closing is friendlier than growing latency without bound.
                    outcome = "backpressure"
                    metrics.rejected_total.labels(reason="stream_backpressure").inc()
                    await ws.close(
                        code=1013,
                        reason=(
                            "client queued more than "
                            f"{settings.stream_max_pending_s:.0f}s of unsynthesised "
                            "text; send slower input or use POST /v1/audio/speech"
                        ),
                    )
                    break
                if time.monotonic() - started > settings.stream_max_session_s:
                    outcome = "max_duration"
                    break
        except WebSocketDisconnect:
            outcome = "client_disconnect"
        except Exception:  # pragma: no cover - defensive
            outcome = "error"
        finally:
            await in_q.put(None)

    async def _worker() -> None:
        nonlocal first_sent, outcome, queued_text_s
        try:
            while True:
                item = await in_q.get()
                if item is None:
                    for sentence in streamer.flush():
                        await _emit(sentence)
                    break
                queued_text_s = max(0.0, queued_text_s - estimate_seconds(item))
                for sentence in streamer.feed(item):
                    await _emit(sentence)
        except asyncio.CancelledError:
            raise
        except SchedulerError as exc:
            outcome = "backpressure" if isinstance(exc, (QueueFullError, QueueTimeoutError)) else "error"
            log.warning("ws synthesis rejected: %s", exc)
        except (EngineError, ValueError) as exc:
            outcome = "error"
            log.warning("ws synthesis failed: %s", exc)
        finally:
            await out_q.put(None)

    async def _emit(sentence: str) -> None:
        nonlocal first_sent
        future = scheduler.submit(
            session, await clean(sentence, params),
            params.synth_params(first=not first_sent, streaming=True),
            first=not first_sent, speed=params.speed or 1.0,
        )
        samples = await future
        data = await encode(samples, response_format)
        if not first_sent:
            first_sent = True
            metrics.first_chunk_seconds.labels(kind="ws").observe(time.monotonic() - started)
        await out_q.put(data)

    async def _sender() -> None:
        while True:
            data = await out_q.get()
            if data is None:
                return
            await ws.send_bytes(data)

    reader_task = asyncio.create_task(_reader())
    worker_task = asyncio.create_task(_worker())
    try:
        await _sender()
    except WebSocketDisconnect:
        outcome = "client_disconnect"
    except Exception:
        outcome = "error"
        raise
    finally:
        worker_task.cancel()
        reader_task.cancel()
        for task in (worker_task, reader_task):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        scheduler.end_session(session)
        _active_streams = max(0, _active_streams - 1)
        metrics.stream_sessions_active.set(_active_streams)
        metrics.stream_sessions_total.labels(outcome=outcome).inc()
        metrics.stream_session_seconds.observe(time.monotonic() - started)
        with contextlib.suppress(Exception):
            await ws.close()


# ── voices ──────────────────────────────────────────────────────────────────


@app.get("/v1/voices")
async def list_voices() -> list[dict]:
    return [v.public() for v in store.list()]


async def _read_upload(upload: UploadFile) -> bytes:
    raw = await upload.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds {settings.max_upload_mb} MB",
        )
    if not raw:
        raise HTTPException(status_code=422, detail="empty upload")
    return raw


@app.post("/v1/voices", status_code=201)
async def create_voice(
    name: Annotated[str, Form()],
    ref_audio: Annotated[UploadFile, File()],
    language: Annotated[str, Form()] = "en",
    ref_text: Annotated[str | None, Form()] = None,
) -> dict:
    """Register a cloneable voice.

    The clip is normalised (mono 24 kHz, edge silence trimmed, cut to
    ``MAX_REF_SECONDS`` on a word boundary) and, when no transcript is supplied,
    transcribed automatically.
    """
    lang = normalize_language(language)
    raw = await _read_upload(ref_audio)
    try:
        voice = await voices.add_voice(
            name=name, audio_bytes=raw, filename=ref_audio.filename or "ref.wav",
            ref_text=ref_text, language=lang,
        )
    except VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return {**voice.public(), "created": True}


@app.get("/v1/voices/{voice_id}/preview")
async def voice_preview(voice_id: str) -> FileResponse:
    try:
        voice = store.get(voice_id)
    except VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return FileResponse(voice.audio_path, media_type=audio_media_type(voice.audio_path))


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict:
    try:
        await asyncio.to_thread(voices.delete_voice, voice_id)
    except VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return {"deleted": voice_id}


@app.get("/v1/languages")
async def languages() -> list[dict]:
    return _LANGUAGES


# ── OpenAI models / health ──────────────────────────────────────────────────

_MODEL_CREATED = int(time.time())


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.served_model_name,
                "object": "model",
                "created": _MODEL_CREATED,
                "owned_by": "local",
            }
        ],
    }


@app.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str) -> dict:
    if model_id not in {settings.served_model_name, settings.model_name}:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return {
        "id": settings.served_model_name,
        "object": "model",
        "created": _MODEL_CREATED,
        "owned_by": "local",
    }


@app.get("/health")
async def health() -> dict:
    ready = engine.ready
    return {
        "status": "ok" if ready else "loading",
        "model_loaded": ready,
        "ready": ready,
        "model": settings.model_name,
        "served_model": settings.served_model_name,
        "engine": engine.stats(),
        "scheduler": scheduler.stats(),
        "voices": {"count": len(store.list())},
        "asr": {"url": settings.asr_url, "available": asr.available},
        "sanitize": sanitizer.stats(),
        "stream_sessions": {"active": _active_streams, "limit": settings.max_stream_sessions},
        "config": {
            "num_steps": settings.num_steps,
            "first_chunk_steps": settings.resolved_first_chunk_steps,
            "encode_workers": pipeline._ENCODE_WORKERS,
            "guidance_scale": settings.guidance_scale,
            "max_input_chars": settings.max_input_chars,
            "max_chunk_chars": settings.max_chunk_chars,
        },
    }


@app.get("/readyz", include_in_schema=False)
async def readyz() -> Response:
    if engine.ready:
        return Response(content="ok", media_type="text/plain")
    return Response(content="loading", status_code=503, media_type="text/plain")


# ── static frontend ─────────────────────────────────────────────────────────

_IMMUTABLE_SUFFIXES = (".js", ".css", ".woff2", ".svg", ".png", ".ico")


def _cache_control(path: str) -> str:
    if path.startswith("assets/") and path.endswith(_IMMUTABLE_SUFFIXES):
        return "public, max-age=31536000, immutable"
    if path.endswith(".html") or path == "":
        return "no-cache"
    return "public, max-age=3600"


class _FrontendFiles(StaticFiles):
    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        origin = scope.get("path", "").lstrip("/") or "index.html"
        response.headers.setdefault("cache-control", _cache_control(origin))
        return response


if settings.static_dir:
    _static = Path(settings.static_dir)
    if _static.is_dir():
        app.mount("/", _FrontendFiles(directory=_static, html=True), name="frontend")
        log.info("serving frontend from %s", _static)
    else:
        log.warning("STATIC_DIR=%s is not a directory; static serving disabled", settings.static_dir)
