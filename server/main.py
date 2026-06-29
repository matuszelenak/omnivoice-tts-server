import asyncio
import base64
import contextlib
import json
import logging
import os
import queue
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

import transformers

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="transformers.*")

import logfire
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from omnivoice import OmniVoice
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID, LANG_NAMES, lang_display_name
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from stream2sentence import generate_sentences

from openai import AsyncOpenAI
from src import audio
from src import voices as voice_store
from src.chunker import split_to_sentences
from src.config import settings
from src.inference import infer
from src.sanitize import sanitize_for_tts

logfire.configure(
    service_name="omnivoice-tts-server",
    send_to_logfire="if-token-present",
    scrubbing=False,
)

# Fewer diffusion steps on the first sentence of a stream trades a little
# quality for a faster first audio chunk.
NUM_STEPS_FIRST_SENTENCE = 16
NUM_STEPS = 32

_executor = ThreadPoolExecutor(max_workers=1)
_model: OmniVoice | None = None
_sanitize_client: AsyncOpenAI | None = None


async def _sanitize(text: str, enabled: bool = True) -> str:
    """Sanitize text for TTS; no-op when disabled or the sanitize LLM is not configured."""
    if not enabled or _sanitize_client is None:
        return text
    return await sanitize_for_tts(text, _sanitize_client, settings.sanitize_llm_model)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model, _sanitize_client
    _model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=settings.device_map,
        dtype=torch.float16,
    )
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, partial(infer, _model, "Hello.", "en"))
        logfire.info("model warm-up complete")
    except Exception as exc:
        logfire.warning("model warm-up failed: {exc}", exc=exc)

    if settings.sanitize_llm_base_url:
        _sanitize_client = AsyncOpenAI(
            base_url=settings.sanitize_llm_base_url,
            api_key=settings.sanitize_llm_api_key or "none",
        )
        logfire.info("text sanitization enabled via {url}", url=settings.sanitize_llm_base_url)
    else:
        logfire.info("text sanitization disabled (SANITIZE_LLM_BASE_URL not set)")

    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="OmniVoice TTS Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built once at import time; both ISO codes and full names are valid inputs.
_LANGUAGES = sorted(
    [{"id": code, "name": lang_display_name(name)} for name, code in LANG_NAME_TO_ID.items()],
    key=lambda x: x["name"],
)


def _language_supported(language: str) -> bool:
    return language in LANG_IDS or language.lower() in LANG_NAMES


def _validate_language(language: str) -> None:
    if not _language_supported(language):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported language '{language}'. "
                "Use an ISO 639-3 code (e.g. 'en', 'sk') or a full name "
                "(e.g. 'English', 'Slovak'). See GET /v1/languages for the full list."
            ),
        )


def _normalize(value: str | None) -> str | None:
    """Collapse empty / whitespace-only optional fields to None."""
    if value is None:
        return None
    return value.strip() or None


def _unlink_quietly(path: str | None) -> None:
    if path:
        with contextlib.suppress(OSError):
            os.unlink(path)


async def _save_upload(ref_audio: UploadFile) -> tuple[str, str]:
    """Write the upload to a temp file; return (tmp_path, original_ext)."""
    ext = os.path.splitext(ref_audio.filename or "")[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await ref_audio.read())
        return tmp.name, ext


# ── OpenAI-compatible speech synthesis ───────────────────────────────────────


class SpeechRequest(BaseModel):
    """Body of ``POST /v1/audio/speech`` — OpenAI's "Create speech" schema.

    ``language`` and ``stream_format`` extend the OpenAI fields: OmniVoice needs
    an explicit language, and ``stream_format`` selects between a single audio
    body (the default) and a Server-Sent-Events stream of audio deltas.
    """

    model: str = "omnivoice"
    input: str
    voice: str | None = None
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = None

    # Extensions (not part of the OpenAI schema):
    language: str = "en"
    sanitize: bool = True
    stream_format: Literal["audio", "sse"] = "audio"


def _resolve_speech_voice(req: SpeechRequest) -> tuple[str | None, str | None, str | None]:
    """Map the request onto OmniVoice inputs: (ref_audio_path, ref_text, instruct).

    ``voice`` names a stored voice (cloning); ``instructions`` drives the
    description-only "design" mode. OpenAI requires ``voice``, but OmniVoice can
    also run from ``instructions`` alone, so we accept either.
    """
    instruct = _normalize(req.instructions)
    voice = _normalize(req.voice)
    if not voice and not instruct:
        raise HTTPException(
            status_code=400,
            detail="Either 'voice' (a stored voice id) or 'instructions' must be provided",
        )
    ref_audio_path = ref_text = None
    if voice:
        ref_audio_path, ref_text = voice_store.resolve_voice(voice, settings.voice_samples_dir, None)
    return ref_audio_path, ref_text, instruct


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    """Generate audio from text — OpenAI "Create speech" compatible endpoint."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _validate_language(req.language)
    ref_audio_path, ref_text, instruct = _resolve_speech_voice(req)
    speed = None if req.speed == 1.0 else req.speed
    fmt = req.response_format
    sanitize_enabled = req.sanitize

    loop = asyncio.get_running_loop()
    synth = partial(
        infer, _model,
        language=req.language, speed=speed,
        ref_audio_path=ref_audio_path, ref_text=ref_text, instruct=instruct,
    )

    if req.stream_format == "sse":
        return StreamingResponse(
            _sse_stream(loop, synth, req.input, fmt, sanitize_enabled),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    text = await _sanitize(req.input, sanitize_enabled)
    samples = await loop.run_in_executor(_executor, partial(synth, text))
    data = await loop.run_in_executor(_executor, partial(audio.encode, samples, fmt))
    return Response(
        content=data,
        media_type=audio.content_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
    )


async def _sse_stream(loop, synth, text: str, fmt: str, sanitize_enabled: bool = True):
    """Yield OpenAI ``speech.audio.delta`` / ``speech.audio.done`` SSE events.

    One delta is emitted per sentence so playback can start before the whole
    input is rendered. Each delta is independently decodable; streamable codecs
    (mp3/opus/aac/pcm) also concatenate into a single file, while wav/flac
    deltas are meant to be decoded chunk by chunk.
    """
    for i, sentence in enumerate(split_to_sentences(text)):
        num_step = NUM_STEPS_FIRST_SENTENCE if i == 0 else NUM_STEPS
        clean = await _sanitize(sentence, sanitize_enabled)
        samples = await loop.run_in_executor(_executor, partial(synth, clean, num_step=num_step))
        data = await loop.run_in_executor(_executor, partial(audio.encode, samples, fmt))
        yield _sse({"type": "speech.audio.delta", "audio": base64.b64encode(data).decode("ascii")})
    yield _sse({"type": "speech.audio.done"})


# ── Voice management ──────────────────────────────────────────────────────────


@app.post("/v1/voices", status_code=201)
async def create_voice(
    name: Annotated[str, Form()],
    ref_text: Annotated[str, Form()],
    ref_audio: Annotated[UploadFile, File()],
    language: Annotated[str, Form()] = "en",
) -> dict:
    """Register a cloned voice from a reference clip + transcript.

    The resulting voice id can then be passed as ``voice`` to
    ``POST /v1/audio/speech``.
    """
    _validate_language(language)
    if not ref_text.strip():
        raise HTTPException(status_code=422, detail="ref_text must not be empty")

    tmp_path: str | None = None
    try:
        tmp_path, ext = await _save_upload(ref_audio)
        voice_store.save_voice_sample(
            name, tmp_path, ext, ref_text.strip(), language, settings.voice_samples_dir
        )
    finally:
        _unlink_quietly(tmp_path)

    voice_id = name.strip().replace(" ", "_")
    return {"id": voice_id, "language": language}


@app.get("/v1/voices")
async def voices() -> list[dict]:
    return voice_store.list_voices(settings.voice_samples_dir)


@app.get("/v1/voices/{voice_id}/preview")
async def voice_preview(voice_id: str) -> FileResponse:
    voice_store.validate_voice_id(voice_id)
    path = voice_store.find_voice_file(settings.voice_samples_dir, voice_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Voice sample not found")
    return FileResponse(path, media_type=voice_store.audio_media_type(path))


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict:
    voice_store.delete_voice(settings.voice_samples_dir, voice_id)
    return {"deleted": voice_id}


@app.get("/v1/languages")
async def languages() -> list[dict]:
    return _LANGUAGES


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


# ── WebSocket streaming (no OpenAI Speech equivalent; kept as an extension) ────


@app.websocket("/v1/ws/synthesize")
async def ws_synthesize(
    ws: WebSocket,
    language: str = "en",
    voice_id: str | None = None,
    speed: float | None = None,
    instruct: str | None = None,
    sanitize: bool = True,
    response_format: str = "wav",
) -> None:
    await ws.accept()

    if _model is None:
        await ws.close(code=1011, reason="Model not loaded")
        return

    if not _language_supported(language):
        await ws.close(code=1008, reason=f"Unsupported language '{language}'")
        return

    if response_format not in audio.RESPONSE_FORMATS:
        await ws.close(code=1008, reason=f"Unsupported response_format '{response_format}'")
        return

    instruct = _normalize(instruct)
    if instruct:
        effective_ref, effective_ref_text = None, None
    elif voice_id:
        try:
            effective_ref, effective_ref_text = voice_store.resolve_voice(
                voice_id, settings.voice_samples_dir, None
            )
        except HTTPException as e:
            await ws.close(code=1008, reason=e.detail)
            return
    else:
        await ws.close(
            code=1008,
            reason="voice cloning over WebSocket requires voice_id (file upload is not supported)",
        )
        return

    loop = asyncio.get_running_loop()
    text_q: queue.Queue[str | None] = queue.Queue()
    audio_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    synth = partial(
        infer, _model,
        language=language, speed=speed,
        ref_audio_path=effective_ref, ref_text=effective_ref_text, instruct=instruct,
    )

    def _text_gen():
        while True:
            chunk = text_q.get()
            if chunk is None:
                return
            yield chunk

    def _process():
        try:
            for sentence in generate_sentences(_text_gen()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                clean = asyncio.run_coroutine_threadsafe(_sanitize(sentence, sanitize), loop).result()
                samples = _executor.submit(partial(synth, clean)).result()
                data = audio.encode(samples, response_format)
                asyncio.run_coroutine_threadsafe(audio_q.put(data), loop).result()
        except Exception as exc:
            logfire.warning("ws synthesis aborted: {exc}", exc=exc)
        finally:
            asyncio.run_coroutine_threadsafe(audio_q.put(None), loop).result()

    threading.Thread(target=_process, daemon=True).start()

    async def _send_audio():
        while True:
            data = await audio_q.get()
            if data is None:
                break
            await ws.send_bytes(data)

    send_task = asyncio.create_task(_send_audio())

    try:
        async for message in ws.iter_text():
            if message == "":  # empty string = end-of-stream sentinel
                text_q.put(None)
                break
            text_q.put(message)
        else:
            text_q.put(None)
    except WebSocketDisconnect:
        text_q.put(None)

    await send_task
    if ws.client_state != WebSocketState.DISCONNECTED:
        await ws.close()


if settings.static_dir:
    _static_path = Path(settings.static_dir)
    if _static_path.is_dir():
        app.mount(
            "/", StaticFiles(directory=_static_path, html=True), name="frontend"
        )
        logfire.info("serving static frontend from {path}", path=str(_static_path))
    else:
        logfire.warning(
            "STATIC_DIR={path} is set but not a directory; static serving disabled",
            path=settings.static_dir,
        )
