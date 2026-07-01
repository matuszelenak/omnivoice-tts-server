import asyncio
import io
import json
import queue
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated

import logfire
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from stream2sentence import generate_sentences
from supertonic import AVAILABLE_LANGUAGES, SUPPORTED_LANGUAGES, TTS

from src import voices as voice_store
from src.chunker import split_to_sentences
from src.config import settings
from src.inference import infer
from src.sanitize import sanitize_for_tts

logfire.configure(
    service_name="supertonic-tts-server",
    send_to_logfire="if-token-present",
    scrubbing=False,
)

TOTAL_STEPS = 16

_executor = ThreadPoolExecutor(max_workers=1)
_model: TTS | None = None
_custom_styles_dir: Path = Path()
_sanitize_client: AsyncOpenAI | None = None


def _resolve_custom_styles_dir() -> Path:
    if settings.custom_styles_dir:
        return Path(settings.custom_styles_dir).expanduser()
    from supertonic.loader import get_cache_dir

    return get_cache_dir("supertonic-3") / "custom_styles"


async def _sanitize(text: str) -> str:
    """Sanitize text for TTS; no-op when the sanitize LLM is not configured."""
    if _sanitize_client is None:
        return text
    return await sanitize_for_tts(text, _sanitize_client, settings.sanitize_llm_model)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model, _custom_styles_dir, _sanitize_client
    model_dir = settings.supertonic_model_dir or None
    _model = TTS(model_dir=model_dir, auto_download=True)
    _custom_styles_dir = _resolve_custom_styles_dir()
    _custom_styles_dir.mkdir(parents=True, exist_ok=True)

    if settings.sanitize_llm_base_url:
        _sanitize_client = AsyncOpenAI(
            base_url=settings.sanitize_llm_base_url,
            api_key=settings.sanitize_llm_api_key or "none",
        )
        logfire.info("text sanitization enabled via {url}", url=settings.sanitize_llm_base_url)
    else:
        logfire.info("text sanitization disabled (SANITIZE_LLM_BASE_URL not set)")

    loop = asyncio.get_running_loop()
    style = _model.get_voice_style(_model.voice_style_names[0])
    try:
        await loop.run_in_executor(_executor, partial(infer, _model, "Hello.", "en", style))
        logfire.info("model warm-up complete")
    except Exception as exc:
        logfire.warning("model warm-up failed: {exc}", exc=exc)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Supertonic TTS Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_LANGUAGES = sorted(
    [{"id": code, "name": code} for code in SUPPORTED_LANGUAGES],
    key=lambda x: x["id"],
)


def _language_supported(language: str) -> bool:
    return language in AVAILABLE_LANGUAGES


def _validate_language(language: str) -> None:
    if not _language_supported(language):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported language '{language}'. "
                "Use an ISO 639-1 code (e.g. 'en', 'sk'). "
                "See GET /v1/languages for the full list."
            ),
        )


@app.post("/v1/synthesize")
async def synthesize(
    text: Annotated[str, Form()],
    voice_id: Annotated[str, Form()],
    language: Annotated[str, Form()] = "en",
    speed: Annotated[float | None, Form()] = None,
    total_steps: Annotated[int, Form()] = TOTAL_STEPS,
    stream: Annotated[bool, Form()] = False,
) -> StreamingResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _validate_language(language)
    voice_style = voice_store.get_voice_style(_model, voice_id, _custom_styles_dir)

    loop = asyncio.get_running_loop()
    synth = partial(infer, _model, language=language, speed=speed, voice_style=voice_style, total_steps=total_steps)

    if not stream:
        clean = await _sanitize(text)
        data = await loop.run_in_executor(_executor, partial(synth, clean))
        return StreamingResponse(
            io.BytesIO(data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )

    sentences = split_to_sentences(text)

    async def generate():
        for sentence in sentences:
            clean = await _sanitize(sentence)
            data = await loop.run_in_executor(_executor, partial(synth, clean))
            yield struct.pack(">I", len(data)) + data

    return StreamingResponse(generate(), media_type="application/octet-stream")


@app.websocket("/v1/ws/synthesize")
async def ws_synthesize(
    ws: WebSocket,
    language: str = "en",
    voice_id: str | None = None,
    speed: float | None = None,
    total_steps: int = TOTAL_STEPS,
) -> None:
    await ws.accept()

    if _model is None:
        await ws.close(code=1011, reason="Model not loaded")
        return

    if not _language_supported(language):
        await ws.close(code=1008, reason=f"Unsupported language '{language}'")
        return

    if not voice_id:
        await ws.close(code=1008, reason="voice_id is required")
        return

    try:
        voice_style = voice_store.get_voice_style(_model, voice_id, _custom_styles_dir)
    except HTTPException as e:
        await ws.close(code=1008, reason=e.detail)
        return

    loop = asyncio.get_running_loop()
    text_q: queue.Queue[str | None] = queue.Queue()
    audio_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    synth = partial(infer, _model, language=language, speed=speed, voice_style=voice_style, total_steps=total_steps)

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
                clean = asyncio.run_coroutine_threadsafe(_sanitize(sentence), loop).result()
                audio = _executor.submit(partial(synth, clean)).result()
                asyncio.run_coroutine_threadsafe(audio_q.put(audio), loop).result()
        except Exception as exc:
            logfire.warning("ws synthesis aborted: {exc}", exc=exc)
        finally:
            asyncio.run_coroutine_threadsafe(audio_q.put(None), loop).result()

    threading.Thread(target=_process, daemon=True).start()

    async def _send_audio():
        while True:
            audio = await audio_q.get()
            if audio is None:
                break
            await ws.send_bytes(audio)

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


@app.get("/v1/languages")
async def languages() -> list[dict]:
    return _LANGUAGES


@app.get("/v1/voices")
async def list_voices() -> list[dict]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return voice_store.list_voices(_model, _custom_styles_dir)


@app.post("/v1/voices")
async def import_voice(
    name: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    raw = await file.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}") from e
    return voice_store.import_voice(name, payload, _model, _custom_styles_dir)


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    voice_store.delete_voice(voice_id, _model, _custom_styles_dir)
    return {"deleted": voice_id}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


if settings.static_dir:
    _static_path = Path(settings.static_dir)
    if _static_path.is_dir():
        app.mount("/", StaticFiles(directory=_static_path, html=True), name="frontend")
        logfire.info("serving static frontend from {path}", path=str(_static_path))
    else:
        logfire.warning(
            "STATIC_DIR={path} is set but not a directory; static serving disabled",
            path=settings.static_dir,
        )
