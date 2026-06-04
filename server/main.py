import asyncio
import contextlib
import io
import logging
import os
import queue
import struct
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated

import transformers

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="transformers.*")

import logfire
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from omnivoice import OmniVoice
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID, LANG_NAMES, lang_display_name
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from stream2sentence import generate_sentences

from src import voices as voice_store
from src.chunker import split_to_sentences
from src.config import settings
from src.inference import infer

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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model
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
                "(e.g. 'English', 'Slovak'). See GET /languages for the full list."
            ),
        )


def _validate_ref_audio_params(
    ref_audio: UploadFile | None,
    ref_text: str | None,
    ref_voice_name: str | None,
) -> None:
    has_audio = ref_audio is not None
    has_text = bool(ref_text and ref_text.strip())
    if has_audio != has_text:
        raise HTTPException(
            status_code=422,
            detail="ref_audio and ref_text must both be provided or both omitted",
        )
    if ref_voice_name and not has_audio:
        raise HTTPException(
            status_code=422,
            detail="ref_voice_name requires ref_audio and ref_text to be provided",
        )


def _normalize(value: str | None) -> str | None:
    """Collapse empty / whitespace-only optional form fields to None."""
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


@app.post("/v1/synthesize")
async def synthesize(
    text: Annotated[str, Form()],
    language: Annotated[str, Form()] = "en",
    speed: Annotated[float | None, Form()] = None,
    ref_text: Annotated[str | None, Form()] = None,
    ref_audio: Annotated[UploadFile | None, File()] = None,
    voice_id: Annotated[str | None, Form()] = None,
    ref_voice_name: Annotated[str | None, Form()] = None,
    instruct: Annotated[str | None, Form()] = None,
    stream: Annotated[bool, Form()] = False,
) -> StreamingResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _validate_language(language)
    _validate_ref_audio_params(ref_audio, ref_text, ref_voice_name)

    instruct = _normalize(instruct)
    if instruct and (ref_audio is not None or voice_id):
        raise HTTPException(status_code=422, detail="instruct cannot be combined with ref_audio or voice_id")
    if not instruct and ref_audio is None and not voice_id:
        raise HTTPException(status_code=422, detail="voice cloning requires ref_audio or voice_id")

    # Common setup — clean up tmp_path on any error before branching.
    tmp_path: str | None = None
    try:
        if ref_audio is not None:
            tmp_path, ext = await _save_upload(ref_audio)
            if ref_voice_name:
                voice_store.save_voice_sample(
                    ref_voice_name, tmp_path, ext, ref_text.strip(), language, settings.voice_samples_dir
                )
            effective_ref, effective_ref_text = tmp_path, ref_text.strip()
        else:
            effective_ref, effective_ref_text = voice_store.resolve_voice(
                voice_id, settings.voice_samples_dir, ref_text
            )
    except Exception:
        _unlink_quietly(tmp_path)
        raise

    loop = asyncio.get_running_loop()
    synth = partial(
        infer, _model,
        language=language, speed=speed,
        ref_audio_path=effective_ref, ref_text=effective_ref_text, instruct=instruct,
    )

    if not stream:
        try:
            data = await loop.run_in_executor(_executor, partial(synth, text))
        finally:
            _unlink_quietly(tmp_path)
        return StreamingResponse(
            io.BytesIO(data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )

    sentences = split_to_sentences(text)

    async def generate():
        try:
            for i, sentence in enumerate(sentences):
                num_step = NUM_STEPS_FIRST_SENTENCE if i == 0 else NUM_STEPS
                data = await loop.run_in_executor(_executor, partial(synth, sentence, num_step=num_step))
                yield struct.pack(">I", len(data)) + data
        finally:
            _unlink_quietly(tmp_path)

    return StreamingResponse(generate(), media_type="application/octet-stream")


@app.websocket("/v1/ws/synthesize")
async def ws_synthesize(
    ws: WebSocket,
    language: str = "en",
    voice_id: str | None = None,
    speed: float | None = None,
    instruct: str | None = None,
) -> None:
    await ws.accept()

    if _model is None:
        await ws.close(code=1011, reason="Model not loaded")
        return

    if not _language_supported(language):
        await ws.close(code=1008, reason=f"Unsupported language '{language}'")
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
        sentence_idx = 0
        try:
            for sentence in generate_sentences(_text_gen()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                num_step = NUM_STEPS_FIRST_SENTENCE if sentence_idx == 0 else NUM_STEPS
                sentence_idx += 1
                audio = _executor.submit(partial(synth, sentence, num_step=num_step)).result()
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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


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
