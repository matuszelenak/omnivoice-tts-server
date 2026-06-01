import asyncio
import io
import logging
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import transformers

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="transformers.*")

import logfire
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from omnivoice import OmniVoice
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID, LANG_NAMES, lang_display_name
from starlette.staticfiles import StaticFiles

from src.chunker import split_text
from src.config import settings


logfire.configure(
    service_name="parakeet-asr-server",
    send_to_logfire="if-token-present",
)


_executor = ThreadPoolExecutor(max_workers=1)
_model: OmniVoice | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model
    device = os.getenv("DEVICE_MAP", "cuda:0")
    _model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float16,
    )
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


def _validate_language(language: str) -> None:
    if language not in LANG_IDS and language.lower() not in LANG_NAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported language '{language}'. "
                "Use an ISO 639-3 code (e.g. 'en', 'sk') or a full name "
                "(e.g. 'English', 'Slovak'). See GET /languages for the full list."
            ),
        )


def _infer(
    text: str,
    language: str,
    speed: float | None,
    ref_audio_path: str | None,
    ref_text: str | None,
) -> io.BytesIO:
    shared: dict = {"language": language}
    if ref_audio_path:
        shared["ref_audio"] = ref_audio_path
    if ref_text:
        shared["ref_text"] = ref_text
    if speed is not None:
        shared["speed"] = speed

    chunks = split_text(text)
    parts: list[np.ndarray] = [
        _model.generate(text=chunk, **shared)[0] for chunk in chunks
    ]

    combined = np.concatenate(parts) if len(parts) > 1 else parts[0]

    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=24000, channels=1,
                      format="WAV", subtype="PCM_16", closefd=False) as f:
        f.write(combined)
    buf.seek(0)
    return buf


@app.post("/v1/synthesize")
async def synthesize(
    text: Annotated[str, Form()],
    language: Annotated[str, Form()] = "en",
    speed: Annotated[float | None, Form()] = None,
    ref_text: Annotated[str | None, Form()] = None,
    ref_audio: Annotated[UploadFile | None, File()] = None,
    voice_id: Annotated[str | None, Form()] = None,
) -> StreamingResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _validate_language(language)

    builtin_path: str | None = None
    if voice_id is not None:
        if "/" in voice_id or "\\" in voice_id or ".." in voice_id:
            raise HTTPException(status_code=400, detail="Invalid voice ID")
        sample = settings.samples_dir / f"{voice_id}.wav"
        if not sample.is_file():
            raise HTTPException(status_code=404, detail="Voice sample not found")
        builtin_path = str(sample)
        if ref_text is None:
            txt = settings.samples_dir / f"{voice_id}.txt"
            if txt.is_file():
                ref_text = txt.read_text(encoding="utf-8").strip() or None

    tmp_path: str | None = None
    try:
        if ref_audio is not None:
            ext = os.path.splitext(ref_audio.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await ref_audio.read())
                tmp_path = tmp.name

        effective_ref = tmp_path or builtin_path

        loop = asyncio.get_event_loop()
        buf = await loop.run_in_executor(
            _executor, _infer, text, language, speed, effective_ref, ref_text
        )
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.get("/v1/languages")
async def languages() -> list[dict]:
    return _LANGUAGES


@app.get("/v1/voices")
async def voices() -> list[dict]:
    if not settings.samples_dir.is_dir():
        return []
    result = []
    for f in sorted(settings.samples_dir.glob("*.wav")):
        voice_id = f.stem
        name = voice_id.replace("_", " ").replace("-", " ").title()
        txt = settings.samples_dir / f"{voice_id}.txt"
        ref_text = txt.read_text(encoding="utf-8").strip() if txt.is_file() else None
        result.append({"id": voice_id, "name": name, "filename": f.name, "ref_text": ref_text})
    return result


@app.get("/v1/voices/{voice_id}/preview")
async def voice_preview(voice_id: str) -> FileResponse:
    if "/" in voice_id or "\\" in voice_id or ".." in voice_id:
        raise HTTPException(status_code=400, detail="Invalid voice ID")
    path = settings.samples_dir / f"{voice_id}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Voice sample not found")
    return FileResponse(path, media_type="audio/wav")


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict:
    if "/" in voice_id or "\\" in voice_id or ".." in voice_id:
        raise HTTPException(status_code=400, detail="Invalid voice ID")
    wav = settings.samples_dir / f"{voice_id}.wav"
    if not wav.is_file():
        raise HTTPException(status_code=404, detail="Voice sample not found")
    wav.unlink()
    txt = settings.samples_dir / f"{voice_id}.txt"
    if txt.is_file():
        txt.unlink()
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
