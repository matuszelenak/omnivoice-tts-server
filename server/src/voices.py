"""Supertonic voice style store.

Manages built-in styles (M1–M5, F1–F5) plus user-imported custom style JSONs.
Custom styles live in a configurable directory alongside the bundled ones.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import HTTPException
from supertonic import TTS
from supertonic.core import Style
from supertonic.utils import validate_voice_style_format

_NAME_RE = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')


def list_voices(model: TTS, custom_styles_dir: Path) -> list[dict]:
    voices = [{"id": n, "name": n, "kind": "builtin"} for n in sorted(model.voice_style_names)]
    if custom_styles_dir.is_dir():
        for p in sorted(custom_styles_dir.glob("*.json")):
            voices.append({"id": p.stem, "name": p.stem, "kind": "custom"})
    return voices


def get_voice_style(model: TTS, voice_id: str, custom_styles_dir: Path) -> Style:
    if voice_id in model.voice_style_names:
        return model.get_voice_style(voice_id)
    custom_path = custom_styles_dir / f"{voice_id}.json"
    if custom_path.is_file():
        return model.get_voice_style_from_path(custom_path)
    raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")


def import_voice(name: str, payload: dict, model: TTS, custom_styles_dir: Path) -> dict:
    if not _NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Voice name must match [A-Za-z0-9_-]{1,64}",
        )
    if name in model.voice_style_names:
        raise HTTPException(
            status_code=409,
            detail=f"'{name}' is a built-in voice and cannot be overwritten",
        )
    if not validate_voice_style_format(payload):
        raise HTTPException(
            status_code=422,
            detail="Invalid style JSON: missing required 'style_ttl' / 'style_dp' fields",
        )
    custom_styles_dir.mkdir(parents=True, exist_ok=True)
    target = custom_styles_dir / f"{name}.json"
    tmp = target.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(target)
    return {"id": name, "name": name, "kind": "custom"}


def delete_voice(voice_id: str, model: TTS, custom_styles_dir: Path) -> None:
    if voice_id in model.voice_style_names:
        raise HTTPException(status_code=400, detail="Built-in voices cannot be deleted")
    target = custom_styles_dir / f"{voice_id}.json"
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    target.unlink()
