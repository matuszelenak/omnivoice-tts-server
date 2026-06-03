import io
import re
import shutil
from pathlib import Path

import soundfile as sf
from fastapi import HTTPException


AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

_AUDIO_MEDIA_TYPES: dict[str, str] = {
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.flac': 'audio/flac',
    '.ogg': 'audio/ogg',
    '.m4a': 'audio/mp4',
}


def find_voice_file(samples_dir: Path, voice_id: str) -> Path | None:
    """Return the first audio file matching *voice_id* stem, or None."""
    for ext in AUDIO_EXTENSIONS:
        p = samples_dir / f"{voice_id}{ext}"
        if p.is_file():
            return p
    return None


def audio_media_type(path: Path) -> str:
    return _AUDIO_MEDIA_TYPES.get(path.suffix.lower(), 'audio/wav')


def resolve_voice(
    voice_id: str | None,
    samples_dir: Path,
    ref_text: str | None,
) -> tuple[str | None, str | None]:
    """Return (ref_audio_path, ref_text) for a built-in voice ID.

    Returns (None, ref_text) unchanged when voice_id is None.
    Raises HTTPException on invalid or missing voice IDs.
    """
    if voice_id is None:
        return None, ref_text
    if "/" in voice_id or "\\" in voice_id or ".." in voice_id:
        raise HTTPException(status_code=400, detail="Invalid voice ID")
    audio_file = find_voice_file(samples_dir, voice_id)
    if audio_file is None:
        raise HTTPException(status_code=404, detail="Voice sample not found")
    if ref_text is None:
        txt = samples_dir / f"{voice_id}.txt"
        if txt.is_file():
            ref_text = txt.read_text(encoding="utf-8").strip() or None
    return str(audio_file), ref_text


def save_voice_sample(
    name: str,
    src_path: str,
    original_ext: str,
    ref_text: str,
    samples_dir: Path,
) -> None:
    """Copy *src_path* into *samples_dir* as a new named voice.

    Raises HTTPException 400 if *name* contains invalid characters.
    """
    slug = name.strip().replace(" ", "_")
    if not re.match(r'^[a-zA-Z0-9_\-]+$', slug):
        raise HTTPException(
            status_code=400,
            detail="Voice name may only contain letters, digits, hyphens, and underscores",
        )
    samples_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, samples_dir / f"{slug}{original_ext}")
    (samples_dir / f"{slug}.txt").write_text(ref_text, encoding="utf-8")


def infer(
    model,
    text: str,
    language: str,
    speed: float | None,
    ref_audio_path: str | None,
    ref_text: str | None,
    instruct: str | None = None,
    num_step: int | None = None,
) -> bytes:
    """Synthesise text and return raw WAV bytes."""
    shared: dict = {"language": language}
    if ref_audio_path:
        shared["ref_audio"] = ref_audio_path
    if ref_text:
        shared["ref_text"] = ref_text
    if speed is not None:
        shared["speed"] = speed
    if instruct:
        shared["instruct"] = instruct
    if num_step is not None:
        shared["num_step"] = num_step

    audio = model.generate(text=text, **shared)[0]

    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=24000, channels=1,
                      format="WAV", subtype="PCM_16", closefd=False) as f:
        f.write(audio)
    buf.seek(0)
    return buf.read()
