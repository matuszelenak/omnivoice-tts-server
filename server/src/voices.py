"""Voice sample store.

Each voice lives in its own directory under the samples dir:
``<samples_dir>/<voice_id>/voice.<ext>`` + ``<samples_dir>/<voice_id>/<language>.txt``.
The directory name is the voice ID; the transcript file's stem is the voice's language.
"""
import re
import shutil
from pathlib import Path

from fastapi import HTTPException

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

_AUDIO_MEDIA_TYPES: dict[str, str] = {
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.flac': 'audio/flac',
    '.ogg': 'audio/ogg',
    '.m4a': 'audio/mp4',
}


def audio_media_type(path: Path) -> str:
    return _AUDIO_MEDIA_TYPES.get(path.suffix.lower(), 'audio/wav')


def validate_voice_id(voice_id: str) -> None:
    """Reject IDs that could escape the samples directory."""
    if "/" in voice_id or "\\" in voice_id or ".." in voice_id:
        raise HTTPException(status_code=400, detail="Invalid voice ID")


def find_voice_file(samples_dir: Path, voice_id: str) -> Path | None:
    """Return the audio file (`voice.<ext>`) inside the voice's directory, or None."""
    voice_dir = samples_dir / voice_id
    for ext in AUDIO_EXTENSIONS:
        p = voice_dir / f"voice{ext}"
        if p.is_file():
            return p
    return None


def find_voice_transcript(samples_dir: Path, voice_id: str) -> tuple[str | None, str | None]:
    """Return (ref_text, language) from the `<lang>.txt` file in the voice's directory.

    Returns (None, None) when the directory or transcript is missing.
    """
    voice_dir = samples_dir / voice_id
    if not voice_dir.is_dir():
        return None, None
    for txt in sorted(voice_dir.glob("*.txt")):
        text = txt.read_text(encoding="utf-8").strip()
        if text:
            return text, txt.stem
    return None, None


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
    validate_voice_id(voice_id)
    audio_file = find_voice_file(samples_dir, voice_id)
    if audio_file is None:
        raise HTTPException(status_code=404, detail="Voice sample not found")
    if ref_text is None:
        ref_text, _ = find_voice_transcript(samples_dir, voice_id)
    return str(audio_file), ref_text


def list_voices(samples_dir: Path) -> list[dict]:
    """Describe every voice in *samples_dir* (id, display name, audio, transcript, language)."""
    if not samples_dir.is_dir():
        return []
    result = []
    for d in sorted(samples_dir.iterdir()):
        if not d.is_dir():
            continue
        audio = find_voice_file(samples_dir, d.name)
        if audio is None:
            continue
        ref_text, language = find_voice_transcript(samples_dir, d.name)
        result.append({
            "id": d.name,
            "name": d.name.replace("_", " ").replace("-", " ").title(),
            "filename": audio.name,
            "ref_text": ref_text,
            "language": language,
        })
    return result


def save_voice_sample(
    name: str,
    src_path: str,
    original_ext: str,
    ref_text: str,
    language: str,
    samples_dir: Path,
) -> None:
    """Copy *src_path* into its own directory under *samples_dir* as a new named voice.

    Raises HTTPException 400 if *name* contains invalid characters.
    """
    slug = name.strip().replace(" ", "_")
    if not re.match(r'^[a-zA-Z0-9_\-]+$', slug):
        raise HTTPException(
            status_code=400,
            detail="Voice name may only contain letters, digits, hyphens, and underscores",
        )
    voice_dir = samples_dir / slug
    voice_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, voice_dir / f"voice{original_ext}")
    (voice_dir / f"{language}.txt").write_text(ref_text, encoding="utf-8")


def delete_voice(samples_dir: Path, voice_id: str) -> None:
    """Remove a voice's directory entirely. Raises 400/404 on bad IDs."""
    validate_voice_id(voice_id)
    if find_voice_file(samples_dir, voice_id) is None:
        raise HTTPException(status_code=404, detail="Voice sample not found")
    shutil.rmtree(samples_dir / voice_id)
