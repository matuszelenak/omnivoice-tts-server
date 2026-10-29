"""Voice store: reference clips on disk, plus the prompt cache in front of them.

A voice is a directory under ``VOICE_SAMPLES_DIR``::

    voices/alice/
        voice.wav     # mono 24 kHz reference, ideally 3-8 s
        en.txt        # transcript of exactly that clip (file stem = language)
        meta.json     # optional: name, language, duration, provenance

Registering a voice is not just "copy the file": the model degrades badly when
the reference transcript does not line up with the reference audio (it *speaks
the reference text* before the target). Preparation therefore trims edge
silence, cuts the clip on an ASR word boundary below ``MAX_REF_SECONDS`` and uses
exactly the words that were kept as the transcript.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import metrics
from .asr import ASRClient, ASRError, Word
from .audio import SAMPLE_RATE, cap_duration, decode_audio, normalize_rms, trim_silence
from .config import settings
from .engine import PromptEntry, TTSEngine

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a")
_AUDIO_MEDIA_TYPES = {
    ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg",
    ".opus": "audio/ogg", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
}
_VOICE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_TERMINAL = ".!?…"


class VoiceError(Exception):
    """Bad request against the voice store (maps to 4xx)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class PreparedReference:
    samples: np.ndarray
    text: str
    language: str | None
    provenance: str


@dataclass
class Voice:
    id: str
    name: str
    language: str | None
    ref_text: str | None
    audio_path: Path
    duration: float | None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "filename": self.audio_path.name,
            "ref_text": self.ref_text,
            "language": self.language,
            "duration": round(self.duration, 2) if self.duration else None,
        }


def audio_media_type(path: Path) -> str:
    return _AUDIO_MEDIA_TYPES.get(path.suffix.lower(), "audio/wav")


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class VoiceStore:
    """Filesystem-backed voice catalogue (no model, no GPU)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- lookup ------------------------------------------------------------
    def _dir(self, voice_id: str) -> Path:
        if not _VOICE_ID_RE.match(voice_id or ""):
            raise VoiceError("invalid voice id")
        return self.root / voice_id

    def _audio_file(self, voice_dir: Path) -> Path | None:
        for suffix in AUDIO_SUFFIXES:
            candidate = voice_dir / f"voice{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _meta(self, voice_dir: Path) -> dict[str, Any]:
        path = voice_dir / "meta.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("ignoring unreadable %s", path)
        return {}

    def _transcript(self, voice_dir: Path) -> tuple[str | None, str | None]:
        meta = self._meta(voice_dir)
        for txt in sorted(voice_dir.glob("*.txt")):
            text = txt.read_text(encoding="utf-8").strip()
            if text:
                return text, meta.get("language") or txt.stem
        return None, meta.get("language")

    def get(self, voice_id: str) -> Voice:
        voice_dir = self._dir(voice_id)
        audio = self._audio_file(voice_dir)
        if audio is None:
            available = ", ".join(v.id for v in self.list()[:8]) or "none registered"
            raise VoiceError(
                f"voice '{voice_id}' not found (available: {available})", status=404
            )
        ref_text, language = self._transcript(voice_dir)
        meta = self._meta(voice_dir)
        return Voice(
            id=voice_id,
            name=meta.get("name") or voice_id.replace("_", " ").replace("-", " ").title(),
            language=language,
            ref_text=ref_text,
            audio_path=audio,
            duration=meta.get("duration"),
        )

    def try_get(self, voice_id: str) -> Voice | None:
        try:
            return self.get(voice_id)
        except VoiceError:
            return None

    def list(self) -> list[Voice]:
        if not self.root.is_dir():
            return []
        voices = []
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir() and self._audio_file(entry) is not None:
                try:
                    voices.append(self.get(entry.name))
                except VoiceError:  # pragma: no cover - defensive
                    continue
        return voices

    def prompt_key(self, voice: Voice) -> str:
        """Cache key that changes whenever the clip or transcript changes."""
        stat = voice.audio_path.stat()
        digest = hashlib.sha256(
            f"{voice.id}|{stat.st_mtime_ns}|{stat.st_size}|{voice.ref_text or ''}".encode()
        ).hexdigest()[:16]
        return f"voice:{voice.id}:{digest}"

    # -- mutation ----------------------------------------------------------
    def save(
        self,
        name: str,
        samples: np.ndarray,
        ref_text: str,
        language: str,
        *,
        audio_format: str = "wav",
        source: str = "upload",
    ) -> Voice:
        slug = re.sub(r"\s+", "_", name.strip())
        if not _VOICE_ID_RE.match(slug):
            raise VoiceError(
                "voice name may only contain letters, digits, hyphens and underscores"
            )
        voice_dir = self.root / slug
        voice_dir.mkdir(parents=True, exist_ok=True)
        from .audio import wav_bytes

        _write_atomic(voice_dir / "voice.wav", wav_bytes(samples))
        (voice_dir / f"{language}.txt").write_text(ref_text, encoding="utf-8")
        meta = {
            "id": slug,
            "name": name.strip(),
            "language": language,
            "ref_text": ref_text,
            "duration": round(len(samples) / SAMPLE_RATE, 3),
            "source": source,
            "format": audio_format,
        }
        (voice_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        # Only one transcript file may exist: drop others (e.g. after a language change).
        for txt in voice_dir.glob("*.txt"):
            if txt.stem != language:
                txt.unlink(missing_ok=True)
        log.info("voice saved: %s (%.2fs, %s)", slug, len(samples) / SAMPLE_RATE, language)
        return self.get(slug)

    def delete(self, voice_id: str) -> None:
        voice_dir = self._dir(voice_id)
        if self._audio_file(voice_dir) is None:
            raise VoiceError(f"voice '{voice_id}' not found", status=404)
        shutil.rmtree(voice_dir)
        log.info("voice deleted: %s", voice_id)


def _pick_reference_window(
    words: list[Word], max_seconds: float, min_seconds: float, preferred_seconds: float = 3.5
) -> tuple[int, int, str] | None:
    """Choose which words to keep and where to cut the audio.

    Returns ``(start_sample, end_sample, text)`` or ``None`` when nothing is
    usable. The rules, in order of importance:

    1. never cut inside a word — the model would finish that word *inside the
       target text* (see docs/pitfalls.md);
    2. prefer a sentence-final word that fits, then a clause-final one;
    3. keep at least ``preferred_seconds`` when possible, using pauses as
       boundaries, because longer references clone the timbre better;
    4. start at the first word, not at a burst the ASR placed mid-phoneme.
    """
    inside = [w for w in words if w.end is not None and w.end <= max_seconds]
    if not inside:
        return None

    def last_ending(marks: str, floor: float) -> Word | None:
        candidates = [w for w in inside if w.text[-1:] in marks and w.end >= floor]
        return candidates[-1] if candidates else None

    chosen = last_ending(_TERMINAL, min_seconds) or last_ending(",;:", min_seconds) or inside[-1]
    if chosen.end < preferred_seconds and chosen is not inside[-1]:
        # Short reference: extend to the last word followed by an audible pause.
        pauses = [
            w for i, w in enumerate(inside[:-1])
            if w.end >= preferred_seconds and inside[i + 1].start - w.end >= 0.15
        ]
        if pauses:
            chosen = pauses[-1]
    idx = inside.index(chosen)
    kept = inside[: idx + 1]
    text = re.sub(r"\s+", " ", " ".join(w.text for w in kept)).strip()
    if text and text[-1] not in _TERMINAL + ",;:":
        text += "."
    start_sample = max(0, round((inside[0].start - 0.05) * SAMPLE_RATE))
    end_sample = round((chosen.end + 0.15) * SAMPLE_RATE)
    return start_sample, end_sample, text


def _prompt_tokens(prompt: Any) -> int:
    """Reference length in audio frames (0 for stubs/unknown)."""
    tokens = getattr(prompt, "ref_audio_tokens", None)
    return int(tokens.shape[-1]) if tokens is not None else 0


def _words_per_second(text: str, seconds: float) -> float:
    return len(text.split()) / max(seconds, 0.1)


class VoiceRegistry:
    """Store + engine: prepares references, owns the prompt cache entries."""

    def __init__(self, store: VoiceStore, engine: TTSEngine, asr: ASRClient) -> None:
        self.store = store
        self.engine = engine
        self.asr = asr

    # -- preparation (CPU + remote ASR, no GPU) ----------------------------
    def prepare_reference(
        self,
        samples: np.ndarray,
        language: str | None,
        ref_text: str | None = None,
        *,
        max_seconds: float | None = None,
        min_seconds: float | None = None,
    ) -> PreparedReference:
        """Normalise a reference clip and make sure its transcript matches.

        Pure CPU + (optional) ASR service call, so it runs on a normal worker
        thread. The prompt itself is built later on the GPU thread.
        """
        max_seconds = max_seconds if max_seconds is not None else settings.max_ref_seconds
        min_seconds = min_seconds if min_seconds is not None else settings.min_ref_seconds
        samples = trim_silence(np.asarray(samples, dtype=np.float32))
        if len(samples) / SAMPLE_RATE < min_seconds:
            raise VoiceError(
                f"reference audio is too short ({len(samples)/SAMPLE_RATE:.2f}s); "
                f"provide at least {min_seconds:.1f}s of speech"
            )
        samples = normalize_rms(samples)
        needs_cut = len(samples) / SAMPLE_RATE > max_seconds
        user_text = (ref_text or "").strip() or None
        provenance = "provided"

        def _asr_words():
            if not self.asr.check():
                return []
            try:
                return self.asr.transcribe_words(samples)
            except ASRError as exc:
                log.warning("reference ASR failed: %s", exc)
                return []

        if needs_cut or user_text is None:
            words = _asr_words()
            if needs_cut:
                window = _pick_reference_window(words, max_seconds, min_seconds)
                if window is None:
                    if user_text is None:
                        raise VoiceError(
                            "reference audio is longer than "
                            f"{max_seconds:.0f}s and could not be transcribed; "
                            "upload a shorter clip or pass ref_text"
                        )
                    samples = cap_duration(samples, max_seconds, min_seconds)
                else:
                    start_sample, end_sample, asr_text = window
                    samples = samples[start_sample:end_sample]
                    if user_text is None:
                        user_text = asr_text
                        provenance = "asr"
            if user_text is None and words:
                asr_text = " ".join(w.text for w in words).strip()
                if asr_text:
                    # Keep exactly the recognised span: a clip that opens or ends
                    # mid-word confuses the model the same way a mid-word cut
                    # does, and trailing silence only wastes prompt tokens.
                    begin = max(0, int((words[0].start - 0.05) * SAMPLE_RATE))
                    finish = min(len(samples), int((words[-1].end + 0.15) * SAMPLE_RATE))
                    samples = samples[begin:finish]
                    user_text = asr_text if asr_text[-1] in _TERMINAL else asr_text + "."
                    provenance = "asr"

        if user_text is None:
            raise VoiceError(
                "reference transcript is required when no ASR service is configured "
                "(set ASR_URL or pass ref_text)"
            )
        seconds = len(samples) / SAMPLE_RATE
        wps = _words_per_second(user_text, seconds)
        if not 0.8 <= wps <= 6.5:
            # A transcript that is far too short or too long for the clip makes
            # the model speak the reference text itself; repair it with ASR.
            log.warning(
                "reference transcript looks misaligned (%.1f words/s over %.2fs); "
                "re-transcribing", wps, seconds,
            )
            asr_text = ""
            if self.asr.check():
                try:
                    asr_text = self.asr.transcribe_text(samples)
                except ASRError as exc:
                    raise VoiceError(f"reference transcript does not match the audio: {exc}") from exc
            if not asr_text:
                raise VoiceError(
                    "reference transcript does not match the audio; pass a transcript "
                    "of exactly the uploaded clip"
                )
            user_text = asr_text if asr_text[-1] in _TERMINAL else asr_text + "."
            provenance = "asr-repaired"
        return PreparedReference(samples, user_text, language, provenance)

    # -- prompt cache ------------------------------------------------------
    def prompt_key_for(self, voice: Voice) -> str:
        return self.store.prompt_key(voice)

    def ensure_prompt(self, voice: Voice) -> str:
        """Return the cache key for *voice*, building the prompt if needed."""
        key = self.prompt_key_for(voice)
        if self.engine.prompts.get(key) is not None:
            return key
        samples = decode_audio(voice.audio_path.read_bytes(), voice.audio_path.name)
        ref_text = voice.ref_text
        if not ref_text or len(samples) / SAMPLE_RATE > settings.max_ref_seconds * 1.5:
            # Legacy clip (no transcript, or never normalised): prepare it now.
            prepared = self.prepare_reference(samples, voice.language, ref_text)
            samples, ref_text = prepared.samples, prepared.text
        prompt = self.engine.create_prompt(samples, ref_text)
        self.engine.prompts.put(
            key,
            PromptEntry(
                prompt=prompt,
                language_hint=voice.language,
                tokens=_prompt_tokens(prompt),
            ),
        )
        log.info("voice prompt built: %s (%.2fs reference)", voice.id, len(samples) / SAMPLE_RATE)
        return key

    def prompt_key_for_upload(self, samples: np.ndarray, ref_text: str) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(samples, dtype=np.float32).tobytes())
        digest.update(b"|")
        digest.update(ref_text.encode("utf-8"))
        return f"upload:{digest.hexdigest()[:24]}"

    def register_upload(self, samples: np.ndarray, ref_text: str) -> str:
        key = self.prompt_key_for_upload(samples, ref_text)
        if self.engine.prompts.get(key) is None:
            prompt = self.engine.create_prompt(samples, ref_text)
            self.engine.prompts.put(
                key,
                PromptEntry(
                    prompt=prompt, language_hint=None, tokens=_prompt_tokens(prompt)
                ),
            )
        return key

    # -- high-level operations (async wrappers) ----------------------------
    async def resolve_voice(self, voice_id: str) -> tuple[str, Any]:
        """Return ``(cache_key, prompt)`` for a stored voice, building it if needed."""
        voice = self.store.get(voice_id)
        loop = asyncio.get_running_loop()
        try:
            key = await loop.run_in_executor(self.engine.executor, self.ensure_prompt, voice)
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(f"could not prepare voice '{voice_id}': {exc}", status=422) from exc
        entry = self.engine.prompts.get(key)
        return key, (entry.prompt if entry is not None else None)

    async def resolve_upload(self, audio_bytes: bytes, filename: str, ref_text: str | None,
                             language: str | None) -> tuple[str, Any]:
        loop = asyncio.get_running_loop()
        try:
            samples = await asyncio.to_thread(decode_audio, audio_bytes, filename)
            prepared = await asyncio.to_thread(
                self.prepare_reference, samples, language, ref_text
            )
            key = await loop.run_in_executor(
                self.engine.executor, self.register_upload, prepared.samples, prepared.text
            )
            entry = self.engine.prompts.get(key)
            return key, (entry.prompt if entry is not None else None)
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(f"could not prepare reference audio: {exc}", status=422) from exc

    async def add_voice(self, name: str, audio_bytes: bytes, filename: str,
                        ref_text: str | None, language: str) -> Voice:
        try:
            samples = await asyncio.to_thread(decode_audio, audio_bytes, filename)
            prepared = await asyncio.to_thread(
                self.prepare_reference, samples, language, ref_text
            )
            voice = await asyncio.to_thread(
                self.store.save, name, prepared.samples, prepared.text,
                prepared.language or language, source="upload",
            )
        except VoiceError as exc:
            metrics.voice_operations_total.labels(operation="create", outcome="rejected").inc()
            raise exc
        await asyncio.to_thread(self.ensure_prompt, voice)
        metrics.voice_operations_total.labels(operation="create", outcome="ok").inc()
        return voice

    def delete_voice(self, voice_id: str) -> None:
        voice = self.store.get(voice_id)
        self.engine.prompts.invalidate(self.store.prompt_key(voice))
        self.store.delete(voice_id)
        metrics.voice_operations_total.labels(operation="delete", outcome="ok").inc()

    async def prewarm(self) -> None:
        """Build prompts for the built-in voices during startup."""
        voices = self.store.list()
        if not voices:
            return
        loop = asyncio.get_running_loop()
        for voice in voices:
            try:
                await loop.run_in_executor(self.engine.executor, self.ensure_prompt, voice)
            except Exception as exc:
                log.warning("could not prewarm voice %s: %s", voice.id, exc)
