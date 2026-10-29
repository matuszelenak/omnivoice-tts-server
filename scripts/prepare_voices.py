#!/usr/bin/env python3
"""Normalise the built-in voice references in ``server/voices``.

The model is extremely sensitive to a reference transcript that does not match
the reference audio: it will *speak the reference text* before the requested
text. Bundled samples are often longer than 10 s and their transcripts cover
only part of the clip, so every voice is rebuilt here:

1. decode → mono 24 kHz, trim edge silence, normalise level
2. ASR with word timestamps → cut at the last complete clause below
   ``MAX_REF_SECONDS`` (a cut inside a word makes the model continue it)
3. write ``voice.wav`` (normalised clip), ``<lang>.txt`` (exact transcript) and
   ``meta.json``; the untouched original is kept as ``source.<ext>``

Usage (from the repository root, with the model on GPU 0):

    HF_HOME=/mnt/nvme/huggingface CUDA_VISIBLE_DEVICES=0 \\
        server/.venv/bin/python scripts/prepare_voices.py [voice_id ...]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from app.asr import ASRClient  # noqa: E402
from app.audio import SAMPLE_RATE, decode_audio, wav_bytes  # noqa: E402
from app.config import settings  # noqa: E402
from app.engine import engine  # noqa: E402
from app.voices import VoiceRegistry, VoiceStore  # noqa: E402

# Voices whose language cannot be inferred from a <lang>.txt / meta.json.
KNOWN_LANGUAGES = {"fico": "sk", "kroner": "sk", "chloe_sample": "en"}


def main() -> int:
    argv = sys.argv[1:]
    overrides: dict[str, str] = {}
    args: list[str] = []
    it = iter(argv)
    for arg in it:
        if arg == "--language":
            for pair in next(it).split(","):
                voice_id, _, code = pair.partition("=")
                overrides[voice_id] = code
        elif arg.startswith("--language="):
            for pair in arg.split("=", 1)[1].split(","):
                voice_id, _, code = pair.partition("=")
                overrides[voice_id] = code
        else:
            args.append(arg)
    wanted = set(args)
    store = VoiceStore(settings.voice_samples_dir)
    engine.load()
    asr = ASRClient(settings.asr_url, settings.asr_timeout_s)
    registry = VoiceRegistry(store, engine, asr)
    if asr.enabled and not asr.check():
        print(f"warning: ASR service at {asr.base_url} unreachable", file=sys.stderr)

    for voice in store.list():
        if wanted and voice.id not in wanted:
            continue
        original = voice.audio_path
        source = original.with_name(f"source{original.suffix}")
        if not source.exists():
            shutil.copy2(original, source)
        raw = decode_audio(source.read_bytes(), source.name)
        language = overrides.get(voice.id) or voice.language or KNOWN_LANGUAGES.get(voice.id)
        if language is None:
            print(f"skip {voice.id}: unknown language (use --language {voice.id}=XX)", file=sys.stderr)
            continue
        prepared = registry.prepare_reference(raw, language, None)
        samples, text, provenance = prepared.samples, prepared.text, prepared.provenance
        voice_dir = original.parent
        (voice_dir / "voice.wav").write_bytes(wav_bytes(samples))
        if original.suffix.lower() != ".wav":
            original.unlink(missing_ok=True)
        language = prepared.language or language
        (voice_dir / f"{language}.txt").write_text(text, encoding="utf-8")
        for txt in voice_dir.glob("*.txt"):
            if txt.stem != language:
                txt.unlink()
        (voice_dir / "meta.json").write_text(
            json.dumps(
                {
                    "id": voice.id,
                    "name": voice.name,
                    "language": language,
                    "ref_text": text,
                    "duration": round(len(samples) / SAMPLE_RATE, 3),
                    "source": "prepare_voices",
                    "provenance": provenance,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{voice.id}: {len(samples)/SAMPLE_RATE:.2f}s [{provenance}] {text}")
    engine.shutdown()
    asr.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
