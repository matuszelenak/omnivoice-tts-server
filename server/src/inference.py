import io

import soundfile as sf
from supertonic import TTS
from supertonic.config import DEFAULT_SPEED, DEFAULT_TOTAL_STEPS
from supertonic.core import Style


def infer(
    model: TTS,
    text: str,
    language: str,
    voice_style: Style,
    speed: float | None = None,
    total_steps: int | None = None,
) -> bytes:
    """Synthesise text and return raw WAV bytes."""
    wav, _ = model.synthesize(
        text,
        voice_style,
        lang=language,
        speed=speed if speed is not None else DEFAULT_SPEED,
        total_steps=total_steps if total_steps is not None else DEFAULT_TOTAL_STEPS,
    )
    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=model.sample_rate, channels=1,
                      format="WAV", subtype="PCM_16", closefd=False) as f:
        f.write(wav.squeeze())
    buf.seek(0)
    return buf.read()
