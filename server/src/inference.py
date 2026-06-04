import io

import soundfile as sf


def infer(
    model,
    text: str,
    language: str,
    speed: float | None = None,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
    num_step: int | None = None,
) -> bytes:
    """Synthesise text and return raw WAV bytes."""
    kwargs: dict = {"language": language}
    if ref_audio_path:
        kwargs["ref_audio"] = ref_audio_path
    if ref_text:
        kwargs["ref_text"] = ref_text
    if speed is not None:
        kwargs["speed"] = speed
    if instruct:
        kwargs["instruct"] = instruct
    if num_step is not None:
        kwargs["num_step"] = num_step

    audio = model.generate(text=text, **kwargs)[0]

    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=24000, channels=1,
                      format="WAV", subtype="PCM_16", closefd=False) as f:
        f.write(audio)
    buf.seek(0)
    return buf.read()
