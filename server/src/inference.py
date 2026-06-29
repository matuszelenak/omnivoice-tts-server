import numpy as np


def infer(
    model,
    text: str,
    language: str,
    speed: float | None = None,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
    num_step: int | None = None,
) -> np.ndarray:
    """Synthesise *text* and return the raw 24 kHz mono float sample array.

    Encoding into a concrete container (WAV, MP3, …) is the caller's job — see
    :mod:`src.audio`.
    """
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

    return model.generate(text=text, **kwargs)[0]
