"""Quality harness: synthesise an en/sk eval set under several decoding configs,
transcribe every clip with the Parakeet ASR server and report WER.

    HF_HOME=/mnt/nvme/huggingface server/.venv/bin/python bench/eval_quality.py --configs base,fast
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fastgen  # noqa: E402
from omnivoice import OmniVoice  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "out"
ASR_URL = "http://localhost:9000/v1/transcribe"

EN = [
    "Hello, this is a test of the speech server.",
    "The quick brown fox jumps over the lazy dog.",
    "Could you please send me the report by Friday afternoon?",
    "Our flight to Vienna was delayed by almost three hours.",
    "I have been looking forward to this concert for months.",
    "The machine learning model produces remarkably natural sounding speech.",
]
SK = [
    "Dobrý deň, ako sa máte?",
    "Naraz sa nás ten Kukučín zmocnil, ale všetkých nás to prekvapilo.",
    "Prosím, pošlite mi ten dokument do piatku popoludní.",
    "Náš let do Viedne meškal takmer tri hodiny.",
    "Na tento koncert sa teším už niekoľko mesiacov.",
    "Tento model strojového učenia vytvára pozoruhodne prirodzene znejúci hlas.",
]

REFS = ROOT / "bench" / "refs"
VOICES = {
    "en": (REFS / "chloe_sample.wav", (REFS / "chloe_sample.txt").read_text(encoding="utf-8")),
    "sk": (REFS / "fico.wav", (REFS / "fico.txt").read_text(encoding="utf-8")),
}

CONFIGS = {
    # name: (num_step, guidance_scale)
    "s32": (32, 2.0),
    "s24": (24, 2.0),
    "s16": (16, 2.0),
    "s12": (12, 2.0),
    "s8": (8, 2.0),
    "g0s16": (16, 0.0),
    "g0s8": (8, 0.0),
}


def norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref).split(), norm(hyp).split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(r), len(h)]) / max(1, len(r))


def transcribe(path: Path) -> str:
    with path.open("rb") as f:
        r = requests.post(ASR_URL, files={"file": (path.name, f, "audio/wav")}, timeout=300)
    r.raise_for_status()
    return r.json()["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="s32,s16,s8")
    ap.add_argument("--modes", default="clone")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patch", default="1")
    args = ap.parse_args()
    if args.patch == "1":
        fastgen.enable(OmniVoice)

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device, dtype=torch.float16)
    prompts = {}
    for lang, (path, txt) in VOICES.items():
        t0 = time.perf_counter()
        prompts[lang] = model.create_voice_clone_prompt(str(path), txt)
        print(f"prompt[{lang}] {time.perf_counter()-t0:.2f}s", flush=True)

    results = {}
    for cfg_name in args.configs.split(","):
        steps, gs = CONFIGS[cfg_name]
        for lang, texts in (("en", EN), ("sk", SK)):
            for mode in args.modes.split(","):
                rows, wers, durs, walls = [], [], [], []
                for i, text in enumerate(texts):
                    kw = {}
                    if mode == "clone":
                        kw["voice_clone_prompt"] = prompts[lang]
                    else:
                        kw["instruct"] = ("female, young adult, british accent"
                                          if lang == "en" else "male, middle-aged")
                    t0 = time.perf_counter()
                    audio = model.generate(text=text, language=lang, num_step=steps,
                                           guidance_scale=gs, **kw)[0]
                    wall = time.perf_counter() - t0
                    dur = len(audio) / model.sampling_rate
                    d = OUT / cfg_name / mode / lang
                    d.mkdir(parents=True, exist_ok=True)
                    p = d / f"{i:02d}.wav"
                    sf.write(p, audio, model.sampling_rate)
                    hyp = transcribe(p)
                    w = wer(text, hyp)
                    rows.append({"text": text, "asr": hyp, "wer": w, "audio_s": dur, "wall_s": wall})
                    wers.append(w)
                    durs.append(dur)
                    walls.append(wall)
                    print(f"{cfg_name:6s} {mode:5s} {lang} {i} audio={dur:5.2f}s wall={wall:5.2f}s wer={w:.3f}", flush=True)
                agg = {
                    "wer": statistics.mean(wers),
                    "audio_s": sum(durs),
                    "wall_s": sum(walls),
                    "rtf": sum(durs) / sum(walls),
                }
                results[f"{cfg_name}/{mode}/{lang}"] = {"agg": agg, "rows": rows}
                print(f"== {cfg_name:6s} {mode:5s} {lang}: WER={agg['wer']*100:.1f}%  audio={agg['audio_s']:.0f}s  RTF={agg['rtf']:.1f}x", flush=True)

    (OUT / f"results-{args.configs.replace(',', '_')}-{args.modes.replace(',', '_')}.json").write_text(json.dumps(results, indent=2))
    print("\n=== summary (WER %) ===")
    for k, v in results.items():
        print(f"{k:24s} {v['agg']['wer']*100:6.2f}%  RTF {v['agg']['rtf']:5.1f}x")


if __name__ == "__main__":
    main()
