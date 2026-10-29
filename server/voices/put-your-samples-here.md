# Voices

Each directory is one voice, identified by its name:

```
voices/
  chloe_sample/
    voice.wav     # prepared reference: mono 24 kHz, 3-8 s, edge silence trimmed
    en.txt        # transcript of exactly that clip (file stem = language)
    meta.json     # name, language, duration, provenance
    source.wav    # the untouched upload (optional, not shipped in the image)
```

The transcript **must** match the audio word for word. OmniVoice aligns the two
inside one sequence; a transcript that covers only half the clip makes the model
speak the reference text before your text (see `docs/pitfalls.md`).

Do not hand-edit these files. Drop a raw clip anywhere and let the server prepare
it:

```bash
# register through the API (recommended: transcribes + normalises for you)
curl -X POST http://localhost:9001/v1/voices \
  -F name=my_voice -F language=sk -F ref_audio=@my_clip.wav

# or prepare clips already placed in this directory
server/.venv/bin/python scripts/prepare_voices.py --language my_voice=sk
```

`scripts/prepare_voices.py` trims silence, normalises level, cuts the clip at a
word boundary below `MAX_REF_SECONDS`, and writes the transcript from the words it
kept (via the Parakeet ASR service, `ASR_URL`).
