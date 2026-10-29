# Pitfalls

Non-obvious failures hit while building and benchmarking this server, and what
the code does about them. Each one cost real time; none are theoretical.

## A reference transcript that does not match the audio

**Symptom**: a cloned voice reads the *reference text* before the requested text,
repeats itself, and produces twice as much audio as it should. English WER
measured 102 % against the input text.

**Cause**: OmniVoice aligns the reference transcript with the reference audio
inside one sequence. If the transcript covers 11 words but the clip contains 41,
the model learns the wrong speaking rate, estimates a target that is 3× too long,
and uses the spare room to keep talking. The bundled `chloe_sample` clip shipped
with a transcript of its first sentence only, which is exactly this bug.

**Fix**: the voice store never trusts a transcript it cannot match. A reference is
cut to `MAX_REF_SECONDS` on a word boundary, the kept words *are* the transcript,
and a transcript whose words-per-second falls outside `0.8–6.5` is re-derived
with the ASR service (`provenance: asr-repaired`). Measured effect on the same
voice: **WER 102 % → 0 %**, audio 58 s → 21 s for the same 6 sentences.

## Cutting a reference in the middle of a word

**Symptom**: every cloned sentence started with the word "pričasť" — the last,
half-spoken word of the reference clip.

**Cause**: references are capped at `MAX_REF_SECONDS`; the first implementation
cut at the lowest-energy 20 ms frame before the limit. Plosives have silent
closures, so the "quietest frame" was inside a word, and the model dutifully
finished that truncated word inside the target text.

**Fix**: cut inside a *run* of quiet frames (≥ 80 ms), prefer the last
sentence-final word that fits, then a clause-final one, and only then any word.
Also start the clip at the first recognised word (`word.start − 50 ms`) so a
recording that begins mid-phoneme is trimmed too.

## Parakeet word timestamps carry unreliable punctuation

**Symptom**: the sentence/preference logic picked a 2.4 s reference even though a
6 s phrase boundary was available; the word list had `Kaliňáka` without its
period.

**Cause**: the ASR service restores punctuation in `text`/`segments`, not
reliably on every word of the word-level response.

**Fix**: prefer punctuation-terminated words, but extend the cut to the last word
followed by a ≥ 150 ms pause when the chosen cut is shorter than 3.5 s
(`_pick_reference_window`). Reference length matters for timbre quality, and a
pause is a safe boundary even without punctuation.

## Whisper translates instead of transcribing

**Symptom**: a Slovak reference clip was stored with an English transcript ("Have
a good evening, dear friends."), which then produced English-accented Slovak.

**Cause**: `whisper-large-v3` is multilingual; given `language="en"` (the store's
default for a voice without a language file) it transcribes *as English*.

**Fix**: reference transcription is delegated to the sibling Parakeet service
(auto-detecting, better on Slovak), and the voice's language is always supplied
explicitly (`POST /v1/voices` requires it, `prepare_voices.py` takes
`--language id=code`). Whisper is no longer loaded at all, which also frees
~1.6 GB of VRAM.

## `guidance_scale=0` does not save the unconditional pass

**Symptom**: setting `GUIDANCE_SCALE=0` changed nothing in latency.

**Cause**: `OmniVoice._generate_iterative` always builds a `2 × batch` forward
pass (conditional + unconditional) and only skips the *combination* when guidance
is off.

**Fix (measured, not shipped)**: a patched decoder that drops the unconditional
half when `guidance_scale == 0` is bit-identical for both settings and **1.63×
faster** at `guidance_scale=0` (411 ms vs 670 ms for a 9-word sentence at 16
steps). It is not enabled by default because turning CFG off costs accuracy —
Slovak WER went 7.9 % → 9.6 % at 16 steps. If you need the throughput, the
patch is ~90 lines against the model's private internals and must be re-verified
on every `omnivoice` upgrade.

## Request batching buys ~1.3×, not 8×

**Symptom**: the obvious "serve concurrent users by batching requests" design does
not work for this model.

**Cause**: a 0.8 B diffusion LM that runs 16 full bidirectional passes per
sentence reaches 54 of this card's ~117 TFLOPS measured GEMM peak at batch size 1
(46 % — the sequence is only 224 tokens, so the matrices are small). Eight
identical sentences in one `generate()` call take 6.1× as long as one; per-item
cost falls from 227 ms to 176 ms (22 %) and efficiency rises only to 71 TFLOPS —
a **~1.3× ceiling**, while the first item in the batch waits for a 1.4 s forward
pass. With *mixed* lengths even that
disappears: every item pads to the longest member under a dense mask, so short
sentences inherit the cost of long neighbours. Forward and `generate()` scale
identically, so this is the GPU, not the Python loop (`bench/batch_probe.py`).

**Fix**: the scheduler's unit is a sentence, not a batch. Requests are interleaved
round-robin, first chunks jump the queue, and throughput stays flat from 1 to 16
clients without adding latency. See [benchmarks](benchmarks.md).

## CUDA graphs and batching fight over the batch dimension

**Symptom**: with batching enabled, every batch failed with
`The size of tensor a (2) must match the size of tensor b (12) at non-singleton
dimension 0`, then silently fell back to per-chunk decoding (visible only as
`tts_split_batches_total` climbing while `tts_batches_total` stayed at zero).

**Cause**: a CUDA graph captures fixed shapes. The graph wrapper allocated its
static buffers for a single item (2 rows: conditional + unconditional), but a
batch of 6 items is 12 rows.

**Fix**: graphs now only capture single-item calls (`input_ids.size(0) == 2`) and
fall through to eager for batches — at that width the launch overhead is already
amortised by the matrix, so capturing every (batch, length) combination would
cost memory for ~3 %. More importantly, `torch.compile` supersedes graphs
entirely on this model (1.65× vs 1.14×), so `CUDA_GRAPHS` defaults to off and the
two are not stacked.

**Lesson**: make the fallback visible. Both of these were silent for several
benchmark runs until `tts_batches_total`/`tts_split_batches_total` and the
`fallbacks` counter in `/health` were checked.

## Per-chunk ffmpeg encoding blocks the event loop

**Symptom**: streaming `mp3`/`opus` made *all* concurrent requests slower, not
just the one being encoded.

**Cause**: every delta is a complete file, so each chunk forks ffmpeg:
~60–70 ms of a `subprocess.run` inside the async generator. During that time the
event loop cannot accept connections, read WebSocket frames or dispatch jobs.

**Fix**: `pipeline.encode()` sends ffmpeg formats to a two-thread CPU pool; `wav`,
`pcm` and `flac` (sub-millisecond) stay inline. Encoding now overlaps with the
GPU rendering the next sentence.

## transformers resets its own log level

**Symptom**: ~200 lines of `[transformers] Accessing ... will be removed in
future versions` at startup, despite `TRANSFORMERS_VERBOSITY=error` and a
`setLevel(ERROR)` before the import.

**Cause**: `transformers.utils.logging._configure_library_root_logger()` runs
during import and re-sets the `transformers` logger level (to `WARNING` when
nothing else is set).

**Fix**: set the level *after* the imports as well (`app/main.py` does it twice,
once before and once after the model-package import), or the first import always
wins.

## Sentence splitting decides your latency

**Symptom**: time-to-first-audio was 3× higher than the first chunk's inference
time.

**Cause**: the first version merged short sentences into one chunk to "save a
diffusion pass". For streaming that is backwards: "Hello there. How are you?" is
one model call, so the listener waits for both sentences.

**Fix**: one chunk per sentence (`split_text`), with a clause-level cap only for
sentences longer than `MAX_CHUNK_CHARS`. Streaming requests additionally use
fewer steps for their first chunk (`FIRST_CHUNK_STEPS`).

## The WebSocket endpoint is not a bulk upload path

**Symptom**: a client that pasted a whole document and closed the socket got
`1013` instead of audio.

**Cause**: live sessions are driven by LLM output, which arrives slower than real
time. A client pushing text faster than the GPU renders it accumulates
unbounded backlog, which turns into latency for everyone else.

**Fix**: `STREAM_MAX_PENDING_S` (default 180 s of unsynthesised text) closes the
session with `1013` and a reason. Bulk text belongs on
`POST /v1/audio/speech` (with `stream_format=sse` if progress matters).

## Root-owned Hugging Face cache

**Symptom**: `Ignored error while writing commit hash to .../refs/main:
Permission denied` on every start, and new model downloads fail.

**Cause**: the container runs as root and writes into a host cache directory owned
by another user.

**Fix**: either keep `HF_HOME` on a directory the container can write, or
pre-download the model as the container user. The warning itself is harmless
(huggingface_hub ignores it), but a read-only cache cannot fetch a new model
revision.

## Word error rate lies about numbers

**Symptom**: Slovak WER jumped from 3.9 % to 14 % with no code change.

**Cause**: the reference says "tri hodiny" (three hours), the ASR returns
"3 hodiny". A naive WER counts that as an error, and one error in an 8-word
sentence is 12 %.

**Fix**: `bench/verify_quality.py` folds digits and spelled-out numerals to a
single `NUM` token before scoring, and averages over the sentence set rather than
reporting a single sentence. Even so, generation is stochastic
(`position_temperature=5.0`): expect ±5 points of run-to-run noise on six
sentences.
