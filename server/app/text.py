"""Text segmentation.

Three jobs:

* ``split_text`` — split a finished text into model-sized chunks. Sentence
  boundaries come from BlingFire; sentences that are still longer than
  ``MAX_CHUNK_CHARS`` are cut again at clause boundaries, then at word
  boundaries. One chunk = one model call = one audio frame on the wire.
* ``SentenceStreamer`` — incremental variant for LLM-style input that arrives a
  few tokens at a time. It yields the first fragment as early as possible
  (latency) and then accumulates proper sentences (natural prosody).
* ``estimate_audio_seconds`` — cheap duration estimate for admission control and
  for deciding whether a chunk needs splitting.

Nothing here imports torch or the model: it is pure text and therefore cheap to
run on the event loop.
"""
from __future__ import annotations

import re

from blingfire import text_to_sentences

# Punctuation that ends a sentence, and weaker marks that can end a clause when
# a sentence has to be cut to size.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")
_CLAUSE_END = re.compile(r"[,;:—–]+[\"'”’)\]]*(?=\s|$)")

# Tokens that end in a period without ending a sentence. Lowercased, trailing
# period stripped. Kept deliberately small: BlingFire handles the batch path,
# this list only matters for the incremental streamer.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "no", "fig", "al", "approx", "dept", "est", "inc", "ltd", "corp", "co",
    "ing", "mgr", "bc", "phd", "a.m", "p.m", "napr", "atď", "t.j", "tzv",
    "č", "str", "obr", "tab", "mil", "mld", "tis",
}

_WS = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    """Collapse runs of whitespace, drop control characters, strip."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch if ch == "\n" or not ch.isspace() or ch == " " else " " for ch in text)
    text = _WS.sub(" ", text)
    return text.strip()


def _words_before(text: str, pos: int) -> str:
    start = max(0, text.rfind(" ", 0, pos) + 1)
    return text[start:pos].strip().lower().rstrip(".")


def _is_boundary(text: str, match: re.Match[str]) -> bool:
    """True when the punctuation match at *match* really ends a sentence.

    Rejects abbreviations ("Dr. Smith"), initials ("J. R. R.") and decimals, and
    requires the next non-space character to start a new sentence (uppercase,
    digit or quote) — "e.g. this" must not split.
    """
    end = match.end()
    tail = text[end:end + 2]
    if tail and not tail[0].isspace():
        return False
    if text[match.start()] == "." and _words_before(text, match.start()) in _ABBREVIATIONS:
        return False
    nxt = text[end:].lstrip()
    if not nxt:
        return True  # end of buffer: a sentence if the caller flushes
    ch = nxt[0]
    return ch.isupper() or ch.isdigit() or ch in "\"'“‘([{"


def _hard_split(text: str, limit: int) -> list[str]:
    """Split *text* into pieces of at most *limit* chars on word boundaries."""
    if len(text) <= limit:
        return [text]
    out, rest = [], text
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return [p for p in out if p]


def _clause_split(sentence: str, limit: int) -> list[str]:
    """Cut an over-long sentence at clause marks, preferring late breaks."""
    if len(sentence) <= limit:
        return [sentence]
    pieces: list[str] = []
    rest = sentence
    while len(rest) > limit:
        window = rest[:limit]
        # Prefer the last clause mark in the second half of the window, so
        # chunks stay long and natural.
        best = -1
        for m in _CLAUSE_END.finditer(window):
            if m.end() >= limit // 2:
                best = m.end()
        if best <= 0:
            cut = window.rfind(" ", limit // 2, limit)
            best = cut if cut > 0 else -1
        if best <= 0:
            break
        pieces.append(rest[:best].strip())
        rest = rest[best:].strip()
    if rest:
        pieces.extend(_hard_split(rest, limit))
    return [p for p in pieces if p]


def split_text(text: str, max_chunk_chars: int = 320) -> list[str]:
    """Split *text* into chunks that each fit in one model call."""
    text = normalize_space(text)
    if not text:
        return []
    sentences = [s.strip() for s in text_to_sentences(text).splitlines() if s.strip()]
    if not sentences:
        sentences = [text]
    chunks: list[str] = []
    for sentence in sentences:
        chunks.extend(_clause_split(sentence, max_chunk_chars))
    return chunks


class SentenceStreamer:
    """Incremental sentence splitter for text that arrives in fragments.

    ``feed(text)`` returns the sentences completed by that fragment (usually an
    empty list). ``flush()`` returns whatever is left. The first emission is
    allowed to be a clause instead of a full sentence so the listener hears
    something as soon as possible; later emissions wait for a real sentence end
    so prosody does not degrade.
    """

    def __init__(
        self,
        max_chunk_chars: int = 320,
        min_chunk_chars: int = 8,
        first_chunk_chars: int = 48,
    ) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.first_chunk_chars = first_chunk_chars
        self._buf = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk = normalize_space(self._buf[:cut])
            self._buf = self._buf[cut:]
            if chunk:
                out.append(chunk)
                self._emitted += 1
        # A single fragment longer than the limit must not grow without bound:
        # pick a clause/word break instead of waiting for punctuation.
        while len(self._buf) > self.max_chunk_chars:
            chunk = _clause_split(self._buf, self.max_chunk_chars)[0]
            if not chunk:
                break
            out.append(chunk)
            self._buf = self._buf[len(chunk):]
            self._emitted += 1
        return out

    def flush(self) -> list[str]:
        chunk = normalize_space(self._buf)
        self._buf = ""
        if not chunk:
            return []
        self._emitted += 1
        return [chunk]

    def _find_cut(self) -> int | None:
        first = self._emitted == 0
        # First fragment: accept a clause break once it is long enough to sound
        # like speech; afterwards require sentence punctuation.
        patterns = (_SENTENCE_END, _CLAUSE_END) if first else (_SENTENCE_END,)
        for pattern in patterns:
            for m in pattern.finditer(self._buf):
                if not _is_boundary(self._buf, m):
                    continue
                if len(self._buf[: m.end()].strip()) < self.min_chunk_chars:
                    continue
                if first and pattern is _CLAUSE_END and len(self._buf[: m.end()]) < self.first_chunk_chars:
                    continue
                return m.end()
        return None


def estimate_audio_seconds(text: str, speed: float = 1.0) -> float:
    """Rough spoken duration of *text* in seconds.

    Deliberately conservative (~15 characters per second, i.e. slower than
    measured synthesis) because it feeds admission control: under-estimating
    would let the queue grow beyond what the GPU can drain.
    """
    chars = len(normalize_space(text))
    return max(0.4, chars / 15.0) / max(0.25, speed)
