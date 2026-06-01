import re

_PARA_BREAK = re.compile(r"\n{2,}")
# Split after sentence-ending punctuation when followed by whitespace.
# The negative lookahead for a digit prevents splitting on decimal numbers ("3.14")
# or common abbreviations followed by a number.
_SENT_BREAK = re.compile(r"(?<=[.!?;…])(?!\d)\s+")


def split_text(text: str, max_chars: int = 300) -> list[str]:
    """
    Split *text* into chunks at sentence and paragraph boundaries so that no
    chunk exceeds *max_chars* (unless a single sentence already does — we never
    break mid-sentence).

    Strategy:
    1. Hard-split on paragraph gaps (blank lines).
    2. Within each paragraph, split on sentence-ending punctuation.
    3. Greedily pack consecutive sentences into chunks up to *max_chars*.
    """
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    for para in _PARA_BREAK.split(text):
        para = para.strip()
        if not para:
            continue
        parts = [s.strip() for s in _SENT_BREAK.split(para) if s.strip()]
        sentences.extend(parts or [para])

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        gap = 1 if current else 0
        if current and current_len + gap + len(sent) > max_chars:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += gap + len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks or [text]
