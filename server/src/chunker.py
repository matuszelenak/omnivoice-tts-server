from blingfire import text_to_sentences


def split_text(text: str) -> list[str]:
    """Split *text* into individual sentences using BlingFire."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in text_to_sentences(text).splitlines() if s.strip()] or [text]
