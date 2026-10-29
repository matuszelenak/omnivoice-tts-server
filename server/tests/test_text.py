"""Text segmentation and duration estimates."""
from __future__ import annotations

from app.text import SentenceStreamer, estimate_audio_seconds, normalize_space, split_text


def test_split_text_keeps_sentences_whole():
    text = "Hello there. How are you? I am fine, thanks!"
    assert split_text(text) == ["Hello there.", "How are you?", "I am fine, thanks!"]


def test_split_text_does_not_split_abbreviations():
    text = "Dr. Smith arrived at 3 p.m. and left immediately. He was tired."
    chunks = split_text(text)
    assert chunks[0].startswith("Dr. Smith arrived at 3 p.m.")
    assert len(chunks) == 2


def test_split_text_cuts_long_sentences_at_clauses():
    clause = "this is a fairly long clause that keeps going for a while"
    text = ", ".join([clause] * 6) + "."
    chunks = split_text(text, max_chunk_chars=120)
    assert all(len(c) <= 120 for c in chunks)
    assert " ".join(chunks).replace(".", "").split() == text.replace(".", "").split()


def test_split_text_hard_cuts_words_that_are_too_long():
    text = "word " * 200
    chunks = split_text(text, max_chunk_chars=80)
    assert all(len(c) <= 80 for c in chunks)


def test_split_text_emits_one_chunk_per_sentence():
    # One chunk per sentence keeps time-to-first-audio low when streaming.
    assert split_text("Yes, that is fine. No, thank you very much.") == [
        "Yes, that is fine.",
        "No, thank you very much.",
    ]


def test_split_text_empty():
    assert split_text("   ") == []


def test_normalize_space():
    assert normalize_space("  a\tb\r\nc  ") == "a b c"


def test_streamer_yields_first_clause_early():
    streamer = SentenceStreamer(max_chunk_chars=200, min_chunk_chars=5, first_chunk_chars=10)
    out = streamer.feed("Hello there, ")
    assert out == ["Hello there,"]
    assert streamer.feed("how are you today?") == ["how are you today?"]
    assert streamer.feed(" I am fine.") == ["I am fine."]
    assert streamer.flush() == []


def test_streamer_waits_for_enough_text():
    streamer = SentenceStreamer(first_chunk_chars=5, min_chunk_chars=8)
    assert streamer.feed("One, ") == []          # too short to be worth a model call
    assert streamer.feed("two, ") == ["One, two,"]  # clause break, long enough now
    assert streamer.feed("three. ") == []        # sentence break but still too short
    assert streamer.flush() == ["three."]


def test_streamer_flush_returns_remainder():
    streamer = SentenceStreamer()
    assert streamer.feed("Hello there") == []
    assert streamer.flush() == ["Hello there"]
    assert streamer.flush() == []


def test_streamer_respects_max_chunk_chars():
    streamer = SentenceStreamer(max_chunk_chars=40, first_chunk_chars=40)
    out = streamer.feed("word " * 40)
    assert out and all(len(c) <= 40 for c in out)


def test_estimate_audio_seconds():
    assert 1.0 < estimate_audio_seconds("This is a normal sentence of about fifty chars.") < 5.0
    assert estimate_audio_seconds("short", speed=2.0) < estimate_audio_seconds("short")
