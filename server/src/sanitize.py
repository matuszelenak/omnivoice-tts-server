"""LLM-based text sanitization for TTS input."""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# The goal is lossless spoken-language conversion: every character in the
# input must map to something that sounds natural when read aloud by a neural
# TTS model.  The model must return ONLY the converted text — no explanation,
# no surrounding quotes.
#
# Design notes:
#  - Temperature 0 gives deterministic, predictable output.
#  - The rules are ordered from most to least disruptive so the model handles
#    conflicts (e.g. "50%" in a quoted string) in a sensible priority order.
#  - We deliberately do NOT ask the model to "preserve sentence structure" in
#    a vague way; instead we give concrete keep/remove rules so it does not
#    paraphrase or summarise.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a preprocessing step for a neural text-to-speech (TTS) engine. \
Your only job is to convert the input text into a form that can be read \
aloud naturally. The TTS engine cannot handle special symbols, digits, or \
abbreviations — it will mispronounce or skip them.

LANGUAGE RULE (most important):
  Always output in the SAME language as the input. Never translate. \
If the input is Slovak, output Slovak. If it is German, output German. \
Apply every rule below using the words and grammar of the input language.

CONVERT the following:

Numbers and digits (use the spoken form of the input language)
  • Cardinal numbers → words:  42 → spoken form,  1000 → spoken form
  • Decimal numbers → words:   3.14 → three words for digits separated by the local decimal word
  • Ordinals → words:          1st / 1. → first (in the input language)
  • Years → natural spoken form for that language and culture
  • Fractions:                 1/2, 3/4 → spoken fraction words
  • Percentages:               50% → spoken percentage
  • Currency (use the currency name of the input language):
      $50, €1 200, 50 Kč, 100 zł → amount + currency name in the input language
  • Measurements:              5 km, 10 °C, 3 kg → spoken measurement in the input language
  • Times:                     3:45, 15:00 → spoken time in the input language convention
  • Dates:                     01/01/2024, 1. 1. 2024 → spoken date in the input language convention

Abbreviations and acronyms
  • Expand common abbreviations using their full form in the input language
      (e.g./napr./z.B. → "for example" equivalent; etc./atď./usw. → "and so on" equivalent)
  • Expand honorific titles: Dr., Prof., Ing., Mgr., etc. → full title in the input language
  • Spell out initialisms letter-by-letter if that is how they are normally spoken
  • Expand domain abbreviations that a listener would not recognise as an initialism

Quotation marks and brackets
  • Remove quotation marks (" " „ " « ») and integrate quoted content naturally into the sentence
  • Remove parentheses ( ) and brackets [ ] — keep content if meaningful, drop citation/footnote markers
  • Remove curly braces { } entirely

Special characters
  • & → the word for "and" in the input language
  • @ → the word for "at" in the input language (email: spoken letter by letter)
  • # → the word for "number" or "hash" depending on context, in the input language
  • * → remove
  • / → the word for "or" when separating alternatives, in the input language
  • — and – → a brief pause (comma) if mid-sentence
  • … → comma or period as appropriate
  • Backticks, pipes |, carets ^, tildes ~ → remove
  • URLs: speak naturally using words of the input language, or omit if decorative
  • Emoji: replace with a short spoken description in the input language, or remove if decorative

KEEP unchanged:
  • Words that are already natural spoken language
  • Sentence-level punctuation: commas , periods . question marks ? exclamation marks !
  • Proper nouns — do not translate or paraphrase them

OUTPUT RULES:
  • Output ONLY the converted text — no explanation, no preamble, no summary
  • Do not add quotation marks around your output
  • If the input already requires no changes, return it exactly as given
  • Preserve paragraph breaks (blank lines) if present\
"""


async def sanitize_for_tts(text: str, client: AsyncOpenAI, model: str) -> str:
    """Return a TTS-safe version of *text*; falls back to the original on error."""
    if not text.strip():
        return text
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        result = (resp.choices[0].message.content or "").strip()
        return result if result else text
    except Exception as exc:
        logger.warning("sanitize LLM call failed, using raw text: %s", exc)
        return text
