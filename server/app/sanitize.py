"""LLM-based text sanitisation before synthesis.

OmniVoice mispronounces digits, abbreviations and symbols ("3 kg", "napr.",
"$50"), so an optional OpenAI-compatible LLM rewrites each chunk into spoken
language first. Two properties matter for a server:

* **per chunk, not per request** — streaming a long text must not wait for the
  whole document to be sanitised before the first sentence can be spoken;
* **never fatal** — a timeout, a 500 or a garbled answer falls back to the raw
  text and the request still produces audio.

Disabled unless ``SANITIZE_LLM_BASE_URL`` is set. The prompt is
language-preserving: the output stays in the input language.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from . import metrics
from .config import settings

log = logging.getLogger(__name__)

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

class Sanitizer:
    """Async façade over the sanitisation LLM (lazily created client)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else settings.sanitize_llm_base_url).strip()
        self.api_key = api_key if api_key is not None else settings.sanitize_llm_api_key
        self.model = model or settings.sanitize_llm_model
        self.timeout_s = timeout_s if timeout_s is not None else settings.sanitize_llm_timeout_s
        self._client: Any = None
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    async def _get_client(self):
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    from openai import AsyncOpenAI

                    self._client = AsyncOpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key or "none",
                        timeout=self.timeout_s,
                        max_retries=0,
                    )
        return self._client

    async def sanitize(self, text: str, *, language: str | None = None) -> str:
        """Return a speakable version of *text*; the raw text on any failure."""
        if not self.enabled or not text.strip():
            return text
        started = time.perf_counter()
        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            result = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            metrics.sanitize_total.labels(outcome="error").inc()
            log.warning("sanitize call failed, using the raw text: %s", exc)
            return text
        finally:
            metrics.sanitize_seconds.observe(time.perf_counter() - started)
        if not result:
            metrics.sanitize_total.labels(outcome="empty").inc()
            return text
        metrics.sanitize_total.labels(outcome="ok").inc()
        log.debug("sanitized %d -> %d chars (%s)", len(text), len(result), language or "?")
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url": self.base_url or None,
            "model": self.model if self.enabled else None,
        }


sanitizer = Sanitizer()
