"""Sanitisation: disabled path, success, and the never-fatal fallback."""
from __future__ import annotations

import asyncio

import pytest

from app.sanitize import Sanitizer


class StubCompletions:
    def __init__(self, reply: str | None = None, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = type("M", (), {"content": self.reply})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class StubClient:
    def __init__(self, completions: StubCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def make(reply=None, error=None, base_url="http://llm/v1"):
    sanitizer = Sanitizer(base_url=base_url, api_key="k", model="m", timeout_s=1.0)
    completions = StubCompletions(reply, error)
    sanitizer._client = StubClient(completions)  # bypass AsyncOpenAI construction
    return sanitizer, completions


def test_disabled_when_no_base_url():
    sanitizer = Sanitizer(base_url="")
    assert sanitizer.enabled is False
    assert asyncio.run(sanitizer.sanitize("50 kg")) == "50 kg"
    assert sanitizer.stats()["enabled"] is False


def test_rewrites_text_and_uses_temperature_zero():
    sanitizer, completions = make(reply="fifty kilograms")
    assert asyncio.run(sanitizer.sanitize("50 kg")) == "fifty kilograms"
    assert completions.calls[0]["temperature"] == 0
    assert completions.calls[0]["model"] == "m"


def test_llm_failure_falls_back_to_raw_text():
    sanitizer, _ = make(error=RuntimeError("boom"))
    assert asyncio.run(sanitizer.sanitize("3:45 pm")) == "3:45 pm"


def test_empty_answer_falls_back_to_raw_text():
    sanitizer, _ = make(reply="   ")
    assert asyncio.run(sanitizer.sanitize("€1 200")) == "€1 200"


def test_blank_input_is_not_sent_to_the_model():
    sanitizer, completions = make(reply="x")
    assert asyncio.run(sanitizer.sanitize("   ")) == "   "
    assert completions.calls == []


def test_system_prompt_forbids_translation():
    sanitizer, completions = make(reply="ok")
    asyncio.run(sanitizer.sanitize("100 Kč", language="sk"))
    system = completions.calls[0]["messages"][0]["content"]
    assert "SAME language" in system and "Never translate" in system


def test_stats_never_leak_the_api_key():
    sanitizer, _ = make()
    stats = sanitizer.stats()
    assert "key" not in str(stats).lower()
    assert stats["url"] == "http://llm/v1"


@pytest.mark.parametrize("text", ["", "   "])
def test_whitespace_only(text):
    sanitizer, _ = make(reply="x")
    assert asyncio.run(sanitizer.sanitize(text)) == text
