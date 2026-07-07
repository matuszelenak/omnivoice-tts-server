"""Runtime configuration, sourced from environment variables."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True, frozen=True)

    # CUDA device the model is loaded on.
    device_map: str = "cuda:0"

    # Directory of a built frontend to serve as static files. Empty (the dev
    # default) disables static serving so the Vite dev server handles the UI.
    # The production image sets this to the baked-in build (e.g. /app/static).
    static_dir: str = ""
    voice_samples_dir: Path = ""

    # LLM-based text sanitization (OpenAI-compatible endpoint).
    # Leave sanitize_llm_base_url empty to disable sanitization entirely.
    sanitize_llm_base_url: str = ""
    sanitize_llm_api_key: str = "none"
    sanitize_llm_model: str = "gpt-4o-mini"


settings = Settings()
