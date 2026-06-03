"""Runtime configuration, sourced from environment variables."""
from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True, frozen=True)

    # Number of model replicas. Defaults to one per visible GPU (resolved at
    # startup when this is 0).
    num_workers: int = 0

    # Explicit device list, e.g. "cuda:0,cuda:1". When unset, devices are
    # assigned round-robin across the visible GPUs.
    devices: list[str] | None = None

    # Directory of a built frontend to serve as static files. Empty (the dev
    # default) disables static serving so the Vite dev server handles the UI.
    # The production image sets this to the baked-in build (e.g. /app/static).
    static_dir: str = ""
    voice_samples_dir: Path = ""

    @field_validator("devices", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> list[str] | None:
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            return parts or None
        return v  # type: ignore[return-value]


settings = Settings()
