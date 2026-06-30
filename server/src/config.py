"""Runtime configuration, sourced from environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True, frozen=True)

    # Optional path to the Supertonic model directory.
    # If empty, defaults to ~/.cache/supertonic3.
    supertonic_model_dir: str = ""

    # Directory where user-imported voice style JSONs are stored.
    # If empty, defaults to ~/.cache/supertonic3/custom_styles.
    custom_styles_dir: str = ""

    # Directory of a built frontend to serve as static files. Empty (the dev
    # default) disables static serving so the Vite dev server handles the UI.
    # The production image sets this to the baked-in build (e.g. /app/static).
    static_dir: str = ""


settings = Settings()
