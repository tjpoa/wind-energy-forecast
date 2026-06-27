"""Runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .paths import project_root


@dataclass(frozen=True)
class WeatherAPIConfig:
    """WeatherAPI settings loaded from environment variables."""

    api_key: str | None
    location: str
    days: int
    end_date: str | None


def load_weather_api_config(
    *,
    load_dotenv_file: bool = False,
    env_file: Path | None = None,
    require_api_key: bool = True,
) -> WeatherAPIConfig:
    """Load WeatherAPI configuration from environment variables.

    Set ``load_dotenv_file=True`` to explicitly load the local ``.env`` file.
    Importing this module never loads environment files or performs network I/O.
    """
    if load_dotenv_file:
        from dotenv import load_dotenv

        load_dotenv(env_file or (project_root() / ".env"))

    api_key = os.getenv("WEATHER_API_KEY") or None
    if require_api_key and not api_key:
        raise RuntimeError("Set WEATHER_API_KEY in a local .env file before calling the API.")

    days_raw = os.getenv("WEATHER_API_DAYS", "44")
    try:
        days = int(days_raw)
    except ValueError as exc:
        raise ValueError("WEATHER_API_DAYS must be an integer.") from exc
    if days <= 0:
        raise ValueError("WEATHER_API_DAYS must be greater than zero.")

    return WeatherAPIConfig(
        api_key=api_key,
        location=os.getenv("WEATHER_API_LOCATION", "41.8345,-7.7889"),
        days=days,
        end_date=os.getenv("WEATHER_API_END_DATE") or None,
    )
