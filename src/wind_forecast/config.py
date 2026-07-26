"""Runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .paths import project_root


PERFORMANCE_ARTIFACT_DIR_ENV = "WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR"
MONITORING_STORE_ROOT_ENV = "WIND_FORECAST_MONITORING_STORE_ROOT"
CORS_ALLOWED_ORIGINS_ENV = "WIND_FORECAST_CORS_ALLOW_ORIGINS"
DEFAULT_CORS_ALLOWED_ORIGINS = ("http://localhost:5173",)


@dataclass(frozen=True)
class WeatherAPIConfig:
    """WeatherAPI settings loaded from environment variables."""

    api_key: str | None
    location: str
    days: int
    end_date: str | None


@dataclass(frozen=True)
class PerformanceArtifactsConfig:
    """Location of one explicitly selected performance-artifact set."""

    artifact_dir: Path | None


@dataclass(frozen=True)
class MonitoringStoreConfig:
    """Location of the immutable Phase 9 monitoring store."""

    store_root: Path


@dataclass(frozen=True)
class CORSConfig:
    """Explicit browser origins allowed to call the FastAPI application."""

    allowed_origins: tuple[str, ...]


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


def load_performance_artifacts_config() -> PerformanceArtifactsConfig:
    """Load the explicitly configured performance-artifact directory.

    Relative paths are resolved from the repository root. The directory is not
    required to exist at configuration-load time; the performance domain
    service owns artifact-readiness validation.
    """
    raw_path = os.getenv(PERFORMANCE_ARTIFACT_DIR_ENV)
    if raw_path is None or not raw_path.strip():
        return PerformanceArtifactsConfig(artifact_dir=None)

    configured_path = Path(raw_path.strip())
    if not configured_path.is_absolute():
        configured_path = project_root() / configured_path
    return PerformanceArtifactsConfig(artifact_dir=configured_path.resolve())


def load_monitoring_store_config() -> MonitoringStoreConfig:
    """Load the monitoring store, resolving relative paths from the project root."""
    raw_path = os.getenv(MONITORING_STORE_ROOT_ENV)
    configured_path = (
        Path(raw_path.strip())
        if raw_path is not None and raw_path.strip()
        else Path("data/processed/v2/monitoring")
    )
    if not configured_path.is_absolute():
        configured_path = project_root() / configured_path
    return MonitoringStoreConfig(store_root=configured_path.resolve())


def load_cors_config() -> CORSConfig:
    """Load validated, comma-separated CORS origins from the environment.

    When the variable is absent, local Vite development is the only allowed
    browser origin. A variable that is present but invalid fails fast instead
    of silently broadening or changing the configured access boundary.
    """
    raw_origins = os.getenv(CORS_ALLOWED_ORIGINS_ENV)
    if raw_origins is None:
        return CORSConfig(allowed_origins=DEFAULT_CORS_ALLOWED_ORIGINS)

    allowed_origins = tuple(origin.strip() for origin in raw_origins.split(","))
    if not allowed_origins or any(not origin for origin in allowed_origins):
        raise ValueError(
            f"{CORS_ALLOWED_ORIGINS_ENV} must contain one or more comma-separated "
            "origins."
        )

    for origin in allowed_origins:
        _validate_cors_origin(origin)

    return CORSConfig(allowed_origins=allowed_origins)


def _validate_cors_origin(origin: str) -> None:
    """Reject values that cannot be used as an exact browser origin."""
    try:
        parsed = urlparse(origin)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(_cors_origin_error()) from exc

    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or "*" in parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in origin)
    ):
        raise ValueError(_cors_origin_error())


def _cors_origin_error() -> str:
    """Return the shared validation message for configured CORS origins."""
    return (
        f"{CORS_ALLOWED_ORIGINS_ENV} entries must be exact http(s) origins "
        "without wildcards, credentials, paths, query strings, or fragments."
    )
