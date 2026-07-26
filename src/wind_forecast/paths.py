"""Repository path helpers.

Paths are resolved from this package location and do not depend on the current
working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the repository root containing the project assets."""
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (
            ((candidate / "README.md").is_file()
             and (candidate / "data").is_dir()
             and (candidate / "models").is_dir())
            or (
                (candidate / "pyproject.toml").is_file()
                and (candidate / "src").is_dir()
                and (candidate / "config").is_dir()
            )
        ):
            return candidate

    raise RuntimeError(
        "Could not locate the project root from the installed wind_forecast package."
    )


def data_dir() -> Path:
    """Return the project data directory."""
    return project_root() / "data"


def raw_data_dir() -> Path:
    """Return the raw data directory."""
    return data_dir() / "raw"


def v1_raw_data_dir() -> Path:
    """Return the current v1-compatible raw data directory."""
    return raw_data_dir()


def v2_raw_data_dir() -> Path:
    """Return the future v2 raw data directory."""
    return raw_data_dir() / "v2"


def v2_raw_production_dir() -> Path:
    """Return the future v2 raw production data directory."""
    return v2_raw_data_dir() / "production"


def v2_raw_weather_dir() -> Path:
    """Return the future v2 raw weather data directory."""
    return v2_raw_data_dir() / "weather"


def processed_data_dir() -> Path:
    """Return the processed data directory."""
    return data_dir() / "processed"


def v1_processed_data_dir() -> Path:
    """Return the current v1-compatible processed data directory."""
    return processed_data_dir()


def v2_processed_data_dir() -> Path:
    """Return the future v2 processed data directory."""
    return processed_data_dir() / "v2"


def v2_processed_daily_merged_dir() -> Path:
    """Return the future v2 daily merged processed data directory."""
    return v2_processed_data_dir() / "daily_merged"


def v2_processed_ml_features_dir() -> Path:
    """Return the future v2 ML feature data directory."""
    return v2_processed_data_dir() / "ml_features"


def manifests_dir() -> Path:
    """Return the dataset manifests directory."""
    return data_dir() / "manifests"


def historical_v2_manifest_path() -> Path:
    """Return the planned historical v2 dataset manifest path."""
    return manifests_dir() / "historical_v2.json"


def pilot_data_dir() -> Path:
    """Return the ignored pilot-output data directory."""
    return data_dir() / "pilot"


def models_dir() -> Path:
    """Return the saved model artifact directory."""
    return project_root() / "models"


def notebooks_dir() -> Path:
    """Return the notebooks directory."""
    return project_root() / "notebooks"


def scripts_dir() -> Path:
    """Return the scripts directory."""
    return project_root() / "scripts"
