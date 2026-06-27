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
            (candidate / "README.md").is_file()
            and (candidate / "data").is_dir()
            and (candidate / "models").is_dir()
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


def processed_data_dir() -> Path:
    """Return the processed data directory."""
    return data_dir() / "processed"


def models_dir() -> Path:
    """Return the saved model artifact directory."""
    return project_root() / "models"


def notebooks_dir() -> Path:
    """Return the notebooks directory."""
    return project_root() / "notebooks"


def scripts_dir() -> Path:
    """Return the scripts directory."""
    return project_root() / "scripts"
