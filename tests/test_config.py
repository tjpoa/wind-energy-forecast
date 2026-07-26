from pathlib import Path

import pytest

from wind_forecast.config import (
    CORS_ALLOWED_ORIGINS_ENV,
    DEFAULT_CORS_ALLOWED_ORIGINS,
    MONITORING_STORE_ROOT_ENV,
    load_cors_config,
    load_monitoring_store_config,
    load_weather_api_config,
)
from wind_forecast.paths import project_root


WEATHER_ENV_VARS = [
    "WEATHER_API_KEY",
    "WEATHER_API_LOCATION",
    "WEATHER_API_DAYS",
    "WEATHER_API_END_DATE",
]


def test_load_weather_api_config_uses_defaults_without_required_key(monkeypatch):
    for name in WEATHER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    config = load_weather_api_config(require_api_key=False)

    assert config.api_key is None
    assert config.location == "41.8345,-7.7889"
    assert config.days == 44
    assert config.end_date is None


def test_load_weather_api_config_reads_environment(monkeypatch):
    monkeypatch.setenv("WEATHER_API_KEY", "test-api-key")
    monkeypatch.setenv("WEATHER_API_LOCATION", "40.0,-8.0")
    monkeypatch.setenv("WEATHER_API_DAYS", "7")
    monkeypatch.setenv("WEATHER_API_END_DATE", "2026-01-31")

    config = load_weather_api_config()

    assert config.api_key == "test-api-key"
    assert config.location == "40.0,-8.0"
    assert config.days == 7
    assert config.end_date == "2026-01-31"


def test_load_weather_api_config_requires_key_by_default(monkeypatch):
    monkeypatch.delenv("WEATHER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="WEATHER_API_KEY"):
        load_weather_api_config()


@pytest.mark.parametrize("days", ["0", "-1", "not-an-int"])
def test_load_weather_api_config_rejects_invalid_days(monkeypatch, days):
    monkeypatch.setenv("WEATHER_API_KEY", "test-api-key")
    monkeypatch.setenv("WEATHER_API_DAYS", days)

    with pytest.raises(ValueError, match="WEATHER_API_DAYS"):
        load_weather_api_config()


def test_load_cors_config_uses_vite_default_when_variable_is_absent(monkeypatch):
    monkeypatch.delenv(CORS_ALLOWED_ORIGINS_ENV, raising=False)

    config = load_cors_config()

    assert config.allowed_origins == DEFAULT_CORS_ALLOWED_ORIGINS


def test_load_cors_config_reads_comma_separated_origins(monkeypatch):
    monkeypatch.setenv(
        CORS_ALLOWED_ORIGINS_ENV,
        "http://localhost:5173, https://dashboard.example.test",
    )

    config = load_cors_config()

    assert config.allowed_origins == (
        "http://localhost:5173",
        "https://dashboard.example.test",
    )


@pytest.mark.parametrize(
    "origins",
    [
        "",
        "http://localhost:5173,",
        "*",
        "ftp://localhost:5173",
        "http://user:password@localhost:5173",
        "http://localhost:not-a-port",
        "http://localhost:5173/",
        "http://localhost:5173?preview=true",
        "http://localhost:5173#section",
    ],
)
def test_load_cors_config_rejects_invalid_origins(monkeypatch, origins):
    monkeypatch.setenv(CORS_ALLOWED_ORIGINS_ENV, origins)

    with pytest.raises(ValueError, match=CORS_ALLOWED_ORIGINS_ENV):
        load_cors_config()


def test_monitoring_store_relative_path_is_resolved_from_project_root(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(MONITORING_STORE_ROOT_ENV, "custom/monitoring")

    config = load_monitoring_store_config()

    assert config.store_root == (project_root() / "custom/monitoring").resolve()
