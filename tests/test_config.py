import pytest

from wind_forecast.config import load_weather_api_config


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
