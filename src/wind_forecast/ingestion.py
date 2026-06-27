"""WeatherAPI ingestion helpers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .schemas import (
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    DATE_COLUMN,
)


WEATHER_API_HISTORY_ENDPOINT = "https://api.weatherapi.com/v1/history.json"
WEATHER_API_TIMEOUT_SECONDS = 10


def build_dates_to_fetch(num_days: int, end_date: str | None = None) -> list[str]:
    """Build the list of past dates to request from WeatherAPI."""
    if num_days <= 0:
        raise ValueError("WEATHER_API_DAYS must be greater than zero.")

    end = pd.to_datetime(end_date).to_pydatetime() if end_date else datetime.today()
    return [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, num_days + 1)]


def build_history_request_params(api_key: str, location: str, date_str: str) -> dict[str, str]:
    """Build WeatherAPI history request parameters."""
    return {"key": api_key, "q": location, "dt": date_str}


def parse_history_forecast_day(date_str: str, data: dict[str, Any]) -> dict[str, Any]:
    """Parse one WeatherAPI history response into the project schema."""
    forecast_day = data["forecast"]["forecastday"][0]
    day_data = forecast_day["day"]
    wind_direction_noon = forecast_day["hour"][12]["wind_degree"]

    return {
        DATE_COLUMN: pd.to_datetime(date_str),
        AVG_TEMPERATURE_COLUMN: day_data["avgtemp_c"],
        AVG_WIND_SPEED_COLUMN: day_data["maxwind_kph"] / 3.6,
        AVG_WIND_DIRECTION_COLUMN: wind_direction_noon,
    }


def fetch_weather_api_data(
    api_key: str | None,
    location: str,
    num_days: int,
    end_date: str | None = None,
    *,
    request_get: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Fetch historical weather data from WeatherAPI."""
    if not api_key:
        raise RuntimeError("Set WEATHER_API_KEY in a local .env file before calling the API.")

    print(f"Fetching weather data for the last {num_days} days...")
    dates_to_fetch = build_dates_to_fetch(num_days, end_date)

    records = []
    if request_get is None:
        request_get = requests.get

    for date_str in dates_to_fetch:
        params = build_history_request_params(api_key, location, date_str)
        try:
            response = request_get(
                WEATHER_API_HISTORY_ENDPOINT,
                params=params,
                timeout=WEATHER_API_TIMEOUT_SECONDS,
            )
            if not response.ok:
                print(f"Error fetching data for {date_str}: HTTP {response.status_code}")
                continue

            data = response.json()
            records.append(parse_history_forecast_day(date_str, data))
        except requests.exceptions.RequestException as exc:
            print(f"Network error fetching data for {date_str}: {exc.__class__.__name__}")
        except (KeyError, ValueError, IndexError) as exc:
            print(f"Error parsing data for {date_str}: {exc.__class__.__name__}")

    if not records:
        print("No data fetched from the API. Exiting.")
        return pd.DataFrame()

    df_api = pd.DataFrame(records).sort_values(DATE_COLUMN).reset_index(drop=True)
    print(f"Successfully fetched {len(df_api)} records from the API.")
    return df_api
