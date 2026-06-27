import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from schema import (
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    DATE_COLUMN,
    RAW_DATE_TIME_COLUMN,
    RAW_WIND_PRODUCTION_COLUMN,
    RAW_PRODUCTION_FILENAME,
    TARGET_COLUMN,
)
from wind_forecast.features import (
    align_output_to_historical_columns,
    apply_feature_engineering,
    handle_final_nans,
)


# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("WEATHER_API_KEY")
API_LOCATION = os.getenv("WEATHER_API_LOCATION", "41.8345,-7.7889")
DAYS_TO_FETCH = int(os.getenv("WEATHER_API_DAYS", "44"))
WEATHER_API_END_DATE = os.getenv("WEATHER_API_END_DATE")

BASE_DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = BASE_DATA_PATH / "raw"
PROCESSED_DATA_PATH = BASE_DATA_PATH / "processed"
HISTORICAL_PROCESSED_FILE = PROCESSED_DATA_PATH / "agg_data_ml.csv"
PRODUCTION_RAW_FILE = RAW_DATA_PATH / RAW_PRODUCTION_FILENAME


def build_dates_to_fetch(num_days: int, end_date: str | None = None) -> list[str]:
    """Build the list of past dates to request from WeatherAPI."""
    if num_days <= 0:
        raise ValueError("WEATHER_API_DAYS must be greater than zero.")

    end = pd.to_datetime(end_date).to_pydatetime() if end_date else datetime.today()
    return [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, num_days + 1)]


def fetch_weather_api_data(api_key: str | None, location: str, num_days: int) -> pd.DataFrame:
    """Fetch historical weather data from WeatherAPI."""
    if not api_key:
        raise RuntimeError("Set WEATHER_API_KEY in a local .env file before calling the API.")

    print(f"Fetching weather data for the last {num_days} days...")
    dates_to_fetch = build_dates_to_fetch(num_days, WEATHER_API_END_DATE)

    records = []
    endpoint = "https://api.weatherapi.com/v1/history.json"

    for date_str in dates_to_fetch:
        params = {"key": api_key, "q": location, "dt": date_str}
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            if not response.ok:
                print(f"Error fetching data for {date_str}: HTTP {response.status_code}")
                continue

            data = response.json()
            forecast_day = data["forecast"]["forecastday"][0]
            day_data = forecast_day["day"]
            wind_direction_noon = forecast_day["hour"][12]["wind_degree"]

            records.append(
                {
                    DATE_COLUMN: pd.to_datetime(date_str),
                    AVG_TEMPERATURE_COLUMN: day_data["avgtemp_c"],
                    AVG_WIND_SPEED_COLUMN: day_data["maxwind_kph"] / 3.6,
                    AVG_WIND_DIRECTION_COLUMN: wind_direction_noon,
                }
            )
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


def load_and_process_production_data(filepath: Path) -> pd.DataFrame:
    """Load source production data and aggregate wind production by day."""
    print(f"Loading production data from: {filepath}")
    df_raw = pd.read_csv(filepath, na_values=-990, sep=";", skiprows=2)
    df_raw.columns = df_raw.columns.str.strip()

    required_columns = {RAW_DATE_TIME_COLUMN, RAW_WIND_PRODUCTION_COLUMN}
    missing_columns = required_columns.difference(df_raw.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in production file: {sorted(missing_columns)}")

    df_production = df_raw[[RAW_DATE_TIME_COLUMN, RAW_WIND_PRODUCTION_COLUMN]].copy()
    df_production[RAW_DATE_TIME_COLUMN] = pd.to_datetime(
        df_production[RAW_DATE_TIME_COLUMN], errors="coerce"
    )
    df_production[RAW_WIND_PRODUCTION_COLUMN] = pd.to_numeric(
        df_production[RAW_WIND_PRODUCTION_COLUMN], errors="coerce"
    )
    df_production = df_production.dropna(subset=[RAW_DATE_TIME_COLUMN, RAW_WIND_PRODUCTION_COLUMN])

    df_daily = (
        df_production.set_index(RAW_DATE_TIME_COLUMN)
        .resample("D")[RAW_WIND_PRODUCTION_COLUMN]
        .sum()
        .reset_index()
        .rename(columns={RAW_DATE_TIME_COLUMN: DATE_COLUMN, RAW_WIND_PRODUCTION_COLUMN: TARGET_COLUMN})
    )

    print(f"Processed production data. Shape: {df_daily.shape}")
    return df_daily[[DATE_COLUMN, TARGET_COLUMN]]


def main() -> None:
    df_api_weather = fetch_weather_api_data(API_KEY, API_LOCATION, DAYS_TO_FETCH)
    if df_api_weather.empty:
        return

    if not PRODUCTION_RAW_FILE.exists():
        raise FileNotFoundError(f"Raw production file not found: {PRODUCTION_RAW_FILE}")

    df_recent_production = load_and_process_production_data(PRODUCTION_RAW_FILE)

    print("Merging API weather data with recent production data...")
    df_new_data_raw = pd.merge(df_api_weather, df_recent_production, on=DATE_COLUMN, how="inner")
    if df_new_data_raw.empty:
        print("No matching dates found between API weather data and production data. Cannot proceed.")
        return

    df_new_data_raw = df_new_data_raw.sort_values(DATE_COLUMN).reset_index(drop=True)
    print(f"Merged new data. Shape: {df_new_data_raw.shape}")

    df_features = apply_feature_engineering(df_new_data_raw)
    df_processed = handle_final_nans(df_features)
    df_final_new_data = df_processed.dropna(subset=[TARGET_COLUMN]).copy()

    print("\n--- Final processed new data: first 5 rows ---")
    print(df_final_new_data.head())

    df_final_new_data = align_output_to_historical_columns(df_final_new_data, HISTORICAL_PROCESSED_FILE)

    if df_final_new_data.isnull().sum().sum() > 0:
        print("\nWARNING: NaNs detected in the final new data after column reordering:")
        print(df_final_new_data.isnull().sum()[df_final_new_data.isnull().sum() > 0])

    output_filename = PROCESSED_DATA_PATH / f"api_data_featured_{datetime.now().strftime('%Y%m%d')}.csv"
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    df_final_new_data.to_csv(output_filename, index=False)
    print(f"\nSuccessfully processed and saved new data to: {output_filename}")


if __name__ == "__main__":
    main()
