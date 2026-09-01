from datetime import datetime
from pathlib import Path

import pandas as pd
from wind_forecast.config import load_weather_api_config
from wind_forecast.features import (
    align_output_to_historical_columns,
    apply_feature_engineering,
    handle_final_nans,
)
from wind_forecast.ingestion import fetch_weather_api_data
from wind_forecast.manifest_validation import validate_v1_source_contract
from wind_forecast.schemas import (
    DATE_COLUMN,
    RAW_DATE_TIME_COLUMN,
    RAW_WIND_PRODUCTION_COLUMN,
    RAW_PRODUCTION_FILENAME,
    TARGET_COLUMN,
)


# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = BASE_DATA_PATH / "raw"
PROCESSED_DATA_PATH = BASE_DATA_PATH / "processed"
HISTORICAL_PROCESSED_FILE = PROCESSED_DATA_PATH / "agg_data_ml.csv"
PRODUCTION_RAW_FILE = RAW_DATA_PATH / RAW_PRODUCTION_FILENAME


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
    validate_v1_source_contract(required_paths=[PRODUCTION_RAW_FILE])
    api_config = load_weather_api_config(load_dotenv_file=True)
    df_api_weather = fetch_weather_api_data(
        api_config.api_key,
        api_config.location,
        api_config.days,
        api_config.end_date,
    )
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
