"""Feature engineering helpers for wind-energy forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import (
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
    columns_to_english,
)


WIND_PRODUCTION_LAGS = [1, 2, 3, 7, 14]
WEATHER_LAGS = [1, 2, 3, 7]
ROLLING_WINDOWS = [3, 7, 14]


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the feature engineering steps used by the forecasting model.

    Expected input columns:
    Date, Wind_Production, Average_Wind_Speed, Average_Temperature, Average_Wind_Direction.
    """
    print("Applying feature engineering...")
    df_features = df.copy()
    df_features[DATE_COLUMN] = pd.to_datetime(df_features[DATE_COLUMN])
    df_features = df_features.sort_values(DATE_COLUMN).reset_index(drop=True)

    df_features["Month"] = df_features[DATE_COLUMN].dt.month
    df_features["Day_Of_Week"] = df_features[DATE_COLUMN].dt.dayofweek
    df_features["Day_Of_Year"] = df_features[DATE_COLUMN].dt.dayofyear
    df_features["ISO_Week"] = df_features[DATE_COLUMN].dt.isocalendar().week.astype(int)
    df_features["Quarter"] = df_features[DATE_COLUMN].dt.quarter
    df_features["Is_Weekend"] = df_features["Day_Of_Week"].isin([5, 6]).astype(int)

    df_features["Wind_Direction_Sin"] = np.sin(np.radians(df_features[AVG_WIND_DIRECTION_COLUMN]))
    df_features["Wind_Direction_Cos"] = np.cos(np.radians(df_features[AVG_WIND_DIRECTION_COLUMN]))
    df_features["Day_Of_Week_Sin"] = np.sin(2 * np.pi * df_features["Day_Of_Week"] / 7)
    df_features["Day_Of_Week_Cos"] = np.cos(2 * np.pi * df_features["Day_Of_Week"] / 7)
    df_features["Month_Sin"] = np.sin(2 * np.pi * df_features["Month"] / 12)
    df_features["Month_Cos"] = np.cos(2 * np.pi * df_features["Month"] / 12)
    df_features["Day_Of_Year_Sin"] = np.sin(2 * np.pi * df_features["Day_Of_Year"] / 366)
    df_features["Day_Of_Year_Cos"] = np.cos(2 * np.pi * df_features["Day_Of_Year"] / 366)

    for lag in WIND_PRODUCTION_LAGS:
        df_features[f"Wind_Production_Lag{lag}"] = df_features[TARGET_COLUMN].shift(lag)

    for lag in WEATHER_LAGS:
        df_features[f"Average_Wind_Speed_Lag{lag}"] = df_features[AVG_WIND_SPEED_COLUMN].shift(lag)
        df_features[f"Average_Temperature_Lag{lag}"] = df_features[AVG_TEMPERATURE_COLUMN].shift(lag)
        df_features[f"Wind_Direction_Sin_Lag{lag}"] = df_features["Wind_Direction_Sin"].shift(lag)
        df_features[f"Wind_Direction_Cos_Lag{lag}"] = df_features["Wind_Direction_Cos"].shift(lag)

    for window in ROLLING_WINDOWS:
        df_features[f"Wind_Production_Rolling_Mean_{window}"] = (
            df_features[TARGET_COLUMN].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df_features[f"Wind_Production_Rolling_Std_{window}"] = (
            df_features[TARGET_COLUMN].shift(1).rolling(window=window, min_periods=1).std()
        )
        df_features[f"Average_Wind_Speed_Rolling_Mean_{window}"] = (
            df_features[AVG_WIND_SPEED_COLUMN].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df_features[f"Average_Wind_Speed_Rolling_Std_{window}"] = (
            df_features[AVG_WIND_SPEED_COLUMN].shift(1).rolling(window=window, min_periods=1).std()
        )
        df_features[f"Average_Temperature_Rolling_Mean_{window}"] = (
            df_features[AVG_TEMPERATURE_COLUMN].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df_features[f"Average_Temperature_Rolling_Std_{window}"] = (
            df_features[AVG_TEMPERATURE_COLUMN].shift(1).rolling(window=window, min_periods=1).std()
        )

    print("Feature engineering completed.")
    return df_features


def handle_final_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Handle NaNs created by lag and rolling-window features."""
    print("Handling final NaNs...")
    df_filled = df.copy()
    lag_roll_columns = [col for col in df_filled.columns if "_Lag" in col or "_Rolling_" in col]

    for col in lag_roll_columns:
        df_filled[col] = df_filled[col].bfill()

    remaining_nans = df_filled.isnull().sum()
    if remaining_nans.sum() > 0:
        print("WARNING: NaNs remaining after backfill:")
        print(remaining_nans[remaining_nans > 0])
        print("Filling remaining NaNs with 0.")
        df_filled = df_filled.fillna(0)
    else:
        print("No NaNs remaining after backfill.")

    return df_filled


def align_output_to_historical_columns(df: pd.DataFrame, historical_file: Path) -> pd.DataFrame:
    """Use the historical feature order when available, converted to English names."""
    if not historical_file.exists():
        print("No historical file available for column reference. Using current column order.")
        return df

    df_historical = pd.read_csv(historical_file, nrows=1)
    expected_columns = columns_to_english(df_historical.columns.tolist())
    for col in df.columns:
        if col not in expected_columns:
            expected_columns.append(col)
    return df.reindex(columns=expected_columns)
