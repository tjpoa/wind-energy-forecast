from pathlib import Path

import numpy as np
import pandas as pd

from wind_forecast.features import (
    align_output_to_historical_columns,
    apply_feature_engineering,
    handle_final_nans,
)
from wind_forecast.schemas import (
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    DATE_COLUMN,
    LEGACY_DATE_COLUMN,
    LEGACY_WIND_SPEED_COLUMN,
    TARGET_COLUMN,
)


def _base_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE_COLUMN: ["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-04"],
            TARGET_COLUMN: [30.0, 10.0, 20.0, 40.0],
            AVG_WIND_SPEED_COLUMN: [3.0, 1.0, 2.0, 4.0],
            AVG_TEMPERATURE_COLUMN: [13.0, 11.0, 12.0, 14.0],
            AVG_WIND_DIRECTION_COLUMN: [180.0, 0.0, 90.0, 270.0],
        }
    )


def test_apply_feature_engineering_sorts_and_builds_deterministic_features():
    source = _base_daily_frame()
    original = source.copy(deep=True)

    features = apply_feature_engineering(source)

    pd.testing.assert_frame_equal(source, original)
    assert features[DATE_COLUMN].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
    ]
    assert features[TARGET_COLUMN].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert features.loc[2, "Wind_Production_Lag1"] == 20.0
    assert features.loc[2, "Wind_Production_Lag2"] == 10.0
    assert features.loc[3, "Wind_Production_Rolling_Mean_3"] == 20.0
    assert np.isclose(features.loc[3, "Wind_Production_Rolling_Std_3"], 10.0)
    assert np.isclose(features.loc[1, "Wind_Direction_Sin"], 1.0)
    assert np.isclose(features.loc[1, "Wind_Direction_Cos"], 0.0)
    assert features.loc[2, "Is_Weekend"] == int(pd.Timestamp("2026-01-03").dayofweek in [5, 6])


def test_handle_final_nans_backfills_lag_and_rolling_columns_then_fills_remaining():
    source = pd.DataFrame(
        {
            "A_Lag1": [np.nan, 2.0, np.nan],
            "B_Rolling_Mean_3": [np.nan, 4.0, 5.0],
            "Other": [1.0, np.nan, 3.0],
        }
    )

    filled = handle_final_nans(source)

    pd.testing.assert_frame_equal(
        filled,
        pd.DataFrame(
            {
                "A_Lag1": [2.0, 2.0, 0.0],
                "B_Rolling_Mean_3": [4.0, 4.0, 5.0],
                "Other": [1.0, 0.0, 3.0],
            }
        ),
    )
    assert source.isna().sum().sum() == 4


def test_align_output_to_historical_columns_uses_legacy_order_and_appends_new_columns(
    tmp_path: Path,
):
    historical_file = tmp_path / "historical.csv"
    pd.DataFrame(columns=[LEGACY_DATE_COLUMN, LEGACY_WIND_SPEED_COLUMN]).to_csv(
        historical_file,
        index=False,
    )
    frame = pd.DataFrame(
        {
            DATE_COLUMN: ["2026-01-01"],
            AVG_WIND_SPEED_COLUMN: [4.2],
            "New_Feature": [99.0],
        }
    )

    aligned = align_output_to_historical_columns(frame, historical_file)

    assert aligned.columns.tolist() == [DATE_COLUMN, AVG_WIND_SPEED_COLUMN, "New_Feature"]
    assert aligned.loc[0, "New_Feature"] == 99.0


def test_align_output_to_historical_columns_returns_input_when_reference_is_missing(tmp_path: Path):
    frame = pd.DataFrame({DATE_COLUMN: ["2026-01-01"]})

    aligned = align_output_to_historical_columns(frame, tmp_path / "missing.csv")

    assert aligned is frame
