from pathlib import Path

import pandas as pd
import pytest

from wind_forecast.paths import project_root
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN
from wind_forecast.training import (
    build_xy,
    load_training_table,
    run_baseline_training,
    temporal_train_test_split,
)


def _training_frame(rows: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            DATE_COLUMN: dates[::-1].strftime("%Y-%m-%d"),
            TARGET_COLUMN: list(reversed(range(100, 100 + rows))),
            "Feature_A": list(reversed(range(rows))),
            "Feature_B": [float(value % 3) for value in reversed(range(rows))],
        }
    )


def test_load_training_table_sorts_dates_without_mutating(tmp_path: Path):
    frame = _training_frame()
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)

    loaded = load_training_table(path)

    assert frame.iloc[0][DATE_COLUMN] == "2026-01-12"
    assert loaded[DATE_COLUMN].dt.strftime("%Y-%m-%d").tolist()[0] == "2026-01-01"
    assert loaded[TARGET_COLUMN].tolist()[0] == 100


def test_temporal_train_test_split_preserves_chronological_holdout():
    frame = _training_frame(rows=10).sort_values(DATE_COLUMN).reset_index(drop=True)
    features, target, dates = build_xy(frame)

    split = temporal_train_test_split(features, target, dates, test_fraction=0.3)

    assert len(split.x_train) == 7
    assert len(split.x_test) == 3
    assert split.train_dates.dt.strftime("%Y-%m-%d").tolist()[-1] == "2026-01-07"
    assert split.test_dates.dt.strftime("%Y-%m-%d").tolist()[0] == "2026-01-08"


def test_run_baseline_training_writes_expected_outputs(tmp_path: Path):
    input_path = tmp_path / "features.csv"
    output_dir = tmp_path / "baseline"
    _training_frame(rows=20).to_csv(input_path, index=False)

    result = run_baseline_training(
        input_path=input_path,
        output_dir=output_dir,
        model_type="extra_trees",
        n_estimators=5,
        seed=42,
        test_fraction=0.25,
    )

    assert result.model_path.exists()
    assert result.metrics_path.exists()
    assert result.predictions_path.exists()
    assert result.summary_path.exists()
    assert result.train_row_count == 15
    assert result.test_row_count == 5
    assert set(result.metrics) == {"R2", "MAE", "RMSE", "MAPE (%)"}
    predictions = pd.read_csv(result.predictions_path)
    assert predictions.columns.tolist() == [
        DATE_COLUMN,
        "Actual_Wind_Production",
        "Predicted_Wind_Production",
    ]
    assert len(predictions) == 5


def test_run_baseline_training_refuses_to_overwrite_known_outputs(tmp_path: Path):
    input_path = tmp_path / "features.csv"
    output_dir = tmp_path / "baseline"
    _training_frame(rows=12).to_csv(input_path, index=False)

    run_baseline_training(
        input_path=input_path,
        output_dir=output_dir,
        n_estimators=3,
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_baseline_training(
            input_path=input_path,
            output_dir=output_dir,
            n_estimators=3,
        )


def test_run_baseline_training_requires_project_outputs_dir_for_local_artifacts(
    tmp_path: Path,
):
    input_path = tmp_path / "features.csv"
    _training_frame(rows=12).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="under outputs"):
        run_baseline_training(
            input_path=input_path,
            output_dir=project_root() / "baseline-training-test-output",
            n_estimators=3,
        )
