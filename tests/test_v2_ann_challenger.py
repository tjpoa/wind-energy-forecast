from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

import wind_forecast.v2_ann_challenger as challenger
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN
from wind_forecast.v2_ann import ANNTrainingConfig, fit_v2_ann_candidate


def _frame(rows: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    target = np.linspace(100.0, 200.0, rows)
    return pd.DataFrame(
        {
            DATE_COLUMN: dates,
            TARGET_COLUMN: target,
            "Feature_A": np.linspace(1.0, 4.0, rows),
            "Wind_Production_Lag1": np.concatenate(([90.0], target[:-1])),
        }
    )


def _candidate(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    frame = _frame()
    input_path = tmp_path / "features.csv"
    frame.to_csv(input_path, index=False)
    from tests.test_v2_ann import _scaler_bundle

    scaler_dir = _scaler_bundle(tmp_path, input_path, frame)
    result = fit_v2_ann_candidate(
        ANNTrainingConfig(
            input_path=input_path,
            scaler_dir=scaler_dir,
            output_dir=tmp_path / "candidate",
            max_epochs=2,
            patience=1,
            batch_size=4,
            train_start="2024-01-01",
            train_end="2024-01-15",
            validation_start="2024-01-16",
            validation_end="2024-01-22",
            test_start="2024-01-23",
            test_end="2024-01-30",
        )
    )
    return result.output_dir, input_path, frame


def test_challenger_backtest_builds_complete_folds_without_writing(
    tmp_path: Path, monkeypatch
) -> None:
    candidate_dir, input_path, frame = _candidate(tmp_path)
    incumbent_dir = tmp_path / "incumbent"
    incumbent_dir.mkdir()
    incumbent = DummyRegressor(strategy="mean").fit(
        frame[["Feature_A", "Wind_Production_Lag1"]].iloc[:22],
        frame[TARGET_COLUMN].iloc[:22],
    )
    joblib.dump(incumbent, incumbent_dir / "model.joblib")
    (incumbent_dir / "model_manifest.json").write_text(
        json.dumps({"model_sha256": "not-used"}), encoding="utf-8"
    )
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    calibration = {
        "calibration_id": "calibration-test",
        "mape_epsilon": 1e-6,
        "policy": {"r2_minimum_samples": {"30": 2}},
        "thresholds": {
            "performance": {
                "30": {
                    "MAE": {"warning": 1e9, "critical": 2e9, "direction": "upper"},
                    "RMSE": {"warning": 1e9, "critical": 2e9, "direction": "upper"},
                    "MAPE_percent": {"warning": 1e9, "critical": 2e9, "direction": "upper"},
                    "R2": {"warning": -1e9, "critical": -2e9, "direction": "lower"},
                    "absolute_bias": {"warning": 1e9, "critical": 2e9, "direction": "upper"},
                }
            }
        },
    }
    (calibration_dir / "calibration.json").write_text(
        json.dumps(calibration), encoding="utf-8"
    )
    monkeypatch.setattr(
        challenger,
        "validate_monitoring_model_bundle",
        lambda path: {
            "root": incumbent_dir,
            "feature_names": ["Feature_A", "Wind_Production_Lag1"],
        },
    )
    monkeypatch.setattr(
        challenger, "load_monitoring_calibration", lambda path: calibration
    )

    result = challenger.run_v2_ann_challenger_backtest(
        challenger.ChallengerBacktestConfig(
            candidate_bundle=candidate_dir,
            incumbent_bundle=incumbent_dir,
            dataset_path=input_path,
            incumbent_calibration=calibration_dir,
            test_start="2024-01-23",
            test_end="2024-01-30",
            fold_size=2,
            dry_run=True,
        )
    )

    assert result.status == "planned"
    assert result.plan.record["fold_count"] == 4
    assert result.plan.record["safeguards"]["test_used_for_selection"] is False
    assert not (tmp_path / "outputs").exists()
