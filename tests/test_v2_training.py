import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

import scripts.train_v2_reference as v2_cli

from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN
from wind_forecast.v2_training import (
    PERSISTENCE_COLUMN,
    add_persistence_skill,
    audit_training_contract,
    chronological_split,
    passes_reference_gate,
    persistence_predictions,
    regression_metrics,
    run_v2_reference_training,
    select_candidate,
    split_assignment_sha256,
)
from wind_forecast.tracking import TrackingConfig


def _frame(rows: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    target = np.linspace(100.0, 200.0, rows) + np.sin(np.arange(rows)) * 5
    lag = np.concatenate(([90.0], target[:-1]))
    return pd.DataFrame(
        {
            DATE_COLUMN: dates,
            TARGET_COLUMN: target,
            "Average_Wind_Speed": target / 20,
            "Average_Temperature": 12 + np.arange(rows) / 10,
            PERSISTENCE_COLUMN: lag,
            "Feature_A": np.arange(rows, dtype=float),
        }
    )


def _split(frame: pd.DataFrame):
    return chronological_split(
        frame,
        train_start="2024-01-01",
        train_end="2024-01-15",
        validation_start="2024-01-16",
        validation_end="2024-01-22",
        test_start="2024-01-23",
        test_end="2024-01-30",
    )


def test_chronological_split_is_fixed_ordered_and_disjoint():
    split = _split(_frame())

    assert [len(split.train), len(split.validation), len(split.test)] == [15, 7, 8]
    assert split.train[DATE_COLUMN].max() < split.validation[DATE_COLUMN].min()
    assert split.validation[DATE_COLUMN].max() < split.test[DATE_COLUMN].min()


def test_chronological_split_rejects_rows_outside_contract():
    with pytest.raises(ValueError, match="exactly one split"):
        chronological_split(
            _frame(),
            train_start="2024-01-02",
            train_end="2024-01-15",
            validation_start="2024-01-16",
            validation_end="2024-01-22",
            test_start="2024-01-23",
            test_end="2024-01-30",
        )


def test_split_assignment_hash_covers_interior_dates():
    original = _split(_frame())
    changed_frame = _frame()
    changed_frame.loc[5, DATE_COLUMN] = pd.Timestamp("2024-01-05 12:00:00")
    changed = _split(changed_frame)

    assert split_assignment_sha256(original) != split_assignment_sha256(changed)


def test_persistence_metrics_bias_skill_selection_and_gate():
    frame = _frame(4)
    predictions = persistence_predictions(frame)
    assert predictions.tolist() == frame[PERSISTENCE_COLUMN].tolist()
    metrics = regression_metrics(frame[TARGET_COLUMN], predictions)
    assert metrics["bias"] == pytest.approx(np.mean(predictions - frame[TARGET_COLUMN]))
    skilled = add_persistence_skill({**metrics, "MAE": metrics["MAE"] / 2}, metrics)
    assert skilled["MAE_skill_vs_persistence"] == pytest.approx(0.5)
    tied = {"extra_trees": {"MAE": 2.0}, "random_forest": {"MAE": 2.0}}
    assert select_candidate(tied) == "extra_trees"
    assert passes_reference_gate({"MAE": 1.99}, {"MAE": 2.0})
    assert not passes_reference_gate({"MAE": 2.0}, {"MAE": 2.0})


def test_leakage_audit_uses_upstream_recomputation_and_detects_duplicate_dates():
    frame = _frame()
    split = _split(frame)
    audit = audit_training_contract(frame, split, upstream_validation={"passed": True})
    assert audit["passed"]
    assert audit["checks"]["upstream_feature_recomputation_passed"]
    assert audit["forecast_contract"] == "historical_daily_hindcast"

    duplicate = frame.copy()
    duplicate.loc[1, DATE_COLUMN] = duplicate.loc[0, DATE_COLUMN]
    duplicate_audit = audit_training_contract(duplicate, split)
    assert not duplicate_audit["passed"]
    assert "dates_unique" in duplicate_audit["failures"]


def test_v2_training_writes_reproducible_contract_and_reload_sample(tmp_path: Path):
    input_path = tmp_path / "features.csv"
    _frame().to_csv(input_path, index=False)
    kwargs = {
        "input_path": input_path,
        "seed": 42,
        "n_estimators": 5,
        "train_start": "2024-01-01",
        "train_end": "2024-01-15",
        "validation_start": "2024-01-16",
        "validation_end": "2024-01-22",
        "test_start": "2024-01-23",
        "test_end": "2024-01-30",
        "upstream_validation": {"passed": True, "errors": []},
    }
    first = run_v2_reference_training(output_dir=tmp_path / "run-1", **kwargs)
    second = run_v2_reference_training(output_dir=tmp_path / "run-2", **kwargs)

    assert first.row_counts == {
        "total": 30,
        "train": 15,
        "validation": 7,
        "test": 8,
        "refit_train_validation": 22,
    }
    assert first.split_sha256 == second.split_sha256
    assert first.metrics == second.metrics
    assert first.paths["test_predictions"].read_bytes() == second.paths[
        "test_predictions"
    ].read_bytes()
    assert set(first.metrics["validation"]) == {
        "persistence",
        "extra_trees",
        "random_forest",
    }
    decision = json.loads(first.paths["decision"].read_text(encoding="utf-8"))
    manifest = json.loads(first.paths["model_manifest"].read_text(encoding="utf-8"))
    assert decision["automatic_promotion"] is False
    assert decision["status"].endswith("not_promoted")
    assert manifest["scaler_required"] is False
    assert manifest["feature_names"] == list(first.feature_names)
    sample = pd.read_csv(first.paths["reload_sample"])
    model = joblib.load(first.paths["model"])
    np.testing.assert_allclose(
        model.predict(sample[list(first.feature_names)]),
        sample["Expected_Prediction"],
        rtol=1e-12,
        atol=1e-9,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        run_v2_reference_training(output_dir=tmp_path / "run-1", **kwargs)


def test_v2_mlflow_logging_uses_one_run_without_registry(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "features.csv"
    _frame().to_csv(input_path, index=False)
    result = run_v2_reference_training(
        input_path=input_path,
        output_dir=tmp_path / "run",
        seed=42,
        n_estimators=3,
        train_start="2024-01-01",
        train_end="2024-01-15",
        validation_start="2024-01-16",
        validation_end="2024-01-22",
        test_start="2024-01-23",
        test_end="2024-01-30",
        upstream_validation={"passed": True, "errors": []},
    )
    calls = []
    monkeypatch.setattr(v2_cli, "log_dataset_input", lambda *args, **kwargs: calls.append("dataset"))
    monkeypatch.setattr(
        v2_cli,
        "log_sklearn_model",
        lambda *args, **kwargs: calls.append("model") or "runs:/run-1/v2_reference_candidate",
    )
    monkeypatch.setattr(
        v2_cli,
        "log_run_data",
        lambda *args, **kwargs: calls.append(("run_data", kwargs)),
    )
    model = joblib.load(result.paths["model"])
    fake_mlflow = SimpleNamespace(
        sklearn=SimpleNamespace(load_model=lambda model_uri: model)
    )
    monkeypatch.setattr(v2_cli, "_load_mlflow", lambda: fake_mlflow)
    config = TrackingConfig(
        tracking_uri="file:///tmp/mlruns",
        experiment_name="wind-energy-forecast-v2-reference",
        registered_model_name="unused-v2-no-registry",
        dataset_version="v2",
    )

    v2_cli._log_run(
        result,
        SimpleNamespace(info=SimpleNamespace(run_id="run-1", experiment_id="exp-1")),
        config,
        input_path,
    )

    assert calls[0:2] == ["dataset", "model"]
    assert result.output_dir.joinpath("mlflow_receipt.json").exists()
    reload_evidence = json.loads(
        result.output_dir.joinpath("mlflow_reload_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert reload_evidence["predictions_equivalent"] is True
    run_payload = calls[2][1]
    assert run_payload["tags"]["registry_used"] is False
    assert run_payload["tags"]["automatic_promotion"] is False
    assert calls[3][0] == "run_data"
