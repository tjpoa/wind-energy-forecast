from __future__ import annotations

from datetime import date
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import wind_forecast.retraining_backtesting as backtesting
from wind_forecast.retraining_backtesting import (
    RetrainingBacktestConfig,
    RetrainingBacktestError,
    RetrainingBacktestPlan,
    evaluate_no_performance_breach,
    load_retraining_backtest,
    run_retraining_backtest,
)
from wind_forecast.retraining_policy import (
    EligibilitySelection,
    ObservationEvidence,
    RetrainingPolicy,
    build_observation_folds,
)


SHA = "a" * 64


def _limits(*, mae_warning: float = 10.0) -> dict[str, dict[str, object]]:
    return {
        "MAE": {
            "warning": mae_warning,
            "critical": mae_warning + 5,
            "direction": "upper",
        },
        "RMSE": {"warning": 20.0, "critical": 30.0, "direction": "upper"},
        "MAPE_percent": {
            "warning": 20.0,
            "critical": 30.0,
            "direction": "upper",
        },
        "R2": {"warning": 0.5, "critical": 0.0, "direction": "lower"},
        "absolute_bias": {
            "warning": 10.0,
            "critical": 20.0,
            "direction": "upper",
        },
    }


def _metrics(**changes: float) -> dict[str, float]:
    return {
        "MAE": 1.0,
        "RMSE": 2.0,
        "MAPE_percent": 3.0,
        "R2": 0.9,
        "bias": -4.0,
        **changes,
    }


def _observation(index: int, day: date) -> ObservationEvidence:
    return ObservationEvidence(
        observation_id=f"observation-{index:03d}",
        target_date=day,
        feature_snapshot_id=f"snapshot-{index}",
        target_revision_id=f"actual-{index}",
        feature_schema_sha256=SHA,
        lineage_sha256=SHA,
        target_contract_id="target-v1",
        transformation_version="features-v2",
        source_revision_ids=(f"source-{index}",),
        feature_values=(float(index), float(index - 1)),
        target_value=float(index),
    )


def _record(
    outcome: str,
    *,
    training: pd.DataFrame | None = None,
    model_bytes: bytes | None = None,
) -> dict[str, object]:
    accepted = outcome == "accepted"
    final_training = (
        {
            "base_dataset_sha256": SHA,
            "incumbent_fit_cutoff": "2025-12-31",
            "data_snapshot_cutoff": "2026-07-31",
            "base_row_count": 1,
            "candidate_observation_ids": ["one", "tail"],
            "row_count": len(training),
            "identity_sha256": SHA,
            "dataset_sha256": backtesting.sha256(
                backtesting._csv_bytes(training)
            ).hexdigest(),
            "candidate_model_sha256": backtesting.sha256(model_bytes).hexdigest(),
        }
        if accepted and training is not None and model_bytes is not None
        else None
    )
    return backtesting._with_id(
        {
            "schema_version": backtesting.BACKTEST_SCHEMA,
            "evaluation_period": "2026-08",
            "evaluation_id": "evaluation",
            "outcome": outcome,
            "cutoffs": {
                "incumbent_fit_cutoff": "2025-12-31",
                "monitoring_evaluation_cutoff": "2026-07-31",
                "data_snapshot_cutoff": "2026-07-31",
                "candidate_fit_cutoff": (
                    "2026-07-31" if outcome == "accepted" else None
                ),
            },
            "identities": {
                "policy_sha256": SHA,
                "evaluation_sha256": SHA,
                "incumbent_model_sha256": SHA,
                "incumbent_dataset_sha256": SHA,
                "calibration_id": "calibration",
                "reference_id": "reference",
                "feature_schema_sha256": SHA,
            },
            "recipe": {
                "model_type": "RandomForestRegressor",
                "parameters": {},
                "source": "exact_incumbent_model_manifest_get_params",
            },
            "fold_plan": {},
            "evaluated_complete_fold_observation_ids": ["one"],
            "candidate_snapshot_observation_ids": ["one", "tail"],
            "final_training": final_training,
            "folds": [],
            "aggregate_metrics": {},
            "gates": {},
            "safeguards": {"rejected_model_persisted": False},
            "git": {"git_sha": "b" * 40, "git_dirty": False},
            "environment": {
                "schema_version": "wind_forecast.retraining_environment.v1",
                "python": "test",
                "packages": {},
            },
        }
    )


def test_no_performance_breach_uses_all_phase9_metrics_and_absolute_bias() -> None:
    passed = evaluate_no_performance_breach(_metrics(), _limits())
    assert passed["passed"] is True
    assert passed["values"]["absolute_bias"] == 4.0

    warning = evaluate_no_performance_breach(
        _metrics(MAE=11.0), _limits(mae_warning=10.0)
    )
    assert warning["passed"] is False
    assert warning["severity"]["MAE"] == "warning"
    assert warning["breached_metrics"] == ["MAE"]


def test_no_performance_breach_rejects_missing_threshold_metric() -> None:
    limits = _limits()
    limits.pop("R2")
    with pytest.raises(RetrainingBacktestError, match="differ"):
        evaluate_no_performance_breach(_metrics(), limits)


def test_folds_preserve_gaps_exclude_tail_and_prevent_future_training() -> None:
    days = pd.date_range("2026-01-01", periods=96, freq="D")
    days = days.delete([10, 40])
    observations = [
        _observation(index, timestamp.date())
        for index, timestamp in enumerate(days, start=1)
    ]
    selection = EligibilitySelection(
        eligible=tuple(observations),
        exclusions={},
        target_contract_id="target-v1",
        transformation_version="features-v2",
        feature_schema_sha256=SHA,
    )
    folds = build_observation_folds(
        selection,
        incumbent_fit_cutoff="2025-12-31",
        fold_observation_count=30,
        minimum_complete_folds=3,
    )
    assert len(folds.folds) == 3
    assert len(folds.trailing_observation_ids) == 4
    assert any(fold.calendar_gap_dates for fold in folds.folds)
    assert all(
        fold.fold_train_cutoff < fold.fold_evaluation_start
        for fold in folds.folds
    )


def test_exact_recipe_rejects_unsupported_model() -> None:
    with pytest.raises(RetrainingBacktestError, match="Unsupported"):
        backtesting._incumbent_recipe(
            {
                "model_manifest": {
                    "model_type": "GradientBoostingRegressor",
                    "parameters": {},
                }
            }
        )


def test_dry_run_creates_no_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record("rejected")
    plan = RetrainingBacktestPlan(
        outcome="rejected",
        evaluation_period="2026-08",
        backtest_id=str(record["backtest_id"]),
        record=record,
        final_model=None,
        predictions=pd.DataFrame(),
        training_frame=None,
        training_evidence=None,
        model_bytes=None,
    )
    monkeypatch.setattr(backtesting, "plan_retraining_backtest", lambda config: plan)
    output = tmp_path / "absent"
    result = run_retraining_backtest(
        RetrainingBacktestConfig(
            evaluation_path=tmp_path / "evaluation.json",
            monitoring_store_root=tmp_path / "monitoring",
            incumbent_bundle=tmp_path / "bundle",
            incumbent_base_dataset=tmp_path / "base.csv",
            calibration_dir=tmp_path / "calibration",
            policy_path=tmp_path / "policy.json",
            output_root=output,
            dry_run=True,
        )
    )
    assert result.status == "planned"
    assert not output.exists()


@pytest.mark.parametrize("outcome", ["accepted", "rejected"])
def test_sealed_bundle_is_idempotent_and_rejects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    features = ["feature", "Wind_Production_Lag1"]
    training = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=12).strftime("%Y-%m-%d"),
            "Wind_Production": np.arange(12, dtype=float),
            "feature": np.arange(12, dtype=float),
            "Wind_Production_Lag1": np.arange(12, dtype=float) - 1,
        }
    )
    model = (
        RandomForestRegressor(n_estimators=2, random_state=1).fit(
            training[features], training["Wind_Production"]
        )
        if outcome == "accepted"
        else None
    )
    model_bytes = None
    evidence = None
    if model is not None:
        buffer = BytesIO()
        backtesting.joblib.dump(model, buffer)
        model_bytes = buffer.getvalue()
        evidence = training.copy()
        evidence["Expected_Prediction"] = model.predict(training[features])
    record = _record(
        outcome,
        training=training if model is not None else None,
        model_bytes=model_bytes,
    )
    predictions = pd.DataFrame(
        {
            "observation_id": ["one"],
            "Date": ["2026-01-01"],
            "model": ["candidate"],
            "Actual_Wind_Production": [1.0],
            "Predicted_Wind_Production": [1.0],
        }
    )
    plan = RetrainingBacktestPlan(
        outcome=outcome,
        evaluation_period="2026-08",
        backtest_id=str(record["backtest_id"]),
        record=record,
        final_model=model,
        predictions=predictions,
        training_frame=training if model is not None else None,
        training_evidence=evidence,
        model_bytes=model_bytes,
    )
    monkeypatch.setattr(backtesting, "plan_retraining_backtest", lambda config: plan)
    config = RetrainingBacktestConfig(
        evaluation_path=tmp_path / "evaluation.json",
        monitoring_store_root=tmp_path / "monitoring",
        incumbent_bundle=tmp_path / "bundle",
        incumbent_base_dataset=tmp_path / "base.csv",
        calibration_dir=tmp_path / "calibration",
        policy_path=tmp_path / "policy.json",
        output_root=tmp_path / "output",
    )
    first = run_retraining_backtest(config)
    second = run_retraining_backtest(config)
    assert first.backtest_dir == second.backtest_dir
    manifest = load_retraining_backtest(first.backtest_dir)
    assert manifest["backtest"]["outcome"] == outcome
    assert (first.backtest_dir / "model.joblib").is_file() is (
        outcome == "accepted"
    )
    if outcome == "accepted":
        dataset_manifest = json.loads(
            (first.backtest_dir / "dataset_manifest.json").read_text()
        )
        assert dataset_manifest["candidate_observation_ids"] == ["one", "tail"]
        assert dataset_manifest["evaluated_complete_fold_observation_ids"] == [
            "one"
        ]
    else:
        (first.backtest_dir / "model.joblib").write_bytes(b"unexpected")
        with pytest.raises(RetrainingBacktestError, match="unexpected"):
            load_retraining_backtest(first.backtest_dir)
        (first.backtest_dir / "model.joblib").unlink()

    (first.backtest_dir / "predictions.csv").write_text(
        "corrupt", encoding="utf-8"
    )
    with pytest.raises(RetrainingBacktestError, match="corrupt"):
        load_retraining_backtest(first.backtest_dir)


def test_conflicting_sealed_period_fails(tmp_path: Path) -> None:
    period = tmp_path / "2026-08"
    record = _record("rejected")
    wrong = period / "wrong"
    wrong.mkdir(parents=True)
    (wrong / "manifest.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(RetrainingBacktestError):
        backtesting._validate_sealed_period(period, str(record["backtest_id"]))


def test_calibration_preflight_rejects_malformed_limits_before_fit(
    tmp_path: Path,
) -> None:
    calibration = {
        "calibration_id": "calibration",
        "policy": {
            "schema_version": "wind_forecast.monitoring_policy.v1",
            "windows_days": [30, 90],
            "r2_minimum_samples": {"30": 20},
        },
        "thresholds": {"performance": {"30": _limits()}},
    }
    calibration["thresholds"]["performance"]["30"]["R2"]["direction"] = "upper"
    config = RetrainingBacktestConfig(
        evaluation_path=tmp_path / "evaluation.json",
        monitoring_store_root=tmp_path / "monitoring",
        incumbent_bundle=tmp_path / "bundle",
        incumbent_base_dataset=tmp_path / "base.csv",
        calibration_dir=tmp_path / "calibration",
        policy_path=tmp_path / "policy.json",
        output_root=tmp_path / "output",
    )
    with pytest.raises(RetrainingBacktestError, match="R2 leaf"):
        backtesting._validate_calibration_preflight(config, calibration)
    assert not config.output_root.exists()


@pytest.mark.parametrize(
    ("mae_warning", "expected_outcome"),
    [(10_000.0, "accepted"), (0.0, "rejected")],
)
def test_plan_retraining_backtest_synthetic_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mae_warning: float,
    expected_outcome: str,
) -> None:
    features = ["feature", "Wind_Production_Lag1"]
    base = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=5).strftime("%Y-%m-%d"),
            "Wind_Production": np.zeros(5),
            "feature": np.arange(5, dtype=float),
            "Wind_Production_Lag1": np.full(5, -100.0),
        }
    )
    base_path = tmp_path / "base.csv"
    base.to_csv(base_path, index=False)
    incumbent = ExtraTreesRegressor(n_estimators=3, random_state=7).fit(
        base[features], base["Wind_Production"]
    )
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    backtesting.joblib.dump(incumbent, bundle_root / "model.joblib")
    bundle = {
        "feature_names": features,
        "model_manifest": {
            "model_type": "extra_trees",
            "parameters": incumbent.get_params(deep=True),
            "model_sha256": backtesting.sha256_file(bundle_root / "model.joblib"),
            "feature_schema_sha256": SHA,
        },
        "dataset_manifest": {
            "sha256": backtesting.sha256_file(base_path),
            "splits": {
                "row_counts": {"refit_train_validation": len(base)}
            },
        },
    }
    days = pd.date_range("2025-01-06", periods=91)
    observations = [
        ObservationEvidence(
            observation_id=f"obs-{index:03d}",
            target_date=timestamp.date(),
            feature_snapshot_id=f"snapshot-{index}",
            target_revision_id=f"actual-{index}",
            feature_schema_sha256=SHA,
            lineage_sha256=SHA,
            target_contract_id="target-v1",
            transformation_version="features-v2",
            source_revision_ids=(f"source-{index}",),
            feature_values=(float(index + 10), -100.0),
            target_value=float(index + 10),
        )
        for index, timestamp in enumerate(days)
    ]
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text("{}", encoding="utf-8")
    evaluation = {
        "evaluation_id": "evaluation",
        "evaluation_period": "2025-05",
        "outcome": "eligible_for_manual_backtest",
        "cutoffs": {
            "incumbent_fit_cutoff": "2025-01-05",
            "monitoring_evaluation_cutoff": days[-1].date().isoformat(),
            "data_snapshot_cutoff": days[-1].date().isoformat(),
        },
    }
    limits = _limits(mae_warning=mae_warning)
    if expected_outcome == "rejected":
        limits["MAE"]["critical"] = 1.0
    else:
        for metric in ("RMSE", "MAPE_percent", "absolute_bias"):
            limits[metric]["warning"] = 10_000.0
            limits[metric]["critical"] = 20_000.0
        limits["R2"]["warning"] = -10_000.0
        limits["R2"]["critical"] = -20_000.0
    calibration = {
        "calibration_id": "calibration",
        "reference_id": "reference",
        "mape_epsilon": 1.0,
        "policy": {
            "schema_version": "wind_forecast.monitoring_policy.v1",
            "windows_days": [30, 90],
            "r2_minimum_samples": {"30": 20},
        },
        "thresholds": {"performance": {"30": limits}},
    }
    monkeypatch.setattr(
        backtesting,
        "load_monthly_retraining_evaluation",
        lambda path: evaluation,
    )
    monkeypatch.setattr(
        backtesting,
        "validate_monitoring_model_bundle",
        lambda path: bundle,
    )
    monkeypatch.setattr(
        backtesting,
        "load_monitoring_calibration",
        lambda path: calibration,
    )
    monkeypatch.setattr(
        backtesting,
        "_verify_pinned_inputs",
        lambda *args: {"as_issued": {}, "actuals": {}},
    )
    monkeypatch.setattr(
        backtesting,
        "_reconstruct_observations",
        lambda *args: observations,
    )
    config = RetrainingBacktestConfig(
        evaluation_path=evaluation_path,
        monitoring_store_root=tmp_path / "monitoring",
        incumbent_bundle=bundle_root,
        incumbent_base_dataset=base_path,
        calibration_dir=tmp_path / "calibration",
        policy_path=Path("config/retraining_policy_v1.json"),
        output_root=tmp_path / "output",
        dry_run=True,
    )
    assert isinstance(RetrainingPolicy.load(config.policy_path), RetrainingPolicy)
    plan = backtesting.plan_retraining_backtest(config)
    assert plan.outcome == expected_outcome
    assert len(plan.record["evaluated_complete_fold_observation_ids"]) == 90
    assert len(plan.record["candidate_snapshot_observation_ids"]) == 91
    if expected_outcome == "accepted":
        assert len(plan.record["final_training"]["candidate_observation_ids"]) == 91
        assert plan.record["final_training"]["dataset_sha256"]
    else:
        assert plan.record["final_training"] is None
