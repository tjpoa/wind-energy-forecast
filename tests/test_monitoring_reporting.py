from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import wind_forecast.monitoring_reporting as reporting
from wind_forecast.monitoring_reporting import (
    CalibrationConfig,
    MonitoringReportConfig,
    calibrate_monitoring_reference,
    load_alert_history,
    load_active_alerts,
    load_monitoring_calibration,
    load_monitoring_report,
    plan_monitoring_report,
    run_monitoring_report,
)


def _sha(path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _calibration_environment(tmp_path, monkeypatch):
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    frame = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Wind_Production": 100 + 20 * np.sin(np.arange(len(dates)) / 20),
            "x": np.linspace(0, 1, len(dates)),
        }
    )
    dataset = tmp_path / "dataset.csv"
    frame.to_csv(dataset, index=False, lineterminator="\n")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    model = DummyRegressor(strategy="mean").fit(frame[["x"]], frame["Wind_Production"])
    joblib.dump(model, bundle / "model.joblib")
    test_dates = pd.date_range("2025-01-01", periods=150, freq="D")
    actual = 100 + 15 * np.sin(np.arange(len(test_dates)) / 10)
    test = pd.DataFrame(
        {
            "Date": test_dates.strftime("%Y-%m-%d"),
            "Actual_Wind_Production": actual,
            "model": "dummy",
            "Predicted_Wind_Production": actual + np.cos(np.arange(len(test_dates))),
        }
    )
    test_path = bundle / "test_predictions.csv"
    test.to_csv(test_path, index=False, lineterminator="\n")
    (bundle / "run_summary.json").write_text(
        json.dumps({"artifact_sha256": {"test_predictions.csv": _sha(test_path)}}),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.monitoring_policy.v1",
                "reference_start": "2022-01-01",
                "reference_end": "2023-12-31",
                "windows_days": [30, 90],
                "warning_quantile": 0.95,
                "critical_quantile": 0.99,
                "minimum_samples": {"30": 15, "90": 45},
                "r2_minimum_samples": {"30": 20, "90": 60},
                "mape_epsilon_quantile": 0.01,
                "alert_persistence_distinct_dates": 3,
                "source_objective_days": 5,
                "source_late_days": 7,
                "hard_quality_tolerance": 0,
                "overrides": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reporting,
        "validate_monitoring_model_bundle",
        lambda _root: {
            "dataset_manifest": {
                "sha256": _sha(dataset),
                "transformation_version": "test-transform",
                "splits": {
                    "row_counts": {"refit_train_validation": len(frame)},
                    "train": {"start": "2022-01-01"},
                    "validation": {"end": "2023-12-31"},
                },
            },
            "model_manifest": {
                "model_sha256": _sha(bundle / "model.joblib"),
                "feature_schema_sha256": "schema-hash",
                "model_type": "dummy",
            },
            "feature_names": ["x"],
        },
    )
    result = calibrate_monitoring_reference(
        CalibrationConfig(
            dataset_path=dataset,
            model_bundle=bundle,
            policy_path=policy,
            output_root=tmp_path / "calibration-output",
            backtest_stride_days=30,
        )
    )
    return result


def test_calibration_is_content_addressed_and_reproducible(tmp_path, monkeypatch) -> None:
    first = _calibration_environment(tmp_path, monkeypatch)
    loaded = load_monitoring_calibration(first.calibration_dir)

    assert loaded["reference_id"] == first.reference_id
    assert loaded["mape_epsilon"] > 0
    assert set(loaded["thresholds"]["feature_drift"]) == {"x"}
    assert set(loaded["thresholds"]["performance"]) == {"30", "90"}

    second = calibrate_monitoring_reference(
        CalibrationConfig(
            dataset_path=tmp_path / "dataset.csv",
            model_bundle=tmp_path / "bundle",
            policy_path=tmp_path / "policy.json",
            output_root=tmp_path / "calibration-output",
            backtest_stride_days=30,
        )
    )
    assert second.reference_id == first.reference_id
    assert second.calibration_id == first.calibration_id
    (first.calibration_dir / "backtest_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(reporting.MonitoringReportingError, match="backtest summary"):
        load_monitoring_calibration(first.calibration_dir)


def test_calibration_reference_path_is_relocatable(tmp_path, monkeypatch) -> None:
    calibrated = _calibration_environment(tmp_path, monkeypatch)
    calibration_path = calibrated.calibration_dir / "calibration.json"
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    payload["reference_dir"] = "C:/previous-host/missing/reference"
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_monitoring_calibration(calibrated.calibration_dir)

    assert loaded["reference_id"] == calibrated.reference_id
    assert Path(loaded["_reference_path"]).is_file()


def test_calibration_rejects_reference_boundaries_outside_train_validation(
    tmp_path, monkeypatch
) -> None:
    _calibration_environment(tmp_path, monkeypatch)
    policy_path = tmp_path / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["reference_start"] = "2022-01-02"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(reporting.MonitoringReportingError, match="exactly match"):
        calibrate_monitoring_reference(
            CalibrationConfig(
                dataset_path=tmp_path / "dataset.csv",
                model_bundle=tmp_path / "bundle",
                policy_path=policy_path,
                output_root=tmp_path / "invalid-calibration",
                backtest_stride_days=30,
            )
        )


def test_calibration_resolves_explicit_threshold_override(tmp_path, monkeypatch) -> None:
    _calibration_environment(tmp_path, monkeypatch)
    policy_path = tmp_path / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["overrides"] = {
        "performance.30.MAE": {"warning": 999.0, "critical": 1000.0}
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    result = calibrate_monitoring_reference(
        CalibrationConfig(
            dataset_path=tmp_path / "dataset.csv",
            model_bundle=tmp_path / "bundle",
            policy_path=policy_path,
            output_root=tmp_path / "override-calibration",
            backtest_stride_days=30,
        )
    )
    limits = load_monitoring_calibration(result.calibration_dir)["thresholds"][
        "performance"
    ]["30"]["MAE"]
    assert (limits["warning"], limits["critical"], limits["override"]) == (
        999.0,
        1000.0,
        True,
    )


def test_v2_quality_is_required_and_manifest_lineage_must_match(tmp_path, monkeypatch) -> None:
    calibrated = _calibration_environment(tmp_path, monkeypatch)
    source_root = tmp_path / "source-contract"
    source_root.mkdir()
    manifest_path = source_root / "manifest.json"
    manifest = {
        "schema_version": "wind_forecast.v2_incremental_run.v2",
        "run_id": "source-contract",
        "status": "failed",
        "command": {"through_date": "2026-01-01"},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = MonitoringReportConfig(
        source_run_manifest=manifest_path,
        monitoring_store_root=tmp_path / "monitoring-contract",
        calibration_dir=calibrated.calibration_dir,
        through_date="2026-01-01",
        dry_run=True,
    )
    with pytest.raises(reporting.MonitoringReportingError, match="sidecar is missing"):
        plan_monitoring_report(config)

    quality_path = source_root / "quality.json"
    quality_path.write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.batch_quality.v1",
                "run_id": "different-run",
                "batch_status": "failed",
                "through_date": "2026-01-01",
            }
        ),
        encoding="utf-8",
    )
    manifest["quality_evidence"] = {
        "path": str(quality_path.resolve()),
        "sha256": _sha(quality_path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(reporting.MonitoringReportingError, match="does not belong"):
        plan_monitoring_report(config)

    manifest["schema_version"] = "wind_forecast.v2_incremental_run.v1"
    manifest.pop("quality_evidence")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert plan_monitoring_report(config).quality_available is False

    mismatch = MonitoringReportConfig(
        source_run_manifest=manifest_path,
        monitoring_store_root=tmp_path / "monitoring-contract",
        calibration_dir=calibrated.calibration_dir,
        through_date="2026-01-02",
        dry_run=True,
    )
    with pytest.raises(reporting.MonitoringReportingError, match="differs"):
        plan_monitoring_report(mismatch)


def test_quality_only_report_is_immutable_and_opens_immediate_alert(tmp_path, monkeypatch) -> None:
    calibrated = _calibration_environment(tmp_path, monkeypatch)
    source_root = tmp_path / "source-run"
    source_root.mkdir()
    quality = {
        "schema_version": "wind_forecast.batch_quality.v1",
        "run_id": "source-1",
        "batch_status": "failed",
        "through_date": "2026-01-01",
        "verdict": "FAIL",
        "issues": [
            {
                "code": "schema_validation_failed",
                "severity": "critical",
                "count": 1,
                "sample": ["missing column"],
            }
        ],
    }
    quality_path = source_root / "quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    manifest = {
        "schema_version": "wind_forecast.v2_incremental_run.v2",
        "run_id": "source-1",
        "status": "failed",
        "command": {"through_date": "2026-01-01"},
        "quality_evidence": {
            "path": str(quality_path.resolve()),
            "sha256": _sha(quality_path),
        },
    }
    manifest_path = source_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    store = tmp_path / "monitoring"

    dry = run_monitoring_report(
        MonitoringReportConfig(
            source_run_manifest=manifest_path,
            monitoring_store_root=store,
            calibration_dir=calibrated.calibration_dir,
            through_date="2026-01-01",
            dry_run=True,
        )
    )
    assert dry.status == "planned"
    assert not store.exists()

    result = run_monitoring_report(
        MonitoringReportConfig(
            source_run_manifest=manifest_path,
            monitoring_store_root=store,
            calibration_dir=calibrated.calibration_dir,
            through_date="2026-01-01",
            now_utc=datetime(2026, 1, 8, 12, tzinfo=timezone.utc),
        )
    )
    report = load_monitoring_report(result.report_path)

    assert report["windows"]["30"]["status"] == "not_available"
    assert result.active_alert_count == 1
    assert result.markdown_path.is_file()
    assert len(load_active_alerts(store)) == 1
    assert [item["event_type"] for item in load_alert_history(store)] == ["opened"]

    repeated = run_monitoring_report(
        MonitoringReportConfig(
            source_run_manifest=manifest_path,
            monitoring_store_root=store,
            calibration_dir=calibrated.calibration_dir,
            through_date="2026-01-01",
            now_utc=datetime(2026, 1, 8, 13, tzinfo=timezone.utc),
        )
    )
    repeated_report = load_monitoring_report(repeated.report_path)
    assert repeated.active_alert_count == 1
    assert repeated_report["alert_events"] == []
    with pytest.raises(reporting.MonitoringReportingError, match="Immutable path"):
        reporting._immutable_json(result.report_path, {"different": True})
    alert_id = next(iter(load_active_alerts(store).values()))
    alert_path = store / "reporting" / "alerts" / f"{alert_id}.json"
    alert_path.write_text("{}", encoding="utf-8")
    with pytest.raises(reporting.MonitoringReportingError, match="content-addressed"):
        load_alert_history(store)


def test_statistical_alert_requires_three_distinct_report_dates() -> None:
    breach = [
        {
            "rule_id": "feature_drift:x:30:global",
            "severity": "warning",
            "category": "feature_drift",
            "immediate": False,
        }
    ]
    state, events = reporting._evaluate_alerts(None, date(2026, 1, 1), breach, 3)
    assert events == []
    state, events = reporting._evaluate_alerts(state, date(2026, 1, 2), breach, 3)
    assert events == []
    state, events = reporting._evaluate_alerts(state, date(2026, 1, 3), breach, 3)
    assert [event["event_type"] for event in events] == ["opened"]

    same_day, events = reporting._evaluate_alerts(state, date(2026, 1, 3), breach, 3)
    assert events == []
    assert same_day["rules"][breach[0]["rule_id"]]["consecutive"] == 3

    _, events = reporting._evaluate_alerts(same_day, date(2026, 1, 4), [], 3)
    assert [event["event_type"] for event in events] == ["resolved"]


def test_statistical_alert_persistence_resets_after_a_date_gap() -> None:
    breach = [
        {
            "rule_id": "performance:mae:30",
            "severity": "warning",
            "category": "performance",
            "immediate": False,
        }
    ]
    state, _ = reporting._evaluate_alerts(None, date(2026, 1, 1), breach, 3)
    state, events = reporting._evaluate_alerts(state, date(2026, 1, 3), breach, 3)
    assert events == []
    assert state["rules"][breach[0]["rule_id"]]["consecutive"] == 1

    state, _ = reporting._evaluate_alerts(state, date(2026, 1, 4), breach, 3)
    state, events = reporting._evaluate_alerts(state, date(2026, 1, 5), breach, 3)
    assert [event["event_type"] for event in events] == ["opened"]


def test_immediate_alert_reopens_same_day_and_report_date_cannot_regress(tmp_path) -> None:
    breach = [
        {
            "rule_id": "quality:schema_validation_failed",
            "severity": "critical",
            "category": "quality",
            "immediate": True,
        }
    ]
    state, opened = reporting._evaluate_alerts(None, date(2026, 1, 2), breach, 3)
    state, resolved = reporting._evaluate_alerts(state, date(2026, 1, 2), [], 3)
    state, reopened = reporting._evaluate_alerts(state, date(2026, 1, 2), breach, 3)
    assert [item["event_type"] for item in opened + resolved + reopened] == [
        "opened",
        "resolved",
        "opened",
    ]
    assert reopened[0]["previous_alert_event_id"] == resolved[0]["alert_event_id"]
    for event in opened + resolved + reopened:
        reporting._immutable_json(
            tmp_path
            / "reporting"
            / "alerts"
            / f"{event['alert_event_id']}.json",
            event,
        )
    assert [item["event_type"] for item in load_alert_history(tmp_path)] == [
        "opened",
        "resolved",
        "opened",
    ]
    with pytest.raises(reporting.MonitoringReportingError, match="cannot precede"):
        reporting._evaluate_alerts(state, date(2026, 1, 1), [], 3)


def test_report_state_rejects_missing_inactive_alert_predecessor(tmp_path) -> None:
    reporting_root = tmp_path / "reporting"
    state_path = reporting_root / "state" / "current.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": reporting.REPORT_STATE_SCHEMA,
                "rules": {
                    "quality:test": {
                        "active": False,
                        "last_event_id": "missing-event",
                    }
                },
                "active_alerts": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(reporting.MonitoringReportingError, match="Invalid JSON artifact"):
        reporting._load_report_state(reporting_root)


def test_full_report_reads_as_issued_evidence_without_mutating_it(tmp_path, monkeypatch) -> None:
    calibrated = _calibration_environment(tmp_path, monkeypatch)
    source_root = tmp_path / "source-success"
    source_root.mkdir()
    quality = {
        "schema_version": "wind_forecast.batch_quality.v1",
        "run_id": "source-success",
        "batch_status": "succeeded",
        "through_date": "2026-03-31",
        "verdict": "PASS",
        "issues": [],
    }
    quality_path = source_root / "quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    manifest_path = source_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.v2_incremental_run.v2",
                "run_id": "source-success",
                "status": "succeeded",
                "command": {"through_date": "2026-03-31"},
                "quality_evidence": {
                    "path": str(quality_path.resolve()),
                    "sha256": _sha(quality_path),
                },
            }
        ),
        encoding="utf-8",
    )
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    as_issued = {day.strftime("%Y-%m-%d"): f"prediction-{index}" for index, day in enumerate(dates)}
    metrics = {f"prediction-{index}": f"metric-{index}" for index in range(len(dates))}
    state = {
        "generation": 1,
        "source_generation": 1,
        "as_issued": as_issued,
        "restated": {},
        "metrics": metrics,
    }

    def evidence(_root, prediction_id):
        index = int(str(prediction_id).split("-")[-1])
        day = dates[index].strftime("%Y-%m-%d")
        actual_id = f"actual-{index}"
        return {
            "prediction": {
                "prediction_id": prediction_id,
                "target_date": day,
                "prediction": 100.0,
            },
            "model_input_snapshot": {
                "model_input_snapshot_id": f"input-{index}",
                "feature_names": ["x"],
                "feature_values": [index / 89],
            },
            "metric_revisions": [
                {
                    "metric_revision_id": f"metric-{index}",
                    "actual_revision_id": actual_id,
                }
            ],
            "actual_revisions": [
                {"actual_revision_id": actual_id, "actual": 100.0 + np.sin(index / 5)}
            ],
        }

    monkeypatch.setattr(reporting, "load_verified_monitoring_state", lambda _root: state)
    monkeypatch.setattr(reporting, "load_prediction_evidence", evidence)
    store = tmp_path / "monitoring-full"
    prediction_sentinel = store / "predictions" / "sentinel.json"
    model_sentinel = store / "model_snapshots" / "sentinel.joblib"
    prediction_sentinel.parent.mkdir(parents=True)
    model_sentinel.parent.mkdir(parents=True)
    prediction_sentinel.write_bytes(b"immutable-prediction")
    model_sentinel.write_bytes(b"immutable-model")
    before = (_sha(prediction_sentinel), _sha(model_sentinel))

    result = run_monitoring_report(
        MonitoringReportConfig(
            source_run_manifest=manifest_path,
            monitoring_store_root=store,
            calibration_dir=calibrated.calibration_dir,
            through_date="2026-03-31",
            now_utc=datetime(2026, 4, 7, 12, tzinfo=timezone.utc),
        )
    )
    report = load_monitoring_report(result.report_path)

    assert report["windows"]["30"]["status"] == "available"
    assert report["windows"]["90"]["status"] == "available"
    assert report["windows"]["30"]["performance"]["status"] == "available"
    assert report["lineage"]["primary_view"] == "as_issued"
    assert (_sha(prediction_sentinel), _sha(model_sentinel)) == before
