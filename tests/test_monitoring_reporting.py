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
    load_reporting_attempt,
    load_reporting_attempts,
    plan_monitoring_report,
    run_monitoring_report,
)


def _sha(path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _alert_model_era() -> dict[str, object]:
    return {
        "model_era_id": "era-test",
        "deployment": {"deployment_id": "deployment-test"},
        "registry": {"model_version": "7"},
    }


def _write_test_alert(store: Path, index: int) -> dict[str, object]:
    event = reporting._alert_event(
        f"quality:test-{index % 3}",
        f"2026-07-{(index % 28) + 1:02d}",
        "opened",
        "warning",
        None,
        _alert_model_era(),
    )
    alerts_root = store / "reporting" / "alerts"
    alerts_root.mkdir(parents=True, exist_ok=True)
    (alerts_root / f"{index:04d}-{event['alert_event_id']}.json").write_text(
        json.dumps(event),
        encoding="utf-8",
    )
    return event


def _write_failed_reporting_attempt(store: Path, index: int) -> dict[str, object]:
    run_id = f"reporting-run-{index:04d}"
    run_root = store / "reporting" / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    request = {
        "schema_version": "wind_forecast.monitoring_report_request.v2",
        "run_id": run_id,
        "requested_at_utc": f"2026-07-30T12:{index % 60:02d}:00Z",
        "plan": {
            "status": "planned",
            "through_date": "2026-07-29",
            "source_run_id": f"source-{index:04d}",
            "source_status": "failed",
            "calibration_id": "calibration-id",
        },
    }
    (run_root / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (run_root / "failure.json").write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.monitoring_report_failure.v1",
                "run_id": run_id,
                "failed_at_utc": f"2026-07-30T13:{index % 60:02d}:00Z",
                "error_type": "SyntheticFailure",
            }
        ),
        encoding="utf-8",
    )
    return request


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
    monkeypatch.setattr(
        reporting,
        "verify_active_model_era",
        lambda *_args, **_kwargs: {
            "model_era_id": "e" * 64,
            "association_kind": "active_deployment",
            "deployment": {
                "deployment_id": "d" * 64,
                "deployment_state_id": "s" * 64,
                "generation": 1,
            },
            "registry": {
                "registered_model_name": "wind-v2",
                "model_version": "1",
            },
            "cutoffs": {
                "fit_cutoff": "2023-12-31",
                "activation_cutoff": "2026-01-01",
            },
            "pins": {
                "model_sha256": _sha(bundle / "model.joblib"),
                "dataset_sha256": _sha(dataset),
            },
            "calibration": {
                "calibration_id": result.calibration_id,
                "reference_id": result.reference_id,
            },
            "monitoring": {"ledger_model_snapshot_id": "m" * 64},
        },
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
        model_bundle=tmp_path / "bundle",
        calibration_dir=calibrated.calibration_dir,
        deployment_root=tmp_path / "deployment",
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
        model_bundle=tmp_path / "bundle",
        calibration_dir=calibrated.calibration_dir,
        deployment_root=tmp_path / "deployment",
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
            model_bundle=tmp_path / "bundle",
            calibration_dir=calibrated.calibration_dir,
            deployment_root=tmp_path / "deployment",
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
            model_bundle=tmp_path / "bundle",
            calibration_dir=calibrated.calibration_dir,
            deployment_root=tmp_path / "deployment",
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
            model_bundle=tmp_path / "bundle",
            calibration_dir=calibrated.calibration_dir,
            deployment_root=tmp_path / "deployment",
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


def test_reporting_attempt_loader_is_verified_ordered_and_sanitized(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "reporting" / "runs"
    run_id = "20260730T120000000000Z-abcdef123456"
    run_root = runs_root / run_id
    run_root.mkdir(parents=True)
    request = {
        "schema_version": "wind_forecast.monitoring_report_request.v2",
        "run_id": run_id,
        "requested_at_utc": "2026-07-30T12:00:00Z",
        "plan": {
            "status": "planned",
            "through_date": "2026-07-29",
            "source_run_id": "source-run",
            "source_status": "failed",
            "calibration_id": "calibration-id",
        },
    }
    (run_root / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (run_root / "failure.json").write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.monitoring_report_failure.v1",
                "run_id": run_id,
                "failed_at_utc": "2026-07-30T12:01:00Z",
                "error_type": "PrivateFailure",
                "error": "C:\\private\\secret.json",
            }
        ),
        encoding="utf-8",
    )

    attempts = load_reporting_attempts(tmp_path)
    exact = load_reporting_attempt(tmp_path, reporting_run_id=run_id)

    assert attempts == [exact]
    assert exact["status"] == "failed"
    assert exact["failure"]["error_type"] == "PrivateFailure"
    assert exact["failure"]["message"] == (
        "The reporting attempt failed. Inspect local operator logs."
    )
    assert "secret.json" not in json.dumps(exact).lower()


def test_large_verified_loaders_use_bounded_parallel_map_without_changing_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item_count = reporting._PARALLEL_LOADER_MIN_ITEMS + 3
    for index in range(item_count):
        _write_test_alert(tmp_path, index + 100)
        _write_failed_reporting_attempt(tmp_path, index)

    original_threshold = reporting._PARALLEL_LOADER_MIN_ITEMS
    monkeypatch.setattr(reporting, "_PARALLEL_LOADER_MIN_ITEMS", item_count + 1)
    sequential_alerts = load_alert_history(tmp_path)
    sequential_attempts = load_reporting_attempts(tmp_path)
    monkeypatch.setattr(reporting, "_PARALLEL_LOADER_MIN_ITEMS", original_threshold)

    real_executor = reporting.ThreadPoolExecutor
    worker_counts: list[int] = []

    class RecordingExecutor:
        def __init__(self, *, max_workers: int) -> None:
            worker_counts.append(max_workers)
            self._delegate = real_executor(max_workers=max_workers)

        def __enter__(self):
            self._delegate.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._delegate.__exit__(exc_type, exc_value, traceback)

        def map(self, function, values):
            return self._delegate.map(function, values)

    monkeypatch.setattr(reporting, "ThreadPoolExecutor", RecordingExecutor)

    assert load_alert_history(tmp_path) == sequential_alerts
    assert load_reporting_attempts(tmp_path) == sequential_attempts
    assert worker_counts == [
        reporting._PARALLEL_LOADER_MAX_WORKERS,
        reporting._PARALLEL_LOADER_MAX_WORKERS,
    ]
    assert [item["run_id"] for item in sequential_attempts] == [
        item["run_id"]
        for item in sorted(
            sequential_attempts,
            key=lambda item: (item["attempted_at_utc"], item["run_id"]),
            reverse=True,
        )
    ]


def test_small_verified_loaders_remain_sequential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alert = _write_test_alert(tmp_path, 1)
    request = _write_failed_reporting_attempt(tmp_path, 1)

    class UnexpectedExecutor:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("Small stores must not construct an executor.")

    monkeypatch.setattr(reporting, "ThreadPoolExecutor", UnexpectedExecutor)

    assert [item["alert_event_id"] for item in load_alert_history(tmp_path)] == [
        alert["alert_event_id"]
    ]
    assert [item["run_id"] for item in load_reporting_attempts(tmp_path)] == [
        request["run_id"]
    ]


def test_parallel_loader_infrastructure_failure_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(reporting._PARALLEL_LOADER_MIN_ITEMS):
        _write_test_alert(tmp_path, index)

    class UnavailableExecutor:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("raw thread infrastructure detail")

    monkeypatch.setattr(reporting, "ThreadPoolExecutor", UnavailableExecutor)
    with pytest.raises(
        reporting.MonitoringReportingUnavailableError,
        match="Parallel monitoring artifact loading is unavailable",
    ) as raised:
        load_alert_history(tmp_path)
    assert "raw thread" not in str(raised.value)


def test_parallel_loaders_report_the_first_sorted_corruption_deterministically(
    tmp_path: Path,
) -> None:
    for index in range(reporting._PARALLEL_LOADER_MIN_ITEMS):
        _write_test_alert(tmp_path, index + 100)
        _write_failed_reporting_attempt(tmp_path, index + 100)

    alerts_root = tmp_path / "reporting" / "alerts"
    (alerts_root / "0000-first-corrupt.json").write_text("{", encoding="utf-8")
    (alerts_root / "0001-second-corrupt.json").write_text("{", encoding="utf-8")
    with pytest.raises(reporting.MonitoringReportingError) as alert_error:
        load_alert_history(tmp_path)
    assert "0000-first-corrupt.json" in str(alert_error.value)
    assert "0001-second-corrupt.json" not in str(alert_error.value)

    runs_root = tmp_path / "reporting" / "runs"
    for name in ("0000-first-corrupt", "0001-second-corrupt"):
        run_root = runs_root / name
        run_root.mkdir()
        (run_root / "request.json").write_text("{", encoding="utf-8")
    with pytest.raises(reporting.MonitoringReportingError) as attempt_error:
        load_reporting_attempts(tmp_path)
    assert "0000-first-corrupt" in str(attempt_error.value)
    assert "0001-second-corrupt" not in str(attempt_error.value)


def test_statistical_alert_requires_three_distinct_report_dates() -> None:
    breach = [
        {
            "rule_id": "feature_drift:x:30:global",
            "severity": "warning",
            "category": "feature_drift",
            "immediate": False,
        }
    ]
    era = _alert_model_era()
    state, events = reporting._evaluate_alerts(None, date(2026, 1, 1), breach, 3, era)
    assert events == []
    state, events = reporting._evaluate_alerts(
        state, date(2026, 1, 2), breach, 3, era
    )
    assert events == []
    state, events = reporting._evaluate_alerts(
        state, date(2026, 1, 3), breach, 3, era
    )
    assert [event["event_type"] for event in events] == ["opened"]

    same_day, events = reporting._evaluate_alerts(
        state, date(2026, 1, 3), breach, 3, era
    )
    assert events == []
    assert same_day["rules"][breach[0]["rule_id"]]["consecutive"] == 3

    _, events = reporting._evaluate_alerts(
        same_day, date(2026, 1, 4), [], 3, era
    )
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
    era = _alert_model_era()
    state, _ = reporting._evaluate_alerts(None, date(2026, 1, 1), breach, 3, era)
    state, events = reporting._evaluate_alerts(
        state, date(2026, 1, 3), breach, 3, era
    )
    assert events == []
    assert state["rules"][breach[0]["rule_id"]]["consecutive"] == 1

    state, _ = reporting._evaluate_alerts(
        state, date(2026, 1, 4), breach, 3, era
    )
    state, events = reporting._evaluate_alerts(
        state, date(2026, 1, 5), breach, 3, era
    )
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
    era = _alert_model_era()
    state, opened = reporting._evaluate_alerts(
        None, date(2026, 1, 2), breach, 3, era
    )
    state, resolved = reporting._evaluate_alerts(
        state, date(2026, 1, 2), [], 3, era
    )
    state, reopened = reporting._evaluate_alerts(
        state, date(2026, 1, 2), breach, 3, era
    )
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
        reporting._evaluate_alerts(
            state, date(2026, 1, 1), [], 3, era
        )


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
            model_bundle=tmp_path / "bundle",
            calibration_dir=calibrated.calibration_dir,
            deployment_root=tmp_path / "deployment",
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


def test_ledger_windows_exclude_predictions_from_other_model_eras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    era = {
        "model_era_id": "a" * 64,
        "deployment": {"deployment_id": "d" * 64},
        "registry": {"model_version": "1"},
        "monitoring": {"ledger_model_snapshot_id": "m" * 64},
    }
    state = {
        "schema_version": "wind_forecast.monitoring_state.v2",
        "as_issued": {
            "2026-01-01": "prediction-a",
            "2026-01-02": "prediction-b",
        },
        "metrics": {},
        "restated": {},
    }

    def evidence(_root: Path, prediction_id: str) -> dict:
        suffix = prediction_id[-1]
        return {
            "prediction": {
                "prediction_id": prediction_id,
                "model_era_id": ("a" if suffix == "a" else "b") * 64,
                "prediction": 1.0,
            },
            "model_input_snapshot": {
                "model_input_snapshot_id": f"input-{suffix}",
                "feature_names": ["x"],
                "feature_values": [1.0],
            },
            "metric_revisions": [],
            "actual_revisions": [],
        }

    monkeypatch.setattr(reporting, "load_prediction_evidence", evidence)
    frame, lineage = reporting._load_ledger_observations(
        tmp_path,
        state,
        date(2026, 1, 2),
        30,
        ["x"],
        era,
    )

    assert frame["Date"].tolist() == ["2026-01-01"]
    assert lineage["prediction_ids"] == ["prediction-a"]
    assert lineage["model_era_id"] == "a" * 64
