from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wind_forecast import operational_projection_cli as cli
from wind_forecast import operational_projection_projector as projector


NOW = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
GIT_COMMIT = "a" * 40
REPORT_ID = "b" * 64
ALERT_ID = "c" * 64
CALIBRATION_ID = "d" * 64
REFERENCE_ID = "e" * 64
MODEL_ERA_ID = "f" * 64
RUN_ID = "20260801T115900000000Z-abcdef123456"


def _era() -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_model_era.v1",
        "model_era_id": MODEL_ERA_ID,
        "association_kind": "active_deployment",
        "deployment": {
            "deployment_id": "1" * 64,
            "generation": 7,
        },
        "registry": {
            "registered_model_name": "wind-v2",
            "model_version": "11",
        },
        "cutoffs": {
            "fit_cutoff": "2025-12-31",
            "activation_cutoff": "2026-07-01",
        },
        "pins": {
            "bundle_sha256": "2" * 64,
            "model_sha256": "3" * 64,
            "dataset_sha256": "4" * 64,
            "feature_schema_sha256": "5" * 64,
            "calibration_sha256": "6" * 64,
            "ledger_sha256": "7" * 64,
        },
        "calibration": {
            "calibration_id": CALIBRATION_ID,
            "reference_id": REFERENCE_ID,
        },
    }


def _calibration() -> dict:
    metrics = {
        name: {"warning": 20.0, "critical": 30.0, "direction": "upper"}
        for name in ("MAE", "RMSE", "absolute_bias", "MAPE_percent", "R2")
    }
    metrics["R2"]["direction"] = "lower"
    return {
        "schema_version": "wind_forecast.monitoring_calibration.v1",
        "calibration_id": CALIBRATION_ID,
        "reference_id": REFERENCE_ID,
        "policy_sha256": "8" * 64,
        "thresholds": {
            "performance": {"30": metrics, "90": metrics},
            "feature_drift": {
                "wind_speed": {
                    "30": {
                        "global": {
                            detector: {
                                "warning": 0.15,
                                "critical": 0.3,
                                "direction": "upper",
                            }
                            for detector in (
                                "ks_statistic",
                                "normalized_wasserstein",
                            )
                        }
                    }
                }
            },
        },
        "_reference_manifest": {
            "schema_version": "wind_forecast.monitoring_reference.v1",
            "reference_id": REFERENCE_ID,
            "period": {"start": "2025-01-01", "end": "2025-12-31"},
        },
        "_reference_path": "must-not-be-projected.csv",
    }


def _report() -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_report.v2",
        "report_id": REPORT_ID,
        "run_id": RUN_ID,
        "created_at_utc": "2026-08-01T11:59:00Z",
        "through_date": "2026-07-31",
        "model_era": {
            "model_era_id": MODEL_ERA_ID,
            "association_kind": "active_deployment",
        },
        "source_batch": {"run_id": "source-run", "status": "succeeded"},
        "reference": {
            "calibration_id": CALIBRATION_ID,
            "reference_id": REFERENCE_ID,
            "policy_sha256": "8" * 64,
        },
        "config": {"minimum_samples": {"30": 15, "90": 45}},
        "quality": {
            "status": "available",
            "batch_status": "succeeded",
            "verdict": "PASS",
            "issues": [
                {"code": "source_late", "severity": "warning"},
                {"code": "duplicate_info", "severity": "informational"},
            ],
            "freshness": {
                "common_validated_watermark": "2026-07-31",
                "watermark_age_days": 1,
                "objective_days": 5,
                "late_days": 7,
                "objective_missed": False,
                "unresolved_late_dates": [],
            },
            "coverage": {
                "date_count": 90,
                "ren_complete_count": 90,
                "era5_complete_count": 90,
                "integration_ready_count": 90,
                "feature_ready_count": 90,
            },
        },
        "windows": {
            "30": {
                "status": "available",
                "calendar_start": "2026-07-02",
                "calendar_end": "2026-07-31",
                "sample_count": 30,
                "coverage_ratio": 1.0,
                "coverage_severity": "ok",
                "performance": {
                    "status": "available",
                    "metrics": {
                        "MAE": 10.0,
                        "RMSE": 12.0,
                        "bias": -1.0,
                        "MAPE_percent": 4.0,
                        "R2": 0.8,
                    },
                    "severity": {
                        "MAE": "ok",
                        "RMSE": "warning",
                        "bias": "ok",
                        "MAPE_percent": "ok",
                        "R2": "ok",
                    },
                },
                "feature_drift": {
                    "wind_speed": {
                        "global": {
                            "normalized_wasserstein": 0.2,
                            "ks_statistic": 0.1,
                            "severity": "warning",
                        }
                    }
                },
            },
            "90": {
                "status": "insufficient_data",
                "sample_count": 30,
                "minimum_samples": 45,
            },
        },
        "active_alerts": {"feature_drift:wind_speed:30:global": ALERT_ID},
        "alert_events": [ALERT_ID],
        "persistence": {},
        "lineage": {"prediction_ids": ["prediction-1"]},
    }


def _alert() -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_alert_event.v2",
        "alert_event_id": ALERT_ID,
        "rule_id": "feature_drift:wind_speed:30:global",
        "through_date": "2026-07-31",
        "event_type": "opened",
        "severity": "warning",
        "previous_alert_event_id": None,
    }


def _attempt() -> dict:
    return {
        "run_id": RUN_ID,
        "attempted_at_utc": "2026-08-01T11:59:00Z",
        "through_date": "2026-07-31",
        "source_pipeline_run_id": "source-run",
        "source_pipeline_status": "succeeded",
        "status": "succeeded",
        "report_id": REPORT_ID,
        "active_alert_count": 1,
        "failure": None,
    }


def _patch_sources(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    values: dict[str, object] = {
        "state": {
            "schema_version": "wind_forecast.monitoring_report_state.v2",
            "latest_report_id": REPORT_ID,
            "latest_through_date": "2026-07-31",
            "active_alerts": {
                "feature_drift:wind_speed:30:global": ALERT_ID,
            },
        },
        "active": {"feature_drift:wind_speed:30:global": ALERT_ID},
        "history": [_alert()],
        "attempts": [_attempt()],
        "report": _report(),
        "calibration": _calibration(),
        "era": _era(),
    }
    monkeypatch.setattr(projector, "load_monitoring_report_state", lambda _root: values["state"])
    monkeypatch.setattr(projector, "load_active_alerts", lambda _root: values["active"])
    monkeypatch.setattr(projector, "load_alert_history", lambda _root: values["history"])
    monkeypatch.setattr(projector, "load_reporting_attempts", lambda _root: values["attempts"])
    monkeypatch.setattr(projector, "load_monitoring_report", lambda _path: values["report"])
    monkeypatch.setattr(projector, "load_monitoring_calibration", lambda _path: values["calibration"])
    monkeypatch.setattr(projector, "resolve_report_model_era", lambda _root, _report: values["era"])
    monkeypatch.setattr(projector, "load_model_era", lambda _root, _era_id: values["era"])
    monkeypatch.setattr(
        projector,
        "load_prediction_evidence",
        lambda _root, prediction_id: {"prediction": {"prediction_id": prediction_id}},
    )
    return values


def _snapshot(tmp_path: Path) -> projector.ProjectionSnapshot:
    return projector.build_projection_snapshot(
        tmp_path,
        environment_id="local",
        source_git_commit=GIT_COMMIT,
        clock=lambda: NOW,
    )


def test_snapshot_normalizes_all_five_domains_and_minimizes_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(monkeypatch)

    snapshot = _snapshot(tmp_path)

    assert snapshot.counts() == {
        "evidence_record_count": 8,
        "generation_evidence_count": 8,
        "model_era_count": 1,
        "monitoring_report_count": 1,
        "quality_issue_count": 1,
        "monitoring_window_count": 2,
        "performance_metric_count": 5,
        "drift_measurement_count": 2,
        "alert_event_count": 1,
        "active_alert_snapshot_count": 1,
        "reporting_attempt_count": 1,
        "lineage_edge_count": 7,
    }
    assert snapshot.rows_for("monitoring_window")[1].value_map()["status"] in {
        "available",
        "not_available",
    }
    serialized = json.dumps(snapshot.manifest.payload(), sort_keys=True)
    assert "must-not-be-projected" not in serialized
    assert "prediction-1" not in serialized
    assert "projected_at" not in serialized
    assert "observed_at" not in serialized


def test_generation_identity_is_timestamp_independent_and_semantic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _patch_sources(monkeypatch)
    first = _snapshot(tmp_path)
    second = projector.build_projection_snapshot(
        tmp_path,
        environment_id="local",
        source_git_commit=GIT_COMMIT,
        clock=lambda: datetime(2026, 8, 2, 12, tzinfo=timezone.utc),
    )
    assert second.generation_id == first.generation_id

    values["attempts"] = [{**_attempt(), "source_pipeline_status": "no_op"}]
    changed = _snapshot(tmp_path)
    assert changed.generation_id != first.generation_id


def test_in_progress_attempt_fails_before_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _patch_sources(monkeypatch)
    values["attempts"] = [{**_attempt(), "status": "in_progress", "report_id": None}]

    with pytest.raises(projector.ProjectionSourceNotStableError):
        _snapshot(tmp_path)


def test_source_change_during_scan_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _patch_sources(monkeypatch)
    active_states = [values["active"], {}]
    monkeypatch.setattr(projector, "load_active_alerts", lambda _root: active_states.pop(0))

    with pytest.raises(projector.ProjectionSourceConflictError):
        _snapshot(tmp_path)


@pytest.mark.parametrize("source_kind", ["history", "attempts"])
def test_append_only_source_change_during_scan_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    values = _patch_sources(monkeypatch)
    first = values[source_kind]
    loader_name = (
        "load_alert_history" if source_kind == "history" else "load_reporting_attempts"
    )
    snapshots = [first, []]
    monkeypatch.setattr(projector, loader_name, lambda _root: snapshots.pop(0))

    with pytest.raises(projector.ProjectionSourceConflictError):
        _snapshot(tmp_path)


def test_broken_alert_chain_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _patch_sources(monkeypatch)
    values["history"] = [{**_alert(), "previous_alert_event_id": "9" * 64}]

    with pytest.raises(projector.ProjectionSourceConflictError):
        _snapshot(tmp_path)


def test_prediction_lineage_mismatch_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(monkeypatch)
    monkeypatch.setattr(
        projector,
        "load_prediction_evidence",
        lambda _root, _prediction_id: {
            "prediction": {"prediction_id": "different-prediction"}
        },
    )

    with pytest.raises(projector.ProjectionSourceConflictError):
        _snapshot(tmp_path)


def test_legacy_unassociated_report_uses_nullable_model_era(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _patch_sources(monkeypatch)
    report = _report()
    report["schema_version"] = "wind_forecast.monitoring_report.v1"
    report.pop("model_era")
    values["report"] = report
    monkeypatch.setattr(
        projector,
        "resolve_report_model_era",
        lambda _root, _report: {"association_kind": "legacy_unassociated"},
    )
    monkeypatch.setattr(
        projector,
        "load_model_era",
        lambda *_args: pytest.fail("legacy-unassociated report must not load an era"),
    )

    snapshot = _snapshot(tmp_path)

    assert snapshot.counts()["model_era_count"] == 0
    report_row = snapshot.rows_for("monitoring_report")[0].value_map()
    assert report_row["model_era_id"] is None


def test_valid_empty_store_produces_ready_snapshot_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(monkeypatch)
    monkeypatch.setattr(projector, "load_monitoring_report_state", lambda _root: None)
    monkeypatch.setattr(projector, "load_active_alerts", lambda _root: {})
    monkeypatch.setattr(projector, "load_alert_history", lambda _root: [])
    monkeypatch.setattr(projector, "load_reporting_attempts", lambda _root: [])

    snapshot = _snapshot(tmp_path)

    assert snapshot.counts()["evidence_record_count"] == 0
    assert snapshot.rows == ()
    assert len(snapshot.generation_id) == 64


def test_scan_does_not_modify_controlled_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(monkeypatch)
    sentinel = tmp_path / "reporting" / "state" / "current.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("immutable-sentinel", encoding="utf-8")
    before = (sha256(sentinel.read_bytes()).hexdigest(), sentinel.stat().st_mtime_ns)

    _snapshot(tmp_path)

    after = (sha256(sentinel.read_bytes()).hexdigest(), sentinel.stat().st_mtime_ns)
    assert after == before
    assert not list(tmp_path.rglob("*.lock"))


def test_data_minimization_rejects_path_like_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _patch_sources(monkeypatch)
    values["attempts"] = [
        {**_attempt(), "source_pipeline_run_id": "C:\\private\\source-run"}
    ]

    with pytest.raises(projector.ProjectionSourceError):
        _snapshot(tmp_path)


def test_source_git_commit_rejects_tracked_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        [
            SimpleNamespace(stdout=GIT_COMMIT),
            SimpleNamespace(stdout=" M src/private.py\n"),
        ]
    )
    monkeypatch.setattr(projector.subprocess, "run", lambda *_a, **_k: next(results))

    with pytest.raises(projector.ProjectionProvenanceError):
        projector.resolve_source_git_commit(Path("."))


def test_cli_failure_does_not_expose_raw_database_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "postgresql://writer:do-not-print@example.test/database"
    monkeypatch.setattr(
        cli,
        "load_operational_projection_database_config",
        lambda _role: SimpleNamespace(dsn=secret, environment_id="local"),
    )
    monkeypatch.setattr(
        cli,
        "load_monitoring_store_config",
        lambda: SimpleNamespace(store_root=Path("monitoring")),
    )
    monkeypatch.setattr(cli, "resolve_source_git_commit", lambda: GIT_COMMIT)
    monkeypatch.setattr(
        cli,
        "project_projection",
        lambda *_a, **_k: (_ for _ in ()).throw(
            projector.ProjectionDatabaseError(f"raw failure for {secret}")
        ),
    )

    assert cli.main(["project"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error_code"] == "database_unavailable"
    assert secret not in captured.err
    assert "raw failure" not in captured.err


def test_projection_imports_do_not_import_psycopg() -> None:
    code = (
        "import sys; "
        "import wind_forecast.operational_projection_cli; "
        "import wind_forecast.operational_projection_projector; "
        "assert 'psycopg' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
