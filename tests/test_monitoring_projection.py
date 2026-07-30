from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import wind_forecast.monitoring_projection as projection
from wind_forecast.api import create_app, get_monitoring_service
from wind_forecast.monitoring_projection import (
    MONITORING_MODE,
    MonitoringProjectionError,
    MonitoringProjectionService,
    MonitoringRunNotFoundError,
)


def _request(run_id: str = "report-run") -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_report_request.v1",
        "run_id": run_id,
        "requested_at_utc": "2026-04-07T11:00:00Z",
        "plan": {
            "status": "planned",
            "through_date": "2026-03-31",
            "source_run_id": "source-run",
            "source_status": "succeeded",
            "calibration_id": "calibration-id",
            "ledger_available": True,
            "quality_available": True,
        },
    }


def _report(run_id: str = "report-run") -> dict:
    return {
        "report_id": "report-id",
        "run_id": run_id,
        "created_at_utc": "2026-04-07T11:00:00Z",
        "through_date": "2026-03-31",
        "source_batch": {"run_id": "source-run", "status": "succeeded"},
        "reference": {
            "calibration_id": "calibration-id",
            "reference_id": "reference-id",
            "policy_sha256": "c" * 64,
        },
        "config": {"source_objective_days": 5, "source_late_days": 7},
        "quality": {
            "status": "available",
            "issues": [],
            "freshness": {
                "common_validated_watermark": "2026-03-31",
                "unresolved_late_dates": [],
            },
        },
        "windows": {
            "30": {
                "status": "available",
                "sample_count": 30,
                "calendar_start": "2026-03-02",
                "calendar_end": "2026-03-31",
                "coverage_ratio": 1.0,
                "coverage_severity": "ok",
                "performance": {
                    "status": "available",
                    "metrics": {
                        "MAE": 12.0,
                        "RMSE": 15.0,
                        "bias": -2.0,
                        "MAPE_percent": 4.0,
                        "R2": None,
                        "R2_status": "insufficient_data",
                    },
                    "severity": {
                        "MAE": "warning",
                        "RMSE": "ok",
                        "bias": "ok",
                        "MAPE_percent": "ok",
                        "R2": "not_available",
                    },
                },
                "feature_drift": {
                    "z": {
                        "global": {
                            "normalized_wasserstein": 0.5,
                            "ks_statistic": 0.3,
                            "severity": "critical",
                        },
                        "seasonal": {
                            "normalized_wasserstein": 0.1,
                            "ks_statistic": 0.1,
                            "severity": "ok",
                        },
                    },
                    "wind_direction_current": {
                        "global": {
                            "normalized_wasserstein": 0.2,
                            "ks_statistic": 0.2,
                            "severity": "warning",
                        },
                    },
                },
            },
            "90": {
                "status": "insufficient_data",
                "sample_count": 30,
                "minimum_samples": 45,
            },
        },
        "active_alerts": {"rule": "alert-id"},
        "lineage": {"prediction_ids": ["prediction-id"]},
    }


def _calibration() -> dict:
    limits = {
        "warning": 10.0,
        "critical": 20.0,
        "direction": "upper",
    }
    drift_limits = {
        comparator: {
            detector: {
                "warning": 0.1,
                "critical": 0.2,
                "direction": "upper",
            }
            for detector in ("normalized_wasserstein", "ks_statistic")
        }
        for comparator in ("global", "seasonal")
    }
    return {
        "calibration_id": "calibration-id",
        "reference_id": "reference-id",
        "policy_sha256": "c" * 64,
        "_reference_manifest": {
            "reference_id": "reference-id",
            "model_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "transformation_version": "transform-v2",
        },
        "thresholds": {
            "performance": {
                "30": {
                    "MAE": limits,
                    "RMSE": limits,
                    "absolute_bias": limits,
                    "MAPE_percent": limits,
                    "R2": {
                        "warning": 0.5,
                        "critical": 0.2,
                        "direction": "lower",
                    },
                },
                "90": {},
            },
            "feature_drift": {
                "z": {"30": drift_limits},
                "wind_direction_current": {"30": drift_limits},
            },
        }
    }


def test_empty_store_is_a_connected_empty_projection(tmp_path: Path) -> None:
    service = MonitoringProjectionService(tmp_path / "missing")

    latest = service.latest(
        now_utc=datetime(2026, 4, 1, tzinfo=timezone.utc)
    )
    history = service.history()

    assert latest["state"] == "empty"
    assert latest["mode"] == MONITORING_MODE
    assert latest["report"] is None
    assert history["state"] == "empty"
    assert history["runs"]["total"] == 0


def test_invalid_store_root_is_unavailable(tmp_path: Path) -> None:
    root = tmp_path / "monitoring.json"
    root.write_text("{}")

    with pytest.raises(MonitoringProjectionError):
        MonitoringProjectionService(root).latest()


def test_structurally_corrupt_state_is_sanitized_as_unavailable(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "reporting" / "state" / "current.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.monitoring_report_state.v1",
                "latest_report_id": "report-id",
                "latest_through_date": "2026-03-31",
                "rules": {},
                "active_alerts": [],
            }
        ),
        encoding="utf-8",
    )

    service = MonitoringProjectionService(tmp_path)
    with pytest.raises(MonitoringProjectionError) as captured:
        service.latest()

    assert "state" not in str(captured.value).lower()


def test_latest_projects_freshness_model_metrics_drift_and_alerts(
    tmp_path: Path, monkeypatch
) -> None:
    service = MonitoringProjectionService(tmp_path)
    monkeypatch.setattr(
        service.__class__,
        "_runs",
        lambda _self: [
            {
                "run_id": "report-run",
                "attempted_at_utc": "2026-04-07T11:00:00Z",
                "through_date": "2026-03-31",
                "source_pipeline_run_id": "source-run",
                "source_pipeline_status": "succeeded",
                "status": "succeeded",
                "report_id": "report-id",
                "active_alert_count": 1,
                "failure": None,
            }
        ],
    )
    monkeypatch.setattr(
        projection,
        "load_monitoring_report_state",
        lambda _root: {
            "latest_report_id": "report-id",
            "latest_through_date": "2026-03-31",
        },
    )
    monkeypatch.setattr(projection, "load_monitoring_report", lambda _path: _report())
    monkeypatch.setattr(projection, "load_monitoring_calibration", lambda _path: _calibration())
    monkeypatch.setattr(
        projection,
        "load_prediction_evidence",
        lambda _root, _prediction_id: {
            "model_snapshot": {
                "model_snapshot_id": "snapshot-id",
                "model": {
                    "model_sha256": "a" * 64,
                    "model_type": "RandomForestRegressor",
                    "reference_status": "selected_not_promoted",
                },
                "dataset": {
                    "dataset_version": "v2",
                    "dataset_sha256": "b" * 64,
                },
                "transformation": {"version": "transform-v2"},
            }
        },
    )
    alert = {
        "alert_event_id": "alert-id",
        "rule_id": "feature_drift:z:30:global",
        "through_date": "2026-03-31",
        "event_type": "opened",
        "severity": "critical",
        "previous_alert_event_id": None,
    }
    unrelated_alert = {
        **alert,
        "alert_event_id": "newer-alert-id",
        "rule_id": "performance:RMSE:30",
        "through_date": "2026-04-01",
    }
    monkeypatch.setattr(
        projection, "load_alert_history", lambda _root: [alert, unrelated_alert]
    )

    before_objective = service.latest(
        now_utc=datetime(2026, 4, 5, 10, 59, tzinfo=timezone.utc)
    )
    at_objective = service.latest(
        now_utc=datetime(2026, 4, 5, 11, 0, tzinfo=timezone.utc)
    )
    at_late = service.latest(
        now_utc=datetime(2026, 4, 7, 11, 0, tzinfo=timezone.utc)
    )

    report = before_objective["report"]
    assert report["freshness"]["status"] == "within_objective"
    assert at_objective["report"]["freshness"]["status"] == "behind_objective"
    assert at_late["report"]["freshness"]["status"] == "late"
    assert report["model"]["status"] == "selected_not_promoted"
    assert report["windows"]["30"]["performance"][0]["metric"] == "MAE"
    assert report["windows"]["30"]["performance"][-1]["status"] == "insufficient_data"
    assert [item["feature"] for item in report["windows"]["30"]["top_drift"]] == [
        "z",
        "wind_direction_current",
    ]
    assert report["active_alerts"][0]["event_type"] == "opened"
    assert [item["alert_event_id"] for item in report["active_alerts"]] == [
        "alert-id"
    ]


def test_latest_ignores_orphan_report_without_current_pointer(
    tmp_path: Path, monkeypatch
) -> None:
    service = MonitoringProjectionService(tmp_path)
    monkeypatch.setattr(
        service.__class__,
        "_runs",
        lambda _self: [
            {
                "run_id": "orphan-run",
                "attempted_at_utc": "2026-04-07T11:00:00Z",
                "through_date": "2026-03-31",
                "source_pipeline_run_id": "source-run",
                "source_pipeline_status": "succeeded",
                "status": "in_progress",
                "report_id": None,
                "active_alert_count": 0,
                "failure": None,
            }
        ],
    )
    monkeypatch.setattr(
        projection, "load_monitoring_report_state", lambda _root: None
    )

    result = service.latest(
        now_utc=datetime(2026, 4, 7, 12, tzinfo=timezone.utc)
    )

    assert result["state"] == "empty"
    assert result["latest_attempt"]["status"] == "in_progress"


def test_reporting_runs_delegate_to_verified_public_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = [
        {
            "run_id": "verified-run",
            "attempted_at_utc": "2026-07-30T12:00:00Z",
            "through_date": "2026-07-29",
            "source_pipeline_run_id": "source-run",
            "source_pipeline_status": "succeeded",
            "status": "in_progress",
            "report_id": None,
            "active_alert_count": 0,
            "failure": None,
        }
    ]
    monkeypatch.setattr(
        projection, "load_reporting_attempts", lambda _root: expected
    )

    assert MonitoringProjectionService(tmp_path)._runs() == expected


def test_freshness_is_unknown_without_verified_source_watermark() -> None:
    report = _report()
    report["quality"] = {"status": "not_available", "issues": []}

    projected = MonitoringProjectionService(Path("."))._freshness(
        report, datetime(2026, 4, 20, tzinfo=timezone.utc)
    )

    assert projected["status"] == "unknown"
    assert projected["watermark_date"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reference_id", "different-reference"),
        ("policy_sha256", "d" * 64),
    ),
)
def test_report_rejects_mismatched_calibration_identity(
    tmp_path: Path, monkeypatch, field: str, value: str
) -> None:
    service = MonitoringProjectionService(tmp_path)
    report = _report()
    report["reference"][field] = value
    monkeypatch.setattr(projection, "load_monitoring_report", lambda _path: report)
    monkeypatch.setattr(
        projection, "load_monitoring_calibration", lambda _path: _calibration()
    )

    with pytest.raises(projection.MonitoringReportingError):
        service._project_report(report, datetime(2026, 4, 1, tzinfo=timezone.utc))


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("dataset", "dataset_sha256", "d" * 64),
        ("transformation", "version", "different-transform"),
    ),
)
def test_report_rejects_mismatched_model_lineage(
    tmp_path: Path,
    monkeypatch,
    section: str,
    field: str,
    value: str,
) -> None:
    service = MonitoringProjectionService(tmp_path)
    report = _report()
    report["active_alerts"] = {}
    monkeypatch.setattr(
        projection, "load_monitoring_calibration", lambda _path: _calibration()
    )
    monkeypatch.setattr(projection, "load_alert_history", lambda _root: [])
    snapshot = {
        "model_snapshot_id": "snapshot-id",
        "model": {
            "model_sha256": "a" * 64,
            "model_type": "RandomForestRegressor",
            "reference_status": "selected_not_promoted",
        },
        "dataset": {
            "dataset_version": "v2",
            "dataset_sha256": "b" * 64,
        },
        "transformation": {"version": "transform-v2"},
    }
    snapshot[section][field] = value
    monkeypatch.setattr(
        projection,
        "load_prediction_evidence",
        lambda _root, _prediction_id: {"model_snapshot": snapshot},
    )

    with pytest.raises(projection.MonitoringReportingError):
        service._project_report(
            report, datetime(2026, 4, 1, tzinfo=timezone.utc)
        )


def test_top_drift_keeps_five_features_in_severity_ratio_order() -> None:
    features = {
        name: {
            "global": {
                "normalized_wasserstein": value,
                "ks_statistic": value / 2,
                "severity": "critical",
            }
        }
        for name, value in {
            "a": 0.05,
            "b": 0.15,
            "c": 0.25,
            "d": 0.35,
            "e": 0.45,
            "f": 0.55,
        }.items()
    }
    limits = {
        name: {
            "30": {
                "global": {
                    detector: {
                        "warning": 0.1,
                        "critical": 0.2,
                        "direction": "upper",
                    }
                    for detector in ("normalized_wasserstein", "ks_statistic")
                }
            }
        }
        for name in features
    }

    ranked = MonitoringProjectionService._top_drift(features, limits, "30")

    assert [item["feature"] for item in ranked] == ["f", "e", "d", "c", "b"]
    assert [item["severity"] for item in ranked[:4]] == ["critical"] * 4
    assert ranked[-1]["severity"] == "warning"


def test_run_history_is_newest_first_paginated_and_failure_is_sanitized(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "reporting" / "runs"
    failed = runs_root / "failed-run"
    failed.mkdir(parents=True)
    (failed / "request.json").write_text(json.dumps(_request("failed-run")))
    (failed / "failure.json").write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.monitoring_report_failure.v1",
                "run_id": "failed-run",
                "failed_at_utc": "2026-04-07T11:01:00Z",
                "error_type": "MonitoringReportingError",
                "error": "C:\\private\\calibration.json is corrupt",
            }
        )
    )
    older = runs_root / "older-run"
    older.mkdir()
    older_request = _request("older-run")
    older_request["requested_at_utc"] = "2026-04-06T11:00:00Z"
    (older / "request.json").write_text(json.dumps(older_request))

    history = MonitoringProjectionService(tmp_path).history(run_limit=1)

    assert history["runs"]["total"] == 2
    assert history["runs"]["items"][0]["run_id"] == "failed-run"
    assert history["runs"]["items"][0]["failure"]["message"] == (
        "The reporting attempt failed. Inspect local operator logs."
    )
    assert "private" not in json.dumps(history)


def test_history_applies_independent_run_and_alert_offsets(
    tmp_path: Path, monkeypatch
) -> None:
    service = MonitoringProjectionService(tmp_path)
    runs = [
        {
            "run_id": f"run-{index}",
            "attempted_at_utc": f"2026-04-0{index}T11:00:00Z",
            "through_date": f"2026-03-0{index}",
            "source_pipeline_run_id": f"source-{index}",
            "source_pipeline_status": "succeeded",
            "status": "in_progress",
            "report_id": None,
            "active_alert_count": 0,
            "failure": None,
        }
        for index in (3, 2, 1)
    ]
    alerts = [
        {
            "alert_event_id": f"alert-{index}",
            "rule_id": "feature_drift:x:30:global",
            "through_date": f"2026-03-0{index}",
            "event_type": event,
            "severity": severity,
            "previous_alert_event_id": previous,
        }
        for index, event, severity, previous in (
            (1, "opened", "warning", None),
            (2, "escalated", "critical", "alert-1"),
            (3, "resolved", "ok", "alert-2"),
        )
    ]
    monkeypatch.setattr(service.__class__, "_runs", lambda _self: runs)
    monkeypatch.setattr(projection, "load_alert_history", lambda _root: alerts)

    result = service.history(
        run_limit=1,
        run_offset=1,
        alert_limit=1,
        alert_offset=2,
    )

    assert result["runs"]["items"][0]["run_id"] == "run-2"
    assert result["alerts"]["items"][0]["event_type"] == "resolved"


def test_run_history_rejects_result_with_mismatched_plan(tmp_path: Path) -> None:
    run = tmp_path / "reporting" / "runs" / "report-run"
    run.mkdir(parents=True)
    request = _request()
    (run / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "run_id": "report-run",
                "report_id": "report-id",
                "active_alert_count": 0,
                "plan": {**request["plan"], "source_run_id": "different-run"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MonitoringProjectionError):
        MonitoringProjectionService(tmp_path).history()


def test_run_rejects_unknown_and_path_like_ids(tmp_path: Path) -> None:
    service = MonitoringProjectionService(tmp_path)
    for run_id in ("missing", "../secret", "nested/run"):
        with pytest.raises(MonitoringRunNotFoundError):
            service.run(run_id)


def test_api_contracts_and_error_mapping() -> None:
    class FakeService:
        def latest(self):
            return {
                "state": "empty",
                "mode": MONITORING_MODE,
                "served_at_utc": "2026-01-01T00:00:00Z",
                "message": "No reports.",
                "latest_attempt": None,
                "report": None,
            }

        def history(self, **kwargs):
            assert kwargs == {
                "run_limit": 2,
                "run_offset": 1,
                "alert_limit": 3,
                "alert_offset": 4,
            }
            return {
                "state": "empty",
                "mode": MONITORING_MODE,
                "runs": {"items": [], "total": 0, "limit": 2, "offset": 1},
                "alerts": {"items": [], "total": 0, "limit": 3, "offset": 4},
            }

        def run(self, run_id):
            if run_id == "missing":
                raise MonitoringRunNotFoundError(run_id)
            if run_id == "corrupt":
                raise MonitoringProjectionError("secret path")
            return {
                "state": "available",
                "mode": MONITORING_MODE,
                "run": {
                    "run_id": run_id,
                    "attempted_at_utc": "2026-01-01T00:00:00Z",
                    "through_date": "2025-12-25",
                    "source_pipeline_run_id": "source",
                    "source_pipeline_status": "succeeded",
                    "status": "in_progress",
                    "report_id": None,
                    "active_alert_count": 0,
                    "failure": None,
                },
                "report": None,
            }

    app = create_app()
    app.dependency_overrides[get_monitoring_service] = lambda: FakeService()
    client = TestClient(app)

    assert client.get("/api/v1/monitoring/latest").status_code == 200
    assert (
        client.get(
            "/api/v1/monitoring/history"
            "?run_limit=2&run_offset=1&alert_limit=3&alert_offset=4"
        ).status_code
        == 200
    )
    assert client.get("/api/v1/monitoring/history?run_limit=101").status_code == 422
    assert client.get("/api/v1/monitoring/runs/missing").status_code == 404
    response = client.get("/api/v1/monitoring/runs/corrupt")
    assert response.status_code == 503
    assert "secret" not in response.text
