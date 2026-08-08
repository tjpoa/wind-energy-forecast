from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

import wind_forecast.operational_query as operational
import wind_forecast.monitoring_reporting as monitoring_reporting
from wind_forecast.deployment_runtime import DeploymentRuntimeConflictError
from wind_forecast.operational_query import OperationalQueryService, TARGET_SCALE
from wind_forecast.operational_query_models import (
    AnswerStatus,
    AuthorizationContext,
    EvidenceState,
    OperationalQuery,
    QueryKind,
)
from wind_forecast.operational_projection_reader import (
    OperationalProjectionTimeoutError,
    OperationalProjectionUnavailableError,
    ProjectedAlerts,
    ProjectedEvidence,
    ProjectedReport,
    ProjectedRow,
)


NOW = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
REPORT_ID = "a" * 64
ALERT_ID = "b" * 64
CALIBRATION_ID = "c" * 64
RUN_ID = "20260730T115900000000Z-abcdef123456"


def _era() -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_model_era.v1",
        "model_era_id": "e" * 64,
        "association_kind": "active_deployment",
        "deployment": {
            "deployment_id": "d" * 64,
            "deployment_state_id": "f" * 64,
            "generation": 7,
            "pointer_sha256": "1" * 64,
            "state_manifest_sha256": "2" * 64,
            "authorizing_receipt_sha256": "3" * 64,
        },
        "registry": {
            "registered_model_name": "wind-v2",
            "model_version": "11",
            "run_id": "registry-run",
            "model_uri": "models:/wind-v2/11",
        },
        "expected_aliases": {
            "candidate": None,
            "champion": "11",
            "stable": "11",
        },
        "cutoffs": {
            "fit_cutoff": "2025-12-31",
            "activation_cutoff": "2026-07-01",
        },
        "pins": {
            "bundle_sha256": "4" * 64,
            "model_sha256": "5" * 64,
            "dataset_sha256": "6" * 64,
            "feature_schema_sha256": "7" * 64,
            "calibration_sha256": "8" * 64,
            "ledger_sha256": "9" * 64,
        },
        "calibration": {
            "calibration_id": CALIBRATION_ID,
            "reference_id": "0" * 64,
        },
        "monitoring": {"ledger_model_snapshot_id": "1" * 64},
        "_runtime_metadata": {
            "model_type": "RandomForestRegressor",
            "dataset_version": "v2",
            "transformation_version": "transform-v2",
        },
    }


def _report() -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_report.v2",
        "report_id": REPORT_ID,
        "run_id": RUN_ID,
        "created_at_utc": "2026-07-30T11:59:00Z",
        "through_date": "2026-07-29",
        "source_batch": {"run_id": "source-run", "status": "succeeded"},
        "reference": {
            "calibration_id": CALIBRATION_ID,
            "reference_id": "0" * 64,
            "policy_sha256": "1" * 64,
        },
        "quality": {
            "status": "available",
            "issues": [{"code": "late_source", "severity": "warning"}],
            "freshness": {
                "common_validated_watermark": "2026-07-29",
                "unresolved_late_dates": [],
            },
        },
        "windows": {
            "30": {
                "status": "available",
                "sample_count": 30,
                "coverage_ratio": 1.0,
                "performance": {
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
            "90": {"status": "insufficient_data"},
        },
        "active_alerts": {"feature_drift:wind_speed:30:global": ALERT_ID},
        "alert_events": [ALERT_ID],
        "persistence": {},
        "lineage": {"prediction_ids": []},
    }


def _calibration() -> dict:
    metric_limits = {
        name: {"warning": 20.0, "critical": 30.0, "direction": "upper"}
        for name in ("MAE", "RMSE", "absolute_bias", "MAPE_percent", "R2")
    }
    metric_limits["R2"]["direction"] = "lower"
    return {
        "schema_version": "wind_forecast.monitoring_calibration.v1",
        "calibration_id": CALIBRATION_ID,
        "reference_id": "0" * 64,
        "policy_sha256": "1" * 64,
        "thresholds": {
            "performance": {"30": metric_limits},
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
    }


def _alert() -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_alert_event.v2",
        "alert_event_id": ALERT_ID,
        "rule_id": "feature_drift:wind_speed:30:global",
        "through_date": "2026-07-29",
        "event_type": "opened",
        "severity": "warning",
        "previous_alert_event_id": None,
    }


def _attempt() -> dict:
    return {
        "run_id": RUN_ID,
        "attempted_at_utc": "2026-07-30T11:59:00Z",
        "through_date": "2026-07-29",
        "source_pipeline_run_id": "source-run",
        "source_pipeline_status": "succeeded",
        "status": "succeeded",
        "report_id": REPORT_ID,
        "active_alert_count": 1,
        "failure": None,
    }


def _service(**overrides) -> OperationalQueryService:
    values = {
        "deployment_root": Path("deployment"),
        "monitoring_store_root": Path("monitoring"),
        "max_deadline_seconds": 300.0,
        "authorization_policy": lambda context, _kind: context.trusted_local,
        "registry_client": object(),
        "registry_timeout_seconds": 10.0,
        "clock": lambda: NOW,
    }
    values.update(overrides)
    return OperationalQueryService(**values)


def _query(
    query_kind: str,
    selector: dict | None = None,
    **overrides,
) -> dict:
    payload = {
        "contract_version": "operational_read_only_copilot_v1",
        "query_kind": query_kind,
        "selector": selector or {"kind": "latest"},
        "window_days": None,
        "pagination": None,
        "requested_at_utc": NOW - timedelta(seconds=30),
        "correlation_id": "query-1",
        "deadline": NOW + timedelta(seconds=30),
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def verified_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(operational, "verify_active_model_era", lambda *_a, **_k: _era())
    monkeypatch.setattr(
        operational,
        "load_monitoring_report_state",
        lambda _root: {
            "schema_version": "wind_forecast.monitoring_report_state.v2",
            "latest_report_id": REPORT_ID,
            "latest_through_date": "2026-07-29",
            "active_alerts": {
                "feature_drift:wind_speed:30:global": ALERT_ID
            },
        },
    )
    monkeypatch.setattr(operational, "load_monitoring_report", lambda _path: _report())
    monkeypatch.setattr(
        operational, "load_monitoring_calibration", lambda _path: _calibration()
    )
    monkeypatch.setattr(
        operational,
        "resolve_report_model_era",
        lambda _root, _report_value: {
            "association_kind": "active_deployment",
            "model_era_id": "e" * 64,
        },
    )
    monkeypatch.setattr(operational, "load_alert_history", lambda _root: [_alert()])
    monkeypatch.setattr(
        operational,
        "load_active_alerts",
        lambda _root: {"feature_drift:wind_speed:30:global": ALERT_ID},
    )
    monkeypatch.setattr(
        operational, "load_reporting_attempt", lambda _root, **_kwargs: _attempt()
    )


@pytest.mark.parametrize(
    ("query_kind", "selector", "extra", "expected_fact"),
    (
        ("operational_summary", {"kind": "latest"}, {}, "monitoring.latest_report_id"),
        ("active_deployment", {"kind": "latest"}, {}, "deployment.deployment_id"),
        ("data_quality", {"kind": "latest"}, {}, "data_quality.freshness"),
        (
            "monitoring_performance",
            {"kind": "latest"},
            {"window_days": 30},
            "monitoring.performance.mae",
        ),
        (
            "monitoring_drift",
            {"kind": "latest"},
            {"window_days": 30},
            "monitoring.drift.1",
        ),
        (
            "monitoring_alerts",
            {"kind": "latest"},
            {},
            "monitoring.alert.1",
        ),
        (
            "active_model_metadata",
            {"kind": "latest"},
            {},
            "model.transformation_version",
        ),
        (
            "reporting_run",
            {
                "kind": "exact_id",
                "id_type": "reporting_run_id",
                "identifier": RUN_ID,
            },
            {},
            "reporting_run.status",
        ),
    ),
)
def test_closed_allowlist_returns_cited_deterministic_answers(
    query_kind: str,
    selector: dict,
    extra: dict,
    expected_fact: str,
) -> None:
    request = _query(query_kind, selector, **extra)

    first = _service().answer(
        request, AuthorizationContext(principal="operator", trusted_local=True)
    )
    second = _service().answer(
        request, AuthorizationContext(principal="operator", trusted_local=True)
    )

    assert first.status == AnswerStatus.ANSWERED
    assert expected_fact in {fact.name for fact in first.facts}
    assert first.model_dump(exclude={"served_at_utc"}) == second.model_dump(
        exclude={"served_at_utc"}
    )
    assert first.evidence
    for fact in first.facts:
        assert fact.evidence_ids
        assert all(f"[{item}]" in first.summary for item in fact.evidence_ids)
    assert all(
        citation.sha256 == citation.sha256.lower()
        and len(citation.sha256) == 64
        for citation in first.evidence
    )
    if query_kind == "operational_summary":
        latest = next(
            fact for fact in first.facts
            if fact.name == "monitoring.latest_report_id"
        )
        source_kinds = {
            citation.source_kind
            for citation in first.evidence
            if citation.evidence_id in latest.evidence_ids
        }
        assert source_kinds == {
            "load_monitoring_report",
            "load_monitoring_report_state",
        }
    assert "models:/" not in first.model_dump_json()


def test_unknown_and_invalid_queries_are_refused_without_operational_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("operational read occurred")

    monkeypatch.setattr(operational, "verify_active_model_era", forbidden)
    service = _service()
    unknown = service.answer(
        _query("future_forecast"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )
    invalid = service.answer(
        _query(
            "monitoring_performance",
            window_days=30,
            selector={
                "kind": "exact_id",
                "id_type": "report_id",
                "identifier": "A" * 64,
            },
        ),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert unknown.status == AnswerStatus.REFUSED
    assert unknown.query_kind is None
    assert invalid.status == AnswerStatus.REFUSED
    assert invalid.query_kind == QueryKind.MONITORING_PERFORMANCE
    assert calls == 0


@pytest.mark.parametrize("value", (None, 42, "query", ["active_deployment"]))
def test_non_mapping_request_is_refused_without_raising(value: object) -> None:
    answer = _service().answer(
        value,
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.REFUSED
    assert answer.query_kind is None


def test_default_deny_authorization_precedes_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational,
        "verify_active_model_era",
        lambda *_a, **_k: pytest.fail("operational read occurred"),
    )

    answer = _service(authorization_policy=None).answer(
        _query("active_deployment"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.UNAUTHORIZED
    assert answer.failure.evidence_state == EvidenceState.UNAUTHORIZED
    assert not answer.facts


def test_non_local_context_is_denied_even_by_permissive_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational,
        "verify_active_model_era",
        lambda *_a, **_k: pytest.fail("operational read occurred"),
    )

    answer = _service(
        authorization_policy=lambda _context, _kind: True
    ).answer(
        _query("active_deployment"),
        AuthorizationContext(principal="operator", trusted_local=False),
    )

    assert answer.status == AnswerStatus.UNAUTHORIZED


def test_expired_deadline_returns_timeout_without_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational,
        "verify_active_model_era",
        lambda *_a, **_k: pytest.fail("operational read occurred"),
    )

    answer = _service().answer(
        _query(
            "active_deployment",
            requested_at_utc=NOW - timedelta(seconds=60),
            deadline=NOW - timedelta(seconds=1),
        ),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.TIMEOUT
    assert answer.failure.retryable is True


def test_deployment_query_requires_timeout_bounded_registry_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational,
        "verify_active_model_era",
        lambda *_a, **_k: pytest.fail("verification must not create a client"),
    )

    answer = _service(
        registry_client=None,
        registry_timeout_seconds=None,
    ).answer(
        _query("active_deployment"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.UNAVAILABLE
    assert answer.failure.code == "required_dependency_unavailable"


def test_deadline_expiring_between_verified_loaders_returns_timeout() -> None:
    calls = 0

    def advancing_clock() -> datetime:
        nonlocal calls
        value = NOW + timedelta(seconds=20 * calls)
        calls += 1
        return value

    answer = _service(clock=advancing_clock).answer(
        _query(
            "data_quality",
            requested_at_utc=NOW - timedelta(seconds=1),
            deadline=NOW + timedelta(seconds=30),
        ),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.TIMEOUT
    assert not answer.facts


def test_changed_model_era_during_summary_is_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eras = [_era(), {**_era(), "model_era_id": "f" * 64}]
    monkeypatch.setattr(
        operational, "verify_active_model_era", lambda *_a, **_k: eras.pop(0)
    )

    answer = _service().answer(
        _query("operational_summary"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CONFLICT
    assert answer.failure.evidence_state == EvidenceState.CONFLICT
    assert not answer.facts


def test_operational_summary_without_report_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational, "load_monitoring_report_state", lambda _root: None
    )

    answer = _service().answer(
        _query("operational_summary"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.EMPTY
    assert answer.summary is None
    assert not answer.facts
    assert not answer.evidence


def test_registry_client_requires_finite_timeout_and_propagates_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="finite positive timeout"):
        _service(registry_client=object(), registry_timeout_seconds=None)

    observed: list[float] = []

    def verify(*_args, **kwargs):
        observed.append(kwargs["registry_timeout_seconds"])
        return _era()

    monkeypatch.setattr(operational, "verify_active_model_era", verify)
    answer = _service(
        registry_client=object(),
        registry_timeout_seconds=60.0,
    ).answer(
        _query("active_deployment"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )
    assert answer.status == AnswerStatus.ANSWERED
    assert observed == [30.0, 30.0]


@pytest.mark.parametrize(
    "query_kind", ("active_deployment", "active_model_metadata")
)
def test_deployment_queries_detect_model_era_change(
    query_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eras = [_era(), {**_era(), "model_era_id": "f" * 64}]
    monkeypatch.setattr(
        operational, "verify_active_model_era", lambda *_a, **_k: eras.pop(0)
    )

    answer = _service().answer(
        _query(query_kind),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CONFLICT
    assert not answer.facts


def test_active_deployment_returns_complete_checksum_pins() -> None:
    answer = _service().answer(
        _query("active_deployment"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    fact = next(
        item for item in answer.facts
        if item.name == "deployment.checksum_pins"
    )
    assert set(fact.value) == {
        "pointer_sha256",
        "state_manifest_sha256",
        "authorizing_receipt_sha256",
        "bundle_sha256",
        "model_sha256",
        "dataset_sha256",
        "feature_schema_sha256",
        "calibration_sha256",
        "ledger_sha256",
    }
    assert fact.evidence_ids


def test_operational_summary_rejects_active_alert_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operational, "load_active_alerts", lambda _root: {})

    answer = _service().answer(
        _query("operational_summary"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CONFLICT
    assert not answer.facts


def test_report_without_requested_window_is_empty() -> None:
    answer = _service().answer(
        _query("monitoring_performance", window_days=90),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.EMPTY
    assert not answer.facts


def test_alert_interval_is_inclusive_and_paginated() -> None:
    answer = _service().answer(
        _query(
            "monitoring_alerts",
            selector={
                "kind": "date_interval",
                "start_date": date(2026, 7, 29),
                "end_date": date(2026, 7, 29),
            },
            pagination={"limit": 50, "offset": 0},
        ),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.ANSWERED
    assert answer.facts[0].value["alert_event_id"] == ALERT_ID
    assert answer.facts[0].value["active"] is True
    assert {
        citation.source_kind for citation in answer.evidence
    } == {"load_alert_history", "load_active_alerts"}
    active_citation = next(
        item for item in answer.evidence if item.source_kind == "load_active_alerts"
    )
    assert active_citation.observed_at_utc == NOW
    assert active_citation.evidence_id in answer.facts[0].evidence_ids


@pytest.mark.parametrize(
    "selector",
    (
        {"kind": "latest"},
        {
            "kind": "exact_id",
            "id_type": "alert_event_id",
            "identifier": ALERT_ID,
        },
        {
            "kind": "date_interval",
            "start_date": "2026-07-29",
            "end_date": "2026-07-29",
        },
    ),
)
def test_alert_state_change_is_conflict_for_every_selector(
    monkeypatch: pytest.MonkeyPatch, selector: dict
) -> None:
    states = [
        {"feature_drift:wind_speed:30:global": ALERT_ID},
        {},
    ]
    monkeypatch.setattr(
        operational, "load_active_alerts", lambda _root: states.pop(0)
    )

    answer = _service().answer(
        _query("monitoring_alerts", selector=selector),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CONFLICT


def test_alert_collection_defaults_to_fifty_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alerts = []
    active = {}
    for index in range(60):
        alert_id = sha256(f"alert-{index}".encode()).hexdigest()
        alerts.append(
            {
                **_alert(),
                "alert_event_id": alert_id,
                "rule_id": f"quality:rule-{index}",
            }
        )
        active[f"quality:rule-{index}"] = alert_id
    monkeypatch.setattr(operational, "load_alert_history", lambda _root: alerts)
    monkeypatch.setattr(operational, "load_active_alerts", lambda _root: active)

    answer = _service().answer(
        _query("monitoring_alerts"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.ANSWERED
    assert len(answer.facts) == 50


def test_reporting_attempt_without_report_is_empty_for_data_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational,
        "load_reporting_attempt",
        lambda _root, **_kwargs: {**_attempt(), "status": "in_progress", "report_id": None},
    )

    answer = _service().answer(
        _query(
            "data_quality",
            selector={
                "kind": "exact_id",
                "id_type": "reporting_run_id",
                "identifier": RUN_ID,
            },
        ),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.EMPTY


def test_missing_report_file_is_unavailable_not_uninitialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational,
        "load_monitoring_report",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing report")),
    )

    answer = _service().answer(
        _query("data_quality"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.UNAVAILABLE
    assert answer.failure.evidence_state == EvidenceState.UNAVAILABLE


def test_broken_latest_report_reference_is_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational,
        "load_monitoring_report",
        monitoring_reporting.load_monitoring_report,
    )

    answer = _service(monitoring_store_root=tmp_path).answer(
        _query("data_quality"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CORRUPT
    assert answer.failure.evidence_state == EvidenceState.CORRUPT


def test_reporting_read_permission_failure_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational,
        "load_monitoring_report",
        monitoring_reporting.load_monitoring_report,
    )
    original_read_text = Path.read_text

    def denied(path: Path, *args, **kwargs):
        if path.name == "report.json":
            raise PermissionError("denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)

    answer = _service(monitoring_store_root=tmp_path).answer(
        _query("data_quality"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.UNAVAILABLE
    assert answer.failure.evidence_state == EvidenceState.UNAVAILABLE


def test_exact_selectors_return_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(monitoring_store_root=tmp_path)
    missing_alert = "f" * 64
    alert = service.answer(
        _query(
            "monitoring_alerts",
            selector={
                "kind": "exact_id",
                "id_type": "alert_event_id",
                "identifier": missing_alert,
            },
        ),
        AuthorizationContext(principal="operator", trusted_local=True),
    )
    report = service.answer(
        _query(
            "data_quality",
            selector={
                "kind": "exact_id",
                "id_type": "report_id",
                "identifier": "f" * 64,
            },
        ),
        AuthorizationContext(principal="operator", trusted_local=True),
    )
    monkeypatch.setattr(
        operational, "load_reporting_attempt", lambda _root, **_kwargs: None
    )
    run = service.answer(
        _query(
            "reporting_run",
            selector={
                "kind": "exact_id",
                "id_type": "reporting_run_id",
                "identifier": RUN_ID,
            },
        ),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert alert.status == AnswerStatus.NOT_FOUND
    assert report.status == AnswerStatus.NOT_FOUND
    assert run.status == AnswerStatus.NOT_FOUND


def test_runtime_conflict_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operational,
        "verify_active_model_era",
        lambda *_a, **_k: (_ for _ in ()).throw(
            DeploymentRuntimeConflictError("C:\\private\\secret.json differs")
        ),
    )

    answer = _service().answer(
        _query("active_deployment"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CONFLICT
    assert "private" not in answer.model_dump_json().lower()
    assert TARGET_SCALE not in answer.model_dump_json()


def test_verified_but_non_public_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = _era()
    unsafe["registry"] = {
        **unsafe["registry"],
        "registered_model_name": "C:\\private\\registered-model",
    }
    monkeypatch.setattr(
        operational, "verify_active_model_era", lambda *_a, **_k: unsafe
    )

    answer = _service().answer(
        _query("active_deployment"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CORRUPT
    assert "private" not in answer.model_dump_json().lower()


def test_executable_models_are_frozen_and_forbid_unknown_fields() -> None:
    query = OperationalQuery.model_validate(_query("active_deployment"), strict=True)
    with pytest.raises(ValidationError):
        OperationalQuery.model_validate(
            {**_query("active_deployment"), "unexpected": True}, strict=True
        )
    with pytest.raises(ValidationError):
        query.correlation_id = "changed"


def test_explicit_iso_utc_and_calendar_date_parsing() -> None:
    request = _query(
        "monitoring_alerts",
        selector={
            "kind": "date_interval",
            "start_date": "2026-07-29",
            "end_date": "2026-07-29",
        },
        requested_at_utc="2026-07-30T11:59:30Z",
        deadline="2026-07-30T12:00:30+00:00",
    )

    answer = _service().answer(
        request,
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.ANSWERED


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("requested_at_utc", "2026-07-30T12:59:30+01:00"),
        ("deadline", "2026-07-30T13:00:30+01:00"),
        ("requested_at_utc", "2026-07-30 11:59:30"),
    ),
)
def test_non_utc_or_implicit_timestamp_strings_are_refused(
    field: str, value: str
) -> None:
    answer = _service().answer(
        _query("active_deployment", **{field: value}),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.REFUSED


def test_reporting_query_does_not_modify_verified_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "reporting" / "runs" / RUN_ID
    run_root.mkdir(parents=True)
    request = {
        "schema_version": "wind_forecast.monitoring_report_request.v2",
        "run_id": RUN_ID,
        "requested_at_utc": "2026-07-30T11:59:00Z",
        "plan": {
            "status": "planned",
            "through_date": "2026-07-29",
            "source_run_id": "source-run",
            "source_status": "failed",
            "calibration_id": CALIBRATION_ID,
        },
    }
    (run_root / "request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    (run_root / "failure.json").write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.monitoring_report_failure.v1",
                "run_id": RUN_ID,
                "failed_at_utc": "2026-07-30T12:00:00Z",
                "error_type": "ExpectedFailure",
                "error": "private detail",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        operational,
        "load_reporting_attempt",
        monitoring_reporting.load_reporting_attempt,
    )

    def snapshot() -> dict[str, tuple[bytes, int, int]]:
        return {
            path.relative_to(tmp_path).as_posix(): (
                path.read_bytes(),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in sorted(tmp_path.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    answer = _service(monitoring_store_root=tmp_path).answer(
        _query(
            "reporting_run",
            selector={
                "kind": "exact_id",
                "id_type": "reporting_run_id",
                "identifier": RUN_ID,
            },
        ),
        AuthorizationContext(principal="operator", trusted_local=True),
    )
    after = snapshot()

    assert answer.status == AnswerStatus.ANSWERED
    assert before == after


@pytest.mark.parametrize(
    "payload",
    (
        _query("operational_summary"),
        _query("active_deployment"),
        _query("data_quality"),
        _query("monitoring_performance", window_days=30),
        _query("monitoring_drift", window_days=30),
        _query("monitoring_alerts"),
        _query("active_model_metadata"),
        _query(
            "reporting_run",
            selector={
                "kind": "exact_id",
                "id_type": "reporting_run_id",
                "identifier": RUN_ID,
            },
        ),
    ),
)
def test_every_query_kind_preserves_operational_store_bytes_and_metadata(
    payload: dict,
    tmp_path: Path,
) -> None:
    deployment_root = tmp_path / "deployment"
    monitoring_root = tmp_path / "monitoring"
    for root, name in (
        (deployment_root, "deployment-state.json"),
        (monitoring_root, "monitoring-state.json"),
    ):
        root.mkdir()
        (root / name).write_text(f"{name}\n", encoding="utf-8")

    def snapshot() -> dict[str, tuple[bytes, int, int]]:
        return {
            path.relative_to(tmp_path).as_posix(): (
                path.read_bytes(),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in sorted(tmp_path.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    answer = _service(
        deployment_root=deployment_root,
        monitoring_store_root=monitoring_root,
    ).answer(
        payload,
        AuthorizationContext(principal="operator", trusted_local=True),
    )
    after = snapshot()

    assert answer.status == AnswerStatus.ANSWERED
    assert before == after


class _MatchingProjectionReader:
    def __init__(self, calibration: dict) -> None:
        self.calibration = calibration
        self.report_override: ProjectedRow | None = None

    def select_report(self, **kwargs) -> ProjectedReport:
        detail = kwargs["detail"]
        window_days = kwargs["window_days"]
        report = _report()
        normalized_report = self.report_override or operational._normalized_report(
            report,
            "e" * 64,
        )
        if detail == "quality":
            return ProjectedReport(
                normalized_report,
                quality_issues=operational._normalized_quality_issues(report),
            )
        return ProjectedReport(
            normalized_report,
            window=operational._normalized_window(report, window_days),
            performance_metrics=(
                operational._normalized_performance_metrics(
                    report,
                    self.calibration,
                    window_days,
                )
                if detail == "performance"
                else ()
            ),
            drift_measurements=(
                operational._normalized_drift_measurements(
                    report,
                    self.calibration,
                    window_days,
                )
                if detail == "drift"
                else ()
            ),
            calibration=operational._normalized_calibration_evidence(
                self.calibration
            ),
            model_era=operational._normalized_model_era(_era()),
        )

    def select_attempt(self, **_kwargs) -> ProjectedRow:
        return operational._normalized_attempt(_attempt())

    def select_alerts(self, **_kwargs) -> ProjectedAlerts:
        active = {"feature_drift:wind_speed:30:global": ALERT_ID}
        digest = operational._digest(active)
        return ProjectedAlerts(
            history=(operational._normalized_alert(_alert()),),
            active=active,
            active_evidence=ProjectedEvidence(
                "alert",
                "load_active_alerts",
                "wind_forecast.verified_active_alert_binding.v1",
                digest,
                digest,
                "2026-07-29",
            ),
            selected_ids=(ALERT_ID,),
        )


def _projection_calibration() -> dict:
    calibration = _calibration()
    calibration["_reference_manifest"] = {
        "schema_version": "wind_forecast.monitoring_reference.v1",
        "reference_id": "0" * 64,
        "period": {"start": "2025-01-01", "end": "2026-07-29"},
    }
    return calibration


@pytest.mark.parametrize(
    "payload",
    (
        _query("data_quality"),
        _query("monitoring_performance", window_days=30),
        _query("monitoring_drift", window_days=30),
        _query("monitoring_alerts"),
        _query(
            "reporting_run",
            selector={
                "kind": "exact_id",
                "id_type": "reporting_run_id",
                "identifier": RUN_ID,
            },
        ),
    ),
)
def test_required_projection_preserves_filesystem_answers(
    payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _projection_calibration()
    monkeypatch.setattr(
        operational,
        "load_monitoring_calibration",
        lambda _path: calibration,
    )
    monkeypatch.setattr(
        operational,
        "load_model_era",
        lambda _root, _model_era_id: _era(),
    )
    monkeypatch.setattr(
        operational,
        "verify_active_model_era",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("projected queries called deployment or MLflow")
        ),
    )
    context = AuthorizationContext(principal="operator", trusted_local=True)

    filesystem = _service().answer(payload, context)
    projected = _service(
        projection_reader=_MatchingProjectionReader(calibration)
    ).answer(payload, context)

    assert projected == filesystem
    assert all(
        "postgres" not in citation.source_kind.lower()
        for citation in projected.evidence
    )


def test_required_projection_divergence_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _projection_calibration()
    projection = _MatchingProjectionReader(calibration)
    expected = operational._normalized_report(_report(), "e" * 64)
    projection.report_override = ProjectedRow(
        {**expected.values, "verdict": "FAIL"},
        expected.evidence,
    )
    monkeypatch.setattr(
        operational,
        "load_monitoring_calibration",
        lambda _path: calibration,
    )

    answer = _service(projection_reader=projection).answer(
        _query("data_quality"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.UNAVAILABLE
    assert answer.failure is not None
    assert answer.failure.code == "required_projection_unavailable"


def test_required_latest_empty_state_is_revalidated_after_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        None,
        {
            "schema_version": "wind_forecast.monitoring_report_state.v2",
            "latest_report_id": REPORT_ID,
            "latest_through_date": "2026-07-29",
            "active_alerts": {},
        },
    ]

    class EmptyProjectionReader:
        def select_report(self, **_kwargs):
            return None

    monkeypatch.setattr(
        operational,
        "load_monitoring_report_state",
        lambda _root: states.pop(0),
    )

    answer = _service(projection_reader=EmptyProjectionReader()).answer(
        _query("data_quality"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CONFLICT
    assert states == []


def test_required_latest_stably_empty_state_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyProjectionReader:
        def select_report(self, **_kwargs):
            return None

    monkeypatch.setattr(
        operational,
        "load_monitoring_report_state",
        lambda _root: None,
    )

    answer = _service(projection_reader=EmptyProjectionReader()).answer(
        _query("data_quality"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.EMPTY


def test_required_projection_preserves_authoritative_corrupt_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _projection_calibration()
    projection = _MatchingProjectionReader(calibration)
    monkeypatch.setattr(
        operational,
        "load_monitoring_report",
        lambda _path: (_ for _ in ()).throw(
            operational.MonitoringReportingError("corrupt")
        ),
    )

    answer = _service(projection_reader=projection).answer(
        _query("data_quality"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CORRUPT


def test_required_projection_preserves_authoritative_conflict_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _projection_calibration()
    projection = _MatchingProjectionReader(calibration)
    conflicting_report = {**_report(), "through_date": "2026-07-28"}
    monkeypatch.setattr(
        operational,
        "load_monitoring_report",
        lambda _path: conflicting_report,
    )

    answer = _service(projection_reader=projection).answer(
        _query("data_quality"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.CONFLICT


def test_projection_failure_affects_only_projected_query_kinds() -> None:
    class UnavailableReader:
        def select_report(self, **_kwargs):
            raise OperationalProjectionUnavailableError("secret")

    service = _service(projection_reader=UnavailableReader())
    context = AuthorizationContext(principal="operator", trusted_local=True)

    projected = service.answer(_query("data_quality"), context)
    direct = service.answer(_query("active_deployment"), context)

    assert projected.status == AnswerStatus.UNAVAILABLE
    assert direct.status == AnswerStatus.ANSWERED


def test_projection_statement_timeout_maps_to_operational_timeout() -> None:
    class TimedOutReader:
        def select_report(self, **_kwargs):
            raise OperationalProjectionTimeoutError("secret")

    answer = _service(projection_reader=TimedOutReader()).answer(
        _query("data_quality"),
        AuthorizationContext(principal="operator", trusted_local=True),
    )

    assert answer.status == AnswerStatus.TIMEOUT
    assert answer.failure is not None
    assert answer.failure.code == "operational_query_timeout"


def test_import_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    result = subprocess.run(
        [sys.executable, "-c", "import wind_forecast.operational_query"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
