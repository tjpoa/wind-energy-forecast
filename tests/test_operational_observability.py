from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient

from wind_forecast.api import create_app
from wind_forecast.config import (
    OPERATIONAL_OBSERVABILITY_ROOT_ENV,
    load_operational_observability_config,
)
from wind_forecast.operational_observability import (
    OBSERVABILITY_EVENTS_FILENAME,
    OBSERVABILITY_SCHEMA_VERSION,
    ObservabilityContext,
    OperationalObservability,
    get_operational_observability,
)


EVENT_FIELDS = {
    "schema_version",
    "event_type",
    "timestamp_utc",
    "correlation_id",
    "trace_id",
    "span_id",
    "parent_span_id",
    "query_kind",
    "tool_name",
    "result",
    "answer_status",
    "http_status",
    "duration_ms",
    "failure_code",
}


def _context(suffix: str = "1") -> ObservabilityContext:
    return ObservabilityContext(
        correlation_id=f"correlation-{suffix}",
        trace_id=f"trace-{suffix}",
        request_span_id=f"request-span-{suffix}",
    )


def _events(root: Path) -> list[dict]:
    path = root / OBSERVABILITY_EVENTS_FILENAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_event_contract_is_strict_correlated_and_sanitized(tmp_path: Path) -> None:
    observability = OperationalObservability(tmp_path)
    context = _context()

    observability.request_started(context, query_kind="data_quality")
    observability.tool_started(
        context,
        span_id="tool-span-1",
        query_kind="data_quality",
    )
    observability.tool_finished(
        context,
        span_id="tool-span-1",
        query_kind="data_quality",
        answer_status="unavailable",
        duration_ms=1.25,
        failure_code=r"C:\private\secret.json token=do-not-log",
    )
    observability.request_finished(
        context,
        query_kind="data_quality",
        answer_status="unavailable",
        http_status=503,
        duration_ms=2.5,
        failure_code="required_evidence_unavailable",
    )

    records = _events(tmp_path)
    assert len(records) == 4
    assert all(set(record) == EVENT_FIELDS for record in records)
    assert all(
        record["schema_version"] == OBSERVABILITY_SCHEMA_VERSION
        for record in records
    )
    assert all(
        datetime.fromisoformat(record["timestamp_utc"].replace("Z", "+00:00"))
        .astimezone(timezone.utc)
        .tzinfo
        == timezone.utc
        for record in records
    )
    assert {record["event_type"] for record in records} == {
        "request.started",
        "request.finished",
        "tool.started",
        "tool.finished",
    }
    assert all(record["correlation_id"] == "correlation-1" for record in records)
    assert all(record["trace_id"] == "trace-1" for record in records)
    assert records[1]["parent_span_id"] == "request-span-1"
    assert records[2]["parent_span_id"] == "request-span-1"
    assert records[2]["failure_code"] == "unspecified_failure"
    assert r"C:\private" not in json.dumps(records)
    assert "token=do-not-log" not in json.dumps(records)


def test_concurrent_events_are_complete_jsonl_records(tmp_path: Path) -> None:
    observability = OperationalObservability(tmp_path)

    def write_event(index: int) -> None:
        context = _context(str(index))
        observability.request_started(context, query_kind="data_quality")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_event, range(120)))

    records = _events(tmp_path)
    assert len(records) == 120
    assert {record["event_type"] for record in records} == {"request.started"}
    assert observability.metrics()["dropped_events"] == 0


def test_writer_failure_is_degraded_and_does_not_raise(tmp_path: Path) -> None:
    blocked_store = tmp_path / "blocked-store"
    blocked_store.write_text("not a directory", encoding="utf-8")
    observability = OperationalObservability(blocked_store)

    observability.request_started(_context(), query_kind="data_quality")

    assert observability.health() == {
        "status": "degraded",
        "dropped_events": 1,
    }
    assert observability.metrics()["event_counts"]["request.started"] == 1


def test_configuration_and_writer_are_lazy(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "observability"
    monkeypatch.setenv(OPERATIONAL_OBSERVABILITY_ROOT_ENV, str(root))
    get_operational_observability.cache_clear()
    try:
        assert load_operational_observability_config().store_root == root.resolve()
        assert not root.exists()
        get_operational_observability()
        assert not root.exists()
    finally:
        get_operational_observability.cache_clear()


def test_observability_endpoints_are_loopback_only_and_secret_free(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "observability"
    monkeypatch.setenv(OPERATIONAL_OBSERVABILITY_ROOT_ENV, str(root))
    get_operational_observability.cache_clear()
    try:
        app = create_app()
        openapi = app.openapi()["paths"]
        assert set(
            openapi["/api/v1/operational-observability/health"]["get"][
                "responses"
            ]
        ) == {"200", "403", "503"}
        assert set(
            openapi["/api/v1/operational-observability/metrics"]["get"][
                "responses"
            ]
        ) == {"200", "403"}
        loopback = TestClient(app, client=("127.0.0.1", 50000))
        remote = TestClient(app, client=("192.0.2.1", 50000))

        assert loopback.get(
            "/api/v1/operational-observability/health"
        ).json() == {"status": "ready", "dropped_events": 0}
        metrics = loopback.get(
            "/api/v1/operational-observability/metrics"
        )
        assert metrics.status_code == 200
        assert metrics.json()["event_counts"] == {
            "request.started": 0,
            "request.finished": 0,
            "tool.started": 0,
            "tool.finished": 0,
        }

        for path in (
            "/api/v1/operational-observability/health",
            "/api/v1/operational-observability/metrics",
        ):
            response = remote.get(
                path,
                headers={"x-forwarded-for": "127.0.0.1"},
            )
            assert response.status_code == 403
            assert response.json() == {"status": "unauthorized"}

        assert str(root) not in loopback.get(
            "/api/v1/operational-observability/metrics"
        ).text
    finally:
        get_operational_observability.cache_clear()
