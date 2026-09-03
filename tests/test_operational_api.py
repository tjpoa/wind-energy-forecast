from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from wind_forecast.api import create_app
from wind_forecast.config import (
    DEPLOYMENT_ROOT_ENV,
    MLFLOW_TRACKING_URI_ENV,
    MONITORING_STORE_ROOT_ENV,
    OPERATIONAL_CALIBRATION_DIR_ENV,
    OPERATIONAL_ENVIRONMENT_ID_ENV,
    OPERATIONAL_MODEL_BUNDLE_ENV,
    OPERATIONAL_OBSERVABILITY_ROOT_ENV,
    OPERATIONAL_PROJECTION_MODE_ENV,
    OPERATIONAL_PROJECTION_READER_DSN_ENV,
    OPERATIONAL_QUERY_TIMEOUT_ENV,
    load_operational_query_config,
)
from wind_forecast.operational_api import (
    _LocalOnlyMlflowRegistryClient,
    MAX_OPERATIONAL_QUERY_BODY_BYTES,
    get_operational_query_service,
    get_operational_query_service_factory,
)
from wind_forecast.operational_observability import (
    OBSERVABILITY_EVENTS_FILENAME,
    get_operational_observability,
)
from wind_forecast.operational_query import OperationalQueryService
from wind_forecast.operational_query_models import (
    AnswerStatus,
    EvidenceCitation,
    EvidenceDomain,
    EvidenceState,
    GroundedFact,
    OperationalAnswer,
    OperationalFailure,
    QueryKind,
)


NOW = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
REPORT_ID = "a" * 64
RUN_ID = "20260731T120000000000Z-abcdef123456"


@pytest.fixture(autouse=True)
def _explicit_operational_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        OPERATIONAL_MODEL_BUNDLE_ENV,
        "outputs/training/v2_reference_mlflow",
    )
    monkeypatch.setenv(
        OPERATIONAL_CALIBRATION_DIR_ENV,
        "data/processed/v2/monitoring/reporting/calibrations/test-calibration",
    )
    monkeypatch.setenv(OPERATIONAL_OBSERVABILITY_ROOT_ENV, str(tmp_path))
    get_operational_observability.cache_clear()
    yield
    get_operational_observability.cache_clear()


def _payload(query_kind: str, **overrides) -> dict:
    payload = {
        "contract_version": "operational_read_only_copilot_v1",
        "query_kind": query_kind,
        "selector": {"kind": "latest"},
        "window_days": None,
        "pagination": None,
    }
    payload.update(overrides)
    return payload


def _answer(
    status: AnswerStatus,
    *,
    query_kind: QueryKind = QueryKind.DATA_QUALITY,
    correlation_id: str = "server-correlation",
) -> OperationalAnswer:
    if status == AnswerStatus.ANSWERED:
        return OperationalAnswer(
            query_kind=query_kind,
            status=status,
            summary="Verified fact [e1].",
            facts=(
                GroundedFact(
                    fact_id="f1",
                    name="quality.status",
                    value="ok",
                    unit_or_scale="status",
                    as_of="2026-07-30",
                    evidence_ids=("e1",),
                ),
            ),
            evidence=(
                EvidenceCitation(
                    evidence_id="e1",
                    domain=EvidenceDomain.MONITORING_REPORT,
                    source_kind="load_monitoring_report",
                    schema_version="wind_forecast.monitoring_report.v2",
                    record_id=REPORT_ID,
                    sha256=REPORT_ID,
                    effective_at="2026-07-30",
                ),
            ),
            limitations=("Historical batch evidence is not real-time.",),
            failure=None,
            served_at_utc=NOW,
            correlation_id=correlation_id,
        )
    failure_state = {
        AnswerStatus.REFUSED: EvidenceState.UNSUPPORTED,
        AnswerStatus.UNAUTHORIZED: EvidenceState.UNAUTHORIZED,
        AnswerStatus.UNAVAILABLE: EvidenceState.UNAVAILABLE,
        AnswerStatus.CORRUPT: EvidenceState.CORRUPT,
        AnswerStatus.CONFLICT: EvidenceState.CONFLICT,
        AnswerStatus.TIMEOUT: EvidenceState.TIMEOUT,
    }.get(status)
    return OperationalAnswer(
        query_kind=query_kind,
        status=status,
        summary=None,
        facts=(),
        evidence=(),
        limitations=(),
        failure=(
            None
            if failure_state is None
            else OperationalFailure(
                code=f"{status.value}_test",
                message="Sanitized operational response.",
                retryable=status in {AnswerStatus.UNAVAILABLE, AnswerStatus.TIMEOUT},
                evidence_state=failure_state,
            )
        ),
        served_at_utc=NOW,
        correlation_id=correlation_id,
    )


class RecordingService:
    max_deadline_seconds = 5.0

    def __init__(self, status: AnswerStatus = AnswerStatus.EMPTY):
        self.status = status
        self.calls = []

    def answer(self, query, authorization):
        self.calls.append((query, authorization))
        status = self.status
        if not authorization.trusted_local:
            status = AnswerStatus.UNAUTHORIZED
        query_kind = (
            query.query_kind
            if hasattr(query, "query_kind")
            else QueryKind(query["query_kind"])
        )
        correlation_id = (
            query.correlation_id
            if hasattr(query, "correlation_id")
            else query["correlation_id"]
        )
        return _answer(
            status,
            query_kind=query_kind,
            correlation_id=correlation_id,
        )


def _client(service, *, host: str = "127.0.0.1") -> TestClient:
    app = create_app()
    app.dependency_overrides[get_operational_query_service_factory] = lambda: (
        lambda: service
    )
    return TestClient(app, client=(host, 50000))


def test_copilot_executes_one_canonical_operational_query() -> None:
    service = RecordingService(AnswerStatus.ANSWERED)

    response = _client(service).post(
        "/api/v1/copilot",
        json={"question": "Como estão a qualidade e a freshness dos dados?"},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "operational"
    assert response.json()["mode"] == "guided_local"
    assert response.json()["answer"]["status"] == "answered"
    assert len(service.calls) == 1
    assert service.calls[0][0].query_kind == QueryKind.DATA_QUALITY


def test_copilot_documentary_route_never_executes_operational_service() -> None:
    service = RecordingService()
    response = _client(service).post(
        "/api/v1/copilot",
        json={"question": "Qual é a metodologia e quais são as limitações?"},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["route"] == "documentary"
    assert payload["mode"] == "rag_local"
    assert payload["failure"] is None
    assert payload["answer"]["evidence"][0]["uri"].startswith("docs://")
    assert service.calls == []


def test_copilot_documentary_no_match_is_refused_without_operational_query(
    monkeypatch,
) -> None:
    from wind_forecast import operational_api

    service = RecordingService()
    monkeypatch.setattr(
        operational_api,
        "get_document_corpus",
        lambda: (),
    )

    response = _client(service).post(
        "/api/v1/copilot",
        json={"question": "Qual é a metodologia?"},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "refused"
    assert response.json()["failure"]["code"] == "documentary_no_match"
    assert service.calls == []


def test_copilot_refuses_unsupported_question_without_operational_query() -> None:
    service = RecordingService()

    response = _client(service).post(
        "/api/v1/copilot",
        json={"question": "Conta uma anedota"},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "refused"
    assert response.json()["failure"]["code"] == "unsupported_question"
    assert service.calls == []


def test_copilot_forecast_replay_refusal_is_visible_and_does_not_query() -> None:
    service = RecordingService()

    response = _client(service).post(
        "/api/v1/copilot",
        json={"question": "Mostra previsões históricas"},
    )

    assert response.json()["route"] == "refused"
    assert response.json()["failure"]["code"] == "forecast_replay_required"
    assert service.calls == []


def test_copilot_rejects_non_loopback_before_operational_query() -> None:
    service = RecordingService()

    response = _client(service, host="192.0.2.1").post(
        "/api/v1/copilot",
        json={"question": "Que deployment está ativo?"},
    )

    assert response.json()["route"] == "operational"
    assert response.json()["answer"]["status"] == "unauthorized"
    assert service.calls == []


@pytest.mark.parametrize("question", ("", " ", "x" * 1001))
def test_copilot_http_question_is_bounded(question: str) -> None:
    service = RecordingService()

    response = _client(service).post(
        "/api/v1/copilot",
        json={"question": question},
    )

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    "payload",
    (
        _payload("operational_summary"),
        _payload("active_deployment"),
        _payload("data_quality"),
        _payload("monitoring_performance", window_days=30),
        _payload("monitoring_drift", window_days=90),
        _payload(
            "monitoring_alerts",
            pagination={"limit": 50, "offset": 0},
        ),
        _payload("active_model_metadata"),
        _payload(
            "reporting_run",
            selector={
                "kind": "exact_id",
                "id_type": "reporting_run_id",
                "identifier": RUN_ID,
            },
        ),
    ),
)
def test_endpoint_accepts_exactly_the_eight_typed_query_kinds(payload):
    service = RecordingService()

    response = _client(service).post("/api/v1/operational-query", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "empty"
    assert len(service.calls) == 1


def test_endpoint_adds_server_controlled_metadata_and_five_second_deadline():
    service = RecordingService()

    response = _client(service).post(
        "/api/v1/operational-query",
        json=_payload("data_quality"),
    )

    query, authorization = service.calls[0]
    assert response.status_code == 200
    assert query["correlation_id"] == response.json()["correlation_id"]
    assert len(query["correlation_id"]) == 32
    assert (
        query["deadline"] - query["requested_at_utc"]
    ).total_seconds() == pytest.approx(5.0)
    assert authorization.principal == "local-api-operator"
    assert authorization.trusted_local is True


def test_endpoint_records_correlated_request_and_tool_spans(tmp_path: Path):
    service = RecordingService()

    response = _client(service).post(
        "/api/v1/operational-query",
        json=_payload("data_quality"),
    )

    events = [
        json.loads(line)
        for line in (tmp_path / OBSERVABILITY_EVENTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert response.status_code == 200
    assert [event["event_type"] for event in events] == [
        "request.started",
        "tool.started",
        "tool.finished",
        "request.finished",
    ]
    assert all(event["query_kind"] == "data_quality" for event in events)
    assert len({event["correlation_id"] for event in events}) == 1
    assert len({event["trace_id"] for event in events}) == 1
    request_span = events[0]["span_id"]
    assert events[1]["parent_span_id"] == request_span
    assert events[2]["parent_span_id"] == request_span
    assert events[1]["tool_name"] == "operational_query"
    assert events[2]["answer_status"] == "empty"
    assert events[3]["answer_status"] == "empty"
    assert events[3]["http_status"] == 200


def test_all_eight_query_kinds_are_observed(tmp_path: Path):
    service = RecordingService()
    payloads = (
        _payload("operational_summary"),
        _payload("active_deployment"),
        _payload("data_quality"),
        _payload("monitoring_performance", window_days=30),
        _payload("monitoring_drift", window_days=90),
        _payload(
            "monitoring_alerts",
            pagination={"limit": 50, "offset": 0},
        ),
        _payload("active_model_metadata"),
        _payload(
            "reporting_run",
            selector={
                "kind": "exact_id",
                "id_type": "reporting_run_id",
                "identifier": RUN_ID,
            },
        ),
    )

    for payload in payloads:
        response = _client(service).post(
            "/api/v1/operational-query",
            json=payload,
        )
        assert response.status_code == 200

    events = [
        json.loads(line)
        for line in (tmp_path / OBSERVABILITY_EVENTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(events) == 32
    assert {event["query_kind"] for event in events} == {
        "operational_summary",
        "active_deployment",
        "data_quality",
        "monitoring_performance",
        "monitoring_drift",
        "monitoring_alerts",
        "active_model_metadata",
        "reporting_run",
    }


def test_writer_failure_does_not_change_operational_query_response(
    monkeypatch,
    tmp_path: Path,
):
    blocked_root = tmp_path / "blocked"
    blocked_root.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(OPERATIONAL_OBSERVABILITY_ROOT_ENV, str(blocked_root))
    get_operational_observability.cache_clear()
    service = RecordingService()

    try:
        response = _client(service).post(
            "/api/v1/operational-query",
            json=_payload("data_quality"),
        )
        assert response.status_code == 200
        assert response.json()["status"] == "empty"

        health = _client(service).get("/api/v1/operational-observability/health")
        metrics = _client(service).get("/api/v1/operational-observability/metrics")
        assert health.status_code == 503
        assert health.json()["status"] == "degraded"
        assert metrics.json()["dropped_events"] == 4
    finally:
        get_operational_observability.cache_clear()


@pytest.mark.parametrize(
    "server_field",
    ("correlation_id", "requested_at_utc", "deadline", "principal"),
)
def test_client_cannot_supply_server_controlled_fields(server_field):
    service = RecordingService()
    payload = _payload("data_quality")
    payload[server_field] = "attacker-controlled"

    response = _client(service).post("/api/v1/operational-query", json=payload)

    assert response.status_code == 400
    assert response.json()["status"] == "refused"
    assert service.calls == []


@pytest.mark.parametrize(
    ("status", "expected_http"),
    (
        (AnswerStatus.ANSWERED, 200),
        (AnswerStatus.EMPTY, 200),
        (AnswerStatus.REFUSED, 400),
        (AnswerStatus.UNAUTHORIZED, 403),
        (AnswerStatus.NOT_FOUND, 404),
        (AnswerStatus.UNAVAILABLE, 503),
        (AnswerStatus.CORRUPT, 503),
        (AnswerStatus.CONFLICT, 503),
        (AnswerStatus.TIMEOUT, 504),
    ),
)
def test_domain_status_maps_to_required_http_status(status, expected_http):
    response = _client(RecordingService(status)).post(
        "/api/v1/operational-query",
        json=_payload("data_quality"),
    )

    assert response.status_code == expected_http
    assert response.json()["status"] == status.value


def test_answer_provenance_is_returned_without_an_alternate_wrapper():
    response = _client(RecordingService(AnswerStatus.ANSWERED)).post(
        "/api/v1/operational-query",
        json=_payload("data_quality"),
    )

    body = response.json()
    assert set(body) == {
        "contract_version",
        "query_kind",
        "status",
        "mode",
        "summary",
        "facts",
        "evidence",
        "limitations",
        "failure",
        "served_at_utc",
        "correlation_id",
    }
    assert body["summary"] == "Verified fact [e1]."
    assert body["facts"][0]["evidence_ids"] == ["e1"]
    assert body["evidence"][0]["sha256"] == REPORT_ID


@pytest.mark.parametrize(
    ("content", "content_type", "expected_code"),
    (
        (b"{", "application/json", "invalid_json_body"),
        (b"[]", "application/json", "invalid_operational_query"),
        (b"{}", "text/plain", "invalid_content_type"),
    ),
)
def test_incompatible_bodies_are_refused_without_service_reads(
    content,
    content_type,
    expected_code,
):
    service = RecordingService()

    response = _client(service).post(
        "/api/v1/operational-query",
        content=content,
        headers={"content-type": content_type},
    )

    assert response.status_code == 400
    assert response.json()["failure"]["code"] == expected_code
    assert service.calls == []


def test_body_larger_than_64_kib_is_refused_before_json_parsing():
    service = RecordingService()
    oversized = b"{" + b" " * MAX_OPERATIONAL_QUERY_BODY_BYTES + b"}"

    response = _client(service).post(
        "/api/v1/operational-query",
        content=oversized,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["failure"]["code"] == "operational_query_body_too_large"
    assert service.calls == []


def test_deeply_nested_json_is_sanitized_as_an_invalid_body():
    service = RecordingService()
    nested = b"[" * 1200 + b"0" + b"]" * 1200

    response = _client(service).post(
        "/api/v1/operational-query",
        content=nested,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["status"] == "refused"
    assert response.json()["failure"]["code"] == "invalid_json_body"
    assert service.calls == []


@pytest.mark.parametrize(
    "payload",
    (
        _payload("predict"),
        _payload(
            "data_quality",
            selector={
                "kind": "exact_id",
                "id_type": "report_id",
                "identifier": "../private/report.json",
            },
        ),
        _payload(
            "monitoring_alerts",
            selector={
                "kind": "date_interval",
                "start_date": "2026-07-31",
                "end_date": "2026-07-30",
            },
        ),
        _payload(
            "monitoring_alerts",
            pagination={"limit": 201, "offset": 0},
        ),
    ),
)
def test_unsupported_or_invalid_queries_are_refused_before_service_reads(payload):
    service = RecordingService()

    response = _client(service).post("/api/v1/operational-query", json=payload)

    assert response.status_code == 400
    assert response.json()["status"] == "refused"
    assert service.calls == []


def test_selector_combination_is_refused_by_query_layer_before_reads(tmp_path):
    service = OperationalQueryService(
        deployment_root=tmp_path / "deployment",
        monitoring_store_root=tmp_path / "monitoring",
        max_deadline_seconds=5.0,
        authorization_policy=lambda _context, _kind: True,
    )

    response = _client(service).post(
        "/api/v1/operational-query",
        json=_payload(
            "active_deployment",
            selector={
                "kind": "exact_id",
                "id_type": "report_id",
                "identifier": REPORT_ID,
            },
        ),
    )

    assert response.status_code == 400
    assert response.json()["status"] == "refused"
    assert response.json()["failure"]["code"] == "invalid_operational_query"


def test_query_layer_authorization_prevents_dispatch_for_remote_socket(
    tmp_path,
    monkeypatch,
):
    service = OperationalQueryService(
        deployment_root=tmp_path / "deployment",
        monitoring_store_root=tmp_path / "monitoring",
        max_deadline_seconds=5.0,
        authorization_policy=lambda context, _kind: context.trusted_local,
    )

    def forbidden_dispatch(_self, _query):
        raise AssertionError("unauthorized request reached operational reads")

    monkeypatch.setattr(OperationalQueryService, "_dispatch", forbidden_dispatch)

    response = _client(service, host="192.0.2.1").post(
        "/api/v1/operational-query",
        json=_payload("data_quality"),
    )

    assert response.status_code == 403
    assert response.json()["status"] == "unauthorized"


@pytest.mark.parametrize("host", ("192.0.2.1", "localhost", "testclient"))
def test_non_loopback_or_nonnumeric_socket_is_unauthorized(host):
    service = RecordingService()

    response = _client(service, host=host).post(
        "/api/v1/operational-query",
        json=_payload("data_quality"),
        headers={
            "forwarded": "for=127.0.0.1",
            "x-forwarded-for": "127.0.0.1",
            "x-real-ip": "127.0.0.1",
        },
    )

    assert response.status_code == 403
    assert response.json()["status"] == "unauthorized"
    assert service.calls[0][1].trusted_local is False


@pytest.mark.parametrize("host", ("127.0.0.1", "::1"))
def test_exact_ipv4_and_ipv6_loopback_are_trusted_and_proxy_headers_ignored(host):
    service = RecordingService()

    response = _client(service, host=host).post(
        "/api/v1/operational-query",
        json=_payload("data_quality"),
        headers={
            "forwarded": "for=192.0.2.1",
            "x-forwarded-for": "192.0.2.1",
            "x-real-ip": "192.0.2.1",
        },
    )

    assert response.status_code == 200
    assert service.calls[0][1].trusted_local is True


def test_absent_request_client_is_not_trusted():
    from wind_forecast.operational_api import _trusted_loopback

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "client": None,
        }
    )

    assert _trusted_loopback(request) is False


def test_unexpected_service_error_is_sanitized():
    class BrokenService:
        max_deadline_seconds = 5.0

        def answer(self, _query, _authorization):
            raise RuntimeError(r"C:\private\secret.json token=do-not-return connection")

    response = _client(BrokenService()).post(
        "/api/v1/operational-query",
        json=_payload("data_quality"),
    )

    serialized = response.text.lower()
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert "private" not in serialized
    assert "token" not in serialized


def test_service_factory_failure_is_sanitized():
    def broken_factory():
        raise ValueError(r"C:\private\config.json token=do-not-return connection")

    app = create_app()
    app.dependency_overrides[get_operational_query_service_factory] = lambda: (
        broken_factory
    )

    response = TestClient(app, client=("127.0.0.1", 50000)).post(
        "/api/v1/operational-query",
        json=_payload("data_quality"),
    )

    serialized = response.text.lower()
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert "private" not in serialized
    assert "token" not in serialized


def test_openapi_documents_request_answer_and_all_response_codes():
    operation = create_app().openapi()["paths"]["/api/v1/operational-query"]["post"]

    assert operation["requestBody"]["required"] is True
    assert (
        operation["requestBody"]["content"]["application/json"]["schema"]["properties"][
            "contract_version"
        ]["const"]
        == "operational_read_only_copilot_v1"
    )
    assert set(operation["responses"]) == {
        "200",
        "400",
        "403",
        "404",
        "503",
        "504",
    }


def test_operational_config_uses_local_defaults(monkeypatch):
    for name in (
        DEPLOYMENT_ROOT_ENV,
        MONITORING_STORE_ROOT_ENV,
        OPERATIONAL_QUERY_TIMEOUT_ENV,
        OPERATIONAL_PROJECTION_MODE_ENV,
        OPERATIONAL_PROJECTION_READER_DSN_ENV,
        MLFLOW_TRACKING_URI_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_operational_query_config()

    assert config.deployment_root.as_posix().endswith("data/processed/v2/deployment")
    assert config.monitoring_store_root.as_posix().endswith(
        "data/processed/v2/monitoring"
    )
    assert config.model_bundle.as_posix().endswith(
        "outputs/training/v2_reference_mlflow"
    )
    assert config.calibration_dir.as_posix().endswith(
        "data/processed/v2/monitoring/reporting/calibrations/test-calibration"
    )
    assert config.timeout_seconds == 5.0
    assert config.registry_uri == "http://127.0.0.1:5000"
    assert config.projection_mode == "disabled"
    assert config.projection_environment_id is None
    assert config.projection_reader_dsn is None


@pytest.mark.parametrize(
    "variable",
    (OPERATIONAL_MODEL_BUNDLE_ENV, OPERATIONAL_CALIBRATION_DIR_ENV),
)
def test_operational_config_requires_explicit_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    monkeypatch.delenv(variable)

    with pytest.raises(ValueError, match=variable):
        load_operational_query_config()


@pytest.mark.parametrize(
    ("variable", "configured_value"),
    (
        (OPERATIONAL_MODEL_BUNDLE_ENV, ""),
        (OPERATIONAL_MODEL_BUNDLE_ENV, "   "),
        (OPERATIONAL_CALIBRATION_DIR_ENV, ""),
        (OPERATIONAL_CALIBRATION_DIR_ENV, "   "),
    ),
)
def test_operational_config_rejects_blank_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    configured_value: str,
) -> None:
    monkeypatch.setenv(variable, configured_value)

    with pytest.raises(ValueError, match=variable):
        load_operational_query_config()


def test_disabled_projection_does_not_read_reader_dsn(monkeypatch) -> None:
    import wind_forecast.config as runtime_config

    original_getenv = runtime_config.os.getenv

    def guarded_getenv(name, default=None):
        if name == OPERATIONAL_PROJECTION_READER_DSN_ENV:
            raise AssertionError("disabled mode read the PostgreSQL DSN")
        return original_getenv(name, default)

    monkeypatch.delenv(OPERATIONAL_PROJECTION_MODE_ENV, raising=False)
    monkeypatch.setattr(runtime_config.os, "getenv", guarded_getenv)

    assert load_operational_query_config().projection_mode == "disabled"


@pytest.mark.parametrize("mode", ("prefer", "REQUIRED", "", " required "))
def test_operational_config_rejects_unsupported_projection_mode(
    monkeypatch,
    mode,
) -> None:
    monkeypatch.setenv(OPERATIONAL_PROJECTION_MODE_ENV, mode)

    with pytest.raises(ValueError, match=OPERATIONAL_PROJECTION_MODE_ENV):
        load_operational_query_config()


def test_required_projection_loads_only_reader_configuration(monkeypatch) -> None:
    monkeypatch.setenv(OPERATIONAL_PROJECTION_MODE_ENV, "required")
    monkeypatch.setenv(OPERATIONAL_ENVIRONMENT_ID_ENV, "local")
    monkeypatch.setenv(
        OPERATIONAL_PROJECTION_READER_DSN_ENV,
        "postgresql://reader:secret@127.0.0.1/projection",
    )

    config = load_operational_query_config()

    assert config.projection_mode == "required"
    assert config.projection_environment_id == "local"
    assert config.projection_reader_dsn == (
        "postgresql://reader:secret@127.0.0.1/projection"
    )


@pytest.mark.parametrize(
    "uri",
    (
        "http://localhost:5000",
        "http://192.0.2.1:5000",
        "file:///tmp/mlruns",
        "sqlite:///mlflow.db",
        "http://user:password@127.0.0.1:5000",
    ),
)
def test_operational_config_disables_non_numeric_or_non_rest_registry(
    monkeypatch,
    uri,
):
    monkeypatch.setenv(MLFLOW_TRACKING_URI_ENV, uri)

    assert load_operational_query_config().registry_uri is None


@pytest.mark.parametrize("uri", ("http://127.0.0.1:5000", "https://[::1]:5000"))
def test_operational_config_accepts_only_exact_loopback_rest_registry(
    monkeypatch,
    uri,
):
    monkeypatch.setenv(MLFLOW_TRACKING_URI_ENV, uri)

    assert load_operational_query_config().registry_uri == uri


@pytest.mark.parametrize("timeout", ("0", "-1", "5.1", "nan", "infinity", "bad"))
def test_operational_config_rejects_invalid_or_unbounded_timeout(
    monkeypatch,
    timeout,
):
    monkeypatch.setenv(OPERATIONAL_QUERY_TIMEOUT_ENV, timeout)

    with pytest.raises(ValueError, match="no greater than 5"):
        load_operational_query_config()


def test_service_factory_uses_configured_paths_timeout_and_registry_gate(
    tmp_path: Path,
    monkeypatch,
):
    import wind_forecast.operational_api as operational_api

    fake_registry = object()
    config = SimpleNamespace(
        deployment_root=tmp_path / "deployment",
        monitoring_store_root=tmp_path / "monitoring",
        model_bundle=tmp_path / "model-bundle",
        calibration_dir=tmp_path / "calibration",
        timeout_seconds=4.0,
        registry_uri="http://127.0.0.1:5000",
    )
    monkeypatch.setattr(
        operational_api,
        "load_operational_query_config",
        lambda: config,
    )
    monkeypatch.setattr(
        operational_api,
        "_registry_client",
        lambda uri: fake_registry if uri == config.registry_uri else None,
    )
    operational_api.get_operational_query_service.cache_clear()
    try:
        service = operational_api.get_operational_query_service()
    finally:
        operational_api.get_operational_query_service.cache_clear()

    assert service.deployment_root == config.deployment_root
    assert service.monitoring_store_root == config.monitoring_store_root
    assert service.model_bundle == config.model_bundle
    assert service.calibration_dir == config.calibration_dir
    assert service.max_deadline_seconds == 4.0
    assert service.registry_client is fake_registry
    assert service.registry_timeout_seconds == 4.0


def test_required_projection_configuration_fails_closed_per_query_kind(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import wind_forecast.operational_api as operational_api
    from wind_forecast.operational_projection_reader import (
        UnavailableOperationalProjectionReader,
    )

    config = SimpleNamespace(
        deployment_root=tmp_path / "deployment",
        monitoring_store_root=tmp_path / "monitoring",
        model_bundle=tmp_path / "model-bundle",
        calibration_dir=tmp_path / "calibration",
        timeout_seconds=4.0,
        registry_uri=None,
        projection_mode="required",
        projection_environment_id="not-local",
        projection_reader_dsn=None,
    )
    monkeypatch.setattr(
        operational_api,
        "load_operational_query_config",
        lambda: config,
    )
    operational_api.get_operational_query_service.cache_clear()
    try:
        service = operational_api.get_operational_query_service()
    finally:
        operational_api.get_operational_query_service.cache_clear()

    assert isinstance(
        service.projection_reader,
        UnavailableOperationalProjectionReader,
    )


def test_disabled_service_factory_does_not_import_postgres_driver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import builtins
    import wind_forecast.operational_api as operational_api

    config = SimpleNamespace(
        deployment_root=tmp_path / "deployment",
        monitoring_store_root=tmp_path / "monitoring",
        model_bundle=tmp_path / "model-bundle",
        calibration_dir=tmp_path / "calibration",
        timeout_seconds=4.0,
        registry_uri=None,
        projection_mode="disabled",
        projection_environment_id=None,
        projection_reader_dsn=None,
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "psycopg" or name.startswith("psycopg."):
            raise AssertionError("disabled mode imported psycopg")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(
        operational_api,
        "load_operational_query_config",
        lambda: config,
    )
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    operational_api.get_operational_query_service.cache_clear()
    try:
        service = operational_api.get_operational_query_service()
    finally:
        operational_api.get_operational_query_service.cache_clear()

    assert service.projection_reader is None


def test_local_registry_adapter_disables_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient
    from mlflow.utils import rest_utils

    calls: list[dict] = []

    def fake_http_request(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            status_code=200,
            reason="OK",
            text=json.dumps(
                {
                    "model_version": {
                        "name": "wind-v2",
                        "version": "11",
                    }
                }
            ),
        )

    monkeypatch.setattr(rest_utils, "http_request", fake_http_request)
    client = MlflowClient(
        tracking_uri="http://127.0.0.1:5000",
        registry_uri="http://127.0.0.1:5000",
    )
    adapter = _LocalOnlyMlflowRegistryClient(client)

    version = adapter.get_model_version_by_alias(
        "wind-v2",
        "champion",
        timeout_seconds=2.5,
    )

    assert version.name == "wind-v2"
    assert version.version == "11"
    assert len(calls) == 1
    assert calls[0]["method"] == "GET"
    assert calls[0]["timeout"] == 2.5
    assert calls[0]["retry_timeout_seconds"] == 2.5
    assert calls[0]["max_retries"] == 0
    assert calls[0]["allow_redirects"] is False
    assert calls[0]["proxies"] == {
        "http": None,
        "https": None,
        "all": None,
    }


def test_import_and_app_creation_do_not_create_configured_stores(
    tmp_path: Path,
    monkeypatch,
):
    deployment = tmp_path / "deployment"
    monitoring = tmp_path / "monitoring"
    monkeypatch.setenv(DEPLOYMENT_ROOT_ENV, str(deployment))
    monkeypatch.setenv(MONITORING_STORE_ROOT_ENV, str(monitoring))
    get_operational_query_service.cache_clear()
    try:
        create_app()
    finally:
        get_operational_query_service.cache_clear()

    assert not deployment.exists()
    assert not monitoring.exists()
