from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import os
import subprocess
import sys
from typing import Any

import pytest

from wind_forecast.config import (
    OPERATIONAL_PROJECTION_MIGRATOR_DSN_ENV,
    OPERATIONAL_PROJECTION_READER_DSN_ENV,
)
from wind_forecast.operational_projection_migrations import (
    discover_migrations,
    migrate,
)
from wind_forecast.operational_projection_models import (
    CONTRACT_VERSION,
    PROJECTOR_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    canonical_sha256,
)
import wind_forecast.operational_projection_reader as reader


INTEGRATION_FLAG = "WIND_FORECAST_OPERATIONAL_PROJECTION_TEST_INTEGRATION"
ATTEMPT_SCHEMA = "wind_forecast.monitoring_reporting_attempt_projection.v1"
RUN_ID = "20260803T120000000000Z-abcdef123456"
OLD_RUN_ID = "20260802T120000000000Z-abcdef123456"
PARENT_ALERT_ID = "f" * 64
CHILD_ALERT_ID = "0" * 64
FAKE_SOURCE = {
    "domain": "reporting_run",
    "source_kind": "load_reporting_attempts",
    "schema_version": ATTEMPT_SCHEMA,
    "record_id": RUN_ID,
    "sha256": "b" * 64,
    "effective_at": "2026-08-03T12:00:00Z",
}


class _FakeCursor:
    def __init__(
        self,
        *,
        head: bool = True,
        advance: Any | None = None,
        migration_checksum_valid: bool = True,
        contract_valid: bool = True,
        source_set_valid: bool = True,
    ) -> None:
        self.head = head
        self.advance = advance
        self.migration_checksum_valid = migration_checksum_valid
        self.contract_valid = contract_valid
        self.source_set_valid = source_set_valid
        self.rows: list[tuple[Any, ...]] = []
        self.executions: list[tuple[str, tuple[Any, ...] | None]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self.executions.append((sql, params))
        if "to_regclass" in sql:
            self.rows = [("operational_projection.schema_migration",)]
        elif "FROM operational_projection.schema_migration" in sql:
            self.rows = [
                (
                    item.version,
                    item.name,
                    (
                        item.sha256
                        if self.migration_checksum_valid or index != 0
                        else "0" * 64
                    ),
                )
                for index, item in enumerate(discover_migrations())
            ]
        elif "FROM operational_projection.projection_head" in sql:
            self.rows = (
                [
                    (
                        "a" * 64,
                        "local",
                        CONTRACT_VERSION if self.contract_valid else "future-contract",
                        RELATIONAL_SCHEMA_VERSION,
                        PROJECTOR_VERSION,
                        datetime(2026, 8, 3, tzinfo=timezone.utc),
                        (
                            canonical_sha256([FAKE_SOURCE])
                            if self.source_set_valid
                            else "0" * 64
                        ),
                        1,
                        0,
                        0,
                    )
                ]
                if self.head
                else []
            )
        elif "FROM operational_projection.generation_evidence ge JOIN" in sql:
            self.rows = [tuple(FAKE_SOURCE.values())]
        elif "FROM operational_projection.reporting_attempt" in sql:
            self.rows = [
                (
                    RUN_ID,
                    datetime(2026, 8, 3, 12, tzinfo=timezone.utc),
                    datetime(2026, 8, 2).date(),
                    "source-run",
                    "succeeded",
                    "failed",
                    None,
                    0,
                    datetime(2026, 8, 3, 12, 1, tzinfo=timezone.utc),
                    "RuntimeError",
                    "sanitized",
                    "reporting_run",
                    "load_reporting_attempts",
                    ATTEMPT_SCHEMA,
                    RUN_ID,
                    "b" * 64,
                    "2026-08-03T12:00:00Z",
                )
            ]
        else:
            self.rows = [(None,)] if "set_config" in sql else []
        if self.advance is not None:
            self.advance()

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self.rows)


class _FakeConnection:
    def __init__(
        self,
        *,
        head: bool = True,
        advance: Any | None = None,
        migration_checksum_valid: bool = True,
        contract_valid: bool = True,
        source_set_valid: bool = True,
    ) -> None:
        self.cursor_value = _FakeCursor(
            head=head,
            advance=advance,
            migration_checksum_valid=migration_checksum_valid,
            contract_valid=contract_valid,
            source_set_valid=source_set_valid,
        )
        self.closed = False

    def transaction(self) -> Any:
        return nullcontext()

    def cursor(self) -> _FakeCursor:
        return self.cursor_value

    def close(self) -> None:
        self.closed = True


class _ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float = 0.02) -> None:
        self.value += seconds


def test_import_does_not_import_psycopg() -> None:
    code = (
        "import sys; "
        "import wind_forecast.operational_projection_reader; "
        "assert 'psycopg' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_reader_validates_schema_head_and_bounds_statement_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _FakeConnection()
    monkeypatch.setattr(reader, "_connect", lambda _dsn, _timeout: connection)

    projected = reader.OperationalProjectionReader("secret-dsn").select_attempt(
        id_type="reporting_run_id",
        identifier=RUN_ID,
        timeout_seconds=99.0,
    )

    assert projected is not None
    assert projected.values["reporting_run_id"] == RUN_ID
    assert projected.evidence.source_kind == "load_reporting_attempts"
    assert connection.closed is True
    assert any(
        params is not None
        and len(params) == 2
        and params[0].endswith("ms")
        and params[1] == params[0]
        for sql, params in connection.cursor_value.executions
        if "statement_timeout" in sql
    )


def test_reader_recomputes_decreasing_timeout_before_every_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ManualClock()
    connection = _FakeConnection(advance=lambda: clock.advance(0.02))
    monkeypatch.setattr(reader, "_connect", lambda _dsn, _budget: connection)

    projected = reader.OperationalProjectionReader(
        "secret-dsn",
        clock=clock,
    ).select_attempt(
        id_type="reporting_run_id",
        identifier=RUN_ID,
        timeout_seconds=2.0,
    )

    assert projected is not None
    timeouts = [
        int(params[0].removesuffix("ms"))
        for sql, params in connection.cursor_value.executions
        if "statement_timeout" in sql and params is not None
    ]
    assert len(timeouts) >= 5
    assert timeouts == sorted(timeouts, reverse=True)
    assert timeouts[-1] < timeouts[0] <= 2_000


def test_subsecond_connect_budget_expires_without_calling_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    clock = _ManualClock()
    budget = reader._DeadlineBudget.create(0.5, clock)
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subsecond budget attempted a libpq connection")
        ),
    )

    with pytest.raises(reader.OperationalProjectionTimeoutError):
        reader._connect("secret-dsn", budget)


def test_connect_timeout_is_sanitized_and_maps_to_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    clock = _ManualClock()
    budget = reader._DeadlineBudget.create(2.9, clock)

    def expire(*_args, **_kwargs):
        raise psycopg.errors.ConnectionTimeout(
            "secret-dsn password=do-not-return"
        )

    monkeypatch.setattr(psycopg, "connect", expire)

    with pytest.raises(
        reader.OperationalProjectionTimeoutError
    ) as captured:
        reader._connect("secret-dsn", budget)

    assert "secret" not in str(captured.value).lower()
    assert "password" not in str(captured.value).lower()


def test_wrapped_connect_timeout_maps_to_sanitized_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    clock = _ManualClock()
    budget = reader._DeadlineBudget.create(2.9, clock)

    def expire(*_args, **_kwargs):
        try:
            raise TimeoutError("secret-dsn password=do-not-return")
        except TimeoutError as exc:
            raise psycopg.OperationalError("driver connection failed") from exc

    monkeypatch.setattr(psycopg, "connect", expire)

    with pytest.raises(
        reader.OperationalProjectionTimeoutError
    ) as captured:
        reader._connect("secret-dsn", budget)

    assert "secret" not in str(captured.value).lower()
    assert "password" not in str(captured.value).lower()


def test_connection_uses_integer_timeout_not_exceeding_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    clock = _ManualClock()
    budget = reader._DeadlineBudget.create(2.9, clock)
    connection = _FakeConnection()
    captured: dict[str, Any] = {}

    def connect(*_args, **kwargs):
        captured.update(kwargs)
        return connection

    monkeypatch.setattr(psycopg, "connect", connect)

    assert reader._connect("secret-dsn", budget) is connection
    assert captured["connect_timeout"] == 2


def test_elapsed_connection_budget_is_timeout_and_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    clock = _ManualClock()
    budget = reader._DeadlineBudget.create(2.0, clock)
    connection = _FakeConnection()

    def connect(*_args, **_kwargs):
        clock.advance(2.1)
        return connection

    monkeypatch.setattr(psycopg, "connect", connect)

    with pytest.raises(reader.OperationalProjectionTimeoutError):
        reader._connect("secret-dsn", budget)
    assert connection.closed is True


def test_reader_fails_closed_for_missing_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _FakeConnection(head=False)
    monkeypatch.setattr(reader, "_connect", lambda _dsn, _timeout: connection)

    with pytest.raises(reader.OperationalProjectionUnavailableError):
        reader.OperationalProjectionReader("secret-dsn").select_attempt(
            id_type="reporting_run_id",
            identifier=RUN_ID,
            timeout_seconds=1.0,
        )


@pytest.mark.parametrize(
    "connection",
    (
        _FakeConnection(migration_checksum_valid=False),
        _FakeConnection(contract_valid=False),
        _FakeConnection(source_set_valid=False),
    ),
)
def test_reader_fails_closed_for_incompatible_or_invalid_projection(
    monkeypatch: pytest.MonkeyPatch,
    connection: _FakeConnection,
) -> None:
    monkeypatch.setattr(reader, "_connect", lambda _dsn, _budget: connection)

    with pytest.raises(reader.OperationalProjectionUnavailableError):
        reader.OperationalProjectionReader("secret-dsn").select_attempt(
            id_type="reporting_run_id",
            identifier=RUN_ID,
            timeout_seconds=1.0,
        )


def test_alert_sql_preserves_loader_causal_order_and_current_generation() -> None:
    sql = reader._alert_chain_sql() + reader._alert_select_clause()

    assert "generation_evidence" in sql
    assert "previous_alert_event_id = parent.alert_event_id" in sql
    assert "child.rule_id = parent.rule_id" in sql
    assert (
        "through_date, chain.rule_id, chain.causal_depth, "
        "chain.alert_event_id"
    ) in sql


@pytest.fixture(scope="module")
def projection_dsns() -> tuple[str, str, str]:
    if os.getenv(INTEGRATION_FLAG) != "1":
        pytest.skip("PostgreSQL integration test was not explicitly enabled.")
    migrator = os.getenv(OPERATIONAL_PROJECTION_MIGRATOR_DSN_ENV, "")
    writer = os.getenv("WIND_FORECAST_OPERATIONAL_PROJECTION_WRITER_DSN", "")
    selected_reader = os.getenv(OPERATIONAL_PROJECTION_READER_DSN_ENV, "")
    if not migrator or not writer or not selected_reader:
        pytest.fail("Explicit integration mode requires all three test DSNs.")
    return migrator, writer, selected_reader


def test_postgres_reader_uses_only_current_ready_generation(
    projection_dsns: tuple[str, str, str],
) -> None:
    import psycopg

    migrator_dsn, writer_dsn, reader_dsn = projection_dsns
    migrate(migrator_dsn)
    current_generation = sha256(b"reader-current-generation").hexdigest()
    old_generation = sha256(b"reader-old-generation").hexdigest()
    current_digest = sha256(b"reader-current-attempt").hexdigest()
    old_digest = sha256(b"reader-old-attempt").hexdigest()
    attempt_sources = {
        OLD_RUN_ID: {
            "domain": "reporting_run",
            "source_kind": "load_reporting_attempts",
            "schema_version": ATTEMPT_SCHEMA,
            "record_id": OLD_RUN_ID,
            "sha256": old_digest,
            "effective_at": "2026-08-03T12:00:00Z",
        },
        RUN_ID: {
            "domain": "reporting_run",
            "source_kind": "load_reporting_attempts",
            "schema_version": ATTEMPT_SCHEMA,
            "record_id": RUN_ID,
            "sha256": current_digest,
            "effective_at": "2026-08-03T12:00:00Z",
        },
    }
    alert_sources = [
        {
            "domain": "alert",
            "source_kind": "load_alert_history",
            "schema_version": "wind_forecast.monitoring_alert_event.v2",
            "record_id": alert_id,
            "sha256": alert_id,
            "effective_at": "2026-08-02",
        }
        for alert_id in (PARENT_ALERT_ID, CHILD_ALERT_ID)
    ]
    generation_sources = {
        old_generation: [attempt_sources[OLD_RUN_ID]],
        current_generation: [attempt_sources[RUN_ID], *alert_sources],
    }

    with psycopg.connect(writer_dsn) as connection:
        with connection.cursor() as cursor:
            for generation_id, sources in generation_sources.items():
                ordered_sources = sorted(
                    sources,
                    key=lambda item: (
                        item["source_kind"],
                        item["schema_version"],
                        item["record_id"],
                        item["sha256"],
                        item["effective_at"],
                        item["domain"],
                    ),
                )
                alert_count = sum(item["domain"] == "alert" for item in sources)
                cursor.execute(
                    "INSERT INTO operational_projection.projection_generation ("
                    "generation_id, environment_id, contract_version, schema_version, "
                    "projector_version, source_git_commit, source_set_sha256, "
                    "evidence_record_count, generation_evidence_count, model_era_count, "
                    "monitoring_report_count, quality_issue_count, monitoring_window_count, "
                    "performance_metric_count, drift_measurement_count, alert_event_count, "
                    "active_alert_snapshot_count, reporting_attempt_count, lineage_edge_count, "
                    "ready_at_utc) VALUES (%s, 'local', %s, %s, %s, %s, %s, "
                    "%s, %s, 0, 0, 0, 0, 0, 0, %s, 0, 1, 0, CURRENT_TIMESTAMP) "
                    "ON CONFLICT (generation_id) DO NOTHING",
                    (
                        generation_id,
                        CONTRACT_VERSION,
                        RELATIONAL_SCHEMA_VERSION,
                        PROJECTOR_VERSION,
                        "a" * 40,
                        canonical_sha256(ordered_sources),
                        len(sources),
                        len(sources),
                        alert_count,
                    ),
                )
            evidence_ids: dict[str, int] = {}
            all_sources = [
                attempt_sources[OLD_RUN_ID],
                attempt_sources[RUN_ID],
                *alert_sources,
            ]
            for source in all_sources:
                cursor.execute(
                    "INSERT INTO operational_projection.evidence_record ("
                    "domain, source_kind, schema_version, record_id, sha256, effective_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (domain, source_kind, schema_version, record_id, sha256) "
                    "DO NOTHING "
                    "RETURNING evidence_record_id",
                    (
                        source["domain"],
                        source["source_kind"],
                        source["schema_version"],
                        source["record_id"],
                        source["sha256"],
                        source["effective_at"],
                    ),
                )
                returned = cursor.fetchone()
                if returned is None:
                    cursor.execute(
                        "SELECT evidence_record_id FROM "
                        "operational_projection.evidence_record WHERE "
                        "domain = %s AND source_kind = %s AND schema_version = %s "
                        "AND record_id = %s AND sha256 = %s",
                        (
                            source["domain"],
                            source["source_kind"],
                            source["schema_version"],
                            source["record_id"],
                            source["sha256"],
                        ),
                    )
                    returned = cursor.fetchone()
                evidence_ids[source["record_id"]] = int(returned[0])
            for generation_id, run_id in (
                (old_generation, OLD_RUN_ID),
                (current_generation, RUN_ID),
            ):
                cursor.execute(
                    "INSERT INTO operational_projection.generation_evidence "
                    "(generation_id, evidence_record_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (generation_id, evidence_ids[run_id]),
                )
                cursor.execute(
                    "INSERT INTO operational_projection.reporting_attempt ("
                    "reporting_run_id, evidence_record_id, attempted_at_utc, through_date, "
                    "source_run_id, source_status, status, report_id, active_alert_count, "
                    "failure_at_utc, failure_type, failure_message) VALUES ("
                    "%s, %s, CURRENT_TIMESTAMP, '2026-08-02', 'source-run', "
                    "'succeeded', 'failed', NULL, 0, CURRENT_TIMESTAMP, "
                    "'RuntimeError', 'sanitized') ON CONFLICT DO NOTHING",
                    (run_id, evidence_ids[run_id]),
                )
            for alert_id in (PARENT_ALERT_ID, CHILD_ALERT_ID):
                cursor.execute(
                    "INSERT INTO operational_projection.generation_evidence "
                    "(generation_id, evidence_record_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (current_generation, evidence_ids[alert_id]),
                )
            cursor.execute(
                "INSERT INTO operational_projection.alert_event ("
                "alert_event_id, evidence_record_id, rule_id, through_date, "
                "event_type, severity, previous_alert_event_id) VALUES ("
                "%s, %s, 'adversarial-rule', '2026-08-02', 'opened', "
                "'warning', NULL) ON CONFLICT DO NOTHING",
                (PARENT_ALERT_ID, evidence_ids[PARENT_ALERT_ID]),
            )
            cursor.execute(
                "INSERT INTO operational_projection.alert_event ("
                "alert_event_id, evidence_record_id, rule_id, through_date, "
                "event_type, severity, previous_alert_event_id) VALUES ("
                "%s, %s, 'adversarial-rule', '2026-08-02', 'escalated', "
                "'critical', %s) ON CONFLICT DO NOTHING",
                (
                    CHILD_ALERT_ID,
                    evidence_ids[CHILD_ALERT_ID],
                    PARENT_ALERT_ID,
                ),
            )
            cursor.execute(
                "INSERT INTO operational_projection.projection_head "
                "(environment_id, generation_id, published_at_utc) "
                "VALUES ('local', %s, CURRENT_TIMESTAMP) "
                "ON CONFLICT (environment_id) DO UPDATE SET "
                "generation_id = EXCLUDED.generation_id, "
                "published_at_utc = EXCLUDED.published_at_utc",
                (current_generation,),
            )

    projection_reader = reader.OperationalProjectionReader(reader_dsn)
    current = projection_reader.select_attempt(
        id_type="reporting_run_id",
        identifier=RUN_ID,
        timeout_seconds=5.0,
    )
    old = projection_reader.select_attempt(
        id_type="reporting_run_id",
        identifier=OLD_RUN_ID,
        timeout_seconds=5.0,
    )

    assert current is not None
    assert current.values["reporting_run_id"] == RUN_ID
    assert old is None
    alerts = projection_reader.select_alerts(
        selector="date_interval",
        start_date=datetime(2026, 8, 2).date(),
        end_date=datetime(2026, 8, 2).date(),
        limit=10,
        offset=0,
        timeout_seconds=5.0,
    )
    assert alerts.selected_ids == (PARENT_ALERT_ID, CHILD_ALERT_ID)


def test_postgres_reader_end_to_end_matches_all_five_loader_queries(
    projection_dsns: tuple[str, str, str],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wind_forecast.operational_projection_projector as projector
    import wind_forecast.operational_query as operational
    from wind_forecast.operational_projection_reader import (
        OperationalProjectionReader,
    )
    from wind_forecast.operational_query import OperationalQueryService
    from wind_forecast.operational_query_models import (
        AuthorizationContext,
        CONTRACT_VERSION as QUERY_CONTRACT_VERSION,
    )
    from test_operational_projection_projector import _patch_sources

    migrator_dsn, writer_dsn, reader_dsn = projection_dsns
    migrate(migrator_dsn)
    values = _patch_sources(monkeypatch)
    projected = projector.project_projection(
        writer_dsn,
        tmp_path,
        environment_id="local",
        source_git_commit="2" * 40,
        clock=lambda: datetime(2026, 8, 3, 12, tzinfo=timezone.utc),
    )
    assert projected.status in {"projected", "no_op"}

    monkeypatch.setattr(
        operational,
        "load_monitoring_report_state",
        lambda _root: values["state"],
    )
    monkeypatch.setattr(
        operational,
        "load_active_alerts",
        lambda _root: values["active"],
    )
    monkeypatch.setattr(
        operational,
        "load_alert_history",
        lambda _root: values["history"],
    )
    monkeypatch.setattr(
        operational,
        "load_monitoring_report",
        lambda _path: values["report"],
    )
    monkeypatch.setattr(
        operational,
        "load_monitoring_calibration",
        lambda _path: values["calibration"],
    )
    monkeypatch.setattr(
        operational,
        "resolve_report_model_era",
        lambda _root, _report: values["era"],
    )
    monkeypatch.setattr(
        operational,
        "load_model_era",
        lambda _root, _model_era_id: values["era"],
    )

    attempt = values["attempts"][0]

    def load_attempt(_root, **kwargs):
        if kwargs.get("reporting_run_id") == attempt["run_id"]:
            return attempt
        if kwargs.get("report_id") == attempt["report_id"]:
            return attempt
        return None

    monkeypatch.setattr(operational, "load_reporting_attempt", load_attempt)

    now = datetime(2026, 8, 3, 12, tzinfo=timezone.utc)
    service_values = {
        "deployment_root": tmp_path / "deployment",
        "monitoring_store_root": tmp_path,
        "max_deadline_seconds": 5.0,
        "authorization_policy": lambda _context, _kind: True,
        "clock": lambda: now,
    }
    filesystem_service = OperationalQueryService(**service_values)
    required_service = OperationalQueryService(
        **service_values,
        projection_reader=OperationalProjectionReader(reader_dsn),
    )
    context = AuthorizationContext(principal="operator", trusted_local=True)

    def query(query_kind: str, **overrides: Any) -> dict[str, Any]:
        value: dict[str, Any] = {
            "contract_version": QUERY_CONTRACT_VERSION,
            "query_kind": query_kind,
            "selector": {"kind": "latest"},
            "window_days": None,
            "pagination": None,
            "requested_at_utc": now,
            "deadline": now + timedelta(seconds=5),
            "correlation_id": f"pg-{query_kind}",
        }
        value.update(overrides)
        return value

    queries = (
        query("data_quality"),
        query("monitoring_performance", window_days=30),
        query("monitoring_drift", window_days=30),
        query("monitoring_alerts"),
        query(
            "reporting_run",
            selector={
                "kind": "exact_id",
                "id_type": "reporting_run_id",
                "identifier": attempt["run_id"],
            },
        ),
    )
    for value in queries:
        filesystem_answer = filesystem_service.answer(value, context)
        required_answer = required_service.answer(value, context)
        assert required_answer == filesystem_answer
        assert required_answer.status.value == "answered"
        assert all(
            "postgres" not in citation.source_kind.lower()
            for citation in required_answer.evidence
        )
