from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_operational_projection as benchmark
from wind_forecast import operational_projection_migrations as migrations
from wind_forecast import operational_projection_projector as projector
from wind_forecast.operational_projection_models import ProjectionSnapshot, RelationalRow
from wind_forecast.operational_projection_projector import build_projection_snapshot


INTEGRATION_FLAG = "WIND_FORECAST_OPERATIONAL_PROJECTION_TEST_INTEGRATION"


def test_profiles_have_exact_contract_cardinalities() -> None:
    full = benchmark.PROFILES["full"]
    assert (
        full.reports,
        full.attempts,
        full.alerts,
        full.drift_measurements,
        full.repetitions,
    ) == (1_000, 10_000, 50_000, 200_000, 30)
    assert benchmark._feature_count(full) == 25
    assert full.enforce_timing_gate is True
    assert full.enforce_plan_gate is True

    smoke = benchmark.PROFILES["smoke"]
    assert smoke.enforce_timing_gate is False
    assert smoke.enforce_plan_gate is False
    assert smoke.repetitions == 3


def test_binary_copy_specs_are_complete_and_exactly_typed() -> None:
    assert tuple(benchmark.COPY_SPECS) == (
        "evidence_record",
        "generation_evidence",
        *benchmark.TABLE_ORDER,
    )
    allowed_types = {
        "bool",
        "bpchar",
        "date",
        "float8",
        "int4",
        "int8",
        "text",
        "timestamptz",
    }
    for table, spec in benchmark.COPY_SPECS.items():
        assert len(spec.columns) == len(spec.postgres_types)
        assert len(spec.columns) == len(set(spec.columns))
        assert set(spec.postgres_types) <= allowed_types
        if table in benchmark.TABLE_ORDER:
            assert spec.columns == projector._TABLE_COLUMNS[table]


def test_copy_rows_uses_binary_format_exact_types_and_bounded_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {"rows": [], "checks": []}

    class RecordingCopy:
        def __enter__(self) -> "RecordingCopy":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def set_types(self, values: tuple[str, ...]) -> None:
            observed["types"] = values

        def write_row(self, row: tuple[object, ...]) -> None:
            observed["rows"].append(row)  # type: ignore[union-attr]

    class RecordingCursor:
        def copy(self, statement: str) -> RecordingCopy:
            observed["statement"] = statement
            return RecordingCopy()

    monkeypatch.setattr(benchmark, "COPY_DEADLINE_CHECK_INTERVAL", 2)
    monkeypatch.setattr(
        benchmark,
        "_check_runtime",
        lambda deadline, phase: observed["checks"].append(  # type: ignore[union-attr]
            (deadline, phase)
        ),
    )
    rows = (("a" * 64, 1), ("b" * 64, 2), ("c" * 64, 3))

    written = benchmark._copy_rows(
        RecordingCursor(),
        "generation_evidence",
        rows,
        deadline_ns=123,
    )

    assert written == 3
    assert observed["statement"] == (
        "COPY operational_projection.generation_evidence "
        "(generation_id, evidence_record_id) FROM STDIN (FORMAT BINARY)"
    )
    assert observed["types"] == ("bpchar", "int8")
    assert observed["rows"] == list(rows)
    assert observed["checks"] == [
        (123, "snapshot_publish:copy_generation_evidence")
    ]


@pytest.mark.parametrize("method", ("binary_copy", "text_copy"))
def test_generation_evidence_copy_probe_is_typed_and_sanitized(
    method: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {"rows": [], "exits": []}

    class RecordingCopy:
        def set_types(self, values: tuple[str, ...]) -> None:
            observed["types"] = values

        def write_row(self, row: tuple[object, ...]) -> None:
            observed["rows"].append(row)  # type: ignore[union-attr]

    class RecordingManager:
        def __enter__(self) -> RecordingCopy:
            return RecordingCopy()

        def __exit__(self, *args: object) -> None:
            observed["exits"].append(args)  # type: ignore[union-attr]

    class RecordingCursor:
        def copy(self, statement: str) -> RecordingManager:
            observed["statement"] = statement
            return RecordingManager()

    timings: dict[str, float] = {}
    written = benchmark._copy_generation_evidence_probe(
        RecordingCursor(),
        method=method,
        generation_id="a" * 64,
        association_count=3,
        deadline_ns=None,
        timings=timings,
    )

    assert written == 3
    assert observed["rows"] == [("a" * 64, 1), ("a" * 64, 2), ("a" * 64, 3)]
    assert observed["exits"] == [(None, None, None)]
    if method == "binary_copy":
        assert str(observed["statement"]).endswith("(FORMAT BINARY)")
        assert observed["types"] == ("bpchar", "int8")
    else:
        assert not str(observed["statement"]).endswith("(FORMAT BINARY)")
        assert "types" not in observed
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert {event["method"] for event in events} == {method}
    assert {event["association_count"] for event in events} == {3}
    assert set(timings) == (
        {"copy_open", "copy_first_row", "copy_remaining_rows", "copy_finalize"}
        | ({"copy_configure"} if method == "binary_copy" else set())
    )
    serialized = json.dumps(events)
    assert "postgresql://" not in serialized.lower()
    assert str(Path.cwd()).lower() not in serialized.lower()


def test_generation_evidence_insert_select_probe_is_counted() -> None:
    class RecordingCursor:
        rowcount = 7

        def execute(self, statement: str, parameters: tuple[object, ...]) -> None:
            self.statement = statement
            self.parameters = parameters

    cursor = RecordingCursor()
    timings: dict[str, float] = {}
    written = benchmark._insert_select_generation_evidence_probe(
        cursor,
        generation_id="b" * 64,
        association_count=7,
        deadline_ns=None,
        timings=timings,
    )

    assert written == 7
    assert "INSERT INTO operational_projection.generation_evidence" in cursor.statement
    assert "FROM operational_projection.evidence_record" in cursor.statement
    assert cursor.parameters == ("b" * 64, 1, 7)
    assert tuple(timings) == ("insert_select",)


def test_generation_evidence_identity_digest_is_deterministic() -> None:
    assert benchmark._generation_evidence_identity_sha256(3) == (
        "ca73761ddabfffcbe51170be0b07f67bafcdbed202545c60707573d36dc935b4"
    )


def test_publication_step_progress_is_sanitized_and_timed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    timings: dict[str, float] = {}

    assert benchmark._run_publication_step(
        "copy_alert_event",
        timings,
        None,
        lambda: 7,
        row_count=7,
    ) == 7

    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert events[0] == {
        "event": "benchmark_progress",
        "phase": "snapshot_publish_step_started",
        "row_count": 7,
        "step": "copy_alert_event",
    }
    assert events[1]["event"] == "benchmark_progress"
    assert events[1]["phase"] == "snapshot_publish_step_completed"
    assert events[1]["row_count"] == 7
    assert events[1]["step"] == "copy_alert_event"
    assert isinstance(events[1]["elapsed_ms"], float)
    assert tuple(timings) == ("copy_alert_event",)
    serialized = json.dumps(events)
    assert str(Path.cwd()).lower() not in serialized.lower()
    assert "postgresql://" not in serialized.lower()


def test_successful_commit_boundary_checks_deadline_only_before_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    monkeypatch.setattr(
        benchmark,
        "_check_runtime",
        lambda _deadline, phase: observed.append(f"check:{phase}"),
    )

    benchmark._run_publication_step(
        "commit",
        {},
        123,
        lambda: observed.append("commit"),
        check_deadline_after=False,
    )

    assert observed == ["check:snapshot_publish:commit", "commit"]


def test_synthetic_store_is_loader_valid_and_projection_complete(
    tmp_path: Path,
) -> None:
    profile = benchmark.PROFILES["smoke"]
    selection = benchmark.generate_synthetic_store(tmp_path, profile)
    snapshot = build_projection_snapshot(
        tmp_path,
        environment_id="local",
        source_git_commit="a" * 40,
    )

    assert snapshot.counts()["monitoring_report_count"] == profile.reports
    assert snapshot.counts()["reporting_attempt_count"] == profile.attempts
    assert snapshot.counts()["alert_event_count"] == profile.alerts
    assert snapshot.counts()["monitoring_window_count"] == profile.reports * 2
    assert snapshot.counts()["performance_metric_count"] == profile.reports * 10
    assert snapshot.counts()["drift_measurement_count"] == profile.drift_measurements
    assert not snapshot.rows_for("model_era")
    for table in benchmark.TABLE_ORDER:
        rows = snapshot.rows_for(table)
        if not rows:
            continue
        actual_columns = {
            *rows[0].value_map(),
            *(link.column for link in rows[0].evidence_links),
        }
        assert actual_columns == set(benchmark.COPY_SPECS[table].columns)

    cases = {case.name: case for case in benchmark.build_query_cases(tmp_path, selection)}
    assert tuple(cases) == (
        "alert_interval_pagination",
        "exact_alert_id",
        "reporting_run_by_run_id",
        "reporting_run_by_report_id",
        "performance_report_window",
        "drift_report_window",
    )
    interval = cases["alert_interval_pagination"]
    assert interval.parameters == (
        selection.alert_start,
        selection.alert_end,
        "local",
        selection.alert_limit,
        selection.alert_offset,
    )
    cte_sql, outer_sql = interval.sql.split(") SELECT", maxsplit=1)
    assert "WITH interval_alerts AS MATERIALIZED" in cte_sql
    assert "WHERE through_date BETWEEN %s AND %s" in cte_sql
    assert "LIMIT" not in cte_sql
    assert "OFFSET" not in cte_sql
    assert "generation_evidence" not in cte_sql
    assert "projection_head" not in cte_sql
    assert "JOIN operational_projection.generation_evidence" in outer_sql
    assert "JOIN operational_projection.projection_head" in outer_sql
    assert (
        "ORDER BY ia.through_date, ia.rule_id, ia.alert_event_id "
        "LIMIT %s OFFSET %s"
    ) in outer_sql
    assert outer_sql.index("projection_head") < outer_sql.index("LIMIT %s")
    assert interval.expected_indexes == ("alert_event_date_idx",)
    assert cases["exact_alert_id"].filesystem() == ((selection.alert_event_id,),)
    assert cases["reporting_run_by_run_id"].filesystem()[0][0] == (
        selection.reporting_run_id
    )
    assert cases["reporting_run_by_report_id"].filesystem()[0][1] == (
        selection.report_id
    )
    assert [row[2] for row in cases["performance_report_window"].filesystem()] == list(
        benchmark.PERFORMANCE_METRICS
    )
    drift = cases["drift_report_window"].filesystem()
    assert len(drift) == profile.drift_measurements // profile.reports // 2
    assert [row[2] for row in drift] == list(range(len(drift)))


@pytest.fixture
def benchmark_projection_dsns() -> dict[str, str]:
    if os.getenv(INTEGRATION_FLAG) != "1":
        pytest.skip("PostgreSQL integration test was not explicitly enabled.")
    variables = {
        "migrator": benchmark.MIGRATOR_DSN_ENV,
        "writer": benchmark.WRITER_DSN_ENV,
        "reader": benchmark.READER_DSN_ENV,
    }
    values = {role: os.getenv(variable, "") for role, variable in variables.items()}
    if any(not value for value in values.values()):
        pytest.fail("Explicit integration mode requires all three test DSNs.")
    return values


def _assert_projection_database_is_empty(reader_dsn: str) -> None:
    import psycopg

    with psycopg.connect(reader_dsn) as connection:
        with connection.cursor() as cursor:
            for table in (
                "projection_head",
                "projection_generation",
                "evidence_record",
                "generation_evidence",
                "monitoring_report",
                "drift_measurement",
            ):
                cursor.execute(
                    f"SELECT count(*) FROM operational_projection.{table}"
                )
                assert cursor.fetchone()[0] == 0


@pytest.mark.parametrize("method", benchmark.GENERATION_EVIDENCE_PROBE_METHODS)
def test_generation_evidence_probe_methods_use_real_schema_and_roll_back(
    benchmark_projection_dsns: dict[str, str],
    method: str,
) -> None:
    result = benchmark.run_generation_evidence_probe(
        method,
        128,
        1,
        max_runtime_seconds=45,
    )

    assert result["decision"] == "PASS"
    assert result["method"] == method
    assert result["association_count"] == 128
    assert result["rows_written"] == 128
    assert result["rolled_back"] is True
    _assert_projection_database_is_empty(benchmark_projection_dsns["reader"])


def test_generation_evidence_probe_rejects_invalid_fk_and_rolls_back(
    benchmark_projection_dsns: dict[str, str],
) -> None:
    import psycopg

    migrations.migrate(benchmark_projection_dsns["migrator"])
    generation_id = "c" * 64
    with psycopg.connect(benchmark_projection_dsns["writer"]) as connection:
        benchmark._register_binary_copy_dumpers(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = '30s'")
            benchmark._assert_probe_database_empty(cursor)
            benchmark._seed_generation_evidence_probe(
                cursor,
                generation_id=generation_id,
                association_count=2,
                deadline_ns=None,
            )
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                benchmark._copy_generation_evidence_probe(
                    cursor,
                    method="binary_copy",
                    generation_id=generation_id,
                    association_count=3,
                    deadline_ns=None,
                    timings={},
                )
        connection.rollback()
    _assert_projection_database_is_empty(benchmark_projection_dsns["reader"])


def test_binary_copy_specs_match_postgres_schema(
    benchmark_projection_dsns: dict[str, str],
) -> None:
    import psycopg

    migrations.migrate(benchmark_projection_dsns["migrator"])
    with psycopg.connect(benchmark_projection_dsns["reader"]) as connection:
        with connection.cursor() as cursor:
            for table, spec in benchmark.COPY_SPECS.items():
                cursor.execute(
                    "SELECT a.attname, t.typname "
                    "FROM pg_catalog.pg_attribute a "
                    "JOIN pg_catalog.pg_class c ON c.oid = a.attrelid "
                    "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                    "JOIN pg_catalog.pg_type t ON t.oid = a.atttypid "
                    "WHERE n.nspname = 'operational_projection' "
                    "AND c.relname = %s AND a.attnum > 0 AND NOT a.attisdropped "
                    "ORDER BY a.attnum",
                    (table,),
                )
                actual = dict(cursor.fetchall())
                assert tuple(actual[column] for column in spec.columns) == (
                    spec.postgres_types
                )


def test_binary_publication_rolls_back_head_and_rows_before_commit(
    benchmark_projection_dsns: dict[str, str],
    tmp_path: Path,
) -> None:
    migrations.migrate(benchmark_projection_dsns["migrator"])
    _assert_projection_database_is_empty(benchmark_projection_dsns["reader"])
    benchmark.generate_synthetic_store(tmp_path, benchmark.PROFILES["smoke"])
    snapshot = build_projection_snapshot(
        tmp_path,
        environment_id="local",
        source_git_commit="a" * 40,
    )

    def fail_before_commit(stage: str) -> None:
        assert stage == "before_commit"
        raise RuntimeError("forced pre-commit failure")

    with pytest.raises(RuntimeError, match="forced pre-commit failure"):
        benchmark._bulk_publish_snapshot(
            benchmark_projection_dsns["writer"],
            snapshot,
            deadline_ns=None,
            failure_hook=fail_before_commit,
        )

    _assert_projection_database_is_empty(benchmark_projection_dsns["reader"])


def test_binary_publication_keeps_constraints_enabled_and_rolls_back(
    benchmark_projection_dsns: dict[str, str],
    tmp_path: Path,
) -> None:
    import psycopg

    migrations.migrate(benchmark_projection_dsns["migrator"])
    _assert_projection_database_is_empty(benchmark_projection_dsns["reader"])
    benchmark.generate_synthetic_store(tmp_path, benchmark.PROFILES["smoke"])
    snapshot = build_projection_snapshot(
        tmp_path,
        environment_id="local",
        source_git_commit="b" * 40,
    )
    rows = list(snapshot.rows)
    drift_index = next(
        index
        for index, row in enumerate(rows)
        if row.table == "drift_measurement"
    )
    drift_row = rows[drift_index]
    invalid_values = drift_row.value_map()
    invalid_values["severity"] = "invalid"
    rows[drift_index] = RelationalRow.create(
        drift_row.table,
        invalid_values,
        evidence_links={
            link.column: link.evidence for link in drift_row.evidence_links
        },
    )
    invalid_snapshot = ProjectionSnapshot(snapshot.manifest, tuple(rows))

    with pytest.raises(psycopg.errors.CheckViolation):
        benchmark._bulk_publish_snapshot(
            benchmark_projection_dsns["writer"],
            invalid_snapshot,
            deadline_ns=None,
        )

    _assert_projection_database_is_empty(benchmark_projection_dsns["reader"])


def _case_result(
    *,
    equivalent: bool = True,
    maximum: float = 1.0,
    speedup: float = 0.5,
    speed_gate: bool = False,
    indexes: bool = True,
) -> dict[str, object]:
    return {
        "equivalent": equivalent,
        "postgres_max_ms": maximum,
        "speedup": speedup,
        "speed_gate": speed_gate,
        "expected_indexes_used": indexes,
    }


def test_gate_is_closed_and_only_two_cases_require_speedup() -> None:
    passing = {
        "alert_interval_pagination": _case_result(speed_gate=True, speedup=0.2),
        "reporting_run_by_report_id": _case_result(speed_gate=True, speedup=0.21),
        "exact_alert_id": _case_result(speed_gate=False, speedup=-1.0),
    }
    assert benchmark.evaluate_gate(
        passing, enforce_timing=True, enforce_plans=True
    ) == ("GO", ())

    too_slow = dict(passing)
    too_slow["reporting_run_by_report_id"] = _case_result(
        speed_gate=True, speedup=0.199999
    )
    decision, failures = benchmark.evaluate_gate(
        too_slow, enforce_timing=True, enforce_plans=True
    )
    assert decision == "NO-GO"
    assert failures == ("reporting_run_by_report_id:speedup",)

    mismatch = {"exact_alert_id": _case_result(equivalent=False)}
    assert benchmark.evaluate_gate(
        mismatch, enforce_timing=False, enforce_plans=False
    ) == ("NO-GO", ("exact_alert_id:identity_order_mismatch",))


def test_gate_rejects_deadline_and_missing_index() -> None:
    cases = {
        "exact_alert_id": _case_result(
            maximum=benchmark.DEADLINE_MS,
            indexes=False,
        )
    }
    assert benchmark.evaluate_gate(
        cases, enforce_timing=True, enforce_plans=True
    ) == ("NO-GO", ("exact_alert_id:deadline", "exact_alert_id:index"))


def test_explain_index_names_are_collected_without_raw_plan_output() -> None:
    plan = [
        {
            "Plan": {
                "Node Type": "Nested Loop",
                "Plans": [
                    {"Node Type": "Index Scan", "Index Name": "alert_event_pkey"},
                    {
                        "Node Type": "Bitmap Heap Scan",
                        "Plans": [
                            {
                                "Node Type": "Bitmap Index Scan",
                                "Index Name": "alert_event_date_idx",
                            }
                        ],
                    },
                ],
            }
        }
    ]
    assert benchmark._index_names(plan) == (
        "alert_event_date_idx",
        "alert_event_pkey",
    )


def test_snapshot_rows_are_grouped_in_one_pass() -> None:
    snapshot = SimpleNamespace(
        manifest=SimpleNamespace(evidence=(1, 2, 3)),
        rows=(
            SimpleNamespace(table="alert_event"),
            SimpleNamespace(table="alert_event"),
            SimpleNamespace(table="reporting_attempt"),
        ),
    )
    grouped, counts = benchmark._group_snapshot_rows(snapshot)
    assert len(grouped["alert_event"]) == 2
    assert len(grouped["reporting_attempt"]) == 1
    assert counts["evidence_record_count"] == 3
    assert counts["generation_evidence_count"] == 3
    assert counts["alert_event_count"] == 2
    assert counts["drift_measurement_count"] == 0


def test_snapshot_publication_precomputes_generation_id_once() -> None:
    class CountingSnapshot:
        rows = (SimpleNamespace(table="alert_event"),)
        manifest = SimpleNamespace(
            evidence=(SimpleNamespace(identity="evidence-1"),)
        )
        accesses = 0

        @property
        def generation_id(self) -> str:
            self.accesses += 1
            return "a" * 64

    snapshot = CountingSnapshot()
    generation_id, evidence_ids, grouped, counts = (
        benchmark._prepare_snapshot_publication(snapshot)
    )

    assert generation_id == "a" * 64
    assert snapshot.accesses == 1
    assert evidence_ids == {"evidence-1": 1}
    assert len(grouped["alert_event"]) == 1
    assert counts["generation_evidence_count"] == 1


def test_measurement_keeps_all_repetitions_but_shares_group_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = 0

    def loader() -> list[str]:
        nonlocal loads
        loads += 1
        return ["expected"]

    def selector(payload: list[str]) -> tuple[tuple[str, ...], ...]:
        return ((payload[0],),)

    cases = tuple(
        benchmark.QueryCase(
            name=name,
            filesystem_group="shared",
            filesystem_loader=loader,
            filesystem_selector=selector,
            sql="SELECT 1",
            parameters=(),
            expected_indexes=(),
        )
        for name in ("first", "second")
    )
    monkeypatch.setattr(
        benchmark,
        "_postgres_rows",
        lambda connection, case: (("expected",),),
    )

    class Cursor:
        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, *args: object) -> None:
            return None

        def fetchone(self) -> tuple[list[dict[str, object]]]:
            return ([{"Plan": {"Node Type": "Result"}}],)

    connection = SimpleNamespace(cursor=lambda: Cursor())
    results = benchmark._measure_cases(connection, cases, 30)
    assert loads == 31  # one warm-up plus exactly 30 measured enumerations
    assert all(result["equivalent"] for result in results.values())


def test_runtime_limit_is_reported_as_sanitized_no_go(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_closed(*args: object, **kwargs: object) -> None:
        raise benchmark.BenchmarkNoGo("benchmark_runtime:query_measurement")

    monkeypatch.setattr(benchmark, "run_benchmark", fail_closed)
    assert benchmark.main(
        ["--profile", "full", "--max-runtime-seconds", "1", "--worker"]
    ) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["decision"] == "NO-GO"
    assert payload["failures"] == ["benchmark_runtime:query_measurement"]
    assert str(Path.cwd()).lower() not in captured.out.lower()
    assert "postgresql://" not in captured.out.lower()


def test_probe_worker_dispatch_and_output_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def run_probe(
        method: str,
        association_count: int,
        trial: int,
        *,
        max_runtime_seconds: float,
    ) -> dict[str, object]:
        observed.update(
            {
                "method": method,
                "association_count": association_count,
                "trial": trial,
                "max_runtime_seconds": max_runtime_seconds,
            }
        )
        return {
            "schema_version": benchmark.GENERATION_EVIDENCE_PROBE_SCHEMA_VERSION,
            "decision": "PASS",
            "method": method,
            "association_count": association_count,
            "trial": trial,
        }

    monkeypatch.setattr(benchmark, "run_generation_evidence_probe", run_probe)
    assert benchmark.main(
        [
            "--generation-evidence-probe-method",
            "insert_select",
            "--generation-evidence-probe-count",
            "1000",
            "--generation-evidence-probe-trial",
            "2",
            "--max-runtime-seconds",
            "45",
            "--worker",
        ]
    ) == 0

    assert observed == {
        "method": "insert_select",
        "association_count": 1000,
        "trial": 2,
        "max_runtime_seconds": 45.0,
    }
    output = capsys.readouterr().out
    assert json.loads(output)["decision"] == "PASS"
    assert "postgresql://" not in output.lower()
    assert str(Path.cwd()).lower() not in output.lower()


def test_probe_supervisor_enforces_its_hard_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def expire(*args: object, **kwargs: object) -> None:
        raise benchmark.subprocess.TimeoutExpired(cmd="probe", timeout=45)

    monkeypatch.setattr(benchmark.subprocess, "run", expire)
    assert benchmark.main(
        [
            "--generation-evidence-probe-method",
            "binary_copy",
            "--generation-evidence-probe-count",
            "1000",
            "--max-runtime-seconds",
            "45",
        ]
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "association_count": 1000,
        "decision": "NO-GO",
        "failures": ["generation_evidence_probe:hard_timeout"],
        "method": "binary_copy",
        "schema_version": benchmark.GENERATION_EVIDENCE_PROBE_SCHEMA_VERSION,
        "trial": 1,
    }


def test_statement_timeout_is_translated_to_no_go() -> None:
    timeout = RuntimeError("raw database detail")
    timeout.sqlstate = "57014"  # type: ignore[attr-defined]
    with pytest.raises(benchmark.BenchmarkNoGo) as raised:
        benchmark._raise_statement_timeout(timeout, "snapshot_publish:deadline")
    assert raised.value.failures == ("snapshot_publish:deadline",)
    other = RuntimeError("not a statement timeout")
    assert benchmark._raise_statement_timeout(other, "ignored") is None


def test_supervisor_enforces_hard_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def expire(*args: object, **kwargs: object) -> None:
        raise benchmark.subprocess.TimeoutExpired(cmd="benchmark", timeout=1)

    monkeypatch.setattr(benchmark.subprocess, "run", expire)
    assert benchmark.main(
        ["--profile", "full", "--max-runtime-seconds", "1"]
    ) == 1
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["decision"] == "NO-GO"
    assert payload["failures"] == ["benchmark_runtime:hard_timeout"]
    assert str(Path.cwd()).lower() not in output.lower()


def test_supervisor_cleanup_removes_only_its_synthetic_workspace(
    tmp_path: Path,
) -> None:
    token = "0123456789abcdef"
    owned = tmp_path / f"wf-projection-benchmark-{token}-owned"
    unrelated = tmp_path / "wf-projection-benchmark-other"
    owned.mkdir()
    unrelated.mkdir()
    (owned / "fixture.json").write_text("{}", encoding="utf-8")
    benchmark._cleanup_worker_stores(token, temp_root=tmp_path)
    assert not owned.exists()
    assert unrelated.is_dir()


def test_cli_configuration_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in (
        benchmark.MIGRATOR_DSN_ENV,
        benchmark.WRITER_DSN_ENV,
        benchmark.READER_DSN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    assert benchmark.main(["--profile", "smoke", "--worker"]) == 2
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["decision"] == "ERROR"
    assert "dsn" not in output.lower()
    assert "postgresql://" not in output.lower()
    assert str(Path.cwd()).lower() not in output.lower()
