"""Deterministic PostgreSQL operational-projection readiness benchmark.

The command creates only synthetic evidence in a temporary directory, projects it
into an isolated PostgreSQL database, and compares identity selection with the
existing verified filesystem loaders. It never accepts a governed store path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import statistics
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any, Callable, Iterable, Mapping, Sequence

from wind_forecast.monitoring_reporting import (
    load_alert_history,
    load_monitoring_report,
    load_reporting_attempts,
)
from wind_forecast.operational_projection_migrations import migrate
from wind_forecast.operational_projection_projector import (
    build_projection_snapshot,
    resolve_source_git_commit,
)


MIGRATOR_DSN_ENV = "WIND_FORECAST_OPERATIONAL_PROJECTION_MIGRATOR_DSN"
WRITER_DSN_ENV = "WIND_FORECAST_OPERATIONAL_PROJECTION_WRITER_DSN"
READER_DSN_ENV = "WIND_FORECAST_OPERATIONAL_PROJECTION_READER_DSN"
ENVIRONMENT_ENV = "WIND_FORECAST_OPERATIONAL_ENVIRONMENT_ID"
PERFORMANCE_METRICS = ("MAE", "RMSE", "bias", "MAPE_percent", "R2")
DETECTORS = ("ks_statistic", "normalized_wasserstein")
COMPARATORS = ("global", "seasonal")
DEADLINE_MS = 5_000.0
MINIMUM_SPEEDUP = 0.20
TABLE_ORDER = (
    "model_era",
    "monitoring_report",
    "quality_issue",
    "monitoring_window",
    "performance_metric",
    "drift_measurement",
    "alert_event",
    "active_alert_snapshot",
    "reporting_attempt",
    "lineage_edge",
)


@dataclass(frozen=True)
class Profile:
    name: str
    reports: int
    attempts: int
    alerts: int
    drift_measurements: int
    repetitions: int
    enforce_timing_gate: bool
    enforce_plan_gate: bool


PROFILES = {
    "smoke": Profile("smoke", 4, 12, 40, 800, 3, False, False),
    "full": Profile("full", 1_000, 10_000, 50_000, 200_000, 30, True, True),
}


@dataclass(frozen=True)
class FixtureSelection:
    latest_report_id: str
    report_id: str
    reporting_run_id: str
    alert_event_id: str
    alert_start: date
    alert_end: date
    alert_limit: int
    alert_offset: int


@dataclass(frozen=True)
class QueryCase:
    name: str
    filesystem_group: str
    filesystem_loader: Callable[[], Any]
    filesystem_selector: Callable[[Any], tuple[tuple[Any, ...], ...]]
    sql: str
    parameters: tuple[Any, ...]
    expected_indexes: tuple[str, ...]
    speed_gate: bool = False

    def filesystem(self) -> tuple[tuple[Any, ...], ...]:
        return self.filesystem_selector(self.filesystem_loader())


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _record(kind: str, id_field: str, body: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(
        body, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        id_field: sha256(kind.encode("utf-8") + b":" + canonical).hexdigest(),
        **body,
    }


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id(index: int) -> str:
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
    return f"{instant.strftime('%Y%m%dT%H%M%S%fZ')}-{index:012x}"


def _feature_count(profile: Profile) -> int:
    denominator = profile.reports * 2 * len(COMPARATORS) * len(DETECTORS)
    if profile.drift_measurements % denominator:
        raise ValueError("Drift fixture cardinality is not exactly representable.")
    return profile.drift_measurements // denominator


def generate_synthetic_store(root: Path, profile: Profile) -> FixtureSelection:
    """Create deterministic, loader-valid synthetic operational evidence."""
    reporting_root = root / "reporting"
    reference_csv = reporting_root / "references" / "pending" / "reference.csv"
    reference_csv.parent.mkdir(parents=True, exist_ok=True)
    reference_csv.write_text("Date,feature_00\n2020-01-01,0.0\n", encoding="utf-8")
    reference_body = {
        "schema_version": "wind_forecast.monitoring_reference.v1",
        "period": {"start": "2020-01-01", "end": "2020-12-31"},
        "feature_names": [f"feature_{index:02d}" for index in range(_feature_count(profile))],
        "reference_csv_sha256": _sha256_file(reference_csv),
    }
    reference = _record(
        "monitoring_reference", "reference_id", reference_body
    )
    reference["reference_path"] = "synthetic-reference.csv"
    reference_dir = reporting_root / "references" / reference["reference_id"]
    reference_dir.mkdir(parents=True)
    reference_csv.replace(reference_dir / "reference.csv")
    _write_json(reference_dir / "manifest.json", reference)

    backtest = reference_dir / "backtest-placeholder.json"
    backtest.write_text("{}\n", encoding="utf-8")
    threshold = {"warning": 0.15, "critical": 0.3, "direction": "upper"}
    features = reference["feature_names"]
    performance_limits = {
        name: {
            "warning": 20.0,
            "critical": 30.0,
            "direction": "lower" if name == "R2" else "upper",
        }
        for name in ("MAE", "RMSE", "absolute_bias", "MAPE_percent", "R2")
    }
    drift_limits = {
        feature: {
            window: {
                comparator: {detector: dict(threshold) for detector in DETECTORS}
                for comparator in COMPARATORS
            }
            for window in ("30", "90")
        }
        for feature in features
    }
    calibration_body = {
        "schema_version": "wind_forecast.monitoring_calibration.v1",
        "reference_id": reference["reference_id"],
        "policy_sha256": "8" * 64,
        "reference_manifest_sha256": _sha256_file(reference_dir / "manifest.json"),
        "backtest_summary_sha256": _sha256_file(backtest),
        "thresholds": {
            "performance": {"30": performance_limits, "90": performance_limits},
            "feature_drift": drift_limits,
        },
    }
    calibration = _record(
        "monitoring_calibration", "calibration_id", calibration_body
    )
    calibration["reference_dir"] = "relocatable-synthetic-reference"
    calibration_dir = reporting_root / "calibrations" / calibration["calibration_id"]
    calibration_dir.mkdir(parents=True)
    _write_json(calibration_dir / "calibration.json", calibration)
    (calibration_dir / "backtest_summary.json").write_bytes(backtest.read_bytes())

    alert_base = date(2020, 1, 1)
    alert_ids: list[str] = []
    for index in range(profile.alerts):
        body = {
            "schema_version": "wind_forecast.monitoring_alert_event.v2",
            "rule_id": f"synthetic-rule-{index:06d}",
            "through_date": (alert_base + timedelta(days=index % 1_000)).isoformat(),
            "event_type": "opened",
            "severity": "warning" if index % 2 else "critical",
            "previous_alert_event_id": None,
            "model_era_id": "synthetic-unassociated",
            "deployment_id": "synthetic-unassociated",
            "model_version": "0",
            "delivery": "local_immutable_record",
        }
        alert = _record("monitoring_alert", "alert_event_id", body)
        alert_ids.append(str(alert["alert_event_id"]))
        _write_json(reporting_root / "alerts" / f"{alert['alert_event_id']}.json", alert)

    report_ids: list[str] = []
    run_ids: list[str] = []
    report_base = date(2023, 1, 1)
    for index in range(profile.reports):
        run_id = _run_id(index)
        through = report_base + timedelta(days=index)
        feature_drift = {
            feature: {
                comparator: {
                    "ks_statistic": 0.1 + (index % 5) / 100,
                    "normalized_wasserstein": 0.2 + (index % 5) / 100,
                    "severity": "warning",
                }
                for comparator in COMPARATORS
            }
            for feature in features
        }
        windows = {}
        for window_days in (30, 90):
            windows[str(window_days)] = {
                "status": "available",
                "calendar_start": (through - timedelta(days=window_days - 1)).isoformat(),
                "calendar_end": through.isoformat(),
                "sample_count": window_days,
                "minimum_samples": 15 if window_days == 30 else 45,
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
                    "severity": {name: "ok" for name in PERFORMANCE_METRICS},
                },
                "feature_drift": feature_drift,
            }
        body = {
            "schema_version": "wind_forecast.monitoring_report.v1",
            "run_id": run_id,
            "created_at_utc": _utc_text(
                datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
            ),
            "through_date": through.isoformat(),
            "source_batch": {"run_id": f"source-{index:05d}", "status": "succeeded"},
            "reference": {
                "calibration_id": calibration["calibration_id"],
                "reference_id": reference["reference_id"],
                "policy_sha256": "8" * 64,
            },
            "config": {"minimum_samples": {"30": 15, "90": 45}},
            "quality": {
                "status": "available",
                "batch_status": "succeeded",
                "verdict": "PASS",
                "issues": [],
                "freshness": {
                    "common_validated_watermark": through.isoformat(),
                    "watermark_age_days": 0,
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
            "windows": windows,
            "active_alerts": {},
            "alert_events": [],
            "persistence": {},
            "lineage": {"prediction_ids": []},
        }
        report = _record("monitoring_report", "report_id", body)
        report_id = str(report["report_id"])
        report_ids.append(report_id)
        run_ids.append(run_id)
        _write_json(
            reporting_root / "reports" / report_id / "report.json", report
        )
        plan = {
            "status": "planned",
            "through_date": through.isoformat(),
            "source_run_id": f"source-{index:05d}",
            "source_status": "succeeded",
            "calibration_id": calibration["calibration_id"],
        }
        run_root = reporting_root / "runs" / run_id
        _write_json(
            run_root / "request.json",
            {
                "schema_version": "wind_forecast.monitoring_report_request.v2",
                "run_id": run_id,
                "requested_at_utc": _utc_text(
                    datetime(2026, 1, 1, tzinfo=timezone.utc)
                    + timedelta(minutes=index)
                ),
                "plan": plan,
            },
        )
        _write_json(
            run_root / "result.json",
            {
                "schema_version": "wind_forecast.monitoring_report_result.v2",
                "run_id": run_id,
                "status": "succeeded",
                "report_id": report_id,
                "active_alert_count": 0,
                "plan": plan,
            },
        )

    for index in range(profile.reports, profile.attempts):
        run_id = _run_id(index)
        run_root = reporting_root / "runs" / run_id
        attempted = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
        plan = {
            "status": "planned",
            "through_date": (report_base + timedelta(days=index % profile.reports)).isoformat(),
            "source_run_id": f"source-{index:05d}",
            "source_status": "succeeded",
            "calibration_id": calibration["calibration_id"],
        }
        _write_json(
            run_root / "request.json",
            {
                "schema_version": "wind_forecast.monitoring_report_request.v2",
                "run_id": run_id,
                "requested_at_utc": _utc_text(attempted),
                "plan": plan,
            },
        )
        _write_json(
            run_root / "failure.json",
            {
                "schema_version": "wind_forecast.monitoring_report_failure.v1",
                "run_id": run_id,
                "failed_at_utc": _utc_text(attempted + timedelta(seconds=1)),
                "error_type": "SyntheticFailure",
                "error": "Synthetic benchmark failure.",
            },
        )

    _write_json(
        reporting_root / "state" / "current.json",
        {
            "schema_version": "wind_forecast.monitoring_report_state.v2",
            "latest_report_id": report_ids[-1],
            "latest_through_date": (report_base + timedelta(days=profile.reports - 1)).isoformat(),
            "active_alerts": {},
            "rules": {},
        },
    )
    interval_start = alert_base + timedelta(days=500 if profile.alerts >= 1_000 else 0)
    return FixtureSelection(
        latest_report_id=report_ids[-1],
        report_id=report_ids[len(report_ids) // 2],
        reporting_run_id=run_ids[len(run_ids) // 2],
        alert_event_id=alert_ids[len(alert_ids) // 2],
        alert_start=interval_start,
        alert_end=interval_start + timedelta(days=6),
        alert_limit=min(50, max(1, profile.alerts // 8)),
        alert_offset=min(100, max(0, profile.alerts // 10)),
    )


def _head_membership(table_alias: str) -> str:
    return (
        f"JOIN operational_projection.generation_evidence ge ON ge.evidence_record_id = {table_alias}.evidence_record_id "
        "JOIN operational_projection.projection_head ph ON ph.generation_id = ge.generation_id "
        "AND ph.environment_id = %s "
    )


def build_query_cases(root: Path, selection: FixtureSelection) -> tuple[QueryCase, ...]:
    report_path = (
        root / "reporting" / "reports" / selection.report_id / "report.json"
    )

    def load_alerts() -> list[dict[str, Any]]:
        return load_alert_history(root)

    def alert_interval(history: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, ...], ...]:
        rows = [
            item
            for item in history
            if selection.alert_start
            <= date.fromisoformat(str(item["through_date"]))
            <= selection.alert_end
        ]
        rows = rows[
            selection.alert_offset : selection.alert_offset + selection.alert_limit
        ]
        return tuple((str(item["alert_event_id"]),) for item in rows)

    def exact_alert(history: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (str(item["alert_event_id"]),)
            for item in history
            if item["alert_event_id"] == selection.alert_event_id
        )

    def load_attempts() -> list[dict[str, Any]]:
        return load_reporting_attempts(root)

    def attempt_by_run(attempts: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, ...], ...]:
        matches = [
            item for item in attempts if item.get("run_id") == selection.reporting_run_id
        ]
        if len(matches) > 1:
            raise ValueError("Synthetic reporting-run identity is not unique.")
        item = matches[0] if matches else None
        return () if item is None else ((str(item["run_id"]), item.get("report_id")),)

    def attempt_by_report(attempts: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, ...], ...]:
        matches = [item for item in attempts if item.get("report_id") == selection.report_id]
        if len(matches) > 1:
            raise ValueError("Synthetic report identity is not unique.")
        item = matches[0] if matches else None
        return () if item is None else ((str(item["run_id"]), str(item["report_id"])),)

    def load_report() -> dict[str, Any]:
        return load_monitoring_report(report_path)

    def performance(report: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
        metrics = report["windows"]["30"]["performance"]["metrics"]
        return tuple(
            (selection.report_id, 30, metric)
            for metric in PERFORMANCE_METRICS
            if metric in metrics
        )

    def drift(report: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
        result: list[tuple[Any, ...]] = []
        position = 0
        for feature in sorted(report["windows"]["90"]["feature_drift"]):
            comparisons = report["windows"]["90"]["feature_drift"][feature]
            for comparator in sorted(comparisons):
                for detector in DETECTORS:
                    if detector in comparisons[comparator]:
                        result.append(
                            (
                                selection.report_id,
                                90,
                                position,
                                feature,
                                comparator,
                                detector,
                            )
                        )
                        position += 1
        return tuple(result)

    member_alert = _head_membership("ae")
    member_attempt = _head_membership("ra")
    member_metric = _head_membership("pm")
    member_drift = _head_membership("dm")
    metric_order = "CASE pm.metric_name " + " ".join(
        f"WHEN '{name}' THEN {index}" for index, name in enumerate(PERFORMANCE_METRICS)
    ) + " END"
    return (
        QueryCase(
            "alert_interval_pagination",
            "alerts",
            load_alerts,
            alert_interval,
            "SELECT ae.alert_event_id FROM operational_projection.alert_event ae "
            + member_alert
            + "WHERE ae.through_date BETWEEN %s AND %s "
            "ORDER BY ae.through_date, ae.rule_id, ae.alert_event_id LIMIT %s OFFSET %s",
            (
                "local",
                selection.alert_start,
                selection.alert_end,
                selection.alert_limit,
                selection.alert_offset,
            ),
            ("alert_event_date_idx",),
            True,
        ),
        QueryCase(
            "exact_alert_id",
            "alerts",
            load_alerts,
            exact_alert,
            "SELECT ae.alert_event_id FROM operational_projection.alert_event ae "
            + member_alert
            + "WHERE ae.alert_event_id = %s",
            ("local", selection.alert_event_id),
            ("alert_event_pkey",),
        ),
        QueryCase(
            "reporting_run_by_run_id",
            "attempts",
            load_attempts,
            attempt_by_run,
            "SELECT ra.reporting_run_id, ra.report_id FROM operational_projection.reporting_attempt ra "
            + member_attempt
            + "WHERE ra.reporting_run_id = %s",
            ("local", selection.reporting_run_id),
            ("reporting_attempt_pkey",),
        ),
        QueryCase(
            "reporting_run_by_report_id",
            "attempts",
            load_attempts,
            attempt_by_report,
            "SELECT ra.reporting_run_id, ra.report_id FROM operational_projection.reporting_attempt ra "
            + member_attempt
            + "WHERE ra.report_id = %s",
            ("local", selection.report_id),
            ("reporting_attempt_report_id_key",),
            True,
        ),
        QueryCase(
            "performance_report_window",
            "report",
            load_report,
            performance,
            "SELECT pm.report_id, pm.window_days, pm.metric_name "
            "FROM operational_projection.performance_metric pm "
            + member_metric
            + f"WHERE pm.report_id = %s AND pm.window_days = %s ORDER BY {metric_order}",
            ("local", selection.report_id, 30),
            ("performance_metric_pkey",),
        ),
        QueryCase(
            "drift_report_window",
            "report",
            load_report,
            drift,
            "SELECT dm.report_id, dm.window_days, dm.position, dm.feature, dm.comparator, dm.detector "
            "FROM operational_projection.drift_measurement dm "
            + member_drift
            + "WHERE dm.report_id = %s AND dm.window_days = %s ORDER BY dm.position",
            ("local", selection.report_id, 90),
            ("drift_measurement_pkey",),
        ),
    )


def _index_names(plan: Any) -> tuple[str, ...]:
    names: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            index = value.get("Index Name")
            if isinstance(index, str):
                names.add(index)
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                visit(child)

    visit(plan)
    return tuple(sorted(names))


def evaluate_gate(
    cases: Mapping[str, Mapping[str, Any]], *, enforce_timing: bool, enforce_plans: bool
) -> tuple[str, tuple[str, ...]]:
    failures: list[str] = []
    for name, result in cases.items():
        if not result["equivalent"]:
            failures.append(f"{name}:identity_order_mismatch")
        if enforce_timing and result["postgres_max_ms"] >= DEADLINE_MS:
            failures.append(f"{name}:deadline")
        if enforce_timing and result["speed_gate"] and result["speedup"] < MINIMUM_SPEEDUP:
            failures.append(f"{name}:speedup")
        if enforce_plans and not result["expected_indexes_used"]:
            failures.append(f"{name}:index")
    return ("GO" if not failures else "NO-GO", tuple(failures))


def _postgres_rows(connection: Any, case: QueryCase) -> tuple[tuple[Any, ...], ...]:
    with connection.cursor() as cursor:
        cursor.execute(case.sql, case.parameters)
        return tuple(tuple(row) for row in cursor.fetchall())


def _measure_cases(
    connection: Any,
    cases: Sequence[QueryCase],
    repetitions: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[QueryCase]] = {}
    for case in cases:
        grouped.setdefault(case.filesystem_group, []).append(case)
    expected: dict[str, tuple[tuple[Any, ...], ...]] = {}
    equivalent = {case.name: True for case in cases}
    filesystem_samples = {case.name: [] for case in cases}
    postgres_samples = {case.name: [] for case in cases}

    for grouped_cases in grouped.values():
        payload = grouped_cases[0].filesystem_loader()
        for case in grouped_cases:
            expected[case.name] = case.filesystem_selector(payload)
    for case in cases:
        equivalent[case.name] = (
            _postgres_rows(connection, case) == expected[case.name]
        )

    for repetition in range(repetitions):
        phases = ("filesystem", "postgres")
        if repetition % 2:
            phases = tuple(reversed(phases))
        for phase in phases:
            if phase == "filesystem":
                for grouped_cases in grouped.values():
                    started = perf_counter_ns()
                    payload = grouped_cases[0].filesystem_loader()
                    load_elapsed = (perf_counter_ns() - started) / 1_000_000
                    for case in grouped_cases:
                        selected_at = perf_counter_ns()
                        observed = case.filesystem_selector(payload)
                        select_elapsed = (perf_counter_ns() - selected_at) / 1_000_000
                        filesystem_samples[case.name].append(
                            load_elapsed + select_elapsed
                        )
                        equivalent[case.name] = (
                            equivalent[case.name]
                            and observed == expected[case.name]
                        )
            else:
                for case in cases:
                    started = perf_counter_ns()
                    observed = _postgres_rows(connection, case)
                    postgres_samples[case.name].append(
                        (perf_counter_ns() - started) / 1_000_000
                    )
                    equivalent[case.name] = (
                        equivalent[case.name]
                        and observed == expected[case.name]
                    )

    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        with connection.cursor() as cursor:
            cursor.execute(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + case.sql,
                case.parameters,
            )
            explain = cursor.fetchone()[0]
        indexes = _index_names(explain)
        filesystem_median = statistics.median(filesystem_samples[case.name])
        postgres_median = statistics.median(postgres_samples[case.name])
        speedup = (
            (filesystem_median - postgres_median) / filesystem_median
            if filesystem_median > 0
            else 0.0
        )
        results[case.name] = {
            "equivalent": equivalent[case.name],
            "result_count": len(expected[case.name]),
            "result_sha256": sha256(
                json.dumps(
                    expected[case.name], default=str, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "filesystem_median_ms": round(filesystem_median, 6),
            "postgres_median_ms": round(postgres_median, 6),
            "postgres_max_ms": round(max(postgres_samples[case.name]), 6),
            "speedup": round(speedup, 6),
            "speed_gate": case.speed_gate,
            "indexes_used": indexes,
            "expected_indexes": case.expected_indexes,
            "expected_indexes_used": all(
                index in indexes for index in case.expected_indexes
            ),
        }
    return results


def _analyze(migrator_dsn: str) -> None:
    import psycopg

    with psycopg.connect(migrator_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE wf_projection_owner")
            cursor.execute("ANALYZE operational_projection.alert_event")
            cursor.execute("ANALYZE operational_projection.reporting_attempt")
            cursor.execute("ANALYZE operational_projection.performance_metric")
            cursor.execute("ANALYZE operational_projection.drift_measurement")


def _copy_rows(cursor: Any, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    statement = (
        f"COPY operational_projection.{table} ({', '.join(columns)}) FROM STDIN"
    )
    with cursor.copy(statement) as copy:
        for row in rows:
            copy.write_row(row)


def _bulk_publish_snapshot(writer_dsn: str, snapshot: Any) -> None:
    """Seed one clean ephemeral benchmark database through writer privileges."""
    import psycopg

    evidence_ids = {
        record.identity: index
        for index, record in enumerate(snapshot.manifest.evidence, start=1)
    }
    counts = snapshot.counts()
    with psycopg.connect(
        writer_dsn,
        application_name="wind_forecast_projection_benchmark_writer",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TIME ZONE 'UTC'")
            cursor.execute("SET statement_timeout = '30s'")
            cursor.execute("SELECT count(*) FROM operational_projection.projection_generation")
            if int(cursor.fetchone()[0]) != 0:
                raise RuntimeError("Benchmark database is not empty.")
            _copy_rows(
                cursor,
                "evidence_record",
                (
                    "evidence_record_id",
                    "domain",
                    "source_kind",
                    "schema_version",
                    "record_id",
                    "sha256",
                    "effective_at",
                    "observed_at_utc",
                ),
                (
                    (
                        evidence_ids[record.identity],
                        record.identity.domain,
                        record.identity.source_kind,
                        record.identity.schema_version,
                        record.identity.record_id,
                        record.identity.sha256,
                        record.effective_at,
                        record.observed_at_utc,
                    )
                    for record in snapshot.manifest.evidence
                ),
            )
            generation_columns = (
                "generation_id",
                "environment_id",
                "contract_version",
                "schema_version",
                "projector_version",
                "source_git_commit",
                "source_set_sha256",
                "evidence_record_count",
                "generation_evidence_count",
                "model_era_count",
                "monitoring_report_count",
                "quality_issue_count",
                "monitoring_window_count",
                "performance_metric_count",
                "drift_measurement_count",
                "alert_event_count",
                "active_alert_snapshot_count",
                "reporting_attempt_count",
                "lineage_edge_count",
            )
            cursor.execute(
                "INSERT INTO operational_projection.projection_generation ("
                + ", ".join(generation_columns)
                + ") VALUES ("
                + ", ".join(["%s"] * len(generation_columns))
                + ")",
                (
                    snapshot.generation_id,
                    snapshot.manifest.environment_id,
                    snapshot.manifest.contract_version,
                    snapshot.manifest.schema_version,
                    snapshot.manifest.projector_version,
                    snapshot.manifest.source_git_commit,
                    snapshot.manifest.source_set_sha256,
                    *(counts[name] for name in generation_columns[7:]),
                ),
            )
            _copy_rows(
                cursor,
                "generation_evidence",
                ("generation_id", "evidence_record_id"),
                (
                    (snapshot.generation_id, evidence_ids[record.identity])
                    for record in snapshot.manifest.evidence
                ),
            )
            for table in TABLE_ORDER:
                table_rows = snapshot.rows_for(table)
                if not table_rows:
                    continue
                columns = tuple(
                    sorted(
                        {
                            *table_rows[0].value_map(),
                            *(link.column for link in table_rows[0].evidence_links),
                        }
                    )
                )

                def values() -> Iterable[tuple[Any, ...]]:
                    for relational_row in table_rows:
                        value_map = relational_row.value_map()
                        value_map.update(
                            {
                                link.column: evidence_ids[link.evidence]
                                for link in relational_row.evidence_links
                            }
                        )
                        yield tuple(value_map[column] for column in columns)

                _copy_rows(cursor, table, columns, values())
            ready_at = datetime.now(timezone.utc)
            cursor.execute(
                "UPDATE operational_projection.projection_generation "
                "SET ready_at_utc = %s WHERE generation_id = %s",
                (ready_at, snapshot.generation_id),
            )
            cursor.execute(
                "INSERT INTO operational_projection.projection_head "
                "(environment_id, generation_id, published_at_utc) VALUES (%s, %s, %s)",
                ("local", snapshot.generation_id, ready_at),
            )


def run_benchmark(profile: Profile) -> dict[str, Any]:
    environment = os.environ.get(ENVIRONMENT_ENV, "local")
    if environment != "local":
        raise ValueError("Benchmark environment is unsupported.")
    dsns = {
        "migrator": os.environ.get(MIGRATOR_DSN_ENV, ""),
        "writer": os.environ.get(WRITER_DSN_ENV, ""),
        "reader": os.environ.get(READER_DSN_ENV, ""),
    }
    if not all(dsns.values()):
        raise ValueError("Required benchmark database configuration is unavailable.")
    import psycopg

    migrate(dsns["migrator"])
    with TemporaryDirectory(prefix="wf-projection-benchmark-") as temporary:
        root = Path(temporary)
        selection = generate_synthetic_store(root, profile)
        source_commit = resolve_source_git_commit()
        snapshot = build_projection_snapshot(
            root,
            environment_id="local",
            source_git_commit=source_commit,
        )
        _bulk_publish_snapshot(dsns["writer"], snapshot)
        _analyze(dsns["migrator"])
        with psycopg.connect(
            dsns["reader"],
            autocommit=True,
            application_name="wind_forecast_projection_benchmark_reader",
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET TIME ZONE 'UTC'")
                cursor.execute("SET statement_timeout = '5s'")
                cursor.execute("SHOW server_version")
                postgres_version = str(cursor.fetchone()[0])
            query_cases = build_query_cases(root, selection)
            cases = _measure_cases(connection, query_cases, profile.repetitions)
    decision, failures = evaluate_gate(
        cases,
        enforce_timing=profile.enforce_timing_gate,
        enforce_plans=profile.enforce_plan_gate,
    )
    return {
        "schema_version": "wind_forecast.operational_projection_benchmark.v1",
        "profile": profile.name,
        "decision": decision,
        "failures": failures,
        "dataset": {
            "reports": profile.reports,
            "reporting_attempts": profile.attempts,
            "alert_events": profile.alerts,
            "drift_measurements": profile.drift_measurements,
            "monitoring_windows": profile.reports * 2,
            "performance_metrics": profile.reports * 2 * len(PERFORMANCE_METRICS),
            "repetitions": profile.repetitions,
        },
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "postgresql": postgres_version,
            "psycopg": psycopg.__version__,
            "source_git_commit": source_commit,
        },
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_benchmark(PROFILES[args.profile])
    except Exception:
        print(
            json.dumps(
                {
                    "schema_version": "wind_forecast.operational_projection_benchmark.v1",
                    "profile": args.profile,
                    "decision": "ERROR",
                    "error": "Benchmark setup or execution failed.",
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), default=str))
    return 0 if summary["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
