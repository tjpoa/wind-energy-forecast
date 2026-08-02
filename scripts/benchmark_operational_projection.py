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
import secrets
import shutil
import statistics
import subprocess
import sys
from tempfile import gettempdir, TemporaryDirectory
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
DEFAULT_MAX_RUNTIME_SECONDS = 3_600.0
COPY_DEADLINE_CHECK_INTERVAL = 4_096
GENERATION_EVIDENCE_PROBE_METHODS = (
    "binary_copy",
    "text_copy",
    "insert_select",
)
GENERATION_EVIDENCE_PROBE_MAX_ASSOCIATIONS = 61_004
GENERATION_EVIDENCE_PROBE_SCHEMA_VERSION = (
    "wind_forecast.operational_projection_generation_evidence_probe.v1"
)
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
ANALYZE_TABLES = (
    "alert_event",
    "reporting_attempt",
    "performance_metric",
    "drift_measurement",
)


@dataclass(frozen=True)
class CopySpec:
    """Allowlisted binary COPY columns and exact PostgreSQL input types."""

    columns: tuple[str, ...]
    postgres_types: tuple[str, ...]


COPY_SPECS = {
    "evidence_record": CopySpec(
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
        ("int8", "text", "text", "text", "text", "bpchar", "text", "timestamptz"),
    ),
    "generation_evidence": CopySpec(
        ("generation_id", "evidence_record_id"),
        ("bpchar", "int8"),
    ),
    "model_era": CopySpec(
        (
            "model_era_id",
            "evidence_record_id",
            "association_kind",
            "deployment_id",
            "deployment_generation",
            "registered_model_name",
            "model_version",
            "fit_cutoff",
            "activation_cutoff",
            "bundle_sha256",
            "model_sha256",
            "dataset_sha256",
            "feature_schema_sha256",
            "calibration_sha256",
            "ledger_sha256",
            "calibration_id",
            "reference_id",
        ),
        (
            "text",
            "int8",
            "text",
            "text",
            "int8",
            "text",
            "text",
            "date",
            "date",
            "bpchar",
            "bpchar",
            "bpchar",
            "bpchar",
            "bpchar",
            "bpchar",
            "text",
            "text",
        ),
    ),
    "monitoring_report": CopySpec(
        (
            "report_id",
            "evidence_record_id",
            "reporting_run_id",
            "created_at_utc",
            "through_date",
            "source_run_id",
            "source_status",
            "calibration_id",
            "reference_id",
            "policy_sha256",
            "quality_status",
            "batch_status",
            "verdict",
            "watermark_date",
            "watermark_age_days",
            "objective_days",
            "late_days",
            "objective_missed",
            "unresolved_late_date_count",
            "date_count",
            "ren_complete_count",
            "era5_complete_count",
            "integration_ready_count",
            "feature_ready_count",
            "model_era_id",
        ),
        (
            "text",
            "int8",
            "text",
            "timestamptz",
            "date",
            "text",
            "text",
            "text",
            "text",
            "bpchar",
            "text",
            "text",
            "text",
            "date",
            "int4",
            "int4",
            "int4",
            "bool",
            "int4",
            "int4",
            "int4",
            "int4",
            "int4",
            "int4",
            "text",
        ),
    ),
    "quality_issue": CopySpec(
        ("report_id", "position", "evidence_record_id", "code", "severity"),
        ("text", "int4", "int8", "text", "text"),
    ),
    "monitoring_window": CopySpec(
        (
            "report_id",
            "window_days",
            "evidence_record_id",
            "status",
            "sample_count",
            "coverage_ratio",
            "coverage_severity",
            "minimum_samples",
            "calendar_start",
            "calendar_end",
        ),
        ("text", "int4", "int8", "text", "int4", "float8", "text", "int4", "date", "date"),
    ),
    "performance_metric": CopySpec(
        (
            "report_id",
            "window_days",
            "evidence_record_id",
            "metric_name",
            "value",
            "value_status",
            "severity",
            "warning_threshold",
            "critical_threshold",
            "direction",
            "unit_or_scale",
        ),
        (
            "text",
            "int4",
            "int8",
            "text",
            "float8",
            "text",
            "text",
            "float8",
            "float8",
            "text",
            "text",
        ),
    ),
    "drift_measurement": CopySpec(
        (
            "report_id",
            "window_days",
            "position",
            "evidence_record_id",
            "feature",
            "comparator",
            "detector",
            "value",
            "severity",
            "warning_threshold",
            "critical_threshold",
            "direction",
        ),
        (
            "text",
            "int4",
            "int4",
            "int8",
            "text",
            "text",
            "text",
            "float8",
            "text",
            "float8",
            "float8",
            "text",
        ),
    ),
    "alert_event": CopySpec(
        (
            "alert_event_id",
            "evidence_record_id",
            "rule_id",
            "through_date",
            "event_type",
            "severity",
            "previous_alert_event_id",
        ),
        ("text", "int8", "text", "date", "text", "text", "text"),
    ),
    "active_alert_snapshot": CopySpec(
        ("generation_id", "rule_id", "evidence_record_id", "alert_event_id"),
        ("bpchar", "text", "int8", "text"),
    ),
    "reporting_attempt": CopySpec(
        (
            "reporting_run_id",
            "evidence_record_id",
            "attempted_at_utc",
            "through_date",
            "source_run_id",
            "source_status",
            "status",
            "report_id",
            "active_alert_count",
            "failure_at_utc",
            "failure_type",
            "failure_message",
        ),
        (
            "text",
            "int8",
            "timestamptz",
            "date",
            "text",
            "text",
            "text",
            "text",
            "int4",
            "timestamptz",
            "text",
            "text",
        ),
    ),
    "lineage_edge": CopySpec(
        (
            "generation_id",
            "edge_type",
            "source_evidence_record_id",
            "target_evidence_record_id",
            "position",
            "evidence_record_id",
        ),
        ("bpchar", "text", "int8", "int8", "int4", "int8"),
    ),
}


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


class BenchmarkNoGo(RuntimeError):
    """Fail-closed benchmark termination with sanitized failure identities."""

    def __init__(self, *failures: str) -> None:
        super().__init__("Benchmark readiness gate failed.")
        self.failures = failures


def _emit_progress(phase: str, **values: int | float | str) -> None:
    print(
        json.dumps(
            {"event": "benchmark_progress", "phase": phase, **values},
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


def _check_runtime(deadline_ns: int | None, phase: str) -> None:
    if deadline_ns is not None and perf_counter_ns() >= deadline_ns:
        raise BenchmarkNoGo(f"benchmark_runtime:{phase}")


def _raise_statement_timeout(exc: Exception, failure: str) -> None:
    if getattr(exc, "sqlstate", None) == "57014":
        raise BenchmarkNoGo(failure) from None


def _cleanup_worker_stores(token: str, *, temp_root: Path | None = None) -> None:
    if len(token) != 16 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError("Benchmark workspace token is invalid.")
    root = (temp_root or Path(gettempdir())).resolve()
    prefix = f"wf-projection-benchmark-{token}-"
    for candidate in root.glob(f"{prefix}*"):
        resolved = candidate.resolve()
        if resolved.parent != root or not resolved.name.startswith(prefix):
            raise RuntimeError("Benchmark temporary cleanup target is invalid.")
        if resolved.is_dir():
            shutil.rmtree(resolved)


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
            "WITH interval_alerts AS MATERIALIZED ("
            "SELECT alert_event_id, evidence_record_id, through_date, rule_id "
            "FROM operational_projection.alert_event "
            "WHERE through_date BETWEEN %s AND %s"
            ") "
            "SELECT ia.alert_event_id FROM interval_alerts ia "
            "JOIN operational_projection.generation_evidence ge "
            "ON ge.evidence_record_id = ia.evidence_record_id "
            "JOIN operational_projection.projection_head ph "
            "ON ph.generation_id = ge.generation_id AND ph.environment_id = %s "
            "ORDER BY ia.through_date, ia.rule_id, ia.alert_event_id "
            "LIMIT %s OFFSET %s",
            (
                selection.alert_start,
                selection.alert_end,
                "local",
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
    *,
    deadline_ns: int | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[QueryCase]] = {}
    for case in cases:
        grouped.setdefault(case.filesystem_group, []).append(case)
    expected: dict[str, tuple[tuple[Any, ...], ...]] = {}
    equivalent = {case.name: True for case in cases}
    filesystem_samples = {case.name: [] for case in cases}
    postgres_samples = {case.name: [] for case in cases}

    _emit_progress("query_warmup_started")
    for group_name, grouped_cases in grouped.items():
        _check_runtime(deadline_ns, f"warmup_{group_name}")
        payload = grouped_cases[0].filesystem_loader()
        for case in grouped_cases:
            expected[case.name] = case.filesystem_selector(payload)
    for case in cases:
        try:
            equivalent[case.name] = (
                _postgres_rows(connection, case) == expected[case.name]
            )
        except Exception as exc:
            _raise_statement_timeout(exc, f"{case.name}:deadline")
            raise
    _emit_progress("query_warmup_completed")

    for repetition in range(repetitions):
        _check_runtime(deadline_ns, f"repetition_{repetition + 1}")
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
                    try:
                        observed = _postgres_rows(connection, case)
                    except Exception as exc:
                        _raise_statement_timeout(exc, f"{case.name}:deadline")
                        raise
                    postgres_samples[case.name].append(
                        (perf_counter_ns() - started) / 1_000_000
                    )
                    equivalent[case.name] = (
                        equivalent[case.name]
                        and observed == expected[case.name]
                    )
        _emit_progress(
            "query_repetition_completed",
            repetition=repetition + 1,
            repetitions=repetitions,
        )

    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        _check_runtime(deadline_ns, f"explain_{case.name}")
        with connection.cursor() as cursor:
            try:
                cursor.execute(
                    "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + case.sql,
                    case.parameters,
                )
            except Exception as exc:
                _raise_statement_timeout(exc, f"{case.name}:deadline")
                raise
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


def _run_publication_step(
    step: str,
    timings: dict[str, float],
    deadline_ns: int | None,
    action: Callable[[], Any],
    *,
    row_count: int | None = None,
    check_deadline_after: bool = True,
) -> Any:
    phase = f"snapshot_publish:{step}"
    _check_runtime(deadline_ns, phase)
    progress: dict[str, int | float | str] = {"step": step}
    if row_count is not None:
        progress["row_count"] = row_count
    _emit_progress("snapshot_publish_step_started", **progress)
    started_ns = perf_counter_ns()
    try:
        result = action()
    except Exception as exc:
        _raise_statement_timeout(exc, f"{phase}:deadline")
        raise
    elapsed_ms = round((perf_counter_ns() - started_ns) / 1_000_000, 3)
    timings[step] = elapsed_ms
    _emit_progress(
        "snapshot_publish_step_completed",
        **progress,
        elapsed_ms=elapsed_ms,
    )
    if check_deadline_after:
        _check_runtime(deadline_ns, phase)
    return result


def _analyze(
    migrator_dsn: str,
    *,
    deadline_ns: int | None,
) -> dict[str, float]:
    import psycopg

    timings: dict[str, float] = {}
    with psycopg.connect(migrator_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE wf_projection_owner")
            for table in ANALYZE_TABLES:
                _run_publication_step(
                    f"analyze_{table}",
                    timings,
                    deadline_ns,
                    lambda table=table: cursor.execute(
                        f"ANALYZE operational_projection.{table}"
                    ),
                )
    return timings


def _copy_rows(
    cursor: Any,
    table: str,
    rows: Iterable[Sequence[Any]],
    *,
    deadline_ns: int | None,
) -> int:
    spec = COPY_SPECS.get(table)
    if spec is None:
        raise RuntimeError("Benchmark COPY table is unsupported.")
    statement = (
        f"COPY operational_projection.{table} ({', '.join(spec.columns)}) "
        "FROM STDIN (FORMAT BINARY)"
    )
    written = 0
    with cursor.copy(statement) as copy:
        copy.set_types(spec.postgres_types)
        for written, row in enumerate(rows, start=1):
            copy.write_row(row)
            if written % COPY_DEADLINE_CHECK_INTERVAL == 0:
                _check_runtime(deadline_ns, f"snapshot_publish:copy_{table}")
    return written


def _register_binary_copy_dumpers(connection: Any) -> None:
    """Register COPY-only binary adaptation missing from Psycopg defaults."""
    from psycopg.types.string import StrBinaryDumper

    class BpcharBinaryDumper(StrBinaryDumper):
        pass

    BpcharBinaryDumper.oid = connection.adapters.types["bpchar"].oid
    connection.adapters.register_dumper(None, BpcharBinaryDumper)


def _generation_evidence_identity_sha256(association_count: int) -> str:
    digest = sha256()
    for evidence_record_id in range(1, association_count + 1):
        digest.update(evidence_record_id.to_bytes(8, "big"))
    return digest.hexdigest()


def _start_probe_step(
    method: str,
    association_count: int,
    step: str,
    deadline_ns: int | None,
) -> int:
    _check_runtime(deadline_ns, f"generation_evidence_probe:{step}")
    _emit_progress(
        "generation_evidence_probe_step_started",
        method=method,
        association_count=association_count,
        step=step,
    )
    return perf_counter_ns()


def _finish_probe_step(
    method: str,
    association_count: int,
    step: str,
    started_ns: int,
    deadline_ns: int | None,
    timings: dict[str, float],
) -> None:
    elapsed_ms = round((perf_counter_ns() - started_ns) / 1_000_000, 3)
    timings[step] = elapsed_ms
    _emit_progress(
        "generation_evidence_probe_step_completed",
        method=method,
        association_count=association_count,
        step=step,
        elapsed_ms=elapsed_ms,
    )
    _check_runtime(deadline_ns, f"generation_evidence_probe:{step}")


def _copy_generation_evidence_probe(
    cursor: Any,
    *,
    method: str,
    generation_id: str,
    association_count: int,
    deadline_ns: int | None,
    timings: dict[str, float],
) -> int:
    if method not in {"binary_copy", "text_copy"}:
        raise ValueError("Generation-evidence COPY probe method is unsupported.")
    statement = (
        "COPY operational_projection.generation_evidence "
        "(generation_id, evidence_record_id) FROM STDIN"
    )
    if method == "binary_copy":
        statement += " (FORMAT BINARY)"

    opened_ns = _start_probe_step(
        method, association_count, "copy_open", deadline_ns
    )
    with cursor.copy(statement) as copy:
        _finish_probe_step(
            method,
            association_count,
            "copy_open",
            opened_ns,
            deadline_ns,
            timings,
        )
        if method == "binary_copy":
            configured_ns = _start_probe_step(
                method, association_count, "copy_configure", deadline_ns
            )
            copy.set_types(COPY_SPECS["generation_evidence"].postgres_types)
            _finish_probe_step(
                method,
                association_count,
                "copy_configure",
                configured_ns,
                deadline_ns,
                timings,
            )

        first_ns = _start_probe_step(
            method, association_count, "copy_first_row", deadline_ns
        )
        copy.write_row((generation_id, 1))
        _finish_probe_step(
            method,
            association_count,
            "copy_first_row",
            first_ns,
            deadline_ns,
            timings,
        )

        remaining_ns = _start_probe_step(
            method, association_count, "copy_remaining_rows", deadline_ns
        )
        for evidence_record_id in range(2, association_count + 1):
            copy.write_row((generation_id, evidence_record_id))
            if evidence_record_id % COPY_DEADLINE_CHECK_INTERVAL == 0:
                _check_runtime(
                    deadline_ns,
                    "generation_evidence_probe:copy_remaining_rows",
                )
                _emit_progress(
                    "generation_evidence_probe_batch_completed",
                    method=method,
                    association_count=association_count,
                    rows_written=evidence_record_id,
                )
        _finish_probe_step(
            method,
            association_count,
            "copy_remaining_rows",
            remaining_ns,
            deadline_ns,
            timings,
        )
        finalized_ns = _start_probe_step(
            method, association_count, "copy_finalize", deadline_ns
        )
    _finish_probe_step(
        method,
        association_count,
        "copy_finalize",
        finalized_ns,
        deadline_ns,
        timings,
    )
    return association_count


def _insert_select_generation_evidence_probe(
    cursor: Any,
    *,
    generation_id: str,
    association_count: int,
    deadline_ns: int | None,
    timings: dict[str, float],
) -> int:
    method = "insert_select"
    inserted_ns = _start_probe_step(
        method, association_count, "insert_select", deadline_ns
    )
    cursor.execute(
        "INSERT INTO operational_projection.generation_evidence "
        "(generation_id, evidence_record_id) "
        "SELECT %s, evidence_record_id "
        "FROM operational_projection.evidence_record "
        "WHERE evidence_record_id BETWEEN %s AND %s "
        "ORDER BY evidence_record_id",
        (generation_id, 1, association_count),
    )
    inserted = int(cursor.rowcount)
    _finish_probe_step(
        method,
        association_count,
        "insert_select",
        inserted_ns,
        deadline_ns,
        timings,
    )
    return inserted


def _assert_probe_database_empty(cursor: Any) -> None:
    cursor.execute(
        "SELECT "
        "(SELECT count(*) FROM operational_projection.projection_generation), "
        "(SELECT count(*) FROM operational_projection.evidence_record), "
        "(SELECT count(*) FROM operational_projection.generation_evidence)"
    )
    if any(int(value) != 0 for value in cursor.fetchone()):
        raise RuntimeError("Generation-evidence probe database is not empty.")


def _seed_generation_evidence_probe(
    cursor: Any,
    *,
    generation_id: str,
    association_count: int,
    deadline_ns: int | None,
) -> None:
    _copy_rows(
        cursor,
        "evidence_record",
        (
            (
                evidence_record_id,
                "probe",
                "probe",
                "probe.v1",
                f"record-{evidence_record_id}",
                f"{evidence_record_id:064x}",
                "2026-08-02",
                None,
            )
            for evidence_record_id in range(1, association_count + 1)
        ),
        deadline_ns=deadline_ns,
    )
    columns = (
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
        + ", ".join(columns)
        + ") VALUES ("
        + ", ".join(["%s"] * len(columns))
        + ")",
        (
            generation_id,
            "local",
            "probe.v1",
            "probe.v1",
            "probe.v1",
            "0" * 40,
            "0" * 64,
            association_count,
            association_count,
            *(0 for _ in range(10)),
        ),
    )


def _verify_generation_evidence_probe(
    cursor: Any,
    *,
    generation_id: str,
    association_count: int,
) -> None:
    cursor.execute(
        "SELECT count(*) FROM operational_projection.generation_evidence "
        "WHERE generation_id = %s",
        (generation_id,),
    )
    if int(cursor.fetchone()[0]) != association_count:
        raise RuntimeError("Generation-evidence probe cardinality differs.")
    cursor.execute(
        "SELECT count(*) FROM generate_series(%s::bigint, %s::bigint) "
        "AS expected(id) "
        "WHERE NOT EXISTS ("
        "SELECT 1 FROM operational_projection.generation_evidence actual "
        "WHERE actual.generation_id = %s "
        "AND actual.evidence_record_id = expected.id) ",
        (1, association_count, generation_id),
    )
    if int(cursor.fetchone()[0]) != 0:
        raise RuntimeError("Generation-evidence probe identities differ.")


def _verify_generation_evidence_probe_rollback(reader_dsn: str) -> None:
    import psycopg

    with psycopg.connect(reader_dsn) as connection:
        with connection.cursor() as cursor:
            _assert_probe_database_empty(cursor)


def run_generation_evidence_probe(
    method: str,
    association_count: int,
    trial: int,
    *,
    max_runtime_seconds: float,
) -> dict[str, Any]:
    if method not in GENERATION_EVIDENCE_PROBE_METHODS:
        raise ValueError("Generation-evidence probe method is unsupported.")
    if not 1 <= association_count <= GENERATION_EVIDENCE_PROBE_MAX_ASSOCIATIONS:
        raise ValueError("Generation-evidence probe cardinality is invalid.")
    if not 1 <= trial <= 3:
        raise ValueError("Generation-evidence probe trial is invalid.")
    if os.environ.get(ENVIRONMENT_ENV, "local") != "local":
        raise ValueError("Generation-evidence probe environment is unsupported.")
    dsns = {
        "migrator": os.environ.get(MIGRATOR_DSN_ENV, ""),
        "writer": os.environ.get(WRITER_DSN_ENV, ""),
        "reader": os.environ.get(READER_DSN_ENV, ""),
    }
    if not all(dsns.values()):
        raise ValueError("Required probe database configuration is unavailable.")

    import psycopg

    started_ns = perf_counter_ns()
    deadline_ns = started_ns + int(max_runtime_seconds * 1_000_000_000)
    migrate(dsns["migrator"])
    _check_runtime(deadline_ns, "generation_evidence_probe:migration")
    generation_id = sha256(
        f"{method}:{association_count}:{trial}".encode("ascii")
    ).hexdigest()
    timings: dict[str, float] = {}
    written = 0
    with psycopg.connect(
        dsns["writer"],
        application_name="wind_forecast_generation_evidence_probe_writer",
    ) as connection:
        try:
            _register_binary_copy_dumpers(connection)
            with connection.cursor() as cursor:
                cursor.execute("SET TIME ZONE 'UTC'")
                cursor.execute("SET statement_timeout = '30s'")
                _assert_probe_database_empty(cursor)
                _seed_generation_evidence_probe(
                    cursor,
                    generation_id=generation_id,
                    association_count=association_count,
                    deadline_ns=deadline_ns,
                )
                if method in {"binary_copy", "text_copy"}:
                    written = _copy_generation_evidence_probe(
                        cursor,
                        method=method,
                        generation_id=generation_id,
                        association_count=association_count,
                        deadline_ns=deadline_ns,
                        timings=timings,
                    )
                else:
                    written = _insert_select_generation_evidence_probe(
                        cursor,
                        generation_id=generation_id,
                        association_count=association_count,
                        deadline_ns=deadline_ns,
                        timings=timings,
                    )
                if written != association_count:
                    raise RuntimeError("Generation-evidence probe write count differs.")
                _verify_generation_evidence_probe(
                    cursor,
                    generation_id=generation_id,
                    association_count=association_count,
                )
                _check_runtime(deadline_ns, "generation_evidence_probe:verify")
            connection.rollback()
        except Exception as exc:
            connection.rollback()
            _raise_statement_timeout(exc, "generation_evidence_probe:deadline")
            raise
    _verify_generation_evidence_probe_rollback(dsns["reader"])
    return {
        "schema_version": GENERATION_EVIDENCE_PROBE_SCHEMA_VERSION,
        "decision": "PASS",
        "method": method,
        "association_count": association_count,
        "trial": trial,
        "rows_written": written,
        "identity_sha256": _generation_evidence_identity_sha256(association_count),
        "timings_ms": timings,
        "total_runtime_ms": round((perf_counter_ns() - started_ns) / 1_000_000, 3),
        "rolled_back": True,
    }


def _group_snapshot_rows(snapshot: Any) -> tuple[dict[str, tuple[Any, ...]], dict[str, int]]:
    grouped: dict[str, list[Any]] = {table: [] for table in TABLE_ORDER}
    for row in snapshot.rows:
        grouped[row.table].append(row)
    frozen = {table: tuple(rows) for table, rows in grouped.items()}
    evidence_count = len(snapshot.manifest.evidence)
    counts = {
        "evidence_record_count": evidence_count,
        "generation_evidence_count": evidence_count,
        **{f"{table}_count": len(rows) for table, rows in frozen.items()},
    }
    return frozen, counts


def _prepare_snapshot_publication(
    snapshot: Any,
) -> tuple[str, dict[Any, int], dict[str, tuple[Any, ...]], dict[str, int]]:
    generation_id = snapshot.generation_id
    evidence_ids = {
        record.identity: index
        for index, record in enumerate(snapshot.manifest.evidence, start=1)
    }
    rows_by_table, counts = _group_snapshot_rows(snapshot)
    return generation_id, evidence_ids, rows_by_table, counts


def _bulk_publish_snapshot(
    writer_dsn: str,
    snapshot: Any,
    *,
    deadline_ns: int | None,
    failure_hook: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """Seed one clean ephemeral benchmark database through writer privileges."""
    import psycopg

    timings: dict[str, float] = {}

    generation_id, evidence_ids, rows_by_table, counts = _run_publication_step(
        "prepare",
        timings,
        deadline_ns,
        lambda: _prepare_snapshot_publication(snapshot),
        row_count=len(snapshot.rows),
    )
    with psycopg.connect(
        writer_dsn,
        application_name="wind_forecast_projection_benchmark_writer",
    ) as connection:
        try:
            _register_binary_copy_dumpers(connection)
            with connection.cursor() as cursor:
                cursor.execute("SET TIME ZONE 'UTC'")
                cursor.execute("SET statement_timeout = '30s'")
                cursor.execute(
                    "SELECT count(*) FROM "
                    "operational_projection.projection_generation"
                )
                if int(cursor.fetchone()[0]) != 0:
                    raise RuntimeError("Benchmark database is not empty.")

                _run_publication_step(
                    "copy_evidence_record",
                    timings,
                    deadline_ns,
                    lambda: _copy_rows(
                        cursor,
                        "evidence_record",
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
                        deadline_ns=deadline_ns,
                    ),
                    row_count=len(snapshot.manifest.evidence),
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

                def insert_generation() -> None:
                    cursor.execute(
                        "INSERT INTO operational_projection.projection_generation ("
                        + ", ".join(generation_columns)
                        + ") VALUES ("
                        + ", ".join(["%s"] * len(generation_columns))
                        + ")",
                        (
                            generation_id,
                            snapshot.manifest.environment_id,
                            snapshot.manifest.contract_version,
                            snapshot.manifest.schema_version,
                            snapshot.manifest.projector_version,
                            snapshot.manifest.source_git_commit,
                            snapshot.manifest.source_set_sha256,
                            *(counts[name] for name in generation_columns[7:]),
                        ),
                    )

                _run_publication_step(
                    "insert_generation",
                    timings,
                    deadline_ns,
                    insert_generation,
                    row_count=1,
                )
                _run_publication_step(
                    "copy_generation_evidence",
                    timings,
                    deadline_ns,
                    lambda: _copy_rows(
                        cursor,
                        "generation_evidence",
                        (
                            (
                                generation_id,
                                evidence_ids[record.identity],
                            )
                            for record in snapshot.manifest.evidence
                        ),
                        deadline_ns=deadline_ns,
                    ),
                    row_count=len(snapshot.manifest.evidence),
                )
                for table in TABLE_ORDER:
                    table_rows = rows_by_table[table]
                    spec = COPY_SPECS[table]
                    if table_rows:
                        first_columns = {
                            *table_rows[0].value_map(),
                            *(link.column for link in table_rows[0].evidence_links),
                        }
                        if first_columns != set(spec.columns):
                            raise RuntimeError(
                                "Benchmark COPY columns differ from the normalized snapshot."
                            )

                    def values(
                        table_rows: tuple[Any, ...] = table_rows,
                        spec: CopySpec = spec,
                    ) -> Iterable[tuple[Any, ...]]:
                        for relational_row in table_rows:
                            value_map = relational_row.value_map()
                            value_map.update(
                                {
                                    link.column: evidence_ids[link.evidence]
                                    for link in relational_row.evidence_links
                                }
                            )
                            yield tuple(
                                value_map[column] for column in spec.columns
                            )

                    _run_publication_step(
                        f"copy_{table}",
                        timings,
                        deadline_ns,
                        (
                            lambda table=table, table_rows=table_rows: (
                                _copy_rows(
                                    cursor,
                                    table,
                                    values(table_rows, COPY_SPECS[table]),
                                    deadline_ns=deadline_ns,
                                )
                                if table_rows
                                else 0
                            )
                        ),
                        row_count=len(table_rows),
                    )
                ready_at = datetime.now(timezone.utc)

                def publish_head() -> None:
                    cursor.execute(
                        "UPDATE operational_projection.projection_generation "
                        "SET ready_at_utc = %s WHERE generation_id = %s",
                        (ready_at, generation_id),
                    )
                    cursor.execute(
                        "INSERT INTO operational_projection.projection_head "
                        "(environment_id, generation_id, published_at_utc) "
                        "VALUES (%s, %s, %s)",
                        ("local", generation_id, ready_at),
                    )

                _run_publication_step(
                    "publish_head",
                    timings,
                    deadline_ns,
                    publish_head,
                    row_count=1,
                )
                if failure_hook is not None:
                    failure_hook("before_commit")
            _run_publication_step(
                "commit",
                timings,
                deadline_ns,
                connection.commit,
                check_deadline_after=False,
            )
        except Exception:
            connection.rollback()
            raise
    return timings


def run_benchmark(
    profile: Profile,
    *,
    max_runtime_seconds: float | None = None,
    workspace_token: str | None = None,
) -> dict[str, Any]:
    if workspace_token is not None and (
        len(workspace_token) != 16
        or any(character not in "0123456789abcdef" for character in workspace_token)
    ):
        raise ValueError("Benchmark workspace token is invalid.")
    started_ns = perf_counter_ns()
    deadline_ns = (
        None
        if max_runtime_seconds is None
        else started_ns + int(max_runtime_seconds * 1_000_000_000)
    )
    phase_timings_ms: dict[str, float] = {}
    snapshot_publish_steps_ms: dict[str, float] = {}

    def start_phase(name: str) -> int:
        _check_runtime(deadline_ns, name)
        _emit_progress(f"{name}_started")
        return perf_counter_ns()

    def finish_phase(name: str, phase_started_ns: int) -> None:
        elapsed_ms = (perf_counter_ns() - phase_started_ns) / 1_000_000
        phase_timings_ms[name] = round(elapsed_ms, 3)
        _emit_progress(f"{name}_completed", elapsed_ms=round(elapsed_ms, 3))
        _check_runtime(deadline_ns, name)

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

    phase_started_ns = start_phase("migration")
    try:
        migrate(dsns["migrator"])
    except Exception as exc:
        _raise_statement_timeout(exc, "migration:deadline")
        raise
    finish_phase("migration", phase_started_ns)
    temporary_prefix = (
        "wf-projection-benchmark-"
        if workspace_token is None
        else f"wf-projection-benchmark-{workspace_token}-"
    )
    with TemporaryDirectory(prefix=temporary_prefix) as temporary:
        root = Path(temporary)
        phase_started_ns = start_phase("fixture_generation")
        selection = generate_synthetic_store(root, profile)
        finish_phase("fixture_generation", phase_started_ns)
        source_commit = resolve_source_git_commit()
        phase_started_ns = start_phase("snapshot_build")
        snapshot = build_projection_snapshot(
            root,
            environment_id="local",
            source_git_commit=source_commit,
        )
        finish_phase("snapshot_build", phase_started_ns)
        phase_started_ns = start_phase("snapshot_publish")
        try:
            snapshot_publish_steps_ms.update(
                _bulk_publish_snapshot(
                    dsns["writer"],
                    snapshot,
                    deadline_ns=deadline_ns,
                )
            )
            snapshot_publish_steps_ms.update(
                _analyze(dsns["migrator"], deadline_ns=deadline_ns)
            )
        except Exception as exc:
            _raise_statement_timeout(exc, "snapshot_publish:deadline")
            raise
        finish_phase("snapshot_publish", phase_started_ns)
        with psycopg.connect(
            dsns["reader"],
            autocommit=True,
            application_name="wind_forecast_projection_benchmark_reader",
        ) as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SET TIME ZONE 'UTC'")
                    cursor.execute("SET statement_timeout = '5s'")
                    cursor.execute("SHOW server_version")
                    postgres_version = str(cursor.fetchone()[0])
            except Exception as exc:
                _raise_statement_timeout(exc, "reader_setup:deadline")
                raise
            query_cases = build_query_cases(root, selection)
            phase_started_ns = start_phase("query_measurement")
            cases = _measure_cases(
                connection,
                query_cases,
                profile.repetitions,
                deadline_ns=deadline_ns,
            )
            finish_phase("query_measurement", phase_started_ns)
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
        "phase_timings_ms": phase_timings_ms,
        "snapshot_publish_steps_ms": snapshot_publish_steps_ms,
        "total_runtime_ms": round((perf_counter_ns() - started_ns) / 1_000_000, 3),
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
    parser.add_argument(
        "--generation-evidence-probe-method",
        choices=GENERATION_EVIDENCE_PROBE_METHODS,
    )
    parser.add_argument("--generation-evidence-probe-count", type=int)
    parser.add_argument("--generation-evidence-probe-trial", type=int, default=1)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=DEFAULT_MAX_RUNTIME_SECONDS,
        help="Fail closed after this total runtime without weakening any query gate.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workspace-token", help=argparse.SUPPRESS)
    return parser


def _run_worker(args: argparse.Namespace) -> int:
    is_probe = args.generation_evidence_probe_method is not None
    try:
        if is_probe:
            summary = run_generation_evidence_probe(
                args.generation_evidence_probe_method,
                args.generation_evidence_probe_count,
                args.generation_evidence_probe_trial,
                max_runtime_seconds=args.max_runtime_seconds,
            )
        else:
            summary = run_benchmark(
                PROFILES[args.profile],
                max_runtime_seconds=args.max_runtime_seconds,
                workspace_token=args.workspace_token,
            )
    except BenchmarkNoGo as exc:
        payload: dict[str, Any] = {
            "schema_version": (
                GENERATION_EVIDENCE_PROBE_SCHEMA_VERSION
                if is_probe
                else "wind_forecast.operational_projection_benchmark.v1"
            ),
            "decision": "NO-GO",
            "failures": exc.failures,
        }
        if is_probe:
            payload.update(
                {
                    "method": args.generation_evidence_probe_method,
                    "association_count": args.generation_evidence_probe_count,
                    "trial": args.generation_evidence_probe_trial,
                }
            )
        else:
            payload["profile"] = args.profile
        print(json.dumps(payload, sort_keys=True))
        return 1
    except Exception:
        payload = {
            "schema_version": (
                GENERATION_EVIDENCE_PROBE_SCHEMA_VERSION
                if is_probe
                else "wind_forecast.operational_projection_benchmark.v1"
            ),
            "decision": "ERROR",
            "error": "Probe execution failed." if is_probe else "Benchmark setup or execution failed.",
        }
        if is_probe:
            payload.update(
                {
                    "method": args.generation_evidence_probe_method,
                    "association_count": args.generation_evidence_probe_count,
                    "trial": args.generation_evidence_probe_trial,
                }
            )
        else:
            payload["profile"] = args.profile
        print(json.dumps(payload, sort_keys=True))
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), default=str))
    return 0 if summary["decision"] in {"GO", "PASS"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    is_probe = args.generation_evidence_probe_method is not None
    if is_probe != (args.generation_evidence_probe_count is not None):
        parser.error("generation-evidence probe method and count are both required")
    if is_probe and not (
        1
        <= args.generation_evidence_probe_count
        <= GENERATION_EVIDENCE_PROBE_MAX_ASSOCIATIONS
    ):
        parser.error("generation-evidence probe count is outside the allowed range")
    if not 1 <= args.generation_evidence_probe_trial <= 3:
        parser.error("generation-evidence probe trial is outside the allowed range")
    if args.worker:
        return _run_worker(args)
    workspace_token = secrets.token_hex(8)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--max-runtime-seconds",
        str(args.max_runtime_seconds),
        "--worker",
    ]
    if is_probe:
        command.extend(
            [
                "--generation-evidence-probe-method",
                args.generation_evidence_probe_method,
                "--generation-evidence-probe-count",
                str(args.generation_evidence_probe_count),
                "--generation-evidence-probe-trial",
                str(args.generation_evidence_probe_trial),
            ]
        )
    else:
        command.extend(
            [
                "--profile",
                args.profile,
                "--workspace-token",
                workspace_token,
            ]
        )
    timed_out = False
    try:
        return_code = subprocess.run(
            command,
            check=False,
            timeout=args.max_runtime_seconds,
        ).returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        return_code = 1
    try:
        if not is_probe:
            _cleanup_worker_stores(workspace_token)
    except Exception:
        print(
            json.dumps(
                {
                    "schema_version": "wind_forecast.operational_projection_benchmark.v1",
                    "profile": args.profile,
                    "decision": "ERROR",
                    "error": "Benchmark temporary cleanup failed.",
                },
                sort_keys=True,
            )
        )
        return 2
    if timed_out:
        payload = {
            "schema_version": (
                GENERATION_EVIDENCE_PROBE_SCHEMA_VERSION
                if is_probe
                else "wind_forecast.operational_projection_benchmark.v1"
            ),
            "decision": "NO-GO",
            "failures": (
                "generation_evidence_probe:hard_timeout"
                if is_probe
                else "benchmark_runtime:hard_timeout",
            ),
        }
        if is_probe:
            payload.update(
                {
                    "method": args.generation_evidence_probe_method,
                    "association_count": args.generation_evidence_probe_count,
                    "trial": args.generation_evidence_probe_trial,
                }
            )
        else:
            payload["profile"] = args.profile
        print(json.dumps(payload, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
