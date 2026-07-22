"""Structured, read-only quality evidence for Phase 8 batch attempts."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wind_forecast.integration import (
    expected_era5_hourly_count,
    expected_ren_interval_count,
)


QUALITY_SCHEMA = "wind_forecast.batch_quality.v1"
REN_REQUIRED_COLUMNS = (
    "timestamp",
    "wind_production_mw",
    "unit",
    "source_date",
    "retrieval_timestamp_utc",
    "endpoint_identifier",
    "raw_response_sha256",
)
ERA5_REQUIRED_COLUMNS = (
    "timestamp_utc",
    "station_id",
    "station_name",
    "station_latitude",
    "station_longitude",
    "grid_latitude",
    "grid_longitude",
    "temperature_2m_k",
    "temperature_2m_c",
    "u10_m_s",
    "v10_m_s",
    "wind_speed_m_s",
    "wind_direction_deg_from",
    "is_calm_or_near_calm",
)
NUMERIC_SCHEMA_COLUMNS = {
    "wind_production_mw",
    "station_latitude",
    "station_longitude",
    "grid_latitude",
    "grid_longitude",
    "temperature_2m_k",
    "temperature_2m_c",
    "u10_m_s",
    "v10_m_s",
    "wind_speed_m_s",
    "wind_direction_deg_from",
}
HARD_QUALITY_CODES = {
    "schema_validation_failed",
    "missing_required_schema_columns",
    "schema_column_order_changed",
    "schema_column_type_changed",
    "duplicate_validation_failed",
    "null_validation_failed",
    "finiteness_validation_failed",
    "interval_validation_failed",
    "invalid_complete_interval_count",
    "timestamp_validation_failed",
    "source_checksum_mismatch",
}


def build_batch_quality_evidence(
    *,
    run_id: str,
    through_date: str,
    evaluated_at_utc: datetime,
    plan: Mapping[str, Any],
    state: Mapping[str, Any] | None,
    status: str,
    source_objective_days: int = 5,
    source_late_days: int = 7,
    hard_quality_tolerance: int = 0,
    policy_evidence: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Summarize quality for a published, no-op, or rejected batch attempt."""
    if evaluated_at_utc.tzinfo is None:
        raise ValueError("evaluated_at_utc must be timezone-aware.")
    through = date.fromisoformat(str(through_date))
    sources = dict((state or {}).get("sources") or {})
    partitions = dict((state or {}).get("partitions") or {})
    watermarks = dict((state or {}).get("watermarks") or {})
    issues: list[dict[str, Any]] = []

    selected_dates = sorted(
        set(plan.get("potentially_affected_dates") or [])
        | set(plan.get("ren_missing_dates") or [])
        | set(plan.get("ren_unavailable_dates") or [])
        | {
            item
            for values in (plan.get("pending_availability_dates") or {}).values()
            for item in values
        }
    )
    if not selected_dates:
        selected_dates = [through.isoformat()]

    freshness = _freshness(
        through,
        evaluated_at_utc,
        watermarks,
        sources,
        source_objective_days,
        source_late_days,
        selected_dates,
        issues,
    )
    coverage = _coverage(selected_dates, sources, partitions, issues)
    intervals = _intervals(selected_dates, sources, issues)
    schemas = _schemas(selected_dates, sources, partitions, issues)

    failed_input = _failed_input_evidence(error) if error else None
    classified_failure = _classify_failure(error, failed_input) if error else None
    if classified_failure:
        issues.append(classified_failure)
    checksums = _checksum_evidence(selected_dates, sources, failed_input)
    mismatches = [
        item for item in checksums["files"] if item.get("matches_recorded") is False
    ]
    if mismatches:
        issues.append(
            {
                "code": "source_checksum_mismatch",
                "severity": "critical",
                "count": len(mismatches),
                "sample": mismatches[:10],
            }
        )
    _apply_quality_tolerance(issues, hard_quality_tolerance)
    hard_issue_count = sum(
        1 for item in issues if item.get("severity") == "critical"
    )
    warning_count = sum(1 for item in issues if item.get("severity") == "warning")
    verdict = "FAIL" if hard_issue_count else (
        "PASS WITH WARNINGS" if warning_count else "PASS"
    )
    if status == "failed" and verdict == "PASS":
        verdict = "NOT EVALUATED"

    return {
        "schema_version": QUALITY_SCHEMA,
        "run_id": run_id,
        "batch_status": status,
        "through_date": through.isoformat(),
        "evaluated_at_utc": evaluated_at_utc.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "scope_dates": selected_dates,
        "policy": dict(policy_evidence or {}),
        "freshness": freshness,
        "coverage": coverage,
        "nulls_and_finiteness": {
            "null_count": 0 if status != "failed" else (failed_input or {}).get("null_count"),
            "non_finite_count": (
                0 if status != "failed" else (failed_input or {}).get("non_finite_count")
            ),
            "validated_by_incremental_contract": status != "failed",
        },
        "duplicates": {
            "duplicate_timestamp_count": (
                0 if status != "failed" else (failed_input or {}).get("duplicate_timestamp_count")
            ),
            "duplicate_date_count": (
                0 if status != "failed" else (failed_input or {}).get("duplicate_date_count")
            ),
            "validated_by_incremental_contract": status != "failed",
        },
        "checksums": checksums,
        "schemas": schemas,
        "intervals": intervals,
        "issues": issues,
        "issue_counts": {"critical": hard_issue_count, "warning": warning_count},
        "verdict": verdict,
        "safeguards": {
            "prediction_mutation": False,
            "model_mutation": False,
            "training": False,
            "network_requests": False,
        },
    }


def _freshness(
    through: date,
    evaluated_at: datetime,
    watermarks: Mapping[str, Any],
    sources: Mapping[str, Any],
    objective_days: int,
    late_days: int,
    candidate_dates: Sequence[str],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    local_timezone = ZoneInfo("Europe/Lisbon")
    evaluated_local = evaluated_at.astimezone(local_timezone)
    local_date = evaluated_local.date()
    common = watermarks.get("common_validated_watermark")
    common_date = date.fromisoformat(common) if common else None
    age = (local_date - common_date).days if common_date else None
    unresolved_late: list[dict[str, str]] = []
    latest_due_day = local_date - timedelta(days=late_days)
    if evaluated_local.timetz().replace(tzinfo=None) < time(12):
        latest_due_day -= timedelta(days=1)
    latest_late_required = min(through, latest_due_day)
    candidates = {
        date.fromisoformat(value)
        for value in candidate_dates
        if date.fromisoformat(value) <= latest_late_required
    }
    if common_date is None:
        candidates.add(latest_late_required)
    else:
        day = common_date + timedelta(days=1)
        while day <= latest_late_required:
            candidates.add(day)
            day += timedelta(days=1)
    for day in sorted(candidates):
        key = day.isoformat()
        ren_complete = ((sources.get("ren") or {}).get(key) or {}).get("status") == "complete"
        era_complete = _era_complete_station_count(
            sources.get("era5_land") or {}, key
        ) == 17
        if not (ren_complete and era_complete):
            unresolved_late.append({"date": key, "status": "source_late"})
    if unresolved_late:
        issues.append(
            {
                "code": "source_late",
                "severity": "critical",
                "count": len(unresolved_late),
                "sample": unresolved_late[:10],
            }
        )
    latest_objective_day = local_date - timedelta(days=objective_days)
    if evaluated_local.timetz().replace(tzinfo=None) < time(12):
        latest_objective_day -= timedelta(days=1)
    objective_required_through = min(through, latest_objective_day)
    objective_missed = common_date is None or common_date < objective_required_through
    if objective_missed and not unresolved_late:
        issues.append(
            {
                "code": "source_objective_missed",
                "severity": "warning",
                "count": 1,
                "sample": [common],
            }
        )
    return {
        "common_validated_watermark": common,
        "watermark_age_days": age,
        "objective_days": objective_days,
        "late_days": late_days,
        "late_required_through_date": latest_late_required.isoformat(),
        "objective_missed": objective_missed,
        "objective_required_through_date": objective_required_through.isoformat(),
        "deadline_local_time": "12:00:00 Europe/Lisbon",
        "unresolved_late_dates": unresolved_late,
    }


def _coverage(
    selected_dates: Sequence[str],
    sources: Mapping[str, Any],
    partitions: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    ren = sources.get("ren") or {}
    era = sources.get("era5_land") or {}
    integrated = partitions.get("integrated") or {}
    features = partitions.get("features") or {}
    rows: list[dict[str, Any]] = []
    for day in selected_dates:
        station_count = _era_complete_station_count(era, day)
        row = {
            "date": day,
            "ren_status": (ren.get(day) or {}).get("status", "missing"),
            "era5_station_count": station_count,
            "era5_expected_station_count": 17,
            "integration_ready": bool((integrated.get(day) or {}).get("integration_ready")),
            "feature_ready": bool((features.get(day) or {}).get("feature_ready")),
        }
        rows.append(row)
    incomplete = [
        row
        for row in rows
        if row["ren_status"] != "complete" or row["era5_station_count"] != 17
    ]
    if incomplete:
        issues.append(
            {
                "code": "incomplete_source_coverage",
                "severity": "warning",
                "count": len(incomplete),
                "sample": incomplete[:10],
            }
        )
    expected = len(rows)
    return {
        "date_count": expected,
        "ren_complete_count": sum(row["ren_status"] == "complete" for row in rows),
        "era5_complete_count": sum(row["era5_station_count"] == 17 for row in rows),
        "integration_ready_count": sum(row["integration_ready"] for row in rows),
        "feature_ready_count": sum(row["feature_ready"] for row in rows),
        "feature_ready_ratio": (
            sum(row["feature_ready"] for row in rows) / expected if expected else None
        ),
        "dates": rows,
    }


def _intervals(
    selected_dates: Sequence[str],
    sources: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    era = sources.get("era5_land") or {}
    ren = sources.get("ren") or {}
    invalid: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for day in selected_dates:
        expected_ren = expected_ren_interval_count(day)
        ren_ref = ren.get(day) or {}
        ren_actual: int | None = None
        ren_path = Path(str(ren_ref.get("primary_path") or ""))
        if ren_ref.get("status") == "complete":
            try:
                ren_actual = len(pd.read_csv(ren_path, usecols=["timestamp"]))
            except (OSError, ValueError, pd.errors.ParserError):
                ren_actual = None
        ren_check = {
            "source": "ren",
            "date": day,
            "expected": expected_ren,
            "actual": ren_actual,
        }
        checks.append(ren_check)
        if ren_ref.get("status") == "complete" and ren_actual != expected_ren:
            invalid.append(ren_check)
        expected_era = expected_era5_hourly_count(day)
        for station_id, station in _era_station_day_counts(era, day).items():
            actual = int(station["actual"])
            check = {
                "source": "era5_land",
                "date": day,
                "station_id": station_id,
                "expected": expected_era,
                "actual": actual,
            }
            checks.append(check)
            if station["declared_complete"] and actual != expected_era:
                invalid.append(check)
    if invalid:
        issues.append(
            {
                "code": "invalid_complete_interval_count",
                "severity": "critical",
                "count": len(invalid),
                "sample": invalid[:10],
            }
        )
    return {"checks": checks, "invalid_complete_count": len(invalid)}


def _era_station_day_counts(
    era: Mapping[str, Any], day: str
) -> dict[str, dict[str, Any]]:
    by_station: dict[str, dict[str, Any]] = {}
    for key, ref in era.items():
        actual = int((ref.get("local_hour_counts") or {}).get(day, 0))
        if actual == 0 and day not in (ref.get("local_dates") or []):
            continue
        station_id = str(ref.get("station_id") or key)
        station = by_station.setdefault(
            station_id, {"actual": 0, "declared_complete": False}
        )
        station["actual"] += actual
        station["declared_complete"] = bool(
            station["declared_complete"] or ref.get("status") == "complete"
        )
    return by_station


def _era_complete_station_count(era: Mapping[str, Any], day: str) -> int:
    expected = expected_era5_hourly_count(day)
    return sum(
        int(station["actual"]) == expected
        for station in _era_station_day_counts(era, day).values()
    )


def _schemas(
    selected_dates: Sequence[str],
    sources: Mapping[str, Any],
    partitions: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates: list[tuple[str, Path, tuple[str, ...]]] = []
    for day in selected_dates:
        ref = (sources.get("ren") or {}).get(day) or {}
        path = Path(str(ref.get("primary_path") or ""))
        if ref.get("status") == "complete" and path.is_file():
            candidates.append(("ren", path, REN_REQUIRED_COLUMNS))
    for ref in (sources.get("era5_land") or {}).values():
        if set(selected_dates).intersection(ref.get("local_dates") or []):
            path = Path(str(ref.get("primary_path") or ""))
            if path.is_file():
                candidates.append(("era5_land", path, ERA5_REQUIRED_COLUMNS))
    fingerprints: list[dict[str, Any]] = []
    schema_changes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source, path, required in candidates:
        key = (source, str(path))
        if key in seen:
            continue
        seen.add(key)
        sample_frame = pd.read_csv(path, nrows=100)
        columns = tuple(str(value) for value in sample_frame.columns)
        missing = sorted(set(required).difference(columns))
        extra = sorted(set(columns).difference(required))
        order_changed = not missing and not extra and columns != required
        dtype_categories = {
            column: _dtype_category(sample_frame[column]) for column in columns
        }
        type_mismatches = sorted(
            column
            for column in NUMERIC_SCHEMA_COLUMNS.intersection(columns)
            if not sample_frame.empty and dtype_categories[column] != "numeric"
        )
        record = {
            "source": source,
            "path": str(path.resolve()),
            "columns": list(columns),
            "fingerprint_sha256": sha256(
                json.dumps(
                    {"columns": columns, "dtypes": dtype_categories},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "dtype_categories": dtype_categories,
            "missing_required": missing,
            "additional_columns": extra,
            "order_changed": order_changed,
            "type_mismatches": type_mismatches,
        }
        fingerprints.append(record)
        if missing:
            schema_changes.append(record)
            issues.append(
                {
                    "code": "missing_required_schema_columns",
                    "severity": "critical",
                    "count": len(missing),
                    "sample": missing,
                }
            )
        elif extra:
            schema_changes.append(record)
            issues.append(
                {
                    "code": "additional_schema_columns",
                    "severity": "warning",
                    "count": len(extra),
                    "sample": extra[:10],
                }
            )
        elif order_changed:
            schema_changes.append(record)
            issues.append(
                {
                    "code": "schema_column_order_changed",
                    "severity": "critical",
                    "count": 1,
                    "sample": [source],
                }
            )
        elif type_mismatches:
            schema_changes.append(record)
            issues.append(
                {
                    "code": "schema_column_type_changed",
                    "severity": "critical",
                    "count": len(type_mismatches),
                    "sample": type_mismatches,
                }
            )
    feature_refs = partitions.get("features") or {}
    feature_fingerprints = sorted(
        {
            str((ref.get("files") or {}).get("feature_ready", {}).get("sha256"))
            for day, ref in feature_refs.items()
            if day in selected_dates and ref.get("feature_ready")
        }
        - {"None", ""}
    )
    return {
        "source_fingerprints": fingerprints,
        "feature_file_sha256": feature_fingerprints,
        "incompatible_schema_count": len(schema_changes),
    }


def _classify_failure(
    error: str, failed_input: Mapping[str, Any] | None
) -> dict[str, Any]:
    lowered = error.casefold()
    code = "batch_execution_failed"
    for token, candidate in (
        ("schema", "schema_validation_failed"),
        ("missing columns", "schema_validation_failed"),
        ("duplicate", "duplicate_validation_failed"),
        ("null", "null_validation_failed"),
        ("non-finite", "finiteness_validation_failed"),
        ("interval", "interval_validation_failed"),
        ("timestamp", "timestamp_validation_failed"),
    ):
        if token in lowered:
            code = candidate
            break
    count_field = {
        "duplicate_validation_failed": "duplicate_timestamp_count",
        "null_validation_failed": "null_count",
        "finiteness_validation_failed": "non_finite_count",
    }.get(code)
    observed_count = int((failed_input or {}).get(count_field) or 0) if count_field else 0
    return {
        "code": code,
        "severity": "critical",
        "count": max(observed_count, 1),
        "sample": [error[:500]],
    }


def _failed_input_evidence(error: str) -> dict[str, Any] | None:
    match = re.search(r"partition\s+(.+?\.csv)(?:\s|$)", error, flags=re.IGNORECASE)
    if match is None:
        return None
    path = Path(match.group(1).strip())
    if not path.is_file():
        return None
    evidence: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "null_count": None,
        "non_finite_count": None,
        "duplicate_timestamp_count": None,
        "duplicate_date_count": None,
    }
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return evidence
    evidence["null_count"] = int(frame.isna().sum().sum())
    numeric = frame.select_dtypes(include=["number"])
    evidence["non_finite_count"] = int(
        np.isinf(numeric.to_numpy(dtype=float)).sum()
    ) if not numeric.empty else 0
    timestamp_column = next(
        (name for name in ("timestamp", "timestamp_utc") if name in frame), None
    )
    evidence["duplicate_timestamp_count"] = (
        int(frame[timestamp_column].duplicated().sum()) if timestamp_column else 0
    )
    date_column = next(
        (name for name in ("Date", "date", "source_date") if name in frame), None
    )
    evidence["duplicate_date_count"] = (
        int(frame[date_column].duplicated().sum()) if date_column == "Date" else 0
    )
    return evidence


def _checksum_evidence(
    selected_dates: Sequence[str],
    sources: Mapping[str, Any],
    failed_input: Mapping[str, Any] | None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    refs: list[tuple[str, str, Mapping[str, Any]]] = []
    for day in selected_dates:
        ref = (sources.get("ren") or {}).get(day) or {}
        if ref:
            refs.append(("ren", day, ref))
    for key, ref in (sources.get("era5_land") or {}).items():
        if set(selected_dates).intersection(ref.get("local_dates") or []):
            refs.append(("era5_land", str(key), ref))
    for source, key, ref in refs:
        path = Path(str(ref.get("primary_path") or ""))
        if not path.is_file() or str(path.resolve()) in seen:
            continue
        seen.add(str(path.resolve()))
        observed = _sha256_file(path)
        recorded = ref.get("physical_sha256")
        entries.append(
            {
                "source": source,
                "logical_key": key,
                "path": str(path.resolve()),
                "observed_sha256": observed,
                "recorded_sha256": recorded,
                "matches_recorded": recorded is None or recorded == observed,
            }
        )
    if failed_input and failed_input.get("path") not in seen:
        entries.append(
            {
                "source": "rejected_input",
                "logical_key": None,
                "path": failed_input["path"],
                "observed_sha256": failed_input["sha256"],
                "recorded_sha256": None,
                "matches_recorded": None,
            }
        )
    return {"count": len(entries), "files": entries}


def _apply_quality_tolerance(
    issues: Sequence[dict[str, Any]], tolerance: int
) -> None:
    for issue in issues:
        if (
            issue.get("code") in HARD_QUALITY_CODES
            and int(issue.get("count") or 0) <= tolerance
        ):
            issue["severity"] = "informational"
            issue["tolerance_applied"] = tolerance


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dtype_category(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    return "string"


__all__ = ["QUALITY_SCHEMA", "build_batch_quality_evidence"]
