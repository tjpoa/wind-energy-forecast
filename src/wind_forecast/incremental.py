"""Transactional incremental updates for the accepted v2 data contract.

This module deliberately separates planning from execution.  Planning only
reads local metadata and never creates files or contacts a provider.  Execution
can receive a source refresher (the command line supplies the REN/CDS adapter),
but all downstream work is performed against immutable, checksum-addressed
source observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time as monotonic_time
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wind_forecast.batch_quality import build_batch_quality_evidence
from wind_forecast.integration import (
    COVERAGE_COLUMNS,
    DATE_LOCAL_COLUMN,
    TRANSFORMATION_VERSION as INTEGRATION_TRANSFORMATION_VERSION,
    aggregate_era5_daily_local,
    aggregate_ren_daily_local,
    build_coverage_table,
    expected_era5_hourly_count,
    iter_local_dates,
    join_integrated_daily,
    parse_local_date,
    sha256_file,
    validate_integrated_outputs,
)
from wind_forecast.monitoring_statistics import MonitoringPolicy
from wind_forecast.paths import (
    v2_processed_daily_merged_dir,
    v2_processed_ml_features_dir,
    v2_raw_production_dir,
    v2_raw_weather_dir,
)
from wind_forecast.schemas import DATE_COLUMN
from wind_forecast.v2_features import (
    FEATURE_COVERAGE_COLUMNS,
    TRANSFORMATION_VERSION as FEATURE_TRANSFORMATION_VERSION,
    build_feature_coverage,
    generate_v2_features,
    load_v1_feature_columns,
    map_integrated_base_columns,
    reindex_full_local_calendar,
    select_feature_ready_rows,
)


RUN_SCHEMA_VERSION = "wind_forecast.v2_incremental_run.v2"
STATE_SCHEMA_VERSION = "wind_forecast.v2_incremental_state.v1"
PARTITION_CONTRACT_VERSION = "wind_forecast.v2_incremental_partition.v1"
DEFAULT_BOOTSTRAP_START = date(2010, 1, 1)
DEFAULT_BOOTSTRAP_END = date(2026, 6, 27)
FEATURE_CONTEXT_DAYS = 14
EXPECTED_INTEGRATED_COLUMNS = 27
EXPECTED_FEATURE_COLUMNS = 58


class IncrementalUpdateError(RuntimeError):
    """Raised when an incremental update cannot be completed safely."""


class ConcurrentUpdateError(IncrementalUpdateError):
    """Raised when another live update owns the store lock."""


@dataclass(frozen=True)
class UpdateConfig:
    """Configuration for one v2 incremental planning/execution request."""

    through_date: str | date
    ren_root: Path = field(default_factory=v2_raw_production_dir)
    era5_root: Path = field(
        default_factory=lambda: v2_raw_weather_dir()
        / "era5_land"
        / "grid_policy=nearest_valid_r1"
        / "request_mode=monthly_bbox"
    )
    station_mapping: Path = Path("data/pilot/ipma/ipma_station_mapping.csv")
    v1_feature_table: Path = Path("data/processed/agg_data_ml.csv")
    baseline_integrated_root: Path = field(
        default_factory=lambda: v2_processed_daily_merged_dir()
        / "integrated_ren_era5_land_v2"
    )
    baseline_feature_root: Path = field(
        default_factory=lambda: v2_processed_ml_features_dir()
        / "feature_ready_ren_era5_land_v2"
    )
    store_root: Path = field(
        default_factory=lambda: v2_processed_daily_merged_dir().parent
        / "incremental_update"
    )
    raw_store_root: Path = Path("data/raw/v2/incremental_update")
    monitoring_policy_path: Path = Path("config/monitoring_policy_v1.json")
    revision_lookback_days: int = 90
    recheck_min_age_hours: int = 24
    recheck_ren_dates: tuple[str, ...] = ()
    recheck_era5_months: tuple[str, ...] = ()
    bootstrap_start: str | date | None = None
    bootstrap_end: str | date | None = None
    dry_run: bool = False
    now_utc: datetime | None = None

    def __post_init__(self) -> None:
        through = parse_local_date(self.through_date, "through_date")
        if self.revision_lookback_days < 0:
            raise ValueError("revision_lookback_days must be zero or greater.")
        if self.recheck_min_age_hours < 0:
            raise ValueError("recheck_min_age_hours must be zero or greater.")
        ren_dates = tuple(
            parse_local_date(item, "recheck_ren_date").isoformat()
            for item in self.recheck_ren_dates
        )
        era_months = tuple(_parse_month(item) for item in self.recheck_era5_months)
        now = self.now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "through_date", through)
        object.__setattr__(self, "ren_root", Path(self.ren_root))
        object.__setattr__(self, "era5_root", Path(self.era5_root))
        object.__setattr__(self, "station_mapping", Path(self.station_mapping))
        object.__setattr__(self, "v1_feature_table", Path(self.v1_feature_table))
        object.__setattr__(self, "baseline_integrated_root", Path(self.baseline_integrated_root))
        object.__setattr__(self, "baseline_feature_root", Path(self.baseline_feature_root))
        object.__setattr__(self, "store_root", Path(self.store_root))
        object.__setattr__(self, "raw_store_root", Path(self.raw_store_root))
        object.__setattr__(
            self, "monitoring_policy_path", Path(self.monitoring_policy_path)
        )
        object.__setattr__(self, "recheck_ren_dates", ren_dates)
        object.__setattr__(self, "recheck_era5_months", era_months)
        object.__setattr__(self, "now_utc", now.astimezone(timezone.utc))
        bootstrap_start = self.bootstrap_start or DEFAULT_BOOTSTRAP_START
        bootstrap_end = self.bootstrap_end or DEFAULT_BOOTSTRAP_END
        object.__setattr__(
            self,
            "bootstrap_start",
            parse_local_date(bootstrap_start, "bootstrap_start"),
        )
        object.__setattr__(
            self,
            "bootstrap_end",
            parse_local_date(bootstrap_end, "bootstrap_end"),
        )
        if self.bootstrap_start > self.bootstrap_end:
            raise ValueError("bootstrap_start must be on or before bootstrap_end.")


@dataclass(frozen=True)
class UpdatePlan:
    """Read-only description of source work and potentially affected outputs."""

    status: str
    through_date: str
    eligible_through: Mapping[str, str]
    bootstrap_required: bool
    ren_missing_dates: tuple[str, ...]
    ren_unavailable_dates: tuple[str, ...]
    ren_recheck_dates: tuple[str, ...]
    era5_missing_months: tuple[str, ...]
    era5_recheck_months: tuple[str, ...]
    pending_availability_dates: Mapping[str, tuple[str, ...]]
    potentially_affected_dates: tuple[str, ...]
    potentially_affected_feature_dates: tuple[str, ...]
    network_requests_planned: Mapping[str, int]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class UpdateResult:
    """Final outcome of planning or executing one incremental update."""

    status: str
    run_id: str | None
    plan: UpdatePlan
    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    current_state_path: Path | None = None
    generation: int | None = None
    affected_dates: tuple[str, ...] = ()
    feature_dates: tuple[str, ...] = ()
    watermarks: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-ready result."""
        return {
            "status": self.status,
            "run_id": self.run_id,
            "generation": self.generation,
            "affected_dates": list(self.affected_dates),
            "feature_dates": list(self.feature_dates),
            "watermarks": _json_ready(self.watermarks),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "manifest_sha256": self.manifest_sha256,
            "current_state_path": (
                str(self.current_state_path) if self.current_state_path else None
            ),
            "plan": self.plan.summary(),
        }


@dataclass(frozen=True)
class RefreshResult:
    """Additional source roots produced by a source refresh adapter."""

    ren_roots: tuple[Path, ...] = ()
    era5_roots: tuple[Path, ...] = ()


SourceRefresher = Callable[[UpdatePlan, Path], RefreshResult]
FailureHook = Callable[[str], None]


def plan_v2_update(config: UpdateConfig) -> UpdatePlan:
    """Plan an update without network calls, locks, directories, or writes."""
    state = _load_current_state(config.store_root, verify=True)
    baseline = _inspect_baseline(config)
    start = baseline["start_date"]
    ren_through, era5_through = _eligible_source_dates(config)
    checked = _recent_checks(config.store_root)

    if state is None:
        ren_index = _scan_ren_sources((config.ren_root,), start, ren_through)
        era_index = _scan_era5_sources(
            (config.era5_root,), era5_through, as_of=config.now_utc.date()
        )
    else:
        ren_index = dict((state.get("sources") or {}).get("ren") or {})
        era_index = dict((state.get("sources") or {}).get("era5_land") or {})

    all_ren_dates = [item.isoformat() for item in iter_local_dates(start, ren_through)]
    ren_missing = tuple(item for item in all_ren_dates if item not in ren_index)
    ren_unavailable = tuple(
        item
        for item in all_ren_dates
        if str((ren_index.get(item) or {}).get("status")) == "unavailable"
    )
    revision_start = ren_through - timedelta(
        days=max(config.revision_lookback_days - 1, 0)
    )
    recent_ren = {
        item
        for item in all_ren_dates
        if config.revision_lookback_days > 0
        and parse_local_date(item) >= revision_start
        and str((ren_index.get(item) or {}).get("status")) in {"complete", "unavailable"}
    }
    wanted_ren_rechecks = recent_ren | set(config.recheck_ren_dates)
    ren_rechecks = tuple(
        sorted(
            item
            for item in wanted_ren_rechecks
            if item <= ren_through.isoformat()
            and not _checked_too_recently(
                checked.get(("ren", item)), config.recheck_min_age_hours, config.now_utc
            )
        )
    )

    expected_months = tuple(_months_between(start, era5_through))
    present_months = _era5_complete_months(era_index, era5_through)
    era_missing = tuple(item for item in expected_months if item not in present_months)
    era_revision_floor = (
        era5_through - timedelta(days=max(config.revision_lookback_days - 1, 0))
    ).strftime("%Y-%m")
    wanted_era_rechecks = {
        item
        for item in expected_months
        if config.revision_lookback_days > 0
        and item >= era_revision_floor
        and item in present_months
    } | set(config.recheck_era5_months)
    era_rechecks = tuple(
        sorted(
            item
            for item in wanted_era_rechecks
            if item <= era5_through.strftime("%Y-%m")
            and not _checked_too_recently(
                checked.get(("era5_land", item)),
                config.recheck_min_age_hours,
                config.now_utc,
            )
        )
    )

    potential = set(ren_missing) | set(ren_rechecks)
    for month in (*era_missing, *era_rechecks):
        potential.update(
            day.isoformat()
            for day in iter_local_dates(start, era5_through)
            if day.strftime("%Y-%m") == month
        )
    feature_potential = _expand_feature_dates(
        potential,
        upper=max(config.through_date, baseline["end_date"]),
    )
    pending_ren = _date_strings(ren_through + timedelta(days=1), config.through_date)
    pending_era = _date_strings(era5_through + timedelta(days=1), config.through_date)
    return UpdatePlan(
        status="planned",
        through_date=config.through_date.isoformat(),
        eligible_through={
            "ren": ren_through.isoformat(),
            "era5_land": era5_through.isoformat(),
        },
        bootstrap_required=state is None,
        ren_missing_dates=tuple(sorted(ren_missing)),
        ren_unavailable_dates=tuple(sorted(ren_unavailable)),
        ren_recheck_dates=ren_rechecks,
        era5_missing_months=era_missing,
        era5_recheck_months=era_rechecks,
        pending_availability_dates={
            "ren": tuple(pending_ren),
            "era5_land": tuple(pending_era),
        },
        potentially_affected_dates=tuple(sorted(potential)),
        potentially_affected_feature_dates=tuple(sorted(feature_potential)),
        network_requests_planned={
            "ren": len(set(ren_missing) | set(ren_rechecks)),
            "era5_land": len(set(era_missing) | set(era_rechecks)),
        },
    )


def _rejected_update_plan(config: UpdateConfig) -> UpdatePlan:
    """Return minimal deterministic context when execution planning is rejected."""
    ren_through, era5_through = _eligible_source_dates(config)
    return UpdatePlan(
        status="rejected",
        through_date=config.through_date.isoformat(),
        eligible_through={
            "ren": ren_through.isoformat(),
            "era5_land": era5_through.isoformat(),
        },
        bootstrap_required=False,
        ren_missing_dates=(),
        ren_unavailable_dates=(),
        ren_recheck_dates=(),
        era5_missing_months=(),
        era5_recheck_months=(),
        pending_availability_dates={"ren": (), "era5_land": ()},
        potentially_affected_dates=(),
        potentially_affected_feature_dates=(),
        network_requests_planned={"ren": 0, "era5_land": 0},
    )


def run_v2_update(
    config: UpdateConfig,
    *,
    source_refresher: SourceRefresher | None = None,
    failure_hook: FailureHook | None = None,
) -> UpdateResult:
    """Execute one atomic incremental update, or return its dry-run plan."""
    if config.dry_run:
        plan = plan_v2_update(config)
        return UpdateResult(status="planned", run_id=None, plan=plan)

    run_id = _new_run_id(config.now_utc)
    paths = _run_paths(config.store_root, run_id)
    lock = _acquire_lock(config.store_root, run_id, config.now_utc)
    started = _utc_text(datetime.now(timezone.utc))
    run_started_monotonic = monotonic_time.perf_counter()
    events: list[dict[str, Any]] = []
    previous_state: dict[str, Any] | None = None
    current_pointer_verified = False
    refresh_performed = False
    source_changes: list[dict[str, Any]] = []
    affected_dates: set[str] = set()
    feature_dates: set[str] = set()
    quality_state: dict[str, Any] | None = None
    quality_policy: MonitoringPolicy | None = None
    plan: UpdatePlan | None = None
    try:
        paths["run"].mkdir(parents=True, exist_ok=False)
        paths["staging"].mkdir(parents=True, exist_ok=False)
        _emit_event(events, paths["events"], run_id, "run", "start", "ok")
        quality_policy = MonitoringPolicy.load(config.monitoring_policy_path)
        plan = plan_v2_update(config)
        # Verify the pointer again while holding the lock so planning cannot race
        # another publisher.  Keeping this inside the try guarantees lock cleanup
        # even when the current state is corrupt.
        previous_state = _load_current_state(config.store_root, verify=True)
        current_pointer_verified = True
        refresh = RefreshResult()
        if source_refresher is not None and any(plan.network_requests_planned.values()):
            refresh_started = monotonic_time.perf_counter()
            refresh = source_refresher(plan, paths["staging"] / "source_refresh")
            refresh_performed = True
            _emit_event(
                events,
                paths["events"],
                run_id,
                "source_refresh",
                "retrieve",
                "ok",
                source="ren+era5_land",
                duration_ms=_elapsed_ms(refresh_started),
            )
        _call_hook(failure_hook, "after_download")

        baseline = _inspect_baseline(config)
        if previous_state is None:
            state = _bootstrap_state(config, baseline, run_id)
        else:
            state = json.loads(json.dumps(previous_state))
        quality_state = state
        before_watermarks = dict(state.get("watermarks") or {})

        validation_started = monotonic_time.perf_counter()
        candidate_sources, affected_dates, source_changes = _refresh_source_index(
            config=config,
            plan=plan,
            current_sources=dict(state.get("sources") or {}),
            refresh=refresh,
        )
        state["sources"] = candidate_sources
        quality_state = state
        _emit_event(
            events,
            paths["events"],
            run_id,
            "source_validation",
            "validate",
            "ok",
            source="ren+era5_land",
            rows=len(source_changes),
            duration_ms=_elapsed_ms(validation_started),
        )
        _call_hook(failure_hook, "after_validation")

        release_written = False
        if affected_dates:
            integration_started = monotonic_time.perf_counter()
            integrated_updates = _build_integrated_partitions(
                config=config,
                run_id=run_id,
                staging_release=paths["staging_release"],
                sources=candidate_sources,
                affected_dates=affected_dates,
            )
            state.setdefault("partitions", {}).setdefault("integrated", {}).update(
                integrated_updates
            )
            for day, ref in sorted(integrated_updates.items()):
                _emit_event(
                    events,
                    paths["events"],
                    run_id,
                    "integration",
                    "recalculate",
                    "ok",
                    source="ren+era5_land",
                    partition=day,
                    checksum=ref["partition_key"],
                    duration_ms=_elapsed_ms(integration_started)
                    / max(len(integrated_updates), 1),
                )
            _call_hook(failure_hook, "after_integration")
            feature_dates = _expand_feature_dates(
                affected_dates,
                upper=max(config.through_date, baseline["end_date"]),
            )
            feature_started = monotonic_time.perf_counter()
            feature_updates = _build_feature_partitions(
                config=config,
                run_id=run_id,
                staging_release=paths["staging_release"],
                state=state,
                output_dates=feature_dates,
            )
            state["partitions"].setdefault("features", {}).update(feature_updates)
            for day, ref in sorted(feature_updates.items()):
                _emit_event(
                    events,
                    paths["events"],
                    run_id,
                    "features",
                    "recalculate",
                    "ok",
                    source="integrated_v2",
                    partition=day,
                    checksum=ref["partition_key"],
                    duration_ms=_elapsed_ms(feature_started)
                    / max(len(feature_updates), 1),
                )
            _validate_partition_index(
                state, config, staging_release=paths["staging_release"]
            )
            release_written = True

        state_changed = previous_state is None or bool(source_changes) or release_written
        if not state_changed:
            status = "no_op"
            watermarks = before_watermarks
            generation = int(previous_state.get("generation", 0)) if previous_state else 0
            quality_evidence = _persist_batch_quality(
                paths["quality"],
                config=config,
                plan=plan,
                run_id=run_id,
                state=previous_state,
                status=status,
                policy=quality_policy,
            )
            manifest = _manifest_payload(
                config=config,
                plan=plan,
                run_id=run_id,
                started_at=started,
                status=status,
                state_before=previous_state,
                state_after=previous_state,
                source_changes=source_changes,
                affected_dates=affected_dates,
                feature_dates=feature_dates,
                events=events,
                source_refresh_performed=refresh_performed,
                current_pointer_verified=current_pointer_verified,
                quality_evidence=quality_evidence,
            )
            manifest_checksum = _write_json(paths["manifest"], manifest)
            _emit_event(
                events,
                paths["events"],
                run_id,
                "run",
                "finish",
                status,
                duration_ms=_elapsed_ms(run_started_monotonic),
            )
            _rewrite_manifest_events(paths["manifest"], manifest, events)
            manifest_checksum = sha256_file(paths["manifest"])
            return UpdateResult(
                status=status,
                run_id=run_id,
                plan=plan,
                manifest_path=paths["manifest"],
                manifest_sha256=manifest_checksum,
                current_state_path=_current_path(config.store_root),
                generation=generation,
                affected_dates=tuple(sorted(affected_dates)),
                feature_dates=tuple(sorted(feature_dates)),
                watermarks=watermarks,
            )

        state["schema_version"] = STATE_SCHEMA_VERSION
        state["generation"] = int((previous_state or {}).get("generation", 0)) + 1
        state["release_id"] = run_id
        state["updated_at_utc"] = _utc_text(datetime.now(timezone.utc))
        state["watermarks"] = _compute_watermarks(config, state, baseline)
        quality_state = state
        _call_hook(failure_hook, "before_publish")
        if release_written:
            paths["release"].parent.mkdir(parents=True, exist_ok=True)
            os.replace(paths["staging_release"], paths["release"])
            _rewrite_staging_paths(state, paths["staging_release"], paths["release"])

        _emit_event(
            events,
            paths["events"],
            run_id,
            "publication",
            "replace_current_pointer",
            "prepared",
        )
        _emit_event(
            events,
            paths["events"],
            run_id,
            "run",
            "finish",
            "succeeded",
            duration_ms=_elapsed_ms(run_started_monotonic),
        )
        quality_evidence = _persist_batch_quality(
            paths["quality"],
            config=config,
            plan=plan,
            run_id=run_id,
            state=state,
            status="succeeded",
            policy=quality_policy,
        )
        manifest = _manifest_payload(
            config=config,
            plan=plan,
            run_id=run_id,
            started_at=started,
            status="succeeded",
            state_before=previous_state,
            state_after=state,
            source_changes=source_changes,
            affected_dates=affected_dates,
            feature_dates=feature_dates,
            events=events,
            source_refresh_performed=refresh_performed,
            current_pointer_verified=current_pointer_verified,
            quality_evidence=quality_evidence,
        )
        manifest_checksum = _write_json(paths["manifest"], manifest)
        pointer = dict(state)
        pointer["manifest_path"] = str(paths["manifest"].resolve())
        pointer["manifest_sha256"] = manifest_checksum
        _atomic_write_json(_current_path(config.store_root), pointer)
        _call_hook(failure_hook, "after_publish")
        return UpdateResult(
            status="succeeded",
            run_id=run_id,
            plan=plan,
            manifest_path=paths["manifest"],
            manifest_sha256=manifest_checksum,
            current_state_path=_current_path(config.store_root),
            generation=int(state["generation"]),
            affected_dates=tuple(sorted(affected_dates)),
            feature_dates=tuple(sorted(feature_dates)),
            watermarks=dict(state["watermarks"]),
        )
    except Exception as exc:
        evidence_plan = plan or _rejected_update_plan(config)
        published = _published_run_is_current(config.store_root, run_id)
        safe_error = _sanitize_error(exc)
        _emit_event(
            events,
            paths["events"],
            run_id,
            "run",
            "finish",
            "failed",
            error=safe_error,
            duration_ms=_elapsed_ms(run_started_monotonic),
        )
        if paths["run"].is_dir() and not published:
            try:
                quality_evidence = _persist_batch_quality(
                    paths["quality"],
                    config=config,
                    plan=evidence_plan,
                    run_id=run_id,
                    state=quality_state or previous_state,
                    status="failed",
                    policy=quality_policy,
                    policy_error=safe_error if quality_policy is None else None,
                    error=safe_error,
                )
            except Exception as quality_exc:
                quality_error = _sanitize_error(quality_exc)
                quality_evidence = _persist_batch_quality(
                    paths["quality"],
                    config=config,
                    plan=evidence_plan,
                    run_id=run_id,
                    state=None,
                    status="failed",
                    policy=quality_policy,
                    policy_error=safe_error if quality_policy is None else None,
                    error=f"{safe_error}; quality scan failed: {quality_error}",
                )
            failed_manifest = _manifest_payload(
                config=config,
                plan=evidence_plan,
                run_id=run_id,
                started_at=started,
                status="failed",
                state_before=previous_state,
                state_after=previous_state,
                source_changes=source_changes,
                affected_dates=affected_dates,
                feature_dates=feature_dates,
                events=events,
                failures=[safe_error],
                source_refresh_performed=refresh_performed,
                current_pointer_verified=current_pointer_verified,
                quality_evidence=quality_evidence,
            )
            _write_json(paths["manifest"], failed_manifest)
        if not published:
            _quarantine_staging(config.store_root, paths["staging"], run_id)
            _quarantine_release(config.store_root, paths["release"], run_id)
        raise
    finally:
        _release_lock(lock, run_id)


def materialize_current_integrated(
    store_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize the checksum-verified current integrated data and coverage."""
    state = _load_current_state(Path(store_root), verify=True)
    if state is None:
        raise FileNotFoundError(f"No current incremental state exists under {store_root}.")
    refs = dict((state.get("partitions") or {}).get("integrated") or {})
    merged_rows: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for day, ref in sorted(refs.items()):
        merged_rows.append(_read_ref_table(ref, "daily_merged", day, cache))
        coverage_rows.append(_read_ref_table(ref, "coverage", day, cache))
    merged = _concat_frames(merged_rows)
    coverage = _concat_frames(coverage_rows, columns=COVERAGE_COLUMNS)
    if not merged.empty:
        merged = merged.sort_values(DATE_LOCAL_COLUMN).reset_index(drop=True)
    coverage = coverage.sort_values(DATE_LOCAL_COLUMN).reset_index(drop=True)
    _require_unique_dates(merged, DATE_LOCAL_COLUMN, "integrated current view")
    _require_unique_dates(coverage, DATE_LOCAL_COLUMN, "integrated coverage current view")
    return merged, coverage


def materialize_current_features(
    store_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize the checksum-verified current feature data and coverage."""
    state = _load_current_state(Path(store_root), verify=True)
    if state is None:
        raise FileNotFoundError(f"No current incremental state exists under {store_root}.")
    refs = dict((state.get("partitions") or {}).get("features") or {})
    feature_rows: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for day, ref in sorted(refs.items()):
        feature_rows.append(_read_ref_table(ref, "feature_ready", day, cache))
        coverage_rows.append(_read_ref_table(ref, "feature_coverage", day, cache))
    features = _concat_frames(feature_rows)
    coverage = _concat_frames(coverage_rows, columns=FEATURE_COVERAGE_COLUMNS)
    if not features.empty:
        features = features.sort_values(DATE_COLUMN).reset_index(drop=True)
    coverage = coverage.sort_values(DATE_LOCAL_COLUMN).reset_index(drop=True)
    _require_unique_dates(features, DATE_COLUMN, "feature current view")
    _require_unique_dates(coverage, DATE_LOCAL_COLUMN, "feature coverage current view")
    return features, coverage


def load_verified_current_state(store_root: str | Path) -> dict[str, Any]:
    """Load the published v2 state after verifying its complete checksum chain.

    This is the supported read-only bridge for downstream consumers.  Callers
    receive a detached JSON-compatible copy so they cannot mutate internal
    state accidentally.
    """
    state = _load_current_state(Path(store_root), verify=True)
    if state is None:
        raise FileNotFoundError(
            f"No current incremental state exists under {store_root}."
        )
    return json.loads(json.dumps(state))


def _inspect_baseline(config: UpdateConfig) -> dict[str, Any]:
    integrated = config.baseline_integrated_root
    features = config.baseline_feature_root
    integrated_files = {
        "daily_merged": integrated / "daily_merged.csv",
        "coverage": integrated / "coverage.csv",
        "validation": integrated / "validation.json",
        "manifest": integrated / "manifest.json",
    }
    feature_files = {
        "feature_ready_daily": features / "feature_ready_daily.csv",
        "feature_coverage": features / "feature_coverage.csv",
        "feature_schema": features / "feature_schema.json",
        "v1_structure_comparison": features / "v1_structure_comparison.json",
        "validation": features / "validation.json",
        "manifest": features / "manifest.json",
    }
    required = (
        *integrated_files.values(),
        *feature_files.values(),
        config.v1_feature_table,
        config.station_mapping,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required accepted v2 baseline inputs are missing: " + ", ".join(missing))
    integrated_validation = _read_json(integrated_files["validation"])
    integrated_manifest = _read_json(integrated_files["manifest"])
    feature_validation = _read_json(feature_files["validation"])
    feature_manifest = _read_json(feature_files["manifest"])
    _require_accepted_baseline_metadata(
        validation=integrated_validation,
        manifest=integrated_manifest,
        validation_schema=None,
        manifest_schema="wind_forecast.integrated_v2_manifest.v1",
        label="integrated",
    )
    _require_accepted_baseline_metadata(
        validation=feature_validation,
        manifest=feature_manifest,
        validation_schema="wind_forecast.feature_ready_validation.v1",
        manifest_schema="wind_forecast.feature_ready_manifest.v1",
        label="feature-ready",
    )
    _verify_manifest_output_checksums(integrated_manifest, integrated_files)
    _verify_manifest_output_checksums(feature_manifest, feature_files)

    coverage = pd.read_csv(integrated_files["coverage"])
    merged = pd.read_csv(integrated_files["daily_merged"])
    feature_coverage = pd.read_csv(feature_files["feature_coverage"])
    feature_ready = pd.read_csv(feature_files["feature_ready_daily"])
    for frame, column, label in (
        (coverage, DATE_LOCAL_COLUMN, "integrated coverage"),
        (merged, DATE_LOCAL_COLUMN, "integrated data"),
        (feature_coverage, DATE_LOCAL_COLUMN, "feature coverage"),
        (feature_ready, DATE_COLUMN, "feature data"),
    ):
        if column not in frame:
            raise IncrementalUpdateError(f"{label} is missing {column}.")
        _require_unique_dates(frame, column, label)
    if list(coverage.columns) != list(COVERAGE_COLUMNS):
        raise IncrementalUpdateError("Accepted integrated coverage schema does not match the v2 contract.")
    if len(merged.columns) != EXPECTED_INTEGRATED_COLUMNS:
        raise IncrementalUpdateError(
            f"Accepted integrated table must have {EXPECTED_INTEGRATED_COLUMNS} columns."
        )
    v1_columns = load_v1_feature_columns(config.v1_feature_table)
    if len(v1_columns) != EXPECTED_FEATURE_COLUMNS or list(feature_ready.columns) != v1_columns:
        raise IncrementalUpdateError(
            f"Accepted feature table must match the {EXPECTED_FEATURE_COLUMNS}-column v1 structure."
        )
    if list(feature_coverage.columns) != list(FEATURE_COVERAGE_COLUMNS):
        raise IncrementalUpdateError("Accepted feature coverage schema does not match the v2 contract.")
    start = config.bootstrap_start
    end = config.bootstrap_end
    expected = {item.isoformat() for item in iter_local_dates(start, end)}
    actual = set(coverage[DATE_LOCAL_COLUMN].astype(str))
    if expected != actual:
        raise IncrementalUpdateError("Accepted integrated coverage does not contain every bootstrap calendar date.")
    if integrated_validation.get("start_date") != start.isoformat() or integrated_validation.get(
        "end_date"
    ) != end.isoformat():
        raise IncrementalUpdateError(
            "Accepted integrated validation coverage does not match the bootstrap contract."
        )
    manifest_start = integrated_manifest.get("coverage_start")
    manifest_end = integrated_manifest.get("coverage_end")
    if manifest_start != start.isoformat() or manifest_end != end.isoformat():
        raise IncrementalUpdateError(
            "Accepted integrated manifest coverage does not match the bootstrap contract."
        )
    integrated_ready_dates = set(
        coverage.loc[
            coverage["integration_ready"].map(_bool_value), DATE_LOCAL_COLUMN
        ].astype(str)
    )
    if set(merged[DATE_LOCAL_COLUMN].astype(str)) != integrated_ready_dates:
        raise IncrementalUpdateError(
            "Accepted integrated rows do not exactly match integration-ready coverage dates."
        )
    feature_calendar = set(feature_coverage[DATE_LOCAL_COLUMN].astype(str))
    if feature_calendar != expected:
        raise IncrementalUpdateError(
            "Accepted feature coverage does not contain every bootstrap calendar date."
        )
    ready_dates = set(
        feature_coverage.loc[
            feature_coverage["feature_ready"].map(_bool_value), DATE_LOCAL_COLUMN
        ].astype(str)
    )
    if set(feature_ready[DATE_COLUMN].astype(str)) != ready_dates:
        raise IncrementalUpdateError(
            "Accepted feature rows do not exactly match feature-ready coverage dates."
        )
    if feature_ready.isna().any().any():
        raise IncrementalUpdateError("Accepted feature-ready table contains null values.")
    feature_numeric = feature_ready.drop(columns=[DATE_COLUMN]).apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(feature_numeric.to_numpy(dtype=float)).all():
        raise IncrementalUpdateError(
            "Accepted feature-ready table contains non-finite numeric values."
        )
    return {
        "start_date": start,
        "end_date": end,
        "integrated_coverage": coverage,
        "integrated": merged,
        "feature_coverage": feature_coverage,
        "features": feature_ready,
        "v1_columns": v1_columns,
    }


def _require_accepted_baseline_metadata(
    *,
    validation: Mapping[str, Any],
    manifest: Mapping[str, Any],
    validation_schema: str | None,
    manifest_schema: str,
    label: str,
) -> None:
    if validation_schema is not None and validation.get("schema_version") != validation_schema:
        raise IncrementalUpdateError(f"Accepted {label} validation schema is invalid.")
    if validation.get("passed") is not True or not str(
        validation.get("verdict") or ""
    ).startswith("PASS"):
        raise IncrementalUpdateError(f"Accepted {label} validation did not pass.")
    if manifest.get("schema_version") != manifest_schema:
        raise IncrementalUpdateError(f"Accepted {label} manifest schema is invalid.")
    if not str(manifest.get("status") or "").startswith("PASS"):
        raise IncrementalUpdateError(f"Accepted {label} manifest status did not pass.")
    if manifest.get("failures"):
        raise IncrementalUpdateError(f"Accepted {label} manifest records failures.")


def _verify_manifest_output_checksums(
    manifest: Mapping[str, Any],
    expected_files: Mapping[str, Path],
) -> None:
    output_files = dict(manifest.get("output_files") or {})
    checksums = dict(manifest.get("sha256_checksums") or {})
    for role, expected_path in expected_files.items():
        if role == "manifest":
            continue
        recorded = output_files.get(role)
        if recorded is None or Path(str(recorded)).resolve() != expected_path.resolve():
            raise IncrementalUpdateError(
                f"Accepted manifest output path for {role} does not match {expected_path}."
            )
        checksum = checksums.get(str(recorded))
        if checksum != sha256_file(expected_path):
            raise IncrementalUpdateError(
                f"Accepted manifest checksum for {expected_path} is missing or invalid."
            )


def _bootstrap_state(
    config: UpdateConfig,
    baseline: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    integrated_files = {
        "daily_merged": config.baseline_integrated_root / "daily_merged.csv",
        "coverage": config.baseline_integrated_root / "coverage.csv",
    }
    feature_files = {
        "feature_ready": config.baseline_feature_root / "feature_ready_daily.csv",
        "feature_coverage": config.baseline_feature_root / "feature_coverage.csv",
    }
    integrated_checksums = {key: sha256_file(path) for key, path in integrated_files.items()}
    feature_checksums = {key: sha256_file(path) for key, path in feature_files.items()}
    ready_dates = set(baseline["features"][DATE_COLUMN].astype(str))
    integrated_index: dict[str, Any] = {}
    for row in baseline["integrated_coverage"].to_dict(orient="records"):
        day = str(row[DATE_LOCAL_COLUMN])
        integrated_index[day] = {
            "partition_key": _partition_key(
                "integrated", day, integrated_checksums.values(), INTEGRATION_TRANSFORMATION_VERSION
            ),
            "storage": "baseline",
            "status": str(row["coverage_status"]),
            "integration_ready": bool(_bool_value(row["integration_ready"])),
            "ren_status": str(row["ren_status"]),
            "era5_status": str(row["era5_status"]),
            "files": {
                key: {"path": str(path.resolve()), "sha256": integrated_checksums[key]}
                for key, path in integrated_files.items()
            },
        }
    feature_by_day = baseline["feature_coverage"].set_index(DATE_LOCAL_COLUMN)
    feature_index: dict[str, Any] = {}
    for day, row in feature_by_day.iterrows():
        day_text = str(day)
        feature_index[day_text] = {
            "partition_key": _partition_key(
                "features", day_text, feature_checksums.values(), FEATURE_TRANSFORMATION_VERSION
            ),
            "storage": "baseline",
            "feature_ready": day_text in ready_dates,
            "status": str(row["feature_coverage_status"]),
            "files": {
                key: {"path": str(path.resolve()), "sha256": feature_checksums[key]}
                for key, path in feature_files.items()
            },
        }
    ren_through, era_through = _eligible_source_dates(config)
    sources = {
        "ren": _scan_ren_sources((config.ren_root,), baseline["start_date"], ren_through),
        "era5_land": _scan_era5_sources(
            (config.era5_root,), era_through, as_of=config.now_utc.date()
        ),
    }
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "generation": 0,
        "release_id": run_id,
        "bootstrap": {
            "start_date": baseline["start_date"].isoformat(),
            "end_date": baseline["end_date"].isoformat(),
            "immutable": True,
        },
        "sources": sources,
        "partitions": {"integrated": integrated_index, "features": feature_index},
    }
    state["watermarks"] = _compute_watermarks(config, state, baseline)
    return state


def _refresh_source_index(
    *,
    config: UpdateConfig,
    plan: UpdatePlan,
    current_sources: Mapping[str, Any],
    refresh: RefreshResult,
) -> tuple[dict[str, Any], set[str], list[dict[str, Any]]]:
    ren = json.loads(json.dumps(dict(current_sources.get("ren") or {})))
    era = json.loads(json.dumps(dict(current_sources.get("era5_land") or {})))
    ren_dates = sorted(set(plan.ren_missing_dates) | set(plan.ren_recheck_dates))
    era_months = sorted(set(plan.era5_missing_months) | set(plan.era5_recheck_months))
    ren_roots = tuple(refresh.ren_roots) + (config.ren_root,)
    era_roots = tuple(refresh.era5_roots) + (config.era5_root,)
    scanned_ren: dict[str, Any] = {}
    if ren_dates:
        scanned_ren = _scan_ren_sources(
            ren_roots,
            parse_local_date(ren_dates[0]),
            parse_local_date(ren_dates[-1]),
            only=set(ren_dates),
        )
    scanned_era = _scan_era5_sources(
        era_roots,
        parse_local_date(plan.eligible_through["era5_land"]),
        only_months=set(era_months),
        as_of=config.now_utc.date(),
    ) if era_months else {}

    affected: set[str] = set()
    changes: list[dict[str, Any]] = []
    for key in ren_dates:
        candidate = scanned_ren.get(key)
        if candidate is None:
            continue
        old = ren.get(key)
        candidate = _persist_observation(config.raw_store_root, "ren", key, candidate, old)
        if (
            old is not None
            and old.get("status") == "complete"
            and candidate.get("status") == "unavailable"
        ):
            preserved = json.loads(json.dumps(old))
            failures = list(preserved.get("failed_observations") or [])
            failure = {
                "status": "unavailable",
                "physical_sha256": candidate.get("physical_sha256"),
                "semantic_sha256": candidate.get("semantic_sha256"),
                "path": candidate.get("primary_path"),
                "supporting_observations": candidate.get(
                    "supporting_observations", []
                ),
            }
            if failure not in failures:
                failures.append(failure)
                preserved["failed_observations"] = failures
                ren[key] = preserved
                changes.append(
                    {
                        "source": "ren",
                        "key": key,
                        "semantic_change": False,
                        "physical_change": True,
                        "finality_change": False,
                        "refresh_failure": "unavailable_preserved_current_complete",
                    }
                )
            continue
        changed = old is None or _logical_fingerprint(old) != _logical_fingerprint(candidate)
        observed = old is not None and old.get("physical_sha256") != candidate.get("physical_sha256")
        metadata_changed = old is not None and old.get("provider_finality") != candidate.get("provider_finality")
        if changed:
            candidate["history"] = _source_history(old)
            candidate["revision"] = int((old or {}).get("revision", 0)) + 1
            candidate["supersedes_id"] = (old or {}).get("revision_id")
            candidate["revision_id"] = _partition_key(
                "ren-source", key, [_logical_fingerprint(candidate)], "ren.v1"
            )
            affected.add(key)
        elif observed:
            candidate["revision"] = int(old.get("revision", 1))
            candidate["revision_id"] = old.get("revision_id")
            candidate["supersedes_id"] = old.get("supersedes_id")
            candidate["semantic_equivalent_to"] = old.get("physical_sha256")
            candidate["history"] = list(old.get("history") or [])
        elif metadata_changed:
            candidate["revision"] = int(old.get("revision", 1))
            candidate["revision_id"] = old.get("revision_id")
            candidate["supersedes_id"] = old.get("supersedes_id")
            candidate["history"] = _source_history(old)
        else:
            continue
        ren[key] = candidate
        changes.append(
            {
                "source": "ren",
                "key": key,
                "semantic_change": changed,
                "physical_change": observed,
                "finality_change": metadata_changed,
            }
        )

    for key, candidate_input in scanned_era.items():
        old = era.get(key)
        if old is not None and (
            set(candidate_input.get("complete_utc_dates") or [])
            != set(old.get("complete_utc_dates") or [])
            or _logical_fingerprint(old) != _logical_fingerprint(candidate_input)
        ):
            candidate_input = _merge_era_observations(
                previous=old,
                candidate=candidate_input,
                raw_store=config.raw_store_root,
                eligible_through=parse_local_date(plan.eligible_through["era5_land"]),
                as_of=config.now_utc.date(),
            )
        candidate = _persist_observation(config.raw_store_root, "era5_land", key, candidate_input, old)
        changed = old is None or _logical_fingerprint(old) != _logical_fingerprint(candidate)
        observed = old is not None and old.get("physical_sha256") != candidate.get("physical_sha256")
        metadata_changed = old is not None and old.get("provider_finality") != candidate.get("provider_finality")
        if changed:
            candidate["history"] = _source_history(old)
            candidate["revision"] = int((old or {}).get("revision", 0)) + 1
            candidate["supersedes_id"] = (old or {}).get("revision_id")
            candidate["revision_id"] = _partition_key(
                "era5-source", key, [_logical_fingerprint(candidate)], "era5-land.v1"
            )
            affected.update(_changed_era_local_dates(old, candidate))
        elif observed:
            candidate["revision"] = int(old.get("revision", 1))
            candidate["revision_id"] = old.get("revision_id")
            candidate["supersedes_id"] = old.get("supersedes_id")
            candidate["semantic_equivalent_to"] = old.get("physical_sha256")
            candidate["history"] = list(old.get("history") or [])
        elif metadata_changed:
            candidate["revision"] = int(old.get("revision", 1))
            candidate["revision_id"] = old.get("revision_id")
            candidate["supersedes_id"] = old.get("supersedes_id")
            candidate["history"] = _source_history(old)
        else:
            continue
        era[key] = candidate
        changes.append(
            {
                "source": "era5_land",
                "key": key,
                "semantic_change": changed,
                "physical_change": observed,
                "finality_change": metadata_changed,
            }
        )
    through = config.through_date.isoformat()
    affected = {item for item in affected if item <= through}
    return {"ren": ren, "era5_land": era}, affected, changes


def _merge_era_observations(
    *,
    previous: Mapping[str, Any],
    candidate: Mapping[str, Any],
    raw_store: Path,
    eligible_through: date,
    as_of: date,
) -> dict[str, Any]:
    old_frame = pd.read_csv(Path(str(previous["primary_path"])))
    new_frame = pd.read_csv(Path(str(candidate["primary_path"])))
    old_frame["_source_priority"] = 0
    new_frame["_source_priority"] = 1
    merged = pd.concat([old_frame, new_frame], ignore_index=True, sort=False)
    timestamps = pd.to_datetime(merged["timestamp_utc"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise IncrementalUpdateError("Cannot merge ERA5 observations with invalid timestamps.")
    merged["_timestamp_sort"] = timestamps
    merged = (
        merged.sort_values(["_timestamp_sort", "_source_priority"])
        .drop_duplicates("_timestamp_sort", keep="last")
        .sort_values("_timestamp_sort")
        .drop(columns=["_timestamp_sort", "_source_priority"])
        .reset_index(drop=True)
    )
    payload = merged.to_csv(index=False, lineterminator="\n")
    checksum = sha256(payload.encode("utf-8")).hexdigest()
    key_hash = sha256(str(candidate["logical_key"]).encode("utf-8")).hexdigest()[:16]
    path = (
        raw_store
        / "era5_land"
        / "merged"
        / f"key={key_hash}"
        / f"sha256={checksum}"
        / "hourly.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(payload, encoding="utf-8", newline="\n")
    elif sha256_file(path) != checksum:
        raise IncrementalUpdateError(f"Merged ERA5 checksum collision at {path}.")
    station_id = str(candidate["station_id"])
    month = str(candidate["month"])
    observation = _inspect_era5_path(
        path=path,
        root=raw_store / "era5_land" / "merged-support",
        station_part=f"station_id={station_id}",
        period_part=f"period={month}-canonical",
        station_id=station_id,
        month=month,
        logical_key=str(candidate["logical_key"]),
        eligible_through=eligible_through,
        as_of=as_of,
    )
    if observation is None:
        raise IncrementalUpdateError("Merged ERA5 observation has no eligible local dates.")
    observation["supporting_paths"] = list(candidate.get("supporting_paths") or [])
    observation["period_label"] = f"{month}-canonical"
    return observation


def _build_integrated_partitions(
    *,
    config: UpdateConfig,
    run_id: str,
    staging_release: Path,
    sources: Mapping[str, Any],
    affected_dates: set[str],
) -> dict[str, Any]:
    view = staging_release.parent / "source_view"
    _materialize_source_view(view, sources, affected_dates)
    updates: dict[str, Any] = {}
    for day in sorted(affected_dates):
        ren_daily, ren_coverage = aggregate_ren_daily_local(view / "production", start_date=day, end_date=day)
        points, aggregate = aggregate_era5_daily_local(
            view / "weather",
            station_mapping=config.station_mapping,
            start_date=day,
            end_date=day,
        )
        dates = iter_local_dates(day, day)
        coverage = build_coverage_table(dates, ren_coverage, aggregate)
        merged = join_integrated_daily(ren_daily, aggregate, coverage)
        validation = validate_integrated_outputs(
            start_date=day,
            end_date=day,
            ren_daily=ren_daily,
            era5_daily_points=points,
            era5_daily_aggregate=aggregate,
            daily_merged=merged,
            coverage=coverage,
        )
        if not validation["passed"]:
            raise IncrementalUpdateError(
                f"Integrated partition {day} failed validation: {validation['failures']}"
            )
        source_hashes = [_logical_fingerprint((sources.get("ren") or {}).get(day) or {})]
        source_hashes.extend(
            _logical_fingerprint(item)
            for item in (sources.get("era5_land") or {}).values()
            if day in set(item.get("local_dates") or [])
        )
        key = _partition_key(
            "integrated", day, source_hashes, INTEGRATION_TRANSFORMATION_VERSION
        )
        root = staging_release / "integrated" / f"date={day}"
        files = {
            "daily_merged": _write_csv(root / "daily_merged.csv", merged),
            "coverage": _write_csv(root / "coverage.csv", coverage),
            "ren_daily": _write_csv(root / "ren_daily.csv", ren_daily),
            "era5_daily_points": _write_csv(root / "era5_daily_points.csv", points),
            "era5_daily_aggregate": _write_csv(root / "era5_daily_aggregate.csv", aggregate),
            "validation": _write_json(root / "validation.json", validation),
        }
        updates[day] = {
            "partition_key": key,
            "storage": "release",
            "release_id": run_id,
            "status": str(coverage.iloc[0]["coverage_status"]),
            "integration_ready": bool(coverage.iloc[0]["integration_ready"]),
            "ren_status": str(coverage.iloc[0]["ren_status"]),
            "era5_status": str(coverage.iloc[0]["era5_status"]),
            "files": {
                name: {"path": str((root / _integrated_filename(name)).resolve()), "sha256": checksum}
                for name, checksum in files.items()
            },
        }
    return updates


def _build_feature_partitions(
    *,
    config: UpdateConfig,
    run_id: str,
    staging_release: Path,
    state: Mapping[str, Any],
    output_dates: set[str],
) -> dict[str, Any]:
    merged, coverage = _materialize_integrated_from_state(
        state,
        staging_release=staging_release,
    )
    if coverage.empty:
        raise IncrementalUpdateError("Cannot calculate features without integrated coverage.")
    v1_columns = load_v1_feature_columns(config.v1_feature_table)
    if len(v1_columns) != EXPECTED_FEATURE_COLUMNS:
        raise IncrementalUpdateError("Feature schema changed from the accepted 58-column contract.")
    calendar_start = parse_local_date(str(coverage[DATE_LOCAL_COLUMN].min()))
    available_dates = set(coverage[DATE_LOCAL_COLUMN].astype(str))
    requested_dates = sorted(day for day in output_dates if day in available_dates)
    ready_by_date: dict[str, pd.DataFrame] = {}
    coverage_rows: list[pd.DataFrame] = []
    for interval_start, interval_end in _contiguous_date_ranges(requested_dates):
        context_start_date = max(
            calendar_start,
            parse_local_date(interval_start) - timedelta(days=FEATURE_CONTEXT_DAYS),
        )
        context_start = context_start_date.isoformat()
        coverage_slice = coverage.loc[
            coverage[DATE_LOCAL_COLUMN].astype(str).between(context_start, interval_end)
        ].copy()
        merged_slice = merged.loc[
            merged[DATE_LOCAL_COLUMN].astype(str).between(context_start, interval_end)
        ].copy()
        mapped = map_integrated_base_columns(merged_slice)
        calendar = reindex_full_local_calendar(coverage_slice, mapped)
        generated = generate_v2_features(calendar)
        generated_coverage = build_feature_coverage(calendar)
        ready = select_feature_ready_rows(generated, generated_coverage, v1_columns)
        output_mask = generated_coverage[DATE_LOCAL_COLUMN].astype(str).between(
            interval_start, interval_end
        )
        coverage_rows.append(generated_coverage.loc[output_mask].copy())
        for row in ready.loc[
            ready[DATE_COLUMN].astype(str).between(interval_start, interval_end)
        ].to_dict(orient="records"):
            ready_by_date[str(row[DATE_COLUMN])] = pd.DataFrame(
                [row], columns=v1_columns
            )
    feature_coverage = _concat_frames(
        coverage_rows, columns=FEATURE_COVERAGE_COLUMNS
    )
    coverage_by_date = feature_coverage.set_index(DATE_LOCAL_COLUMN)
    integrated_index = dict((state.get("partitions") or {}).get("integrated") or {})
    updates: dict[str, Any] = {}
    schema_checksum = sha256("\n".join(v1_columns).encode("utf-8")).hexdigest()
    for day in requested_dates:
        if day not in coverage_by_date.index:
            continue
        context_start = (parse_local_date(day) - timedelta(days=FEATURE_CONTEXT_DAYS)).isoformat()
        context_hashes = [
            ref["partition_key"]
            for context_day, ref in sorted(integrated_index.items())
            if context_start <= context_day <= day
        ]
        context_hashes.append(schema_checksum)
        key = _partition_key(
            "features", day, context_hashes, FEATURE_TRANSFORMATION_VERSION
        )
        root = staging_release / "features" / f"date={day}"
        ready = ready_by_date.get(day, pd.DataFrame(columns=v1_columns))
        feature_cov = pd.DataFrame(
            [coverage_by_date.loc[day].to_dict() | {DATE_LOCAL_COLUMN: day}],
            columns=list(FEATURE_COVERAGE_COLUMNS),
        )
        if not ready.empty:
            numeric = ready.drop(columns=[DATE_COLUMN]).apply(pd.to_numeric, errors="coerce")
            if ready.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise IncrementalUpdateError(f"Feature partition {day} contains null or non-finite values.")
        files = {
            "feature_ready": _write_csv(root / "feature_ready.csv", ready),
            "feature_coverage": _write_csv(root / "feature_coverage.csv", feature_cov),
        }
        updates[day] = {
            "partition_key": key,
            "storage": "release",
            "release_id": run_id,
            "feature_ready": not ready.empty,
            "status": str(feature_cov.iloc[0]["feature_coverage_status"]),
            "context_start": context_start,
            "files": {
                name: {"path": str((root / _feature_filename(name)).resolve()), "sha256": checksum}
                for name, checksum in files.items()
            },
        }
    return updates


def _scan_ren_sources(
    roots: Sequence[Path],
    start: date,
    end: date,
    *,
    only: set[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for day in iter_local_dates(start, end):
        key = day.isoformat()
        if only is not None and key not in only:
            continue
        for root in roots:
            ren_base = root if root.name == "ren" else root / "ren"
            normalized = ren_base / "normalized" / f"date={key}" / "production_15min.csv"
            status_path = ren_base / "metadata" / f"date={key}" / "status.json"
            raw_path = ren_base / "raw" / f"date={key}" / "response.json"
            status = _read_json_optional(status_path)
            validation = dict(status.get("validation") or {})
            if normalized.is_file():
                daily, coverage = aggregate_ren_daily_local(root, start_date=key, end_date=key)
                row = coverage.iloc[0]
                if str(row["ren_status"]) != "complete" or daily.empty:
                    raise IncrementalUpdateError(
                        f"REN partition {normalized} for {key} is present but invalid: "
                        f"{row['message']}"
                    )
                semantic = _semantic_csv_sha256(
                    normalized,
                    columns=("timestamp", "wind_production_mw", "unit", "source_date"),
                    sort_by=("timestamp",),
                )
                result[key] = {
                    "logical_key": key,
                    "status": "complete",
                    "provider_finality": "unknown",
                    "physical_sha256": sha256_file(normalized),
                    "semantic_sha256": semantic,
                    "primary_path": str(normalized.resolve()),
                    "supporting_paths": _existing_paths((raw_path, status_path)),
                    "local_dates": [key],
                }
                break
            if validation.get("validation_status") == "unavailable":
                result[key] = {
                    "logical_key": key,
                    "status": "unavailable",
                    "provider_finality": "unknown",
                    "physical_sha256": sha256_file(status_path),
                    "semantic_sha256": sha256(
                        f"ren-unavailable:{key}".encode("utf-8")
                    ).hexdigest(),
                    "primary_path": str(status_path.resolve()),
                    "supporting_paths": [],
                    "local_dates": [key],
                }
                break
    for key, ref in result.items():
        ref.setdefault("revision", 1)
        ref.setdefault(
            "revision_id",
            _partition_key("ren-source", key, [_logical_fingerprint(ref)], "ren.v1"),
        )
        ref.setdefault("supersedes_id", None)
        ref.setdefault("history", [])
        ref.setdefault(
            "observations",
            [{"physical_sha256": ref["physical_sha256"], "path": ref["primary_path"]}],
        )
    return result


def _scan_era5_sources(
    roots: Sequence[Path],
    eligible_through: date,
    *,
    only_months: set[str] | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for root in roots:
        hourly_root = root / "hourly"
        if not hourly_root.is_dir():
            continue
        root_candidates: dict[str, dict[str, Any]] = {}
        for path in sorted(hourly_root.glob("station_id=*/period=*/hourly.csv")):
            station_part = path.parent.parent.name
            period_part = path.parent.name
            period = period_part.removeprefix("period=")
            month = period[:7]
            if only_months is not None and month not in only_months:
                continue
            station_id = station_part.removeprefix("station_id=")
            logical_key = f"{station_part}/month={month}"
            observation = _inspect_era5_path(
                path=path,
                root=root,
                station_part=station_part,
                period_part=period_part,
                station_id=station_id,
                month=month,
                logical_key=logical_key,
                eligible_through=eligible_through,
                as_of=as_of,
            )
            if observation is None:
                continue
            current = root_candidates.get(logical_key)
            score = (
                len(observation["complete_utc_dates"]),
                int(observation["hour_count"]),
                str(observation["period_label"]),
            )
            current_score = (
                len(current["complete_utc_dates"]),
                int(current["hour_count"]),
                str(current["period_label"]),
            ) if current else None
            if current is None or score > current_score:
                root_candidates[logical_key] = observation
        for key, observation in root_candidates.items():
            result.setdefault(key, observation)
    for key, ref in result.items():
        ref.setdefault("revision", 1)
        ref.setdefault(
            "revision_id",
            _partition_key(
                "era5-source", key, [_logical_fingerprint(ref)], "era5-land.v1"
            ),
        )
        ref.setdefault("supersedes_id", None)
        ref.setdefault("history", [])
        ref.setdefault(
            "observations",
            [{"physical_sha256": ref["physical_sha256"], "path": ref["primary_path"]}],
        )
    return result


def _inspect_era5_path(
    *,
    path: Path,
    root: Path,
    station_part: str,
    period_part: str,
    station_id: str,
    month: str,
    logical_key: str,
    eligible_through: date,
    as_of: date | None,
) -> dict[str, Any] | None:
    frame = pd.read_csv(path)
    required = {
        "timestamp_utc",
        "station_id",
        "temperature_2m_k",
        "temperature_2m_c",
        "u10_m_s",
        "v10_m_s",
        "wind_speed_m_s",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise IncrementalUpdateError(f"ERA5 partition {path} is missing columns: {missing}.")
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    if (
        timestamps.isna().any()
        or timestamps.duplicated().any()
        or not timestamps.is_monotonic_increasing
    ):
        raise IncrementalUpdateError(
            f"ERA5 partition {path} has invalid, duplicate, or unsorted timestamps."
        )
    numeric = frame[list(required - {"timestamp_utc", "station_id"})].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise IncrementalUpdateError(
            f"ERA5 partition {path} contains null or non-finite required values."
        )
    if set(frame["station_id"].astype(str)) != {station_id}:
        raise IncrementalUpdateError(
            f"ERA5 station IDs do not match partition {logical_key}."
        )
    utc_date_text = timestamps.dt.strftime("%Y-%m-%d")
    utc_hour_counts = {
        str(key): int(value)
        for key, value in utc_date_text.value_counts().sort_index().items()
        if str(key) <= eligible_through.isoformat()
    }
    complete_utc_dates = sorted(
        key for key, count in utc_hour_counts.items() if count == 24
    )
    local_date_text = timestamps.dt.tz_convert("Europe/Lisbon").dt.strftime(
        "%Y-%m-%d"
    )
    local_hour_counts = {
        str(key): int(value)
        for key, value in local_date_text.value_counts().sort_index().items()
        if str(key) <= eligible_through.isoformat()
    }
    local_dates = sorted(local_hour_counts)
    if not local_dates:
        return None
    semantic_columns = tuple(
        column
        for column in _era5_semantic_columns()
        if column in frame
    )
    expected_utc_dates = _era_expected_utc_dates(month, eligible_through)
    month_end = _month_end(datetime.strptime(month, "%Y-%m").date())
    finality = (
        "consolidated_window"
        if (as_of or datetime.now(timezone.utc).date())
        > month_end + timedelta(days=90)
        else "preliminary_window"
    )
    status_path = root / "metadata" / station_part / period_part / "status.json"
    daily_path = root / "daily_points" / station_part / period_part / "daily_points.csv"
    shared_raw = root / "raw" / period_part / "era5_land.nc"
    station_raw = root / "raw" / station_part / period_part / "era5_land.nc"
    return {
        "logical_key": logical_key,
        "month": month,
        "station_id": station_id,
        "period_label": period_part.removeprefix("period="),
        "status": (
            "complete"
            if set(expected_utc_dates).issubset(complete_utc_dates)
            else "partial"
        ),
        "provider_finality": finality,
        "physical_sha256": sha256_file(path),
        "semantic_sha256": _semantic_csv_sha256(
            path, columns=semantic_columns, sort_by=("timestamp_utc",)
        ),
        "primary_path": str(path.resolve()),
        "supporting_paths": _existing_paths(
            (shared_raw, station_raw, daily_path, status_path)
        ),
        "local_dates": local_dates,
        "local_hour_counts": local_hour_counts,
        "utc_hour_counts": utc_hour_counts,
        "complete_utc_dates": complete_utc_dates,
        "hour_count": int(len(frame)),
    }


def _era5_semantic_columns() -> tuple[str, ...]:
    return (
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
        "is_calm_or_near_calm",
    )


def _era_expected_utc_dates(month: str, through: date) -> list[str]:
    return [
        item.isoformat()
        for item in _days_in_month(month)
        if item <= through
    ]


def _persist_observation(
    raw_store: Path,
    source: str,
    key: str,
    observation: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(observation)))
    primary = Path(str(result["primary_path"]))
    safe_key = sha256(key.encode("utf-8")).hexdigest()[:16]
    destination = (
        raw_store
        / source
        / f"key={safe_key}"
        / f"sha256={result['physical_sha256']}"
        / primary.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(primary, destination)
    elif sha256_file(destination) != result["physical_sha256"]:
        raise IncrementalUpdateError(f"Content-addressed raw collision at {destination}.")
    result["primary_path"] = str(destination.resolve())
    observations = list((previous or {}).get("observations") or [])
    evidence = {
        "physical_sha256": result["physical_sha256"],
        "path": str(destination.resolve()),
    }
    if evidence not in observations:
        observations.append(evidence)
    result["observations"] = observations
    supporting = list((previous or {}).get("supporting_observations") or [])
    for value in result.get("supporting_paths") or []:
        source_path = Path(str(value))
        checksum = sha256_file(source_path)
        support_destination = raw_store / "blobs" / f"sha256={checksum}" / source_path.name
        support_destination.parent.mkdir(parents=True, exist_ok=True)
        if not support_destination.exists():
            shutil.copyfile(source_path, support_destination)
        elif sha256_file(support_destination) != checksum:
            raise IncrementalUpdateError(
                f"Content-addressed supporting-file collision at {support_destination}."
            )
        support_ref = {
            "sha256": checksum,
            "path": str(support_destination.resolve()),
            "filename": source_path.name,
        }
        if support_ref not in supporting:
            supporting.append(support_ref)
    result["supporting_observations"] = supporting
    result.pop("supporting_paths", None)
    return result


def _materialize_source_view(
    root: Path,
    sources: Mapping[str, Any],
    affected_dates: set[str],
) -> None:
    for day in sorted(affected_dates):
        ref = (sources.get("ren") or {}).get(day)
        if ref is None:
            raise IncrementalUpdateError(f"No REN source observation exists for affected date {day}.")
        source_path = Path(ref["primary_path"])
        if ref.get("status") == "unavailable":
            target = root / "production" / "ren" / "metadata" / f"date={day}" / "status.json"
        else:
            target = root / "production" / "ren" / "normalized" / f"date={day}" / "production_15min.csv"
        _copy_verified(source_path, target, str(ref["physical_sha256"]))
    for ref in (sources.get("era5_land") or {}).values():
        if not affected_dates.intersection(ref.get("local_dates") or []):
            continue
        target = (
            root
            / "weather"
            / "hourly"
            / f"station_id={ref['station_id']}"
            / f"period={ref['period_label']}"
            / "hourly.csv"
        )
        _copy_verified(Path(ref["primary_path"]), target, str(ref["physical_sha256"]))


def _materialize_integrated_from_state(
    state: Mapping[str, Any],
    *,
    staging_release: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    refs = dict((state.get("partitions") or {}).get("integrated") or {})
    merged_rows: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for day, ref in sorted(refs.items()):
        merged_rows.append(
            _read_ref_table(ref, "daily_merged", day, cache, staging_release=staging_release)
        )
        coverage_rows.append(
            _read_ref_table(ref, "coverage", day, cache, staging_release=staging_release)
        )
    return _concat_frames(merged_rows), _concat_frames(coverage_rows, columns=COVERAGE_COLUMNS)


def _materialize_features_from_state(
    state: Mapping[str, Any],
    *,
    staging_release: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    refs = dict((state.get("partitions") or {}).get("features") or {})
    feature_rows: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for day, ref in sorted(refs.items()):
        feature_rows.append(
            _read_ref_table(
                ref,
                "feature_ready",
                day,
                cache,
                staging_release=staging_release,
            )
        )
        coverage_rows.append(
            _read_ref_table(
                ref,
                "feature_coverage",
                day,
                cache,
                staging_release=staging_release,
            )
        )
    return _concat_frames(feature_rows), _concat_frames(
        coverage_rows, columns=FEATURE_COVERAGE_COLUMNS
    )


def _read_ref_table(
    ref: Mapping[str, Any],
    role: str,
    day: str,
    cache: dict[tuple[str, str], pd.DataFrame],
    *,
    staging_release: Path | None = None,
) -> pd.DataFrame:
    file_ref = dict((ref.get("files") or {}).get(role) or {})
    if not file_ref:
        return pd.DataFrame()
    path = Path(str(file_ref["path"]))
    if staging_release is not None and not path.exists():
        release_marker = f"{os.sep}releases{os.sep}{ref.get('release_id')}{os.sep}"
        if release_marker in str(path):
            suffix = str(path).split(release_marker, 1)[1]
            path = staging_release / suffix
    checksum = str(file_ref["sha256"])
    cache_key = (str(path), checksum)
    if cache_key not in cache:
        if not path.is_file() or sha256_file(path) != checksum:
            raise IncrementalUpdateError(f"Current partition file is missing or corrupt: {path}.")
        cache[cache_key] = pd.read_csv(path)
    frame = cache[cache_key]
    column = DATE_COLUMN if role == "feature_ready" else DATE_LOCAL_COLUMN
    if column not in frame:
        if frame.empty:
            return frame.copy()
        raise IncrementalUpdateError(f"Partition file {path} is missing date key {column}.")
    return frame.loc[frame[column].astype(str).eq(day)].copy()


def _validate_partition_index(
    state: Mapping[str, Any],
    config: UpdateConfig,
    *,
    staging_release: Path,
) -> None:
    merged, coverage = _materialize_integrated_from_state(state, staging_release=staging_release)
    _require_unique_dates(merged, DATE_LOCAL_COLUMN, "candidate integrated view")
    _require_unique_dates(coverage, DATE_LOCAL_COLUMN, "candidate coverage view")
    if list(coverage.columns) != list(COVERAGE_COLUMNS):
        raise IncrementalUpdateError("Candidate integrated coverage schema changed.")
    if not merged.empty and len(merged.columns) != EXPECTED_INTEGRATED_COLUMNS:
        raise IncrementalUpdateError("Candidate integrated schema changed.")
    expected_rows = len((state.get("partitions") or {}).get("integrated") or {})
    if len(coverage) != expected_rows:
        raise IncrementalUpdateError("Candidate coverage does not contain one row per indexed date.")
    features, feature_coverage = _materialize_features_from_state(
        state, staging_release=staging_release
    )
    _require_unique_dates(features, DATE_COLUMN, "candidate feature view")
    _require_unique_dates(
        feature_coverage, DATE_LOCAL_COLUMN, "candidate feature coverage view"
    )
    v1_columns = load_v1_feature_columns(config.v1_feature_table)
    if list(features.columns) != v1_columns:
        raise IncrementalUpdateError("Candidate feature schema changed.")
    if list(feature_coverage.columns) != list(FEATURE_COVERAGE_COLUMNS):
        raise IncrementalUpdateError("Candidate feature coverage schema changed.")
    expected_feature_rows = len(
        (state.get("partitions") or {}).get("features") or {}
    )
    if len(feature_coverage) != expected_feature_rows:
        raise IncrementalUpdateError(
            "Candidate feature coverage does not contain one row per indexed date."
        )
    ready_dates = set(
        feature_coverage.loc[
            feature_coverage["feature_ready"].map(_bool_value), DATE_LOCAL_COLUMN
        ].astype(str)
    )
    if set(features[DATE_COLUMN].astype(str)) != ready_dates:
        raise IncrementalUpdateError(
            "Candidate feature rows do not match feature-ready coverage dates."
        )
    if features.isna().any().any():
        raise IncrementalUpdateError("Candidate feature view contains null values.")
    numeric = features.drop(columns=[DATE_COLUMN]).apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise IncrementalUpdateError(
            "Candidate feature view contains non-finite numeric values."
        )


def _compute_watermarks(
    config: UpdateConfig,
    state: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    ren = dict((state.get("sources") or {}).get("ren") or {})
    era = dict((state.get("sources") or {}).get("era5_land") or {})
    integrated = dict((state.get("partitions") or {}).get("integrated") or {})
    ren_observed = sorted(ren)
    ren_valid = sorted(key for key, ref in ren.items() if ref.get("status") == "complete")
    era_dates = sorted({day for ref in era.values() for day in ref.get("local_dates") or []})
    era_valid = sorted(
        key for key, ref in integrated.items() if ref.get("era5_status") == "complete"
    )
    published = sorted(
        key for key, ref in integrated.items() if bool(ref.get("integration_ready"))
    )
    complete_dates = sorted(set(ren_valid).intersection(era_valid))
    start = baseline["start_date"]
    ren_through, era_through = _eligible_source_dates(config)
    ren_gaps = _source_gaps(start, ren_through, ren, source="ren")
    era_gaps = _era_gaps(start, era_through, era)
    return {
        "ren": {
            "observed_through": ren_observed[-1] if ren_observed else None,
            "validated_watermark": ren_valid[-1] if ren_valid else None,
            "published_watermark": published[-1] if published else None,
            "gaps": ren_gaps,
        },
        "era5_land": {
            "observed_through": era_dates[-1] if era_dates else None,
            "validated_watermark": era_valid[-1] if era_valid else None,
            "published_watermark": published[-1] if published else None,
            "gaps": era_gaps,
        },
        "common_validated_watermark": complete_dates[-1] if complete_dates else None,
    }


def _source_gaps(
    start: date,
    end: date,
    index: Mapping[str, Any],
    *,
    source: str,
) -> list[dict[str, str]]:
    gaps = []
    for day in iter_local_dates(start, end):
        key = day.isoformat()
        ref = index.get(key)
        status = str((ref or {}).get("status") or "missing")
        if status != "complete":
            gaps.append({"date": key, "status": status, "source": source})
    return gaps


def _era_gaps(start: date, end: date, index: Mapping[str, Any]) -> list[dict[str, str]]:
    counts_by_station_date: dict[tuple[str, str], int] = {}
    for ref in index.values():
        station_id = str(ref.get("station_id"))
        for day, count in (ref.get("local_hour_counts") or {}).items():
            key = (station_id, str(day))
            counts_by_station_date[key] = counts_by_station_date.get(key, 0) + int(
                count
            )
    return [
        {"date": day.isoformat(), "status": "missing", "source": "era5_land"}
        for day in iter_local_dates(start, end)
        if len(
            {
                station_id
                for station_id in {
                    str(ref.get("station_id")) for ref in index.values()
                }
                if counts_by_station_date.get((station_id, day.isoformat()), 0)
                == expected_era5_hourly_count(day)
            }
        )
        != 17
    ]


def _load_current_state(store_root: Path, *, verify: bool) -> dict[str, Any] | None:
    path = _current_path(store_root)
    if not path.is_file():
        return None
    payload = _read_json(path)
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise IncrementalUpdateError(f"Unsupported current-state schema in {path}.")
    if verify:
        manifest = Path(str(payload.get("manifest_path") or ""))
        expected = str(payload.get("manifest_sha256") or "")
        if not manifest.is_file() or sha256_file(manifest) != expected:
            raise IncrementalUpdateError("Current state references a missing or corrupt run manifest.")
        _verify_current_state_files(payload)
    return payload


def _verify_current_state_files(state: Mapping[str, Any]) -> None:
    expected_by_path: dict[str, str] = {}
    for group in ((state.get("partitions") or {}).values()):
        for ref in group.values():
            for file_ref in (ref.get("files") or {}).values():
                path = str(file_ref.get("path") or "")
                checksum = str(file_ref.get("sha256") or "")
                if path and checksum:
                    expected_by_path[path] = checksum
    for source_group in ((state.get("sources") or {}).values()):
        for ref in source_group.values():
            path = str(ref.get("primary_path") or "")
            checksum = str(ref.get("physical_sha256") or "")
            if path and checksum:
                expected_by_path[path] = checksum
    for path_text, expected in expected_by_path.items():
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != expected:
            raise IncrementalUpdateError(
                f"Current state references a missing or corrupt file: {path}."
            )


def _eligible_source_dates(config: UpdateConfig) -> tuple[date, date]:
    today_local = config.now_utc.astimezone(ZoneInfo("Europe/Lisbon")).date()
    ren = min(config.through_date, today_local - timedelta(days=1))
    era = min(config.through_date, today_local - timedelta(days=6))
    return ren, era


def _era5_complete_months(index: Mapping[str, Any], through: date) -> set[str]:
    by_month: dict[str, dict[str, set[str]]] = {}
    for ref in index.values():
        month = str(ref.get("month") or "")
        if not month:
            continue
        station_id = str(ref.get("station_id"))
        by_month.setdefault(month, {})[station_id] = set(
            ref.get("complete_utc_dates") or []
        )
    complete: set[str] = set()
    for month, stations in by_month.items():
        expected_days = set(_era_expected_utc_dates(month, through))
        if len(stations) == 17 and all(
            expected_days.issubset(days) for days in stations.values()
        ):
            complete.add(month)
    return complete


def _recent_checks(store_root: Path) -> dict[tuple[str, str], datetime]:
    checks: dict[tuple[str, str], datetime] = {}
    runs = store_root / "runs"
    if not runs.is_dir():
        return checks
    for path in runs.glob("*/manifest.json"):
        try:
            payload = _read_json(path)
            if payload.get("status") not in {"succeeded", "no_op"}:
                continue
            if not payload.get("source_refresh_performed"):
                continue
            timestamp = datetime.fromisoformat(str(payload["finished_at_utc"]).replace("Z", "+00:00"))
        except (KeyError, ValueError, IncrementalUpdateError):
            continue
        plan = dict(payload.get("plan") or {})
        for key in set(plan.get("ren_missing_dates") or []) | set(plan.get("ren_recheck_dates") or []):
            checks[("ren", str(key))] = max(checks.get(("ren", str(key)), timestamp), timestamp)
        for key in set(plan.get("era5_missing_months") or []) | set(plan.get("era5_recheck_months") or []):
            checks[("era5_land", str(key))] = max(
                checks.get(("era5_land", str(key)), timestamp), timestamp
            )
    return checks


def _checked_too_recently(
    checked_at: datetime | None,
    min_age_hours: int,
    now: datetime,
) -> bool:
    return checked_at is not None and now - checked_at < timedelta(hours=min_age_hours)


def _acquire_lock(store_root: Path, run_id: str, now: datetime) -> Path:
    state_root = store_root / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    lock = state_root / "update.lock"
    payload = {
        "run_id": run_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "created_at_utc": _utc_text(now),
    }
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = _read_json(lock)
        if existing.get("host") != socket.gethostname() or _pid_is_alive(int(existing.get("pid", -1))):
            raise ConcurrentUpdateError(
                f"Incremental update lock is owned by live run {existing.get('run_id')}."
            )
        _record_abandoned_run(store_root, existing, now)
        lock.unlink()
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return lock


def _record_abandoned_run(store_root: Path, lock: Mapping[str, Any], now: datetime) -> None:
    old_id = str(lock.get("run_id") or "unknown")
    if _current_points_to_valid_run(store_root, old_id):
        return
    run_root = store_root / "runs" / old_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = run_root / "manifest.json"
    _write_json(
        run_root / "abandoned.json",
        {
            "schema_version": "wind_forecast.v2_incremental_abandonment.v1",
            "run_id": old_id,
            "status": "abandoned",
            "started_at_utc": lock.get("created_at_utc"),
            "recovered_at_utc": _utc_text(now),
            "original_manifest_path": str(manifest.resolve()) if manifest.exists() else None,
            "original_manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
            "recovered_by_next_run": True,
        },
    )
    staging = store_root / "staging" / old_id
    _quarantine_staging(store_root, staging, old_id)
    _quarantine_release(store_root, store_root / "releases" / old_id, old_id)


def _current_points_to_valid_run(store_root: Path, run_id: str) -> bool:
    current = _current_path(store_root)
    if not current.is_file():
        return False
    try:
        payload = _read_json(current)
        if str(payload.get("release_id")) != run_id:
            return False
        manifest = Path(str(payload.get("manifest_path") or ""))
        expected = str(payload.get("manifest_sha256") or "")
        return manifest.is_file() and bool(expected) and sha256_file(manifest) == expected
    except IncrementalUpdateError:
        return False


def _release_lock(lock: Path, run_id: str) -> None:
    if not lock.exists():
        return
    try:
        current = _read_json(lock)
    except IncrementalUpdateError:
        return
    if current.get("run_id") == run_id:
        lock.unlink()


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _manifest_payload(
    *,
    config: UpdateConfig,
    plan: UpdatePlan,
    run_id: str,
    started_at: str,
    status: str,
    state_before: Mapping[str, Any] | None,
    state_after: Mapping[str, Any] | None,
    source_changes: Sequence[Mapping[str, Any]],
    affected_dates: Iterable[str],
    feature_dates: Iterable[str],
    events: Sequence[Mapping[str, Any]],
    failures: Sequence[str] = (),
    source_refresh_performed: bool = False,
    current_pointer_verified: bool = True,
    quality_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_text(datetime.now(timezone.utc)),
        "command": {
            "through_date": config.through_date.isoformat(),
            "revision_lookback_days": config.revision_lookback_days,
            "recheck_min_age_hours": config.recheck_min_age_hours,
            "recheck_ren_dates": list(config.recheck_ren_dates),
            "recheck_era5_months": list(config.recheck_era5_months),
            "monitoring_policy_path": str(config.monitoring_policy_path),
            "monitoring_policy_sha256": (
                sha256_file(config.monitoring_policy_path)
                if config.monitoring_policy_path.is_file()
                else None
            ),
        },
        "git_commit": _git_commit(),
        "versions": {
            "state": STATE_SCHEMA_VERSION,
            "partition": PARTITION_CONTRACT_VERSION,
            "integration": INTEGRATION_TRANSFORMATION_VERSION,
            "features": FEATURE_TRANSFORMATION_VERSION,
        },
        "plan": plan.summary(),
        "inputs": {
            "ren_root": str(config.ren_root),
            "era5_root": str(config.era5_root),
            "station_mapping": _file_evidence(config.station_mapping),
            "v1_feature_table": _file_evidence(config.v1_feature_table),
            "baseline_integrated_root": str(config.baseline_integrated_root),
            "baseline_feature_root": str(config.baseline_feature_root),
        },
        "outputs": _release_output_evidence(state_after, run_id),
        "watermarks_before": dict((state_before or {}).get("watermarks") or {}),
        "watermarks_after": dict((state_after or {}).get("watermarks") or {}),
        "gaps": {
            "ren_missing": list(plan.ren_missing_dates),
            "ren_unavailable": list(plan.ren_unavailable_dates),
            "era5_missing_months": list(plan.era5_missing_months),
            "pending_availability": _json_ready(plan.pending_availability_dates),
        },
        "revisions": _json_ready(source_changes),
        "source_changes": _json_ready(source_changes),
        "source_refresh_performed": source_refresh_performed,
        "intervals": {
            "integrated_recalculated": sorted(affected_dates),
            "features_recalculated": sorted(feature_dates),
            "feature_context_days": FEATURE_CONTEXT_DAYS,
        },
        "validations": {
            "current_pointer_verified": current_pointer_verified,
            "schema": True if status != "failed" else None,
            "checksums": True if status != "failed" else None,
            "duplicates": True if status != "failed" else None,
            "nulls_and_finiteness": True if status != "failed" else None,
        },
        "quality_evidence": _json_ready(quality_evidence),
        "warnings": [],
        "failures": list(failures),
        "events": _json_ready(events),
        "safeguards": {
            "raw_immutable": True,
            "atomic_current_pointer": True,
            "network_during_dry_run": False,
            "training": False,
            "notebooks": False,
        },
    }


def _file_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _release_output_evidence(
    state: Mapping[str, Any] | None,
    run_id: str,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"integrated": {}, "features": {}}
    for group_name, group in ((state or {}).get("partitions") or {}).items():
        if group_name not in evidence:
            continue
        for day, ref in group.items():
            if ref.get("release_id") != run_id:
                continue
            evidence[group_name][day] = {
                "partition_key": ref.get("partition_key"),
                "files": dict(ref.get("files") or {}),
            }
    return evidence


def _emit_event(
    events: list[dict[str, Any]],
    path: Path,
    run_id: str,
    stage: str,
    action: str,
    result: str,
    *,
    source: str | None = None,
    partition: str | None = None,
    rows: int | None = None,
    checksum: str | None = None,
    error: str | None = None,
    duration_ms: float | None = None,
) -> None:
    event = {
        "timestamp": _utc_text(datetime.now(timezone.utc)),
        "run_id": run_id,
        "stage": stage,
        "source": source or "coordinator",
        "partition": partition,
        "action": action,
        "result": result,
        "duration_ms": round(max(duration_ms or 0.0, 0.0), 3),
        "rows": rows,
        "checksum": checksum,
        "error": error,
    }
    events.append(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps(event, ensure_ascii=True, sort_keys=True))


def _elapsed_ms(started: float) -> float:
    return (monotonic_time.perf_counter() - started) * 1000.0


def _rewrite_manifest_events(path: Path, manifest: dict[str, Any], events: Sequence[Mapping[str, Any]]) -> None:
    manifest["events"] = _json_ready(events)
    manifest["finished_at_utc"] = _utc_text(datetime.now(timezone.utc))
    _write_json(path, manifest)


def _run_paths(store_root: Path, run_id: str) -> dict[str, Path]:
    run = store_root / "runs" / run_id
    staging = store_root / "staging" / run_id
    return {
        "run": run,
        "events": run / "events.jsonl",
        "manifest": run / "manifest.json",
        "quality": run / "quality.json",
        "staging": staging,
        "staging_release": staging / "release",
        "release": store_root / "releases" / run_id,
    }


def _persist_batch_quality(
    path: Path,
    *,
    config: UpdateConfig,
    plan: UpdatePlan,
    run_id: str,
    state: Mapping[str, Any] | None,
    status: str,
    policy: MonitoringPolicy | None,
    policy_error: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    objective_days = policy.source_objective_days if policy else 5
    late_days = policy.source_late_days if policy else 7
    hard_tolerance = policy.hard_quality_tolerance if policy else 0
    policy_evidence = {
        "schema_version": policy.schema_version if policy else None,
        "path": str(config.monitoring_policy_path.resolve()),
        "sha256": (
            sha256_file(config.monitoring_policy_path)
            if config.monitoring_policy_path.is_file()
            else None
        ),
        "status": "valid" if policy else "invalid",
        "error": policy_error,
        "source_objective_days": objective_days,
        "source_late_days": late_days,
        "hard_quality_tolerance": hard_tolerance,
    }
    payload = build_batch_quality_evidence(
        run_id=run_id,
        through_date=config.through_date.isoformat(),
        evaluated_at_utc=config.now_utc,
        plan=plan.summary(),
        state=state,
        status=status,
        source_objective_days=objective_days,
        source_late_days=late_days,
        hard_quality_tolerance=hard_tolerance,
        policy_evidence=policy_evidence,
        error=error,
    )
    checksum = _write_json(path, payload)
    return {
        "schema_version": payload["schema_version"],
        "path": str(path.resolve()),
        "sha256": checksum,
        "verdict": payload["verdict"],
    }


def _quarantine_staging(store_root: Path, staging: Path, run_id: str) -> None:
    if not staging.exists():
        return
    quarantine = store_root / "quarantine" / run_id
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    if quarantine.exists():
        quarantine = quarantine.with_name(f"{run_id}-{uuid4().hex[:8]}")
    os.replace(staging, quarantine)


def _quarantine_release(store_root: Path, release: Path, run_id: str) -> None:
    if not release.exists():
        return
    quarantine = store_root / "quarantine" / run_id
    quarantine.mkdir(parents=True, exist_ok=True)
    target = quarantine / "unpublished_release"
    if target.exists():
        target = quarantine / f"unpublished_release-{uuid4().hex[:8]}"
    os.replace(release, target)


def _rewrite_staging_paths(state: dict[str, Any], staging: Path, release: Path) -> None:
    staging_text = str(staging.resolve())
    release_text = str(release.resolve())
    for group in (state.get("partitions") or {}).values():
        for ref in group.values():
            for file_ref in (ref.get("files") or {}).values():
                path = str(file_ref.get("path") or "")
                if path.startswith(staging_text):
                    file_ref["path"] = release_text + path[len(staging_text):]


def _current_path(store_root: Path) -> Path:
    return store_root / "state" / "current.json"


def _published_run_is_current(store_root: Path, run_id: str) -> bool:
    path = _current_path(store_root)
    if not path.is_file():
        return False
    try:
        return str(_read_json(path).get("release_id")) == run_id
    except IncrementalUpdateError:
        return False


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    _write_json(temporary, payload)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sha256_file(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IncrementalUpdateError(f"Invalid JSON file: {path}.") from exc
    if not isinstance(payload, dict):
        raise IncrementalUpdateError(f"JSON file must contain an object: {path}.")
    return payload


def _read_json_optional(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _semantic_csv_sha256(
    path: Path,
    *,
    columns: Sequence[str],
    sort_by: Sequence[str],
) -> str:
    frame = pd.read_csv(path, usecols=list(columns))
    normalized = frame.sort_values(list(sort_by)).reset_index(drop=True)
    payload = normalized.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return sha256(payload.encode("utf-8")).hexdigest()


def _partition_key(role: str, logical_key: str, inputs: Iterable[str], version: str) -> str:
    payload = {
        "contract": PARTITION_CONTRACT_VERSION,
        "role": role,
        "logical_key": logical_key,
        "inputs": sorted(str(item) for item in inputs),
        "transformation_version": version,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _logical_fingerprint(ref: Mapping[str, Any]) -> str:
    return str(ref.get("semantic_sha256") or f"status:{ref.get('status')}")


def _source_history(previous: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not previous:
        return []
    history = list(previous.get("history") or [])
    snapshot = {
        "revision": previous.get("revision"),
        "revision_id": previous.get("revision_id"),
        "semantic_sha256": previous.get("semantic_sha256"),
        "physical_sha256": previous.get("physical_sha256"),
        "provider_finality": previous.get("provider_finality"),
        "primary_path": previous.get("primary_path"),
    }
    if snapshot not in history:
        history.append(snapshot)
    return history


def _changed_era_local_dates(
    previous: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
) -> set[str]:
    if not previous:
        return set(candidate.get("local_dates") or [])
    columns = (
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
        "is_calm_or_near_calm",
    )
    try:
        old_frame = pd.read_csv(Path(str(previous["primary_path"])))
        new_frame = pd.read_csv(Path(str(candidate["primary_path"])))
        comparable = [
            item for item in columns if item in old_frame.columns and item in new_frame.columns
        ]
        if "timestamp_utc" not in comparable:
            raise ValueError("timestamp_utc is unavailable")
        old = old_frame[comparable].copy().set_index("timestamp_utc")
        new = new_frame[comparable].copy().set_index("timestamp_utc")
        identities = old.index.union(new.index)
        old = old.reindex(identities)
        new = new.reindex(identities)
        for column in old.columns:
            old[column] = old[column].map(_canonical_semantic_value)
            new[column] = new[column].map(_canonical_semantic_value)
        equal = old.eq(new) | (old.isna() & new.isna())
        changed_timestamps = identities[~equal.all(axis=1)]
        parsed = pd.to_datetime(changed_timestamps, utc=True, errors="coerce")
        if parsed.isna().any():
            raise ValueError("unparseable timestamp")
        return set(parsed.tz_convert("Europe/Lisbon").strftime("%Y-%m-%d"))
    except (OSError, KeyError, ValueError, pd.errors.ParserError):
        return set(previous.get("local_dates") or []) | set(
            candidate.get("local_dates") or []
        )


def _canonical_semantic_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format(float(value), ".12g")
    return str(value).strip()


def _copy_verified(source: Path, target: Path, checksum: str) -> None:
    if not source.is_file() or sha256_file(source) != checksum:
        raise IncrementalUpdateError(f"Source observation is missing or corrupt: {source}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _require_unique_dates(frame: pd.DataFrame, column: str, label: str) -> None:
    if column in frame and frame[column].astype(str).duplicated().any():
        raise IncrementalUpdateError(f"{label} contains duplicate {column} values.")


def _concat_frames(frames: Sequence[pd.DataFrame], columns: Sequence[str] | None = None) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=list(columns or ()))
    return pd.concat(non_empty, ignore_index=True)


def _integrated_filename(role: str) -> str:
    return {
        "daily_merged": "daily_merged.csv",
        "coverage": "coverage.csv",
        "ren_daily": "ren_daily.csv",
        "era5_daily_points": "era5_daily_points.csv",
        "era5_daily_aggregate": "era5_daily_aggregate.csv",
        "validation": "validation.json",
    }[role]


def _feature_filename(role: str) -> str:
    return {"feature_ready": "feature_ready.csv", "feature_coverage": "feature_coverage.csv"}[role]


def _expand_feature_dates(days: Iterable[str], *, upper: date) -> set[str]:
    result: set[str] = set()
    for item in days:
        start = parse_local_date(item)
        for offset in range(FEATURE_CONTEXT_DAYS + 1):
            candidate = start + timedelta(days=offset)
            if candidate <= upper:
                result.add(candidate.isoformat())
    return result


def _date_strings(start: date, end: date) -> list[str]:
    if start > end:
        return []
    return [item.isoformat() for item in iter_local_dates(start, end)]


def _contiguous_date_ranges(values: Sequence[str]) -> list[tuple[str, str]]:
    if not values:
        return []
    dates = sorted({parse_local_date(item) for item in values})
    ranges: list[tuple[str, str]] = []
    start = previous = dates[0]
    for current in dates[1:]:
        if current != previous + timedelta(days=1):
            ranges.append((start.isoformat(), previous.isoformat()))
            start = current
        previous = current
    ranges.append((start.isoformat(), previous.isoformat()))
    return ranges


def _months_between(start: date, end: date) -> list[str]:
    cursor = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)
    values = []
    while cursor <= final:
        values.append(cursor.strftime("%Y-%m"))
        cursor = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
    return values


def _days_in_month(month: str) -> list[date]:
    first = datetime.strptime(month, "%Y-%m").date().replace(day=1)
    end = _month_end(first)
    return iter_local_dates(first, end)


def _month_end(value: date) -> date:
    next_month = date(value.year + (value.month == 12), 1 if value.month == 12 else value.month + 1, 1)
    return next_month - timedelta(days=1)


def _parse_month(value: str) -> str:
    try:
        return datetime.strptime(str(value), "%Y-%m").strftime("%Y-%m")
    except ValueError as exc:
        raise ValueError("recheck_era5_month must be formatted as YYYY-MM.") from exc


def _existing_paths(paths: Iterable[Path]) -> list[str]:
    return [str(path.resolve()) for path in paths if path.is_file()]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _new_run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:12]}"


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _call_hook(hook: FailureHook | None, stage: str) -> None:
    if hook is not None:
        hook(stage)


def _sanitize_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"[:1000]
    patterns = (
        r"(?i)(api[_-]?key|token|password|secret)\s*[=:]\s*[^\s,;]+",
        r"(?i)(authorization)\s*:\s*bearer\s+[^\s]+",
    )
    for pattern in patterns:
        message = re.sub(pattern, r"\1=[REDACTED]", message)
    return message


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


__all__ = [
    "ConcurrentUpdateError",
    "IncrementalUpdateError",
    "RefreshResult",
    "UpdateConfig",
    "UpdatePlan",
    "UpdateResult",
    "materialize_current_features",
    "materialize_current_integrated",
    "load_verified_current_state",
    "plan_v2_update",
    "run_v2_update",
]
