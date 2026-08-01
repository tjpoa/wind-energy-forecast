"""Verified, all-or-nothing projection of operational artifacts to PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping

from wind_forecast.monitoring import load_model_era, load_prediction_evidence
from wind_forecast.monitoring_reporting import (
    load_active_alerts,
    load_alert_history,
    load_monitoring_calibration,
    load_monitoring_report,
    load_monitoring_report_state,
    load_reporting_attempts,
    resolve_report_model_era,
)
from wind_forecast.monitoring_statistics import threshold_severity
from wind_forecast.operational_projection_migrations import (
    AppliedMigration,
    OperationalProjectionMigrationError,
    _compare,
    discover_migrations,
)
from wind_forecast.operational_projection_models import (
    EvidenceIdentity,
    EvidenceRecord,
    GenerationManifest,
    LineageRelation,
    ProjectionSnapshot,
    RelationalRow,
    canonical_sha256,
    ordered_evidence,
    ordered_relations,
    ordered_rows,
)
from wind_forecast.paths import project_root


PROJECTION_LOCK_KEY = 8_144_735_432_071_719_011
ATTEMPT_SCHEMA = "wind_forecast.monitoring_reporting_attempt_projection.v1"
ACTIVE_ALERT_STATE_SCHEMA = "wind_forecast.verified_active_alert_binding.v1"
COMMAND_SCHEMA_VERSION = "wind_forecast.operational_projection.command.v1"
ERROR_SCHEMA_VERSION = "wind_forecast.operational_projection.error.v1"

_HEX_40_OR_64 = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
_SQL_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_TABLE_ORDER = (
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
_TABLE_COLUMNS = {
    "model_era": (
        "model_era_id", "evidence_record_id", "association_kind",
        "deployment_id", "deployment_generation", "registered_model_name",
        "model_version", "fit_cutoff", "activation_cutoff", "bundle_sha256",
        "model_sha256", "dataset_sha256", "feature_schema_sha256",
        "calibration_sha256", "ledger_sha256", "calibration_id", "reference_id",
    ),
    "monitoring_report": (
        "report_id", "evidence_record_id", "reporting_run_id", "created_at_utc",
        "through_date", "source_run_id", "source_status", "calibration_id",
        "reference_id", "policy_sha256", "quality_status", "batch_status",
        "verdict", "watermark_date", "watermark_age_days", "objective_days",
        "late_days", "objective_missed", "unresolved_late_date_count",
        "date_count", "ren_complete_count", "era5_complete_count",
        "integration_ready_count", "feature_ready_count", "model_era_id",
    ),
    "quality_issue": (
        "report_id", "position", "evidence_record_id", "code", "severity",
    ),
    "monitoring_window": (
        "report_id", "window_days", "evidence_record_id", "status",
        "sample_count", "coverage_ratio", "coverage_severity", "minimum_samples",
        "calendar_start", "calendar_end",
    ),
    "performance_metric": (
        "report_id", "window_days", "evidence_record_id", "metric_name", "value",
        "value_status", "severity", "warning_threshold", "critical_threshold",
        "direction", "unit_or_scale",
    ),
    "drift_measurement": (
        "report_id", "window_days", "position", "evidence_record_id", "feature",
        "comparator", "detector", "value", "severity", "warning_threshold",
        "critical_threshold", "direction",
    ),
    "alert_event": (
        "alert_event_id", "evidence_record_id", "rule_id", "through_date",
        "event_type", "severity", "previous_alert_event_id",
    ),
    "active_alert_snapshot": (
        "generation_id", "rule_id", "evidence_record_id", "alert_event_id",
    ),
    "reporting_attempt": (
        "reporting_run_id", "evidence_record_id", "attempted_at_utc", "through_date",
        "source_run_id", "source_status", "status", "report_id",
        "active_alert_count", "failure_at_utc", "failure_type", "failure_message",
    ),
    "lineage_edge": (
        "generation_id", "edge_type", "source_evidence_record_id",
        "target_evidence_record_id", "position", "evidence_record_id",
    ),
}
_COUNT_COLUMNS = (
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


class OperationalProjectionError(RuntimeError):
    """Base class for failures whose public representation must be sanitized."""

    code = "projection_failed"


class ProjectionSourceError(OperationalProjectionError):
    code = "source_corrupt"


class ProjectionSourceConflictError(OperationalProjectionError):
    code = "source_conflict"


class ProjectionSourceNotStableError(OperationalProjectionError):
    code = "source_not_stable"


class ProjectionProvenanceError(OperationalProjectionError):
    code = "source_provenance_unavailable"


class ProjectionDatabaseError(OperationalProjectionError):
    code = "database_unavailable"


class ProjectionIncompatibleError(OperationalProjectionError):
    code = "incompatible_schema"


class ProjectionMismatchError(OperationalProjectionError):
    code = "projection_mismatch"


@dataclass(frozen=True)
class ProjectionCommandResult:
    """Sanitized result shared by plan, project, and verify."""

    command: str
    status: str
    environment_id: str
    generation_id: str
    head_generation_id: str | None
    counts: Mapping[str, int]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": COMMAND_SCHEMA_VERSION,
            "command": self.command,
            "status": self.status,
            "environment_id": self.environment_id,
            "generation_id": self.generation_id,
            "head_generation_id": self.head_generation_id,
            "counts": dict(sorted(self.counts.items())),
        }


@dataclass
class _SnapshotBuilder:
    environment_id: str
    source_git_commit: str
    observed_at_utc: datetime
    evidence: list[EvidenceRecord]
    rows: list[RelationalRow]
    relations: list[LineageRelation]

    def add_evidence(
        self,
        *,
        domain: str,
        source_kind: str,
        schema_version: str,
        record_id: str,
        digest: str,
        effective_at: str,
        mutable: bool = False,
    ) -> EvidenceIdentity:
        identity = EvidenceIdentity(
            domain=domain,
            source_kind=source_kind,
            schema_version=schema_version,
            record_id=record_id,
            sha256=digest,
        )
        self.evidence.append(
            EvidenceRecord(
                identity=identity,
                effective_at=effective_at,
                observed_at_utc=self.observed_at_utc if mutable else None,
            )
        )
        return identity

    def add_row(
        self,
        table: str,
        values: Mapping[str, Any],
        *,
        evidence_links: Mapping[str, EvidenceIdentity] | None = None,
    ) -> None:
        row = RelationalRow.create(
            table,
            values,
            evidence_links=evidence_links,
        )
        _validate_minimized_row(row)
        self.rows.append(row)

    def add_relation(
        self,
        edge_type: str,
        source: EvidenceIdentity,
        target: EvidenceIdentity,
        *,
        position: int,
        evidence: EvidenceIdentity,
    ) -> None:
        self.relations.append(
            LineageRelation(edge_type, source, target, position, evidence)
        )


def resolve_source_git_commit(root: Path | None = None) -> str:
    """Return a committed projector revision and reject tracked local changes."""
    checkout = root or project_root()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProjectionProvenanceError(
            "Projector source provenance is unavailable."
        ) from exc
    if not _HEX_40_OR_64.fullmatch(revision) or dirty:
        raise ProjectionProvenanceError(
            "Projector source provenance is unavailable."
        )
    return revision


def build_projection_snapshot(
    monitoring_store_root: str | Path,
    *,
    environment_id: str,
    source_git_commit: str,
    clock: Callable[[], datetime] | None = None,
) -> ProjectionSnapshot:
    """Read only verified loaders and normalize one stable source snapshot."""
    if environment_id != "local" or not _HEX_40_OR_64.fullmatch(source_git_commit):
        raise ProjectionProvenanceError("Projection provenance is invalid.")
    observed = (clock or (lambda: datetime.now(timezone.utc)))()
    if observed.tzinfo is None:
        raise ProjectionProvenanceError("Projection clock must be timezone-aware.")
    observed = observed.astimezone(timezone.utc)
    store_root = Path(monitoring_store_root)

    try:
        state_before = load_monitoring_report_state(store_root)
        active_before = load_active_alerts(store_root)
        history_before = load_alert_history(store_root)
        attempts_before = load_reporting_attempts(store_root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ProjectionSourceError("Operational source evidence is invalid.") from exc

    if any(item.get("status") == "in_progress" for item in attempts_before):
        raise ProjectionSourceNotStableError(
            "A reporting attempt is not yet terminal."
        )
    if state_before is None and active_before:
        raise ProjectionSourceConflictError("Operational source states disagree.")
    if state_before is not None and dict(state_before.get("active_alerts") or {}) != active_before:
        raise ProjectionSourceConflictError("Operational source states disagree.")

    builder = _SnapshotBuilder(
        environment_id=environment_id,
        source_git_commit=source_git_commit,
        observed_at_utc=observed,
        evidence=[],
        rows=[],
        relations=[],
    )
    try:
        state_identity, active_state_identity = _normalize_state(
            builder,
            state_before,
            active_before,
        )
        alert_identities = _normalize_alerts(builder, history_before)
        report_identities, report_alerts = _normalize_attempts_and_reports(
            builder,
            store_root,
            attempts_before,
            alert_identities,
        )
        _normalize_active_snapshot_relations(
            builder,
            active_before,
            active_state_identity,
            alert_identities,
        )
        if state_before is not None:
            latest_id = str(state_before.get("latest_report_id") or "")
            if latest_id not in report_identities:
                raise ProjectionSourceConflictError(
                    "Latest report is absent from verified reporting attempts."
                )
        for report_id, referenced_ids in report_alerts.items():
            report_evidence = report_identities[report_id]
            for position, alert_id in enumerate(sorted(referenced_ids)):
                target = alert_identities.get(alert_id)
                if target is None:
                    raise ProjectionSourceConflictError(
                        "Report alert lineage is incomplete."
                    )
                builder.add_relation(
                    "report_alert_event",
                    report_evidence,
                    target,
                    position=position,
                    evidence=report_evidence,
                )
    except OperationalProjectionError:
        raise
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        raise ProjectionSourceError("Operational source normalization failed.") from exc

    _revalidate_source_snapshots(
        store_root,
        state_before,
        active_before,
        history_before,
        attempts_before,
    )

    evidence = ordered_evidence(builder.evidence)
    relations = ordered_relations(builder.relations)
    manifest = GenerationManifest(
        environment_id=environment_id,
        source_git_commit=source_git_commit,
        evidence=evidence,
        report_state_sha256=canonical_sha256(state_before),
        active_alert_state_sha256=canonical_sha256(dict(sorted(active_before.items()))),
        relations=relations,
    )
    generation_id = manifest.generation_id
    for relation in relations:
        builder.add_row(
            "lineage_edge",
            {
                "generation_id": generation_id,
                "edge_type": relation.edge_type,
                "position": relation.position,
            },
            evidence_links={
                "source_evidence_record_id": relation.source,
                "target_evidence_record_id": relation.target,
                "evidence_record_id": relation.evidence,
            },
        )
    _add_active_snapshot_rows(
        builder,
        generation_id,
        active_before,
        active_state_identity,
    )
    return ProjectionSnapshot(manifest=manifest, rows=ordered_rows(builder.rows))


def plan_projection(
    dsn: str,
    monitoring_store_root: str | Path,
    *,
    environment_id: str,
    source_git_commit: str,
) -> ProjectionCommandResult:
    snapshot = build_projection_snapshot(
        monitoring_store_root,
        environment_id=environment_id,
        source_git_commit=source_git_commit,
    )
    connection = _connect(dsn, role="reader")
    try:
        _require_current_schema(connection)
        head = _read_head(connection, environment_id)
        status = (
            "no_op"
            if head == snapshot.generation_id
            and _snapshot_matches(connection, snapshot, require_ready=True)
            else "planned"
        )
        return ProjectionCommandResult(
            "plan",
            status,
            environment_id,
            snapshot.generation_id,
            head,
            snapshot.counts(),
        )
    finally:
        connection.close()


def verify_projection(
    dsn: str,
    monitoring_store_root: str | Path,
    *,
    environment_id: str,
    source_git_commit: str,
) -> ProjectionCommandResult:
    snapshot = build_projection_snapshot(
        monitoring_store_root,
        environment_id=environment_id,
        source_git_commit=source_git_commit,
    )
    connection = _connect(dsn, role="reader")
    try:
        try:
            _require_current_schema(connection)
        except ProjectionIncompatibleError:
            status = "incompatible"
            head = None
        else:
            head = _read_head(connection, environment_id)
            if head is None:
                status = "missing"
            elif head != snapshot.generation_id:
                status = "stale"
            elif _snapshot_matches(connection, snapshot, require_ready=True):
                status = "ready"
            else:
                status = "mismatch"
        return ProjectionCommandResult(
            "verify",
            status,
            environment_id,
            snapshot.generation_id,
            head,
            snapshot.counts(),
        )
    finally:
        connection.close()


def project_projection(
    dsn: str,
    monitoring_store_root: str | Path,
    *,
    environment_id: str,
    source_git_commit: str,
    clock: Callable[[], datetime] | None = None,
    failure_hook: Callable[[str], None] | None = None,
) -> ProjectionCommandResult:
    """Serialize, insert, validate, and atomically publish one generation."""
    connection = _connect(dsn, role="writer")
    locked = False
    try:
        _require_current_schema(connection)
        _acquire_projection_lock(connection, environment_id)
        locked = True
        snapshot = build_projection_snapshot(
            monitoring_store_root,
            environment_id=environment_id,
            source_git_commit=source_git_commit,
            clock=clock,
        )
        head_before = _read_head(connection, environment_id)
        if (
            head_before == snapshot.generation_id
            and _snapshot_matches(connection, snapshot, require_ready=True)
        ):
            return ProjectionCommandResult(
                "project",
                "no_op",
                environment_id,
                snapshot.generation_id,
                head_before,
                snapshot.counts(),
            )
        try:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                    cursor.execute(
                        "SET LOCAL search_path TO operational_projection, pg_catalog"
                    )
                evidence_ids = _insert_evidence(connection, snapshot.manifest.evidence)
                _insert_generation(connection, snapshot)
                _insert_generation_evidence(connection, snapshot, evidence_ids)
                for table in _TABLE_ORDER:
                    for row in snapshot.rows_for(table):
                        _insert_immutable_row(connection, row, evidence_ids)
                _call_hook(failure_hook, "before_validation")
                if not _snapshot_matches(
                    connection,
                    snapshot,
                    require_ready=False,
                ):
                    raise ProjectionMismatchError(
                        "Inserted projection differs from its normalized snapshot."
                    )
                _call_hook(failure_hook, "before_publish")
                published_at = _utc_now(clock)
                with connection.cursor() as cursor:
                    cursor.execute(
                        "UPDATE operational_projection.projection_generation "
                        "SET ready_at_utc = COALESCE(ready_at_utc, %s) "
                        "WHERE generation_id = %s",
                        (published_at, snapshot.generation_id),
                    )
                    cursor.execute(
                        "INSERT INTO operational_projection.projection_head "
                        "(environment_id, generation_id, published_at_utc) "
                        "VALUES (%s, %s, %s) "
                        "ON CONFLICT (environment_id) DO UPDATE SET "
                        "generation_id = EXCLUDED.generation_id, "
                        "published_at_utc = EXCLUDED.published_at_utc",
                        (environment_id, snapshot.generation_id, published_at),
                    )
                _call_hook(failure_hook, "before_commit")
        except OperationalProjectionError:
            raise
        except Exception as exc:
            raise ProjectionDatabaseError(
                "PostgreSQL could not publish the projection."
            ) from exc
        return ProjectionCommandResult(
            "project",
            "projected",
            environment_id,
            snapshot.generation_id,
            snapshot.generation_id,
            snapshot.counts(),
        )
    finally:
        if locked:
            _release_projection_lock(connection, environment_id)
        connection.close()


def _normalize_state(
    builder: _SnapshotBuilder,
    state: Mapping[str, Any] | None,
    active: Mapping[str, Any],
) -> tuple[EvidenceIdentity | None, EvidenceIdentity | None]:
    if state is None:
        return None, None
    state_digest = canonical_sha256(state)
    effective = str(state.get("latest_through_date") or "not_available")
    state_identity = builder.add_evidence(
        domain="monitoring_report",
        source_kind="load_monitoring_report_state",
        schema_version=str(state["schema_version"]),
        record_id=state_digest,
        digest=state_digest,
        effective_at=effective,
        mutable=True,
    )
    active_digest = canonical_sha256(dict(sorted(active.items())))
    active_identity = builder.add_evidence(
        domain="alert",
        source_kind="load_active_alerts",
        schema_version=ACTIVE_ALERT_STATE_SCHEMA,
        record_id=active_digest,
        digest=active_digest,
        effective_at=effective,
        mutable=True,
    )
    return state_identity, active_identity


def _normalize_alerts(
    builder: _SnapshotBuilder,
    history: Iterable[Mapping[str, Any]],
) -> dict[str, EvidenceIdentity]:
    identities: dict[str, EvidenceIdentity] = {}
    events = list(history)
    for event in events:
        alert_id = str(event["alert_event_id"])
        identity = builder.add_evidence(
            domain="alert",
            source_kind="load_alert_history",
            schema_version=str(event["schema_version"]),
            record_id=alert_id,
            digest=alert_id,
            effective_at=str(event["through_date"]),
        )
        identities[alert_id] = identity
        builder.add_row(
            "alert_event",
            {
                "alert_event_id": alert_id,
                "rule_id": str(event["rule_id"]),
                "through_date": _date(event["through_date"]),
                "event_type": str(event["event_type"]),
                "severity": str(event["severity"]),
                "previous_alert_event_id": event.get("previous_alert_event_id"),
            },
            evidence_links={"evidence_record_id": identity},
        )
    for event in events:
        previous = event.get("previous_alert_event_id")
        if previous is None:
            continue
        source = identities[str(event["alert_event_id"])]
        target = identities.get(str(previous))
        if target is None:
            raise ProjectionSourceConflictError("Alert predecessor is absent.")
        builder.add_relation(
            "alert_predecessor",
            source,
            target,
            position=0,
            evidence=source,
        )
    return identities


def _normalize_attempts_and_reports(
    builder: _SnapshotBuilder,
    store_root: Path,
    attempts: Iterable[Mapping[str, Any]],
    alert_identities: Mapping[str, EvidenceIdentity],
) -> tuple[dict[str, EvidenceIdentity], dict[str, set[str]]]:
    report_identities: dict[str, EvidenceIdentity] = {}
    report_alerts: dict[str, set[str]] = {}
    reports: dict[str, Mapping[str, Any]] = {}
    attempt_items = list(attempts)
    for attempt in attempt_items:
        report_id = attempt.get("report_id")
        if report_id and str(report_id) not in reports:
            path = (
                store_root
                / "reporting"
                / "reports"
                / str(report_id)
                / "report.json"
            )
            report = load_monitoring_report(path)
            if report.get("report_id") != report_id:
                raise ProjectionSourceConflictError(
                    "Reporting attempt and report identity differ."
                )
            reports[str(report_id)] = report

    for report_id in sorted(reports):
        identity, referenced = _normalize_report(
            builder,
            store_root,
            reports[report_id],
            alert_identities,
        )
        report_identities[report_id] = identity
        report_alerts[report_id] = referenced

    for attempt in attempt_items:
        run_id = str(attempt["run_id"])
        attempt_digest = canonical_sha256(attempt)
        identity = builder.add_evidence(
            domain="reporting_run",
            source_kind="load_reporting_attempts",
            schema_version=ATTEMPT_SCHEMA,
            record_id=run_id,
            digest=attempt_digest,
            effective_at=str(attempt["attempted_at_utc"]),
        )
        failure = attempt.get("failure")
        builder.add_row(
            "reporting_attempt",
            {
                "reporting_run_id": run_id,
                "attempted_at_utc": _utc(attempt["attempted_at_utc"]),
                "through_date": _date(attempt["through_date"]),
                "source_run_id": str(attempt["source_pipeline_run_id"]),
                "source_status": str(attempt["source_pipeline_status"]),
                "status": str(attempt["status"]),
                "report_id": attempt.get("report_id"),
                "active_alert_count": _nonnegative_int(
                    attempt.get("active_alert_count"), default=0
                ),
                "failure_at_utc": (
                    _utc(failure["failed_at_utc"])
                    if isinstance(failure, Mapping)
                    else None
                ),
                "failure_type": (
                    str(failure["error_type"])
                    if isinstance(failure, Mapping)
                    else None
                ),
                "failure_message": (
                    str(failure["message"])
                    if isinstance(failure, Mapping)
                    else None
                ),
            },
            evidence_links={"evidence_record_id": identity},
        )
        if attempt.get("report_id"):
            target = report_identities[str(attempt["report_id"])]
            builder.add_relation(
                "reporting_attempt_report",
                identity,
                target,
                position=0,
                evidence=identity,
            )
    return report_identities, report_alerts


def _normalize_report(
    builder: _SnapshotBuilder,
    store_root: Path,
    report: Mapping[str, Any],
    alert_identities: Mapping[str, EvidenceIdentity],
) -> tuple[EvidenceIdentity, set[str]]:
    report_id = str(report["report_id"])
    report_identity = builder.add_evidence(
        domain="monitoring_report",
        source_kind="load_monitoring_report",
        schema_version=str(report["schema_version"]),
        record_id=report_id,
        digest=report_id,
        effective_at=str(report["through_date"]),
    )
    reference = _mapping(report.get("reference"), "Report reference is invalid.")
    calibration_id = str(reference["calibration_id"])
    calibration = load_monitoring_calibration(
        store_root / "reporting" / "calibrations" / calibration_id
    )
    if (
        calibration.get("calibration_id") != calibration_id
        or calibration.get("reference_id") != reference.get("reference_id")
        or calibration.get("policy_sha256") != reference.get("policy_sha256")
    ):
        raise ProjectionSourceConflictError("Report calibration lineage differs.")
    calibration_identity = builder.add_evidence(
        domain="calibration_reference",
        source_kind="load_monitoring_calibration",
        schema_version=str(calibration["schema_version"]),
        record_id=calibration_id,
        digest=calibration_id,
        effective_at=str((calibration["_reference_manifest"]["period"])["end"]),
    )
    reference_manifest = _mapping(
        calibration.get("_reference_manifest"),
        "Calibration reference manifest is invalid.",
    )
    reference_id = str(reference_manifest["reference_id"])
    reference_identity = builder.add_evidence(
        domain="calibration_reference",
        source_kind="load_monitoring_calibration",
        schema_version=str(reference_manifest["schema_version"]),
        record_id=reference_id,
        digest=reference_id,
        effective_at=str(_mapping(reference_manifest["period"], "Invalid period.")["end"]),
    )
    builder.add_relation(
        "calibration_reference",
        calibration_identity,
        reference_identity,
        position=0,
        evidence=calibration_identity,
    )
    builder.add_relation(
        "report_calibration",
        report_identity,
        calibration_identity,
        position=0,
        evidence=report_identity,
    )
    builder.add_relation(
        "report_reference",
        report_identity,
        reference_identity,
        position=0,
        evidence=report_identity,
    )

    resolved_era = resolve_report_model_era(store_root, report)
    model_era_identity: EvidenceIdentity | None = None
    model_era_id: str | None = None
    if resolved_era.get("association_kind") in {
        "active_deployment",
        "bootstrap_adopted",
    }:
        model_era_id = str(resolved_era["model_era_id"])
        stored_era = load_model_era(store_root, model_era_id)
        if stored_era.get("model_era_id") != model_era_id:
            raise ProjectionSourceConflictError("Report model era differs.")
        model_era_identity = _normalize_model_era(builder, stored_era)
        builder.add_relation(
            "report_model_era",
            report_identity,
            model_era_identity,
            position=0,
            evidence=report_identity,
        )

    lineage = report.get("lineage") or {}
    for prediction_id in sorted(
        {str(value) for value in (lineage.get("prediction_ids") or [])}
    ):
        prediction = load_prediction_evidence(store_root, prediction_id)
        if (prediction.get("prediction") or {}).get("prediction_id") != prediction_id:
            raise ProjectionSourceConflictError("Prediction lineage differs.")

    _add_monitoring_report_row(
        builder,
        report,
        report_identity,
        model_era_id,
    )
    _add_report_children(builder, report, calibration, report_identity)
    referenced_alerts = {
        str(value)
        for value in (
            list(report.get("alert_events") or [])
            + list((report.get("active_alerts") or {}).values())
        )
    }
    if not referenced_alerts.issubset(alert_identities):
        raise ProjectionSourceConflictError("Report references an unknown alert.")
    return report_identity, referenced_alerts


def _normalize_model_era(
    builder: _SnapshotBuilder,
    era: Mapping[str, Any],
) -> EvidenceIdentity:
    model_era_id = str(era["model_era_id"])
    identity = builder.add_evidence(
        domain="prediction_model_era",
        source_kind="load_model_era",
        schema_version=str(era["schema_version"]),
        record_id=model_era_id,
        digest=model_era_id,
        effective_at=str(_mapping(era["cutoffs"], "Invalid cutoffs.")["activation_cutoff"]),
    )
    deployment = _mapping(era["deployment"], "Invalid deployment.")
    registry = _mapping(era["registry"], "Invalid registry.")
    cutoffs = _mapping(era["cutoffs"], "Invalid cutoffs.")
    pins = _mapping(era["pins"], "Invalid pins.")
    calibration = _mapping(era["calibration"], "Invalid calibration.")
    builder.add_row(
        "model_era",
        {
            "model_era_id": model_era_id,
            "association_kind": str(era["association_kind"]),
            "deployment_id": str(deployment["deployment_id"]),
            "deployment_generation": _nonnegative_int(deployment["generation"]),
            "registered_model_name": str(registry["registered_model_name"]),
            "model_version": str(registry["model_version"]),
            "fit_cutoff": _date(cutoffs["fit_cutoff"]),
            "activation_cutoff": _date(cutoffs["activation_cutoff"]),
            "bundle_sha256": str(pins["bundle_sha256"]),
            "model_sha256": str(pins["model_sha256"]),
            "dataset_sha256": str(pins["dataset_sha256"]),
            "feature_schema_sha256": str(pins["feature_schema_sha256"]),
            "calibration_sha256": str(pins["calibration_sha256"]),
            "ledger_sha256": str(pins["ledger_sha256"]),
            "calibration_id": str(calibration["calibration_id"]),
            "reference_id": str(calibration["reference_id"]),
        },
        evidence_links={"evidence_record_id": identity},
    )
    return identity


def _add_monitoring_report_row(
    builder: _SnapshotBuilder,
    report: Mapping[str, Any],
    identity: EvidenceIdentity,
    model_era_id: str | None,
) -> None:
    quality = _mapping(report.get("quality"), "Report quality is invalid.")
    freshness = quality.get("freshness") or {}
    coverage = quality.get("coverage") or {}
    source = _mapping(report.get("source_batch"), "Report source is invalid.")
    reference = _mapping(report.get("reference"), "Report reference is invalid.")
    quality_status = str(
        quality.get("status") or quality.get("batch_status") or "not_available"
    )
    batch_status = str(quality.get("batch_status") or source.get("status"))
    builder.add_row(
        "monitoring_report",
        {
            "report_id": str(report["report_id"]),
            "reporting_run_id": str(report["run_id"]),
            "created_at_utc": _utc(report["created_at_utc"]),
            "through_date": _date(report["through_date"]),
            "source_run_id": str(source["run_id"]),
            "source_status": str(source["status"]),
            "calibration_id": str(reference["calibration_id"]),
            "reference_id": str(reference["reference_id"]),
            "policy_sha256": str(reference["policy_sha256"]),
            "quality_status": quality_status,
            "batch_status": batch_status,
            "verdict": str(quality.get("verdict") or "not_available"),
            "watermark_date": _optional_date(freshness.get("common_validated_watermark")),
            "watermark_age_days": _optional_nonnegative_int(freshness.get("watermark_age_days")),
            "objective_days": _optional_nonnegative_int(freshness.get("objective_days")),
            "late_days": _optional_nonnegative_int(freshness.get("late_days")),
            "objective_missed": bool(freshness.get("objective_missed") or False),
            "unresolved_late_date_count": len(freshness.get("unresolved_late_dates") or []),
            "date_count": _nonnegative_int(coverage.get("date_count"), default=0),
            "ren_complete_count": _nonnegative_int(coverage.get("ren_complete_count"), default=0),
            "era5_complete_count": _nonnegative_int(coverage.get("era5_complete_count"), default=0),
            "integration_ready_count": _nonnegative_int(coverage.get("integration_ready_count"), default=0),
            "feature_ready_count": _nonnegative_int(coverage.get("feature_ready_count"), default=0),
            "model_era_id": model_era_id,
        },
        evidence_links={"evidence_record_id": identity},
    )
    for position, issue in enumerate(quality.get("issues") or []):
        if not isinstance(issue, Mapping) or issue.get("severity") not in {
            "warning",
            "critical",
        }:
            continue
        builder.add_row(
            "quality_issue",
            {
                "report_id": str(report["report_id"]),
                "position": position,
                "code": str(issue["code"]),
                "severity": str(issue["severity"]),
            },
            evidence_links={"evidence_record_id": identity},
        )


def _add_report_children(
    builder: _SnapshotBuilder,
    report: Mapping[str, Any],
    calibration: Mapping[str, Any],
    evidence: EvidenceIdentity,
) -> None:
    windows = report.get("windows") or {}
    configured_minimums = (report.get("config") or {}).get("minimum_samples") or {}
    for window_days in (30, 90):
        window = str(window_days)
        payload = windows.get(window) or {}
        available = payload.get("status") == "available"
        minimum = payload.get("minimum_samples", configured_minimums.get(window))
        builder.add_row(
            "monitoring_window",
            {
                "report_id": str(report["report_id"]),
                "window_days": window_days,
                "status": "available" if available else "not_available",
                "sample_count": _nonnegative_int(payload.get("sample_count"), default=0),
                "coverage_ratio": _optional_finite(payload.get("coverage_ratio")),
                "coverage_severity": (
                    str(payload.get("coverage_severity") or "not_available")
                    if available
                    else "not_available"
                ),
                "minimum_samples": _nonnegative_int(minimum, default=0),
                "calendar_start": _optional_date(payload.get("calendar_start")),
                "calendar_end": _optional_date(payload.get("calendar_end")),
            },
            evidence_links={"evidence_record_id": evidence},
        )
        if available:
            _add_performance_rows(
                builder,
                report,
                payload,
                calibration,
                evidence,
                window_days,
            )
            _add_drift_rows(
                builder,
                report,
                payload,
                calibration,
                evidence,
                window_days,
            )


def _add_performance_rows(
    builder: _SnapshotBuilder,
    report: Mapping[str, Any],
    window_payload: Mapping[str, Any],
    calibration: Mapping[str, Any],
    evidence: EvidenceIdentity,
    window_days: int,
) -> None:
    performance = window_payload.get("performance") or {}
    metrics = performance.get("metrics") or {}
    severities = performance.get("severity") or {}
    thresholds = (
        calibration.get("thresholds", {}).get("performance", {}).get(str(window_days), {})
    )
    for metric in ("MAE", "RMSE", "bias", "MAPE_percent", "R2"):
        limit_key = "absolute_bias" if metric == "bias" else metric
        limits = thresholds.get(limit_key)
        if not isinstance(limits, Mapping):
            continue
        value = _optional_finite(metrics.get(metric))
        if performance.get("status") != "available":
            value_status = (
                "insufficient_data"
                if performance.get("status") == "insufficient_data"
                else "not_available"
            )
        elif value is None and metric == "R2":
            candidate = str(metrics.get("R2_status") or "not_available")
            value_status = (
                candidate
                if candidate in {"insufficient_data", "constant_target"}
                else "not_available"
            )
        else:
            value_status = "available" if value is not None else "not_available"
        unit = (
            "sum_of_15_minute_MW_observations"
            if metric in {"MAE", "RMSE", "bias"}
            else "percent"
            if metric == "MAPE_percent"
            else "not_applicable"
        )
        builder.add_row(
            "performance_metric",
            {
                "report_id": str(report["report_id"]),
                "window_days": window_days,
                "metric_name": metric,
                "value": value,
                "value_status": value_status,
                "severity": str(severities.get(metric) or "not_available"),
                "warning_threshold": _optional_finite(limits.get("warning")),
                "critical_threshold": _optional_finite(limits.get("critical")),
                "direction": str(limits.get("direction") or "upper"),
                "unit_or_scale": unit,
            },
            evidence_links={"evidence_record_id": evidence},
        )


def _add_drift_rows(
    builder: _SnapshotBuilder,
    report: Mapping[str, Any],
    window_payload: Mapping[str, Any],
    calibration: Mapping[str, Any],
    evidence: EvidenceIdentity,
    window_days: int,
) -> None:
    position = 0
    drift = window_payload.get("feature_drift") or {}
    calibrated = calibration.get("thresholds", {}).get("feature_drift", {})
    for feature in sorted(drift):
        for comparator in sorted((drift.get(feature) or {})):
            statistics = (drift[feature] or {}).get(comparator) or {}
            for detector in ("ks_statistic", "normalized_wasserstein"):
                value = _optional_finite(statistics.get(detector))
                limits = (
                    calibrated.get(feature, {})
                    .get(str(window_days), {})
                    .get(comparator, {})
                    .get(detector)
                )
                if value is None or not isinstance(limits, Mapping):
                    continue
                builder.add_row(
                    "drift_measurement",
                    {
                        "report_id": str(report["report_id"]),
                        "window_days": window_days,
                        "position": position,
                        "feature": str(feature),
                        "comparator": str(comparator),
                        "detector": detector,
                        "value": value,
                        "severity": threshold_severity(value, limits),
                        "warning_threshold": _finite(limits["warning"]),
                        "critical_threshold": _finite(limits["critical"]),
                        "direction": str(limits.get("direction") or "upper"),
                    },
                    evidence_links={"evidence_record_id": evidence},
                )
                position += 1


def _normalize_active_snapshot_relations(
    builder: _SnapshotBuilder,
    active: Mapping[str, Any],
    active_state_identity: EvidenceIdentity | None,
    alert_identities: Mapping[str, EvidenceIdentity],
) -> None:
    if not active:
        return
    if active_state_identity is None:
        raise ProjectionSourceConflictError("Active alerts have no verified state.")
    for position, (rule_id, alert_id) in enumerate(sorted(active.items())):
        target = alert_identities.get(str(alert_id))
        if target is None:
            raise ProjectionSourceConflictError("Active alert event is absent.")
        builder.add_relation(
            "active_alert_event",
            active_state_identity,
            target,
            position=position,
            evidence=active_state_identity,
        )


def _add_active_snapshot_rows(
    builder: _SnapshotBuilder,
    generation_id: str,
    active: Mapping[str, Any],
    active_state_identity: EvidenceIdentity | None,
) -> None:
    if not active:
        return
    assert active_state_identity is not None
    for rule_id, alert_id in sorted(active.items()):
        builder.add_row(
            "active_alert_snapshot",
            {
                "generation_id": generation_id,
                "rule_id": str(rule_id),
                "alert_event_id": str(alert_id),
            },
            evidence_links={"evidence_record_id": active_state_identity},
        )


def _revalidate_source_snapshots(
    store_root: Path,
    state_before: Mapping[str, Any] | None,
    active_before: Mapping[str, Any],
    history_before: Iterable[Mapping[str, Any]],
    attempts_before: Iterable[Mapping[str, Any]],
) -> None:
    try:
        values = (
            load_monitoring_report_state(store_root),
            load_active_alerts(store_root),
            load_alert_history(store_root),
            load_reporting_attempts(store_root),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ProjectionSourceError("Operational source revalidation failed.") from exc
    before = (
        state_before,
        dict(active_before),
        list(history_before),
        list(attempts_before),
    )
    if canonical_sha256(before) != canonical_sha256(values):
        raise ProjectionSourceConflictError(
            "Operational source changed during projection scan."
        )


def _connect(dsn: str, *, role: str) -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise ProjectionDatabaseError("PostgreSQL driver is unavailable.") from exc
    try:
        connection = psycopg.connect(
            dsn,
            autocommit=True,
            application_name=f"wind_forecast_projection_{role}",
            connect_timeout=5,
        )
        with connection.cursor() as cursor:
            cursor.execute("SET TIME ZONE 'UTC'")
            cursor.execute("SET statement_timeout = '30s'")
            cursor.execute("SET lock_timeout = '5s'")
            cursor.execute(
                "SET search_path TO operational_projection, pg_catalog"
            )
        return connection
    except Exception as exc:
        raise ProjectionDatabaseError("PostgreSQL is unavailable.") from exc


def _require_current_schema(connection: Any) -> None:
    try:
        migrations = discover_migrations()
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(
                    "SELECT to_regclass('operational_projection.schema_migration')"
                )
                if cursor.fetchone()[0] is None:
                    raise ProjectionIncompatibleError(
                        "Projection schema is unavailable."
                    )
                cursor.execute(
                    "SELECT version, name, sha256 "
                    "FROM operational_projection.schema_migration ORDER BY version"
                )
                applied = tuple(
                    AppliedMigration(int(version), str(name), str(checksum))
                    for version, name, checksum in cursor.fetchall()
                )
        status = _compare(migrations, applied)
        if status.pending:
            raise ProjectionIncompatibleError("Projection schema is not current.")
    except ProjectionIncompatibleError:
        raise
    except OperationalProjectionMigrationError as exc:
        raise ProjectionIncompatibleError(
            "Projection schema is incompatible."
        ) from exc
    except Exception as exc:
        raise ProjectionDatabaseError("Projection schema is unavailable.") from exc


def _read_head(connection: Any, environment_id: str) -> str | None:
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT generation_id FROM operational_projection.projection_head "
                "WHERE environment_id = %s",
                (environment_id,),
            )
            row = cursor.fetchone()
            return None if row is None else str(row[0])
    except Exception as exc:
        raise ProjectionDatabaseError("Projection head is unavailable.") from exc


def _acquire_projection_lock(connection: Any, environment_id: str) -> None:
    if environment_id != "local":
        raise ProjectionProvenanceError("Projection environment is unsupported.")
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_lock(%s)", (PROJECTION_LOCK_KEY,))
    except Exception as exc:
        raise ProjectionDatabaseError("Projection lock is unavailable.") from exc


def _release_projection_lock(connection: Any, environment_id: str) -> None:
    if environment_id != "local":
        return
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_unlock(%s)", (PROJECTION_LOCK_KEY,))
    except Exception:
        pass


def _insert_evidence(
    connection: Any,
    evidence: Iterable[EvidenceRecord],
) -> dict[EvidenceIdentity, int]:
    result: dict[EvidenceIdentity, int] = {}
    with connection.cursor() as cursor:
        for record in evidence:
            item = record.identity
            cursor.execute(
                "INSERT INTO operational_projection.evidence_record "
                "(domain, source_kind, schema_version, record_id, sha256, "
                "effective_at, observed_at_utc) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (domain, source_kind, schema_version, record_id, sha256) "
                "DO NOTHING RETURNING evidence_record_id",
                (
                    item.domain,
                    item.source_kind,
                    item.schema_version,
                    item.record_id,
                    item.sha256,
                    record.effective_at,
                    record.observed_at_utc,
                ),
            )
            inserted = cursor.fetchone()
            if inserted is None:
                cursor.execute(
                    "SELECT evidence_record_id, effective_at "
                    "FROM operational_projection.evidence_record "
                    "WHERE domain = %s AND source_kind = %s AND schema_version = %s "
                    "AND record_id = %s AND sha256 = %s",
                    (
                        item.domain,
                        item.source_kind,
                        item.schema_version,
                        item.record_id,
                        item.sha256,
                    ),
                )
                existing = cursor.fetchone()
                if existing is None or str(existing[1]) != record.effective_at:
                    raise ProjectionMismatchError(
                        "Existing evidence metadata is incompatible."
                    )
                result[item] = int(existing[0])
            else:
                result[item] = int(inserted[0])
    return result


def _insert_generation(connection: Any, snapshot: ProjectionSnapshot) -> None:
    counts = snapshot.counts()
    columns = [
        "generation_id",
        "environment_id",
        "contract_version",
        "schema_version",
        "projector_version",
        "source_git_commit",
        "source_set_sha256",
        *_COUNT_COLUMNS,
    ]
    values = [
        snapshot.generation_id,
        snapshot.manifest.environment_id,
        snapshot.manifest.contract_version,
        snapshot.manifest.schema_version,
        snapshot.manifest.projector_version,
        snapshot.manifest.source_git_commit,
        snapshot.manifest.source_set_sha256,
        *(counts[column] for column in _COUNT_COLUMNS),
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO operational_projection.projection_generation ("
            + ", ".join(columns)
            + f") VALUES ({placeholders}) ON CONFLICT (generation_id) DO NOTHING",
            values,
        )


def _insert_generation_evidence(
    connection: Any,
    snapshot: ProjectionSnapshot,
    evidence_ids: Mapping[EvidenceIdentity, int],
) -> None:
    with connection.cursor() as cursor:
        for record in snapshot.manifest.evidence:
            cursor.execute(
                "INSERT INTO operational_projection.generation_evidence "
                "(generation_id, evidence_record_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                (snapshot.generation_id, evidence_ids[record.identity]),
            )


def _insert_immutable_row(
    connection: Any,
    row: RelationalRow,
    evidence_ids: Mapping[EvidenceIdentity, int],
) -> None:
    if row.table not in _TABLE_ORDER:
        raise ProjectionMismatchError("Projection table is unsupported.")
    values = row.value_map()
    values.update(
        {
            link.column: evidence_ids[link.evidence]
            for link in row.evidence_links
        }
    )
    columns = sorted(values)
    if not all(_SQL_IDENTIFIER.fullmatch(column) for column in columns):
        raise ProjectionMismatchError("Projection column is unsupported.")
    placeholders = ", ".join(["%s"] * len(columns))
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO operational_projection.{row.table} "
            f"({', '.join(columns)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
            [values[column] for column in columns],
        )


def _snapshot_matches(
    connection: Any,
    snapshot: ProjectionSnapshot,
    *,
    require_ready: bool,
) -> bool:
    try:
        evidence_ids = _read_generation_evidence(connection, snapshot)
        if evidence_ids is None:
            return False
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT environment_id, contract_version, schema_version, "
                "projector_version, source_git_commit, source_set_sha256, "
                + ", ".join(_COUNT_COLUMNS)
                + ", ready_at_utc FROM operational_projection.projection_generation "
                "WHERE generation_id = %s",
                (snapshot.generation_id,),
            )
            generation = cursor.fetchone()
        if generation is None:
            return False
        expected = (
            snapshot.manifest.environment_id,
            snapshot.manifest.contract_version,
            snapshot.manifest.schema_version,
            snapshot.manifest.projector_version,
            snapshot.manifest.source_git_commit,
            snapshot.manifest.source_set_sha256,
            *(snapshot.counts()[column] for column in _COUNT_COLUMNS),
        )
        if tuple(generation[:-1]) != expected:
            return False
        if require_ready and generation[-1] is None:
            return False
        for table in _TABLE_ORDER:
            expected_rows = {
                _resolved_row_tuple(row, evidence_ids)
                for row in snapshot.rows_for(table)
            }
            actual_rows = _read_table_rows(
                connection,
                table,
                snapshot.rows_for(table),
                snapshot.generation_id,
            )
            if expected_rows != actual_rows:
                return False
        return True
    except OperationalProjectionError:
        raise
    except Exception as exc:
        raise ProjectionDatabaseError("Projection verification failed.") from exc


def _read_generation_evidence(
    connection: Any,
    snapshot: ProjectionSnapshot,
) -> dict[EvidenceIdentity, int] | None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT e.evidence_record_id, e.domain, e.source_kind, "
            "e.schema_version, e.record_id, e.sha256, e.effective_at "
            "FROM operational_projection.generation_evidence ge "
            "JOIN operational_projection.evidence_record e "
            "ON e.evidence_record_id = ge.evidence_record_id "
            "WHERE ge.generation_id = %s",
            (snapshot.generation_id,),
        )
        rows = cursor.fetchall()
    actual: dict[EvidenceIdentity, tuple[int, str]] = {}
    for row in rows:
        identity = EvidenceIdentity(
            domain=str(row[1]),
            source_kind=str(row[2]),
            schema_version=str(row[3]),
            record_id=str(row[4]),
            sha256=str(row[5]),
        )
        actual[identity] = (int(row[0]), str(row[6]))
    expected = {
        record.identity: record.effective_at
        for record in snapshot.manifest.evidence
    }
    if set(actual) != set(expected):
        return None
    if any(actual[key][1] != effective for key, effective in expected.items()):
        return None
    return {key: value[0] for key, value in actual.items()}


def _read_table_rows(
    connection: Any,
    table: str,
    expected: tuple[RelationalRow, ...],
    generation_id: str,
) -> set[tuple[tuple[str, Any], ...]]:
    columns = sorted(_TABLE_COLUMNS[table])
    if table in {"active_alert_snapshot", "lineage_edge"}:
        query = (
            f"SELECT {', '.join('t.' + column for column in columns)} "
            f"FROM operational_projection.{table} t WHERE t.generation_id = %s"
        )
    else:
        query = (
            f"SELECT {', '.join('t.' + column for column in columns)} "
            f"FROM operational_projection.{table} t "
            "JOIN operational_projection.generation_evidence ge "
            "ON ge.evidence_record_id = t.evidence_record_id "
            "WHERE ge.generation_id = %s"
        )
    with connection.cursor() as cursor:
        cursor.execute(query, (generation_id,))
        return {
            tuple((column, value) for column, value in zip(columns, values, strict=True))
            for values in cursor.fetchall()
        }


def _resolved_row_tuple(
    row: RelationalRow,
    evidence_ids: Mapping[EvidenceIdentity, int],
) -> tuple[tuple[str, Any], ...]:
    values = row.value_map()
    values.update(
        {
            link.column: evidence_ids[link.evidence]
            for link in row.evidence_links
        }
    )
    return tuple(sorted(values.items()))


def _validate_minimized_row(row: RelationalRow) -> None:
    for _column, value in row.values:
        if isinstance(value, str):
            lowered = value.lower()
            if (
                any(ord(character) < 32 or ord(character) == 127 for character in value)
                or "/" in value
                or "\\" in value
                or "://" in lowered
                or "models:/" in lowered
                or any(
                    marker in lowered
                    for marker in (
                        "password=",
                        "token=",
                        "secret=",
                        "connection_string",
                        "tracking_uri",
                    )
                )
            ):
                raise ProjectionSourceError(
                    "A normalized field violates data-minimization rules."
                )


def _mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectionSourceError(message)
    return value


def _date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _optional_date(value: Any) -> date | None:
    return None if value in {None, ""} else _date(value)


def _utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ProjectionSourceError("Projected timestamp is not timezone-aware.")
    return parsed.astimezone(timezone.utc)


def _utc_now(clock: Callable[[], datetime] | None) -> datetime:
    value = (clock or (lambda: datetime.now(timezone.utc)))()
    if value.tzinfo is None:
        raise ProjectionProvenanceError("Projection clock must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _nonnegative_int(value: Any, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectionSourceError("Projected count is invalid.")
    return value


def _optional_nonnegative_int(value: Any) -> int | None:
    return None if value is None else _nonnegative_int(value)


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectionSourceError("Projected numeric value is invalid.")
    number = float(value)
    if not math.isfinite(number):
        raise ProjectionSourceError("Projected numeric value is not finite.")
    return number


def _optional_finite(value: Any) -> float | None:
    return None if value is None else _finite(value)


def _call_hook(hook: Callable[[str], None] | None, stage: str) -> None:
    if hook is not None:
        hook(stage)


__all__ = [
    "ACTIVE_ALERT_STATE_SCHEMA",
    "ATTEMPT_SCHEMA",
    "COMMAND_SCHEMA_VERSION",
    "ERROR_SCHEMA_VERSION",
    "OperationalProjectionError",
    "PROJECTION_LOCK_KEY",
    "ProjectionCommandResult",
    "ProjectionDatabaseError",
    "ProjectionIncompatibleError",
    "ProjectionMismatchError",
    "ProjectionProvenanceError",
    "ProjectionSourceConflictError",
    "ProjectionSourceError",
    "ProjectionSourceNotStableError",
    "build_projection_snapshot",
    "plan_projection",
    "project_projection",
    "resolve_source_git_commit",
    "verify_projection",
]
