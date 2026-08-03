"""Deadline-bounded reader for the derived operational PostgreSQL projection.

The module deliberately does not import psycopg at import time.  PostgreSQL
selects identities, order, and pagination; callers must still revalidate every
returned row against the authoritative verified loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from time import monotonic
from typing import Any, Callable, Literal, Mapping, Sequence

from .operational_projection_migrations import discover_migrations
from .operational_projection_models import (
    CONTRACT_VERSION,
    PROJECTOR_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    canonical_sha256,
)


class OperationalProjectionReaderError(RuntimeError):
    """Base class for sanitized reader failures."""


class OperationalProjectionUnavailableError(OperationalProjectionReaderError):
    """The required projection is absent, stale, or incompatible."""


class OperationalProjectionTimeoutError(OperationalProjectionReaderError):
    """The remaining operational-query deadline was exhausted."""


@dataclass(frozen=True)
class ProjectedEvidence:
    """Original verified evidence identity stored with one projected row."""

    domain: str
    source_kind: str
    schema_version: str
    record_id: str
    sha256: str
    effective_at: str


@dataclass(frozen=True)
class ProjectedRow:
    """One normalized row selected from the current ready generation."""

    values: Mapping[str, Any]
    evidence: ProjectedEvidence


@dataclass(frozen=True)
class ProjectedReport:
    """Projected values required to revalidate one report-backed query."""

    report: ProjectedRow
    quality_issues: tuple[ProjectedRow, ...] = ()
    window: ProjectedRow | None = None
    performance_metrics: tuple[ProjectedRow, ...] = ()
    drift_measurements: tuple[ProjectedRow, ...] = ()
    calibration: ProjectedEvidence | None = None
    model_era: ProjectedRow | None = None


@dataclass(frozen=True)
class ProjectedAlerts:
    """Complete current alert snapshot plus SQL-selected ordered candidates."""

    history: tuple[ProjectedRow, ...]
    active: Mapping[str, str]
    active_evidence: ProjectedEvidence | None
    selected_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ProjectionHead:
    generation_id: str
    alert_event_count: int
    active_alert_snapshot_count: int


@dataclass(frozen=True)
class _DeadlineBudget:
    deadline: float
    clock: Callable[[], float]

    @classmethod
    def create(
        cls,
        timeout_seconds: float,
        clock: Callable[[], float],
    ) -> "_DeadlineBudget":
        timeout = _bounded_timeout(timeout_seconds)
        return cls(clock() + timeout, clock)

    def remaining(self) -> float:
        value = self.deadline - self.clock()
        if not math.isfinite(value) or value <= 0:
            raise OperationalProjectionTimeoutError(
                "Projection query deadline expired."
            )
        return value


class _DeadlineCursor:
    """Apply the current remaining deadline before every SQL statement."""

    def __init__(self, cursor: Any, budget: _DeadlineBudget) -> None:
        self._cursor = cursor
        self._budget = budget

    def execute(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> Any:
        milliseconds = _statement_timeout_milliseconds(
            self._budget.remaining()
        )
        self._cursor.execute(
            "SELECT set_config('statement_timeout', %s, true), "
            "set_config('lock_timeout', %s, true)",
            (f"{milliseconds}ms", f"{milliseconds}ms"),
        )
        _statement_timeout_milliseconds(self._budget.remaining())
        result = self._cursor.execute(sql, params)
        self._budget.remaining()
        return result

    def fetchone(self) -> Any:
        value = self._cursor.fetchone()
        self._budget.remaining()
        return value

    def fetchall(self) -> Any:
        values = self._cursor.fetchall()
        self._budget.remaining()
        return values


_REPORT_COLUMNS = (
    "report_id",
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
)
_WINDOW_COLUMNS = (
    "report_id",
    "window_days",
    "status",
    "sample_count",
    "coverage_ratio",
    "coverage_severity",
    "minimum_samples",
    "calendar_start",
    "calendar_end",
)
_PERFORMANCE_COLUMNS = (
    "report_id",
    "window_days",
    "metric_name",
    "value",
    "value_status",
    "severity",
    "warning_threshold",
    "critical_threshold",
    "direction",
    "unit_or_scale",
)
_DRIFT_COLUMNS = (
    "report_id",
    "window_days",
    "position",
    "feature",
    "comparator",
    "detector",
    "value",
    "severity",
    "warning_threshold",
    "critical_threshold",
    "direction",
)
_ATTEMPT_COLUMNS = (
    "reporting_run_id",
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
)
_MODEL_ERA_COLUMNS = (
    "model_era_id",
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
)
_ALERT_COLUMNS = (
    "alert_event_id",
    "rule_id",
    "through_date",
    "event_type",
    "severity",
    "previous_alert_event_id",
)


@dataclass(frozen=True)
class OperationalProjectionReader:
    """Read only the current compatible projection with bounded statements."""

    dsn: str
    environment_id: str = "local"
    clock: Callable[[], float] = monotonic

    def __post_init__(self) -> None:
        if self.environment_id != "local" or not self.dsn.strip():
            raise ValueError("Operational projection reader configuration is invalid.")

    def select_report(
        self,
        *,
        selector: Literal["latest", "exact"],
        report_id: str | None,
        report_state_sha256: str | None,
        report_state_schema_version: str | None,
        report_state_effective_at: str | None,
        detail: Literal["quality", "performance", "drift"],
        window_days: int | None,
        timeout_seconds: float,
    ) -> ProjectedReport | None:
        """Select one report identity and its query-scoped normalized values."""

        def operation(cursor: Any, head: _ProjectionHead) -> ProjectedReport | None:
            if selector == "latest":
                self._require_report_state(
                    cursor,
                    head.generation_id,
                    report_state_sha256,
                    report_state_schema_version,
                    report_state_effective_at,
                )
            params: list[Any] = [head.generation_id]
            where = ""
            order = (
                " AND mr.report_id = (SELECT ra.report_id FROM "
                "operational_projection.reporting_attempt ra JOIN "
                "operational_projection.generation_evidence attempt_ge ON "
                "attempt_ge.evidence_record_id = ra.evidence_record_id "
                "WHERE attempt_ge.generation_id = %s AND ra.report_id IS NOT NULL "
                "ORDER BY ra.attempted_at_utc DESC, ra.reporting_run_id DESC "
                "LIMIT 1)"
            )
            params.append(head.generation_id)
            if selector == "exact":
                where = " AND mr.report_id = %s"
                params = [head.generation_id, report_id]
                order = ""
            cursor.execute(
                _entity_select("monitoring_report", "mr", _REPORT_COLUMNS)
                + where
                + order,
                tuple(params),
            )
            report_row = _projected_row(cursor, cursor.fetchone(), _REPORT_COLUMNS)
            if report_row is None:
                return None
            selected_report_id = str(report_row.values["report_id"])
            issues: tuple[ProjectedRow, ...] = ()
            window: ProjectedRow | None = None
            metrics: tuple[ProjectedRow, ...] = ()
            drift: tuple[ProjectedRow, ...] = ()
            calibration: ProjectedEvidence | None = None
            model_era: ProjectedRow | None = None

            if detail == "quality":
                cursor.execute(
                    _entity_select(
                        "quality_issue",
                        "qi",
                        ("report_id", "position", "code", "severity"),
                    )
                    + " AND qi.report_id = %s ORDER BY qi.position",
                    (head.generation_id, selected_report_id),
                )
                issues = _projected_rows(
                    cursor,
                    cursor.fetchall(),
                    ("report_id", "position", "code", "severity"),
                )
            else:
                if window_days not in {30, 90}:
                    raise OperationalProjectionUnavailableError(
                        "Projection query window is incompatible."
                    )
                cursor.execute(
                    _entity_select("monitoring_window", "mw", _WINDOW_COLUMNS)
                    + " AND mw.report_id = %s AND mw.window_days = %s",
                    (head.generation_id, selected_report_id, window_days),
                )
                window = _projected_row(cursor, cursor.fetchone(), _WINDOW_COLUMNS)
                if window is None:
                    raise OperationalProjectionUnavailableError(
                        "Projected monitoring window is unavailable."
                    )
                calibration = self._lineage_evidence(
                    cursor,
                    head.generation_id,
                    report_row.evidence,
                    "report_calibration",
                )
                model_era_id = report_row.values.get("model_era_id")
                if model_era_id is not None:
                    relation = self._lineage_evidence(
                        cursor,
                        head.generation_id,
                        report_row.evidence,
                        "report_model_era",
                    )
                    cursor.execute(
                        _entity_select("model_era", "me", _MODEL_ERA_COLUMNS)
                        + " AND me.model_era_id = %s",
                        (head.generation_id, model_era_id),
                    )
                    model_era = _projected_row(
                        cursor,
                        cursor.fetchone(),
                        _MODEL_ERA_COLUMNS,
                    )
                    if model_era is None or model_era.evidence != relation:
                        raise OperationalProjectionUnavailableError(
                            "Projected model-era lineage is unavailable."
                        )
                if detail == "performance":
                    cursor.execute(
                        _entity_select(
                            "performance_metric",
                            "pm",
                            _PERFORMANCE_COLUMNS,
                        )
                        + " AND pm.report_id = %s AND pm.window_days = %s "
                        "ORDER BY pm.metric_name",
                        (head.generation_id, selected_report_id, window_days),
                    )
                    metrics = _projected_rows(
                        cursor,
                        cursor.fetchall(),
                        _PERFORMANCE_COLUMNS,
                    )
                else:
                    cursor.execute(
                        _entity_select(
                            "drift_measurement",
                            "dm",
                            _DRIFT_COLUMNS,
                        )
                        + " AND dm.report_id = %s AND dm.window_days = %s "
                        "ORDER BY dm.position",
                        (head.generation_id, selected_report_id, window_days),
                    )
                    drift = _projected_rows(
                        cursor,
                        cursor.fetchall(),
                        _DRIFT_COLUMNS,
                    )
            return ProjectedReport(
                report=report_row,
                quality_issues=issues,
                window=window,
                performance_metrics=metrics,
                drift_measurements=drift,
                calibration=calibration,
                model_era=model_era,
            )

        return self._read(timeout_seconds, operation)

    def select_attempt(
        self,
        *,
        id_type: Literal["reporting_run_id", "report_id"],
        identifier: str,
        timeout_seconds: float,
    ) -> ProjectedRow | None:
        """Select one reporting attempt by an existing exact identifier."""

        def operation(cursor: Any, head: _ProjectionHead) -> ProjectedRow | None:
            column = "reporting_run_id" if id_type == "reporting_run_id" else "report_id"
            cursor.execute(
                _entity_select("reporting_attempt", "ra", _ATTEMPT_COLUMNS)
                + f" AND ra.{column} = %s",
                (head.generation_id, identifier),
            )
            rows = cursor.fetchall()
            if len(rows) > 1:
                raise OperationalProjectionUnavailableError(
                    "Projected reporting-attempt identity is ambiguous."
                )
            return _projected_row(cursor, rows[0], _ATTEMPT_COLUMNS) if rows else None

        return self._read(timeout_seconds, operation)

    def select_alerts(
        self,
        *,
        selector: Literal["latest", "exact", "date_interval"],
        identifier: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
        offset: int | None = None,
        timeout_seconds: float,
    ) -> ProjectedAlerts:
        """Select alert identities in the loader's causal order in PostgreSQL."""

        def operation(cursor: Any, head: _ProjectionHead) -> ProjectedAlerts:
            active, active_evidence = self._active_snapshot(cursor, head)
            base_sql = _alert_chain_sql()
            cursor.execute(
                base_sql + _alert_select_clause(),
                (head.generation_id,),
            )
            history = _projected_rows(cursor, cursor.fetchall(), _ALERT_COLUMNS)
            if len(history) != head.alert_event_count:
                raise OperationalProjectionUnavailableError(
                    "Projected alert history is incomplete or non-causal."
                )

            where = ""
            params: list[Any] = [head.generation_id]
            if selector == "latest":
                where = (
                    " WHERE chain.alert_event_id IN ("
                    "SELECT alert_event_id FROM operational_projection."
                    "active_alert_snapshot WHERE generation_id = %s)"
                )
                params.append(head.generation_id)
            elif selector == "exact":
                where = " WHERE chain.alert_event_id = %s"
                params.append(identifier)
            elif selector == "date_interval":
                where = " WHERE chain.through_date BETWEEN %s AND %s"
                params.extend((start_date, end_date))
            else:
                raise OperationalProjectionUnavailableError(
                    "Projection alert selector is incompatible."
                )
            pagination = ""
            if selector != "exact":
                pagination = " LIMIT %s OFFSET %s"
                params.extend((limit, offset))
            cursor.execute(
                base_sql
                + " SELECT chain.alert_event_id FROM chain"
                + where
                + " ORDER BY chain.through_date, chain.rule_id, "
                "chain.causal_depth, chain.alert_event_id"
                + pagination,
                tuple(params),
            )
            selected_ids = tuple(str(row[0]) for row in cursor.fetchall())
            return ProjectedAlerts(history, active, active_evidence, selected_ids)

        return self._read(timeout_seconds, operation)

    def _read(
        self,
        timeout_seconds: float,
        operation: Callable[[Any, _ProjectionHead], Any],
    ) -> Any:
        budget = _DeadlineBudget.create(timeout_seconds, self.clock)
        connection = _connect(self.dsn, budget)
        try:
            with connection.transaction():
                with connection.cursor() as raw_cursor:
                    _statement_timeout_milliseconds(budget.remaining())
                    raw_cursor.execute("SET TRANSACTION READ ONLY")
                    budget.remaining()
                    cursor = _DeadlineCursor(raw_cursor, budget)
                    self._require_migrations(cursor)
                    head = self._require_head(cursor)
                    return operation(cursor, head)
        except OperationalProjectionReaderError:
            raise
        except Exception as exc:
            if _is_statement_timeout(exc):
                raise OperationalProjectionTimeoutError(
                    "Projection query deadline expired."
                ) from exc
            raise OperationalProjectionUnavailableError(
                "Required operational projection is unavailable."
            ) from exc
        finally:
            connection.close()

    @staticmethod
    def _require_migrations(cursor: Any) -> None:
        cursor.execute(
            "SELECT to_regclass('operational_projection.schema_migration')"
        )
        row = cursor.fetchone()
        if row is None or row[0] is None:
            raise OperationalProjectionUnavailableError(
                "Projection schema is unavailable."
            )
        cursor.execute(
            "SELECT version, name, sha256 FROM "
            "operational_projection.schema_migration ORDER BY version"
        )
        applied = tuple(
            (int(version), str(name), str(checksum))
            for version, name, checksum in cursor.fetchall()
        )
        expected = tuple(
            (item.version, item.name, item.sha256)
            for item in discover_migrations()
        )
        if applied != expected:
            raise OperationalProjectionUnavailableError(
                "Projection schema is incompatible."
            )

    def _require_head(self, cursor: Any) -> _ProjectionHead:
        cursor.execute(
            "SELECT h.generation_id, g.environment_id, g.contract_version, "
            "g.schema_version, g.projector_version, g.ready_at_utc, "
            "g.source_set_sha256, g.generation_evidence_count, g.alert_event_count, "
            "g.active_alert_snapshot_count "
            "FROM operational_projection.projection_head h "
            "JOIN operational_projection.projection_generation g "
            "ON g.generation_id = h.generation_id "
            "WHERE h.environment_id = %s",
            (self.environment_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise OperationalProjectionUnavailableError(
                "Projection head is unavailable."
            )
        (
            generation_id,
            environment_id,
            contract_version,
            schema_version,
            projector_version,
            ready_at_utc,
            source_set_sha256,
            evidence_count,
            alert_count,
            active_count,
        ) = row
        if (
            str(environment_id) != self.environment_id
            or str(contract_version) != CONTRACT_VERSION
            or str(schema_version) != RELATIONAL_SCHEMA_VERSION
            or str(projector_version) != PROJECTOR_VERSION
            or ready_at_utc is None
        ):
            raise OperationalProjectionUnavailableError(
                "Projection head is incompatible."
            )
        cursor.execute(
            "SELECT er.domain, er.source_kind, er.schema_version, er.record_id, "
            "er.sha256, er.effective_at FROM "
            "operational_projection.generation_evidence ge JOIN "
            "operational_projection.evidence_record er ON "
            "er.evidence_record_id = ge.evidence_record_id "
            "WHERE ge.generation_id = %s ORDER BY er.source_kind, "
            "er.schema_version, er.record_id, er.sha256, er.effective_at, er.domain",
            (generation_id,),
        )
        source_rows = cursor.fetchall()
        source_manifest = [
            {
                "domain": str(domain),
                "source_kind": str(source_kind),
                "schema_version": str(source_schema),
                "record_id": str(record_id),
                "sha256": str(checksum),
                "effective_at": str(effective_at),
            }
            for (
                domain,
                source_kind,
                source_schema,
                record_id,
                checksum,
                effective_at,
            ) in source_rows
        ]
        if (
            len(source_rows) != int(evidence_count)
            or canonical_sha256(source_manifest) != str(source_set_sha256)
        ):
            raise OperationalProjectionUnavailableError(
                "Projection generation is incomplete."
            )
        return _ProjectionHead(
            str(generation_id),
            int(alert_count),
            int(active_count),
        )

    @staticmethod
    def _require_report_state(
        cursor: Any,
        generation_id: str,
        report_state_sha256: str | None,
        report_state_schema_version: str | None,
        report_state_effective_at: str | None,
    ) -> None:
        cursor.execute(
            "SELECT er.domain, er.source_kind, er.schema_version, er.record_id, "
            "er.sha256, er.effective_at "
            "FROM operational_projection.evidence_record er "
            "JOIN operational_projection.generation_evidence ge "
            "ON ge.evidence_record_id = er.evidence_record_id "
            "WHERE ge.generation_id = %s "
            "AND er.source_kind = 'load_monitoring_report_state'",
            (generation_id,),
        )
        rows = cursor.fetchall()
        if report_state_sha256 is None:
            if rows:
                raise OperationalProjectionUnavailableError(
                    "Projection report state is stale."
                )
            return
        if len(rows) != 1:
            raise OperationalProjectionUnavailableError(
                "Projection report state is unavailable."
            )
        evidence = _evidence(rows[0])
        if (
            evidence.domain != "monitoring_report"
            or evidence.source_kind != "load_monitoring_report_state"
            or evidence.schema_version != report_state_schema_version
            or evidence.record_id != report_state_sha256
            or evidence.sha256 != report_state_sha256
            or evidence.effective_at != report_state_effective_at
        ):
            raise OperationalProjectionUnavailableError(
                "Projection report state is stale."
            )

    @staticmethod
    def _lineage_evidence(
        cursor: Any,
        generation_id: str,
        source: ProjectedEvidence,
        edge_type: str,
    ) -> ProjectedEvidence:
        cursor.execute(
            "SELECT target.domain, target.source_kind, target.schema_version, "
            "target.record_id, target.sha256, target.effective_at "
            "FROM operational_projection.lineage_edge le "
            "JOIN operational_projection.evidence_record source "
            "ON source.evidence_record_id = le.source_evidence_record_id "
            "JOIN operational_projection.evidence_record target "
            "ON target.evidence_record_id = le.target_evidence_record_id "
            "JOIN operational_projection.generation_evidence target_ge "
            "ON target_ge.evidence_record_id = target.evidence_record_id "
            "AND target_ge.generation_id = le.generation_id "
            "WHERE le.generation_id = %s AND le.edge_type = %s "
            "AND source.domain = %s AND source.source_kind = %s "
            "AND source.schema_version = %s AND source.record_id = %s "
            "AND source.sha256 = %s",
            (
                generation_id,
                edge_type,
                source.domain,
                source.source_kind,
                source.schema_version,
                source.record_id,
                source.sha256,
            ),
        )
        rows = cursor.fetchall()
        if len(rows) != 1:
            raise OperationalProjectionUnavailableError(
                "Projected evidence lineage is unavailable."
            )
        return _evidence(rows[0])

    @staticmethod
    def _active_snapshot(
        cursor: Any,
        head: _ProjectionHead,
    ) -> tuple[dict[str, str], ProjectedEvidence | None]:
        cursor.execute(
            "SELECT aas.rule_id, aas.alert_event_id, er.domain, er.source_kind, "
            "er.schema_version, er.record_id, er.sha256, er.effective_at "
            "FROM operational_projection.active_alert_snapshot aas "
            "JOIN operational_projection.evidence_record er "
            "ON er.evidence_record_id = aas.evidence_record_id "
            "JOIN operational_projection.generation_evidence ge "
            "ON ge.evidence_record_id = er.evidence_record_id "
            "AND ge.generation_id = aas.generation_id "
            "WHERE aas.generation_id = %s ORDER BY aas.rule_id",
            (head.generation_id,),
        )
        rows = cursor.fetchall()
        if len(rows) != head.active_alert_snapshot_count:
            raise OperationalProjectionUnavailableError(
                "Projected active-alert state is incomplete."
            )
        active: dict[str, str] = {}
        evidence: ProjectedEvidence | None = None
        for row in rows:
            active[str(row[0])] = str(row[1])
            candidate = _evidence(row[2:])
            if evidence is not None and evidence != candidate:
                raise OperationalProjectionUnavailableError(
                    "Projected active-alert evidence is inconsistent."
                )
            evidence = candidate
        if evidence is None:
            cursor.execute(
                "SELECT er.domain, er.source_kind, er.schema_version, er.record_id, "
                "er.sha256, er.effective_at "
                "FROM operational_projection.evidence_record er "
                "JOIN operational_projection.generation_evidence ge "
                "ON ge.evidence_record_id = er.evidence_record_id "
                "WHERE ge.generation_id = %s "
                "AND er.source_kind = 'load_active_alerts'",
                (head.generation_id,),
            )
            evidence_rows = cursor.fetchall()
            if len(evidence_rows) > 1:
                raise OperationalProjectionUnavailableError(
                    "Projected active-alert evidence is inconsistent."
                )
            if evidence_rows:
                evidence = _evidence(evidence_rows[0])
        if evidence is not None and (
            evidence.domain != "alert"
            or evidence.source_kind != "load_active_alerts"
            or evidence.record_id != evidence.sha256
        ):
            raise OperationalProjectionUnavailableError(
                "Projected active-alert evidence is incompatible."
            )
        return active, evidence


@dataclass(frozen=True)
class UnavailableOperationalProjectionReader:
    """Fail-closed required-mode reader used for invalid local configuration."""

    def select_report(self, **_kwargs: Any) -> None:
        raise OperationalProjectionUnavailableError(
            "Required operational projection is unavailable."
        )

    def select_attempt(self, **_kwargs: Any) -> None:
        raise OperationalProjectionUnavailableError(
            "Required operational projection is unavailable."
        )

    def select_alerts(self, **_kwargs: Any) -> None:
        raise OperationalProjectionUnavailableError(
            "Required operational projection is unavailable."
        )


def _connect(dsn: str, budget: _DeadlineBudget) -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise OperationalProjectionUnavailableError(
            "PostgreSQL driver is unavailable."
        ) from exc
    try:
        remaining = budget.remaining()
        if remaining < 1.0:
            raise OperationalProjectionTimeoutError(
                "Projection query deadline expired."
            )
        connect_timeout = min(5, math.floor(remaining))
        milliseconds = _statement_timeout_milliseconds(remaining)
        connection = psycopg.connect(
            dsn,
            autocommit=True,
            application_name="wind_forecast_projection_reader",
            connect_timeout=connect_timeout,
            options=(
                f"-c statement_timeout={milliseconds} "
                f"-c lock_timeout={milliseconds} "
                "-c timezone=UTC "
                "-c search_path=operational_projection,pg_catalog"
            ),
        )
        try:
            budget.remaining()
        except OperationalProjectionTimeoutError:
            connection.close()
            raise
        return connection
    except OperationalProjectionTimeoutError:
        raise
    except Exception as exc:
        if _is_connection_timeout(exc, psycopg):
            raise OperationalProjectionTimeoutError(
                "Projection connection deadline expired."
            ) from exc
        raise OperationalProjectionUnavailableError(
            "Required operational projection is unavailable."
        ) from exc


def _bounded_timeout(value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise OperationalProjectionTimeoutError(
            "Projection query deadline expired."
        )
    return min(float(value), 5.0)


def _statement_timeout_milliseconds(remaining_seconds: float) -> int:
    milliseconds = min(5_000, math.floor(remaining_seconds * 1_000))
    if milliseconds < 1:
        raise OperationalProjectionTimeoutError(
            "Projection query deadline expired."
        )
    return milliseconds


def _is_statement_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or getattr(exc, "sqlstate", None) in {
        "55P03",
        "57014",
    }


def _is_connection_timeout(exc: BaseException, psycopg: Any) -> bool:
    timeout_types = (TimeoutError, psycopg.errors.ConnectionTimeout)
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, timeout_types):
            return True
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _entity_select(table: str, alias: str, columns: Sequence[str]) -> str:
    selected = ", ".join(f"{alias}.{column}" for column in columns)
    return (
        f"SELECT {selected}, er.domain, er.source_kind, er.schema_version, "
        "er.record_id, er.sha256, er.effective_at "
        f"FROM operational_projection.{table} {alias} "
        "JOIN operational_projection.evidence_record er "
        f"ON er.evidence_record_id = {alias}.evidence_record_id "
        "JOIN operational_projection.generation_evidence ge "
        "ON ge.evidence_record_id = er.evidence_record_id "
        "WHERE ge.generation_id = %s"
    )


def _projected_row(
    _cursor: Any,
    row: Sequence[Any] | None,
    columns: Sequence[str],
) -> ProjectedRow | None:
    if row is None:
        return None
    width = len(columns)
    return ProjectedRow(
        dict(zip(columns, row[:width], strict=True)),
        _evidence(row[width:]),
    )


def _projected_rows(
    cursor: Any,
    rows: Sequence[Sequence[Any]],
    columns: Sequence[str],
) -> tuple[ProjectedRow, ...]:
    return tuple(_projected_row(cursor, row, columns) for row in rows)  # type: ignore[misc]


def _evidence(values: Sequence[Any]) -> ProjectedEvidence:
    if len(values) != 6:
        raise OperationalProjectionUnavailableError(
            "Projected evidence identity is invalid."
        )
    return ProjectedEvidence(*(str(value) for value in values))


def _alert_chain_sql() -> str:
    selected = ", ".join(f"ae.{column}" for column in _ALERT_COLUMNS)
    recursive_columns = ", ".join(_ALERT_COLUMNS) + (
        ", evidence_domain, evidence_source_kind, evidence_schema_version, "
        "evidence_record_id, evidence_sha256, evidence_effective_at, causal_depth"
    )
    root_values = ", ".join(_ALERT_COLUMNS)
    child_values = ", ".join(f"child.{column}" for column in _ALERT_COLUMNS)
    return (
        "WITH RECURSIVE current_alerts AS MATERIALIZED ("
        f"SELECT {selected}, er.domain AS evidence_domain, "
        "er.source_kind AS evidence_source_kind, "
        "er.schema_version AS evidence_schema_version, "
        "er.record_id AS evidence_record_id, er.sha256 AS evidence_sha256, "
        "er.effective_at AS evidence_effective_at "
        "FROM operational_projection.alert_event ae "
        "JOIN operational_projection.evidence_record er "
        "ON er.evidence_record_id = ae.evidence_record_id "
        "JOIN operational_projection.generation_evidence ge "
        "ON ge.evidence_record_id = er.evidence_record_id "
        "WHERE ge.generation_id = %s), "
        f"chain ({recursive_columns}) AS ("
        f"SELECT {root_values}, evidence_domain, evidence_source_kind, "
        "evidence_schema_version, evidence_record_id, evidence_sha256, "
        "evidence_effective_at, 0 FROM current_alerts "
        "WHERE previous_alert_event_id IS NULL UNION ALL "
        f"SELECT {child_values}, child.evidence_domain, "
        "child.evidence_source_kind, child.evidence_schema_version, "
        "child.evidence_record_id, child.evidence_sha256, "
        "child.evidence_effective_at, parent.causal_depth + 1 "
        "FROM current_alerts child JOIN chain parent "
        "ON child.previous_alert_event_id = parent.alert_event_id "
        "AND child.rule_id = parent.rule_id)"
    )


def _alert_select_clause() -> str:
    columns = ", ".join(f"chain.{column}" for column in _ALERT_COLUMNS)
    return (
        f" SELECT {columns}, chain.evidence_domain, "
        "chain.evidence_source_kind, chain.evidence_schema_version, "
        "chain.evidence_record_id, chain.evidence_sha256, "
        "chain.evidence_effective_at FROM chain "
        "ORDER BY chain.through_date, chain.rule_id, chain.causal_depth, "
        "chain.alert_event_id"
    )


__all__ = [
    "OperationalProjectionReader",
    "OperationalProjectionReaderError",
    "OperationalProjectionTimeoutError",
    "OperationalProjectionUnavailableError",
    "ProjectedAlerts",
    "ProjectedEvidence",
    "ProjectedReport",
    "ProjectedRow",
    "UnavailableOperationalProjectionReader",
]
