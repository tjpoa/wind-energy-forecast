"""Deterministic internal models for the operational PostgreSQL projection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping


CONTRACT_VERSION = "operational_postgres_projection_v1"
RELATIONAL_SCHEMA_VERSION = "operational_postgres_projection_schema_v1"
PROJECTOR_VERSION = "operational_postgres_projector_v1"


def canonical_json(value: Any) -> str:
    """Return the ASCII JSON representation used by every projection digest."""
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Hash one canonical JSON value."""
    return sha256(canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True, order=True)
class EvidenceIdentity:
    """Stable identity of one verified loader result."""

    domain: str
    source_kind: str
    schema_version: str
    record_id: str
    sha256: str


@dataclass(frozen=True)
class EvidenceRecord:
    """One minimized evidence row and its non-identity temporal context."""

    identity: EvidenceIdentity
    effective_at: str
    observed_at_utc: datetime | None = None

    def manifest_entry(self) -> dict[str, str]:
        """Return only deterministic fields admitted to the source-set digest."""
        return {
            "domain": self.identity.domain,
            "source_kind": self.identity.source_kind,
            "schema_version": self.identity.schema_version,
            "record_id": self.identity.record_id,
            "sha256": self.identity.sha256,
            "effective_at": self.effective_at,
        }


@dataclass(frozen=True)
class EvidenceLink:
    """Resolve one relational foreign-key column from an evidence identity."""

    column: str
    evidence: EvidenceIdentity


@dataclass(frozen=True)
class RelationalRow:
    """One normalized row with symbolic evidence foreign keys."""

    table: str
    values: tuple[tuple[str, Any], ...]
    evidence_links: tuple[EvidenceLink, ...] = ()

    @classmethod
    def create(
        cls,
        table: str,
        values: Mapping[str, Any],
        *,
        evidence_links: Mapping[str, EvidenceIdentity] | None = None,
    ) -> "RelationalRow":
        return cls(
            table=table,
            values=tuple(sorted(values.items())),
            evidence_links=tuple(
                sorted(
                    (
                        EvidenceLink(column, evidence)
                        for column, evidence in (evidence_links or {}).items()
                    ),
                    key=lambda item: item.column,
                )
            ),
        )

    def value_map(self) -> dict[str, Any]:
        return dict(self.values)

    def comparison_payload(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "values": self.value_map(),
            "evidence_links": {
                item.column: _identity_payload(item.evidence)
                for item in self.evidence_links
            },
        }


@dataclass(frozen=True, order=True)
class LineageRelation:
    """One deterministic relationship among verified evidence records."""

    edge_type: str
    source: EvidenceIdentity
    target: EvidenceIdentity
    position: int
    evidence: EvidenceIdentity

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "edge_type": self.edge_type,
            "source": _identity_payload(self.source),
            "target": _identity_payload(self.target),
            "position": self.position,
            "evidence": _identity_payload(self.evidence),
        }


@dataclass(frozen=True)
class GenerationManifest:
    """Canonical, timestamp-independent identity of one complete projection."""

    environment_id: str
    source_git_commit: str
    evidence: tuple[EvidenceRecord, ...]
    report_state_sha256: str
    active_alert_state_sha256: str
    relations: tuple[LineageRelation, ...]
    contract_version: str = CONTRACT_VERSION
    schema_version: str = RELATIONAL_SCHEMA_VERSION
    projector_version: str = PROJECTOR_VERSION

    @property
    def source_set_sha256(self) -> str:
        return canonical_sha256(
            [item.manifest_entry() for item in self.evidence]
        )

    @property
    def generation_id(self) -> str:
        return canonical_sha256(self.payload())

    def payload(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "contract_version": self.contract_version,
            "schema_version": self.schema_version,
            "projector_version": self.projector_version,
            "source_git_commit": self.source_git_commit,
            "source_set_sha256": self.source_set_sha256,
            "sources": [item.manifest_entry() for item in self.evidence],
            "report_state_sha256": self.report_state_sha256,
            "active_alert_state_sha256": self.active_alert_state_sha256,
            "relations": [item.manifest_entry() for item in self.relations],
        }


@dataclass(frozen=True)
class ProjectionSnapshot:
    """Complete normalized source snapshot ready for comparison or publication."""

    manifest: GenerationManifest
    rows: tuple[RelationalRow, ...]

    @property
    def generation_id(self) -> str:
        return self.manifest.generation_id

    def rows_for(self, table: str) -> tuple[RelationalRow, ...]:
        return tuple(row for row in self.rows if row.table == table)

    def counts(self) -> dict[str, int]:
        row_counts = {
            table: len(self.rows_for(table))
            for table in (
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
        }
        return {
            "evidence_record_count": len(self.manifest.evidence),
            "generation_evidence_count": len(self.manifest.evidence),
            **{f"{table}_count": count for table, count in row_counts.items()},
        }


def ordered_evidence(records: Iterable[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
    """Deduplicate evidence identities and return canonical source-set order."""
    by_identity: dict[EvidenceIdentity, EvidenceRecord] = {}
    for record in records:
        prior = by_identity.get(record.identity)
        if prior is not None and prior.effective_at != record.effective_at:
            raise ValueError("One evidence identity has conflicting effective times.")
        if prior is None:
            by_identity[record.identity] = record
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda item: (
                item.identity.source_kind,
                item.identity.schema_version,
                item.identity.record_id,
                item.identity.sha256,
                item.effective_at,
                item.identity.domain,
            ),
        )
    )


def ordered_rows(rows: Iterable[RelationalRow]) -> tuple[RelationalRow, ...]:
    return tuple(
        sorted(rows, key=lambda row: canonical_json(row.comparison_payload()))
    )


def ordered_relations(
    relations: Iterable[LineageRelation],
) -> tuple[LineageRelation, ...]:
    return tuple(sorted(set(relations)))


def _identity_payload(identity: EvidenceIdentity) -> dict[str, str]:
    return {
        "domain": identity.domain,
        "source_kind": identity.source_kind,
        "schema_version": identity.schema_version,
        "record_id": identity.record_id,
        "sha256": identity.sha256,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


__all__ = [
    "CONTRACT_VERSION",
    "EvidenceIdentity",
    "EvidenceLink",
    "EvidenceRecord",
    "GenerationManifest",
    "LineageRelation",
    "PROJECTOR_VERSION",
    "ProjectionSnapshot",
    "RELATIONAL_SCHEMA_VERSION",
    "RelationalRow",
    "canonical_json",
    "canonical_sha256",
    "ordered_evidence",
    "ordered_relations",
    "ordered_rows",
]
