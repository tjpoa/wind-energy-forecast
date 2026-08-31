"""Fail-closed readiness receipt validation for local automation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal, Mapping


READINESS_SCHEMA = "wind_forecast.automation_readiness.v1"
READINESS_STATUSES = ("GO", "NO-GO")


class ReadinessError(ValueError):
    """Raised when an automation readiness receipt is invalid or insufficient."""


@dataclass(frozen=True)
class AutomationReadiness:
    environment_id: str
    status: Literal["GO", "NO-GO"]
    allowed_workflows: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    updated_at_utc: str

    def allows(self, workflow: str) -> bool:
        """Return whether the receipt explicitly permits one workflow."""
        return self.status == "GO" and workflow in self.allowed_workflows

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_workflows": list(self.allowed_workflows),
            "environment_id": self.environment_id,
            "evidence_refs": list(self.evidence_refs),
            "reason_codes": list(self.reason_codes),
            "schema_version": READINESS_SCHEMA,
            "status": self.status,
            "updated_at_utc": self.updated_at_utc,
        }


def load_automation_readiness(
    path: str | Path,
    *,
    environment_id: str,
    workflow: str | None = None,
) -> AutomationReadiness:
    """Load and validate a readiness receipt without side effects."""
    receipt_path = Path(path)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"Invalid readiness receipt: {receipt_path}.") from exc
    if not isinstance(payload, Mapping):
        raise ReadinessError("Readiness receipt must be a JSON object.")
    required = {
        "allowed_workflows",
        "environment_id",
        "evidence_refs",
        "reason_codes",
        "schema_version",
        "status",
        "updated_at_utc",
    }
    if set(payload) != required or payload.get("schema_version") != READINESS_SCHEMA:
        raise ReadinessError("Readiness receipt fields or schema are invalid.")
    if payload.get("environment_id") != environment_id:
        raise ReadinessError("Readiness receipt belongs to another environment.")
    status = payload.get("status")
    if status not in READINESS_STATUSES:
        raise ReadinessError("Readiness status must be GO or NO-GO.")
    allowed = _text_tuple(payload.get("allowed_workflows"), "allowed_workflows")
    reasons = _text_tuple(payload.get("reason_codes"), "reason_codes")
    evidence = _text_tuple(payload.get("evidence_refs"), "evidence_refs")
    if status == "NO-GO" and allowed:
        raise ReadinessError("A NO-GO receipt must not allow workflows.")
    if status == "GO" and (reasons or not evidence or not allowed):
        raise ReadinessError(
            "A GO receipt requires evidence, no reason codes, and an allowed workflow."
        )
    updated = payload.get("updated_at_utc")
    if not isinstance(updated, str):
        raise ReadinessError("updated_at_utc must be an ISO-8601 UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(updated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReadinessError("updated_at_utc must be an ISO-8601 UTC timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReadinessError("updated_at_utc must be an ISO-8601 UTC timestamp.")
    receipt = AutomationReadiness(
        environment_id=environment_id,
        status=status,
        allowed_workflows=allowed,
        reason_codes=reasons,
        evidence_refs=evidence,
        updated_at_utc=updated,
    )
    if workflow is not None and not receipt.allows(workflow):
        raise ReadinessError(
            f"Readiness receipt does not permit workflow {workflow!r}."
        )
    return receipt


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ReadinessError(f"{name} must be a list of non-empty strings.")
    return tuple(value)


__all__ = [
    "AutomationReadiness",
    "READINESS_SCHEMA",
    "ReadinessError",
    "load_automation_readiness",
]
