"""Strict executable contract for read-only operational queries."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
import math
import re
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


CONTRACT_VERSION = "operational_read_only_copilot_v1"
OPERATIONAL_MODE = "retrospective_historical_batch_not_real_time"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REPORTING_RUN_PATTERN = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{12}$")
SAFE_OPAQUE_PATTERN = re.compile(r"^[^\x00-\x1f\x7f/\\]{1,128}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class QueryKind(str, Enum):
    OPERATIONAL_SUMMARY = "operational_summary"
    ACTIVE_DEPLOYMENT = "active_deployment"
    DATA_QUALITY = "data_quality"
    MONITORING_PERFORMANCE = "monitoring_performance"
    MONITORING_DRIFT = "monitoring_drift"
    MONITORING_ALERTS = "monitoring_alerts"
    ACTIVE_MODEL_METADATA = "active_model_metadata"
    REPORTING_RUN = "reporting_run"


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    EMPTY = "empty"
    NOT_FOUND = "not_found"
    REFUSED = "refused"
    UNAUTHORIZED = "unauthorized"
    UNAVAILABLE = "unavailable"
    CORRUPT = "corrupt"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"


class EvidenceState(str, Enum):
    EMPTY = "empty"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    CORRUPT = "corrupt"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"
    UNSUPPORTED = "unsupported"


class EvidenceDomain(str, Enum):
    DEPLOYMENT = "deployment"
    MONITORING_REPORT = "monitoring_report"
    ALERT = "alert"
    MODEL_ERA = "prediction_model_era"
    MODEL_BUNDLE = "model_bundle"
    CALIBRATION = "calibration_reference"
    REGISTRY = "verified_registry_binding"
    REPORTING_RUN = "reporting_run"


class LatestSelector(StrictModel):
    kind: Literal["latest"] = "latest"


class ExactIdSelector(StrictModel):
    kind: Literal["exact_id"] = "exact_id"
    id_type: Literal["report_id", "reporting_run_id", "alert_event_id"]
    identifier: StrictStr

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, value: str, info: Any) -> str:
        if not SAFE_OPAQUE_PATTERN.fullmatch(value) or value in {".", ".."}:
            raise ValueError("identifier is malformed")
        id_type = info.data.get("id_type")
        if id_type in {"report_id", "alert_event_id"}:
            if not SHA256_PATTERN.fullmatch(value):
                raise ValueError("content-addressed identifier is malformed")
        elif id_type == "reporting_run_id":
            if not REPORTING_RUN_PATTERN.fullmatch(value):
                raise ValueError("reporting-run identifier is malformed")
        return value


class DateIntervalSelector(StrictModel):
    kind: Literal["date_interval"] = "date_interval"
    start_date: date
    end_date: date

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_iso_date(cls, value: Any) -> date:
        if isinstance(value, datetime):
            raise ValueError("calendar date must not contain a time")
        if isinstance(value, date):
            return value
        if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", value
        ):
            raise ValueError("calendar date must use YYYY-MM-DD")
        return date.fromisoformat(value)

    @model_validator(mode="after")
    def validate_interval(self) -> "DateIntervalSelector":
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        return self


Selector = Annotated[
    Union[LatestSelector, ExactIdSelector, DateIntervalSelector],
    Field(discriminator="kind"),
]


class Pagination(StrictModel):
    limit: StrictInt = Field(default=50, ge=1, le=200)
    offset: StrictInt = Field(default=0, ge=0)


class OperationalQuery(StrictModel):
    contract_version: Literal[CONTRACT_VERSION]
    query_kind: QueryKind
    selector: Selector
    window_days: Literal[30, 90] | None = None
    pagination: Pagination | None = None
    requested_at_utc: datetime
    correlation_id: StrictStr = Field(min_length=1, max_length=128)
    deadline: datetime

    @field_validator("query_kind", mode="before")
    @classmethod
    def parse_query_kind(cls, value: Any) -> QueryKind:
        return value if isinstance(value, QueryKind) else QueryKind(value)

    @field_validator("requested_at_utc", "deadline", mode="before")
    @classmethod
    def require_utc(cls, value: Any) -> datetime:
        if isinstance(value, str):
            if not re.fullmatch(
                (
                    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
                    r"(?:\.\d{1,6})?(?:Z|\+00:00)"
                ),
                value,
            ):
                raise ValueError("timestamp string must use ISO UTC")
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("timestamp string is invalid") from exc
        if not isinstance(value, datetime):
            raise ValueError("timestamp must be a datetime or ISO UTC string")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("timestamp must use the UTC offset")
        return value.astimezone(timezone.utc)

    @field_validator("correlation_id")
    @classmethod
    def validate_correlation_id(cls, value: str) -> str:
        if not SAFE_OPAQUE_PATTERN.fullmatch(value):
            raise ValueError("correlation_id must be printable opaque text")
        return value

    @model_validator(mode="after")
    def validate_query_contract(self) -> "OperationalQuery":
        latest_only = {
            QueryKind.OPERATIONAL_SUMMARY,
            QueryKind.ACTIVE_DEPLOYMENT,
            QueryKind.ACTIVE_MODEL_METADATA,
        }
        report_queries = {
            QueryKind.DATA_QUALITY,
            QueryKind.MONITORING_PERFORMANCE,
            QueryKind.MONITORING_DRIFT,
        }
        if self.deadline <= self.requested_at_utc:
            raise ValueError("deadline must be after requested_at_utc")
        if self.query_kind in latest_only and not isinstance(
            self.selector, LatestSelector
        ):
            raise ValueError("query_kind requires latest selector")
        if self.query_kind in report_queries:
            if isinstance(self.selector, DateIntervalSelector):
                raise ValueError("query_kind does not accept a date interval")
            if isinstance(self.selector, ExactIdSelector):
                allowed = (
                    {"report_id", "reporting_run_id"}
                    if self.query_kind == QueryKind.DATA_QUALITY
                    else {"report_id"}
                )
                if self.selector.id_type not in allowed:
                    raise ValueError("query_kind does not accept this identifier")
        if self.query_kind == QueryKind.MONITORING_ALERTS:
            if isinstance(self.selector, ExactIdSelector) and (
                self.selector.id_type != "alert_event_id"
            ):
                raise ValueError("monitoring_alerts requires an alert-event ID")
        if self.query_kind == QueryKind.REPORTING_RUN:
            if not isinstance(self.selector, ExactIdSelector) or (
                self.selector.id_type not in {"report_id", "reporting_run_id"}
            ):
                raise ValueError("reporting_run requires an exact run or report ID")
        window_required = self.query_kind in {
            QueryKind.MONITORING_PERFORMANCE,
            QueryKind.MONITORING_DRIFT,
        }
        if window_required != (self.window_days is not None):
            raise ValueError("window_days is present only for windowed queries")
        if self.pagination is not None and (
            self.query_kind != QueryKind.MONITORING_ALERTS
            or isinstance(self.selector, ExactIdSelector)
        ):
            raise ValueError("pagination is accepted only for alert collections")
        return self


FactScalar = StrictStr | StrictInt | StrictFloat | StrictBool | None
FactValue = FactScalar | list[FactScalar] | dict[StrictStr, FactScalar]


class GroundedFact(StrictModel):
    fact_id: StrictStr = Field(pattern=r"^f[1-9][0-9]*$")
    name: StrictStr = Field(pattern=r"^[a-z][a-z0-9_.]{0,127}$")
    value: FactValue
    unit_or_scale: StrictStr
    as_of: StrictStr
    evidence_ids: tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def finite_numbers(cls, value: FactValue) -> FactValue:
        def check(item: Any) -> None:
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("fact numbers must be finite")
            if isinstance(item, list):
                for member in item:
                    check(member)
            if isinstance(item, dict):
                for member in item.values():
                    check(member)

        check(value)
        return value


class EvidenceCitation(StrictModel):
    evidence_id: StrictStr = Field(pattern=r"^e[1-9][0-9]*$")
    domain: EvidenceDomain
    source_kind: StrictStr
    schema_version: StrictStr
    record_id: StrictStr
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    effective_at: StrictStr
    observed_at_utc: datetime | None = None

    @field_validator("observed_at_utc")
    @classmethod
    def observed_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("observed_at_utc must be timezone-aware")
        return None if value is None else value.astimezone(timezone.utc)


class OperationalFailure(StrictModel):
    code: StrictStr
    message: StrictStr
    retryable: StrictBool
    evidence_state: EvidenceState


class OperationalAnswer(StrictModel):
    contract_version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    query_kind: QueryKind | None
    status: AnswerStatus
    mode: Literal[OPERATIONAL_MODE] = OPERATIONAL_MODE
    summary: StrictStr | None
    facts: tuple[GroundedFact, ...] = ()
    evidence: tuple[EvidenceCitation, ...] = ()
    limitations: tuple[StrictStr, ...] = ()
    failure: OperationalFailure | None
    served_at_utc: datetime
    correlation_id: StrictStr = Field(min_length=1, max_length=128)

    @field_validator("served_at_utc")
    @classmethod
    def served_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("served_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("correlation_id")
    @classmethod
    def answer_correlation_is_safe(cls, value: str) -> str:
        if not SAFE_OPAQUE_PATTERN.fullmatch(value):
            raise ValueError("correlation_id must be printable opaque text")
        return value

    @model_validator(mode="after")
    def validate_answer_integrity(self) -> "OperationalAnswer":
        failure_states = {
            AnswerStatus.REFUSED,
            AnswerStatus.UNAUTHORIZED,
            AnswerStatus.UNAVAILABLE,
            AnswerStatus.CORRUPT,
            AnswerStatus.CONFLICT,
            AnswerStatus.TIMEOUT,
        }
        if (self.status in failure_states) != (self.failure is not None):
            raise ValueError("failure must match terminal failure status")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        referenced = {
            evidence_id
            for fact in self.facts
            for evidence_id in fact.evidence_ids
        }
        if not referenced.issubset(evidence_ids):
            raise ValueError("facts reference missing evidence")
        if set(evidence_ids) != referenced:
            raise ValueError("evidence citations must be referenced")
        if self.summary is not None:
            for evidence_id in referenced:
                if f"[{evidence_id}]" not in self.summary:
                    raise ValueError("summary must cite every returned fact")
        if self.status in {AnswerStatus.REFUSED, AnswerStatus.UNAUTHORIZED} and (
            self.facts or self.evidence or self.summary is not None
        ):
            raise ValueError("refused and unauthorized answers contain no facts")
        return self


class AuthorizationContext(StrictModel):
    principal: StrictStr
    trusted_local: StrictBool


__all__ = [
    "AnswerStatus",
    "AuthorizationContext",
    "CONTRACT_VERSION",
    "DateIntervalSelector",
    "EvidenceCitation",
    "EvidenceDomain",
    "EvidenceState",
    "ExactIdSelector",
    "GroundedFact",
    "LatestSelector",
    "OPERATIONAL_MODE",
    "OperationalAnswer",
    "OperationalFailure",
    "OperationalQuery",
    "Pagination",
    "QueryKind",
]
