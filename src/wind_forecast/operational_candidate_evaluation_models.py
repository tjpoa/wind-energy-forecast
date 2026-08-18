"""Strict additive contracts for offline operational-candidate evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictStr,
    field_validator,
    model_validator,
)

from .operational_evaluation_models import EvaluationMetrics


RECEIPT_SCHEMA_VERSION = "wind_forecast.operational_candidate_evaluation_receipt.v1"
RECEIPT_ACCEPTANCE_STATE = "candidate evaluated; Copilot disabled by default"
OFFLINE_EXECUTION_MODE = "offline_injected"
ENGLISH_LANGUAGE = "en"
DIGEST_ONLY_RETENTION_POLICY = "digest_only_no_candidate_payload"
DEFAULT_SELECTOR_TIMEOUT_SECONDS = 1.0
DEFAULT_TOTAL_DEADLINE_SECONDS = 5.0
SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CandidateMetadata(StrictModel):
    """Non-secret identity and execution policy for one injected candidate."""

    candidate_id: StrictStr = Field(min_length=1, max_length=128)
    provider: StrictStr = Field(min_length=1, max_length=128)
    model: StrictStr = Field(min_length=1, max_length=128)
    execution_mode: Literal[OFFLINE_EXECUTION_MODE] = OFFLINE_EXECUTION_MODE
    language: Literal[ENGLISH_LANGUAGE] = ENGLISH_LANGUAGE
    egress_allowed: Literal[False] = False
    retention_policy: Literal[DIGEST_ONLY_RETENTION_POLICY] = (
        DIGEST_ONLY_RETENTION_POLICY
    )
    selector_timeout_seconds: StrictFloat = Field(
        default=DEFAULT_SELECTOR_TIMEOUT_SECONDS,
        gt=0.0,
        le=DEFAULT_SELECTOR_TIMEOUT_SECONDS,
    )
    total_deadline_seconds: StrictFloat = Field(
        default=DEFAULT_TOTAL_DEADLINE_SECONDS,
        gt=0.0,
        le=DEFAULT_TOTAL_DEADLINE_SECONDS,
    )

    @field_validator("candidate_id", "provider", "model")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if not value.isprintable() or "/" in value or "\\" in value:
            raise ValueError("candidate metadata labels must be bounded opaque text")
        return value

    @model_validator(mode="after")
    def validate_deadline_order(self) -> "CandidateMetadata":
        if self.selector_timeout_seconds > self.total_deadline_seconds:
            raise ValueError("selector timeout must not exceed total deadline")
        return self


class CandidateEvaluationReceipt(StrictModel):
    """A passed-evaluation receipt that contains no prompt or answer payloads."""

    schema_version: Literal[RECEIPT_SCHEMA_VERSION] = RECEIPT_SCHEMA_VERSION
    acceptance_state: Literal[RECEIPT_ACCEPTANCE_STATE] = RECEIPT_ACCEPTANCE_STATE
    candidate_id: StrictStr = Field(min_length=1, max_length=128)
    provider: StrictStr = Field(min_length=1, max_length=128)
    model: StrictStr = Field(min_length=1, max_length=128)
    execution_mode: Literal[OFFLINE_EXECUTION_MODE] = OFFLINE_EXECUTION_MODE
    language: Literal[ENGLISH_LANGUAGE] = ENGLISH_LANGUAGE
    egress_allowed: Literal[False] = False
    retention_policy: Literal[DIGEST_ONLY_RETENTION_POLICY] = (
        DIGEST_ONLY_RETENTION_POLICY
    )
    selector_timeout_seconds: StrictFloat = Field(gt=0.0, le=1.0)
    total_deadline_seconds: StrictFloat = Field(gt=0.0, le=5.0)
    dataset_id: StrictStr = Field(min_length=1, max_length=128)
    dataset_version: StrictStr = Field(min_length=1, max_length=64)
    dataset_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    response_set_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    report_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    candidate_config_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    source_commit: StrictStr = Field(pattern=COMMIT_PATTERN)
    metrics: EvaluationMetrics
    evaluated_at_utc: datetime

    @field_validator("candidate_id", "provider", "model")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if not value.isprintable() or "/" in value or "\\" in value:
            raise ValueError("candidate receipt labels must be bounded opaque text")
        return value

    @field_validator("evaluated_at_utc")
    @classmethod
    def validate_receipt_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluated_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_deadline_order(self) -> "CandidateEvaluationReceipt":
        if self.selector_timeout_seconds > self.total_deadline_seconds:
            raise ValueError("selector timeout must not exceed total deadline")
        return self


__all__ = [
    "CandidateEvaluationReceipt",
    "CandidateMetadata",
    "DEFAULT_SELECTOR_TIMEOUT_SECONDS",
    "DEFAULT_TOTAL_DEADLINE_SECONDS",
    "DIGEST_ONLY_RETENTION_POLICY",
    "ENGLISH_LANGUAGE",
    "OFFLINE_EXECUTION_MODE",
    "RECEIPT_ACCEPTANCE_STATE",
    "RECEIPT_SCHEMA_VERSION",
]
