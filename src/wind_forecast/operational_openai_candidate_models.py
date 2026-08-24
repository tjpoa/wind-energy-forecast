"""Strict additive contracts for the approved remote OpenAI candidate."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .operational_evaluation_models import EXPECTED_CASE_COUNT, EvaluationMetrics


OPENAI_RECEIPT_SCHEMA_VERSION = (
    "wind_forecast.operational_openai_candidate_evaluation_receipt.v1"
)
OPENAI_RECEIPT_ACCEPTANCE_STATE = (
    "candidate evaluated; Copilot disabled by default"
)
OPENAI_PROVIDER = "openai"
OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"
OPENAI_API = "responses"
OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"
REMOTE_EXECUTION_MODE = "remote_provider_evaluation"
SEALED_SYNTHETIC_EGRESS_SCOPE = "sealed_synthetic_dataset_only"
DIGEST_ONLY_RETENTION_POLICY = "digest_only_no_candidate_payload"
ENGLISH_LANGUAGE = "en"
OPENAI_PROMPT_VERSION = "operational_openai_candidate_prompt.v1"
OPENAI_SELECTOR_VERSION = "openai_responses_operational_query_selector.v1"
REMOTE_SELECTOR_TIMEOUT_SECONDS = 5.0
REMOTE_TOTAL_DEADLINE_SECONDS = 5.0
OPENAI_MAX_RESPONSE_BYTES = 64 * 1024
SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OpenAICandidateMetadata(StrictModel):
    """Fixed non-secret configuration for the approved remote candidate."""

    candidate_id: StrictStr = Field(min_length=1, max_length=128)
    provider: Literal[OPENAI_PROVIDER] = OPENAI_PROVIDER
    model: Literal[OPENAI_MODEL] = OPENAI_MODEL
    api: Literal[OPENAI_API] = OPENAI_API
    endpoint: Literal[OPENAI_ENDPOINT] = OPENAI_ENDPOINT
    execution_mode: Literal[REMOTE_EXECUTION_MODE] = REMOTE_EXECUTION_MODE
    egress_allowed: Literal[True] = True
    egress_scope: Literal[SEALED_SYNTHETIC_EGRESS_SCOPE] = (
        SEALED_SYNTHETIC_EGRESS_SCOPE
    )
    store: Literal[False] = False
    language: Literal[ENGLISH_LANGUAGE] = ENGLISH_LANGUAGE
    reasoning_effort: Literal["none"] = "none"
    tool_choice: Literal["auto"] = "auto"
    parallel_tool_calls: Literal[False] = False
    zero_retries: Literal[True] = True
    retention_policy: Literal[DIGEST_ONLY_RETENTION_POLICY] = (
        DIGEST_ONLY_RETENTION_POLICY
    )
    prompt_version: Literal[OPENAI_PROMPT_VERSION] = OPENAI_PROMPT_VERSION
    selector_version: Literal[OPENAI_SELECTOR_VERSION] = OPENAI_SELECTOR_VERSION
    selector_timeout_seconds: StrictFloat = Field(
        default=REMOTE_SELECTOR_TIMEOUT_SECONDS,
        ge=REMOTE_SELECTOR_TIMEOUT_SECONDS,
        le=REMOTE_SELECTOR_TIMEOUT_SECONDS,
    )
    total_deadline_seconds: StrictFloat = Field(
        default=REMOTE_TOTAL_DEADLINE_SECONDS,
        ge=REMOTE_TOTAL_DEADLINE_SECONDS,
        le=REMOTE_TOTAL_DEADLINE_SECONDS,
    )
    max_response_bytes: StrictInt = Field(
        default=OPENAI_MAX_RESPONSE_BYTES,
        ge=OPENAI_MAX_RESPONSE_BYTES,
        le=OPENAI_MAX_RESPONSE_BYTES,
    )

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        if not value.isprintable() or "/" in value or "\\" in value:
            raise ValueError("candidate ID must be bounded opaque text")
        return value

    @model_validator(mode="after")
    def validate_deadline_order(self) -> "OpenAICandidateMetadata":
        if self.selector_timeout_seconds > self.total_deadline_seconds:
            raise ValueError("selector timeout must not exceed total deadline")
        return self


class OpenAICandidateEvaluationReceipt(StrictModel):
    """Passed remote evaluation metadata without prompts or model payloads."""

    schema_version: Literal[OPENAI_RECEIPT_SCHEMA_VERSION] = (
        OPENAI_RECEIPT_SCHEMA_VERSION
    )
    acceptance_state: Literal[OPENAI_RECEIPT_ACCEPTANCE_STATE] = (
        OPENAI_RECEIPT_ACCEPTANCE_STATE
    )
    candidate_id: StrictStr = Field(min_length=1, max_length=128)
    provider: Literal[OPENAI_PROVIDER]
    model: Literal[OPENAI_MODEL]
    api: Literal[OPENAI_API]
    execution_mode: Literal[REMOTE_EXECUTION_MODE]
    egress_allowed: Literal[True]
    egress_scope: Literal[SEALED_SYNTHETIC_EGRESS_SCOPE]
    store: Literal[False]
    language: Literal[ENGLISH_LANGUAGE]
    reasoning_effort: Literal["none"]
    tool_choice: Literal["auto"]
    parallel_tool_calls: Literal[False]
    zero_retries: Literal[True]
    retention_policy: Literal[DIGEST_ONLY_RETENTION_POLICY]
    prompt_version: Literal[OPENAI_PROMPT_VERSION]
    selector_version: Literal[OPENAI_SELECTOR_VERSION]
    selector_timeout_seconds: StrictFloat = Field(
        ge=REMOTE_SELECTOR_TIMEOUT_SECONDS,
        le=REMOTE_SELECTOR_TIMEOUT_SECONDS,
    )
    total_deadline_seconds: StrictFloat = Field(
        ge=REMOTE_TOTAL_DEADLINE_SECONDS,
        le=REMOTE_TOTAL_DEADLINE_SECONDS,
    )
    max_response_bytes: StrictInt = Field(
        ge=OPENAI_MAX_RESPONSE_BYTES,
        le=OPENAI_MAX_RESPONSE_BYTES,
    )
    calls_completed: Literal[EXPECTED_CASE_COUNT]
    dataset_id: StrictStr = Field(min_length=1, max_length=128)
    dataset_version: StrictStr = Field(min_length=1, max_length=64)
    dataset_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    response_set_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    report_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    candidate_config_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    source_commit: StrictStr = Field(pattern=COMMIT_PATTERN)
    metrics: EvaluationMetrics
    evaluated_at_utc: datetime

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        if not value.isprintable() or "/" in value or "\\" in value:
            raise ValueError("candidate ID must be bounded opaque text")
        return value

    @field_validator("evaluated_at_utc")
    @classmethod
    def validate_receipt_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluated_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)


__all__ = [
    "DIGEST_ONLY_RETENTION_POLICY",
    "ENGLISH_LANGUAGE",
    "OPENAI_API",
    "OPENAI_ENDPOINT",
    "OPENAI_MAX_RESPONSE_BYTES",
    "OPENAI_MODEL",
    "OPENAI_PROMPT_VERSION",
    "OPENAI_PROVIDER",
    "OPENAI_RECEIPT_ACCEPTANCE_STATE",
    "OPENAI_RECEIPT_SCHEMA_VERSION",
    "OPENAI_SELECTOR_VERSION",
    "OpenAICandidateEvaluationReceipt",
    "OpenAICandidateMetadata",
    "REMOTE_EXECUTION_MODE",
    "REMOTE_SELECTOR_TIMEOUT_SECONDS",
    "REMOTE_TOTAL_DEADLINE_SECONDS",
    "SEALED_SYNTHETIC_EGRESS_SCOPE",
]
