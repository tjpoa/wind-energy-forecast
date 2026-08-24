"""Strict additive contracts for the approved remote Gemini candidate."""

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
)

from .operational_evaluation_models import EXPECTED_CASE_COUNT, EvaluationMetrics

GEMINI_RECEIPT_SCHEMA_VERSION = (
    "wind_forecast.operational_gemini_candidate_evaluation_receipt.v1"
)
GEMINI_RECEIPT_ACCEPTANCE_STATE = "candidate evaluated; Copilot disabled by default"
GEMINI_PROVIDER = "google"
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_API = "interactions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
REMOTE_EXECUTION_MODE = "remote_provider_evaluation"
SEALED_SYNTHETIC_EGRESS_SCOPE = "sealed_synthetic_dataset_only"
DIGEST_ONLY_RETENTION_POLICY = "digest_only_no_candidate_payload"
ENGLISH_LANGUAGE = "en"
GEMINI_PROMPT_VERSION = "operational_gemini_candidate_prompt.v1"
GEMINI_SELECTOR_VERSION = "gemini_interactions_operational_query_selector.v1"
REMOTE_SELECTOR_TIMEOUT_SECONDS = 5.0
REMOTE_TOTAL_DEADLINE_SECONDS = 5.0
GEMINI_MAX_RESPONSE_BYTES = 64 * 1024
SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GeminiCandidateMetadata(StrictModel):
    candidate_id: StrictStr = Field(min_length=1, max_length=128)
    provider: Literal[GEMINI_PROVIDER] = GEMINI_PROVIDER
    model: Literal[GEMINI_MODEL] = GEMINI_MODEL
    api: Literal[GEMINI_API] = GEMINI_API
    endpoint: Literal[GEMINI_ENDPOINT] = GEMINI_ENDPOINT
    execution_mode: Literal[REMOTE_EXECUTION_MODE] = REMOTE_EXECUTION_MODE
    egress_allowed: Literal[True] = True
    egress_scope: Literal[SEALED_SYNTHETIC_EGRESS_SCOPE] = SEALED_SYNTHETIC_EGRESS_SCOPE
    store: Literal[False] = False
    language: Literal[ENGLISH_LANGUAGE] = ENGLISH_LANGUAGE
    zero_retries: Literal[True] = True
    retention_policy: Literal[DIGEST_ONLY_RETENTION_POLICY] = (
        DIGEST_ONLY_RETENTION_POLICY
    )
    prompt_version: Literal[GEMINI_PROMPT_VERSION] = GEMINI_PROMPT_VERSION
    selector_version: Literal[GEMINI_SELECTOR_VERSION] = GEMINI_SELECTOR_VERSION
    selector_timeout_seconds: StrictFloat = Field(default=5.0, ge=5.0, le=5.0)
    total_deadline_seconds: StrictFloat = Field(default=5.0, ge=5.0, le=5.0)
    max_response_bytes: StrictInt = Field(
        default=GEMINI_MAX_RESPONSE_BYTES,
        ge=GEMINI_MAX_RESPONSE_BYTES,
        le=GEMINI_MAX_RESPONSE_BYTES,
    )

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        if not value.isprintable() or "/" in value or "\\" in value:
            raise ValueError("candidate ID must be bounded opaque text")
        return value


class GeminiCandidateEvaluationReceipt(StrictModel):
    schema_version: Literal[GEMINI_RECEIPT_SCHEMA_VERSION] = (
        GEMINI_RECEIPT_SCHEMA_VERSION
    )
    acceptance_state: Literal[GEMINI_RECEIPT_ACCEPTANCE_STATE] = (
        GEMINI_RECEIPT_ACCEPTANCE_STATE
    )
    candidate_id: StrictStr = Field(min_length=1, max_length=128)
    provider: Literal[GEMINI_PROVIDER]
    model: Literal[GEMINI_MODEL]
    api: Literal[GEMINI_API]
    execution_mode: Literal[REMOTE_EXECUTION_MODE]
    egress_allowed: Literal[True]
    egress_scope: Literal[SEALED_SYNTHETIC_EGRESS_SCOPE]
    store: Literal[False]
    language: Literal[ENGLISH_LANGUAGE]
    zero_retries: Literal[True]
    retention_policy: Literal[DIGEST_ONLY_RETENTION_POLICY]
    prompt_version: Literal[GEMINI_PROMPT_VERSION]
    selector_version: Literal[GEMINI_SELECTOR_VERSION]
    selector_timeout_seconds: StrictFloat = Field(ge=5.0, le=5.0)
    total_deadline_seconds: StrictFloat = Field(ge=5.0, le=5.0)
    max_response_bytes: StrictInt = Field(
        ge=GEMINI_MAX_RESPONSE_BYTES, le=GEMINI_MAX_RESPONSE_BYTES
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
