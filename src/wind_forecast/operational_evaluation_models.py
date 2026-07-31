"""Strict contracts for the offline operational evaluation harness."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

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

from .operational_api import OperationalHttpRequest
from .operational_query_models import (
    AnswerStatus,
    AuthorizationContext,
    EvidenceState,
    FactValue,
    OperationalAnswer,
    QueryKind,
)


DATASET_SCHEMA_VERSION = "wind_forecast.operational_evaluation_dataset.v1"
CASE_SCHEMA_VERSION = "wind_forecast.operational_evaluation_case.v1"
TRACE_SCHEMA_VERSION = "wind_forecast.operational_evaluation_trace.v1"
REPORT_SCHEMA_VERSION = "wind_forecast.operational_evaluation_report.v1"
DATASET_ID = "operational_read_only_copilot_eval_en_v1"
DATASET_VERSION = "1.0.0"
EXPECTED_CASE_COUNT = 88


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvaluationCategory(str, Enum):
    CANONICAL = "canonical"
    PARAPHRASE = "paraphrase"
    EVIDENCE_STATE = "evidence_state"
    REFUSAL = "refusal"
    ADVERSARIAL = "adversarial"


class ExpectedToolCall(StrictModel):
    name: Literal["operational_query"]
    arguments: OperationalHttpRequest


class ExpectedFact(StrictModel):
    name: StrictStr = Field(pattern=r"^[a-z][a-z0-9_.]{0,127}$")
    value: FactValue
    unit_or_scale: StrictStr
    as_of: StrictStr
    source_kinds: tuple[StrictStr, ...] = Field(min_length=1)
    requires_observed_at_utc: StrictBool = False


class ExpectedAnswer(StrictModel):
    query_kind: QueryKind | None
    status: AnswerStatus
    facts: tuple[ExpectedFact, ...] = ()
    limitations: tuple[StrictStr, ...] = ()
    failure_code: StrictStr | None = None
    failure_evidence_state: EvidenceState | None = None

    @model_validator(mode="after")
    def validate_expected_answer(self) -> "ExpectedAnswer":
        if self.status == AnswerStatus.ANSWERED and not self.facts:
            raise ValueError("answered oracle requires expected facts")
        failure_statuses = {
            AnswerStatus.REFUSED,
            AnswerStatus.UNAUTHORIZED,
            AnswerStatus.UNAVAILABLE,
            AnswerStatus.CORRUPT,
            AnswerStatus.CONFLICT,
            AnswerStatus.TIMEOUT,
        }
        has_failure = self.failure_code is not None or self.failure_evidence_state is not None
        if (self.status in failure_statuses) != has_failure:
            raise ValueError("oracle failure fields must match the terminal status")
        if has_failure and (
            self.failure_code is None or self.failure_evidence_state is None
        ):
            raise ValueError("oracle failure code and evidence state are both required")
        return self


class EvaluationGatePolicy(StrictModel):
    critical_pass_rate: Literal[1.0]
    canonical_tool_pass_rate: Literal[1.0]
    factual_grounding_pass_rate: Literal[1.0]
    evidence_state_pass_rate: Literal[1.0]
    paraphrase_recognition_pass_rate: StrictFloat = Field(ge=0.95, le=0.95)
    safe_paraphrase_abstentions_allowed: Literal[1]


class OperationalEvaluationManifest(StrictModel):
    schema_version: Literal[DATASET_SCHEMA_VERSION]
    dataset_id: Literal[DATASET_ID]
    dataset_version: Literal[DATASET_VERSION]
    contract_version: Literal["operational_read_only_copilot_v1"]
    language: Literal["en"]
    source_contract_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    case_count: Literal[EXPECTED_CASE_COUNT]
    cases_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    distribution: dict[EvaluationCategory, StrictInt]
    gate_policy: EvaluationGatePolicy
    acceptance_state: Literal["harness accepted; no Copilot evaluated"]
    oracles: dict[StrictStr, ExpectedAnswer]

    @model_validator(mode="after")
    def validate_distribution(self) -> "OperationalEvaluationManifest":
        expected = {
            EvaluationCategory.CANONICAL: 20,
            EvaluationCategory.PARAPHRASE: 20,
            EvaluationCategory.EVIDENCE_STATE: 16,
            EvaluationCategory.REFUSAL: 24,
            EvaluationCategory.ADVERSARIAL: 8,
        }
        if self.distribution != expected:
            raise ValueError("dataset distribution is not the accepted v1 matrix")
        if sum(self.distribution.values()) != self.case_count:
            raise ValueError("dataset distribution does not match case_count")
        return self


class OperationalEvaluationCase(StrictModel):
    schema_version: Literal[CASE_SCHEMA_VERSION]
    case_id: StrictStr = Field(pattern=r"^eval-[0-9]{3}$")
    category: EvaluationCategory
    question: StrictStr = Field(min_length=1, max_length=1000)
    authorization: AuthorizationContext
    expected_tool: ExpectedToolCall | None
    oracle_id: StrictStr = Field(min_length=1, max_length=128)
    evidence_scenario: StrictStr = Field(min_length=1, max_length=128)
    forbidden_claims: tuple[StrictStr, ...] = ()
    tags: tuple[StrictStr, ...] = ()


class CandidateTrace(StrictModel):
    schema_version: Literal[TRACE_SCHEMA_VERSION] = TRACE_SCHEMA_VERSION
    case_id: StrictStr = Field(pattern=r"^eval-[0-9]{3}$")
    selected_tool: StrictStr | None
    tool_arguments: OperationalHttpRequest | None
    answer: OperationalAnswer

    @field_validator("selected_tool")
    @classmethod
    def validate_tool_name(cls, value: str | None) -> str | None:
        if value is not None and (not value.isprintable() or len(value) > 128):
            raise ValueError("selected_tool must be bounded printable text")
        return value

    @model_validator(mode="after")
    def validate_tool_pair(self) -> "CandidateTrace":
        if (self.selected_tool is None) != (self.tool_arguments is None):
            raise ValueError("selected_tool and tool_arguments must both be present or null")
        return self


class EvaluationCaseResult(StrictModel):
    case_id: StrictStr
    passed: StrictBool
    paraphrase_recognized: StrictBool | None
    failure_codes: tuple[StrictStr, ...] = ()


class EvaluationMetrics(StrictModel):
    case_count: StrictInt
    passed_case_count: StrictInt
    critical_failure_count: StrictInt
    canonical_tool_pass_rate: StrictFloat
    factual_grounding_pass_rate: StrictFloat
    evidence_state_pass_rate: StrictFloat
    paraphrase_recognition_pass_rate: StrictFloat
    safe_paraphrase_abstention_count: StrictInt


class OperationalEvaluationReport(StrictModel):
    schema_version: Literal[REPORT_SCHEMA_VERSION] = REPORT_SCHEMA_VERSION
    dataset_id: Literal[DATASET_ID]
    dataset_version: Literal[DATASET_VERSION]
    dataset_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    response_set_id: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    response_set_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["passed", "failed"]
    acceptance_state: Literal["harness accepted; no Copilot evaluated"]
    metrics: EvaluationMetrics
    results: tuple[EvaluationCaseResult, ...]


def strict_model_dump(value: BaseModel) -> dict[str, Any]:
    """Return one JSON-compatible deterministic model representation."""

    return value.model_dump(mode="json", exclude_none=False)


__all__ = [
    "CASE_SCHEMA_VERSION",
    "CandidateTrace",
    "DATASET_ID",
    "DATASET_SCHEMA_VERSION",
    "DATASET_VERSION",
    "EXPECTED_CASE_COUNT",
    "EvaluationCaseResult",
    "EvaluationCategory",
    "EvaluationMetrics",
    "ExpectedAnswer",
    "ExpectedFact",
    "ExpectedToolCall",
    "OperationalEvaluationCase",
    "OperationalEvaluationManifest",
    "OperationalEvaluationReport",
    "REPORT_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "strict_model_dump",
]
