"""Strict provider-neutral contracts for the operational Copilot core."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from .operational_query_models import (
    CONTRACT_VERSION,
    OperationalAnswer,
    OperationalFailure,
    Pagination,
    QueryKind,
    Selector,
)


TOOL_NAME = "operational_query"
LOCAL_OPERATOR_PRINCIPAL = "local-api-operator"
MAX_COPILOT_QUESTION_LENGTH = 1_000


class StrictModel(BaseModel):
    """Frozen, extra-forbid model used at the Copilot boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OperationalHttpRequest(StrictModel):
    """Five public fields accepted by both the HTTP and Copilot boundaries."""

    contract_version: Literal[CONTRACT_VERSION]
    query_kind: QueryKind
    selector: Selector
    window_days: Literal[30, 90] | None = None
    pagination: Pagination | None = None

    @field_validator("query_kind", mode="before")
    @classmethod
    def parse_query_kind(cls, value: Any) -> QueryKind:
        return value if isinstance(value, QueryKind) else QueryKind(value)


class CopilotRequest(StrictModel):
    """Boundary request supplied to a tool selector, without operational data."""

    question: StrictStr = Field(
        min_length=1,
        max_length=MAX_COPILOT_QUESTION_LENGTH,
    )
    requested_at_utc: datetime
    correlation_id: StrictStr = Field(min_length=1, max_length=128)
    deadline: datetime

    @field_validator("requested_at_utc", "deadline")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("question")
    @classmethod
    def require_non_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @model_validator(mode="after")
    def require_future_deadline(self) -> "CopilotRequest":
        if self.deadline <= self.requested_at_utc:
            raise ValueError("deadline must be after requested_at_utc")
        return self


class CopilotHttpRequest(StrictModel):
    """Small browser-facing request for the local Copilot."""

    question: StrictStr = Field(min_length=1, max_length=MAX_COPILOT_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def require_non_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class DocumentaryAnswer(StrictModel):
    """Documentary answer grounded in the closed, versioned corpus."""

    summary: StrictStr
    evidence: tuple["DocumentaryEvidence", ...]


class DocumentaryEvidence(StrictModel):
    chunk_id: StrictStr
    document_id: StrictStr
    title: StrictStr
    heading: StrictStr
    uri: StrictStr
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class ProviderFailure(StrictModel):
    code: StrictStr
    message: StrictStr


class CopilotOperationalResponse(StrictModel):
    route: Literal["operational"]
    mode: Literal["guided_local"]
    answer: OperationalAnswer
    limitations: tuple[StrictStr, ...] = ()
    failure: None = None


class CopilotDocumentaryResponse(StrictModel):
    route: Literal["documentary"]
    mode: Literal["rag_local", "rag_openai"]
    answer: DocumentaryAnswer
    limitations: tuple[StrictStr, ...] = ()
    failure: None = None
    provider_failure: ProviderFailure | None = None


class CopilotRefusedResponse(StrictModel):
    route: Literal["refused"]
    mode: Literal["guided_local"]
    answer: None = None
    limitations: tuple[StrictStr, ...] = ()
    failure: OperationalFailure


CopilotResponse = (
    CopilotOperationalResponse | CopilotDocumentaryResponse | CopilotRefusedResponse
)


class OperationalToolDefinition(StrictModel):
    """The only tool contract exposed to an injected selector."""

    name: Literal[TOOL_NAME] = TOOL_NAME
    description: StrictStr
    input_schema: dict[str, Any]


class OperationalToolSelection(StrictModel):
    """The complete selector output; it cannot carry an answer or evidence."""

    tool_name: StrictStr
    arguments: dict[str, Any]


def operational_query_tool() -> OperationalToolDefinition:
    """Return the immutable-shaped description of the sole allowed tool."""
    return OperationalToolDefinition(
        description=(
            "Answer one bounded operational question from verified local evidence."
        ),
        input_schema=OperationalHttpRequest.model_json_schema(),
    )


def allowed_operational_tools() -> tuple[OperationalToolDefinition, ...]:
    """Return the closed tool catalog supplied to a selector."""
    return (operational_query_tool(),)


__all__ = [
    "CopilotRequest",
    "CopilotHttpRequest",
    "CopilotOperationalResponse",
    "CopilotDocumentaryResponse",
    "CopilotRefusedResponse",
    "CopilotResponse",
    "DocumentaryAnswer",
    "DocumentaryEvidence",
    "ProviderFailure",
    "LOCAL_OPERATOR_PRINCIPAL",
    "MAX_COPILOT_QUESTION_LENGTH",
    "OperationalToolDefinition",
    "OperationalHttpRequest",
    "OperationalToolSelection",
    "TOOL_NAME",
    "allowed_operational_tools",
    "operational_query_tool",
]
