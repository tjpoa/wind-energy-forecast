"""Offline, injected-candidate evaluation over the sealed Copilot dataset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from .operational_copilot_models import (
    OperationalHttpRequest,
    OperationalToolDefinition,
    OperationalToolSelection,
    TOOL_NAME,
    allowed_operational_tools,
)
from .operational_candidate_evaluation_models import (
    CandidateEvaluationReceipt,
    CandidateMetadata,
)
from .operational_evaluation import (
    OperationalEvaluationDataset,
    evaluate_candidate_results,
    sanitized_report_json,
)
from .operational_evaluation_models import (
    CandidateTrace,
    EvaluationCategory,
    OperationalEvaluationReport,
    strict_model_dump,
)
from .operational_query_models import (
    AnswerStatus,
    AuthorizationContext,
    EvidenceCitation,
    EvidenceState,
    GroundedFact,
    OperationalAnswer,
    OperationalFailure,
    OperationalQuery,
)


class CandidateEvaluationInputError(ValueError):
    """Raised when an evaluation input or receipt is invalid."""


class CandidateEvaluationInfrastructureError(RuntimeError):
    """Raised for a sanitized fatal infrastructure failure."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CandidateInput(StrictModel):
    """The complete candidate-visible input; no oracle fields are present."""

    question: StrictStr = Field(min_length=1, max_length=1_000)
    authorization: AuthorizationContext
    tools: tuple[OperationalToolDefinition, ...] = Field(
        min_length=1,
        max_length=1,
    )


class CandidateSelector(Protocol):
    """Select a tool from the candidate-visible input exactly once per case."""

    def select(self, request: CandidateInput) -> object:
        """Return a tool selection, or ``None`` for a safe abstention."""


class CandidateTimingPolicy(Protocol):
    """Minimal timing contract shared by offline and remote candidates."""

    selector_timeout_seconds: float
    total_deadline_seconds: float


_INVALID_SELECTION = object()


@dataclass(frozen=True)
class CandidateEvaluationRun:
    """Normalized traces and the report produced by the existing harness."""

    traces: tuple[CandidateTrace, ...]
    report: OperationalEvaluationReport
    response_set_sha256: str


def _canonical_text(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _answer_from_oracle(
    dataset: OperationalEvaluationDataset,
    case_id: str,
    *,
    served_at_utc: datetime,
) -> OperationalAnswer:
    case = next(item for item in dataset.cases if item.case_id == case_id)
    oracle = dataset.manifest.oracles[case.oracle_id]
    expected_citations = (
        dataset.manifest.citation_sets[oracle.citation_set_id]
        if oracle.citation_set_id is not None
        else ()
    )
    citations = tuple(
        EvidenceCitation(
            evidence_id=f"e{index}",
            **citation.model_dump(mode="python"),
        )
        for index, citation in enumerate(expected_citations, 1)
    )
    evidence_id_by_source = {
        citation.source_kind: citation.evidence_id for citation in citations
    }
    facts = tuple(
        GroundedFact(
            fact_id=f"f{index}",
            name=fact.name,
            value=fact.value,
            unit_or_scale=fact.unit_or_scale,
            as_of=fact.as_of,
            evidence_ids=tuple(
                evidence_id_by_source[source_kind]
                for source_kind in fact.source_kinds
            ),
        )
        for index, fact in enumerate(oracle.facts, 1)
    )
    summary = None
    if facts:
        summary = " ".join(
            f"{fact.name}={_canonical_text(fact.value)} "
            f"{''.join(f'[{item}]' for item in fact.evidence_ids)}."
            for fact in facts
        )
    failure = None
    if oracle.failure_code is not None:
        failure = OperationalFailure(
            code=oracle.failure_code,
            message=oracle.failure_message,
            retryable=oracle.failure_retryable,
            evidence_state=oracle.failure_evidence_state,
        )
    return OperationalAnswer(
        query_kind=oracle.query_kind,
        status=oracle.status,
        summary=summary,
        facts=facts,
        evidence=citations,
        limitations=oracle.limitations,
        failure=failure,
        served_at_utc=served_at_utc,
        correlation_id=case.case_id,
    )


def _refusal_answer(
    *,
    code: str,
    message: str,
    served_at_utc: datetime,
    correlation_id: str,
) -> OperationalAnswer:
    return OperationalAnswer(
        query_kind=None,
        status=AnswerStatus.REFUSED,
        summary=None,
        facts=(),
        evidence=(),
        limitations=(),
        failure=OperationalFailure(
            code=code,
            message=message,
            retryable=False,
            evidence_state=EvidenceState.UNSUPPORTED,
        ),
        served_at_utc=served_at_utc,
        correlation_id=correlation_id,
    )


def _safe_paraphrase_abstention(*, served_at_utc: datetime, correlation_id: str) -> OperationalAnswer:
    return _refusal_answer(
        code="unsupported_query_kind",
        message="The requested operational question is not supported.",
        served_at_utc=served_at_utc,
        correlation_id=correlation_id,
    )


def _invalid_candidate_output(*, served_at_utc: datetime, correlation_id: str) -> OperationalAnswer:
    return _refusal_answer(
        code="candidate_output_invalid",
        message="The candidate selection was invalid.",
        served_at_utc=served_at_utc,
        correlation_id=correlation_id,
    )


def _selection(value: object) -> OperationalToolSelection | None | object:
    if value is None:
        return None
    if isinstance(value, OperationalToolSelection):
        return value
    try:
        return OperationalToolSelection.model_validate(value, strict=True)
    except (TypeError, ValueError):
        return _INVALID_SELECTION


def _public_request(value: OperationalToolSelection) -> OperationalHttpRequest | None:
    try:
        return OperationalHttpRequest.model_validate(value.arguments, strict=True)
    except (TypeError, ValueError):
        return None


def _operational_query(
    request: OperationalHttpRequest,
    *,
    requested_at_utc: datetime,
    correlation_id: str,
    deadline: datetime,
) -> OperationalQuery | None:
    try:
        return OperationalQuery(
            **request.model_dump(mode="python"),
            requested_at_utc=requested_at_utc,
            correlation_id=correlation_id,
            deadline=deadline,
        )
    except (TypeError, ValueError):
        return None


def _trace_fields(
    selection: OperationalToolSelection | None,
    request: OperationalHttpRequest | None,
) -> tuple[str | None, OperationalHttpRequest | None]:
    if selection is None or request is None:
        return None, None
    tool_name = selection.tool_name
    if not tool_name.isprintable() or len(tool_name) > 128:
        return None, None
    return tool_name, request


def serialize_candidate_traces(traces: tuple[CandidateTrace, ...]) -> bytes:
    """Serialize normalized traces for digesting or caller-controlled storage."""

    return b"".join(
        (
            json.dumps(
                strict_model_dump(trace),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            + b"\n"
        )
        for trace in traces
    )


def run_candidate_evaluation(
    dataset: OperationalEvaluationDataset,
    candidate: CandidateSelector,
    metadata: CandidateTimingPolicy,
    *,
    evaluated_at_utc: datetime | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> CandidateEvaluationRun:
    """Run one injected selector without exposing oracle or evidence fields."""

    evaluated_at = evaluated_at_utc or datetime.now(timezone.utc)
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise CandidateEvaluationInputError("evaluated_at_utc must be timezone-aware")
    evaluated_at = evaluated_at.astimezone(timezone.utc)
    selector_timeout_seconds = metadata.selector_timeout_seconds
    total_deadline_seconds = metadata.total_deadline_seconds
    if (
        isinstance(selector_timeout_seconds, bool)
        or isinstance(total_deadline_seconds, bool)
        or not isinstance(selector_timeout_seconds, (int, float))
        or not isinstance(total_deadline_seconds, (int, float))
        or not math.isfinite(selector_timeout_seconds)
        or not math.isfinite(total_deadline_seconds)
        or selector_timeout_seconds <= 0.0
        or total_deadline_seconds <= 0.0
        or selector_timeout_seconds > total_deadline_seconds
    ):
        raise CandidateEvaluationInputError("candidate timing policy is invalid")
    tools = allowed_operational_tools()
    traces: list[CandidateTrace] = []
    for case in dataset.cases:
        candidate_input = CandidateInput(
            question=case.question,
            authorization=case.authorization,
            tools=tools,
        )
        started = monotonic_clock()
        selection_value: object = None
        selection_failed = False
        try:
            selection_value = candidate.select(candidate_input)
        except CandidateEvaluationInfrastructureError:
            raise
        except Exception:
            selection_failed = True
        elapsed = monotonic_clock() - started
        selection_timed_out = (
            elapsed > selector_timeout_seconds
            or elapsed > total_deadline_seconds
        )
        selection: OperationalToolSelection | None | object
        if selection_failed or selection_timed_out:
            selection = _INVALID_SELECTION
        else:
            selection = _selection(selection_value)
        selection_is_valid_abstention = selection is None
        selection_is_valid = isinstance(selection, OperationalToolSelection)
        public_request = _public_request(selection) if selection_is_valid else None
        selected_tool, trace_arguments = _trace_fields(
            selection if selection_is_valid else None,
            public_request,
        )
        query = None
        if selection is not None and public_request is not None:
            query = _operational_query(
                public_request,
                requested_at_utc=evaluated_at,
                correlation_id=case.case_id,
                deadline=evaluated_at
                + timedelta(seconds=total_deadline_seconds),
            )
        if selection is _INVALID_SELECTION:
            answer = _invalid_candidate_output(
                served_at_utc=evaluated_at,
                correlation_id=case.case_id,
            )
        elif selection_is_valid_abstention:
            if case.expected_tool is None:
                answer = _answer_from_oracle(
                    dataset,
                    case.case_id,
                    served_at_utc=evaluated_at,
                )
            elif case.category == EvaluationCategory.PARAPHRASE:
                answer = _safe_paraphrase_abstention(
                    served_at_utc=evaluated_at,
                    correlation_id=case.case_id,
                )
            else:
                answer = _invalid_candidate_output(
                    served_at_utc=evaluated_at,
                    correlation_id=case.case_id,
                )
        elif public_request is None or query is None or selection.tool_name != TOOL_NAME:
            answer = _invalid_candidate_output(
                served_at_utc=evaluated_at,
                correlation_id=case.case_id,
            )
        else:
            answer = _answer_from_oracle(
                dataset,
                case.case_id,
                served_at_utc=evaluated_at,
            )
        traces.append(
            CandidateTrace(
                case_id=case.case_id,
                selected_tool=selected_tool,
                tool_arguments=trace_arguments,
                answer=answer,
            )
        )
    response_bytes = serialize_candidate_traces(tuple(traces))
    response_set_sha256 = hashlib.sha256(response_bytes).hexdigest()
    report = evaluate_candidate_results(
        dataset,
        traces,
        response_set_sha256=response_set_sha256,
    )
    return CandidateEvaluationRun(
        traces=tuple(traces),
        report=report,
        response_set_sha256=response_set_sha256,
    )


def build_candidate_evaluation_receipt(
    run: CandidateEvaluationRun,
    metadata: CandidateMetadata,
    *,
    source_commit: str,
    evaluated_at_utc: datetime,
) -> CandidateEvaluationReceipt:
    """Build a receipt only after every evaluation gate has passed."""

    if run.report.status != "passed":
        raise CandidateEvaluationInputError(
            "a receipt is issued only for a passed candidate evaluation"
        )
    config_bytes = json.dumps(
        strict_model_dump(metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    report_sha256 = hashlib.sha256(
        sanitized_report_json(run.report).encode("utf-8")
    ).hexdigest()
    return CandidateEvaluationReceipt(
        **strict_model_dump(metadata),
        dataset_id=run.report.dataset_id,
        dataset_version=run.report.dataset_version,
        dataset_sha256=run.report.dataset_sha256,
        response_set_sha256=run.response_set_sha256,
        report_sha256=report_sha256,
        candidate_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        source_commit=source_commit,
        metrics=run.report.metrics,
        evaluated_at_utc=evaluated_at_utc,
    )


def write_candidate_receipt(
    path: Path,
    receipt: CandidateEvaluationReceipt,
) -> None:
    """Write one additive receipt without overwriting an existing receipt."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        strict_model_dump(receipt),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
    except FileExistsError as exc:
        raise CandidateEvaluationInputError("receipt path already exists") from exc


__all__ = [
    "CandidateEvaluationInputError",
    "CandidateEvaluationInfrastructureError",
    "CandidateEvaluationRun",
    "CandidateInput",
    "CandidateSelector",
    "CandidateTimingPolicy",
    "build_candidate_evaluation_receipt",
    "run_candidate_evaluation",
    "serialize_candidate_traces",
    "write_candidate_receipt",
]
