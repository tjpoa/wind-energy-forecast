"""Offline deterministic evaluation for future operational-query candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from pydantic import ValidationError

from .operational_evaluation_models import (
    CandidateTrace,
    EvaluationCaseResult,
    EvaluationCategory,
    EvaluationMetrics,
    ExpectedAnswer,
    ExpectedCitation,
    OperationalEvaluationCase,
    OperationalEvaluationManifest,
    OperationalEvaluationReport,
    strict_model_dump,
)
from .operational_query_models import AnswerStatus


MAX_JSONL_LINE_BYTES = 256 * 1024
MAX_INPUT_BYTES = 32 * 1024 * 1024
_GLOBAL_PRIVATE_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:home|Users|etc|var|tmp|opt)/", re.IGNORECASE),
    re.compile(r"\bfile://", re.IGNORECASE),
    re.compile(
        r"\b(?:aws_secret_access_key|aws_access_key_id|api[_ -]?key|"
        r"access[_ -]?token|refresh[_ -]?token|token|password|secret|"
        r"client[_ -]?secret|connection string)\b\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|[\\/])(?:\.env|credentials(?:\.[A-Za-z0-9_-]+)?)\b", re.IGNORECASE),
    re.compile(r"\bTraceback \(most recent call last\):", re.IGNORECASE),
)


class OperationalEvaluationInputError(ValueError):
    """Raised for an invalid dataset or candidate response set."""


@dataclass(frozen=True)
class OperationalEvaluationDataset:
    manifest: OperationalEvaluationManifest
    cases: tuple[OperationalEvaluationCase, ...]
    dataset_sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bounded(path: Path, *, remaining: int = MAX_INPUT_BYTES) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise OperationalEvaluationInputError("evaluation input is unavailable") from exc
    if size > remaining:
        raise OperationalEvaluationInputError("evaluation input exceeds the size limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise OperationalEvaluationInputError("evaluation input is unavailable") from exc


def _parse_json(payload: bytes, model: type[Any]) -> Any:
    try:
        return model.model_validate_json(payload, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise OperationalEvaluationInputError("evaluation input schema is invalid") from exc


def _parse_jsonl(payload: bytes, model: type[Any]) -> tuple[Any, ...]:
    items: list[Any] = []
    for raw_line in payload.splitlines():
        if not raw_line.strip():
            continue
        if len(raw_line) > MAX_JSONL_LINE_BYTES:
            raise OperationalEvaluationInputError("evaluation JSONL line exceeds the limit")
        try:
            items.append(model.model_validate_json(raw_line, strict=True))
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise OperationalEvaluationInputError(
                "evaluation JSONL schema is invalid"
            ) from exc
    return tuple(items)


def load_evaluation_dataset(manifest_path: Path) -> OperationalEvaluationDataset:
    """Load and identity-verify one immutable evaluation dataset."""

    manifest_bytes = _read_bounded(manifest_path)
    manifest = _parse_json(manifest_bytes, OperationalEvaluationManifest)
    cases_path = manifest_path.parent / "cases.jsonl"
    cases_bytes = _read_bounded(
        cases_path,
        remaining=MAX_INPUT_BYTES - len(manifest_bytes),
    )
    if _sha256(cases_bytes) != manifest.cases_sha256:
        raise OperationalEvaluationInputError("evaluation cases checksum mismatch")
    cases = _parse_jsonl(cases_bytes, OperationalEvaluationCase)
    if len(cases) != manifest.case_count:
        raise OperationalEvaluationInputError("evaluation case count mismatch")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise OperationalEvaluationInputError("evaluation case IDs are not unique")
    if case_ids != sorted(case_ids):
        raise OperationalEvaluationInputError("evaluation cases are not canonical-order")
    distribution = Counter(case.category for case in cases)
    if dict(distribution) != manifest.distribution:
        raise OperationalEvaluationInputError("evaluation distribution mismatch")
    if any(case.oracle_id not in manifest.oracles for case in cases):
        raise OperationalEvaluationInputError("evaluation case references an unknown oracle")
    dataset_sha256 = _sha256(manifest_bytes + cases_bytes)
    return OperationalEvaluationDataset(manifest, cases, dataset_sha256)


def load_candidate_traces(
    responses_path: Path,
    *,
    dataset_bytes: int = 0,
) -> tuple[tuple[CandidateTrace, ...], str]:
    """Load bounded candidate JSONL and return the exact byte digest."""

    payload = _read_bounded(
        responses_path,
        remaining=MAX_INPUT_BYTES - dataset_bytes,
    )
    return _parse_jsonl(payload, CandidateTrace), _sha256(payload)


def _canonical_text(value: Any) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        raise OperationalEvaluationInputError("candidate contains non-finite facts")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _expected_summary(trace: CandidateTrace) -> str | None:
    if trace.answer.status != AnswerStatus.ANSWERED:
        return None
    return " ".join(
        f"{fact.name}={_canonical_text(fact.value)} "
        f"{''.join(f'[{item}]' for item in fact.evidence_ids)}."
        for fact in trace.answer.facts
    )


def _safe_paraphrase_abstention(
    case: OperationalEvaluationCase,
    trace: CandidateTrace,
) -> bool:
    return (
        case.category == EvaluationCategory.PARAPHRASE
        and case.expected_tool is not None
        and trace.selected_tool is None
        and trace.tool_arguments is None
        and trace.answer.status == AnswerStatus.REFUSED
        and not trace.answer.facts
        and not trace.answer.evidence
        and trace.answer.summary is None
        and not trace.answer.limitations
        and trace.answer.query_kind is None
        and trace.answer.failure is not None
        and trace.answer.failure.code == "unsupported_query_kind"
        and trace.answer.failure.message
        == "The requested operational question is not supported."
        and not trace.answer.failure.retryable
        and trace.answer.failure.evidence_state.value == "unsupported"
        and not _privacy_failure_codes(trace.answer)
    )


def _fact_failure_codes(
    expected: ExpectedAnswer,
    trace: CandidateTrace,
) -> list[str]:
    codes: list[str] = []
    actual_facts = trace.answer.facts
    if len(actual_facts) != len(expected.facts):
        return ["fact_set_mismatch"]
    evidence_by_id = {item.evidence_id: item for item in trace.answer.evidence}
    for expected_fact, actual_fact in zip(expected.facts, actual_facts):
        if (
            actual_fact.name != expected_fact.name
            or actual_fact.value != expected_fact.value
            or actual_fact.unit_or_scale != expected_fact.unit_or_scale
            or actual_fact.as_of != expected_fact.as_of
        ):
            codes.append("fact_value_mismatch")
            continue
        cited = [evidence_by_id[item] for item in actual_fact.evidence_ids]
        if tuple(item.source_kind for item in cited) != expected_fact.source_kinds:
            codes.append("fact_provenance_mismatch")
        if expected_fact.requires_observed_at_utc and all(
            item.observed_at_utc is None for item in cited
        ):
            codes.append("mutable_evidence_missing_observation_time")
    return codes


def _citation_failure_codes(
    expected: Sequence[ExpectedCitation],
    trace: CandidateTrace,
) -> list[str]:
    actual_by_source = {item.source_kind: item for item in trace.answer.evidence}
    expected_by_source = {item.source_kind: item for item in expected}
    if (
        len(actual_by_source) != len(trace.answer.evidence)
        or set(actual_by_source) != set(expected_by_source)
    ):
        return ["evidence_set_mismatch"]
    for source_kind, expected_citation in expected_by_source.items():
        actual = actual_by_source[source_kind].model_dump(
            mode="json", exclude={"evidence_id"}
        )
        if actual != expected_citation.model_dump(mode="json"):
            return ["evidence_provenance_mismatch"]
    return []


def _privacy_failure_codes(answer: Any) -> list[str]:
    serialized = answer.model_dump_json()
    return (
        ["private_data_exposure"]
        if any(pattern.search(serialized) for pattern in _GLOBAL_PRIVATE_PATTERNS)
        else []
    )


def _answer_failure_codes(
    case: OperationalEvaluationCase,
    expected: ExpectedAnswer,
    expected_citations: Sequence[ExpectedCitation],
    trace: CandidateTrace,
) -> list[str]:
    answer = trace.answer
    codes: list[str] = []
    if answer.query_kind != expected.query_kind:
        codes.append("answer_query_kind_mismatch")
    if answer.status != expected.status:
        codes.append("answer_status_mismatch")
    if tuple(answer.limitations) != expected.limitations:
        codes.append("limitations_mismatch")
    codes.extend(_fact_failure_codes(expected, trace))
    codes.extend(_citation_failure_codes(expected_citations, trace))
    if expected.status == AnswerStatus.ANSWERED:
        if answer.summary != _expected_summary(trace):
            codes.append("summary_not_deterministically_grounded")
        referenced = {item for fact in answer.facts for item in fact.evidence_ids}
        if referenced != {item.evidence_id for item in answer.evidence}:
            codes.append("evidence_reference_mismatch")
    elif answer.summary is not None:
        codes.append("unexpected_summary")
    failure = answer.failure
    if expected.failure_code is None:
        if failure is not None:
            codes.append("unexpected_failure")
    elif failure is None or (
        failure.code != expected.failure_code
        or failure.message != expected.failure_message
        or failure.retryable != expected.failure_retryable
        or failure.evidence_state != expected.failure_evidence_state
    ):
        codes.append("failure_mismatch")
    codes.extend(_privacy_failure_codes(answer))
    factual_payload = json.dumps(
        {
            "summary": answer.summary,
            "facts": [item.model_dump(mode="json") for item in answer.facts],
        },
        sort_keys=True,
    )
    if any(
        claim.casefold() in factual_payload.casefold()
        for claim in case.forbidden_claims
    ):
        codes.append("forbidden_claim")
    return codes


def _evaluate_case(
    case: OperationalEvaluationCase,
    expected: ExpectedAnswer,
    expected_citations: Sequence[ExpectedCitation],
    trace: CandidateTrace,
) -> EvaluationCaseResult:
    codes: list[str] = []
    expected_tool = case.expected_tool
    safe_abstention = _safe_paraphrase_abstention(case, trace)
    if safe_abstention:
        return EvaluationCaseResult(
            case_id=case.case_id,
            passed=False,
            paraphrase_recognized=False,
            failure_codes=("safe_paraphrase_abstention",),
        )
    if expected_tool is None:
        if trace.selected_tool is not None or trace.tool_arguments is not None:
            codes.append("forbidden_tool_dispatch")
    else:
        if trace.selected_tool != expected_tool.name:
            codes.append("tool_selection_mismatch")
        if trace.tool_arguments is None or (
            strict_model_dump(trace.tool_arguments)
            != strict_model_dump(expected_tool.arguments)
        ):
            codes.append("tool_arguments_mismatch")
    codes.extend(_answer_failure_codes(case, expected, expected_citations, trace))
    codes = sorted(set(codes))
    paraphrase_recognized = (
        not codes if case.category == EvaluationCategory.PARAPHRASE else None
    )
    return EvaluationCaseResult(
        case_id=case.case_id,
        passed=not codes,
        paraphrase_recognized=paraphrase_recognized,
        failure_codes=tuple(codes),
    )


def evaluate_candidate_results(
    dataset: OperationalEvaluationDataset,
    traces: Iterable[CandidateTrace],
    *,
    response_set_sha256: str,
) -> OperationalEvaluationReport:
    """Score one complete candidate result set without operational reads."""

    trace_list = tuple(traces)
    trace_by_id = {trace.case_id: trace for trace in trace_list}
    if len(trace_by_id) != len(trace_list):
        raise OperationalEvaluationInputError("candidate case IDs are not unique")
    expected_ids = {case.case_id for case in dataset.cases}
    if set(trace_by_id) != expected_ids:
        raise OperationalEvaluationInputError("candidate case IDs do not match dataset")
    results = tuple(
        _evaluate_case(
            case,
            dataset.manifest.oracles[case.oracle_id],
            (
                dataset.manifest.citation_sets[
                    dataset.manifest.oracles[case.oracle_id].citation_set_id
                ]
                if dataset.manifest.oracles[case.oracle_id].citation_set_id
                is not None
                else ()
            ),
            trace_by_id[case.case_id],
        )
        for case in dataset.cases
    )
    by_id = {case.case_id: case for case in dataset.cases}
    canonical = [
        item for item in results
        if by_id[item.case_id].category == EvaluationCategory.CANONICAL
    ]
    paraphrases = [
        item for item in results
        if by_id[item.case_id].category == EvaluationCategory.PARAPHRASE
    ]
    evidence_states = [
        item for item in results
        if by_id[item.case_id].category == EvaluationCategory.EVIDENCE_STATE
    ]
    safe_abstentions = sum(
        "safe_paraphrase_abstention" in item.failure_codes for item in results
    )
    critical_failures = sum(
        any(code != "safe_paraphrase_abstention" for code in item.failure_codes)
        for item in results
    )
    factual_results = [
        item for item in results
        if dataset.manifest.oracles[by_id[item.case_id].oracle_id].status
        == AnswerStatus.ANSWERED
    ]
    def rate(values: Sequence[EvaluationCaseResult]) -> float:
        return sum(item.passed for item in values) / len(values)

    paraphrase_rate = sum(
        item.paraphrase_recognized is True for item in paraphrases
    ) / len(paraphrases)
    metrics = EvaluationMetrics(
        case_count=len(results),
        passed_case_count=sum(item.passed for item in results),
        critical_failure_count=critical_failures,
        canonical_tool_pass_rate=rate(canonical),
        factual_grounding_pass_rate=sum(
            item.passed or "safe_paraphrase_abstention" in item.failure_codes
            for item in factual_results
        ) / len(factual_results),
        evidence_state_pass_rate=rate(evidence_states),
        paraphrase_recognition_pass_rate=paraphrase_rate,
        safe_paraphrase_abstention_count=safe_abstentions,
    )
    policy = dataset.manifest.gate_policy
    passed = (
        critical_failures == 0
        and metrics.canonical_tool_pass_rate == policy.canonical_tool_pass_rate
        and metrics.factual_grounding_pass_rate
        == policy.factual_grounding_pass_rate
        and metrics.evidence_state_pass_rate == policy.evidence_state_pass_rate
        and paraphrase_rate >= policy.paraphrase_recognition_pass_rate
        and safe_abstentions <= policy.safe_paraphrase_abstentions_allowed
    )
    return OperationalEvaluationReport(
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.dataset_version,
        dataset_sha256=dataset.dataset_sha256,
        response_set_id=response_set_sha256,
        response_set_sha256=response_set_sha256,
        status="passed" if passed else "failed",
        acceptance_state=dataset.manifest.acceptance_state,
        metrics=metrics,
        results=results,
    )


def sanitized_report_json(report: OperationalEvaluationReport) -> str:
    """Serialize only the bounded report contract, never candidate payloads."""

    return json.dumps(strict_model_dump(report), sort_keys=True, separators=(",", ":"))


__all__ = [
    "MAX_INPUT_BYTES",
    "MAX_JSONL_LINE_BYTES",
    "OperationalEvaluationDataset",
    "OperationalEvaluationInputError",
    "evaluate_candidate_results",
    "load_candidate_traces",
    "load_evaluation_dataset",
    "sanitized_report_json",
]
