from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

import pytest

from wind_forecast.operational_copilot import (
    OfflineOperationalCopilotRunner,
    OperationalCopilot,
)
from wind_forecast.operational_copilot_models import (
    CopilotRequest,
    OperationalHttpRequest,
    OperationalToolSelection,
)
from wind_forecast.operational_query_models import (
    AnswerStatus,
    AuthorizationContext,
    OperationalAnswer,
    QueryKind,
)


NOW = datetime(2026, 8, 17, 18, tzinfo=timezone.utc)
AUTHORIZATION = AuthorizationContext(
    principal="local-api-operator",
    trusted_local=True,
)


def _arguments(query_kind: str) -> dict[str, Any]:
    selector: dict[str, Any] = {"kind": "latest"}
    payload: dict[str, Any] = {
        "contract_version": "operational_read_only_copilot_v1",
        "query_kind": query_kind,
        "selector": selector,
        "window_days": None,
        "pagination": None,
    }
    if query_kind in {"monitoring_performance", "monitoring_drift"}:
        payload["window_days"] = 30
    if query_kind == "reporting_run":
        payload["selector"] = {
            "kind": "exact_id",
            "id_type": "report_id",
            "identifier": "a" * 64,
        }
    return payload


def _empty_answer() -> OperationalAnswer:
    return OperationalAnswer(
        query_kind=QueryKind.OPERATIONAL_SUMMARY,
        status=AnswerStatus.EMPTY,
        summary=None,
        facts=(),
        evidence=(),
        limitations=(),
        failure=None,
        served_at_utc=NOW,
        correlation_id="executor-answer-correlation",
    )


class RecordingSelector:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[tuple[CopilotRequest, tuple[object, ...], float]] = []

    def select(
        self,
        request: CopilotRequest,
        *,
        tools: tuple[object, ...],
        timeout_seconds: float,
    ) -> object:
        self.calls.append((request, tools, timeout_seconds))
        return self.output


_UNSET = object()


class RecordingAnswerer:
    def __init__(self, output: object = _UNSET, *, error: Exception | None = None):
        self.output = _empty_answer() if output is _UNSET else output
        self.error = error
        self.calls: list[tuple[object, object]] = []

    def answer(self, value: object, authorization: object) -> object:
        self.calls.append((value, authorization))
        if self.error is not None:
            raise self.error
        return self.output


def _copilot(
    selector: object,
    answerer: object,
    **overrides: object,
) -> OperationalCopilot:
    values: dict[str, object] = {
        "selector": selector,
        "query_service": answerer,
        "observability_readiness": lambda: "ready",
        "clock": lambda: NOW,
    }
    values.update(overrides)
    return OperationalCopilot(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "query_kind",
    [item.value for item in QueryKind],
)
def test_valid_selection_supports_the_closed_eight_kind_allowlist(
    query_kind: str,
) -> None:
    selector = RecordingSelector(
        {"tool_name": "operational_query", "arguments": _arguments(query_kind)}
    )
    answerer = RecordingAnswerer()

    result = _copilot(selector, answerer).answer(
        "What verified state is available?",
        AUTHORIZATION,
    )

    assert result is answerer.output
    assert len(selector.calls) == 1
    assert len(answerer.calls) == 1
    request, tools, timeout_seconds = selector.calls[0]
    assert request.question == "What verified state is available?"
    assert request.deadline > request.requested_at_utc
    assert timeout_seconds == pytest.approx(1.0)
    assert len(tools) == 1
    assert tools[0].name == "operational_query"
    assert set(tools[0].input_schema["properties"]) == {
        "contract_version",
        "query_kind",
        "selector",
        "window_days",
        "pagination",
    }


def test_selector_output_and_http_arguments_are_strict() -> None:
    invalid_selection = RecordingSelector(
        {
            "tool_name": "operational_query",
            "arguments": _arguments("operational_summary"),
            "answer": "model-generated fact",
        }
    )
    answerer = RecordingAnswerer()

    result = _copilot(invalid_selection, answerer).answer(
        "question",
        AUTHORIZATION,
    )

    assert result.status == AnswerStatus.REFUSED
    assert result.failure is not None
    assert result.failure.code == "invalid_tool_selection"
    assert not answerer.calls

    invalid_arguments = RecordingSelector(
        {
            "tool_name": "operational_query",
            "arguments": {
                **_arguments("operational_summary"),
                "unexpected": True,
            },
        }
    )
    result = _copilot(invalid_arguments, answerer).answer(
        "question",
        AUTHORIZATION,
    )
    assert result.failure is not None
    assert result.failure.code == "invalid_operational_request"
    assert not answerer.calls

    semantic_invalid = RecordingSelector(
        {
            "tool_name": "operational_query",
            "arguments": _arguments("monitoring_performance")
            | {"window_days": None},
        }
    )
    result = _copilot(semantic_invalid, answerer).answer(
        "question",
        AUTHORIZATION,
    )
    assert result.failure is not None
    assert result.failure.code == "invalid_operational_request"
    assert not answerer.calls


def test_unknown_tool_is_refused_without_executor_call() -> None:
    selector = RecordingSelector(
        {"tool_name": "web_search", "arguments": {"query": "secret"}}
    )
    answerer = RecordingAnswerer()

    result = _copilot(selector, answerer).answer("question", AUTHORIZATION)

    assert result.status == AnswerStatus.REFUSED
    assert result.failure is not None
    assert result.failure.code == "unsupported_tool"
    assert not answerer.calls


def test_unauthorized_context_is_rejected_before_selector_or_operational_reads() -> None:
    selector = RecordingSelector(
        {"tool_name": "operational_query", "arguments": _arguments("data_quality")}
    )
    answerer = RecordingAnswerer()

    result = _copilot(selector, answerer).answer(
        "question",
        AuthorizationContext(principal="remote", trusted_local=False),
    )

    assert result.status == AnswerStatus.UNAUTHORIZED
    assert result.failure is not None
    assert result.failure.code == "operator_not_authorized"
    assert not selector.calls
    assert not answerer.calls

    local_but_unknown_principal = _copilot(selector, answerer).answer(
        "question",
        AuthorizationContext(principal="another-local-principal", trusted_local=True),
    )
    assert local_but_unknown_principal.status == AnswerStatus.UNAUTHORIZED
    assert not selector.calls
    assert not answerer.calls


def test_observability_degraded_refuses_before_selector() -> None:
    selector = RecordingSelector(
        {"tool_name": "operational_query", "arguments": _arguments("data_quality")}
    )
    answerer = RecordingAnswerer()

    result = _copilot(
        selector,
        answerer,
        observability_readiness=lambda: "degraded",
    ).answer("question", AUTHORIZATION)

    assert result.status == AnswerStatus.REFUSED
    assert result.failure is not None
    assert result.failure.code == "copilot_observability_degraded"
    assert not selector.calls
    assert not answerer.calls


def test_selector_exception_is_not_retried_or_leaked() -> None:
    selector = RecordingSelector(object())

    def fail(*_args: object, **_kwargs: object) -> object:
        selector.calls.append((None, (), 0.0))  # type: ignore[arg-type]
        raise RuntimeError("secret path and provider response")

    selector.select = fail  # type: ignore[method-assign]
    answerer = RecordingAnswerer()

    result = _copilot(selector, answerer).answer("question", AUTHORIZATION)

    assert result.status == AnswerStatus.REFUSED
    assert result.failure is not None
    assert result.failure.code == "selector_failed"
    assert "secret" not in result.failure.message
    assert len(selector.calls) == 1
    assert not answerer.calls


def test_selector_timeout_error_maps_to_timeout() -> None:
    selector = RecordingSelector(object())

    def timeout(*args: object, **kwargs: object) -> object:
        selector.calls.append((None, (), 0.0))  # type: ignore[arg-type]
        raise TimeoutError("provider timeout")

    selector.select = timeout  # type: ignore[method-assign]
    answerer = RecordingAnswerer()

    result = _copilot(selector, answerer).answer("question", AUTHORIZATION)

    assert result.status == AnswerStatus.TIMEOUT
    assert result.failure is not None
    assert result.failure.code == "selector_timeout"
    assert len(selector.calls) == 1
    assert not answerer.calls


def test_selector_timeout_is_deterministic_and_does_not_execute_tool() -> None:
    class SlowSelector(RecordingSelector):
        def select(self, request, *, tools, timeout_seconds):  # type: ignore[no-untyped-def]
            self.calls.append((request, tools, timeout_seconds))
            time.sleep(0.03)
            return self.output

    selector = SlowSelector(
        {"tool_name": "operational_query", "arguments": _arguments("data_quality")}
    )
    answerer = RecordingAnswerer()

    result = _copilot(
        selector,
        answerer,
        selector_timeout_seconds=0.001,
    ).answer("question", AUTHORIZATION)

    assert result.status == AnswerStatus.TIMEOUT
    assert result.failure is not None
    assert result.failure.code == "selector_timeout"
    assert len(selector.calls) == 1
    assert not answerer.calls


def test_expired_deadline_prevents_selector() -> None:
    class ExpiredClock:
        def __init__(self) -> None:
            self.values = iter((0.0, 1.0))

        def __call__(self) -> float:
            return next(self.values)

    selector = RecordingSelector(
        {"tool_name": "operational_query", "arguments": _arguments("data_quality")}
    )
    answerer = RecordingAnswerer()

    result = _copilot(
        selector,
        answerer,
        max_deadline_seconds=0.1,
        selector_timeout_seconds=0.05,
        monotonic_clock=ExpiredClock(),
    ).answer("question", AUTHORIZATION)

    assert result.status == AnswerStatus.TIMEOUT
    assert result.failure is not None
    assert result.failure.code == "copilot_deadline_exceeded"
    assert not selector.calls
    assert not answerer.calls


def test_deadline_expiry_after_selection_prevents_executor() -> None:
    class ExpiringClock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            return self.value

    clock = ExpiringClock()

    class Selector(RecordingSelector):
        def select(self, request, *, tools, timeout_seconds):  # type: ignore[no-untyped-def]
            self.calls.append((request, tools, timeout_seconds))
            clock.value = 10.0
            return self.output

    selector = Selector(
        {"tool_name": "operational_query", "arguments": _arguments("data_quality")}
    )
    answerer = RecordingAnswerer()

    result = _copilot(
        selector,
        answerer,
        monotonic_clock=clock,
    ).answer("question", AUTHORIZATION)

    assert result.status == AnswerStatus.TIMEOUT
    assert result.failure is not None
    assert result.failure.code == "selector_timeout"
    assert not answerer.calls


def test_answer_passthrough_preserves_identity_and_all_fields() -> None:
    expected = OperationalAnswer(
        query_kind=QueryKind.OPERATIONAL_SUMMARY,
        status=AnswerStatus.EMPTY,
        summary=None,
        facts=(),
        evidence=(),
        limitations=("recorded limitation",),
        failure=None,
        served_at_utc=NOW,
        correlation_id="answer-correlation",
    )
    selector = RecordingSelector(
        {"tool_name": "operational_query", "arguments": _arguments("data_quality")}
    )
    answerer = RecordingAnswerer(expected)

    result = _copilot(selector, answerer).answer("question", AUTHORIZATION)

    assert result is expected
    assert result.model_dump() == expected.model_dump()
    query, authorization = answerer.calls[0]
    assert query.correlation_id
    assert query.deadline > query.requested_at_utc
    assert authorization is AUTHORIZATION


@pytest.mark.parametrize("output", [object(), None, {"status": "answered"}])
def test_invalid_executor_output_is_refused(output: object) -> None:
    selector = RecordingSelector(
        {"tool_name": "operational_query", "arguments": _arguments("data_quality")}
    )
    answerer = RecordingAnswerer(output)

    result = _copilot(selector, answerer).answer("question", AUTHORIZATION)

    assert result.status == AnswerStatus.REFUSED
    assert result.failure is not None
    assert result.failure.code == "invalid_tool_output"


def test_executor_exception_is_sanitized_and_not_retried() -> None:
    selector = RecordingSelector(
        {"tool_name": "operational_query", "arguments": _arguments("data_quality")}
    )
    answerer = RecordingAnswerer(error=RuntimeError("private path /tmp/secret"))

    result = _copilot(selector, answerer).answer("question", AUTHORIZATION)

    assert result.status == AnswerStatus.UNAVAILABLE
    assert result.failure is not None
    assert result.failure.code == "operational_query_executor_unavailable"
    assert "/tmp" not in result.failure.message
    assert len(answerer.calls) == 1


def test_offline_runner_is_in_memory_and_calls_once_per_question() -> None:
    selector = RecordingSelector(
        {"tool_name": "operational_query", "arguments": _arguments("data_quality")}
    )
    answerer = RecordingAnswerer()
    runner = OfflineOperationalCopilotRunner(_copilot(selector, answerer))

    results = runner.run_many(("first", "second"), AUTHORIZATION)

    assert results == (answerer.output, answerer.output)
    assert len(selector.calls) == 2
    assert len(answerer.calls) == 2


def test_boundary_models_reject_extra_fields() -> None:
    with pytest.raises(ValueError):
        OperationalHttpRequest.model_validate(
            {**_arguments("data_quality"), "prompt": "do something else"},
            strict=True,
        )
    with pytest.raises(ValueError):
        OperationalToolSelection.model_validate(
            {
                "tool_name": "operational_query",
                "arguments": _arguments("data_quality"),
                "facts": [],
            },
            strict=True,
        )
