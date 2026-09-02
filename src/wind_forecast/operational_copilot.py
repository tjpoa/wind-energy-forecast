"""Provider-neutral, single-tool operational Copilot core."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from time import monotonic
from typing import Any, Callable, Protocol
from uuid import uuid4

from pydantic import ValidationError

from .operational_copilot_models import (
    CopilotRequest,
    LOCAL_OPERATOR_PRINCIPAL,
    OperationalHttpRequest,
    OperationalToolDefinition,
    OperationalToolSelection,
    allowed_operational_tools,
)
from .operational_observability import (
    get_operational_observability,
)
from .operational_query_models import (
    AnswerStatus,
    AuthorizationContext,
    EvidenceState,
    OperationalAnswer,
    OperationalFailure,
    OperationalQuery,
)


DEFAULT_COPILOT_DEADLINE_SECONDS = 5.0
DEFAULT_SELECTOR_TIMEOUT_SECONDS = 1.0


class OperationalSelector(Protocol):
    """Injectable selector that can choose only from the supplied tool catalog."""

    def select(
        self,
        request: CopilotRequest,
        *,
        tools: tuple[OperationalToolDefinition, ...],
        timeout_seconds: float,
    ) -> object:
        """Return one JSON-like tool selection without facts or an answer."""


class OperationalQueryAnswerer(Protocol):
    """Minimal read-only executor boundary used by the Copilot."""

    def answer(
        self,
        value: OperationalQuery,
        authorization: AuthorizationContext | Mapping[str, Any] | None,
    ) -> OperationalAnswer:
        """Execute one already-selected operational query."""


def _default_observability_readiness() -> str:
    """Read local readiness without creating a store or performing network I/O."""
    try:
        return str(get_operational_observability().health().get("status"))
    except Exception:
        return "degraded"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _failure_answer(
    *,
    status: AnswerStatus,
    code: str,
    message: str,
    retryable: bool,
    evidence_state: EvidenceState,
    correlation_id: str,
    served_at_utc: datetime,
) -> OperationalAnswer:
    return OperationalAnswer(
        query_kind=None,
        status=status,
        summary=None,
        facts=(),
        evidence=(),
        limitations=(),
        failure=OperationalFailure(
            code=code,
            message=message,
            retryable=retryable,
            evidence_state=evidence_state,
        ),
        served_at_utc=served_at_utc,
        correlation_id=correlation_id,
    )


def _unauthorized_answer(
    *,
    correlation_id: str,
    served_at_utc: datetime,
) -> OperationalAnswer:
    return _failure_answer(
        status=AnswerStatus.UNAUTHORIZED,
        code="operator_not_authorized",
        message="The local operator is not authorized for this query.",
        retryable=False,
        evidence_state=EvidenceState.UNAUTHORIZED,
        correlation_id=correlation_id,
        served_at_utc=served_at_utc,
    )


def _refused_answer(
    *,
    code: str,
    message: str,
    correlation_id: str,
    served_at_utc: datetime,
) -> OperationalAnswer:
    return _failure_answer(
        status=AnswerStatus.REFUSED,
        code=code,
        message=message,
        retryable=False,
        evidence_state=EvidenceState.UNSUPPORTED,
        correlation_id=correlation_id,
        served_at_utc=served_at_utc,
    )


def _timeout_answer(
    *,
    code: str,
    message: str,
    correlation_id: str,
    served_at_utc: datetime,
) -> OperationalAnswer:
    return _failure_answer(
        status=AnswerStatus.TIMEOUT,
        code=code,
        message=message,
        retryable=True,
        evidence_state=EvidenceState.TIMEOUT,
        correlation_id=correlation_id,
        served_at_utc=served_at_utc,
    )


@dataclass(frozen=True)
class OperationalCopilot:
    """Run one provider-neutral selection and one deterministic query.

    The selector is deliberately synchronous and receives its own cooperative
    timeout. The core measures that budget and refuses late results; it does
    not create a thread that could continue an external provider call after a
    timeout. The injected query service owns authorization and operational
    reads, including its own cooperative deadline propagation.
    """

    selector: OperationalSelector
    query_service: OperationalQueryAnswerer
    max_deadline_seconds: float = DEFAULT_COPILOT_DEADLINE_SECONDS
    selector_timeout_seconds: float = DEFAULT_SELECTOR_TIMEOUT_SECONDS
    observability_readiness: Callable[[], str] = _default_observability_readiness
    clock: Callable[[], datetime] = _utc_now
    monotonic_clock: Callable[[], float] = monotonic

    def __post_init__(self) -> None:
        for name, value in (
            ("max_deadline_seconds", self.max_deadline_seconds),
            ("selector_timeout_seconds", self.selector_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.selector_timeout_seconds > self.max_deadline_seconds:
            raise ValueError(
                "selector_timeout_seconds must not exceed max_deadline_seconds"
            )

    def answer(
        self,
        question: Any,
        authorization: AuthorizationContext | Mapping[str, Any] | None,
    ) -> OperationalAnswer:
        """Select one accepted tool and pass its answer through unchanged."""
        requested_at_utc = self._now()
        correlation_id = uuid4().hex
        deadline = requested_at_utc + timedelta(
            seconds=self.max_deadline_seconds
        )
        started = self.monotonic_clock()

        if self._readiness() != "ready":
            return _refused_answer(
                code="copilot_observability_degraded",
                message="The operational Copilot is not ready.",
                correlation_id=correlation_id,
                served_at_utc=requested_at_utc,
            )

        context = self._authorization_context(authorization)
        if (
            context is None
            or not context.trusted_local
            or context.principal != LOCAL_OPERATOR_PRINCIPAL
        ):
            return _unauthorized_answer(
                correlation_id=correlation_id,
                served_at_utc=requested_at_utc,
            )

        try:
            request = CopilotRequest(
                question=question,
                requested_at_utc=requested_at_utc,
                correlation_id=correlation_id,
                deadline=deadline,
            )
        except (ValidationError, TypeError, ValueError):
            return _refused_answer(
                code="invalid_copilot_question",
                message="The Copilot question is invalid.",
                correlation_id=correlation_id,
                served_at_utc=requested_at_utc,
            )

        remaining = self._remaining(started)
        if remaining <= 0:
            return _timeout_answer(
                code="copilot_deadline_exceeded",
                message="The operational Copilot deadline expired.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )
        selector_timeout = min(self.selector_timeout_seconds, remaining)
        selector_started = self.monotonic_clock()
        try:
            selected = self.selector.select(
                request,
                tools=allowed_operational_tools(),
                timeout_seconds=selector_timeout,
            )
        except TimeoutError:
            return _timeout_answer(
                code="selector_timeout",
                message="The operational Copilot selector exceeded its timeout.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )
        except ValueError as exc:
            code = str(exc) or "selector_failed"
            return _refused_answer(
                code=code,
                message="A pergunta não está disponível no catálogo do Copilot.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )
        except Exception:
            return _refused_answer(
                code="selector_failed",
                message="The operational Copilot selector failed.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )
        if (
            self._remaining(started) <= 0
            or self._elapsed(selector_started) > selector_timeout
        ):
            return _timeout_answer(
                code="selector_timeout",
                message="The operational Copilot selector exceeded its timeout.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )

        try:
            selection = OperationalToolSelection.model_validate(
                selected,
                strict=True,
            )
        except (ValidationError, TypeError, ValueError):
            return _refused_answer(
                code="invalid_tool_selection",
                message="The Copilot tool selection is invalid.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )

        if selection.tool_name != "operational_query":
            return _refused_answer(
                code="unsupported_tool",
                message="The requested Copilot tool is not supported.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )

        try:
            public_request = OperationalHttpRequest.model_validate(
                selection.arguments,
                strict=True,
            )
        except (ValidationError, TypeError, ValueError):
            return _refused_answer(
                code="invalid_operational_request",
                message="The selected operational query is invalid.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )

        if self._remaining(started) <= 0:
            return _timeout_answer(
                code="copilot_deadline_exceeded",
                message="The operational Copilot deadline expired.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )

        try:
            query = OperationalQuery(
                **public_request.model_dump(mode="python"),
                requested_at_utc=requested_at_utc,
                correlation_id=correlation_id,
                deadline=deadline,
            )
        except (ValidationError, TypeError, ValueError):
            return _refused_answer(
                code="invalid_operational_request",
                message="The selected operational query is invalid.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )
        try:
            answer = self.query_service.answer(query, context)
        except Exception:
            return _failure_answer(
                status=AnswerStatus.UNAVAILABLE,
                code="operational_query_executor_unavailable",
                message="The operational query service is unavailable.",
                retryable=True,
                evidence_state=EvidenceState.UNAVAILABLE,
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )

        if self._remaining(started) <= 0:
            return _timeout_answer(
                code="copilot_deadline_exceeded",
                message="The operational Copilot deadline expired.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )
        if not isinstance(answer, OperationalAnswer):
            return _refused_answer(
                code="invalid_tool_output",
                message="The operational query tool returned an invalid answer.",
                correlation_id=correlation_id,
                served_at_utc=self._now(),
            )
        return answer

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _elapsed(self, started: float) -> float:
        return max(0.0, self.monotonic_clock() - started)

    def _remaining(self, started: float) -> float:
        return self.max_deadline_seconds - self._elapsed(started)

    def _readiness(self) -> str:
        try:
            return self.observability_readiness()
        except Exception:
            return "degraded"

    @staticmethod
    def _authorization_context(
        authorization: AuthorizationContext | Mapping[str, Any] | None,
    ) -> AuthorizationContext | None:
        try:
            return (
                authorization
                if isinstance(authorization, AuthorizationContext)
                else AuthorizationContext.model_validate(
                    authorization,
                    strict=True,
                )
            )
        except (ValidationError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class OfflineOperationalCopilotRunner:
    """In-memory runner for synthetic selectors and query answerers only."""

    copilot: OperationalCopilot

    def run(
        self,
        question: str,
        authorization: AuthorizationContext | Mapping[str, Any] | None,
    ) -> OperationalAnswer:
        """Run one synthetic/offline request without filesystem or network code."""
        return self.copilot.answer(question, authorization)

    def run_many(
        self,
        questions: Iterable[str],
        authorization: AuthorizationContext | Mapping[str, Any] | None,
    ) -> tuple[OperationalAnswer, ...]:
        """Run a bounded caller-provided sequence in memory."""
        return tuple(self.run(question, authorization) for question in questions)


__all__ = [
    "DEFAULT_COPILOT_DEADLINE_SECONDS",
    "DEFAULT_SELECTOR_TIMEOUT_SECONDS",
    "OfflineOperationalCopilotRunner",
    "OperationalCopilot",
    "OperationalQueryAnswerer",
    "OperationalSelector",
]
