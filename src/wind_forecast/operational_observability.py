"""Process-local, sanitized observability for the operational query API."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
from threading import Lock
from typing import Literal

from .config import load_operational_observability_config


OBSERVABILITY_SCHEMA_VERSION = (
    "wind_forecast.operational_observability_event.v1"
)
OBSERVABILITY_EVENTS_FILENAME = "events.jsonl"
OBSERVABILITY_TOOL_NAME = "operational_query"

EventType = Literal[
    "request.started",
    "request.finished",
    "tool.started",
    "tool.finished",
]

_EVENT_TYPES: tuple[EventType, ...] = (
    "request.started",
    "request.finished",
    "tool.started",
    "tool.finished",
)
_QUERY_KINDS = frozenset(
    {
        "operational_summary",
        "active_deployment",
        "data_quality",
        "monitoring_performance",
        "monitoring_drift",
        "monitoring_alerts",
        "active_model_metadata",
        "reporting_run",
    }
)
_ANSWER_STATUSES = frozenset(
    {
        "answered",
        "empty",
        "not_found",
        "refused",
        "unauthorized",
        "unavailable",
        "corrupt",
        "conflict",
        "timeout",
    }
)
_SUCCESS_STATUSES = frozenset({"answered", "empty", "not_found"})
_FAILURE_CODE_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")


def utc_timestamp(value: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp without host-local information."""
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def sanitize_failure_code(value: object | None) -> str | None:
    """Keep only the bounded domain-code form; never serialize raw failures."""
    if value is None:
        return None
    candidate = str(value)
    if _FAILURE_CODE_PATTERN.fullmatch(candidate):
        return candidate
    return "unspecified_failure"


def _query_kind(value: object | None) -> str | None:
    candidate = value.value if hasattr(value, "value") else value
    return candidate if isinstance(candidate, str) and candidate in _QUERY_KINDS else None


def _answer_status(value: object | None) -> str | None:
    candidate = value.value if hasattr(value, "value") else value
    return (
        candidate
        if isinstance(candidate, str) and candidate in _ANSWER_STATUSES
        else None
    )


def _http_status(value: int | None) -> int | None:
    if isinstance(value, int) and 100 <= value <= 599:
        return value
    return None


def _duration_ms(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return None
    return round(max(numeric, 0.0), 3)


@dataclass(frozen=True)
class ObservabilityContext:
    """Identifiers for one request and its child deterministic-tool span."""

    correlation_id: str
    trace_id: str
    request_span_id: str


class OperationalObservability:
    """Write bounded event records and expose process-local counters.

    The lock protects both JSONL writes and counters within one process. A
    multi-worker deployment would have one counter set per worker and requires
    a separately reviewed cross-process writer policy.
    """

    def __init__(
        self,
        store_root: Path | None,
        *,
        initially_degraded: bool = False,
    ) -> None:
        self._events_path = (
            None
            if store_root is None
            else Path(store_root) / OBSERVABILITY_EVENTS_FILENAME
        )
        self._lock = Lock()
        self._degraded = initially_degraded
        self._dropped_events = 0
        self._event_counts = Counter({event_type: 0 for event_type in _EVENT_TYPES})
        self._answer_status_counts = Counter(
            {answer_status: 0 for answer_status in sorted(_ANSWER_STATUSES)}
        )

    @property
    def readiness(self) -> Literal["ready", "degraded"]:
        """Return whether event writes are currently usable."""
        with self._lock:
            return "degraded" if self._degraded else "ready"

    def health(self) -> dict[str, object]:
        """Return a sanitized readiness payload without exposing the store path."""
        with self._lock:
            return {
                "status": "degraded" if self._degraded else "ready",
                "dropped_events": self._dropped_events,
            }

    def metrics(self) -> dict[str, object]:
        """Return process-local counters with only bounded low-cardinality keys."""
        with self._lock:
            return {
                "status": "degraded" if self._degraded else "ready",
                "event_counts": {
                    event_type: self._event_counts[event_type]
                    for event_type in _EVENT_TYPES
                },
                "answer_status_counts": {
                    answer_status: self._answer_status_counts[answer_status]
                    for answer_status in sorted(_ANSWER_STATUSES)
                },
                "dropped_events": self._dropped_events,
            }

    def request_started(
        self,
        context: ObservabilityContext,
        *,
        query_kind: object | None,
    ) -> None:
        self._record(
            event_type="request.started",
            context=context,
            span_id=context.request_span_id,
            parent_span_id=None,
            query_kind=query_kind,
            tool_name=None,
            result=None,
            answer_status=None,
            http_status=None,
            duration_ms=None,
            failure_code=None,
        )

    def request_finished(
        self,
        context: ObservabilityContext,
        *,
        query_kind: object | None,
        answer_status: object | None,
        http_status: int,
        duration_ms: float,
        failure_code: object | None,
    ) -> None:
        status = _answer_status(answer_status)
        self._record(
            event_type="request.finished",
            context=context,
            span_id=context.request_span_id,
            parent_span_id=None,
            query_kind=query_kind,
            tool_name=None,
            result=("success" if status in _SUCCESS_STATUSES else "failure"),
            answer_status=status,
            http_status=http_status,
            duration_ms=duration_ms,
            failure_code=failure_code,
        )

    def tool_started(
        self,
        context: ObservabilityContext,
        *,
        span_id: str,
        query_kind: object | None,
    ) -> None:
        self._record(
            event_type="tool.started",
            context=context,
            span_id=span_id,
            parent_span_id=context.request_span_id,
            query_kind=query_kind,
            tool_name=OBSERVABILITY_TOOL_NAME,
            result=None,
            answer_status=None,
            http_status=None,
            duration_ms=None,
            failure_code=None,
        )

    def tool_finished(
        self,
        context: ObservabilityContext,
        *,
        span_id: str,
        query_kind: object | None,
        answer_status: object | None,
        duration_ms: float,
        failure_code: object | None,
    ) -> None:
        status = _answer_status(answer_status)
        self._record(
            event_type="tool.finished",
            context=context,
            span_id=span_id,
            parent_span_id=context.request_span_id,
            query_kind=query_kind,
            tool_name=OBSERVABILITY_TOOL_NAME,
            result=("success" if status in _SUCCESS_STATUSES else "failure"),
            answer_status=status,
            http_status=None,
            duration_ms=duration_ms,
            failure_code=failure_code,
        )

    def _record(
        self,
        *,
        event_type: EventType,
        context: ObservabilityContext,
        span_id: str,
        parent_span_id: str | None,
        query_kind: object | None,
        tool_name: str | None,
        result: Literal["success", "failure"] | None,
        answer_status: object | None,
        http_status: int | None,
        duration_ms: float | None,
        failure_code: object | None,
    ) -> None:
        event = {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "event_type": event_type,
            "timestamp_utc": utc_timestamp(),
            "correlation_id": context.correlation_id,
            "trace_id": context.trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "query_kind": _query_kind(query_kind),
            "tool_name": tool_name,
            "result": result,
            "answer_status": _answer_status(answer_status),
            "http_status": _http_status(http_status),
            "duration_ms": _duration_ms(duration_ms),
            "failure_code": sanitize_failure_code(failure_code),
        }
        line = json.dumps(
            event,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"

        with self._lock:
            self._event_counts[event_type] += 1
            status = event["answer_status"]
            if status is not None:
                self._answer_status_counts[status] += 1
            try:
                if self._events_path is None:
                    raise OSError("observability event writer is unavailable")
                self._events_path.parent.mkdir(parents=True, exist_ok=True)
                with self._events_path.open(
                    "a",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    handle.write(line)
                    handle.flush()
            except Exception:
                self._dropped_events += 1
                self._degraded = True


_UNAVAILABLE_OBSERVABILITY = OperationalObservability(
    None,
    initially_degraded=True,
)


def unavailable_observability() -> OperationalObservability:
    """Return a non-writing degraded instance for configuration failures."""
    return _UNAVAILABLE_OBSERVABILITY


@lru_cache(maxsize=1)
def get_operational_observability() -> OperationalObservability:
    """Create the lazy writer without creating its directory or file."""
    config = load_operational_observability_config()
    return OperationalObservability(config.store_root)


__all__ = [
    "OBSERVABILITY_EVENTS_FILENAME",
    "OBSERVABILITY_SCHEMA_VERSION",
    "OBSERVABILITY_TOOL_NAME",
    "EventType",
    "ObservabilityContext",
    "OperationalObservability",
    "get_operational_observability",
    "sanitize_failure_code",
    "unavailable_observability",
    "utc_timestamp",
]
