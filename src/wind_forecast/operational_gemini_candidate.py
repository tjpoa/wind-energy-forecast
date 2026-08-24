"""Remote Gemini Interactions selector for the sealed synthetic evaluation."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import requests

from .operational_candidate_evaluation import (
    CandidateEvaluationInfrastructureError,
    CandidateEvaluationInputError,
    CandidateEvaluationRun,
    CandidateInput,
)
from .operational_copilot_models import OperationalToolSelection, TOOL_NAME
from .operational_evaluation import sanitized_report_json
from .operational_evaluation_models import EXPECTED_CASE_COUNT, strict_model_dump
from .operational_gemini_candidate_models import (
    GEMINI_ENDPOINT,
    GEMINI_MAX_RESPONSE_BYTES,
    GEMINI_MODEL,
    GeminiCandidateEvaluationReceipt,
    GeminiCandidateMetadata,
)

_STATIC_INSTRUCTIONS = "Select at most one approved operational tool for the supplied synthetic operator request. Call operational_query only when the request is authorized and supported by its schema. Otherwise make no tool call. Never invent another tool or answer the operational question."
_INVALID_OUTPUT = {"candidate_output": "invalid"}
_MAX_ARGUMENT_BYTES = 32 * 1024


@dataclass(frozen=True)
class GeminiTransportResponse:
    status_code: int
    body: bytes


class GeminiTransport(Protocol):
    def send(
        self,
        *,
        endpoint: str,
        api_key: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> GeminiTransportResponse: ...


class RequestsGeminiTransport:
    def __repr__(self) -> str:
        return "RequestsGeminiTransport()"

    def send(
        self,
        *,
        endpoint: str,
        api_key: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> GeminiTransportResponse:
        if endpoint != GEMINI_ENDPOINT:
            raise CandidateEvaluationInfrastructureError("gemini_endpoint_not_allowed")
        try:
            session = requests.Session()
        except Exception:
            raise CandidateEvaluationInfrastructureError(
                "gemini_transport_failure"
            ) from None
        session.trust_env = False
        try:
            response = session.post(
                GEMINI_ENDPOINT,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                json=dict(payload),
                timeout=timeout_seconds,
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException:
            session.close()
            raise CandidateEvaluationInfrastructureError(
                "gemini_transport_failure"
            ) from None
        try:
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    if int(declared) > max_response_bytes:
                        raise CandidateEvaluationInfrastructureError(
                            "gemini_response_too_large"
                        )
                except ValueError:
                    pass
            body = bytearray()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    body.extend(chunk)
                    if len(body) > max_response_bytes:
                        raise CandidateEvaluationInfrastructureError(
                            "gemini_response_too_large"
                        )
            return GeminiTransportResponse(response.status_code, bytes(body))
        except requests.RequestException:
            raise CandidateEvaluationInfrastructureError(
                "gemini_transport_failure"
            ) from None
        finally:
            response.close()
            session.close()


def _gemini_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    def convert(item: Any) -> Any:
        if isinstance(item, list):
            return [convert(value) for value in item]
        if not isinstance(item, dict):
            return item
        result = {}
        for key, value in item.items():
            if key in {"default", "discriminator", "title"}:
                continue
            if key == "const":
                result["enum"] = [convert(value)]
            else:
                result["anyOf" if key == "oneOf" else key] = convert(value)
        return result

    result = convert(dict(value))
    if not isinstance(result, dict) or result.get("type") != "object":
        raise CandidateEvaluationInfrastructureError("operational_tool_schema_invalid")
    return result


def _candidate_payload(request: CandidateInput) -> dict[str, Any]:
    if len(request.tools) != 1 or request.tools[0].name != TOOL_NAME:
        raise CandidateEvaluationInfrastructureError("candidate_tool_catalog_invalid")
    tool = request.tools[0]
    visible = json.dumps(
        {
            "authorization": request.authorization.model_dump(mode="json"),
            "question": request.question,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return {
        "model": GEMINI_MODEL,
        "input": visible,
        "system_instruction": _STATIC_INSTRUCTIONS,
        "tools": [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": _gemini_schema(tool.input_schema),
            }
        ],
        "store": False,
    }


def _failure_code(status: int) -> str:
    if status in {401, 403}:
        return "gemini_authentication_failure"
    if status == 429:
        return "gemini_rate_limit_failure"
    if status >= 500:
        return "gemini_service_failure"
    return "gemini_request_rejected"


def _parse_selection(body: bytes) -> object:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _INVALID_OUTPUT
    if not isinstance(payload, dict):
        return _INVALID_OUTPUT
    if payload.get("object") != "interaction" or payload.get("model") != GEMINI_MODEL:
        raise CandidateEvaluationInfrastructureError("gemini_response_mismatch")
    status = payload.get("status")
    if status not in {"completed", "requires_action"}:
        raise CandidateEvaluationInfrastructureError("gemini_response_incomplete")
    if not isinstance(payload.get("steps"), list) or len(payload["steps"]) > 32:
        return _INVALID_OUTPUT
    calls = []
    model_output_count = 0
    for step in payload["steps"]:
        if not isinstance(step, dict):
            return _INVALID_OUTPUT
        step_type = step.get("type")
        if step_type == "function_call":
            calls.append(step)
        elif step_type == "model_output":
            model_output_count += 1
        elif step_type != "thought":
            return _INVALID_OUTPUT
    if not calls:
        if status != "completed" or model_output_count < 1:
            raise CandidateEvaluationInfrastructureError("gemini_response_mismatch")
        return None
    if len(calls) != 1:
        return _INVALID_OUTPUT
    if status != "requires_action" or model_output_count:
        raise CandidateEvaluationInfrastructureError("gemini_response_mismatch")
    name, arguments = calls[0].get("name"), calls[0].get("arguments")
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 128
        or not name.isprintable()
        or not isinstance(arguments, dict)
    ):
        return _INVALID_OUTPUT
    if len(json.dumps(arguments, separators=(",", ":")).encode()) > _MAX_ARGUMENT_BYTES:
        return _INVALID_OUTPUT
    return OperationalToolSelection(tool_name=name, arguments=arguments)


class GeminiInteractionsCandidateSelector:
    __slots__ = ("_api_key", "_calls_completed", "_metadata", "_transport")

    def __init__(
        self,
        *,
        api_key: str,
        metadata: GeminiCandidateMetadata,
        transport: GeminiTransport | None = None,
    ) -> None:
        if (
            not isinstance(api_key, str)
            or not api_key
            or api_key.strip() != api_key
            or not api_key.isprintable()
        ):
            raise CandidateEvaluationInputError("GEMINI_API_KEY is unavailable")
        self._api_key, self._metadata, self._transport, self._calls_completed = (
            api_key,
            metadata,
            transport or RequestsGeminiTransport(),
            0,
        )

    def __repr__(self) -> str:
        return f"GeminiInteractionsCandidateSelector(model={GEMINI_MODEL!r}, calls_completed={self._calls_completed})"

    @property
    def calls_completed(self) -> int:
        return self._calls_completed

    def select(self, request: CandidateInput) -> object:
        if self._calls_completed >= EXPECTED_CASE_COUNT:
            raise CandidateEvaluationInfrastructureError("gemini_call_limit_exceeded")
        self._calls_completed += 1
        response = self._transport.send(
            endpoint=GEMINI_ENDPOINT,
            api_key=self._api_key,
            payload=_candidate_payload(request),
            timeout_seconds=self._metadata.selector_timeout_seconds,
            max_response_bytes=self._metadata.max_response_bytes,
        )
        if response.status_code != 200:
            raise CandidateEvaluationInfrastructureError(
                _failure_code(response.status_code)
            )
        if len(response.body) > GEMINI_MAX_RESPONSE_BYTES:
            raise CandidateEvaluationInfrastructureError("gemini_response_too_large")
        return _parse_selection(response.body)


def build_gemini_candidate_evaluation_receipt(
    run: CandidateEvaluationRun,
    metadata: GeminiCandidateMetadata,
    *,
    calls_completed: int,
    source_commit: str,
    evaluated_at_utc: datetime,
) -> GeminiCandidateEvaluationReceipt:
    if run.report.status != "passed" or calls_completed != EXPECTED_CASE_COUNT:
        raise CandidateEvaluationInputError(
            "a remote receipt requires a passed 88-call evaluation"
        )
    config = strict_model_dump(metadata)
    config_bytes = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    report_hash = hashlib.sha256(sanitized_report_json(run.report).encode()).hexdigest()
    return GeminiCandidateEvaluationReceipt(
        **{k: v for k, v in config.items() if k != "endpoint"},
        calls_completed=calls_completed,
        dataset_id=run.report.dataset_id,
        dataset_version=run.report.dataset_version,
        dataset_sha256=run.report.dataset_sha256,
        response_set_sha256=run.response_set_sha256,
        report_sha256=report_hash,
        candidate_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        source_commit=source_commit,
        metrics=run.report.metrics,
        evaluated_at_utc=evaluated_at_utc,
    )


def write_gemini_candidate_receipt(
    path: Path, receipt: GeminiCandidateEvaluationReceipt
) -> None:
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
