from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest
import requests

from scripts import evaluate_openai_operational_copilot_candidate as openai_cli
from scripts.evaluate_openai_operational_copilot_candidate import main as openai_main
from wind_forecast.operational_candidate_evaluation import (
    CandidateEvaluationInfrastructureError,
    CandidateEvaluationInputError,
    CandidateInput,
    run_candidate_evaluation,
)
from wind_forecast.operational_copilot_models import allowed_operational_tools
from wind_forecast.operational_evaluation import load_evaluation_dataset
from wind_forecast.operational_openai_candidate import (
    OpenAIResponsesCandidateSelector,
    OpenAITransportResponse,
    RequestsOpenAITransport,
    build_openai_candidate_evaluation_receipt,
    write_openai_candidate_receipt,
)
from wind_forecast.operational_openai_candidate_models import (
    OPENAI_ENDPOINT,
    OPENAI_MAX_RESPONSE_BYTES,
    OPENAI_MODEL,
    OPENAI_RECEIPT_SCHEMA_VERSION,
    OpenAICandidateMetadata,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation/operational_read_only_copilot/v1/manifest.json"
NOW = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)
SECRET = "sk-test-secret-value"


def _dataset():
    return load_evaluation_dataset(MANIFEST)


def _metadata() -> OpenAICandidateMetadata:
    return OpenAICandidateMetadata(
        candidate_id="openai-gpt-5-4-mini-2026-03-17-en-v1"
    )


def _candidate_input(case_index: int = 0) -> CandidateInput:
    case = _dataset().cases[case_index]
    return CandidateInput(
        question=case.question,
        authorization=case.authorization,
        tools=allowed_operational_tools(),
    )


def _provider_body(output: list[dict[str, Any]], **extra: Any) -> bytes:
    return json.dumps(
        {
            "model": OPENAI_MODEL,
            "output": output,
            "status": "completed",
            **extra,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class RecordingTransport:
    def __init__(self, response: OpenAITransportResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def send(
        self,
        *,
        endpoint: str,
        api_key: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> OpenAITransportResponse:
        self.calls.append(
            {
                "endpoint": endpoint,
                "api_key": api_key,
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
            }
        )
        return self.response


class DatasetTransport:
    def __init__(self, dataset) -> None:
        self.expected = {
            case.question: case.expected_tool for case in dataset.cases
        }
        self.calls: list[dict[str, Any]] = []

    def send(
        self,
        *,
        endpoint: str,
        api_key: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> OpenAITransportResponse:
        self.calls.append(dict(payload))
        visible = json.loads(payload["input"][0]["content"][0]["text"])
        expected = self.expected[visible["question"]]
        if expected is None:
            output = [
                {
                    "content": [{"text": "refused", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
        else:
            output = [
                {
                    "arguments": json.dumps(
                        expected.arguments.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "name": expected.name,
                    "status": "completed",
                    "type": "function_call",
                }
            ]
        return OpenAITransportResponse(
            status_code=200,
            body=_provider_body(output),
        )


def test_request_contains_only_approved_candidate_input_and_fixed_policy() -> None:
    case = _dataset().cases[0]
    expected = case.expected_tool
    assert expected is not None
    transport = RecordingTransport(
        OpenAITransportResponse(
            status_code=200,
            body=_provider_body(
                [
                    {
                        "arguments": json.dumps(
                            expected.arguments.model_dump(mode="json")
                        ),
                        "name": expected.name,
                        "type": "function_call",
                    }
                ]
            ),
        )
    )
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=_metadata(),
        transport=transport,
    )

    selection = selector.select(_candidate_input())

    assert selection.tool_name == "operational_query"
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["endpoint"] == OPENAI_ENDPOINT
    assert call["timeout_seconds"] == 5.0
    assert call["max_response_bytes"] == OPENAI_MAX_RESPONSE_BYTES
    payload = call["payload"]
    assert payload["model"] == OPENAI_MODEL
    assert payload["store"] is False
    assert payload["parallel_tool_calls"] is False
    assert payload["tool_choice"] == "auto"
    assert payload["reasoning"] == {"effort": "none"}
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["name"] == "operational_query"
    assert payload["tools"][0]["strict"] is True
    visible = json.loads(payload["input"][0]["content"][0]["text"])
    assert set(visible) == {"authorization", "question"}
    serialized = json.dumps(payload, sort_keys=True)
    assert "oracle_id" not in serialized
    assert "expected_tool" not in serialized
    assert "evidence_scenario" not in serialized
    assert SECRET not in serialized
    assert SECRET not in repr(selector)


def test_openai_strict_schema_requires_nullable_public_fields() -> None:
    transport = DatasetTransport(_dataset())
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=_metadata(),
        transport=transport,
    )
    selector.select(_candidate_input())
    schema = transport.calls[0]["tools"][0]["parameters"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "window_days" in schema["required"]
    assert "pagination" in schema["required"]
    assert "oneOf" not in json.dumps(schema)
    assert "discriminator" not in json.dumps(schema)


def test_no_function_call_is_a_safe_abstention() -> None:
    transport = RecordingTransport(
        OpenAITransportResponse(
            status_code=200,
            body=_provider_body(
                [{"content": [], "role": "assistant", "type": "message"}]
            ),
        )
    )
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=_metadata(),
        transport=transport,
    )
    assert selector.select(_candidate_input()) is None


@pytest.mark.parametrize(
    "output",
    [
        [
            {"arguments": "{}", "name": "operational_query", "type": "function_call"},
            {"arguments": "{}", "name": "operational_query", "type": "function_call"},
        ],
        [{"arguments": "not-json", "name": "operational_query", "type": "function_call"}],
        [{"arguments": "[]", "name": "operational_query", "type": "function_call"}],
        [{"type": "web_search_call"}],
    ],
)
def test_malformed_or_multiple_provider_outputs_fail_closed(output) -> None:
    transport = RecordingTransport(
        OpenAITransportResponse(status_code=200, body=_provider_body(output))
    )
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=_metadata(),
        transport=transport,
    )
    result = selector.select(_candidate_input())
    assert result is not None
    assert getattr(result, "tool_name", None) is None


@pytest.mark.parametrize(
    ("status_code", "failure_code"),
    [
        (400, "openai_request_rejected"),
        (401, "openai_authentication_failure"),
        (403, "openai_authentication_failure"),
        (429, "openai_rate_limit_failure"),
        (500, "openai_service_failure"),
    ],
)
def test_http_failures_are_sanitized_and_fatal(status_code, failure_code) -> None:
    transport = RecordingTransport(
        OpenAITransportResponse(status_code=status_code, body=b"private-body")
    )
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=_metadata(),
        transport=transport,
    )
    with pytest.raises(CandidateEvaluationInfrastructureError) as exc_info:
        selector.select(_candidate_input())
    assert str(exc_info.value) == failure_code
    assert "private-body" not in str(exc_info.value)
    assert SECRET not in str(exc_info.value)
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        _provider_body([], status="incomplete"),
        _provider_body([], model="unexpected-model"),
    ],
)
def test_invalid_envelope_or_model_does_not_leak_payload(body: bytes) -> None:
    transport = RecordingTransport(
        OpenAITransportResponse(status_code=200, body=body)
    )
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=_metadata(),
        transport=transport,
    )
    if body == b"not-json":
        assert selector.select(_candidate_input()) is not None
    else:
        with pytest.raises(CandidateEvaluationInfrastructureError):
            selector.select(_candidate_input())


def test_response_size_limit_is_fatal() -> None:
    transport = RecordingTransport(
        OpenAITransportResponse(
            status_code=200,
            body=b"x" * (OPENAI_MAX_RESPONSE_BYTES + 1),
        )
    )
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=_metadata(),
        transport=transport,
    )
    with pytest.raises(
        CandidateEvaluationInfrastructureError,
        match="openai_response_too_large",
    ):
        selector.select(_candidate_input())


def test_requests_transport_disables_environment_proxies_and_closes(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"Content-Length": "2"}
        closed = False

        def iter_content(self, *, chunk_size: int):
            assert chunk_size == 8192
            yield b"{}"

        def close(self) -> None:
            self.closed = True

    class FakeSession:
        trust_env = True
        closed = False

        def post(self, endpoint, **kwargs):
            assert self.trust_env is False
            assert endpoint == OPENAI_ENDPOINT
            assert kwargs["headers"]["Authorization"] == f"Bearer {SECRET}"
            assert kwargs["timeout"] == 5.0
            assert kwargs["stream"] is True
            assert kwargs["allow_redirects"] is False
            return fake_response

        def close(self) -> None:
            self.closed = True

    fake_response = FakeResponse()
    fake_session = FakeSession()
    monkeypatch.setattr(requests, "Session", lambda: fake_session)

    result = RequestsOpenAITransport().send(
        endpoint=OPENAI_ENDPOINT,
        api_key=SECRET,
        payload={"store": False},
        timeout_seconds=5.0,
        max_response_bytes=OPENAI_MAX_RESPONSE_BYTES,
    )

    assert result.body == b"{}"
    assert fake_response.closed is True
    assert fake_session.closed is True


def test_requests_transport_sanitizes_timeout_and_closes_session(monkeypatch) -> None:
    class TimeoutSession:
        trust_env = True
        closed = False

        def post(self, _endpoint, **_kwargs):
            raise requests.Timeout(f"private {SECRET}")

        def close(self) -> None:
            self.closed = True

    session = TimeoutSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    with pytest.raises(
        CandidateEvaluationInfrastructureError,
        match="openai_transport_failure",
    ) as exc_info:
        RequestsOpenAITransport().send(
            endpoint=OPENAI_ENDPOINT,
            api_key=SECRET,
            payload={"store": False},
            timeout_seconds=5.0,
            max_response_bytes=OPENAI_MAX_RESPONSE_BYTES,
        )
    assert SECRET not in str(exc_info.value)
    assert session.closed is True


def test_remote_candidate_replays_all_88_cases_and_builds_sanitized_receipt(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    transport = DatasetTransport(dataset)
    metadata = _metadata()
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=metadata,
        transport=transport,
    )

    run = run_candidate_evaluation(
        dataset,
        selector,
        metadata,
        evaluated_at_utc=NOW,
    )

    assert run.report.status == "passed"
    assert selector.calls_completed == 88
    assert len(transport.calls) == 88
    receipt = build_openai_candidate_evaluation_receipt(
        run,
        metadata,
        calls_completed=selector.calls_completed,
        source_commit="a" * 40,
        evaluated_at_utc=NOW,
    )
    payload = receipt.model_dump(mode="json")
    assert payload["schema_version"] == OPENAI_RECEIPT_SCHEMA_VERSION
    assert payload["model"] == OPENAI_MODEL
    assert payload["egress_allowed"] is True
    assert payload["store"] is False
    assert payload["calls_completed"] == 88
    serialized = json.dumps(payload, sort_keys=True)
    assert SECRET not in serialized
    assert "question" not in serialized
    assert "facts" not in serialized
    receipt_path = tmp_path / "receipt.json"
    write_openai_candidate_receipt(receipt_path, receipt)
    with pytest.raises(CandidateEvaluationInputError):
        write_openai_candidate_receipt(receipt_path, receipt)


def test_remote_receipt_requires_all_calls() -> None:
    dataset = _dataset()
    transport = DatasetTransport(dataset)
    metadata = _metadata()
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=metadata,
        transport=transport,
    )
    run = run_candidate_evaluation(
        dataset,
        selector,
        metadata,
        evaluated_at_utc=NOW,
    )
    with pytest.raises(CandidateEvaluationInputError):
        build_openai_candidate_evaluation_receipt(
            run,
            metadata,
            calls_completed=87,
            source_commit="a" * 40,
            evaluated_at_utc=NOW,
        )


def test_remote_candidate_refuses_more_than_88_calls() -> None:
    dataset = _dataset()
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=_metadata(),
        transport=DatasetTransport(dataset),
    )
    for case in dataset.cases:
        selector.select(
            CandidateInput(
                question=case.question,
                authorization=case.authorization,
                tools=allowed_operational_tools(),
            )
        )

    with pytest.raises(
        CandidateEvaluationInfrastructureError,
        match="openai_call_limit_exceeded",
    ):
        selector.select(_candidate_input())


def test_failed_transport_attempts_still_consume_the_88_call_limit() -> None:
    class FailingTransport:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, **_kwargs) -> OpenAITransportResponse:
            self.calls += 1
            raise CandidateEvaluationInfrastructureError("transport_failed")

    transport = FailingTransport()
    selector = OpenAIResponsesCandidateSelector(
        api_key=SECRET,
        metadata=_metadata(),
        transport=transport,
    )
    for _ in range(88):
        with pytest.raises(CandidateEvaluationInfrastructureError):
            selector.select(_candidate_input())

    with pytest.raises(
        CandidateEvaluationInfrastructureError,
        match="openai_call_limit_exceeded",
    ):
        selector.select(_candidate_input())
    assert selector.calls_completed == 88
    assert transport.calls == 88


def test_cli_requires_explicit_egress_and_never_prints_the_key(
    tmp_path: Path,
    capsys,
) -> None:
    result = openai_main(
        [
            "--dataset",
            str(MANIFEST),
            "--receipt-out",
            str(tmp_path / "receipt.json"),
            "--candidate-id",
            "candidate-v1",
            "--source-commit",
            "a" * 40,
        ],
        environ={"OPENAI_API_KEY": SECRET},
    )
    assert result == 2
    assert "synthetic_egress_not_confirmed" in capsys.readouterr().out


def test_cli_rejects_dataset_that_does_not_match_sealed_digest(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    substituted = replace(_dataset(), dataset_sha256="0" * 64)
    monkeypatch.setattr(openai_cli, "load_evaluation_dataset", lambda _path: substituted)
    transport = DatasetTransport(substituted)

    result = openai_main(
        [
            "--dataset",
            str(MANIFEST),
            "--receipt-out",
            str(tmp_path / "receipt.json"),
            "--candidate-id",
            "candidate-v1",
            "--source-commit",
            "a" * 40,
            "--confirm-synthetic-egress",
        ],
        environ={"OPENAI_API_KEY": SECRET},
        transport=transport,
    )

    assert result == 2
    assert not transport.calls
    assert "candidate_evaluation_input_invalid" in capsys.readouterr().out


def test_cli_evaluates_in_memory_and_writes_only_a_passed_receipt(
    tmp_path: Path,
    capsys,
) -> None:
    dataset = _dataset()
    transport = DatasetTransport(dataset)
    receipt_path = tmp_path / "receipt.json"
    result = openai_main(
        [
            "--dataset",
            str(MANIFEST),
            "--receipt-out",
            str(receipt_path),
            "--candidate-id",
            "candidate-v1",
            "--source-commit",
            "a" * 40,
            "--evaluated-at-utc",
            "2026-08-24T12:00:00Z",
            "--confirm-synthetic-egress",
        ],
        environ={"OPENAI_API_KEY": SECRET},
        transport=transport,
    )
    stdout = capsys.readouterr().out
    assert result == 0
    assert receipt_path.exists()
    assert len(transport.calls) == 88
    assert SECRET not in stdout
    assert dataset.cases[0].question not in stdout
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["store"] is False


def test_cli_aborts_after_first_infrastructure_failure_without_receipt(
    tmp_path: Path,
    capsys,
) -> None:
    transport = RecordingTransport(
        OpenAITransportResponse(status_code=429, body=b"private")
    )
    receipt_path = tmp_path / "receipt.json"
    result = openai_main(
        [
            "--dataset",
            str(MANIFEST),
            "--receipt-out",
            str(receipt_path),
            "--candidate-id",
            "candidate-v1",
            "--source-commit",
            "a" * 40,
            "--confirm-synthetic-egress",
        ],
        environ={"OPENAI_API_KEY": SECRET},
        transport=transport,
    )
    stdout = capsys.readouterr().out
    assert result == 2
    assert len(transport.calls) == 1
    assert not receipt_path.exists()
    assert "candidate_infrastructure_failure" in stdout
    assert SECRET not in stdout


@pytest.mark.parametrize("api_key", ["", " padded", "padded ", "line\nbreak"])
def test_selector_rejects_missing_or_unsafe_api_keys(api_key: str) -> None:
    with pytest.raises(CandidateEvaluationInputError):
        OpenAIResponsesCandidateSelector(
            api_key=api_key,
            metadata=_metadata(),
            transport=DatasetTransport(_dataset()),
        )
