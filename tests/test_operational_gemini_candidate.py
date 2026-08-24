from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
import requests

import scripts.evaluate_gemini_operational_copilot_candidate as cli
from scripts.evaluate_gemini_operational_copilot_candidate import main
from wind_forecast.operational_candidate_evaluation import (
    CandidateEvaluationInfrastructureError,
    CandidateEvaluationInputError,
    CandidateInput,
    run_candidate_evaluation,
)
from wind_forecast.operational_copilot import allowed_operational_tools
from wind_forecast.operational_evaluation import load_evaluation_dataset
from wind_forecast.operational_gemini_candidate import (
    GeminiInteractionsCandidateSelector,
    GeminiTransportResponse,
    RequestsGeminiTransport,
    build_gemini_candidate_evaluation_receipt,
    write_gemini_candidate_receipt,
)
from wind_forecast.operational_gemini_candidate_models import (
    GEMINI_ENDPOINT,
    GEMINI_MAX_RESPONSE_BYTES,
    GEMINI_MODEL,
    GEMINI_RECEIPT_SCHEMA_VERSION,
    GeminiCandidateMetadata,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation/operational_read_only_copilot/v1/manifest.json"
SECRET = "gemini-test-secret"
NOW = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


def dataset():
    return load_evaluation_dataset(MANIFEST)


def metadata():
    return GeminiCandidateMetadata(candidate_id="gemini-2-5-flash-lite-en-v1")


def candidate_input(index=0):
    case = dataset().cases[index]
    return CandidateInput(
        question=case.question,
        authorization=case.authorization,
        tools=allowed_operational_tools(),
    )


def body(steps):
    status = "requires_action" if any(s.get("type") == "function_call" for s in steps) else "completed"
    return json.dumps(
        {
            "model": GEMINI_MODEL,
            "object": "interaction",
            "status": status,
            "steps": steps,
        }
    ).encode()


class RecordingTransport:
    def __init__(self, response):
        self.response, self.calls = response, []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class DatasetTransport:
    def __init__(self, data):
        self.expected = {c.question: c.expected_tool for c in data.cases}
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        visible = json.loads(kwargs["payload"]["input"])
        expected = self.expected[visible["question"]]
        steps = (
            [{"type": "model_output", "content": [{"type": "text", "text": "refused"}]}]
            if expected is None
            else [
                {
                    "type": "function_call",
                    "name": expected.name,
                    "arguments": expected.arguments.model_dump(mode="json"),
                }
            ]
        )
        return GeminiTransportResponse(200, body(steps))


def selector(transport):
    return GeminiInteractionsCandidateSelector(
        api_key=SECRET, metadata=metadata(), transport=transport
    )


def test_payload_and_fixed_boundary():
    expected = dataset().cases[0].expected_tool
    transport = RecordingTransport(
        GeminiTransportResponse(
            200,
            body(
                [
                    {
                        "type": "function_call",
                        "name": expected.name,
                        "arguments": expected.arguments.model_dump(mode="json"),
                    }
                ]
            ),
        )
    )
    result = selector(transport).select(candidate_input())
    assert result.tool_name == "operational_query"
    call = transport.calls[0]
    assert call["endpoint"] == GEMINI_ENDPOINT and call["timeout_seconds"] == 5.0
    payload = call["payload"]
    assert payload["model"] == GEMINI_MODEL and payload["store"] is False
    assert set(json.loads(payload["input"])) == {"authorization", "question"}
    serialized = json.dumps(payload)
    assert all(
        term not in serialized
        for term in ("oracle_id", "expected_tool", "evidence_scenario", SECRET)
    )


def test_abstention_and_malformed_fail_closed():
    assert (
        selector(
            RecordingTransport(
                GeminiTransportResponse(
                    200,
                    body(
                        [
                            {
                                "type": "model_output",
                                "content": [{"type": "text", "text": "refused"}],
                            }
                        ]
                    ),
                )
            )
        ).select(candidate_input())
        is None
    )
    result = selector(
        RecordingTransport(
            GeminiTransportResponse(
                200, body([{"type": "function_call"}, {"type": "function_call"}])
            )
        )
    ).select(candidate_input())
    assert result is not None and getattr(result, "tool_name", None) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"model": GEMINI_MODEL, "object": "interaction", "status": "incomplete", "steps": []},
        {"model": "wrong-model", "object": "interaction", "status": "completed", "steps": []},
        {"model": GEMINI_MODEL, "object": "interaction", "status": "completed", "steps": [{"type": "function_call", "name": "operational_query", "arguments": {}}]},
        {"model": GEMINI_MODEL, "object": "interaction", "status": "completed", "steps": []},
        {"model": GEMINI_MODEL, "object": "interaction", "status": "requires_action", "steps": [{"type": "function_call", "name": "operational_query", "arguments": {}}, {"type": "model_output", "content": []}]},
    ],
)
def test_real_interaction_envelope_mismatches_fail_closed(payload):
    with pytest.raises(CandidateEvaluationInfrastructureError):
        selector(
            RecordingTransport(
                GeminiTransportResponse(200, json.dumps(payload).encode())
            )
        ).select(candidate_input())


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "gemini_request_rejected"),
        (401, "gemini_authentication_failure"),
        (403, "gemini_authentication_failure"),
        (429, "gemini_rate_limit_failure"),
        (500, "gemini_service_failure"),
    ],
)
def test_http_errors_sanitized(status, code):
    with pytest.raises(CandidateEvaluationInfrastructureError, match=code) as exc:
        selector(
            RecordingTransport(GeminiTransportResponse(status, b"private"))
        ).select(candidate_input())
    assert SECRET not in str(exc.value)


def test_transport_security_and_bounds(monkeypatch):
    class Response:
        status_code = 200
        headers = {"Content-Length": "2"}

        def iter_content(self, *, chunk_size):
            assert chunk_size == 8192
            yield b"{}"

        def close(self):
            self.closed = True

    class Session:
        trust_env = True

        def post(self, endpoint, **kwargs):
            assert self.trust_env is False and endpoint == GEMINI_ENDPOINT
            assert kwargs["headers"]["x-goog-api-key"] == SECRET
            assert kwargs["allow_redirects"] is False and kwargs["stream"] is True
            return response

        def close(self):
            self.closed = True

    response, session = Response(), Session()
    monkeypatch.setattr(requests, "Session", lambda: session)
    assert (
        RequestsGeminiTransport()
        .send(
            endpoint=GEMINI_ENDPOINT,
            api_key=SECRET,
            payload={"store": False},
            timeout_seconds=5.0,
            max_response_bytes=GEMINI_MAX_RESPONSE_BYTES,
        )
        .body
        == b"{}"
    )
    assert response.closed and session.closed


def test_transport_sanitizes_session_construction_failure(monkeypatch):
    monkeypatch.setattr(requests, "Session", lambda: (_ for _ in ()).throw(RuntimeError(SECRET)))
    with pytest.raises(CandidateEvaluationInfrastructureError, match="gemini_transport_failure") as exc:
        RequestsGeminiTransport().send(
            endpoint=GEMINI_ENDPOINT,
            api_key=SECRET,
            payload={"store": False},
            timeout_seconds=5.0,
            max_response_bytes=GEMINI_MAX_RESPONSE_BYTES,
        )
    assert SECRET not in str(exc.value)


def test_88_case_replay_receipt_and_limit(tmp_path):
    data, transport, meta = dataset(), DatasetTransport(dataset()), metadata()
    selected = selector(transport)
    run = run_candidate_evaluation(data, selected, meta, evaluated_at_utc=NOW)
    assert run.report.status == "passed" and selected.calls_completed == 88
    receipt = build_gemini_candidate_evaluation_receipt(
        run, meta, calls_completed=88, source_commit="a" * 40, evaluated_at_utc=NOW
    )
    payload = receipt.model_dump(mode="json")
    assert (
        payload["schema_version"] == GEMINI_RECEIPT_SCHEMA_VERSION
        and payload["store"] is False
    )
    assert SECRET not in json.dumps(payload) and "question" not in json.dumps(payload)
    path = tmp_path / "receipt.json"
    write_gemini_candidate_receipt(path, receipt)
    with pytest.raises(CandidateEvaluationInputError):
        write_gemini_candidate_receipt(path, receipt)
    with pytest.raises(
        CandidateEvaluationInfrastructureError, match="gemini_call_limit_exceeded"
    ):
        selected.select(candidate_input())


def test_cli_requires_confirmation_and_pinned_dataset(tmp_path, capsys, monkeypatch):
    args = [
        "--dataset",
        str(MANIFEST),
        "--receipt-out",
        str(tmp_path / "r.json"),
        "--candidate-id",
        "v1",
        "--source-commit",
        "a" * 40,
    ]
    assert main(args, environ={"GEMINI_API_KEY": SECRET}) == 2
    monkeypatch.setattr(cli, "_checkout_state", lambda: ("a" * 40, True))
    substituted = replace(dataset(), dataset_sha256="0" * 64)
    monkeypatch.setattr(cli, "load_evaluation_dataset", lambda _: substituted)
    transport = DatasetTransport(substituted)
    assert (
        main(
            args + ["--confirm-synthetic-egress"],
            environ={"GEMINI_API_KEY": SECRET},
            transport=transport,
        )
        == 2
    )
    assert not transport.calls and SECRET not in capsys.readouterr().out


def test_cli_success_and_failure_do_not_retain_provider_payloads(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_checkout_state", lambda: ("a" * 40, True))
    args = [
        "--dataset",
        str(MANIFEST),
        "--receipt-out",
        str(tmp_path / "r.json"),
        "--candidate-id",
        "v1",
        "--source-commit",
        "a" * 40,
        "--evaluated-at-utc",
        "2026-08-24T12:00:00Z",
        "--confirm-synthetic-egress",
    ]
    assert (
        main(
            args,
            environ={"GEMINI_API_KEY": SECRET},
            transport=DatasetTransport(dataset()),
        )
        == 0
    )
    assert (tmp_path / "r.json").exists() and SECRET not in capsys.readouterr().out
    failed = tmp_path / "failed.json"
    args[3] = str(failed)
    transport = RecordingTransport(GeminiTransportResponse(429, b"private"))
    assert main(args, environ={"GEMINI_API_KEY": SECRET}, transport=transport) == 2
    assert not failed.exists() and len(transport.calls) == 1


def test_cli_rejects_mismatched_or_dirty_checkout(tmp_path, monkeypatch):
    args = [
        "--dataset", str(MANIFEST), "--receipt-out", str(tmp_path / "r.json"),
        "--candidate-id", "v1", "--source-commit", "a" * 40,
        "--confirm-synthetic-egress",
    ]
    monkeypatch.setattr(cli, "_checkout_state", lambda: ("b" * 40, True))
    assert main(args, environ={"GEMINI_API_KEY": SECRET}) == 2
    monkeypatch.setattr(cli, "_checkout_state", lambda: ("a" * 40, False))
    assert main(args, environ={"GEMINI_API_KEY": SECRET}) == 2


@pytest.mark.parametrize("key", ["", " padded", "padded ", "line\nbreak"])
def test_unsafe_keys_rejected(key):
    with pytest.raises(CandidateEvaluationInputError):
        GeminiInteractionsCandidateSelector(
            api_key=key, metadata=metadata(), transport=DatasetTransport(dataset())
        )
