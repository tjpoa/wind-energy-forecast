from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate_operational_copilot_candidate import main as evaluation_main
from wind_forecast.operational_candidate_evaluation import (
    CandidateEvaluationInputError,
    CandidateInput,
    build_candidate_evaluation_receipt,
    run_candidate_evaluation,
    serialize_candidate_traces,
    write_candidate_receipt,
)
from wind_forecast.operational_candidate_evaluation_models import (
    CandidateEvaluationReceipt,
    CandidateMetadata,
)
from wind_forecast.operational_evaluation import load_evaluation_dataset
from wind_forecast.operational_copilot_models import (
    OperationalToolSelection,
    TOOL_NAME,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation/operational_read_only_copilot/v1/manifest.json"
NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


def _metadata() -> CandidateMetadata:
    return CandidateMetadata(
        candidate_id="fixture-candidate-v1",
        provider="local-fixture",
        model="selector-fixture-v1",
    )


class PerfectFixture:
    def __init__(self, dataset) -> None:
        self._expected = {
            case.question: case.expected_tool for case in dataset.cases
        }
        self.requests: list[CandidateInput] = []

    def select(self, request: CandidateInput) -> object:
        self.requests.append(request)
        expected = self._expected[request.question]
        if expected is None:
            return None
        return OperationalToolSelection(
            tool_name=expected.name,
            arguments=expected.arguments.model_dump(mode="python"),
        )


def test_candidate_view_contains_only_question_authorization_and_tool_schema() -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    candidate = PerfectFixture(dataset)

    run = run_candidate_evaluation(
        dataset,
        candidate,
        _metadata(),
        evaluated_at_utc=NOW,
        monotonic_clock=lambda: 0.0,
    )

    assert run.report.status == "passed"
    assert run.report.metrics.passed_case_count == 88
    assert len(candidate.requests) == 88
    assert set(candidate.requests[0].model_dump()) == {
        "question",
        "authorization",
        "tools",
    }
    assert all(len(request.tools) == 1 for request in candidate.requests)
    assert all(request.tools[0].name == TOOL_NAME for request in candidate.requests)
    assert all(
        "oracle_id" not in request.model_dump()
        and "expected_tool" not in request.model_dump()
        and "evidence_scenario" not in request.model_dump()
        for request in candidate.requests
    )


def test_runner_produces_normalized_traces_and_digest_without_store_access() -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    candidate = PerfectFixture(dataset)
    run = run_candidate_evaluation(
        dataset,
        candidate,
        _metadata(),
        evaluated_at_utc=NOW,
        monotonic_clock=lambda: 0.0,
    )

    serialized = serialize_candidate_traces(run.traces)
    assert len(run.traces) == 88
    assert len(serialized) > 0
    assert run.response_set_sha256 == hashlib.sha256(serialized).hexdigest()
    assert all(trace.case_id == f"eval-{index:03d}" for index, trace in enumerate(run.traces, 1))


def test_wrong_arguments_fail_the_existing_harness() -> None:
    dataset = load_evaluation_dataset(MANIFEST)

    class WrongArguments(PerfectFixture):
        def select(self, request: CandidateInput) -> object:
            selection = super().select(request)
            if isinstance(selection, OperationalToolSelection):
                arguments = selection.arguments.copy()
                arguments["selector"] = {"kind": "latest"}
                return OperationalToolSelection(
                    tool_name=selection.tool_name,
                    arguments=arguments,
                )
            return selection

    run = run_candidate_evaluation(
        dataset,
        WrongArguments(dataset),
        _metadata(),
        evaluated_at_utc=NOW,
        monotonic_clock=lambda: 0.0,
    )
    assert run.report.status == "failed"
    assert run.report.metrics.critical_failure_count > 0


@pytest.mark.parametrize(
    "candidate_factory",
    [
        lambda _dataset: type("UnknownTool", (), {
            "select": lambda _self, _request: {
                "tool_name": "unknown_tool",
                "arguments": {},
            }
        })(),
        lambda _dataset: type("InvalidOutput", (), {
            "select": lambda _self, _request: {
                "tool_name": TOOL_NAME,
                "arguments": {},
                "answer": "forbidden",
            }
        })(),
        lambda _dataset: type("Raises", (), {
            "select": lambda _self, _request: (_ for _ in ()).throw(
                RuntimeError("candidate failure")
            )
        })(),
    ],
)
def test_invalid_candidate_behaviour_fails_closed(candidate_factory) -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    run = run_candidate_evaluation(
        dataset,
        candidate_factory(dataset),
        _metadata(),
        evaluated_at_utc=NOW,
        monotonic_clock=lambda: 0.0,
    )
    assert run.report.status == "failed"
    assert run.report.metrics.critical_failure_count > 0


def test_receipt_is_additive_and_contains_only_digests_metadata_and_metrics(
    tmp_path: Path,
) -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    run = run_candidate_evaluation(
        dataset,
        PerfectFixture(dataset),
        _metadata(),
        evaluated_at_utc=NOW,
        monotonic_clock=lambda: 0.0,
    )
    receipt = build_candidate_evaluation_receipt(
        run,
        _metadata(),
        source_commit="a" * 40,
        evaluated_at_utc=NOW,
    )
    assert isinstance(receipt, CandidateEvaluationReceipt)
    assert receipt.acceptance_state == "candidate evaluated; Copilot disabled by default"
    payload = receipt.model_dump(mode="json")
    assert set(payload) >= {
        "candidate_id",
        "provider",
        "model",
        "dataset_sha256",
        "response_set_sha256",
        "report_sha256",
        "candidate_config_sha256",
        "source_commit",
        "metrics",
    }
    assert "question" not in json.dumps(payload)
    assert "facts" not in json.dumps(payload)

    receipt_path = tmp_path / "receipts" / "candidate.json"
    write_candidate_receipt(receipt_path, receipt)
    assert receipt_path.exists()
    with pytest.raises(CandidateEvaluationInputError):
        write_candidate_receipt(receipt_path, receipt)


def test_receipt_requires_a_passed_run() -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    run = run_candidate_evaluation(
        dataset,
        type("Bad", (), {"select": lambda _self, _request: None})(),
        _metadata(),
        evaluated_at_utc=NOW,
        monotonic_clock=lambda: 0.0,
    )
    assert run.report.status == "failed"
    with pytest.raises(CandidateEvaluationInputError):
        build_candidate_evaluation_receipt(
            run,
            _metadata(),
            source_commit="a" * 40,
            evaluated_at_utc=NOW,
        )


def test_candidate_evaluation_cli_writes_receipt_for_passed_response_set(
    tmp_path: Path,
) -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    run = run_candidate_evaluation(
        dataset,
        PerfectFixture(dataset),
        _metadata(),
        evaluated_at_utc=NOW,
        monotonic_clock=lambda: 0.0,
    )
    responses = tmp_path / "responses.jsonl"
    responses.write_bytes(serialize_candidate_traces(run.traces))
    receipt = tmp_path / "receipt.json"

    result = evaluation_main(
        [
            "--dataset",
            str(MANIFEST),
            "--responses",
            str(responses),
            "--receipt-out",
            str(receipt),
            "--candidate-id",
            "fixture-candidate-v1",
            "--provider",
            "local-fixture",
            "--model",
            "selector-fixture-v1",
            "--source-commit",
            "a" * 40,
            "--evaluated-at-utc",
            "2026-08-18T12:00:00Z",
        ]
    )

    assert result == 0
    assert receipt.exists()
    assert json.loads(receipt.read_text(encoding="utf-8"))["egress_allowed"] is False
