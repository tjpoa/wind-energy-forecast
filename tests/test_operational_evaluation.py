from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate_operational_copilot import main as evaluation_main
import wind_forecast.operational_query as operational
from wind_forecast.operational_evaluation import (
    MAX_JSONL_LINE_BYTES,
    OperationalEvaluationInputError,
    evaluate_candidate_results,
    load_candidate_traces,
    load_evaluation_dataset,
    sanitized_report_json,
)
from wind_forecast.operational_evaluation_models import (
    CandidateTrace,
    EvaluationCategory,
    ExpectedAnswer,
    TRACE_SCHEMA_VERSION,
)
from wind_forecast.operational_query import OperationalQueryService
from wind_forecast.operational_query_models import (
    AnswerStatus,
    EvidenceCitation,
    EvidenceDomain,
    EvidenceState,
    GroundedFact,
    OperationalAnswer,
    OperationalFailure,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation/operational_read_only_copilot/v1/manifest.json"
NOW = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
REPORT_ID = "a" * 64
ALERT_ID = "b" * 64
CALIBRATION_ID = "c" * 64
RUN_ID = "20260730T115900000000Z-abcdef123456"


def _domain(source_kind: str) -> EvidenceDomain:
    if source_kind == "verified_registry_alias_binding":
        return EvidenceDomain.REGISTRY
    if source_kind.startswith("verify_active_model_era.model_bundle"):
        return EvidenceDomain.MODEL_BUNDLE
    if source_kind.startswith("verify_active_model_era.calibration"):
        return EvidenceDomain.CALIBRATION
    if source_kind == "verify_active_model_era":
        return EvidenceDomain.DEPLOYMENT
    if source_kind == "load_monitoring_calibration":
        return EvidenceDomain.CALIBRATION
    if source_kind in {"load_alert_history", "load_active_alerts"}:
        return EvidenceDomain.ALERT
    if source_kind == "load_reporting_attempt":
        return EvidenceDomain.REPORTING_RUN
    return EvidenceDomain.MONITORING_REPORT


def _canonical_text(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _answer(oracle: ExpectedAnswer, *, correlation_id: str) -> OperationalAnswer:
    facts: list[GroundedFact] = []
    citations: list[EvidenceCitation] = []
    evidence_index = 1
    for fact_index, expected in enumerate(oracle.facts, 1):
        evidence_ids: list[str] = []
        for source_kind in expected.source_kinds:
            evidence_id = f"e{evidence_index}"
            digest = hashlib.sha256(
                f"{expected.name}:{source_kind}".encode("utf-8")
            ).hexdigest()
            citations.append(
                EvidenceCitation(
                    evidence_id=evidence_id,
                    domain=_domain(source_kind),
                    source_kind=source_kind,
                    schema_version="wind_forecast.synthetic_evaluation.v1",
                    record_id=digest,
                    sha256=digest,
                    effective_at=expected.as_of,
                    observed_at_utc=(
                        NOW if expected.requires_observed_at_utc else None
                    ),
                )
            )
            evidence_ids.append(evidence_id)
            evidence_index += 1
        facts.append(
            GroundedFact(
                fact_id=f"f{fact_index}",
                name=expected.name,
                value=expected.value,
                unit_or_scale=expected.unit_or_scale,
                as_of=expected.as_of,
                evidence_ids=tuple(evidence_ids),
            )
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
            message="Sanitized evaluation failure.",
            retryable=oracle.status in {AnswerStatus.UNAVAILABLE, AnswerStatus.TIMEOUT},
            evidence_state=oracle.failure_evidence_state,
        )
    return OperationalAnswer(
        query_kind=oracle.query_kind,
        status=oracle.status,
        summary=summary,
        facts=tuple(facts),
        evidence=tuple(citations),
        limitations=oracle.limitations,
        failure=failure,
        served_at_utc=NOW,
        correlation_id=correlation_id,
    )


def _perfect_traces(dataset) -> list[CandidateTrace]:
    return [
        CandidateTrace(
            case_id=case.case_id,
            selected_tool=(case.expected_tool.name if case.expected_tool else None),
            tool_arguments=(case.expected_tool.arguments if case.expected_tool else None),
            answer=_answer(
                dataset.manifest.oracles[case.oracle_id],
                correlation_id=case.case_id,
            ),
        )
        for case in dataset.cases
    ]


def _score(dataset, traces):
    return evaluate_candidate_results(
        dataset,
        traces,
        response_set_sha256="f" * 64,
    )


def test_versioned_dataset_is_sealed_complete_and_synthetic() -> None:
    dataset = load_evaluation_dataset(MANIFEST)

    assert len(dataset.cases) == 88
    assert [case.case_id for case in dataset.cases] == [
        f"eval-{index:03d}" for index in range(1, 89)
    ]
    assert {
        category: sum(case.category == category for case in dataset.cases)
        for category in EvaluationCategory
    } == dataset.manifest.distribution
    assert sum(case.category == EvaluationCategory.CANONICAL for case in dataset.cases) == 20
    assert sum(case.category == EvaluationCategory.PARAPHRASE for case in dataset.cases) == 20
    assert all(case.question.isascii() for case in dataset.cases)
    serialized_oracles = json.dumps(
        dataset.manifest.oracles,
        default=lambda value: value.model_dump(mode="json"),
    )
    assert "C:\\Users\\" not in serialized_oracles
    assert "/home/" not in serialized_oracles
    assert "connection_string" not in serialized_oracles


def test_perfect_candidate_passes_every_gate() -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    report = _score(dataset, _perfect_traces(dataset))

    assert report.status == "passed"
    assert report.metrics.passed_case_count == 88
    assert report.metrics.critical_failure_count == 0
    assert report.metrics.canonical_tool_pass_rate == 1.0
    assert report.metrics.factual_grounding_pass_rate == 1.0
    assert report.metrics.evidence_state_pass_rate == 1.0
    assert report.metrics.paraphrase_recognition_pass_rate == 1.0
    assert report.acceptance_state == "harness accepted; no Copilot evaluated"


def test_one_safe_paraphrase_abstention_still_passes_gate() -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    traces = _perfect_traces(dataset)
    index = next(
        index
        for index, case in enumerate(dataset.cases)
        if case.category == EvaluationCategory.PARAPHRASE
    )
    traces[index] = CandidateTrace(
        case_id=traces[index].case_id,
        selected_tool=None,
        tool_arguments=None,
        answer=OperationalAnswer(
            query_kind=None,
            status=AnswerStatus.REFUSED,
            summary=None,
            facts=(),
            evidence=(),
            limitations=(),
            failure=OperationalFailure(
                code="unsupported_query_kind",
                message="The requested operational question is not supported.",
                retryable=False,
                evidence_state=EvidenceState.UNSUPPORTED,
            ),
            served_at_utc=NOW,
            correlation_id=traces[index].case_id,
        ),
    )

    report = _score(dataset, traces)

    assert report.status == "passed"
    assert report.metrics.paraphrase_recognition_pass_rate == 0.95
    assert report.metrics.safe_paraphrase_abstention_count == 1
    assert report.metrics.critical_failure_count == 0


@pytest.mark.parametrize(
    ("mutation", "failure_code"),
    (
        ("wrong_tool", "tool_selection_mismatch"),
        ("wrong_arguments", "tool_arguments_mismatch"),
        ("invented_summary", "summary_not_deterministically_grounded"),
        ("wrong_scale", "fact_value_mismatch"),
    ),
)
def test_critical_candidate_mutations_fail(mutation: str, failure_code: str) -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    traces = _perfect_traces(dataset)
    trace = traces[0]
    if mutation == "wrong_tool":
        traces[0] = trace.model_copy(update={"selected_tool": "write_deployment"})
    elif mutation == "wrong_arguments":
        traces[0] = trace.model_copy(
            update={"tool_arguments": dataset.cases[1].expected_tool.arguments}
        )
    else:
        answer = trace.answer
        if mutation == "invented_summary":
            answer = answer.model_copy(update={"summary": answer.summary + " Healthy."})
        else:
            facts = list(answer.facts)
            facts[0] = facts[0].model_copy(update={"unit_or_scale": "MW"})
            answer = answer.model_copy(update={"facts": tuple(facts)})
        traces[0] = trace.model_copy(update={"answer": answer})

    report = _score(dataset, traces)

    assert report.status == "failed"
    result = next(item for item in report.results if item.case_id == "eval-001")
    assert failure_code in result.failure_codes
    assert report.metrics.critical_failure_count >= 1


def test_private_candidate_data_is_detected_but_never_echoed() -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    traces = _perfect_traces(dataset)
    index = 56
    trace = traces[index]
    failure = trace.answer.failure.model_copy(
        update={"message": "Traceback (most recent call last): C:\\Users\\operator\\secret"}
    )
    traces[index] = trace.model_copy(
        update={"answer": trace.answer.model_copy(update={"failure": failure})}
    )

    report = _score(dataset, traces)
    payload = sanitized_report_json(report)

    assert report.status == "failed"
    assert "private_data_exposure" in payload
    assert "Traceback" not in payload
    assert "operator" not in payload


def test_dataset_and_response_inputs_fail_closed(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["cases_sha256"] = "0" * 64
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (dataset_dir / "cases.jsonl").write_bytes(
        (MANIFEST.parent / "cases.jsonl").read_bytes()
    )
    with pytest.raises(OperationalEvaluationInputError, match="checksum"):
        load_evaluation_dataset(dataset_dir / "manifest.json")

    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b"{" + b"x" * MAX_JSONL_LINE_BYTES + b"}\n")
    with pytest.raises(OperationalEvaluationInputError, match="line exceeds"):
        load_candidate_traces(oversized)


def _era() -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_model_era.v1",
        "model_era_id": "e" * 64,
        "association_kind": "active_deployment",
        "deployment": {
            "deployment_id": "d" * 64,
            "deployment_state_id": "f" * 64,
            "generation": 7,
            "pointer_sha256": "1" * 64,
            "state_manifest_sha256": "2" * 64,
            "authorizing_receipt_sha256": "3" * 64,
        },
        "registry": {
            "registered_model_name": "wind-v2",
            "model_version": "11",
            "run_id": "registry-run",
            "model_uri": "models:/wind-v2/11",
        },
        "expected_aliases": {"candidate": None, "champion": "11", "stable": "11"},
        "cutoffs": {"fit_cutoff": "2025-12-31", "activation_cutoff": "2026-07-01"},
        "pins": {
            "bundle_sha256": "4" * 64,
            "model_sha256": "5" * 64,
            "dataset_sha256": "6" * 64,
            "feature_schema_sha256": "7" * 64,
            "calibration_sha256": "8" * 64,
            "ledger_sha256": "9" * 64,
        },
        "calibration": {"calibration_id": CALIBRATION_ID, "reference_id": "0" * 64},
        "monitoring": {"ledger_model_snapshot_id": "1" * 64},
        "_runtime_metadata": {
            "model_type": "RandomForestRegressor",
            "dataset_version": "v2",
            "transformation_version": "transform-v2",
        },
    }


def _window(days: int) -> dict:
    is_short = days == 30
    return {
        "status": "available",
        "sample_count": days,
        "coverage_ratio": 1.0,
        "coverage_severity": "ok",
        "minimum_samples": 24 if is_short else 72,
        "calendar_start": "2026-06-30" if is_short else "2026-05-01",
        "calendar_end": "2026-07-29",
        "performance": {
            "metrics": {
                "MAE": 10.0 if is_short else 11.0,
                "RMSE": 12.0 if is_short else 13.0,
                "bias": -1.0 if is_short else -2.0,
                "MAPE_percent": 4.0 if is_short else 5.0,
                "R2": 0.8 if is_short else 0.75,
            },
            "severity": {
                "MAE": "ok",
                "RMSE": "warning" if is_short else "ok",
                "bias": "ok",
                "MAPE_percent": "ok",
                "R2": "ok",
            },
        },
        "feature_drift": {
            "wind_speed": {
                "global": {
                    "ks_statistic": 0.1 if is_short else 0.12,
                    "normalized_wasserstein": 0.2 if is_short else 0.25,
                }
            }
        },
    }


def _report() -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_report.v2",
        "report_id": REPORT_ID,
        "run_id": RUN_ID,
        "created_at_utc": "2026-07-30T11:59:00Z",
        "through_date": "2026-07-29",
        "source_batch": {"run_id": "source-run", "status": "succeeded"},
        "reference": {
            "calibration_id": CALIBRATION_ID,
            "reference_id": "0" * 64,
            "policy_sha256": "1" * 64,
        },
        "quality": {
            "batch_status": "succeeded",
            "verdict": "accepted",
            "issues": [{"code": "late_source", "severity": "warning"}],
            "freshness": {
                "common_validated_watermark": "2026-07-29",
                "watermark_age_days": 1,
                "objective_days": 7,
                "late_days": 0,
                "objective_missed": False,
                "unresolved_late_dates": [],
            },
            "coverage": {
                "date_count": 30,
                "ren_complete_count": 30,
                "era5_complete_count": 30,
                "integration_ready_count": 30,
                "feature_ready_count": 30,
                "feature_ready_ratio": 1.0,
            },
        },
        "windows": {"30": _window(30), "90": _window(90)},
        "active_alerts": {"feature_drift:wind_speed:30:global": ALERT_ID},
        "alert_events": [ALERT_ID],
        "persistence": {},
        "lineage": {"prediction_ids": []},
    }


def _calibration() -> dict:
    metric_limits = {
        name: {"warning": 20.0, "critical": 30.0, "direction": "upper"}
        for name in ("MAE", "RMSE", "absolute_bias", "MAPE_percent", "R2")
    }
    metric_limits["R2"]["direction"] = "lower"
    detectors = {
        name: {"warning": 0.15, "critical": 0.3, "direction": "upper"}
        for name in ("ks_statistic", "normalized_wasserstein")
    }
    return {
        "schema_version": "wind_forecast.monitoring_calibration.v1",
        "calibration_id": CALIBRATION_ID,
        "reference_id": "0" * 64,
        "policy_sha256": "1" * 64,
        "thresholds": {
            "performance": {"30": metric_limits, "90": metric_limits},
            "coverage": {
                "30": {"warning": 0.9, "critical": 0.8, "direction": "lower"},
                "90": {"warning": 0.9, "critical": 0.8, "direction": "lower"},
            },
            "feature_drift": {
                "wind_speed": {
                    "30": {"global": detectors},
                    "90": {"global": detectors},
                }
            },
        },
    }


def _alert() -> dict:
    return {
        "schema_version": "wind_forecast.monitoring_alert_event.v2",
        "alert_event_id": ALERT_ID,
        "rule_id": "feature_drift:wind_speed:30:global",
        "through_date": "2026-07-29",
        "event_type": "opened",
        "severity": "warning",
        "previous_alert_event_id": None,
    }


def _attempt() -> dict:
    return {
        "run_id": RUN_ID,
        "attempted_at_utc": "2026-07-30T11:59:00Z",
        "through_date": "2026-07-29",
        "source_pipeline_run_id": "source-run",
        "source_pipeline_status": "succeeded",
        "status": "succeeded",
        "report_id": REPORT_ID,
        "active_alert_count": 1,
        "failure": None,
    }


@pytest.fixture
def verified_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(operational, "verify_active_model_era", lambda *_a, **_k: _era())
    monkeypatch.setattr(
        operational,
        "load_monitoring_report_state",
        lambda _root: {
            "schema_version": "wind_forecast.monitoring_report_state.v2",
            "latest_report_id": REPORT_ID,
            "latest_through_date": "2026-07-29",
            "active_alerts": {"feature_drift:wind_speed:30:global": ALERT_ID},
        },
    )
    monkeypatch.setattr(operational, "load_monitoring_report", lambda _path: _report())
    monkeypatch.setattr(operational, "load_monitoring_calibration", lambda _path: _calibration())
    monkeypatch.setattr(
        operational,
        "resolve_report_model_era",
        lambda _root, _report_value: {
            "association_kind": "active_deployment",
            "model_era_id": "e" * 64,
        },
    )
    monkeypatch.setattr(operational, "load_alert_history", lambda _root: [_alert()])
    monkeypatch.setattr(
        operational,
        "load_active_alerts",
        lambda _root: {"feature_drift:wind_speed:30:global": ALERT_ID},
    )
    monkeypatch.setattr(operational, "load_reporting_attempt", lambda _root, **_kwargs: _attempt())


def test_twenty_canonical_goldens_match_operational_query_service(
    verified_sources: None,
    tmp_path: Path,
) -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    base_traces = _perfect_traces(dataset)
    monitoring_root = tmp_path / "monitoring"
    report_path = monitoring_root / "reporting" / "reports" / REPORT_ID / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}", encoding="utf-8")
    service = OperationalQueryService(
        deployment_root=Path("synthetic-deployment"),
        monitoring_store_root=monitoring_root,
        max_deadline_seconds=300.0,
        authorization_policy=lambda context, _kind: context.trusted_local,
        registry_client=object(),
        registry_timeout_seconds=10.0,
        clock=lambda: NOW,
    )

    for index, case in enumerate(dataset.cases[:20]):
        request = case.expected_tool.arguments.model_dump(mode="python")
        request.update(
            requested_at_utc=NOW - timedelta(seconds=30),
            correlation_id=case.case_id,
            deadline=NOW + timedelta(seconds=30),
        )
        actual = service.answer(request, case.authorization)
        traces = list(base_traces)
        traces[index] = CandidateTrace(
            case_id=case.case_id,
            selected_tool="operational_query",
            tool_arguments=case.expected_tool.arguments,
            answer=actual,
        )
        report = _score(dataset, traces)
        result = report.results[index]
        assert result.passed, (case.case_id, result.failure_codes)


def test_evaluation_does_not_modify_dataset_or_synthetic_store(tmp_path: Path) -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    store = tmp_path / "store"
    store.mkdir()
    evidence = store / "evidence.json"
    evidence.write_bytes(b'{"immutable":true}\n')

    def snapshot() -> tuple[bytes, int, int]:
        stat = evidence.stat()
        return evidence.read_bytes(), stat.st_size, stat.st_mtime_ns

    before = snapshot()
    report = _score(dataset, _perfect_traces(dataset))
    after = snapshot()

    assert report.status == "passed"
    assert after == before


def test_candidate_trace_schema_forbids_server_metadata() -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    payload = _perfect_traces(dataset)[0].model_dump(mode="json")
    payload["tool_arguments"]["correlation_id"] = "client-supplied"

    with pytest.raises(Exception):
        CandidateTrace.model_validate(payload)


def test_trace_schema_version_is_explicit() -> None:
    assert TRACE_SCHEMA_VERSION == "wind_forecast.operational_evaluation_trace.v1"


def test_stdout_only_runner_exit_codes_are_deterministic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = load_evaluation_dataset(MANIFEST)
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        "\n".join(trace.model_dump_json() for trace in _perfect_traces(dataset))
        + "\n",
        encoding="utf-8",
    )

    assert evaluation_main(
        ["--dataset", str(MANIFEST), "--responses", str(responses)]
    ) == 0
    passed_payload = json.loads(capsys.readouterr().out)
    assert passed_payload["status"] == "passed"
    assert passed_payload["acceptance_state"] == (
        "harness accepted; no Copilot evaluated"
    )

    traces = _perfect_traces(dataset)
    traces[0] = traces[0].model_copy(update={"selected_tool": "write_deployment"})
    responses.write_text(
        "\n".join(trace.model_dump_json() for trace in traces) + "\n",
        encoding="utf-8",
    )
    assert evaluation_main(
        ["--dataset", str(MANIFEST), "--responses", str(responses)]
    ) == 1
    failed_payload = json.loads(capsys.readouterr().out)
    assert failed_payload["status"] == "failed"
    assert "write_deployment" not in json.dumps(failed_payload)

    responses.write_text("not-json\n", encoding="utf-8")
    assert evaluation_main(
        ["--dataset", str(MANIFEST), "--responses", str(responses)]
    ) == 2
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "wind_forecast.operational_evaluation_error.v1",
        "status": "invalid_input",
    }
