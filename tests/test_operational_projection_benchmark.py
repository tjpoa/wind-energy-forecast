from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import benchmark_operational_projection as benchmark
from wind_forecast.operational_projection_projector import build_projection_snapshot


def test_profiles_have_exact_contract_cardinalities() -> None:
    full = benchmark.PROFILES["full"]
    assert (
        full.reports,
        full.attempts,
        full.alerts,
        full.drift_measurements,
        full.repetitions,
    ) == (1_000, 10_000, 50_000, 200_000, 30)
    assert benchmark._feature_count(full) == 25
    assert full.enforce_timing_gate is True
    assert full.enforce_plan_gate is True

    smoke = benchmark.PROFILES["smoke"]
    assert smoke.enforce_timing_gate is False
    assert smoke.enforce_plan_gate is False
    assert smoke.repetitions == 3


def test_synthetic_store_is_loader_valid_and_projection_complete(
    tmp_path: Path,
) -> None:
    profile = benchmark.PROFILES["smoke"]
    selection = benchmark.generate_synthetic_store(tmp_path, profile)
    snapshot = build_projection_snapshot(
        tmp_path,
        environment_id="local",
        source_git_commit="a" * 40,
    )

    assert snapshot.counts()["monitoring_report_count"] == profile.reports
    assert snapshot.counts()["reporting_attempt_count"] == profile.attempts
    assert snapshot.counts()["alert_event_count"] == profile.alerts
    assert snapshot.counts()["monitoring_window_count"] == profile.reports * 2
    assert snapshot.counts()["performance_metric_count"] == profile.reports * 10
    assert snapshot.counts()["drift_measurement_count"] == profile.drift_measurements
    assert not snapshot.rows_for("model_era")

    cases = {case.name: case for case in benchmark.build_query_cases(tmp_path, selection)}
    assert tuple(cases) == (
        "alert_interval_pagination",
        "exact_alert_id",
        "reporting_run_by_run_id",
        "reporting_run_by_report_id",
        "performance_report_window",
        "drift_report_window",
    )
    assert cases["exact_alert_id"].filesystem() == ((selection.alert_event_id,),)
    assert cases["reporting_run_by_run_id"].filesystem()[0][0] == (
        selection.reporting_run_id
    )
    assert cases["reporting_run_by_report_id"].filesystem()[0][1] == (
        selection.report_id
    )
    assert [row[2] for row in cases["performance_report_window"].filesystem()] == list(
        benchmark.PERFORMANCE_METRICS
    )
    drift = cases["drift_report_window"].filesystem()
    assert len(drift) == profile.drift_measurements // profile.reports // 2
    assert [row[2] for row in drift] == list(range(len(drift)))


def _case_result(
    *,
    equivalent: bool = True,
    maximum: float = 1.0,
    speedup: float = 0.5,
    speed_gate: bool = False,
    indexes: bool = True,
) -> dict[str, object]:
    return {
        "equivalent": equivalent,
        "postgres_max_ms": maximum,
        "speedup": speedup,
        "speed_gate": speed_gate,
        "expected_indexes_used": indexes,
    }


def test_gate_is_closed_and_only_two_cases_require_speedup() -> None:
    passing = {
        "alert_interval_pagination": _case_result(speed_gate=True, speedup=0.2),
        "reporting_run_by_report_id": _case_result(speed_gate=True, speedup=0.21),
        "exact_alert_id": _case_result(speed_gate=False, speedup=-1.0),
    }
    assert benchmark.evaluate_gate(
        passing, enforce_timing=True, enforce_plans=True
    ) == ("GO", ())

    too_slow = dict(passing)
    too_slow["reporting_run_by_report_id"] = _case_result(
        speed_gate=True, speedup=0.199999
    )
    decision, failures = benchmark.evaluate_gate(
        too_slow, enforce_timing=True, enforce_plans=True
    )
    assert decision == "NO-GO"
    assert failures == ("reporting_run_by_report_id:speedup",)

    mismatch = {"exact_alert_id": _case_result(equivalent=False)}
    assert benchmark.evaluate_gate(
        mismatch, enforce_timing=False, enforce_plans=False
    ) == ("NO-GO", ("exact_alert_id:identity_order_mismatch",))


def test_gate_rejects_deadline_and_missing_index() -> None:
    cases = {
        "exact_alert_id": _case_result(
            maximum=benchmark.DEADLINE_MS,
            indexes=False,
        )
    }
    assert benchmark.evaluate_gate(
        cases, enforce_timing=True, enforce_plans=True
    ) == ("NO-GO", ("exact_alert_id:deadline", "exact_alert_id:index"))


def test_explain_index_names_are_collected_without_raw_plan_output() -> None:
    plan = [
        {
            "Plan": {
                "Node Type": "Nested Loop",
                "Plans": [
                    {"Node Type": "Index Scan", "Index Name": "alert_event_pkey"},
                    {
                        "Node Type": "Bitmap Heap Scan",
                        "Plans": [
                            {
                                "Node Type": "Bitmap Index Scan",
                                "Index Name": "alert_event_date_idx",
                            }
                        ],
                    },
                ],
            }
        }
    ]
    assert benchmark._index_names(plan) == (
        "alert_event_date_idx",
        "alert_event_pkey",
    )


def test_cli_configuration_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in (
        benchmark.MIGRATOR_DSN_ENV,
        benchmark.WRITER_DSN_ENV,
        benchmark.READER_DSN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    assert benchmark.main(["--profile", "smoke"]) == 2
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["decision"] == "ERROR"
    assert "dsn" not in output.lower()
    assert "postgresql://" not in output.lower()
    assert str(Path.cwd()).lower() not in output.lower()
