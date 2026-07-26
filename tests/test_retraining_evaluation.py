from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Barrier

import pytest

import wind_forecast.retraining_evaluation as evaluation
from wind_forecast.monitoring_reporting import MonitoringReportingError
from wind_forecast.retraining_evaluation import (
    MonthlyRetrainingEvaluationConfig,
    RetrainingEvaluationError,
    load_monthly_retraining_evaluation,
    plan_monthly_retraining_evaluation,
    run_monthly_retraining_evaluation,
)


POLICY_PATH = Path("config/retraining_policy_v1.json")
INCUMBENT_ID = "incumbent-model-snapshot"


def _environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    observation_count: int = 90,
    alert_category: str | None = "feature_drift",
    alert_severity: str = "warning",
) -> tuple[MonthlyRetrainingEvaluationConfig, dict]:
    store = tmp_path / "monitoring"
    report_path = (
        store
        / "reporting"
        / "reports"
        / "report-1"
        / "report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"synthetic": true}\n', encoding="utf-8")
    reporting_state_path = store / "reporting" / "state" / "current.json"
    reporting_state_path.parent.mkdir(parents=True)
    reporting_state_path.write_text('{"synthetic": true}\n', encoding="utf-8")
    ledger_state_path = store / "state" / "current.json"
    ledger_state_path.parent.mkdir(parents=True)
    ledger_state_path.write_text('{"synthetic": true}\n', encoding="utf-8")

    days = [
        (date(2026, 1, 1) + timedelta(days=index)).isoformat()
        for index in range(observation_count)
    ]
    ledger = {
        "generation": 7,
        "model_snapshot_id": INCUMBENT_ID,
        "as_issued": {
            day: f"prediction-{index:03d}" for index, day in enumerate(days)
        },
        "actuals": {
            day: f"actual-{index:03d}" for index, day in enumerate(days)
        },
        "restated": (
            {days[5]: "restated-005"} if len(days) > 5 else {}
        ),
    }
    active_alerts = {}
    breaches = []
    persistence = {}
    history = []
    if alert_category is not None:
        rule_id = f"{alert_category}:synthetic"
        event_id = "alert-1"
        active_alerts[rule_id] = event_id
        breaches.append(
            {
                "rule_id": rule_id,
                "category": alert_category,
                "severity": alert_severity,
                "immediate": alert_category == "quality",
            }
        )
        required = 1 if alert_category == "quality" else 3
        persistence[rule_id] = {
            "active": True,
            "severity": alert_severity,
            "consecutive": required,
            "required": required,
            "last_date": "2026-03-31",
            "last_event_id": event_id,
        }
        history.append(
            {
                "alert_event_id": event_id,
                "rule_id": rule_id,
                "event_type": "opened",
                "severity": alert_severity,
                "through_date": "2026-03-31",
            }
        )
    report = {
        "report_id": "report-1",
        "through_date": "2026-03-31",
        "active_alerts": active_alerts,
        "breaches": breaches,
        "persistence": persistence,
        "lineage": {
            "primary_view": "as_issued",
            "ledger_generation": 6,
            "prediction_ids": list(ledger["as_issued"].values())[-30:],
        },
    }
    report_state = {
        "generation": 4,
        "latest_report_id": "report-1",
        "latest_through_date": "2026-03-31",
    }

    monkeypatch.setattr(evaluation, "load_monitoring_report", lambda _path: report)
    monkeypatch.setattr(
        evaluation,
        "load_monitoring_report_state",
        lambda _root: report_state,
    )
    monkeypatch.setattr(
        evaluation,
        "load_alert_history",
        lambda _root: history,
    )
    monkeypatch.setattr(
        evaluation,
        "load_verified_monitoring_state",
        lambda _root: ledger,
    )

    def evidence(_root: Path, prediction_id: str) -> dict:
        index = int(prediction_id.rsplit("-", 1)[-1])
        day = days[index]
        actual_id = f"actual-{index:03d}"
        return {
            "prediction": {
                "prediction_id": prediction_id,
                "view": "as_issued",
                "target_date": day,
                "model_snapshot_id": INCUMBENT_ID,
            },
            "model_input_snapshot": {
                "model_input_snapshot_id": f"input-{index:03d}",
                "target_date": day,
                "feature_values": [float(index), float(index + 1)],
                "feature_schema_sha256": "a" * 64,
                "transformation": {"version": "v2_features_v1"},
                "dependencies": {
                    "Wind_Production_Lag1": {
                        "source_revisions": [
                            {
                                "revision_id": f"ren-feature-{index:03d}",
                            }
                        ]
                    },
                    "Average_Wind_Speed": {
                        "source_revisions": [
                            {
                                "revision_id": f"era-feature-{index:03d}",
                            }
                        ]
                    },
                },
            },
            "model_snapshot": {"model_snapshot_id": INCUMBENT_ID},
            "actual_revisions": [
                {
                    "actual_revision_id": actual_id,
                    "target_contract_id": "ren_wind_production_15min_mw_sum_v1",
                    "actual": 100.0 + index,
                    "source_revision_id": f"ren-target-{index:03d}",
                }
            ],
        }

    monkeypatch.setattr(evaluation, "load_prediction_evidence", evidence)
    config = MonthlyRetrainingEvaluationConfig(
        policy_path=POLICY_PATH,
        monitoring_store_root=store,
        monitoring_report_path=report_path,
        incumbent_id=INCUMBENT_ID,
        incumbent_fit_cutoff="2025-12-31",
        output_root=tmp_path / "evaluations",
        evaluated_at_utc=datetime(2026, 4, 8, 12, tzinfo=timezone.utc),
    )
    return config, {
        "ledger": ledger,
        "report": report,
        "report_state": report_state,
        "history": history,
        "evidence": evidence,
    }


def test_before_schedule_is_not_due_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _environment(tmp_path, monkeypatch)
    early = MonthlyRetrainingEvaluationConfig(
        **{
            **config.__dict__,
            "evaluated_at_utc": datetime(
                2026, 4, 8, 11, 59, tzinfo=timezone.utc
            ),
        }
    )
    result = run_monthly_retraining_evaluation(early)

    assert result.status == "not_due"
    assert result.outcome == "not_due"
    assert result.evaluation_path is None
    assert not early.output_root.exists()

    late = MonthlyRetrainingEvaluationConfig(
        **{
            **config.__dict__,
            "evaluated_at_utc": datetime(
                2026, 5, 1, 12, tzinfo=timezone.utc
            ),
        }
    )
    catch_up = plan_monthly_retraining_evaluation(late)
    assert catch_up.status == "planned"
    assert catch_up.evaluation_period == "2026-04"


@pytest.mark.parametrize(
    ("observation_count", "alert_category", "outcome"),
    [
        (89, "feature_drift", "insufficient_observations"),
        (90, None, "no_trigger"),
        (90, "feature_drift", "eligible_for_manual_backtest"),
        (90, "performance", "eligible_for_manual_backtest"),
        (90, "quality", "blocked_quality"),
    ],
)
def test_monthly_outcomes_are_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observation_count: int,
    alert_category: str | None,
    outcome: str,
) -> None:
    config, _ = _environment(
        tmp_path,
        monkeypatch,
        observation_count=observation_count,
        alert_category=alert_category,
    )

    plan = plan_monthly_retraining_evaluation(config)

    assert plan.outcome == outcome
    assert plan.record["outcome"] == outcome
    assert plan.record["eligibility"]["eligible_observation_count"] == (
        observation_count
    )
    assert plan.record["incumbent"]["champion_claim"] is False
    assert plan.record["safeguards"]["training"] is False
    assert plan.record["safeguards"]["monitoring_persistence_increment"] is False


def test_as_issued_features_current_target_and_restatement_are_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _environment(tmp_path, monkeypatch)

    plan = plan_monthly_retraining_evaluation(config)
    eligibility = plan.record["eligibility"]

    assert eligibility["feature_view"] == "as_issued"
    assert eligibility["target_revision_view"] == "current"
    assert eligibility["restatements_ignored"] == [
        {
            "target_date": "2026-01-06",
            "as_issued_prediction_id": "prediction-005",
            "ignored_restatement_prediction_id": "restated-005",
        }
    ]
    assert eligibility["eligible_observations"][5]["target_revision_id"] == (
        "actual-005"
    )


def test_dry_run_writes_nothing_and_sealed_rerun_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _environment(tmp_path, monkeypatch)
    dry_config = MonthlyRetrainingEvaluationConfig(
        **{**config.__dict__, "dry_run": True}
    )

    dry = run_monthly_retraining_evaluation(dry_config)

    assert dry.status == "planned"
    assert dry.evaluation_id
    assert not config.output_root.exists()

    first = run_monthly_retraining_evaluation(config)
    before = first.evaluation_path.read_bytes()
    second = run_monthly_retraining_evaluation(config)

    assert first.evaluation_id == second.evaluation_id
    assert first.evaluation_path == second.evaluation_path
    assert second.evaluation_path.read_bytes() == before
    loaded = load_monthly_retraining_evaluation(first.evaluation_path)
    assert loaded["evaluation_id"] == first.evaluation_id


def test_conflicting_evidence_for_sealed_period_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _environment(tmp_path, monkeypatch)
    run_monthly_retraining_evaluation(config)
    changed = MonthlyRetrainingEvaluationConfig(
        **{
            **config.__dict__,
            "evaluated_at_utc": datetime(
                2026, 4, 8, 12, 1, tzinfo=timezone.utc
            ),
        }
    )

    with pytest.raises(RetrainingEvaluationError, match="Conflicting evidence"):
        run_monthly_retraining_evaluation(changed)


def test_corrupt_or_cross_store_report_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _environment(tmp_path, monkeypatch)
    outside = tmp_path / "outside" / "report.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    cross_store = MonthlyRetrainingEvaluationConfig(
        **{**config.__dict__, "monitoring_report_path": outside}
    )
    with pytest.raises(RetrainingEvaluationError, match="must belong"):
        plan_monthly_retraining_evaluation(cross_store)

    monkeypatch.setattr(
        evaluation,
        "load_monitoring_report",
        lambda _path: (_ for _ in ()).throw(
            MonitoringReportingError("Corrupt report")
        ),
    )
    with pytest.raises(RetrainingEvaluationError, match="Corrupt report"):
        plan_monthly_retraining_evaluation(config)


def test_incumbent_alert_and_numeric_mismatches_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, values = _environment(tmp_path, monkeypatch)
    values["ledger"]["model_snapshot_id"] = "different-model"
    with pytest.raises(RetrainingEvaluationError, match="incumbent_id differs"):
        plan_monthly_retraining_evaluation(config)

    values["ledger"]["model_snapshot_id"] = INCUMBENT_ID
    values["report"]["persistence"]["feature_drift:synthetic"][
        "last_event_id"
    ] = "different-alert"
    with pytest.raises(RetrainingEvaluationError, match="disagrees"):
        plan_monthly_retraining_evaluation(config)

    values["report"]["persistence"]["feature_drift:synthetic"][
        "last_event_id"
    ] = "alert-1"

    def nonfinite(root: Path, prediction_id: str) -> dict:
        result = values["evidence"](root, prediction_id)
        result["model_input_snapshot"]["feature_values"][0] = float("nan")
        return result

    monkeypatch.setattr(evaluation, "load_prediction_evidence", nonfinite)
    with pytest.raises(RetrainingEvaluationError, match="non-finite"):
        plan_monthly_retraining_evaluation(config)


@pytest.mark.parametrize(
    ("event_change", "match"),
    [
        ({"severity": "critical"}, "disagrees"),
        ({"through_date": "2026-04-01"}, "disagrees"),
    ],
)
def test_active_alert_event_severity_and_date_must_match_pinned_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event_change: dict[str, str],
    match: str,
) -> None:
    config, values = _environment(tmp_path, monkeypatch)
    values["history"][0].update(event_change)

    with pytest.raises(RetrainingEvaluationError, match=match):
        plan_monthly_retraining_evaluation(config)


def test_latched_critical_state_accepts_current_warning_breach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, values = _environment(tmp_path, monkeypatch)
    rule_id = "feature_drift:synthetic"
    values["report"]["persistence"][rule_id]["severity"] = "critical"
    values["history"][0]["severity"] = "critical"

    plan = plan_monthly_retraining_evaluation(config)

    assert plan.outcome == "eligible_for_manual_backtest"
    assert plan.record["active_triggers"] == [
        {
            "rule_id": rule_id,
            "category": "feature_drift",
            "severity": "critical",
            "breach_severity": "warning",
            "alert_event_id": "alert-1",
            "event_type": "opened",
            "event_through_date": "2026-03-31",
            "consecutive": 3,
            "required": 3,
        }
    ]


def test_current_critical_breach_cannot_exceed_warning_persisted_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, values = _environment(tmp_path, monkeypatch)
    values["report"]["breaches"][0]["severity"] = "critical"

    with pytest.raises(RetrainingEvaluationError, match="disagrees"):
        plan_monthly_retraining_evaluation(config)


def test_missing_current_target_is_an_explicit_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, values = _environment(tmp_path, monkeypatch)
    values["ledger"]["actuals"].pop("2026-03-31")

    plan = plan_monthly_retraining_evaluation(config)

    assert plan.outcome == "insufficient_observations"
    assert plan.record["eligibility"]["eligible_observation_count"] == 89
    assert plan.record["eligibility"]["excluded"] == {
        "prediction-089": ["missing_current_target_revision"]
    }


def test_cutoff_lateness_boundary_and_strict_loader_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, values = _environment(tmp_path, monkeypatch)
    values["report"]["through_date"] = "2026-04-02"
    values["report_state"]["latest_through_date"] = "2026-04-02"
    later_period = plan_monthly_retraining_evaluation(config)
    assert later_period.status == "not_due"
    assert later_period.evaluation_period == "2026-05"

    values["report"]["through_date"] = "2026-03-31"
    values["report_state"]["latest_through_date"] = "2026-03-31"
    invalid_cutoff = MonthlyRetrainingEvaluationConfig(
        **{**config.__dict__, "incumbent_fit_cutoff": "2026-03-31"}
    )
    with pytest.raises(RetrainingEvaluationError, match="must precede"):
        plan_monthly_retraining_evaluation(invalid_cutoff)

    result = run_monthly_retraining_evaluation(config)
    payload = json.loads(result.evaluation_path.read_text(encoding="utf-8"))
    payload["outcome"] = "eligible_for_automatic_training"
    result.evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RetrainingEvaluationError, match="content-addressed"):
        load_monthly_retraining_evaluation(result.evaluation_path)


def test_concurrent_different_records_atomically_seal_exactly_one_period(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _environment(tmp_path, monkeypatch)
    first = plan_monthly_retraining_evaluation(config).record
    changed = MonthlyRetrainingEvaluationConfig(
        **{
            **config.__dict__,
            "evaluated_at_utc": datetime(
                2026, 4, 8, 12, 1, tzinfo=timezone.utc
            ),
        }
    )
    second = plan_monthly_retraining_evaluation(changed).record
    period_root = config.output_root / "2026-04"
    barrier = Barrier(2)
    publish = evaluation._publish_prepared_period

    def synchronized_publish(prepared: Path, destination: Path) -> None:
        barrier.wait(timeout=5)
        publish(prepared, destination)

    monkeypatch.setattr(
        evaluation,
        "_publish_prepared_period",
        synchronized_publish,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(evaluation._seal_period, period_root, record)
            for record in (first, second)
        ]
        outcomes = []
        for future in futures:
            try:
                future.result(timeout=10)
                outcomes.append("succeeded")
            except RetrainingEvaluationError as exc:
                outcomes.append(str(exc))

    assert outcomes.count("succeeded") == 1
    assert sum("Conflicting evidence" in item for item in outcomes) == 1
    entries = list(period_root.iterdir())
    assert len(entries) == 1
    sealed = load_monthly_retraining_evaluation(entries[0] / "evaluation.json")
    assert sealed["evaluation_id"] in {
        first["evaluation_id"],
        second["evaluation_id"],
    }
    assert list(config.output_root.glob(".*.tmp")) == []
    evaluation._seal_period(period_root, sealed)
