from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from wind_forecast import monthly_governance as governance
from wind_forecast.monthly_governance import (
    MonthlyGovernanceConfig,
    MonthlyGovernanceError,
    canonical_monthly_logical_time,
    load_monthly_governance_recommendation,
    run_monthly_governance,
    select_month_end_report,
)
from wind_forecast.retraining_policy import RetrainingPolicy


POLICY = Path("config/retraining_policy_v1.json")
MONITORING_POLICY = Path("config/monitoring_policy_v1.json")


def _config(tmp_path: Path, *, dry_run: bool = False) -> MonthlyGovernanceConfig:
    return MonthlyGovernanceConfig(
        policy_path=POLICY,
        monitoring_policy_path=MONITORING_POLICY,
        monitoring_store_root=tmp_path / "monitoring",
        deployment_root=tmp_path / "deployment",
        logical_at_utc="2026-07-08T12:00:00Z",
        output_root=tmp_path / "recommendations",
        evaluation_output_root=tmp_path / "evaluations",
        dry_run=dry_run,
    )


def _state(status: str = "stable") -> dict:
    return {
        "deployment_id": "d" * 64,
        "deployment_state_id": "s" * 64,
        "generation": 2,
        "lifecycle_status": status,
        "registry": {
            "registered_model_name": "wind",
            "model_version": "2",
        },
        "expected_aliases": {
            "candidate": None,
            "champion": "2",
            "stable": "2" if status == "stable" else "1",
        },
        "cutoffs": {
            "fit_cutoff": "2026-01-31",
            "promotion_effective_date": "2026-02-01",
            "observation_cutoff": None,
        },
    }


def _patch_common(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "stable",
) -> tuple[Path, dict]:
    selected = tmp_path / "monitoring" / "reporting" / "reports" / "month" / "report.json"
    selected.parent.mkdir(parents=True)
    selected.write_text("{}", encoding="utf-8")
    pointer = tmp_path / "deployment" / "state" / "current.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("{}", encoding="utf-8")
    selected_report = {
        "report_id": "month",
        "through_date": "2026-06-30",
        "created_at_utc": "2026-07-08T11:00:00Z",
        "model_era": {
            "model_era_id": "era",
            "deployment_id": "d" * 64,
            "deployment_state_id": "s" * 64,
            "deployment_generation": 2,
            "registered_model_name": "wind",
            "model_version": "2",
            "cutoffs": _state(status)["cutoffs"],
            "pins": {},
        },
    }
    state = _state(status)
    monkeypatch.setattr(
        governance,
        "select_month_end_report",
        lambda *_args: (selected, selected_report),
    )
    monkeypatch.setattr(
        governance,
        "load_verified_monitoring_state",
        lambda _root: {
            "active_model_era_id": "era",
            "model_snapshot_id": "model",
            "as_issued": {},
            "actuals": {},
        },
    )
    monkeypatch.setattr(
        governance,
        "load_monitoring_report_state",
        lambda _root: {
            "latest_report_id": "latest",
            "latest_through_date": "2026-07-08",
        },
    )
    monkeypatch.setattr(
        governance,
        "load_verified_deployment_pointer",
        lambda *_args, **_kwargs: {
            "pointer_path": pointer,
            "pointer": {"generation": 2},
            "state": state,
        },
    )
    monkeypatch.setattr(
        governance,
        "load_model_era",
        lambda *_args: {
            "model_era_id": "era",
            "deployment": {
                "deployment_id": "d" * 64,
                "deployment_state_id": "s" * 64,
                "generation": 2,
            },
            "registry": {
                "registered_model_name": "wind",
                "model_version": "2",
            },
            "cutoffs": state["cutoffs"],
            "pins": {},
        },
    )
    return selected, state


def test_canonical_schedule_respects_lisbon_dst() -> None:
    policy = RetrainingPolicy.load(POLICY)
    assert canonical_monthly_logical_time(policy, "2026-01") == datetime(
        2026, 1, 8, 13, tzinfo=timezone.utc
    )
    assert canonical_monthly_logical_time(policy, "2026-07") == datetime(
        2026, 7, 8, 12, tzinfo=timezone.utc
    )


def test_month_end_report_selects_latest_created_and_rejects_tie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = tmp_path / "reporting" / "reports"
    paths = [reports / name / "report.json" for name in ("old", "latest")]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
    values = {
        "old": {
            "report_id": "old",
            "through_date": "2026-06-30",
            "created_at_utc": "2026-07-08T10:00:00Z",
        },
        "latest": {
            "report_id": "latest",
            "through_date": "2026-06-30",
            "created_at_utc": "2026-07-08T11:00:00Z",
        },
    }
    monkeypatch.setattr(
        governance,
        "load_monitoring_report",
        lambda path: values[Path(path).parent.name],
    )
    selected, _report = select_month_end_report(tmp_path, date(2026, 6, 30))
    assert selected.parent.name == "latest"

    values["old"]["created_at_utc"] = values["latest"]["created_at_utc"]
    with pytest.raises(MonthlyGovernanceError, match="ambiguous"):
        select_month_end_report(tmp_path, date(2026, 6, 30))


def test_stable_monthly_run_is_recommendation_only_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(tmp_path, monkeypatch)
    monkeypatch.setattr(
        governance,
        "run_monthly_retraining_evaluation",
        lambda _config: SimpleNamespace(
            outcome="no_trigger",
            evaluation_id="evaluation",
            evaluation_path=tmp_path / "evaluation.json",
            plan=SimpleNamespace(reasons=("no_trigger",)),
        ),
    )
    config = _config(tmp_path)
    first = run_monthly_governance(config)
    second = run_monthly_governance(config)
    assert first == second
    payload = load_monthly_governance_recommendation(
        first.recommendation_path
    )
    assert payload["retraining"]["decision"] == "evaluated"
    assert payload["stability"]["decision"] == "not_applicable_stable"
    assert payload["safeguards"] == {
        "backtest": False,
        "deployment_write": False,
        "monitoring_state_write": False,
        "promotion": False,
        "recommendation_only": True,
        "registry_write": False,
        "rollback": False,
        "stability_transition": False,
        "training": False,
    }
    recommendation_path = first.recommendation_path
    assert recommendation_path is not None
    recommendation_path.chmod(stat.S_IWRITE)
    payload["stability"]["reason"] = "tampered"
    recommendation_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MonthlyGovernanceError, match="identity is corrupt"):
        load_monthly_governance_recommendation(recommendation_path)


@pytest.mark.parametrize(
    ("count", "healthy", "exclude_first", "expected", "expected_cutoff"),
    [
        (89, True, False, "insufficient_observations", None),
        (90, True, False, "ready_for_second_manual_approval", "2026-05-01"),
        (91, True, False, "ready_for_second_manual_approval", "2026-05-01"),
        (91, True, True, "ready_for_second_manual_approval", "2026-05-02"),
        (90, False, False, "blocked_health", "2026-05-01"),
    ],
)
def test_probation_stability_uses_first_90_and_current_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    healthy: bool,
    exclude_first: bool,
    expected: str,
    expected_cutoff: str | None,
) -> None:
    config = _config(tmp_path)
    _selected, state = _patch_common(
        tmp_path,
        monkeypatch,
        status="probationary",
    )
    latest = (
        config.monitoring_store_root
        / "reporting"
        / "reports"
        / "latest"
        / "report.json"
    )
    latest.parent.mkdir(parents=True)
    latest.write_text("{}", encoding="utf-8")
    days = {
        date.fromordinal(date(2026, 2, 1).toordinal() + index).isoformat(): f"p{index}"
        for index in range(count)
    }
    ledger = {
        "active_model_era_id": "era",
        "as_issued": days,
        "actuals": {day: f"a{index}" for index, day in enumerate(days)},
    }
    monitoring_policy = json.loads(
        MONITORING_POLICY.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        governance,
        "load_monitoring_report",
        lambda _path: {
            "report_id": "latest",
            "through_date": "2026-07-08",
            "active_alerts": {} if healthy else {"rule": "alert"},
            "breaches": [],
            "quality": {"issues": []},
            "reference": {
                "policy_sha256": governance.sha256_file(MONITORING_POLICY)
            },
            "config": monitoring_policy,
            "model_era": {
                "model_era_id": "era",
                "deployment_id": "d" * 64,
                "deployment_state_id": "s" * 64,
                "deployment_generation": 2,
                "registered_model_name": "wind",
                "model_version": "2",
                "cutoffs": state["cutoffs"],
                "pins": {},
            },
        },
    )
    monkeypatch.setattr(
        governance,
        "load_prediction_evidence",
        lambda _root, prediction_id: {
            "prediction": {
                "model_era_id": "era",
                "issuance_kind": (
                    "manual"
                    if exclude_first and prediction_id == "p0"
                    else "scheduled"
                ),
                "target_date": next(
                    day for day, value in days.items() if value == prediction_id
                ),
                "prediction": 1.0,
            },
            "actual_revisions": [
                {
                    "actual_revision_id": next(
                        actual
                        for day, actual in ledger["actuals"].items()
                        if days[day] == prediction_id
                    ),
                    "target_date": next(
                        day for day, value in days.items() if value == prediction_id
                    ),
                    "actual": 1.0,
                }
            ],
        },
    )
    result = governance._stability_recommendation(
        config,
        RetrainingPolicy.load(POLICY),
        state=state,
        verified={
            "pointer_path": config.deployment_root / "state" / "current.json"
        },
        ledger=ledger,
        report_state={
            "latest_report_id": "latest",
            "latest_through_date": "2026-07-08",
            "active_alerts": {} if healthy else {"rule": "alert"},
        },
        active_era_id="era",
        lifecycle_status="probationary",
    )
    assert result["decision"] == expected
    if expected_cutoff is not None:
        assert len(result["fixed_observations"]) == 90
        assert result["observation_cutoff"] == expected_cutoff


def test_noncanonical_logical_time_fails_without_output(
    tmp_path: Path,
) -> None:
    config = MonthlyGovernanceConfig(
        **{
            **_config(tmp_path).__dict__,
            "logical_at_utc": "2026-07-08T12:00:01Z",
        }
    )
    with pytest.raises(MonthlyGovernanceError, match="policy-fixed"):
        run_monthly_governance(config)
    assert not config.output_root.exists()


def test_dry_run_is_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(tmp_path, monkeypatch)
    monkeypatch.setattr(
        governance,
        "run_monthly_retraining_evaluation",
        lambda _config: SimpleNamespace(
            outcome="no_trigger",
            evaluation_id="evaluation",
            evaluation_path=None,
            plan=SimpleNamespace(reasons=("no_trigger",)),
        ),
    )
    config = _config(tmp_path, dry_run=True)
    result = run_monthly_governance(config)
    assert result.status == "planned"
    assert result.recommendation_path is None
    assert not config.output_root.exists()
    assert not config.evaluation_output_root.exists()
