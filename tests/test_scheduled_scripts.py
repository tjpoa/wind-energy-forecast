from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"


def test_windows_daily_runner_is_owner_guarded() -> None:
    source = (SCRIPTS / "run_scheduled_batch.ps1").read_text(encoding="utf-8")
    assert "manage_scheduler_owner.py" in source
    assert "--scheduler windows_task_scheduler" in source
    assert "--workflow historical_daily_batch" in source
    assert "release" in source


def test_windows_monthly_task_runs_recommendations_only() -> None:
    runner = (
        SCRIPTS / "run_scheduled_monthly_governance.ps1"
    ).read_text(encoding="utf-8")
    registration = (
        SCRIPTS / "register_local_monthly_governance_task.ps1"
    ).read_text(encoding="utf-8")
    assert "run_monthly_governance.py" in runner
    assert "--workflow monthly_governance" in runner
    assert "MSFT_TaskMonthlyTrigger" in registration
    assert "DaysOfMonth = [uint32]128" in registration
    assert "StartWhenAvailable" in registration
    assert "IgnoreNew" in registration
    for prohibited in (
        "backtest_retraining_candidate.py",
        "train_v2_reference.py",
        "register_retraining_candidate.py",
        "manage_v2_deployment.py",
    ):
        assert prohibited not in runner
