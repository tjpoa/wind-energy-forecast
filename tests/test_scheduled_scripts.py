import os
from pathlib import Path
import subprocess

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"


def _script(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_windows_daily_runner_is_owner_guarded() -> None:
    source = _script("run_scheduled_batch.ps1")
    assert "manage_scheduler_owner.py" in source
    assert "--scheduler windows_task_scheduler" in source
    assert "--workflow historical_daily_batch" in source
    assert "release" in source


def test_windows_monthly_task_runs_recommendations_only() -> None:
    runner = _script("run_scheduled_monthly_governance.ps1")
    registration = _script("register_local_monthly_governance_task.ps1")
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


def test_local_mlflow_runner_is_fixed_to_existing_loopback_store() -> None:
    source = _script("run_local_mlflow.ps1")
    assert 'sqlite:///var/mlflow/mlflow.db' in source
    assert './var/mlflow/artifacts' in source
    assert '--host "127.0.0.1"' in source
    assert '--port "5000"' in source
    assert 'var\\local_services' in source
    assert '*>> $logFile' in source
    assert '$nativeErrorActionPreference = $ErrorActionPreference' in source
    assert '$ErrorActionPreference = "Continue"' in source
    assert '$ErrorActionPreference = $nativeErrorActionPreference' in source
    assert 'Resolve-Path -LiteralPath $PythonExecutable' not in source
    for prohibited in ("--serve-artifacts", "0.0.0.0", "Start-Process"):
        assert prohibited not in source


def test_local_operational_api_runner_has_bounded_explicit_configuration() -> None:
    source = _script("run_local_operational_api.ps1")
    assert '[ValidateRange(1, 600)]' in source
    assert '$MlflowHealthTimeoutSeconds = 120' in source
    assert 'if ($remaining -lt 1)' in source
    assert 'http://127.0.0.1:5000/health' in source
    assert 'WIND_FORECAST_DEPLOYMENT_ROOT' in source
    assert 'WIND_FORECAST_MONITORING_STORE_ROOT' in source
    assert 'WIND_FORECAST_OPERATIONAL_MODEL_BUNDLE' in source
    assert 'WIND_FORECAST_OPERATIONAL_CALIBRATION_DIR' in source
    assert '$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"' in source
    assert '$env:WIND_FORECAST_OPERATIONAL_PROJECTION_MODE = "disabled"' in source
    assert 'wind_forecast.api:app' in source
    assert '--host "127.0.0.1"' in source
    assert '--port "8000"' in source
    assert 'var\\local_services' in source
    assert '$nativeErrorActionPreference = $ErrorActionPreference' in source
    assert '$ErrorActionPreference = "Continue"' in source
    assert '$ErrorActionPreference = $nativeErrorActionPreference' in source
    for prohibited in ("--reload", "--env-file", "0.0.0.0", "docker", "proxy"):
        assert prohibited not in source.lower()


def test_local_service_registrations_are_reviewable_and_do_not_start_tasks() -> None:
    contracts = {
        "register_local_mlflow_task.ps1": "WindForecastMlflow",
        "register_local_operational_api_task.ps1": (
            "WindForecastOperationalApi"
        ),
    }
    for filename, task_name in contracts.items():
        source = _script(filename)
        assert 'SupportsShouldProcess = $true' in source
        assert f'$TaskName = "{task_name}"' in source
        assert 'New-ScheduledTaskTrigger -AtLogOn -User $currentUser' in source
        assert '-LogonType Interactive' in source
        assert '-RunLevel Limited' in source
        assert '-ExecutionTimeLimit (New-TimeSpan -Seconds 0)' in source
        assert '-RestartCount 3' in source
        assert '-RestartInterval (New-TimeSpan -Minutes 1)' in source
        assert '-MultipleInstances IgnoreNew' in source
        assert '-StartWhenAvailable' in source
        assert '$powershell = Join-Path $PSHOME "powershell.exe"' in source
        assert '-Execute $powershell' in source
        assert 'Register-ScheduledTask' in source
        assert 'Start-ScheduledTask' not in source
        assert 'Unregister-ScheduledTask' not in source
    api_registration = _script("register_local_operational_api_task.ps1")
    assert '[string]$ModelBundle' in api_registration
    assert '[string]$CalibrationDirectory' in api_registration
    assert '"-ModelBundle", (Quote-TaskArgument $model)' in api_registration
    assert (
        '"-CalibrationDirectory", (Quote-TaskArgument $calibration)'
        in api_registration
    )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_mlflow_runner_preserves_native_stderr_and_exit_code(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "var" / "mlflow" / "artifacts").mkdir(parents=True)
    (repository / "var" / "mlflow" / "mlflow.db").touch()
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "@echo off\n"
        "echo native-stderr-evidence 1>&2\n"
        "exit /b 7\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPTS / "run_local_mlflow.ps1"),
            "-PythonExecutable",
            str(fake_python),
            "-RepositoryRoot",
            str(repository),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 7
    logs = list((repository / "var" / "local_services").glob("mlflow-*.log"))
    assert len(logs) == 1
    assert "native-stderr-evidence" in logs[0].read_text(encoding="utf-16")
