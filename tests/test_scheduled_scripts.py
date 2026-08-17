import json
import os
from pathlib import Path
import subprocess
import time

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"


def _script(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def _read_log(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError(f"Could not decode runner log {path}")


def _events(repository: Path, runner: str) -> list[dict[str, object]]:
    paths = list(
        (repository / "var" / "local_services").glob(
            f"{runner}-*.events.jsonl"
        )
    )
    assert len(paths) == 1
    return [
        json.loads(line)
        for line in paths[0].read_text(encoding="utf-8").splitlines()
    ]


def _assert_event_contract(events: list[dict[str, object]], runner: str) -> None:
    assert events
    run_ids = {event["run_id"] for event in events}
    runner_pids = {event["runner_pid"] for event in events}
    assert len(run_ids) == 1
    assert len(runner_pids) == 1
    assert all(
        event["schema_version"] == "wind_forecast.runner_event.v1"
        for event in events
    )
    assert all(event["runner"] == runner for event in events)
    assert all(
        set(event)
        == {
            "schema_version",
            "timestamp_utc",
            "runner",
            "run_id",
            "stage",
            "status",
            "runner_pid",
            "child_exit_code",
            "exception_type",
            "exception_message",
        }
        for event in events
    )


def _run_powershell(
    script: str,
    arguments: list[str],
    **kwargs: object,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPTS / script),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        **kwargs,
    )


def _batch_arguments(repository: Path, fake_python: Path) -> list[str]:
    return [
        "-PythonExecutable",
        str(fake_python),
        "-RepositoryRoot",
        str(repository),
        "-ModelBundle",
        "model-bundle",
        "-CalibrationDirectory",
        "calibration",
        "-DeploymentRoot",
        "deployment",
        "-SchedulerStateRoot",
        "scheduler",
        "-EnvironmentId",
        "local",
        "-ActivationDate",
        "2026-06-28",
    ]


def _prepare_batch_repository(repository: Path) -> None:
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_batch_pipeline.py").touch()
    (scripts / "manage_scheduler_owner.py").touch()


def test_windows_daily_runner_is_owner_guarded() -> None:
    source = _script("run_scheduled_batch.ps1")
    assert "manage_scheduler_owner.py" in source
    assert '"--scheduler", "windows_task_scheduler"' in source
    assert '"--workflow", "historical_daily_batch"' in source
    assert "release" in source


def test_windows_daily_registration_uses_s4u_without_changing_schedule() -> None:
    source = _script("register_local_batch_task.ps1")
    assert 'SupportsShouldProcess = $true' in source
    assert '$TaskName = "WindForecastHistoricalBatch"' in source
    assert (
        "$currentUser = "
        "[System.Security.Principal.WindowsIdentity]::GetCurrent().Name"
        in source
    )
    assert 'New-ScheduledTaskTrigger -Daily -At "12:00"' in source
    assert '-LogonType S4U' in source
    assert 'LogonType = "S4U"' in source
    assert '-LogonType Interactive' not in source
    assert '-RunLevel Limited' in source
    assert '-ExecutionTimeLimit (New-TimeSpan -Hours 6)' in source
    assert '-RestartCount 2' in source
    assert '-RestartInterval (New-TimeSpan -Minutes 30)' in source
    assert '-MultipleInstances IgnoreNew' in source
    assert '-StartWhenAvailable' in source
    assert '"-File", (Quote-TaskArgument $runner)' in source
    assert '"-EnvironmentId", (Quote-TaskArgument $EnvironmentId)' in source
    assert 'StartsTask = $false' in source
    assert 'Register-ScheduledTask' in source
    assert 'Start-ScheduledTask' not in source
    assert 'Unregister-ScheduledTask' not in source


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_windows_daily_registration_whatif_reports_s4u_without_starting(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    for relative in (
        "scripts",
        "model-bundle",
        "calibration",
        "deployment",
        "scheduler",
    ):
        (repository / relative).mkdir(parents=True, exist_ok=True)
    (repository / "scripts" / "run_scheduled_batch.ps1").touch()
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text("@echo off\nexit /b 0\n", encoding="utf-8")

    completed = _run_powershell(
        "register_local_batch_task.ps1",
        [
            "-PythonExecutable",
            str(fake_python),
            "-RepositoryRoot",
            str(repository),
            "-ModelBundle",
            str(repository / "model-bundle"),
            "-CalibrationDirectory",
            str(repository / "calibration"),
            "-DeploymentRoot",
            str(repository / "deployment"),
            "-SchedulerStateRoot",
            str(repository / "scheduler"),
            "-EnvironmentId",
            "local",
            "-ActivationDate",
            "2026-06-28",
            "-WhatIf",
        ],
    )

    assert completed.returncode == 0, completed.stderr
    assert "Daily at 12:00 local time" in completed.stdout
    assert "LogonType" in completed.stdout
    assert "S4U" in completed.stdout
    assert "StartsTask" in completed.stdout
    assert "False" in completed.stdout


def test_windows_monthly_task_runs_recommendations_only() -> None:
    runner = _script("run_scheduled_monthly_governance.ps1")
    registration = _script("register_local_monthly_governance_task.ps1")
    assert "run_monthly_governance.py" in runner
    assert "--workflow monthly_governance" in runner
    assert "MSFT_TaskMonthlyTrigger" in registration
    assert "DaysOfMonth = [uint32]128" in registration
    assert "StartWhenAvailable" in registration
    assert "IgnoreNew" in registration
    assert "-LogonType Interactive" in registration
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
    assert '--workers "1"' in source
    assert '--host "127.0.0.1"' in source
    assert '--port "5000"' in source
    assert 'var\\local_services' in source
    assert '*>> $outputFile' in source
    assert 'schema_version = "wind_forecast.runner_event.v1"' in source
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
    assert '*>> $outputFile' in source
    assert 'schema_version = "wind_forecast.runner_event.v1"' in source
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
        assert '-LogonType S4U' in source
        assert 'LogonType = "S4U"' in source
        assert '-LogonType Interactive' not in source
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
        "echo native-args:%* 1>&2\n"
        "echo native-stderr-evidence 1>&2\n"
        "exit /b 7\n",
        encoding="utf-8",
    )

    completed = _run_powershell(
        "run_local_mlflow.ps1",
        [
            "-PythonExecutable",
            str(fake_python),
            "-RepositoryRoot",
            str(repository),
        ],
    )

    assert completed.returncode == 7
    logs = list(
        (repository / "var" / "local_services").glob(
            "mlflow-*.output.log"
        )
    )
    assert len(logs) == 1
    output = _read_log(logs[0])
    assert "native-stderr-evidence" in output
    assert "--workers 1" in output
    events = _events(repository, "mlflow")
    _assert_event_contract(events, "mlflow")
    assert events[-2]["stage"] == "child"
    assert events[-2]["child_exit_code"] == 7
    assert events[-1]["stage"] == "runner_exit"
    assert events[-1]["child_exit_code"] == 7


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_mlflow_setup_failure_is_logged_and_sanitized(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    secret = "DO-NOT-LOG-THIS-SECRET"
    environment = {**os.environ, "WIND_FORECAST_SECRET_TEST_MARKER": secret}

    completed = _run_powershell(
        "run_local_mlflow.ps1",
        [
            "-PythonExecutable",
            str(repository / secret / "python.exe"),
            "-RepositoryRoot",
            str(repository),
        ],
        env=environment,
    )

    assert completed.returncode == 1
    events = _events(repository, "mlflow")
    _assert_event_contract(events, "mlflow")
    assert [event["stage"] for event in events] == [
        "observability",
        "setup",
        "setup",
        "runner_exit",
    ]
    assert events[-2]["status"] == "failed"
    assert events[-2]["exception_type"]
    assert events[-1]["child_exit_code"] is None
    all_evidence = completed.stdout + completed.stderr
    for path in (repository / "var" / "local_services").iterdir():
        all_evidence += _read_log(path)
    assert secret not in all_evidence


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_api_health_timeout_writes_events_before_child_start(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    for relative in (
        "deployment",
        "monitoring",
        "model",
        "calibration",
        "src/wind_forecast",
    ):
        (repository / relative).mkdir(parents=True, exist_ok=True)
    (repository / "src" / "wind_forecast" / "api.py").touch()
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text("@echo off\nexit /b 0\n", encoding="utf-8")

    completed = _run_powershell(
        "run_local_operational_api.ps1",
        [
            "-PythonExecutable",
            str(fake_python),
            "-RepositoryRoot",
            str(repository),
            "-DeploymentRoot",
            str(repository / "deployment"),
            "-MonitoringStoreRoot",
            str(repository / "monitoring"),
            "-ModelBundle",
            str(repository / "model"),
            "-CalibrationDirectory",
            str(repository / "calibration"),
            "-MlflowHealthTimeoutSeconds",
            "1",
        ],
    )

    assert completed.returncode == 1
    events = _events(repository, "operational-api")
    _assert_event_contract(events, "operational-api")
    assert any(
        event["stage"] == "health_wait" and event["status"] == "failed"
        for event in events
    )
    assert not any(event["stage"] == "child" for event in events)
    assert events[-1]["child_exit_code"] is None


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_batch_preserves_child_json_and_successful_lease_lifecycle(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _prepare_batch_repository(repository)
    fake_python = tmp_path / "fake-python.cmd"
    child_json = '{"schema_version":"child.v1","status":"succeeded"}'
    fake_python.write_text(
        "@echo off\n"
        "if \"%~2\"==\"acquire\" (echo {\"lease_id\":\"lease-1\"}& exit /b 0)\n"
        f'if "%~2"=="run" (echo {child_json}& exit /b 0)\n'
        "if \"%~2\"==\"release\" (echo {\"status\":\"released\"}& exit /b 0)\n"
        "exit /b 99\n",
        encoding="utf-8",
    )

    completed = _run_powershell(
        "run_scheduled_batch.ps1",
        _batch_arguments(repository, fake_python),
    )

    assert completed.returncode == 0
    assert completed.stdout == child_json + "\n"
    events = _events(repository, "scheduled-batch")
    _assert_event_contract(events, "scheduled-batch")
    stages = [(event["stage"], event["status"]) for event in events]
    assert ("lease_acquire", "succeeded") in stages
    assert ("child", "succeeded") in stages
    assert ("lease_release", "succeeded") in stages
    assert stages[-1] == ("runner_exit", "succeeded")


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_batch_preflight_failure_preserves_stderr_without_manifest(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _prepare_batch_repository(repository)
    fake_python = tmp_path / "fake-python.cmd"
    error_json = '{"schema_version":"batch_cli_error.v1","status":"failed"}'
    fake_python.write_text(
        "@echo off\n"
        "if \"%~2\"==\"acquire\" (echo {\"lease_id\":\"lease-1\"}& exit /b 0)\n"
        f'if "%~2"=="run" (echo {error_json} 1>&2& exit /b 7)\n'
        "if \"%~2\"==\"release\" (echo {\"status\":\"released\"}& exit /b 0)\n"
        "exit /b 99\n",
        encoding="utf-8",
    )

    completed = _run_powershell(
        "run_scheduled_batch.ps1",
        _batch_arguments(repository, fake_python),
    )

    assert completed.returncode == 7
    assert completed.stdout == ""
    assert error_json in completed.stderr
    orchestration = repository / "data" / "processed" / "v2" / "orchestration"
    assert not orchestration.exists()
    output_logs = list(
        (repository / "var" / "local_services").glob(
            "scheduled-batch-*.output.log"
        )
    )
    assert len(output_logs) == 1
    assert error_json in _read_log(output_logs[0])
    events = _events(repository, "scheduled-batch")
    assert any(
        event["stage"] == "child"
        and event["status"] == "failed"
        and event["child_exit_code"] == 7
        for event in events
    )
    assert any(
        event["stage"] == "lease_release" and event["status"] == "succeeded"
        for event in events
    )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_batch_release_failure_is_fail_closed(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _prepare_batch_repository(repository)
    fake_python = tmp_path / "fake-python.cmd"
    child_json = '{"schema_version":"child.v1","status":"succeeded"}'
    fake_python.write_text(
        "@echo off\n"
        "if \"%~2\"==\"acquire\" (echo {\"lease_id\":\"lease-1\"}& exit /b 0)\n"
        f'if "%~2"=="run" (echo {child_json}& exit /b 0)\n'
        "if \"%~2\"==\"release\" (echo release-failed 1>&2& exit /b 9)\n"
        "exit /b 99\n",
        encoding="utf-8",
    )

    completed = _run_powershell(
        "run_scheduled_batch.ps1",
        _batch_arguments(repository, fake_python),
    )

    assert completed.returncode == 1
    assert completed.stdout == child_json + "\n"
    events = _events(repository, "scheduled-batch")
    assert any(
        event["stage"] == "lease_release" and event["status"] == "failed"
        for event in events
    )
    assert events[-1]["stage"] == "runner_exit"
    assert events[-1]["status"] == "failed"
    assert events[-1]["child_exit_code"] == 0


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_batch_acquire_failure_stops_before_child_or_release(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _prepare_batch_repository(repository)
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "@echo off\n"
        "if \"%~2\"==\"acquire\" (echo acquire-failed 1>&2& exit /b 8)\n"
        "exit /b 99\n",
        encoding="utf-8",
    )

    completed = _run_powershell(
        "run_scheduled_batch.ps1",
        _batch_arguments(repository, fake_python),
    )

    assert completed.returncode == 1
    assert "acquire-failed" in completed.stderr
    events = _events(repository, "scheduled-batch")
    assert any(
        event["stage"] == "lease_acquire" and event["status"] == "failed"
        for event in events
    )
    assert not any(
        event["stage"] in {"child", "lease_release"} for event in events
    )
    assert events[-1]["child_exit_code"] is None


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_batch_mirrors_native_output_before_child_exits(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _prepare_batch_repository(repository)
    fake_python = tmp_path / "fake-python.cmd"
    child_json = '{"schema_version":"child.v1","status":"succeeded"}'
    fake_python.write_text(
        "@echo off\n"
        "if \"%~2\"==\"acquire\" (echo {\"lease_id\":\"lease-1\"}& exit /b 0)\n"
        "if \"%~2\"==\"run\" ("
        "echo early-native-evidence 1>&2& "
        "ping 127.0.0.1 -n 4 >nul& "
        f"echo {child_json}& exit /b 0)\n"
        "if \"%~2\"==\"release\" (echo {\"status\":\"released\"}& exit /b 0)\n"
        "exit /b 99\n",
        encoding="utf-8",
    )
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPTS / "run_scheduled_batch.ps1"),
        *_batch_arguments(repository, fake_python),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    observed_while_running = False
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline:
        logs = list(
            (repository / "var" / "local_services").glob(
                "scheduled-batch-*.output.log"
            )
        )
        if logs and "early-native-evidence" in _read_log(logs[0]):
            observed_while_running = process.poll() is None
            break
        time.sleep(0.05)
    stdout, stderr = process.communicate(timeout=10)

    assert observed_while_running
    assert process.returncode == 0
    assert stdout == child_json + "\n"
    assert "early-native-evidence" in stderr


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell 5.1")
def test_batch_logger_failure_after_acquire_still_releases_lease(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _prepare_batch_repository(repository)
    release_marker = tmp_path / "release-attempted.txt"
    events_glob = repository / "var" / "local_services" / "*.events.jsonl"
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "@echo off\n"
        "if \"%~2\"==\"acquire\" goto acquire\n"
        "if \"%~2\"==\"release\" goto release\n"
        "exit /b 99\n"
        ":acquire\n"
        "echo {\"lease_id\":\"lease-1\"}\n"
        f'for %%F in ("{events_glob}") do (del "%%~fF" & mkdir "%%~fF")\n'
        "exit /b 0\n"
        ":release\n"
        f'echo attempted>"{release_marker}"\n'
        "exit /b 0\n",
        encoding="utf-8",
    )

    completed = _run_powershell(
        "run_scheduled_batch.ps1",
        _batch_arguments(repository, fake_python),
    )

    assert completed.returncode == 1
    assert release_marker.read_text(encoding="utf-8").strip() == "attempted"
