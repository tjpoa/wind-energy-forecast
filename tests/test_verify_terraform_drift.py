from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from scripts.verify_terraform_drift import run_drift_check


class FakeTerraform:
    def __init__(self, plan_exit: int, plan_payload: dict[str, Any] | None = None) -> None:
        self.plan_exit = plan_exit
        self.plan_payload = plan_payload
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if "show" in command:
            stdout = json.dumps(self.plan_payload) if self.plan_payload is not None else "{}"
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(command, self.plan_exit, stdout="", stderr="")


def _outputs(tmp_path: Path, monkeypatch: Any) -> tuple[Path, Path]:
    output_path = tmp_path / "github-output.txt"
    summary_path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    return output_path, summary_path


def test_drift_check_accepts_a_clean_plan_and_writes_exit_code(tmp_path: Path, monkeypatch: Any) -> None:
    output_path, summary_path = _outputs(tmp_path, monkeypatch)
    runner = FakeTerraform(plan_exit=0)

    result = run_drift_check(
        directory=Path("infra/azure/terraform/production"),
        plan_path=Path("post-deployment.tfplan"),
        operation="deployment",
        runner=runner,
    )

    assert result == 0
    assert output_path.read_text(encoding="utf-8") == "plan_exit_code=0\n"
    assert "drift-free" in summary_path.read_text(encoding="utf-8")
    assert len(runner.commands) == 1
    assert "-detailed-exitcode" in runner.commands[0]
    assert "-input=false" in runner.commands[0]
    assert "-lock-timeout=5m" in runner.commands[0]
    assert "-out=post-deployment.tfplan" in runner.commands[0]


def test_drift_check_reports_only_sanitized_resource_changes(tmp_path: Path, monkeypatch: Any) -> None:
    output_path, summary_path = _outputs(tmp_path, monkeypatch)
    runner = FakeTerraform(
        plan_exit=2,
        plan_payload={
            "resource_changes": [
                {
                    "address": "azurerm_container_app.api",
                    "change": {"actions": ["update"], "sensitive_value": "must-not-leak"},
                },
                {"address": "azurerm_container_app.noop", "change": {"actions": ["no-op"]}},
            ]
        },
    )

    result = run_drift_check(
        directory=Path("infra/azure/terraform/production"),
        plan_path=Path("post-deployment.tfplan"),
        operation="deployment",
        runner=runner,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert result == 1
    assert output_path.read_text(encoding="utf-8") == "plan_exit_code=2\n"
    assert '"address": "azurerm_container_app.api"' in summary
    assert '"actions": ["update"]' in summary
    assert "must-not-leak" not in summary
    assert "azurerm_container_app.noop" not in summary
    assert len(runner.commands) == 2


def test_drift_check_propagates_terraform_errors_without_reading_plan(
    tmp_path: Path, monkeypatch: Any
) -> None:
    output_path, summary_path = _outputs(tmp_path, monkeypatch)
    runner = FakeTerraform(plan_exit=1)

    result = run_drift_check(
        directory=Path("infra/azure/terraform/foundation"),
        plan_path=Path("post-foundation.tfplan"),
        operation="foundation-apply",
        runner=runner,
    )

    assert result == 1
    assert output_path.read_text(encoding="utf-8") == "plan_exit_code=1\n"
    assert "exit code 1" in summary_path.read_text(encoding="utf-8")
    assert len(runner.commands) == 1


def test_drift_check_fails_closed_when_plan_details_cannot_be_read(
    tmp_path: Path, monkeypatch: Any
) -> None:
    output_path, summary_path = _outputs(tmp_path, monkeypatch)

    class ShowFails(FakeTerraform):
        def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            self.commands.append(command)
            if "show" in command:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="sensitive")
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="")

    runner = ShowFails(plan_exit=2)
    result = run_drift_check(
        directory=Path("infra/azure/terraform/production"),
        plan_path=Path("post-rollback.tfplan"),
        operation="rollback",
        runner=runner,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert result == 1
    assert output_path.read_text(encoding="utf-8") == "plan_exit_code=2\n"
    assert "Change details were unavailable" in summary
    assert "sensitive" not in summary
    assert len(runner.commands) == 2
