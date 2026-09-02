"""Run the shared fail-closed Terraform post-apply drift gate."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


_OPERATION_LABELS = {
    "deployment": "post-deployment",
    "rollback": "post-rollback",
    "foundation-apply": "post-foundation",
}
_PLAN_CHANGED = 2

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument("--operation", choices=sorted(_OPERATION_LABELS), required=True)
    return parser.parse_args(argv)


def run_drift_check(
    *,
    directory: Path,
    plan_path: Path,
    operation: str,
    runner: CommandRunner = subprocess.run,
) -> int:
    """Run Terraform and return a workflow-safe exit code."""
    if operation not in _OPERATION_LABELS:
        raise ValueError(f"Unsupported Terraform drift operation: {operation}")

    label = _OPERATION_LABELS[operation]
    plan_command = [
        "terraform",
        f"-chdir={directory}",
        "plan",
        "-detailed-exitcode",
        "-input=false",
        "-lock-timeout=5m",
        f"-out={plan_path}",
    ]
    try:
        completed = runner(plan_command, check=False, text=True)
    except OSError as exc:
        _write_output(1)
        _append_summary(
            [
                f"## Terraform {label} drift check",
                "",
                "- Result: Terraform could not start; the workflow failed closed.",
            ]
        )
        print(f"Terraform {label} plan could not start: {exc}", file=sys.stderr)
        return 1

    plan_exit = completed.returncode
    _write_output(plan_exit)

    if plan_exit == 0:
        _append_summary(
            [
                f"## Terraform {label} drift check",
                "",
                "- Result: drift-free (Terraform detailed exit code 0).",
                "- The binary plan remains on this ephemeral runner and is not uploaded.",
            ]
        )
        return 0

    if plan_exit == _PLAN_CHANGED:
        changes = _read_plan_changes(directory, plan_path, runner)
        summary = [
            f"## Terraform {label} drift check",
            "",
            "- Result: drift detected; the workflow failed closed.",
            "- The binary plan remains on this ephemeral runner and is not uploaded.",
        ]
        if changes is None:
            summary.append(
                "- Change details were unavailable; no raw Terraform plan was emitted."
            )
        summary.extend(
            [
                "",
                "```json",
                json.dumps({"resource_changes": changes or []}, sort_keys=True),
                "```",
            ]
        )
        _append_summary(summary)
        print(f"Terraform detected drift after the protected {operation}.", file=sys.stderr)
        if changes is not None:
            print(json.dumps({"resource_changes": changes}, sort_keys=True), file=sys.stderr)
        return 1

    _append_summary(
        [
            f"## Terraform {label} drift check",
            "",
            f"- Result: plan failed with exit code {plan_exit}; the workflow failed closed.",
        ]
    )
    print(
        f"Terraform {label} plan failed with exit code {plan_exit}.",
        file=sys.stderr,
    )
    return plan_exit if 1 <= plan_exit <= 255 else 1


def _read_plan_changes(
    directory: Path,
    plan_path: Path,
    runner: CommandRunner,
) -> list[dict[str, Any]] | None:
    try:
        completed = runner(
            [
                "terraform",
                f"-chdir={directory}",
                "show",
                "-json",
                str(plan_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    try:
        plan = json.loads(completed.stdout)
    except (TypeError, ValueError):
        return None
    return _sanitize_resource_changes(plan)


def _sanitize_resource_changes(plan: Any) -> list[dict[str, Any]]:
    if not isinstance(plan, Mapping):
        return []
    resource_changes = plan.get("resource_changes")
    if not isinstance(resource_changes, list):
        return []

    changes: list[dict[str, Any]] = []
    for resource in resource_changes:
        if not isinstance(resource, Mapping):
            continue
        address = resource.get("address")
        change = resource.get("change")
        if not isinstance(address, str) or not isinstance(change, Mapping):
            continue
        actions = change.get("actions")
        if not isinstance(actions, list) or not all(
            isinstance(action, str) for action in actions
        ):
            continue
        if actions == ["no-op"]:
            continue
        changes.append({"address": address, "actions": actions})
    return changes


def _write_output(plan_exit: int) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(f"plan_exit_code={plan_exit}\n")


def _append_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_drift_check(
        directory=args.directory,
        plan_path=args.plan_path,
        operation=args.operation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
