"""CLI adapter for the pure Azure workflow policy functions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from azure_workflow_policy import (
    WorkflowPolicyError,
    build_receipt,
    forbidden_plan_changes,
    forbidden_tracked_paths,
    require_configuration,
    require_release_enabled,
    require_run_id,
    validate_active_images,
    validate_release_manifest,
    validate_rollback_manifest,
)


BASE_CONFIGURATION = (
    "AZURE_ACR_NAME",
    "AZURE_TENANT_ID",
    "AZURE_SUBSCRIPTION_ID",
    "TFSTATE_RESOURCE_GROUP_NAME",
    "TFSTATE_STORAGE_ACCOUNT_NAME",
)

RELEASE_CONFIGURATION = BASE_CONFIGURATION + (
    "AZURE_RESOURCE_GROUP",
    "AZURE_BUDGET_ALERT_EMAIL",
    "AZURE_BUDGET_START_DATE",
    "AZURE_BUDGET_END_DATE",
)

PUBLISH_CONFIGURATION = (
    "AZURE_ACR_NAME",
    "AZURE_RESOURCE_GROUP",
    "AZURE_TENANT_ID",
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_PUBLISHER_CLIENT_ID",
)

MODE_CONFIGURATION = {
    "release-publish": PUBLISH_CONFIGURATION,
    "release-plan": RELEASE_CONFIGURATION + ("AZURE_PLANNER_CLIENT_ID",),
    "release-deploy": RELEASE_CONFIGURATION + ("AZURE_DEPLOYER_CLIENT_ID",),
    "rollback-request": (),
    "rollback-plan": RELEASE_CONFIGURATION + ("AZURE_PLANNER_CLIENT_ID",),
    "rollback-deploy": RELEASE_CONFIGURATION + ("AZURE_DEPLOYER_CLIENT_ID",),
    "foundation-plan": BASE_CONFIGURATION + ("AZURE_PLANNER_CLIENT_ID",),
    "foundation-apply": BASE_CONFIGURATION + ("AZURE_DEPLOYER_CLIENT_ID",),
    "legacy-bicep": (
        "AZURE_TENANT_ID",
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_CLIENT_ID",
        "AZURE_PRINCIPAL_OBJECT_ID",
        "AZURE_BUDGET_ALERT_EMAIL",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--mode", choices=sorted(MODE_CONFIGURATION), required=True)
    preflight.add_argument("--release-enabled", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--path", type=Path, required=True)
    manifest.add_argument("--expected-source-sha")
    manifest.add_argument("--expected-release-run-id")

    rollback_manifest = subparsers.add_parser("rollback-manifest")
    rollback_manifest.add_argument("--path", type=Path, required=True)
    rollback_manifest.add_argument("--expected-rollback-run-id")

    run_id = subparsers.add_parser("run-id")
    run_id.add_argument("--value", required=True)
    run_id.add_argument("--name", default="run_id")

    plan = subparsers.add_parser("plan")
    plan.add_argument("--path", type=Path)
    plan.add_argument("--stdin", action="store_true")

    active = subparsers.add_parser("active-images")
    active.add_argument("--manifest", type=Path, required=True)
    active.add_argument("--observed", type=Path, required=True)
    active.add_argument("--kind", choices=("release", "rollback"), required=True)

    receipt = subparsers.add_parser("receipt")
    receipt.add_argument("--operation", choices=("release", "rollback"), required=True)
    receipt.add_argument("--manifest", type=Path, required=True)
    receipt.add_argument("--active-images", type=Path, required=True)
    receipt.add_argument("--smoke-tests", type=Path, required=True)
    receipt.add_argument("--api-revision", required=True)
    receipt.add_argument("--frontend-revision", required=True)
    receipt.add_argument("--terraform-post-plan-exit-code", type=int, required=True)
    receipt.add_argument("--workflow-run-id", required=True)
    receipt.add_argument("--output", type=Path, required=True)

    subparsers.add_parser("tracked")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "preflight":
            require_release_enabled(args.release_enabled)
            require_configuration(os.environ, MODE_CONFIGURATION[args.mode])
            print(f"Azure workflow preflight passed for {args.mode}.")
        elif args.command == "manifest":
            validate_release_manifest(
                _read_json(args.path),
                expected_source_sha=args.expected_source_sha,
                expected_release_run_id=args.expected_release_run_id,
            )
            print("Release manifest validation passed.")
        elif args.command == "rollback-manifest":
            validate_rollback_manifest(
                _read_json(args.path),
                expected_rollback_run_id=args.expected_rollback_run_id,
            )
            print("Rollback manifest validation passed.")
        elif args.command == "run-id":
            require_run_id(args.value, name=args.name)
            print("Run id validation passed.")
        elif args.command == "plan":
            plan = _read_json(args.path) if not args.stdin else _read_stdin_json()
            forbidden = forbidden_plan_changes(plan)
            if forbidden:
                addresses = ", ".join(item["address"] for item in forbidden)
                raise WorkflowPolicyError(
                    "Terraform plan contains forbidden delete/replacement changes: "
                    + addresses
                )
            print("Terraform plan contains no deletes or replacements.")
        elif args.command == "active-images":
            manifest = _read_json(args.manifest)
            if args.kind == "release":
                identity = validate_release_manifest(manifest)
            else:
                identity = validate_rollback_manifest(manifest)
            validate_active_images(identity, _read_json(args.observed))
            print("Active image references match the registered manifest.")
        elif args.command == "receipt":
            receipt = build_receipt(
                operation=args.operation,
                manifest=_read_json(args.manifest),
                active_images=_read_json(args.active_images),
                api_revision=args.api_revision,
                frontend_revision=args.frontend_revision,
                smoke_tests=_read_json(args.smoke_tests),
                terraform_post_plan_exit_code=args.terraform_post_plan_exit_code,
                workflow_run_id=args.workflow_run_id,
            )
            args.output.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print("Deployment receipt created.")
        elif args.command == "tracked":
            tracked = subprocess.run(
                ["git", "ls-files", "-z"],
                check=True,
                capture_output=True,
            ).stdout.decode("utf-8").split("\0")
            forbidden = forbidden_tracked_paths(path for path in tracked if path)
            if forbidden:
                raise WorkflowPolicyError(
                    "Forbidden tracked deployment artifacts: " + ", ".join(forbidden)
                )
            print("No forbidden state, plan, variable, password, or client-secret paths are tracked.")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WorkflowPolicyError(f"{path} must contain a JSON object.")
    return data


def _read_stdin_json() -> dict[str, Any]:
    data = json.load(sys.stdin)
    if not isinstance(data, dict):
        raise WorkflowPolicyError("Terraform plan JSON must be an object.")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
