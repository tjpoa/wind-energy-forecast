"""Bootstrap the accepted v2 bundle into deployment governance exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wind_forecast.retraining_deployment import (
    DeploymentBootstrapConfig,
    bootstrap_v2_deployment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate accepted v2 evidence and initialize generation-one "
            "stable/champion deployment state."
        )
    )
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--monitoring-store-root", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--registry-lock-root", type=Path, required=True)
    parser.add_argument("--registered-model-name", required=True)
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--approval-path", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument(
        "--expect-no-deployment-pointer",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--expect-no-v2-registry-state",
        action="store_true",
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run and (
        args.approval_path is None or args.approval_sha256 is None
    ):
        raise SystemExit(
            "ERROR: bootstrap requires --approval-path and --approval-sha256."
        )
    result = bootstrap_v2_deployment(
        DeploymentBootstrapConfig(
            model_bundle=args.model_bundle,
            calibration_dir=args.calibration_dir,
            monitoring_store_root=args.monitoring_store_root,
            deployment_root=args.deployment_root,
            registry_lock_root=args.registry_lock_root,
            registered_model_name=args.registered_model_name,
            tracking_uri=args.mlflow_tracking_uri,
            approval_path=args.approval_path,
            approval_sha256=args.approval_sha256,
            expect_no_deployment_pointer=args.expect_no_deployment_pointer,
            expect_no_v2_registry_state=args.expect_no_v2_registry_state,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
