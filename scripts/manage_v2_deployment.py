"""Plan or execute one explicitly approved V2 deployment transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.retraining_lifecycle import (
    ExpectedDeploymentState,
    LifecycleConfig,
    execute_lifecycle_transition,
)


def _alias(value: str) -> str | None:
    if value == "none":
        return None
    if not value.strip():
        raise argparse.ArgumentTypeError("Alias expectation must be a version or 'none'.")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manually promote, stabilize, or roll back a governed V2 deployment. "
            "No transition is automatic."
        )
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--deployment-root", type=Path, required=True)
    common.add_argument("--registry-lock-root", type=Path, required=True)
    common.add_argument("--registered-model-name", required=True)
    common.add_argument("--tracking-uri", required=True)
    common.add_argument("--expected-generation", type=int, required=True)
    common.add_argument("--expected-deployment-state-id", required=True)
    common.add_argument("--expected-pointer-sha256", required=True)
    common.add_argument("--expected-candidate", type=_alias, required=True)
    common.add_argument("--expected-champion", type=_alias, required=True)
    common.add_argument("--expected-stable", type=_alias, required=True)
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--approval-path", type=Path)
    common.add_argument("--approval-sha256")

    actions = parser.add_subparsers(dest="action", required=True)
    promote = actions.add_parser("promote", parents=[common])
    promote.add_argument("--candidate-bundle", type=Path, required=True)
    promote.add_argument("--candidate-calibration", type=Path, required=True)
    promote.add_argument("--incumbent-bundle", type=Path, required=True)
    promote.add_argument("--incumbent-calibration", type=Path, required=True)
    promote.add_argument("--registration-receipt", type=Path, required=True)
    promote.add_argument("--promotion-effective-date", required=True)

    stabilize = actions.add_parser("stabilize", parents=[common])
    stabilize.add_argument("--monitoring-store-root", type=Path, required=True)
    stabilize.add_argument("--monitoring-report", type=Path, required=True)
    stabilize.add_argument("--retraining-policy", type=Path, required=True)
    stabilize.add_argument("--monitoring-policy", type=Path, required=True)
    stabilize.add_argument("--observation-cutoff", required=True)

    rollback = actions.add_parser("rollback", parents=[common])
    rollback.add_argument("--promotion-receipt", type=Path, required=True)
    rollback.add_argument("--expected-rollback-state-id", required=True)
    rollback.add_argument("--observation-cutoff")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = LifecycleConfig(
        action=args.action,
        deployment_root=args.deployment_root,
        registry_lock_root=args.registry_lock_root,
        registered_model_name=args.registered_model_name,
        tracking_uri=args.tracking_uri,
        expected=ExpectedDeploymentState(
            generation=args.expected_generation,
            deployment_state_id=args.expected_deployment_state_id,
            pointer_sha256=args.expected_pointer_sha256,
            candidate=args.expected_candidate,
            champion=args.expected_champion,
            stable=args.expected_stable,
        ),
        approval_path=args.approval_path,
        approval_sha256=args.approval_sha256,
        dry_run=args.dry_run,
        candidate_bundle=getattr(args, "candidate_bundle", None),
        candidate_calibration=getattr(args, "candidate_calibration", None),
        incumbent_bundle=getattr(args, "incumbent_bundle", None),
        incumbent_calibration=getattr(args, "incumbent_calibration", None),
        registration_receipt=getattr(args, "registration_receipt", None),
        promotion_effective_date=getattr(args, "promotion_effective_date", None),
        monitoring_store_root=getattr(args, "monitoring_store_root", None),
        monitoring_report=getattr(args, "monitoring_report", None),
        policy_path=getattr(args, "retraining_policy", None),
        monitoring_policy_path=getattr(args, "monitoring_policy", None),
        observation_cutoff=getattr(args, "observation_cutoff", None),
        promotion_receipt=getattr(args, "promotion_receipt", None),
        expected_rollback_state_id=getattr(
            args, "expected_rollback_state_id", None
        ),
    )
    result = execute_lifecycle_transition(config)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
