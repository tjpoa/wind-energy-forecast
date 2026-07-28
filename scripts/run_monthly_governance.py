"""Run one recommendation-only controlled-retraining monthly check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.monthly_governance import (
    MonthlyGovernanceConfig,
    canonical_monthly_logical_time,
    run_monthly_governance,
)
from wind_forecast.retraining_policy import RetrainingPolicy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit monthly retraining/stability recommendations without "
            "training or lifecycle transitions."
        )
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=Path("config/retraining_policy_v1.json"),
    )
    parser.add_argument(
        "--monitoring-policy-path",
        type=Path,
        default=Path("config/monitoring_policy_v1.json"),
    )
    parser.add_argument(
        "--monitoring-store-root",
        type=Path,
        default=Path("data/processed/v2/monitoring"),
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=Path("data/processed/v2/deployment"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/processed/v2/retraining/monthly_recommendations"
        ),
    )
    parser.add_argument(
        "--evaluation-output-root",
        type=Path,
        default=Path("data/processed/v2/retraining/evaluations"),
    )
    logical = parser.add_mutually_exclusive_group(required=True)
    logical.add_argument(
        "--logical-at-utc",
        help="Exact policy-canonical UTC schedule timestamp.",
    )
    logical.add_argument(
        "--evaluation-period",
        help="YYYY-MM; converted to the policy-canonical UTC timestamp.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.logical_at_utc:
        logical = args.logical_at_utc
    else:
        logical = canonical_monthly_logical_time(
            RetrainingPolicy.load(args.policy_path),
            args.evaluation_period,
        )
    result = run_monthly_governance(
        MonthlyGovernanceConfig(
            policy_path=args.policy_path,
            monitoring_policy_path=args.monitoring_policy_path,
            monitoring_store_root=args.monitoring_store_root,
            deployment_root=args.deployment_root,
            logical_at_utc=logical,
            output_root=args.output_root,
            evaluation_output_root=args.evaluation_output_root,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
