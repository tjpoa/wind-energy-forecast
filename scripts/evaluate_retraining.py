"""Evaluate one operator-pinned monthly controlled-retraining decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.retraining_evaluation import (
    MonthlyRetrainingEvaluationConfig,
    run_monthly_retraining_evaluation,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Phase 9 evidence and emit an immutable recommendation only."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=Path("config/retraining_policy_v1.json"),
    )
    parser.add_argument(
        "--monitoring-store-root",
        type=Path,
        default=Path("data/processed/v2/monitoring"),
    )
    parser.add_argument(
        "--monitoring-report-path",
        type=Path,
        required=True,
        help="Exact immutable Phase 9 report.json selected by the operator.",
    )
    parser.add_argument(
        "--incumbent-id",
        required=True,
        help="Explicit transitional incumbent model_snapshot_id; not a champion alias.",
    )
    parser.add_argument("--incumbent-fit-cutoff", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/v2/retraining/evaluations"),
    )
    parser.add_argument(
        "--evaluated-at-utc",
        required=True,
        help="Explicit reproducible UTC timestamp, for example 2026-04-08T12:00:00Z.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify and plan without writing any evaluation artifact.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_monthly_retraining_evaluation(
        MonthlyRetrainingEvaluationConfig(
            policy_path=args.policy_path,
            monitoring_store_root=args.monitoring_store_root,
            monitoring_report_path=args.monitoring_report_path,
            incumbent_id=args.incumbent_id,
            incumbent_fit_cutoff=args.incumbent_fit_cutoff,
            output_root=args.output_root,
            evaluated_at_utc=args.evaluated_at_utc,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
