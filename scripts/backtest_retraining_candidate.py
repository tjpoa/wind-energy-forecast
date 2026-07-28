"""Run one operator-pinned controlled-retraining temporal backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wind_forecast.retraining_backtesting import (
    RetrainingBacktestConfig,
    run_retraining_backtest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify, backtest, and optionally seal one retraining candidate."
    )
    parser.add_argument("--evaluation-path", type=Path, required=True)
    parser.add_argument("--monitoring-store-root", type=Path, required=True)
    parser.add_argument("--incumbent-bundle", type=Path, required=True)
    parser.add_argument("--incumbent-base-dataset", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_retraining_backtest(
        RetrainingBacktestConfig(
            evaluation_path=args.evaluation_path,
            monitoring_store_root=args.monitoring_store_root,
            incumbent_bundle=args.incumbent_bundle,
            incumbent_base_dataset=args.incumbent_base_dataset,
            calibration_dir=args.calibration_dir,
            policy_path=args.policy_path,
            output_root=args.output_root,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
