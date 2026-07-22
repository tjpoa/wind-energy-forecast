"""Build the immutable Phase 9 reference and calibrated thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.monitoring_reporting import (
    CalibrationConfig,
    calibrate_monitoring_reference,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate historical batch drift/performance thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/processed/v2/ml_features/feature_ready_ren_era5_land_v2/feature_ready_daily.csv"
        ),
    )
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/monitoring_policy_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/v2/monitoring/reporting"),
    )
    parser.add_argument("--backtest-stride-days", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = calibrate_monitoring_reference(
        CalibrationConfig(
            dataset_path=args.dataset,
            model_bundle=args.model_bundle,
            policy_path=args.policy,
            output_root=args.output_root,
            backtest_stride_days=args.backtest_stride_days,
        )
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
