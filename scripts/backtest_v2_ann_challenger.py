"""Run the sealed-test ANN v2 challenger backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.v2_ann import DEFAULT_DATASET_PATH
from wind_forecast.v2_ann_challenger import (
    DEFAULT_OUTPUT_ROOT,
    ChallengerBacktestConfig,
    run_v2_ann_challenger_backtest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest one frozen ANN v2 candidate on the sealed test period."
    )
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--incumbent-bundle", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--incumbent-calibration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evaluation-period", default="sealed_test_2025_2026")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-06-27")
    parser.add_argument("--fold-size", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_v2_ann_challenger_backtest(
        ChallengerBacktestConfig(
            candidate_bundle=args.candidate_bundle,
            incumbent_bundle=args.incumbent_bundle,
            dataset_path=args.dataset_path,
            incumbent_calibration=args.incumbent_calibration,
            output_root=args.output_root,
            evaluation_period=args.evaluation_period,
            test_start=args.test_start,
            test_end=args.test_end,
            fold_size=args.fold_size,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
