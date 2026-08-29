"""Log an accepted ANN v2 challenger in local MLflow without Registry mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.v2_ann_registry import ANNRunConfig, log_ann_candidate_run


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log one ANN v2 candidate run in MLflow.")
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--backtest-bundle", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment-name", default="wind-energy-forecast-v2-ann-challenger")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = log_ann_candidate_run(
        ANNRunConfig(
            candidate_bundle=args.candidate_bundle,
            backtest_bundle=args.backtest_bundle,
            calibration_dir=args.calibration_dir,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
        )
    )
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
