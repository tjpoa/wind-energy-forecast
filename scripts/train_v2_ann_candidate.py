"""Train the explicitly governed scaled ANN v2 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.v2_ann import (
    ANNTrainingConfig,
    DEFAULT_DATASET_PATH,
    DEFAULT_SCALER_DIR,
    fit_v2_ann_candidate,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and seal one local scaled ANN v2 candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--scaler-dir", type=Path, default=DEFAULT_SCALER_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-start", default="2010-01-15")
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--validation-start", default="2023-01-01")
    parser.add_argument("--validation-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-06-27")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = fit_v2_ann_candidate(
        ANNTrainingConfig(
            input_path=args.input_path,
            scaler_dir=args.scaler_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            train_start=args.train_start,
            train_end=args.train_end,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
