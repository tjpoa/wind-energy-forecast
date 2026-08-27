"""Fit immutable v2 ANN scalers from the accepted feature-ready dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wind_forecast.manifests import sha256_file
from wind_forecast.paths import project_root
from wind_forecast.v2_scaling import (
    DEFAULT_FIT_END,
    DEFAULT_FIT_START,
    DEFAULT_TRANSFORMATION_VERSION,
    fit_v2_scalers,
)
from wind_forecast.validation.feature_ready import (
    serialize_validation_report,
    validate_feature_ready_v2_dataset,
)


DEFAULT_FEATURE_ROOT = Path(
    "data/processed/v2/ml_features/feature_ready_ren_era5_land_v2"
)
DEFAULT_INPUT = DEFAULT_FEATURE_ROOT / "feature_ready_daily.csv"
DEFAULT_INTEGRATED_ROOT = Path(
    "data/processed/v2/daily_merged/integrated_ren_era5_land_v2"
)
DEFAULT_V1_FEATURE_TABLE = Path("data/processed/agg_data_ml.csv")
DEFAULT_OUTPUT = project_root() / "models" / "v2" / "scalers" / "feature_ready_ren_era5_land_v2"
ACCEPTED_INPUT_SHA256 = "d0d073748c5d963cba30212e6b0ab666ec2000197b8f61a5c439b4aaf786b2a6"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit versioned v2 MinMax scalers without modifying v1 artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--integrated-root", type=Path, default=DEFAULT_INTEGRATED_ROOT)
    parser.add_argument("--v1-feature-table", type=Path, default=DEFAULT_V1_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-start", default=DEFAULT_FIT_START)
    parser.add_argument("--fit-end", default=DEFAULT_FIT_END)
    parser.add_argument(
        "--transformation-version",
        default=DEFAULT_TRANSFORMATION_VERSION,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.input.resolve().parent != args.feature_root.resolve():
        raise SystemExit("ERROR: --input must belong to --feature-root.")
    actual_sha256 = sha256_file(args.input)
    if actual_sha256 != ACCEPTED_INPUT_SHA256:
        raise SystemExit(
            "ERROR: v2 input SHA-256 does not match the accepted dataset: "
            f"{actual_sha256}"
        )

    validation = validate_feature_ready_v2_dataset(
        feature_root=args.feature_root,
        integrated_root=args.integrated_root,
        v1_feature_table=args.v1_feature_table,
    )
    if validation.has_errors:
        raise SystemExit(
            "ERROR: accepted v2 dataset validation failed:\n"
            + serialize_validation_report(validation)
        )

    result = fit_v2_scalers(
        input_path=args.input,
        output_dir=args.output_dir,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        transformation_version=args.transformation_version,
    )
    print(
        json.dumps(
            {
                "output_dir": result.output_dir.as_posix(),
                "input_sha256": result.input_sha256,
                "fit_scope": result.fit_scope,
                "fit_start": result.fit_start,
                "fit_end": result.fit_end,
                "fit_row_count": result.fit_row_count,
                "total_row_count": result.total_row_count,
                "feature_count": len(result.feature_names),
                "manifest": result.paths["manifest"].as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
