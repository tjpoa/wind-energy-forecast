"""Build the feature-ready REN + ERA5-Land v2 daily dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wind_forecast.v2_features import build_feature_ready_v2_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments without reading datasets or writing outputs."""
    parser = argparse.ArgumentParser(
        description="Build the feature-ready v2 local-day daily dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Accepted Step 2A.17 integrated dataset root.",
    )
    parser.add_argument(
        "--v1-feature-table",
        type=Path,
        required=True,
        help="Local v1 feature table used for exact column order.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory for generated feature-ready outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in an existing output directory. Without this flag, existing output roots are refused.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = build_feature_ready_v2_dataset(
        input_root=args.input_root,
        v1_feature_table=args.v1_feature_table,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
