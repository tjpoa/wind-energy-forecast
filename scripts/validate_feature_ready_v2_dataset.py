"""Validate the feature-ready REN + ERA5-Land v2 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wind_forecast.validation.feature_ready import (
    serialize_validation_report,
    validate_feature_ready_v2_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments without reading or writing datasets."""
    parser = argparse.ArgumentParser(
        description="Validate the feature-ready v2 local-day daily dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
        help="Feature-ready v2 dataset root.",
    )
    parser.add_argument(
        "--integrated-root",
        type=Path,
        required=True,
        help="Accepted Step 2A.17 integrated dataset root.",
    )
    parser.add_argument(
        "--v1-feature-table",
        type=Path,
        required=True,
        help="Local v1 feature table used for exact column-order validation.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optional path for the deterministic JSON validation report.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    report = validate_feature_ready_v2_dataset(
        feature_root=args.feature_root,
        integrated_root=args.integrated_root,
        v1_feature_table=args.v1_feature_table,
    )
    report_json = serialize_validation_report(report)
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(report_json, encoding="utf-8", newline="\n")
    print(report_json, end="")
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
