"""Build the local-day integrated REN + ERA5-Land v2 daily dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wind_forecast.integration import build_integrated_v2_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments without reading datasets or writing outputs."""
    parser = argparse.ArgumentParser(
        description="Build the integrated REN + ERA5-Land v2 local-day daily dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True, help="Inclusive Europe/Lisbon local start date.")
    parser.add_argument("--end-date", required=True, help="Inclusive Europe/Lisbon local end date.")
    parser.add_argument(
        "--ren-root",
        type=Path,
        required=True,
        help="REN v2 production root containing the ren/ partition tree.",
    )
    parser.add_argument(
        "--era5-root",
        type=Path,
        required=True,
        help="ERA5-Land monthly-bbox root containing hourly/ station partitions.",
    )
    parser.add_argument(
        "--station-mapping",
        type=Path,
        required=True,
        help="Approved IPMA station mapping CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory for generated integrated outputs.",
    )
    parser.add_argument(
        "--v1-production",
        type=Path,
        default=Path("data/raw/ReparticaoProducao.csv"),
        help="Frozen v1 production CSV used only for comparison evidence.",
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
    result = build_integrated_v2_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        ren_root=args.ren_root,
        era5_root=args.era5_root,
        station_mapping=args.station_mapping,
        output_root=args.output_root,
        v1_production=args.v1_production,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
