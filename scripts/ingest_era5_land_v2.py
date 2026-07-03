"""Ingest ERA5-Land weather into versioned v2 raw weather partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wind_forecast.data_sources.era5_land import (
    EXPECTED_STATION_COUNT,
    run_ingestion,
)
from wind_forecast.paths import v2_raw_weather_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments without contacting CDS or writing outputs."""
    parser = argparse.ArgumentParser(
        description="Ingest ERA5-Land v2 raw weather station chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date, formatted as YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Inclusive end date, formatted as YYYY-MM-DD.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=v2_raw_weather_dir(),
        help="Root directory for v2 ERA5-Land weather outputs.",
    )
    parser.add_argument(
        "--station-mapping",
        type=Path,
        default=Path("data/pilot/ipma/ipma_station_mapping.csv"),
        help="Approved IPMA station mapping CSV.",
    )
    parser.add_argument(
        "--station-id",
        action="append",
        dest="station_ids",
        help="Optional approved station ID to ingest; repeat for multiple stations.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=EXPECTED_STATION_COUNT,
        help="Maximum station/month request chunks allowed for this run.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between ordered CDS requests.",
    )
    parser.add_argument(
        "--prior-pilot-dir",
        type=Path,
        default=Path("data/pilot/era5_land"),
        help="Existing ERA5-Land pilot directory for optional overlap comparison.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip verified complete existing chunks.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly overwrite existing chunks.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without network or writes.")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive.")
    if args.max_chunks <= 0:
        parser.error("--max-chunks must be greater than zero.")
    if args.request_delay < 0:
        parser.error("--request-delay must be zero or greater.")
    return args


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = run_ingestion(
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
        station_mapping=args.station_mapping,
        station_ids=args.station_ids,
        max_chunks=args.max_chunks,
        request_delay=args.request_delay,
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        prior_pilot_dir=args.prior_pilot_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
