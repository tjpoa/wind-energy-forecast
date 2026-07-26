"""Run the append-only historical hindcast monitoring ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.monitoring import MonitoringConfig, run_historical_monitoring


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments without reading sources or writing ledger state."""
    parser = argparse.ArgumentParser(
        description="Persist historical v2 hindcast evidence append-only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--through-date", required=True, help="Inclusive YYYY-MM-DD.")
    parser.add_argument(
        "--source-store-root",
        type=Path,
        default=Path("data/processed/v2/incremental_update"),
        help="Verified Phase 8 incremental store.",
    )
    parser.add_argument(
        "--monitoring-store-root",
        type=Path,
        default=Path("data/processed/v2/monitoring"),
    )
    parser.add_argument(
        "--model-bundle",
        type=Path,
        required=True,
        help="Accepted, unpromoted v2 reference output directory.",
    )
    parser.add_argument(
        "--activation-date",
        help="Required on the first run and immutable afterwards.",
    )
    parser.add_argument("--backfill-start", help="Explicit pre-activation YYYY-MM-DD.")
    parser.add_argument("--backfill-end", help="Explicit pre-activation YYYY-MM-DD.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and verify only; create no lock, directory, or record.",
    )
    args = parser.parse_args(argv)
    if bool(args.backfill_start) != bool(args.backfill_end):
        parser.error("--backfill-start and --backfill-end must be supplied together.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = MonitoringConfig(
        source_store_root=args.source_store_root,
        monitoring_store_root=args.monitoring_store_root,
        model_bundle=args.model_bundle,
        through_date=args.through_date,
        activation_date=args.activation_date,
        backfill_start=args.backfill_start,
        backfill_end=args.backfill_end,
        dry_run=args.dry_run,
    )
    result = run_historical_monitoring(config)
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
