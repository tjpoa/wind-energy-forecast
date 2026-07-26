"""Safely refresh and incrementally publish the accepted v2 dataset."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Sequence

try:
    from scripts.ingest_ren_production_v2 import run_ingestion as run_ren_ingestion
except ModuleNotFoundError:  # Direct ``python scripts/update_v2_dataset.py`` execution.
    from ingest_ren_production_v2 import run_ingestion as run_ren_ingestion
from wind_forecast.data_sources.era5_land import (
    GRID_POLICY_NEAREST_VALID,
    REQUEST_MODE_MONTHLY_BBOX,
    era5_land_root,
    run_ingestion as run_era5_ingestion,
)
from wind_forecast.incremental import (
    RefreshResult,
    UpdateConfig,
    UpdatePlan,
    run_v2_update,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments without reading data or creating outputs."""
    parser = argparse.ArgumentParser(
        description="Run one transactional incremental update of the v2 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--through-date",
        required=True,
        help="Inclusive Europe/Lisbon date, formatted as YYYY-MM-DD.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only: no network, locks, logs, or writes.")
    parser.add_argument("--revision-lookback-days", type=int, default=90)
    parser.add_argument("--recheck-min-age-hours", type=int, default=24)
    parser.add_argument(
        "--recheck-ren-date",
        action="append",
        default=[],
        help="Explicit old REN date to recheck; repeat as needed.",
    )
    parser.add_argument(
        "--recheck-era5-month",
        action="append",
        default=[],
        help="Explicit old ERA5-Land month (YYYY-MM) to recheck; repeat as needed.",
    )
    parser.add_argument("--ren-root", type=Path, default=Path("data/raw/v2/production"))
    parser.add_argument(
        "--era5-root",
        type=Path,
        default=Path(
            "data/raw/v2/weather/era5_land/"
            "grid_policy=nearest_valid_r1/request_mode=monthly_bbox"
        ),
    )
    parser.add_argument(
        "--station-mapping",
        type=Path,
        default=Path("data/pilot/ipma/ipma_station_mapping.csv"),
    )
    parser.add_argument(
        "--v1-feature-table",
        type=Path,
        default=Path("data/processed/agg_data_ml.csv"),
    )
    parser.add_argument(
        "--baseline-integrated-root",
        type=Path,
        default=Path(
            "data/processed/v2/daily_merged/integrated_ren_era5_land_v2"
        ),
    )
    parser.add_argument(
        "--baseline-feature-root",
        type=Path,
        default=Path(
            "data/processed/v2/ml_features/feature_ready_ren_era5_land_v2"
        ),
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path("data/processed/v2/incremental_update"),
    )
    parser.add_argument(
        "--raw-store-root",
        type=Path,
        default=Path("data/raw/v2/incremental_update"),
    )
    parser.add_argument(
        "--monitoring-policy",
        type=Path,
        default=Path("config/monitoring_policy_v1.json"),
        help="Versioned Phase 9 quality/freshness policy used by batch sidecars.",
    )
    parser.add_argument(
        "--no-source-refresh",
        action="store_true",
        help="Use only already-present local raw partitions; mainly for controlled recovery.",
    )
    args = parser.parse_args(argv)
    if args.revision_lookback_days < 0:
        parser.error("--revision-lookback-days must be zero or greater.")
    if args.recheck_min_age_hours < 0:
        parser.error("--recheck-min-age-hours must be zero or greater.")
    return args


def _consecutive_ranges(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return []
    days = sorted(datetime.strptime(item, "%Y-%m-%d").date() for item in set(values))
    result: list[tuple[str, str]] = []
    start = previous = days[0]
    for item in days[1:]:
        if item != previous + timedelta(days=1):
            result.append((start.isoformat(), previous.isoformat()))
            start = item
        previous = item
    result.append((start.isoformat(), previous.isoformat()))
    return result


def _month_bounds(month: str, eligible_through: str) -> tuple[str, str]:
    first = datetime.strptime(month, "%Y-%m").date().replace(day=1)
    next_month = date(
        first.year + (first.month == 12),
        1 if first.month == 12 else first.month + 1,
        1,
    )
    last = min(next_month - timedelta(days=1), datetime.strptime(eligible_through, "%Y-%m-%d").date())
    return first.isoformat(), last.isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    config = UpdateConfig(
        through_date=args.through_date,
        ren_root=args.ren_root,
        era5_root=args.era5_root,
        station_mapping=args.station_mapping,
        v1_feature_table=args.v1_feature_table,
        baseline_integrated_root=args.baseline_integrated_root,
        baseline_feature_root=args.baseline_feature_root,
        store_root=args.store_root,
        raw_store_root=args.raw_store_root,
        monitoring_policy_path=args.monitoring_policy,
        revision_lookback_days=args.revision_lookback_days,
        recheck_min_age_hours=args.recheck_min_age_hours,
        recheck_ren_dates=tuple(args.recheck_ren_date),
        recheck_era5_months=tuple(args.recheck_era5_month),
        dry_run=args.dry_run,
    )

    if args.dry_run or args.no_source_refresh:
        refresher = None
    else:
        # Capture the configured station mapping without changing the public
        # refresher protocol used by tests and other callers.
        def refresher(plan: UpdatePlan, staging_root: Path) -> RefreshResult:
            ren_output = staging_root / "production"
            ren_dates = list(set(plan.ren_missing_dates) | set(plan.ren_recheck_dates))
            for start, end in _consecutive_ranges(ren_dates):
                run_ren_ingestion(
                    start_date=start,
                    end_date=end,
                    output_root=ren_output,
                    resume=True,
                    retry_unavailable=True,
                )
            weather_output = staging_root / "weather"
            era_months = sorted(
                set(plan.era5_missing_months) | set(plan.era5_recheck_months)
            )
            for month in era_months:
                start, end = _month_bounds(month, plan.eligible_through["era5_land"])
                run_era5_ingestion(
                    start_date=start,
                    end_date=end,
                    output_root=weather_output,
                    station_mapping=config.station_mapping,
                    max_chunks=1,
                    resume=True,
                    grid_policy=GRID_POLICY_NEAREST_VALID,
                    grid_search_radius=1,
                    request_mode=REQUEST_MODE_MONTHLY_BBOX,
                )
            return RefreshResult(
                ren_roots=(ren_output,) if ren_dates else (),
                era5_roots=(
                    era5_land_root(
                        weather_output,
                        grid_policy=GRID_POLICY_NEAREST_VALID,
                        grid_search_radius=1,
                        request_mode=REQUEST_MODE_MONTHLY_BBOX,
                    ),
                )
                if era_months
                else (),
            )

    result = run_v2_update(config, source_refresher=refresher)
    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
