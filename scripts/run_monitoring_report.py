"""Generate one immutable Phase 9 drift/performance report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.monitoring_reporting import (
    MonitoringReportConfig,
    run_monitoring_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an immutable historical batch monitoring report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument(
        "--monitoring-store-root",
        type=Path,
        default=Path("data/processed/v2/monitoring"),
    )
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--through-date", required=True, help="Inclusive YYYY-MM-DD.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-on-active-alert",
        action="store_true",
        help="Exit 2 after a successful report when one or more alerts are active.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_monitoring_report(
        MonitoringReportConfig(
            source_run_manifest=args.source_run_manifest,
            monitoring_store_root=args.monitoring_store_root,
            model_bundle=args.model_bundle,
            calibration_dir=args.calibration_dir,
            deployment_root=args.deployment_root,
            through_date=args.through_date,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    if args.fail_on_active_alert and result.active_alert_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
