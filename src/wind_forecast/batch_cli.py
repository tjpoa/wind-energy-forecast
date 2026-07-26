"""Stable command-line interface for local historical batch orchestration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from wind_forecast.orchestration import (
    BatchConfig,
    BatchOrchestrationError,
    load_verified_batch_run,
    plan_batch,
    run_batch,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan, run, or inspect the accepted historical batch workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--through-date", help="Inclusive YYYY-MM-DD; defaults to the current Lisbon date.")
        child.add_argument("--model-bundle", type=Path)
        child.add_argument("--calibration-dir", type=Path)
        child.add_argument("--activation-date")
        child.add_argument("--backfill-start")
        child.add_argument("--backfill-end")
        child.add_argument(
            "--source-store-root",
            type=Path,
            default=Path("data/processed/v2/incremental_update"),
        )
        child.add_argument(
            "--monitoring-store-root",
            type=Path,
            default=Path("data/processed/v2/monitoring"),
        )
        child.add_argument(
            "--orchestration-root",
            type=Path,
            default=Path("data/processed/v2/orchestration"),
        )
        child.add_argument("--no-source-refresh", action="store_true")
        child.add_argument("--fail-on-active-alert", action="store_true")
        child.add_argument(
            "--env-file",
            type=Path,
            help="Optional ignored dotenv file loaded explicitly before configuration.",
        )
    status = subparsers.add_parser("status")
    status.add_argument(
        "--orchestration-root",
        type=Path,
        default=Path("data/processed/v2/orchestration"),
    )
    status.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command in {"plan", "run"}:
        if bool(args.backfill_start) != bool(args.backfill_end):
            parser.error("--backfill-start and --backfill-end must be supplied together.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "status":
            payload = load_verified_batch_run(
                args.orchestration_root,
                args.manifest,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.env_file:
            from dotenv import load_dotenv

            load_dotenv(args.env_file, override=False)
        model_bundle = args.model_bundle or os.getenv(
            "WIND_FORECAST_BATCH_MODEL_BUNDLE"
        )
        calibration_dir = args.calibration_dir or os.getenv(
            "WIND_FORECAST_BATCH_CALIBRATION_DIR"
        )
        if not model_bundle or not calibration_dir:
            raise BatchOrchestrationError(
                "Set --model-bundle and --calibration-dir, or their batch environment variables."
            )
        config = BatchConfig(
            through_date=args.through_date,
            model_bundle=Path(model_bundle),
            calibration_dir=Path(calibration_dir),
            activation_date=args.activation_date,
            backfill_start=args.backfill_start,
            backfill_end=args.backfill_end,
            source_store_root=args.source_store_root,
            monitoring_store_root=args.monitoring_store_root,
            orchestration_root=args.orchestration_root,
            no_source_refresh=args.no_source_refresh,
            fail_on_active_alert=args.fail_on_active_alert,
        )
        result = plan_batch(config) if args.command == "plan" else run_batch(config)
        print(json.dumps(result.summary(), indent=2, sort_keys=True))
        if (
            args.command == "run"
            and args.fail_on_active_alert
            and result.active_alert_count
        ):
            return 2
        return 0
    except (BatchOrchestrationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "wind_forecast.batch_cli_error.v1",
                    "status": "failed",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
