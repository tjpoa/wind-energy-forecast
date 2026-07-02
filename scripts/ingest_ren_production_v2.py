"""Ingest REN production-breakdown responses into versioned v2 raw partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import pandas as pd

from wind_forecast.data_sources.ren import (
    DEFAULT_TIMEOUT_SECONDS,
    RenHTTPError,
    RenIngestionError,
    build_manifest,
    compare_normalized_with_v1,
    deterministic_json_sha256,
    fetch_ren_production_day,
    iter_inclusive_dates,
    load_all_partition_summaries,
    load_partition_summary,
    manifest_path,
    normalize_ren_payload,
    partition_is_unavailable_status_only,
    partition_is_verified,
    ren_partition_paths,
    utc_timestamp,
    validate_ren_normalized_day,
    write_daily_partition,
    write_manifest,
    write_unavailable_status,
)
from wind_forecast.paths import v2_raw_production_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest REN production v2 raw daily partitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date, formatted as YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Inclusive end date, formatted as YYYY-MM-DD.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=v2_raw_production_dir(),
        help="Root directory for v2 REN production outputs.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between ordered one-date requests.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip verified existing daily partitions.")
    parser.add_argument(
        "--retry-unavailable",
        action="store_true",
        help="With --resume, retry status-only unavailable daily partitions.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Explicitly overwrite existing daily partitions.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without network or writes.")
    parser.add_argument("--compare-v1-csv", type=Path, help="Optional frozen v1 production CSV for overlap metrics.")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive.")
    if args.retry_unavailable and not args.resume:
        parser.error("--retry-unavailable requires --resume.")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero.")
    if args.request_delay < 0:
        parser.error("--request-delay must be zero or greater.")
    return args


def run_ingestion(
    *,
    start_date: str,
    end_date: str,
    output_root: Path,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    request_delay: float = 0.0,
    resume: bool = False,
    retry_unavailable: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    compare_v1_csv: Path | None = None,
    request_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run a controlled REN ingestion over an inclusive date range."""
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive.")
    if retry_unavailable and not resume:
        raise ValueError("retry_unavailable requires resume.")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")
    if request_delay < 0:
        raise ValueError("request_delay must be zero or greater.")

    dates = iter_inclusive_dates(start_date, end_date)
    output_root = Path(output_root)
    requested_start = dates[0].isoformat()
    requested_end = dates[-1].isoformat()
    run_timestamp = utc_timestamp()

    if dry_run:
        return {
            "dry_run": True,
            "network_requests_planned": len(dates),
            "writes_planned": False,
            "requested_start_date": requested_start,
            "requested_end_date": requested_end,
            "output_root": str(output_root),
            "partitions": [
                {
                    "source_date": item.isoformat(),
                    "paths": {
                        "raw_response": str(ren_partition_paths(output_root, item).raw_response),
                        "normalized_csv": str(ren_partition_paths(output_root, item).normalized_csv),
                        "status_json": str(ren_partition_paths(output_root, item).status_json),
                    },
                }
                for item in dates
            ],
            "manifest_path": str(manifest_path(output_root)),
        }

    daily_results: list[dict[str, Any]] = []
    comparison_results: dict[str, Any] = {}
    requests_made = 0

    for index, source_day in enumerate(dates):
        source_date_text = source_day.isoformat()
        retrying_unavailable = False
        if resume:
            if partition_is_verified(output_root, source_day):
                summary = load_partition_summary(output_root, source_day)
                summary["skipped_existing"] = True
                daily_results.append(summary)
                continue
            retrying_unavailable = (
                retry_unavailable
                and partition_is_unavailable_status_only(output_root, source_day)
            )
        paths = ren_partition_paths(output_root, source_day)
        existing_paths = [paths.raw_response, paths.normalized_csv, paths.status_json]
        partition_overwrite = overwrite or retrying_unavailable
        if not partition_overwrite and any(path.exists() for path in existing_paths):
            raise FileExistsError(
                "REN partition already exists; use --resume to skip verified partitions "
                "or --overwrite to replace it explicitly."
            )

        try:
            capture = fetch_ren_production_day(
                source_day,
                timeout=timeout,
                request_get=request_get,
            )
            requests_made += 1
            raw_checksum = deterministic_json_sha256(capture.payload)
            normalized = normalize_ren_payload(
                source_day,
                capture.payload,
                retrieval_timestamp_utc=capture.retrieval_timestamp_utc,
                raw_response_sha256=raw_checksum,
            )
            validation = validate_ren_normalized_day(normalized, source_day)
            if compare_v1_csv is not None:
                comparison_results[source_date_text] = compare_normalized_with_v1(normalized, compare_v1_csv)
            daily_results.append(
                write_daily_partition(
                    output_root,
                    capture,
                    normalized,
                    validation,
                    overwrite=partition_overwrite,
                )
            )
        except RenHTTPError as exc:
            requests_made += 1
            daily_results.append(
                write_unavailable_status(
                    output_root,
                    source_day,
                    message=str(exc),
                    retrieval_timestamp_utc=utc_timestamp(),
                    status_code=exc.status_code,
                    overwrite=partition_overwrite,
                )
            )
        except RenIngestionError:
            raise

        if request_delay and index < len(dates) - 1:
            time.sleep(request_delay)

    manifest_daily_results = _merge_daily_results(
        load_all_partition_summaries(output_root),
        daily_results,
    )
    if compare_v1_csv is not None:
        comparison_results.update(
            _comparison_results_for_manifest(
                manifest_daily_results,
                compare_v1_csv=compare_v1_csv,
                existing_results=comparison_results,
            )
        )
    requested_ranges = _updated_requested_ranges(
        output_root,
        {
            "start_date": requested_start,
            "end_date": requested_end,
            "inclusive": True,
        },
    )
    manifest = build_manifest(
        output_root=output_root,
        requested_start_date=requested_start,
        requested_end_date=requested_end,
        retrieval_timestamp_utc=run_timestamp,
        daily_results=manifest_daily_results,
        requested_ranges=requested_ranges,
        compare_v1_csv=compare_v1_csv,
        comparison_results=comparison_results,
    )
    manifest_checksum = write_manifest(output_root, manifest)
    return {
        "dry_run": False,
        "requested_start_date": requested_start,
        "requested_end_date": requested_end,
        "output_root": str(output_root),
        "requests_made": requests_made,
        "daily_results": _json_ready(daily_results),
        "comparison_results": comparison_results,
        "manifest_path": str(manifest_path(output_root)),
        "manifest_sha256": manifest_checksum,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _merge_daily_results(
    existing_results: list[dict[str, Any]],
    current_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge existing and current partition summaries by source date."""
    by_date = {str(item["source_date"]): item for item in existing_results}
    for item in current_results:
        by_date[str(item["source_date"])] = item
    return [by_date[key] for key in sorted(by_date)]


def _updated_requested_ranges(
    output_root: Path,
    current_range: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return previous manifest ranges plus the current range, deduplicated."""
    ranges: list[dict[str, Any]] = []
    path = manifest_path(output_root)
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_ranges = (
                previous.get("extra_metadata", {}).get("requested_ranges", [])
                if isinstance(previous, dict)
                else []
            )
        except (OSError, json.JSONDecodeError):
            previous_ranges = []
        if isinstance(previous_ranges, list):
            ranges.extend(item for item in previous_ranges if isinstance(item, dict))

    ranges.append(current_range)
    unique: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for item in ranges:
        key = (item.get("start_date"), item.get("end_date"), item.get("inclusive"))
        unique[key] = {
            "start_date": item.get("start_date"),
            "end_date": item.get("end_date"),
            "inclusive": bool(item.get("inclusive", True)),
        }
    return [unique[key] for key in sorted(unique)]


def _comparison_results_for_manifest(
    daily_results: list[dict[str, Any]],
    *,
    compare_v1_csv: Path,
    existing_results: dict[str, Any],
) -> dict[str, Any]:
    """Return v1 comparison metrics for existing normalized partitions."""
    results = dict(existing_results)
    for item in daily_results:
        source_date = str(item["source_date"])
        if source_date in results:
            continue
        normalized_path = (item.get("paths") or {}).get("normalized_csv")
        if normalized_path is None:
            continue
        path = Path(normalized_path)
        if not path.is_file():
            continue
        normalized = pd.read_csv(path)
        results[source_date] = compare_normalized_with_v1(normalized, compare_v1_csv)
    return results


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = run_ingestion(
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
        timeout=args.timeout,
        request_delay=args.request_delay,
        resume=args.resume,
        retry_unavailable=args.retry_unavailable,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        compare_v1_csv=args.compare_v1_csv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
