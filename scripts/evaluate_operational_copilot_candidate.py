"""Evaluate a supplied offline candidate response set and issue a receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from wind_forecast.operational_candidate_evaluation import (
    CandidateEvaluationRun,
    build_candidate_evaluation_receipt,
    write_candidate_receipt,
)
from wind_forecast.operational_candidate_evaluation_models import CandidateMetadata
from wind_forecast.operational_evaluation import (
    OperationalEvaluationInputError,
    evaluate_candidate_results,
    load_candidate_traces,
    load_evaluation_dataset,
    sanitized_report_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an offline candidate response set and write its receipt."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--evaluated-at-utc")
    return parser.parse_args(argv)


def _evaluated_at(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--evaluated-at-utc must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = load_evaluation_dataset(args.dataset)
        dataset_bytes = args.dataset.stat().st_size + (
            args.dataset.parent / "cases.jsonl"
        ).stat().st_size
        traces, response_sha256 = load_candidate_traces(
            args.responses,
            dataset_bytes=dataset_bytes,
        )
        report = evaluate_candidate_results(
            dataset,
            traces,
            response_set_sha256=response_sha256,
        )
    except (OperationalEvaluationInputError, OSError, ValueError):
        print(
            '{"schema_version":"wind_forecast.operational_candidate_evaluation_error.v1",'
            '"status":"invalid_input"}'
        )
        return 2

    print(sanitized_report_json(report))
    if report.status != "passed":
        return 1

    try:
        run = CandidateEvaluationRun(
            traces=traces,
            report=report,
            response_set_sha256=response_sha256,
        )
        metadata = CandidateMetadata(
            candidate_id=args.candidate_id,
            provider=args.provider,
            model=args.model,
        )
        receipt = build_candidate_evaluation_receipt(
            run,
            metadata,
            source_commit=args.source_commit,
            evaluated_at_utc=_evaluated_at(args.evaluated_at_utc),
        )
        write_candidate_receipt(args.receipt_out, receipt)
    except (OperationalEvaluationInputError, OSError, ValueError):
        print(
            '{"schema_version":"wind_forecast.operational_candidate_evaluation_error.v1",'
            '"status":"invalid_input"}'
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
