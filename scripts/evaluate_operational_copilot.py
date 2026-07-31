"""Score offline operational-query candidate traces against a sealed dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from wind_forecast.operational_evaluation import (
    OperationalEvaluationInputError,
    evaluate_candidate_results,
    load_candidate_traces,
    load_evaluation_dataset,
    sanitized_report_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate offline operational-query candidate traces."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    return parser.parse_args(argv)


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
            '{"schema_version":"wind_forecast.operational_evaluation_error.v1",'
            '"status":"invalid_input"}'
        )
        return 2
    print(sanitized_report_json(report))
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
