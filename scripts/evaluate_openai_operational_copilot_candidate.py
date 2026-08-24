"""Evaluate the fixed OpenAI candidate without persisting provider payloads."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from wind_forecast.operational_candidate_evaluation import (
    CandidateEvaluationInfrastructureError,
    CandidateEvaluationInputError,
    run_candidate_evaluation,
)
from wind_forecast.operational_evaluation import (
    OperationalEvaluationInputError,
    load_evaluation_dataset,
    sanitized_report_json,
)
from wind_forecast.operational_evaluation_models import (
    DATASET_ID,
    DATASET_VERSION,
    EXPECTED_CASE_COUNT,
)
from wind_forecast.operational_openai_candidate import (
    OpenAIResponsesCandidateSelector,
    OpenAITransport,
    build_openai_candidate_evaluation_receipt,
    write_openai_candidate_receipt,
)
from wind_forecast.operational_openai_candidate_models import (
    ENGLISH_LANGUAGE,
    OpenAICandidateMetadata,
)


SEALED_DATASET_SHA256 = (
    "74c6438edf636d3061f0f142bb315d669e716fc2b20e5adafaab6154aa08afb6"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed OpenAI operational Copilot candidate."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--evaluated-at-utc")
    parser.add_argument(
        "--confirm-synthetic-egress",
        action="store_true",
        help="Confirm that the sealed synthetic candidate inputs may be sent to OpenAI.",
    )
    return parser.parse_args(argv)


def _evaluated_at(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--evaluated-at-utc must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _sanitized_error(code: str) -> None:
    print(json.dumps({"code": code, "status": "error"}, sort_keys=True))


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: OpenAITransport | None = None,
) -> int:
    args = parse_args(argv)
    if not args.confirm_synthetic_egress:
        _sanitized_error("synthetic_egress_not_confirmed")
        return 2
    environment = os.environ if environ is None else environ
    api_key = environment.get("OPENAI_API_KEY", "")
    if not api_key.strip():
        _sanitized_error("openai_api_key_unavailable")
        return 2
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        _sanitized_error("source_commit_invalid")
        return 2
    if args.receipt_out.exists():
        _sanitized_error("receipt_path_exists")
        return 2
    try:
        evaluated_at = _evaluated_at(args.evaluated_at_utc)
        dataset = load_evaluation_dataset(args.dataset)
        if (
            dataset.manifest.dataset_id != DATASET_ID
            or dataset.manifest.dataset_version != DATASET_VERSION
            or dataset.manifest.language != ENGLISH_LANGUAGE
            or dataset.manifest.case_count != EXPECTED_CASE_COUNT
            or len(dataset.cases) != EXPECTED_CASE_COUNT
            or dataset.dataset_sha256 != SEALED_DATASET_SHA256
        ):
            raise CandidateEvaluationInputError(
                "sealed evaluation dataset identity is invalid"
            )
        metadata = OpenAICandidateMetadata(candidate_id=args.candidate_id)
        selector = OpenAIResponsesCandidateSelector(
            api_key=api_key,
            metadata=metadata,
            transport=transport,
        )
        run = run_candidate_evaluation(
            dataset,
            selector,
            metadata,
            evaluated_at_utc=evaluated_at,
        )
    except CandidateEvaluationInfrastructureError:
        _sanitized_error("candidate_infrastructure_failure")
        return 2
    except (
        CandidateEvaluationInputError,
        OperationalEvaluationInputError,
        OSError,
        ValueError,
    ):
        _sanitized_error("candidate_evaluation_input_invalid")
        return 2

    print(sanitized_report_json(run.report))
    if run.report.status != "passed":
        return 1
    try:
        receipt = build_openai_candidate_evaluation_receipt(
            run,
            metadata,
            calls_completed=selector.calls_completed,
            source_commit=args.source_commit,
            evaluated_at_utc=evaluated_at,
        )
        write_openai_candidate_receipt(args.receipt_out, receipt)
    except (CandidateEvaluationInputError, OSError, ValueError):
        _sanitized_error("candidate_receipt_invalid")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
