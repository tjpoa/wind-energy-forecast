"""Evaluate the fixed Gemini candidate without persisting provider payloads."""

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

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
from wind_forecast.operational_gemini_candidate import (
    GeminiInteractionsCandidateSelector,
    GeminiTransport,
    build_gemini_candidate_evaluation_receipt,
    write_gemini_candidate_receipt,
)
from wind_forecast.operational_gemini_candidate_models import (
    ENGLISH_LANGUAGE,
    GeminiCandidateMetadata,
)

SEALED_DATASET_SHA256 = (
    "74c6438edf636d3061f0f142bb315d669e716fc2b20e5adafaab6154aa08afb6"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed Gemini operational Copilot candidate."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--evaluated-at-utc")
    parser.add_argument("--confirm-synthetic-egress", action="store_true")
    return parser.parse_args(argv)


def _error(code: str) -> None:
    print(json.dumps({"code": code, "status": "error"}, sort_keys=True))


def _time(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timezone required")
    return result.astimezone(timezone.utc)


def _checkout_state() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        raise CandidateEvaluationInputError("checkout provenance is unavailable") from None
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CandidateEvaluationInputError("checkout provenance is invalid")
    return commit, not bool(status)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: GeminiTransport | None = None,
) -> int:
    args = parse_args(argv)
    if not args.confirm_synthetic_egress:
        _error("synthetic_egress_not_confirmed")
        return 2
    api_key = (os.environ if environ is None else environ).get("GEMINI_API_KEY", "")
    if not api_key.strip():
        _error("gemini_api_key_unavailable")
        return 2
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        _error("source_commit_invalid")
        return 2
    if args.receipt_out.exists():
        _error("receipt_path_exists")
        return 2
    try:
        checkout_commit, checkout_clean = _checkout_state()
        if args.source_commit != checkout_commit or not checkout_clean:
            raise CandidateEvaluationInputError("checkout provenance does not match")
        evaluated_at = _time(args.evaluated_at_utc)
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
        metadata = GeminiCandidateMetadata(candidate_id=args.candidate_id)
        selector = GeminiInteractionsCandidateSelector(
            api_key=api_key, metadata=metadata, transport=transport
        )
        run = run_candidate_evaluation(
            dataset, selector, metadata, evaluated_at_utc=evaluated_at
        )
    except CandidateEvaluationInfrastructureError:
        _error("candidate_infrastructure_failure")
        return 2
    except (
        CandidateEvaluationInputError,
        OperationalEvaluationInputError,
        OSError,
        ValueError,
    ):
        _error("candidate_evaluation_input_invalid")
        return 2
    print(sanitized_report_json(run.report))
    if run.report.status != "passed":
        return 1
    try:
        receipt = build_gemini_candidate_evaluation_receipt(
            run,
            metadata,
            calls_completed=selector.calls_completed,
            source_commit=args.source_commit,
            evaluated_at_utc=evaluated_at,
        )
        write_gemini_candidate_receipt(args.receipt_out, receipt)
    except (CandidateEvaluationInputError, OSError, ValueError):
        _error("candidate_receipt_invalid")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
