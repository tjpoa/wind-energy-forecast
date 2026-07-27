"""Register one sealed accepted retraining bundle as the v2 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wind_forecast.retraining_registry import (
    RetrainingRegistrationConfig,
    register_retraining_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and register one accepted v2 retraining candidate."
    )
    parser.add_argument("--backtest-bundle", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registered-model-name", required=True)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--expected-current-candidate-version")
    candidate.add_argument(
        "--expect-no-current-candidate",
        action="store_true",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = register_retraining_candidate(
        RetrainingRegistrationConfig(
            backtest_bundle=args.backtest_bundle,
            run_id=args.run_id,
            registered_model_name=args.registered_model_name,
            expected_current_candidate_version=(
                None
                if args.expect_no_current_candidate
                else args.expected_current_candidate_version
            ),
            output_root=args.output_root,
        )
    )
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
