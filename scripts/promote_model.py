import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wind_forecast.registry import (
    PromotionReceipt,
    promote_candidate,
    rollback_promotion,
)
from wind_forecast.tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_TRACKING_URI,
    TrackingConfig,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote or roll back MLflow aliases.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--registered-model", default=DEFAULT_REGISTERED_MODEL_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--expected-candidate-version", required=True)
    promote.add_argument("--expected-champion-version", required=True)
    promote.add_argument("--approval-note", required=True)
    promote.add_argument("--receipt", type=Path)
    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = TrackingConfig(
        tracking_uri=args.tracking_uri,
        registered_model_name=args.registered_model,
    )
    if args.command == "rollback":
        receipt = PromotionReceipt.from_path(args.receipt)
        rollback_promotion(receipt, config=config)
        print(f"Rolled back promotion receipt: {args.receipt}")
        return
    receipt_path = args.receipt or Path("outputs/registry") / (
        f"promotion-v{args.expected_candidate_version}.json"
    )
    receipt = promote_candidate(
        config=config,
        expected_candidate_version=args.expected_candidate_version,
        expected_champion_version=args.expected_champion_version,
        approval_note=args.approval_note,
        receipt_path=receipt_path,
    )
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
