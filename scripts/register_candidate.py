import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wind_forecast.registry import register_candidate, write_receipt
from wind_forecast.tracking import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_TRACKING_URI,
    TrackingConfig,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and register an MLflow candidate.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--registered-model", default=DEFAULT_REGISTERED_MODEL_NAME)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = TrackingConfig(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model,
        dataset_version=args.dataset_version,
    )
    receipt = register_candidate(args.run_id, config=config)
    receipt_path = args.receipt or Path("outputs/registry") / f"candidate-v{receipt.model_version}.json"
    write_receipt(receipt, receipt_path)
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
