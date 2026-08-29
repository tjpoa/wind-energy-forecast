"""Register an accepted ANN v2 challenger under the Registry candidate alias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.v2_ann_registry import ANNRegistrationConfig, register_ann_candidate


def _alias(value: str) -> str | None:
    if value == "none":
        return None
    if not value.strip():
        raise argparse.ArgumentTypeError("Alias expectation must be a version or 'none'.")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register one ANN v2 candidate only.")
    parser.add_argument("--challenger-bundle", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registered-model-name", default="wind-forecast-v2-hindcast")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--expected-candidate", type=_alias, required=True)
    parser.add_argument("--expected-champion", type=_alias, required=True)
    parser.add_argument("--expected-stable", type=_alias, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry-lock-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = register_ann_candidate(
        ANNRegistrationConfig(
            challenger_bundle=args.challenger_bundle,
            calibration_dir=args.calibration_dir,
            run_id=args.run_id,
            registered_model_name=args.registered_model_name,
            tracking_uri=args.tracking_uri,
            expected_candidate=args.expected_candidate,
            expected_champion=args.expected_champion,
            expected_stable=args.expected_stable,
            output_root=args.output_root,
            registry_lock_root=args.registry_lock_root,
        )
    )
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
