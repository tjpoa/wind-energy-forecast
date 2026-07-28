"""Verify the active deployment against explicit monitoring artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.deployment_runtime import (
    DeploymentRuntimeError,
    verify_active_model_era,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify pointer, MLflow aliases and explicit v2 artifacts."
    )
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = verify_active_model_era(
            args.deployment_root,
            args.model_bundle,
            calibration_dir=args.calibration_dir,
        )
    except (DeploymentRuntimeError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "wind_forecast.deployment_preflight_error.v1",
                    "status": "failed",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"status": "verified", **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
