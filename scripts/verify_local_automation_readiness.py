"""Verify a local automation readiness receipt for one workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wind_forecast.readiness import ReadinessError, load_automation_readiness
from wind_forecast.paths import project_root


DEFAULT_PATH = project_root() / "config" / "local_automation_readiness_v1.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--environment-id", default="local")
    parser.add_argument("--workflow", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        receipt = load_automation_readiness(
            args.path,
            environment_id=args.environment_id,
            workflow=args.workflow,
        )
    except ReadinessError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(receipt.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
