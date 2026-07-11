import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from wind_forecast.artifacts import bundle_temp_extract, validate_release, verify_bundle
from wind_forecast.paths import project_root
from wind_forecast.training import run_baseline_training


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify or reproduce a release bundle.")
    parser.add_argument("--release", required=True)
    parser.add_argument("--bundle-root", type=Path, default=project_root() / "artifacts" / "releases")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_release(args.release)
    version = args.release.removeprefix("artifacts-")
    root = args.bundle_root / args.release
    archive = root / f"wind-energy-forecast-artifacts-{version}.zip"
    checksum = root / f"wind-energy-forecast-artifacts-{version}.sha256"
    manifest = verify_bundle(archive, checksum)
    if args.retrain:
        _retrain_and_compare(archive, args.release, overwrite=args.overwrite)
    print(f"Verified reproduction bundle: {manifest['release']}")


def _retrain_and_compare(archive: Path, release: str, *, overwrite: bool) -> None:
    with bundle_temp_extract(archive) as extracted:
        model_manifest = json.loads(
            (extracted / "manifests" / "model_manifest.json").read_text(encoding="utf-8")
        )
        output = project_root() / "outputs" / "reproduction" / release
        result = run_baseline_training(
            input_path=extracted / "data" / "agg_data_ml.csv",
            output_dir=output,
            model_type=model_manifest["model_type"],
            seed=int(model_manifest["seed"]),
            test_fraction=float(model_manifest["test_fraction"]),
            n_estimators=int(model_manifest["n_estimators"]),
            overwrite=overwrite,
            dataset_version=release,
        )
        expected_metrics = json.loads(
            (extracted / "baseline" / "metrics.json").read_text(encoding="utf-8")
        )
        for name, expected in expected_metrics.items():
            np.testing.assert_allclose(result.metrics[name], expected, rtol=1e-12, atol=1e-9)
        actual_predictions = pd.read_csv(result.predictions_path)["Predicted_Wind_Production"]
        expected_predictions = pd.read_csv(extracted / "baseline" / "predictions.csv")[
            "Predicted_Wind_Production"
        ]
        np.testing.assert_allclose(actual_predictions, expected_predictions, rtol=1e-12, atol=1e-9)


if __name__ == "__main__":
    main()
