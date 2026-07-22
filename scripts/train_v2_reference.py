"""Train and track the first reproducible v2 reference-model candidate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wind_forecast.manifests import sha256_file
from wind_forecast.paths import project_root
from wind_forecast.schemas import TARGET_COLUMN
from wind_forecast.tracking import (
    DEFAULT_TRACKING_URI,
    ArtifactReference,
    MLflowNotInstalledError,
    MLflowTrackingError,
    TrackingConfig,
    flatten_metric_groups,
    git_state,
    log_dataset_input,
    log_run_data,
    log_sklearn_model,
    run_receipt,
    start_tracking_run,
    _load_mlflow,
)
from wind_forecast.training import load_training_table
from wind_forecast.v2_training import (
    DEFAULT_TEST_END,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VALIDATION_END,
    DEFAULT_VALIDATION_START,
    run_v2_reference_training,
)
from wind_forecast.validation.feature_ready import (
    serialize_validation_report,
    validate_feature_ready_v2_dataset,
)


DEFAULT_FEATURE_ROOT = Path(
    "data/processed/v2/ml_features/feature_ready_ren_era5_land_v2"
)
DEFAULT_INPUT = DEFAULT_FEATURE_ROOT / "feature_ready_daily.csv"
DEFAULT_INTEGRATED_ROOT = Path(
    "data/processed/v2/daily_merged/integrated_ren_era5_land_v2"
)
DEFAULT_V1_FEATURE_TABLE = Path("data/processed/agg_data_ml.csv")
DEFAULT_OUTPUT = project_root() / "outputs" / "training" / "v2_reference"
DEFAULT_EXPERIMENT = "wind-energy-forecast-v2-reference"
ACCEPTED_INPUT_SHA256 = "d0d073748c5d963cba30212e6b0ab666ec2000197b8f61a5c439b4aaf786b2a6"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first reproducible v2 hindcast reference candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--integrated-root", type=Path, default=DEFAULT_INTEGRATED_ROOT)
    parser.add_argument("--v1-feature-table", type=Path, default=DEFAULT_V1_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=DEFAULT_TEST_END)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--tracking-mode", choices=["local", "off"], default="local")
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--mlflow-experiment-name", default=DEFAULT_EXPERIMENT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.input.resolve().parent != args.feature_root.resolve():
        raise SystemExit("ERROR: --input must belong to --feature-root for a verifiable v2 run.")
    actual_sha256 = sha256_file(args.input)
    if actual_sha256 != ACCEPTED_INPUT_SHA256:
        raise SystemExit(
            "ERROR: v2 input SHA-256 does not match the accepted dataset: "
            f"{actual_sha256}"
        )
    validation = validate_feature_ready_v2_dataset(
        feature_root=args.feature_root,
        integrated_root=args.integrated_root,
        v1_feature_table=args.v1_feature_table,
    )
    if validation.has_errors:
        raise SystemExit(
            "ERROR: accepted v2 dataset validation failed:\n"
            + serialize_validation_report(validation)
        )
    upstream_validation = validation.to_dict()
    config = TrackingConfig(
        mode=args.tracking_mode,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        registered_model_name="unused-v2-no-registry",
        dataset_version="v2",
    )
    lineage = git_state()
    context = nullcontext(None)
    if config.mode == "local":
        context = start_tracking_run(
            "train-v2-reference",
            config=config,
            tags={
                "workflow": "train_v2_reference",
                "forecast_contract": "historical_daily_hindcast",
                "promotion": "disabled",
                **lineage,
            },
        )
    try:
        with context as active_run:
            result = run_v2_reference_training(
                input_path=args.input,
                output_dir=args.output_dir,
                seed=args.seed,
                n_estimators=args.n_estimators,
                train_start=args.train_start,
                train_end=args.train_end,
                validation_start=args.validation_start,
                validation_end=args.validation_end,
                test_start=args.test_start,
                test_end=args.test_end,
                upstream_validation=upstream_validation,
            )
            if active_run is not None:
                _log_run(result, active_run, config, args.input)
    except (MLflowNotInstalledError, MLflowTrackingError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"V2 reference training complete: {result.paths['summary']}")
    print(
        json.dumps(
            {
                "selected_model": result.selected_model,
                "accepted_as_reference": result.accepted_as_reference,
                "test_metrics": result.metrics["test"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _log_run(result, active_run, config: TrackingConfig, input_path: Path) -> None:
    table = load_training_table(input_path)
    sample = pd.read_csv(result.paths["reload_sample"])
    x_sample = sample[list(result.feature_names)]
    expected = sample["Expected_Prediction"].to_numpy(float)
    model = joblib.load(result.paths["model"])
    log_dataset_input(
        table,
        source=input_path.as_posix(),
        name="feature_ready_ren_era5_land_v2",
        target=TARGET_COLUMN,
        context="training",
        digest=result.input_sha256,
    )
    model_uri = log_sklearn_model(
        model,
        name="v2_reference_candidate",
        input_example=x_sample,
        predictions=expected,
    )
    reload_evidence_path = _verify_logged_model(
        model_uri=model_uri,
        input_example=x_sample,
        expected=expected,
        output_dir=result.output_dir,
    )
    flat_metrics = flatten_metric_groups(
        {
            f"{split}_{model_name}": values
            for split, models in result.metrics.items()
            for model_name, values in models.items()
        }
    )
    log_run_data(
        params={
            "workflow": "train_v2_reference",
            "selected_model": result.selected_model,
            "seed": model.random_state,
            "n_estimators": model.n_estimators,
            "dataset_version": "v2",
            "dataset_sha256": result.input_sha256,
            "split_assignment_sha256": result.split_sha256,
            "feature_count": len(result.feature_names),
            "scaler_required": False,
            "logged_model_uri": model_uri,
        },
        metrics=flat_metrics,
        tags={
            "forecast_contract": "historical_daily_hindcast",
            "reference_gate_passed": result.accepted_as_reference,
            "reference_status": (
                "selected_not_promoted"
                if result.accepted_as_reference
                else "rejected_not_promoted"
            ),
            "registry_used": False,
            "automatic_promotion": False,
        },
        artifact_paths=[
            ArtifactReference(path, "v2_reference")
            for name, path in result.paths.items()
            if name != "model" and path.is_file()
        ]
        + [ArtifactReference(reload_evidence_path, "validation")],
    )
    receipt = run_receipt(active_run, config, model_uri=model_uri)
    receipt_path = result.output_dir / "mlflow_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    log_run_data(artifact_paths=[ArtifactReference(receipt_path, "receipts")])


def _verify_logged_model(
    *,
    model_uri: str,
    input_example: pd.DataFrame,
    expected,
    output_dir: Path,
) -> Path:
    """Reload the MLflow model and persist prediction-equivalence evidence."""
    mlflow = _load_mlflow()
    loaded_model = mlflow.sklearn.load_model(model_uri)
    actual = loaded_model.predict(input_example)
    if not np.isfinite(np.asarray(actual, dtype=float)).all():
        raise MLflowTrackingError("Reloaded MLflow model returned non-finite predictions.")
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    equivalent = np.allclose(actual_array, expected_array, rtol=1e-12, atol=1e-9)
    if not equivalent:
        raise MLflowTrackingError(
            "Reloaded MLflow model predictions differ from the saved validation sample."
        )
    evidence_path = output_dir / "mlflow_reload_validation.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.mlflow_reload_validation.v1",
                "model_uri": model_uri,
                "row_count": len(expected_array),
                "predictions_equivalent": True,
                "rtol": 1e-12,
                "atol": 1e-9,
                "max_absolute_difference": float(
                    np.max(np.abs(actual_array - expected_array))
                ),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return evidence_path


if __name__ == "__main__":
    main()
