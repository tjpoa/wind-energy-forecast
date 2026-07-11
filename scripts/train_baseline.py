import argparse
import json
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import joblib
import pandas as pd

from wind_forecast.paths import processed_data_dir, project_root
from wind_forecast.schemas import TARGET_COLUMN
from wind_forecast.tracking import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_TRACKING_URI,
    ArtifactReference,
    MLflowNotInstalledError,
    MLflowTrackingError,
    TrackingConfig,
    git_state,
    log_dataset_input,
    log_run_data,
    log_sklearn_model,
    run_receipt,
    start_tracking_run,
)
from wind_forecast.training import load_training_table, run_baseline_training


DEFAULT_INPUT_PATH = processed_data_dir() / "agg_data_ml.csv"
DEFAULT_OUTPUT_DIR = project_root() / "outputs" / "training" / "baseline"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train a reproducible baseline wind-production model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model", choices=["extra_trees", "random_forest"], default="extra_trees"
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--tracking-mode",
        choices=["local", "off"],
        default="local",
        help="Log to the configured MLflow server or run without tracking.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Deprecated compatibility alias for --tracking-mode local.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri", default=DEFAULT_TRACKING_URI
    )
    parser.add_argument(
        "--mlflow-experiment-name", default=DEFAULT_EXPERIMENT_NAME
    )
    parser.add_argument(
        "--registered-model-name", default=DEFAULT_REGISTERED_MODEL_NAME
    )
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the baseline training workflow."""
    args = parse_args(argv)
    if args.mlflow:
        warnings.warn(
            "--mlflow is deprecated; use --tracking-mode local.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.tracking_mode = "local"

    config = TrackingConfig(
        mode=args.tracking_mode,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        registered_model_name=args.registered_model_name,
        dataset_version=args.dataset_version,
    )
    lineage = (
        git_state()
        if config.mode == "local"
        else {"git_sha": "not_recorded", "git_dirty": False}
    )
    run_context = nullcontext(None)
    if config.mode == "local":
        run_context = start_tracking_run(
            "train-baseline",
            config=config,
            tags={
                "workflow": "train_baseline",
                "model_type": args.model,
                "target_contract": "original",
                "git_sha": lineage["git_sha"],
                "git_dirty": lineage["git_dirty"],
            },
        )

    try:
        with run_context as active_run:
            result = run_baseline_training(
                input_path=args.input,
                output_dir=args.output_dir,
                model_type=args.model,
                seed=args.seed,
                test_fraction=args.test_fraction,
                n_estimators=args.n_estimators,
                overwrite=args.overwrite,
                dataset_version=args.dataset_version,
            )
            if active_run is not None:
                receipt_path = _log_training_run(result, active_run, config, lineage)
                print(f"MLflow receipt: {receipt_path}")
    except (MLflowNotInstalledError, MLflowTrackingError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(f"Baseline training complete: {result.summary_path}")
    print(result.metrics)


def _log_training_run(result, active_run, config: TrackingConfig, lineage: dict) -> Path:
    table = load_training_table(result.input_path)
    validation = pd.read_csv(result.validation_sample_path)
    feature_names = list(result.feature_names)
    input_example = validation[feature_names]
    expected = validation["Expected_Prediction"].to_numpy(dtype=float)
    model = joblib.load(result.model_path)

    log_dataset_input(
        table,
        source=result.input_path.as_posix(),
        name=f"wind-production-{result.dataset_version}",
        target=TARGET_COLUMN,
        context="training",
        digest=result.input_sha256,
    )
    model_uri = log_sklearn_model(
        model,
        name="model",
        input_example=input_example,
        predictions=expected,
    )
    log_run_data(
        params={
            "workflow": "train_baseline",
            "model_type": result.model_type,
            "seed": result.seed,
            "test_fraction": result.test_fraction,
            "n_estimators": result.n_estimators,
            "input_path": result.input_path,
            "dataset_version": result.dataset_version,
            "dataset_sha256": result.input_sha256,
            "feature_schema_sha256": result.feature_schema_sha256,
            "row_count": result.row_count,
            "feature_count": result.feature_count,
            "train_row_count": result.train_row_count,
            "test_row_count": result.test_row_count,
            "train_start_date": result.train_start_date,
            "train_end_date": result.train_end_date,
            "test_start_date": result.test_start_date,
            "test_end_date": result.test_end_date,
            "git_sha": lineage["git_sha"],
            "git_dirty": lineage["git_dirty"],
            "target_contract": "original",
            "logged_model_uri": model_uri,
        },
        metrics=result.metrics,
        tags={
            "validation_status": "pending_candidate_validation",
            "registered_model_name": config.registered_model_name,
        },
        artifact_paths=[
            ArtifactReference(result.model_path, "baseline"),
            ArtifactReference(result.metrics_path, "baseline"),
            ArtifactReference(result.predictions_path, "baseline"),
            ArtifactReference(result.summary_path, "baseline"),
            ArtifactReference(result.plot_path, "evaluation"),
            ArtifactReference(result.dataset_manifest_path, "manifests"),
            ArtifactReference(result.model_manifest_path, "manifests"),
            ArtifactReference(result.environment_path, "environment"),
            ArtifactReference(result.validation_sample_path, "validation"),
        ],
    )
    receipt = run_receipt(active_run, config, model_uri=model_uri)
    receipt_path = result.output_dir / "mlflow_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt_path


if __name__ == "__main__":
    main()
