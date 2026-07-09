import argparse
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

from wind_forecast.paths import processed_data_dir, project_root
from wind_forecast.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_DIRNAME,
    ArtifactReference,
    MLflowNotInstalledError,
    log_run_data,
    start_local_run,
)
from wind_forecast.training import run_baseline_training


DEFAULT_INPUT_PATH = processed_data_dir() / "agg_data_ml.csv"
DEFAULT_OUTPUT_DIR = project_root() / "outputs" / "training" / "baseline"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train a reproducible baseline wind-production model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Feature-ready training CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model, metrics, predictions, and run summary.",
    )
    parser.add_argument(
        "--model",
        choices=["extra_trees", "random_forest"],
        default="extra_trees",
        help="Baseline estimator to train.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Chronological final holdout fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic estimators.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees for the baseline estimator.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite known output files in the output directory.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log this training run to a local MLflow tracking store.",
    )
    parser.add_argument(
        "--mlflow-tracking-dir",
        type=Path,
        default=project_root() / DEFAULT_TRACKING_DIRNAME,
        help="Local MLflow tracking directory.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the baseline training workflow."""
    args = parse_args(argv)
    run_context = nullcontext()
    if args.mlflow:
        run_context = start_local_run(
            "train-baseline",
            tracking_dir=args.mlflow_tracking_dir,
            experiment_name=args.mlflow_experiment_name,
            tags={
                "workflow": "train_baseline",
                "model_type": args.model,
            },
        )

    try:
        with run_context:
            result = run_baseline_training(
                input_path=args.input,
                output_dir=args.output_dir,
                model_type=args.model,
                seed=args.seed,
                test_fraction=args.test_fraction,
                n_estimators=args.n_estimators,
                overwrite=args.overwrite,
            )
            if args.mlflow:
                log_run_data(
                    params={
                        "workflow": "train_baseline",
                        "model_type": result.model_type,
                        "seed": result.seed,
                        "test_fraction": result.test_fraction,
                        "n_estimators": result.n_estimators,
                        "input_path": result.input_path,
                        "output_dir": result.output_dir,
                        "row_count": result.row_count,
                        "feature_count": result.feature_count,
                        "train_row_count": result.train_row_count,
                        "test_row_count": result.test_row_count,
                        "train_start_date": result.train_start_date,
                        "train_end_date": result.train_end_date,
                        "test_start_date": result.test_start_date,
                        "test_end_date": result.test_end_date,
                    },
                    metrics=result.metrics,
                    artifact_paths=[
                        ArtifactReference(result.model_path, "model"),
                        ArtifactReference(result.metrics_path, "metrics"),
                        ArtifactReference(result.predictions_path, "predictions"),
                        ArtifactReference(result.summary_path, "summary"),
                    ],
                )
    except MLflowNotInstalledError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(f"Baseline training complete: {result.summary_path}")
    print(result.metrics)


if __name__ == "__main__":
    main()
