import argparse
import os
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wind_forecast.evaluation import evaluate_predictions
from wind_forecast.inference import (
    BEST_MODEL_LOG_NAME_FROM_NOTEBOOK,
    BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK,
    build_prediction_comparison,
    historical_positive_cap,
    load_new_data,
    load_trained_model_and_scalers,
    make_predictions,
    model_and_scaler_paths,
    prepare_data_for_prediction,
    select_log_x_scaler,
)
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN
from wind_forecast.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_TRACKING_URI,
    ArtifactReference,
    TrackingConfig,
    flatten_metric_groups,
    log_run_data,
    start_tracking_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = BASE_DATA_PATH / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
HISTORICAL_PROCESSED_FILE = PROCESSED_DATA_PATH / "agg_data_ml.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Apply saved wind-forecast models to the latest API feature data.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Compatibility alias for --tracking-mode local.",
    )
    parser.add_argument(
        "--tracking-mode",
        choices=["local", "off"],
        default="off",
        help="Evaluation remains opt-in to preserve the interactive workflow.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"MLflow experiment name. Defaults to {DEFAULT_EXPERIMENT_NAME}.",
    )
    return parser.parse_args(argv)


def select_latest_api_data_file(processed_data_path: Path) -> Path:
    """Select the latest processed API feature file."""
    api_data_files = sorted(processed_data_path.glob("api_data_featured_*.csv"))
    if not api_data_files:
        print("No 'api_data_featured_*.csv' file found in data/processed/.")
        print("Run scripts/process_api_data.py first.")
        raise SystemExit(1)

    latest_api_data_file = api_data_files[-1]
    print(f"Using latest API dataset: {latest_api_data_file}")
    return latest_api_data_file


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mlflow:
        args.tracking_mode = "local"
    config = TrackingConfig(
        mode=args.tracking_mode,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
    )
    run_context = nullcontext()
    if config.mode == "local":
        run_context = start_tracking_run(
            "apply-models-to-api-data",
            config=config,
            tags={
                "workflow": "apply_models_to_api_data",
                "phase": "4A",
            },
        )

    with run_context:
        _run_workflow(args)


def _run_workflow(args: argparse.Namespace) -> None:
    latest_api_data_file = select_latest_api_data_file(PROCESSED_DATA_PATH)
    df_new = load_new_data(latest_api_data_file)
    if df_new.empty:
        print("No new data to process after dropping rows with missing target values.")
        return

    y_true_new = df_new[TARGET_COLUMN]
    positive_inf_cap = historical_positive_cap(HISTORICAL_PROCESSED_FILE)

    print("\n--- Processing best original-target model ---")
    model_orig, scaler_x_orig, scaler_y_orig = load_trained_model_and_scalers(
        BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK, "original", MODELS_PATH
    )
    if not model_orig:
        print(f"Could not load original-target model {BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}.")
        preds_orig_final = np.full_like(y_true_new, np.nan)
    elif "ANN" in BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK and (not scaler_x_orig or not scaler_y_orig):
        print("X or Y scaler missing for the original ANN model. Predictions cannot be generated correctly.")
        preds_orig_final = np.full_like(y_true_new, np.nan)
    else:
        x_new_prepared_orig, _ = prepare_data_for_prediction(
            df_new.copy(), scaler_x_orig, HISTORICAL_PROCESSED_FILE
        )
        preds_orig_final = make_predictions(model_orig, x_new_prepared_orig, scaler_y_orig, is_log_target=False)

    print("\n--- Processing best log-transformed-target model ---")
    model_log, scaler_x_log, scaler_y_log = load_trained_model_and_scalers(
        BEST_MODEL_LOG_NAME_FROM_NOTEBOOK, "log", MODELS_PATH
    )
    if not model_log:
        print(f"Could not load log-target model {BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}.")
        preds_log_final = np.full_like(y_true_new, np.nan)
    else:
        actual_scaler_x_for_log = select_log_x_scaler(scaler_x_log, scaler_x_orig)

        if "ANN" in BEST_MODEL_LOG_NAME_FROM_NOTEBOOK and (not actual_scaler_x_for_log or not scaler_y_log):
            print("X or Y scaler missing for the log ANN model. Predictions cannot be generated correctly.")
            preds_log_final = np.full_like(y_true_new, np.nan)
        else:
            x_new_prepared_log, _ = prepare_data_for_prediction(
                df_new.copy(), actual_scaler_x_for_log, HISTORICAL_PROCESSED_FILE
            )
            preds_log_final = make_predictions(
                model_log,
                x_new_prepared_log,
                scaler_y_log,
                is_log_target=True,
                positive_inf_cap=positive_inf_cap,
            )

    metrics_results = {}
    if not np.isnan(preds_orig_final).all():
        metrics_results[f"Best_Original_{BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}"] = evaluate_predictions(
            y_true_new, preds_orig_final, f"Best model (original target - {BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK})"
        )
    if not np.isnan(preds_log_final).all():
        metrics_results[f"Best_Log_{BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}"] = evaluate_predictions(
            y_true_new, preds_log_final, f"Best model (log target - {BEST_MODEL_LOG_NAME_FROM_NOTEBOOK})"
        )

    df_metrics_summary = pd.DataFrame(metrics_results).T
    print("\n--- Consolidated metrics on new data ---")
    print(df_metrics_summary)

    comparison_df_new = build_prediction_comparison(
        df_new,
        preds_orig_final,
        preds_log_final,
        BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK,
        BEST_MODEL_LOG_NAME_FROM_NOTEBOOK,
    )

    print("\n--- Final comparison DataFrame: first and last rows ---")
    print(comparison_df_new.head())
    print(comparison_df_new.tail())

    plt.figure(figsize=(15, 7))
    plot_limit = min(100, len(comparison_df_new))
    plt.plot(
        comparison_df_new[DATE_COLUMN][:plot_limit],
        comparison_df_new["Actual_Wind_Production"][:plot_limit],
        label="Actual",
        marker="o",
        linestyle="-",
        alpha=0.7,
    )
    if f"Pred_Best_Original_{BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}" in comparison_df_new.columns:
        plt.plot(
            comparison_df_new[DATE_COLUMN][:plot_limit],
            comparison_df_new[f"Pred_Best_Original_{BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}"][:plot_limit],
            label=f"Best original-target prediction ({BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK})",
            marker="x",
            linestyle="--",
            alpha=0.7,
        )
    if f"Pred_Best_Log_{BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}" in comparison_df_new.columns:
        plt.plot(
            comparison_df_new[DATE_COLUMN][:plot_limit],
            comparison_df_new[f"Pred_Best_Log_{BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}"][:plot_limit],
            label=f"Best log-target prediction ({BEST_MODEL_LOG_NAME_FROM_NOTEBOOK})",
            marker="s",
            linestyle=":",
            alpha=0.7,
        )

    plt.title(f"Wind production on new API data: first {plot_limit} points")
    plt.xlabel("Date")
    plt.ylabel("Wind energy production (kW)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    output_comparison_filename = PROCESSED_DATA_PATH / f"api_data_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    comparison_df_new.to_csv(output_comparison_filename, index=False)
    print(f"\nComparison DataFrame saved to: {output_comparison_filename}")

    if args.tracking_mode == "local":
        _log_mlflow_evaluation_run(
            latest_api_data_file=latest_api_data_file,
            output_comparison_filename=output_comparison_filename,
            row_count=len(df_new),
            positive_inf_cap=positive_inf_cap,
            metrics_results=metrics_results,
        )


def _log_mlflow_evaluation_run(
    *,
    latest_api_data_file: Path,
    output_comparison_filename: Path,
    row_count: int,
    positive_inf_cap: float,
    metrics_results: dict,
) -> None:
    original_model_path, original_scaler_x_path, original_scaler_y_path = model_and_scaler_paths(
        BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK,
        "original",
        MODELS_PATH,
    )
    log_model_path, log_scaler_x_path, log_scaler_y_path = model_and_scaler_paths(
        BEST_MODEL_LOG_NAME_FROM_NOTEBOOK,
        "log",
        MODELS_PATH,
    )

    params = {
        "workflow": "apply_models_to_api_data",
        "input_data_path": _display_path(latest_api_data_file),
        "historical_feature_path": _display_path(HISTORICAL_PROCESSED_FILE),
        "prediction_output_path": _display_path(output_comparison_filename),
        "row_count": row_count,
        "positive_inf_cap": positive_inf_cap,
        "original_model_name": BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK,
        "original_model_path": _display_path(original_model_path),
        "original_scaler_x_path": _display_optional_path(original_scaler_x_path),
        "original_scaler_y_path": _display_optional_path(original_scaler_y_path),
        "log_model_name": BEST_MODEL_LOG_NAME_FROM_NOTEBOOK,
        "log_model_path": _display_path(log_model_path),
        "log_scaler_x_path": _display_optional_path(log_scaler_x_path),
        "log_scaler_y_path": _display_optional_path(log_scaler_y_path),
    }

    log_run_data(
        params=params,
        metrics=flatten_metric_groups(metrics_results),
        artifact_paths=[
            ArtifactReference(
                path=output_comparison_filename,
                artifact_path="predictions",
            )
        ],
    )
    print("MLflow run logged to local tracking store.")


def _display_optional_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return _display_path(path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    os.makedirs(MODELS_PATH, exist_ok=True)
    main()
