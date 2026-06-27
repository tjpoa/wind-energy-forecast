import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN
from wind_forecast.evaluation import evaluate_predictions
from wind_forecast.inference import (
    BEST_MODEL_LOG_NAME_FROM_NOTEBOOK,
    BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK,
    build_prediction_comparison,
    historical_positive_cap,
    load_new_data,
    load_trained_model_and_scalers,
    make_predictions,
    prepare_data_for_prediction,
    select_log_x_scaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = BASE_DATA_PATH / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
HISTORICAL_PROCESSED_FILE = PROCESSED_DATA_PATH / "agg_data_ml.csv"


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


def main() -> None:
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


if __name__ == "__main__":
    os.makedirs(MODELS_PATH, exist_ok=True)
    main()
