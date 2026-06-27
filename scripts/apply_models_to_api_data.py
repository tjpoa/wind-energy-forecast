import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from schema import DATE_COLUMN, TARGET_COLUMN, english_column_to_legacy, rename_legacy_columns_to_english
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = BASE_DATA_PATH / "processed"
MODELS_PATH = PROJECT_ROOT / "models"


api_data_files = sorted(PROCESSED_DATA_PATH.glob("api_data_featured_*.csv"))
if not api_data_files:
    print("No 'api_data_featured_*.csv' file found in data/processed/.")
    print("Run scripts/process_api_data.py first.")
    raise SystemExit(1)

LATEST_API_DATA_FILE = api_data_files[-1]
print(f"Using latest API dataset: {LATEST_API_DATA_FILE}")


BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK = "ANN_Tuned"
BEST_MODEL_LOG_NAME_FROM_NOTEBOOK = "ANN_Tuned"


def load_new_data(filepath: Path) -> pd.DataFrame:
    """Load the latest processed API dataset and normalize it to English columns."""
    print(f"Loading new data from: {filepath}")
    df = pd.read_csv(filepath)
    df = rename_legacy_columns_to_english(df)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def load_trained_model_and_scalers(model_name: str, target_type: str, models_dir: Path):
    """Load a trained model and its associated scalers when required."""
    print(f"Loading best {target_type} target model: {model_name}")
    model_instance = None
    scaler_x_instance = None
    scaler_y_instance = None

    if "ANN" in model_name:
        model_path = models_dir / f"best_model_{target_type}_target_{model_name}.keras"
        scaler_x_path = models_dir / f"scaler_X_{target_type}_ann.joblib"
        scaler_y_path = models_dir / f"scaler_y_{target_type}_ann.joblib"

        if model_path.exists():
            model_instance = load_model(model_path)
        else:
            print(f"WARNING: ANN model file not found: {model_path}")

        if scaler_x_path.exists():
            scaler_x_instance = joblib.load(scaler_x_path)
        else:
            print(f"WARNING: X scaler file not found: {scaler_x_path} (required for ANN models)")

        if scaler_y_path.exists():
            scaler_y_instance = joblib.load(scaler_y_path)
        else:
            print(f"WARNING: Y scaler file not found: {scaler_y_path} (required for ANN models)")
    else:
        model_path = models_dir / f"best_model_{target_type}_target_{model_name}.joblib"
        if model_path.exists():
            model_instance = joblib.load(model_path)
        else:
            print(f"WARNING: model file not found: {model_path}")

    return model_instance, scaler_x_instance, scaler_y_instance


def load_training_feature_columns() -> list[str]:
    """Load the feature order used by the saved models."""
    df_historical_for_cols = pd.read_csv(PROCESSED_DATA_PATH / "agg_data_ml.csv", nrows=1)
    df_historical_for_cols = rename_legacy_columns_to_english(df_historical_for_cols)
    return df_historical_for_cols.drop(columns=[DATE_COLUMN, TARGET_COLUMN]).columns.tolist()


def prepare_data_for_prediction(df_new: pd.DataFrame, scaler_x=None):
    """Prepare feature columns for prediction using the saved training feature order."""
    x_new_english = df_new.drop(columns=[DATE_COLUMN, TARGET_COLUMN])
    feature_columns_english = load_training_feature_columns()

    x_ordered_english = pd.DataFrame(index=x_new_english.index)
    for english_col in feature_columns_english:
        if english_col in x_new_english.columns:
            x_ordered_english[english_col] = x_new_english[english_col]
        else:
            x_ordered_english[english_col] = 0

    # Existing saved scalers were trained with the original training schema.
    # Future retrained scalers may use English names, so choose the boundary
    # schema from the scaler's recorded feature names when available.
    x_ordered_training_schema = x_ordered_english.rename(
        columns={column: english_column_to_legacy(column) for column in x_ordered_english.columns}
    )
    x_for_model = x_ordered_training_schema

    expected_features = getattr(scaler_x, "feature_names_in_", None) if scaler_x is not None else None
    if expected_features is not None:
        expected_features = list(expected_features)
        if all(column in x_ordered_english.columns for column in expected_features):
            x_for_model = x_ordered_english.reindex(columns=expected_features)
        elif all(column in x_ordered_training_schema.columns for column in expected_features):
            x_for_model = x_ordered_training_schema.reindex(columns=expected_features)

    if scaler_x:
        print("Applying X scaling...")
        x_new_scaled = scaler_x.transform(x_for_model)
        return x_new_scaled, x_ordered_english.columns
    return x_for_model, x_ordered_english.columns


def make_predictions(model, x_data, scaler_y=None, is_log_target=False, positive_inf_cap=1e9):
    """Generate predictions and reverse scaling/log transforms when needed."""
    preds_transformed = model.predict(x_data)
    if scaler_y:
        preds_unscaled_transformed = scaler_y.inverse_transform(preds_transformed).flatten()
    else:
        preds_unscaled_transformed = preds_transformed.flatten()

    if is_log_target:
        preds_final = np.expm1(preds_unscaled_transformed)
        preds_final = np.nan_to_num(preds_final, nan=0.0, posinf=positive_inf_cap, neginf=0.0)
        preds_final[preds_final < 0] = 0
    else:
        preds_final = preds_unscaled_transformed
        preds_final[preds_final < 0] = 0

    return preds_final


def evaluate_predictions(y_true, y_pred, model_label: str):
    """Calculate and print prediction metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_safe = y_true.copy()
    y_true_safe[y_true_safe == 0] = 1e-6
    y_pred_safe = y_pred.copy()
    y_pred_safe[y_pred_safe == 0] = 1e-6
    mape = mean_absolute_percentage_error(y_true_safe, y_pred_safe) * 100

    print(f"\nMetrics for {model_label}:")
    print(f"  R2:   {r2:.6f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def historical_positive_cap() -> float:
    """Return a conservative cap for positive infinity after log inverse transforms."""
    historical_file = PROCESSED_DATA_PATH / "agg_data_ml.csv"
    if not historical_file.exists():
        return 1e9

    df_historical = pd.read_csv(historical_file)
    df_historical = rename_legacy_columns_to_english(df_historical)
    return df_historical[TARGET_COLUMN].max() * 1.5


def main() -> None:
    df_new = load_new_data(LATEST_API_DATA_FILE)
    if df_new.empty:
        print("No new data to process after dropping rows with missing target values.")
        return

    y_true_new = df_new[TARGET_COLUMN]
    positive_inf_cap = historical_positive_cap()

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
        x_new_prepared_orig, _ = prepare_data_for_prediction(df_new.copy(), scaler_x_orig)
        preds_orig_final = make_predictions(model_orig, x_new_prepared_orig, scaler_y_orig, is_log_target=False)

    print("\n--- Processing best log-transformed-target model ---")
    model_log, scaler_x_log, scaler_y_log = load_trained_model_and_scalers(
        BEST_MODEL_LOG_NAME_FROM_NOTEBOOK, "log", MODELS_PATH
    )
    if not model_log:
        print(f"Could not load log-target model {BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}.")
        preds_log_final = np.full_like(y_true_new, np.nan)
    else:
        actual_scaler_x_for_log = scaler_x_log if scaler_x_log else scaler_x_orig

        if "ANN" in BEST_MODEL_LOG_NAME_FROM_NOTEBOOK and (not actual_scaler_x_for_log or not scaler_y_log):
            print("X or Y scaler missing for the log ANN model. Predictions cannot be generated correctly.")
            preds_log_final = np.full_like(y_true_new, np.nan)
        else:
            x_new_prepared_log, _ = prepare_data_for_prediction(df_new.copy(), actual_scaler_x_for_log)
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

    comparison_df_new = pd.DataFrame(
        {
            DATE_COLUMN: df_new[DATE_COLUMN],
            "Actual_Wind_Production": y_true_new,
        }
    )
    if not np.isnan(preds_orig_final).all():
        comparison_df_new[f"Pred_Best_Original_{BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}"] = preds_orig_final
    if not np.isnan(preds_log_final).all():
        comparison_df_new[f"Pred_Best_Log_{BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}"] = preds_log_final

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
