"""Inference helpers for saved wind-energy forecasting models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import processed_data_dir
from .schemas import DATE_COLUMN, TARGET_COLUMN, english_column_to_legacy, rename_legacy_columns_to_english


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


def model_and_scaler_paths(model_name: str, target_type: str, models_dir: Path):
    """Return model and scaler paths using the saved artifact naming convention."""
    if "ANN" in model_name:
        return (
            models_dir / f"best_model_{target_type}_target_{model_name}.keras",
            models_dir / f"scaler_X_{target_type}_ann.joblib",
            models_dir / f"scaler_y_{target_type}_ann.joblib",
        )

    return models_dir / f"best_model_{target_type}_target_{model_name}.joblib", None, None


def load_trained_model_and_scalers(model_name: str, target_type: str, models_dir: Path):
    """Load a trained model and its associated scalers when required."""
    import joblib

    print(f"Loading best {target_type} target model: {model_name}")
    model_instance = None
    scaler_x_instance = None
    scaler_y_instance = None

    model_path, scaler_x_path, scaler_y_path = model_and_scaler_paths(model_name, target_type, models_dir)
    if "ANN" in model_name:
        from tensorflow.keras.models import load_model

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
        if model_path.exists():
            model_instance = joblib.load(model_path)
        else:
            print(f"WARNING: model file not found: {model_path}")

    return model_instance, scaler_x_instance, scaler_y_instance


def load_training_feature_columns(historical_file: Path | None = None) -> list[str]:
    """Load the feature order used by the saved models."""
    historical_file = historical_file or (processed_data_dir() / "agg_data_ml.csv")
    df_historical_for_cols = pd.read_csv(historical_file, nrows=1)
    df_historical_for_cols = rename_legacy_columns_to_english(df_historical_for_cols)
    return df_historical_for_cols.drop(columns=[DATE_COLUMN, TARGET_COLUMN]).columns.tolist()


def prepare_data_for_prediction(
    df_new: pd.DataFrame,
    scaler_x=None,
    historical_file: Path | None = None,
):
    """Prepare feature columns for prediction using the saved training feature order."""
    x_new_english = df_new.drop(columns=[DATE_COLUMN, TARGET_COLUMN])
    feature_columns_english = load_training_feature_columns(historical_file)

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


def select_log_x_scaler(scaler_x_log, scaler_x_orig):
    """Use the log X scaler when available, otherwise fall back to the original X scaler."""
    return scaler_x_log if scaler_x_log else scaler_x_orig


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


def historical_positive_cap(historical_file: Path | None = None) -> float:
    """Return a conservative cap for positive infinity after log inverse transforms."""
    historical_file = historical_file or (processed_data_dir() / "agg_data_ml.csv")
    if not historical_file.exists():
        return 1e9

    df_historical = pd.read_csv(historical_file)
    df_historical = rename_legacy_columns_to_english(df_historical)
    return df_historical[TARGET_COLUMN].max() * 1.5


def build_prediction_comparison(
    df_new: pd.DataFrame,
    preds_orig_final,
    preds_log_final,
    original_model_name: str,
    log_model_name: str,
) -> pd.DataFrame:
    """Build the final prediction comparison DataFrame."""
    comparison_df_new = pd.DataFrame(
        {
            DATE_COLUMN: df_new[DATE_COLUMN],
            "Actual_Wind_Production": df_new[TARGET_COLUMN],
        }
    )
    if not np.isnan(preds_orig_final).all():
        comparison_df_new[f"Pred_Best_Original_{original_model_name}"] = preds_orig_final
    if not np.isnan(preds_log_final).all():
        comparison_df_new[f"Pred_Best_Log_{log_model_name}"] = preds_log_final

    return comparison_df_new
