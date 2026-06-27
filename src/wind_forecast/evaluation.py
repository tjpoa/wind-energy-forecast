"""Evaluation helpers for wind-energy prediction outputs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


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
