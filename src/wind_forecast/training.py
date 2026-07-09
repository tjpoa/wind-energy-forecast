"""Deterministic baseline training helpers for wind-energy forecasting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from .paths import project_root
from .schemas import DATE_COLUMN, TARGET_COLUMN, rename_legacy_columns_to_english


BaselineModelType = Literal["extra_trees", "random_forest"]

OUTPUT_FILENAMES = {
    "model": "model.joblib",
    "metrics": "metrics.json",
    "predictions": "predictions.csv",
    "summary": "run_summary.json",
}


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological train/test split used for baseline evaluation."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_dates: pd.Series
    test_dates: pd.Series


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Outputs from a completed baseline training run."""

    model_type: BaselineModelType
    seed: int
    test_fraction: float
    n_estimators: int
    input_path: Path
    output_dir: Path
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    summary_path: Path
    row_count: int
    feature_count: int
    train_row_count: int
    test_row_count: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    metrics: dict[str, float]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready run summary."""
        return {
            "model_type": self.model_type,
            "seed": self.seed,
            "test_fraction": self.test_fraction,
            "n_estimators": self.n_estimators,
            "input_path": _display_path(self.input_path),
            "output_dir": _display_path(self.output_dir),
            "model_path": _display_path(self.model_path),
            "metrics_path": _display_path(self.metrics_path),
            "predictions_path": _display_path(self.predictions_path),
            "summary_path": _display_path(self.summary_path),
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "train_row_count": self.train_row_count,
            "test_row_count": self.test_row_count,
            "train_start_date": self.train_start_date,
            "train_end_date": self.train_end_date,
            "test_start_date": self.test_start_date,
            "test_end_date": self.test_end_date,
            "metrics": self.metrics,
        }


def load_training_table(path: str | Path) -> pd.DataFrame:
    """Load a feature-ready training table and normalize known legacy columns."""
    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(f"Training feature table is missing: {table_path}")

    frame = pd.read_csv(table_path)
    frame = rename_legacy_columns_to_english(frame)
    missing_columns = {DATE_COLUMN, TARGET_COLUMN}.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Training feature table is missing required columns: {sorted(missing_columns)}"
        )

    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    if result[DATE_COLUMN].isna().any():
        examples = frame.loc[result[DATE_COLUMN].isna(), DATE_COLUMN].astype(str).head(5)
        raise ValueError(f"Training table contains unparseable dates: {examples.tolist()}")
    result = result.sort_values(DATE_COLUMN).reset_index(drop=True)
    return result


def build_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return feature matrix, target vector, and dates from a training table."""
    missing_columns = {DATE_COLUMN, TARGET_COLUMN}.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Training frame is missing columns: {sorted(missing_columns)}")

    features = frame.drop(columns=[DATE_COLUMN, TARGET_COLUMN]).copy()
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError("Training target contains missing or non-numeric values.")

    numeric_features = features.apply(pd.to_numeric, errors="coerce")
    if numeric_features.isna().any().any():
        bad_columns = numeric_features.columns[numeric_features.isna().any()].tolist()
        raise ValueError(f"Training features contain missing or non-numeric values: {bad_columns}")

    return numeric_features, target, pd.to_datetime(frame[DATE_COLUMN])


def temporal_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    dates: pd.Series,
    *,
    test_fraction: float = 0.2,
) -> TemporalSplit:
    """Split rows chronologically, preserving the notebook's final holdout intent."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if len(features) != len(target) or len(features) != len(dates):
        raise ValueError("features, target, and dates must have the same length.")
    if len(features) < 2:
        raise ValueError("At least two rows are required for a train/test split.")

    split_index = int(len(features) * (1 - test_fraction))
    split_index = min(max(split_index, 1), len(features) - 1)
    return TemporalSplit(
        x_train=features.iloc[:split_index].copy(),
        x_test=features.iloc[split_index:].copy(),
        y_train=target.iloc[:split_index].copy(),
        y_test=target.iloc[split_index:].copy(),
        train_dates=dates.iloc[:split_index].copy(),
        test_dates=dates.iloc[split_index:].copy(),
    )


def train_baseline_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_type: BaselineModelType = "extra_trees",
    seed: int = 42,
    n_estimators: int = 100,
) -> Any:
    """Train a lightweight deterministic tree baseline."""
    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than zero.")

    if model_type == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported baseline model type: {model_type}")

    return model.fit(x_train, y_train)


def calculate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate stable regression metrics on the original target scale."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    y_true_safe = y_true_array.copy()
    y_pred_safe = y_pred_array.copy()
    y_true_safe[y_true_safe == 0] = 1e-6
    y_pred_safe[y_pred_safe == 0] = 1e-6

    return {
        "R2": float(r2_score(y_true_array, y_pred_array)),
        "MAE": float(mean_absolute_error(y_true_array, y_pred_array)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "MAPE (%)": float(mean_absolute_percentage_error(y_true_safe, y_pred_safe) * 100),
    }


def run_baseline_training(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    model_type: BaselineModelType = "extra_trees",
    seed: int = 42,
    test_fraction: float = 0.2,
    n_estimators: int = 100,
    overwrite: bool = False,
) -> BaselineTrainingResult:
    """Train a baseline model, write reproducible outputs, and return metadata."""
    resolved_input = Path(input_path)
    resolved_output = Path(output_dir)
    _validate_project_output_dir(resolved_output)
    output_paths = _output_paths(resolved_output)
    _ensure_output_paths_available(output_paths, overwrite=overwrite)

    table = load_training_table(resolved_input)
    features, target, dates = build_xy(table)
    split = temporal_train_test_split(
        features,
        target,
        dates,
        test_fraction=test_fraction,
    )
    model = train_baseline_model(
        split.x_train,
        split.y_train,
        model_type=model_type,
        seed=seed,
        n_estimators=n_estimators,
    )
    predictions = np.asarray(model.predict(split.x_test), dtype=float)
    metrics = calculate_regression_metrics(split.y_test, predictions)

    resolved_output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_paths["model"])

    predictions_frame = pd.DataFrame(
        {
            DATE_COLUMN: split.test_dates.dt.strftime("%Y-%m-%d"),
            "Actual_Wind_Production": split.y_test.to_numpy(dtype=float),
            "Predicted_Wind_Production": predictions,
        }
    )
    predictions_frame.to_csv(output_paths["predictions"], index=False, lineterminator="\n")

    result = BaselineTrainingResult(
        model_type=model_type,
        seed=seed,
        test_fraction=test_fraction,
        n_estimators=n_estimators,
        input_path=resolved_input,
        output_dir=resolved_output,
        model_path=output_paths["model"],
        metrics_path=output_paths["metrics"],
        predictions_path=output_paths["predictions"],
        summary_path=output_paths["summary"],
        row_count=len(table),
        feature_count=len(features.columns),
        train_row_count=len(split.x_train),
        test_row_count=len(split.x_test),
        train_start_date=_date_text(split.train_dates.iloc[0]),
        train_end_date=_date_text(split.train_dates.iloc[-1]),
        test_start_date=_date_text(split.test_dates.iloc[0]),
        test_end_date=_date_text(split.test_dates.iloc[-1]),
        metrics=metrics,
    )

    _write_json(output_paths["metrics"], metrics)
    _write_json(output_paths["summary"], result.summary())
    return result


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}


def _ensure_output_paths_available(paths: dict[str, Path], *, overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        display_paths = ", ".join(_display_path(path) for path in existing)
        raise FileExistsError(
            "Baseline training outputs already exist; rerun with --overwrite "
            f"to replace them: {display_paths}"
        )


def _validate_project_output_dir(output_dir: Path) -> None:
    root = project_root().resolve()
    resolved = output_dir.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return
    if not relative.parts or relative.parts[0] != "outputs":
        raise ValueError(
            "Project-local baseline training outputs must be written under outputs/."
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    root = project_root().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)
