"""Deterministic reference-model training for the accepted v2 dataset."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from .manifests import sha256_file
from .paths import project_root
from .schemas import DATE_COLUMN, TARGET_COLUMN
from .tracking import git_state
from .training import build_xy, load_training_table


DEFAULT_TRAIN_START = "2010-01-15"
DEFAULT_TRAIN_END = "2022-12-31"
DEFAULT_VALIDATION_START = "2023-01-01"
DEFAULT_VALIDATION_END = "2024-12-31"
DEFAULT_TEST_START = "2025-01-01"
DEFAULT_TEST_END = "2026-06-27"
PERSISTENCE_COLUMN = "Wind_Production_Lag1"

OUTPUT_FILENAMES = {
    "model": "model.joblib",
    "validation_predictions": "validation_predictions.csv",
    "test_predictions": "test_predictions.csv",
    "metrics": "metrics.json",
    "decision": "reference_decision.json",
    "leakage_audit": "leakage_audit.json",
    "dataset_manifest": "dataset_manifest.json",
    "model_manifest": "model_manifest.json",
    "environment": "environment.json",
    "reload_sample": "reload_sample.csv",
    "plot": "model_comparison.png",
    "summary": "run_summary.json",
}


@dataclass(frozen=True)
class V2TemporalSplit:
    """Fixed chronological train, validation, and sealed-test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class V2TrainingResult:
    """Paths and decisions emitted by a v2 reference-model run."""

    output_dir: Path
    paths: Mapping[str, Path]
    selected_model: str
    accepted_as_reference: bool
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
    feature_names: tuple[str, ...]
    input_sha256: str
    split_sha256: str
    row_counts: Mapping[str, int]


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    validation_start: str = DEFAULT_VALIDATION_START,
    validation_end: str = DEFAULT_VALIDATION_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> V2TemporalSplit:
    """Create explicit date-bounded partitions and reject overlap or stray rows."""
    if DATE_COLUMN not in frame:
        raise ValueError(f"Training frame is missing {DATE_COLUMN!r}.")
    dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        raise ValueError("Training frame contains invalid dates.")
    bounds = [
        pd.Timestamp(train_start),
        pd.Timestamp(train_end),
        pd.Timestamp(validation_start),
        pd.Timestamp(validation_end),
        pd.Timestamp(test_start),
        pd.Timestamp(test_end),
    ]
    if not (bounds[0] <= bounds[1] < bounds[2] <= bounds[3] < bounds[4] <= bounds[5]):
        raise ValueError("Chronological split bounds must be ordered and disjoint.")
    masks = (
        dates.between(bounds[0], bounds[1]),
        dates.between(bounds[2], bounds[3]),
        dates.between(bounds[4], bounds[5]),
    )
    assigned = masks[0].astype(int) + masks[1].astype(int) + masks[2].astype(int)
    if not assigned.eq(1).all():
        bad = dates[assigned.ne(1)].dt.strftime("%Y-%m-%d").head(5).tolist()
        raise ValueError(f"Every input row must belong to exactly one split: {bad}")
    parts = tuple(frame.loc[mask].copy().reset_index(drop=True) for mask in masks)
    if any(part.empty for part in parts):
        raise ValueError("Train, validation, and test splits must all be non-empty.")
    return V2TemporalSplit(*parts)


def persistence_predictions(frame: pd.DataFrame) -> np.ndarray:
    """Return the one-day production-lag persistence prediction."""
    if PERSISTENCE_COLUMN not in frame:
        raise ValueError(f"Persistence requires feature {PERSISTENCE_COLUMN!r}.")
    values = pd.to_numeric(frame[PERSISTENCE_COLUMN], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Persistence feature contains missing or non-finite values.")
    return values


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calculate original-scale metrics, including signed bias."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted values must have the same non-empty shape.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values.")
    return {
        "MAE": _stable_float(mean_absolute_error(actual, predicted)),
        "RMSE": _stable_float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": _stable_float(r2_score(actual, predicted)),
        "bias": _stable_float(np.mean(predicted - actual)),
        "MAPE (%)": _stable_float(
            mean_absolute_percentage_error(actual, predicted) * 100
        ),
    }


def add_persistence_skill(
    metrics: Mapping[str, float], persistence: Mapping[str, float]
) -> dict[str, float]:
    """Add MAE skill, where positive values improve on persistence."""
    baseline_mae = float(persistence["MAE"])
    if baseline_mae <= 0:
        raise ValueError("Persistence MAE must be greater than zero to calculate skill.")
    return {
        **metrics,
        "MAE_skill_vs_persistence": _stable_float(
            1.0 - float(metrics["MAE"]) / baseline_mae
        ),
    }


def select_candidate(candidate_metrics: Mapping[str, Mapping[str, float]]) -> str:
    """Select validation winner by MAE, preferring ExtraTrees on exact ties."""
    required = {"extra_trees", "random_forest"}
    if set(candidate_metrics) != required:
        raise ValueError(f"Candidate metrics must contain exactly {sorted(required)}.")
    extra_mae = float(candidate_metrics["extra_trees"]["MAE"])
    forest_mae = float(candidate_metrics["random_forest"]["MAE"])
    return "extra_trees" if extra_mae <= forest_mae else "random_forest"


def passes_reference_gate(model_metrics: Mapping[str, float], persistence: Mapping[str, float]) -> bool:
    """Accept only a strictly lower test MAE than persistence."""
    return float(model_metrics["MAE"]) < float(persistence["MAE"])


def split_assignment_sha256(split: V2TemporalSplit) -> str:
    """Hash every ordered date-to-split assignment for the accepted daily key."""
    assignments = [
        {"date": pd.Timestamp(value).strftime("%Y-%m-%d"), "split": split_name}
        for split_name, part in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        for value in part[DATE_COLUMN]
    ]
    return _json_sha256(assignments)


def audit_training_contract(
    frame: pd.DataFrame,
    split: V2TemporalSplit,
    *,
    upstream_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit training-time leakage controls and upstream feature recomputation."""
    dates = pd.to_datetime(frame[DATE_COLUMN])
    features, _, _ = build_xy(frame)
    checks = {
        "dates_unique": bool(dates.is_unique),
        "dates_strictly_increasing": bool(dates.is_monotonic_increasing and dates.is_unique),
        "target_excluded_from_features": TARGET_COLUMN not in features.columns,
        "persistence_lag_present": PERSISTENCE_COLUMN in features.columns,
        "features_are_finite": bool(np.isfinite(features.to_numpy(dtype=float)).all()),
        "splits_disjoint": _splits_are_disjoint(split),
        "splits_ordered": _splits_are_ordered(split),
        "no_scaler_or_global_transform": True,
        "test_not_used_for_selection": True,
        "contemporaneous_weather_allowed_for_hindcast_only": True,
    }
    if upstream_validation is not None:
        checks["upstream_feature_recomputation_passed"] = bool(
            upstream_validation.get("passed")
        )
        checks["upstream_validation_has_no_errors"] = not bool(
            upstream_validation.get("errors")
        )
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": "wind_forecast.v2_leakage_audit.v1",
        "forecast_contract": "historical_daily_hindcast",
        "contemporaneous_weather_policy": (
            "ERA5-Land weather for the target date is permitted only for historical hindcast; "
            "this run is not a day-ahead operational forecast."
        ),
        "feature_recomputation_source": "wind_forecast.validation.feature_ready",
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "upstream_validation": upstream_validation,
    }


def run_v2_reference_training(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    n_estimators: int = 100,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    validation_start: str = DEFAULT_VALIDATION_START,
    validation_end: str = DEFAULT_VALIDATION_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
    dataset_version: str = "v2",
    transformation_version: str = "feature_ready_ren_era5_land_v2_2A.18",
    upstream_validation: Mapping[str, Any] | None = None,
) -> V2TrainingResult:
    """Train, select, refit, evaluate once, and persist the v2 reference candidate."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    _validate_output_dir(output_dir)
    paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    existing = [path for path in paths.values() if path.exists()]
    if output_dir.exists() or existing:
        raise FileExistsError(f"V2 output directory already exists: {output_dir}")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than zero.")

    frame = load_training_table(input_path)
    if frame[DATE_COLUMN].duplicated().any():
        raise ValueError("V2 training dates must be unique.")
    split = chronological_split(
        frame,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
    )
    audit = audit_training_contract(frame, split, upstream_validation=upstream_validation)
    if not audit["passed"]:
        raise ValueError(f"Temporal leakage audit failed: {audit['failures']}")

    x_train, y_train, _ = build_xy(split.train)
    x_validation, y_validation, validation_dates = build_xy(split.validation)
    x_test, y_test, test_dates = build_xy(split.test)
    feature_names = tuple(str(column) for column in x_train.columns)
    if tuple(x_validation.columns) != feature_names or tuple(x_test.columns) != feature_names:
        raise ValueError("Feature order differs between chronological splits.")

    candidate_models = {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1
        ),
    }
    validation_persistence = persistence_predictions(split.validation)
    validation_metrics: dict[str, dict[str, float]] = {
        "persistence": regression_metrics(y_validation, validation_persistence)
    }
    validation_predictions: dict[str, np.ndarray] = {
        "persistence": validation_persistence
    }
    for name, model in candidate_models.items():
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_validation), dtype=float)
        validation_predictions[name] = predictions
        validation_metrics[name] = add_persistence_skill(
            regression_metrics(y_validation, predictions),
            validation_metrics["persistence"],
        )
    selected = select_candidate(
        {name: validation_metrics[name] for name in candidate_models}
    )

    train_validation = pd.concat([split.train, split.validation], ignore_index=True)
    x_refit, y_refit, _ = build_xy(train_validation)
    final_model = candidate_models[selected].__class__(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1
    ).fit(x_refit, y_refit)
    test_model_predictions = np.asarray(final_model.predict(x_test), dtype=float)
    test_persistence = persistence_predictions(split.test)
    test_persistence_metrics = regression_metrics(y_test, test_persistence)
    test_model_metrics = add_persistence_skill(
        regression_metrics(y_test, test_model_predictions), test_persistence_metrics
    )
    accepted = passes_reference_gate(test_model_metrics, test_persistence_metrics)
    metrics = {
        "validation": validation_metrics,
        "test": {selected: test_model_metrics, "persistence": test_persistence_metrics},
    }

    output_dir.mkdir(parents=True)
    joblib.dump(final_model, paths["model"])
    _prediction_frame(
        validation_dates, y_validation, validation_predictions
    ).to_csv(
        paths["validation_predictions"],
        index=False,
        lineterminator="\n",
        float_format="%.9f",
    )
    _prediction_frame(
        test_dates,
        y_test,
        {selected: test_model_predictions, "persistence": test_persistence},
    ).to_csv(
        paths["test_predictions"],
        index=False,
        lineterminator="\n",
        float_format="%.9f",
    )

    reload_sample = x_test.head(5).copy()
    reload_sample["Expected_Prediction"] = test_model_predictions[: len(reload_sample)]
    reload_sample.to_csv(
        paths["reload_sample"], index=False, lineterminator="\n", float_format="%.9f"
    )
    reloaded = joblib.load(paths["model"])
    if not np.allclose(
        np.asarray(reloaded.predict(reload_sample[list(feature_names)])),
        reload_sample["Expected_Prediction"].to_numpy(),
        rtol=1e-12,
        atol=1e-9,
    ):
        raise RuntimeError("Reloaded model predictions differ from the saved sample.")

    _write_comparison_plot(metrics, paths["plot"])
    row_counts = {
        "total": len(frame),
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
        "refit_train_validation": len(train_validation),
    }
    split_payload = {
        "train": _period(split.train),
        "validation": _period(split.validation),
        "test": _period(split.test),
        "row_counts": row_counts,
    }
    split_sha256 = split_assignment_sha256(split)
    input_sha256 = sha256_file(input_path)
    schema_sha256 = _json_sha256(list(feature_names))
    lineage = git_state()
    _write_json(paths["metrics"], {"schema_version": "wind_forecast.v2_metrics.v1", **metrics})
    _write_json(
        paths["decision"],
        {
            "schema_version": "wind_forecast.v2_reference_decision.v1",
            "selected_model": selected,
            "selection_split": "validation",
            "selection_metric": "MAE",
            "tie_break": "extra_trees",
            "gate": "selected test MAE strictly lower than persistence test MAE",
            "accepted_as_reference": accepted,
            "status": "selected_not_promoted" if accepted else "rejected_not_promoted",
            "automatic_promotion": False,
        },
    )
    _write_json(paths["leakage_audit"], audit)
    dataset_manifest = {
        "schema_version": "wind_forecast.v2_training_dataset.v1",
        "dataset_version": dataset_version,
        "transformation_version": transformation_version,
        "role": "accepted_v2_reference_training_input",
        "path": _display_path(input_path),
        "sha256": input_sha256,
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "target": TARGET_COLUMN,
        "feature_names": list(feature_names),
        "feature_schema_sha256": schema_sha256,
        "split_assignment_sha256": split_sha256,
        "splits": split_payload,
    }
    _write_json(paths["dataset_manifest"], dataset_manifest)
    _write_json(paths["environment"], _environment_manifest(lineage))
    _write_json(
        paths["model_manifest"],
        {
            "schema_version": "wind_forecast.v2_model_manifest.v1",
            "task": "daily_wind_production_historical_hindcast",
            "model_type": selected,
            "model_sha256": sha256_file(paths["model"]),
            "dataset_version": dataset_version,
            "dataset_sha256": input_sha256,
            "feature_names": list(feature_names),
            "feature_schema_sha256": schema_sha256,
            "split_assignment_sha256": split_sha256,
            "parameters": final_model.get_params(deep=True),
            "scaler_required": False,
            "scaler": None,
            "metrics": metrics,
            "reference_status": "selected_not_promoted" if accepted else "rejected_not_promoted",
        },
    )
    artifact_hashes = {
        path.name: sha256_file(path)
        for name, path in paths.items()
        if name != "summary" and path.is_file()
    }
    _write_json(
        paths["summary"],
        {
            "schema_version": "wind_forecast.v2_training_run.v1",
            "selected_model": selected,
            "accepted_as_reference": accepted,
            "dataset_version": dataset_version,
            "dataset_sha256": input_sha256,
            "split_assignment_sha256": split_sha256,
            "row_counts": row_counts,
            "feature_count": len(feature_names),
            "seed": seed,
            "n_estimators": n_estimators,
            "scaler_required": False,
            "test_opened_once_after_selection": True,
            "artifact_sha256": artifact_hashes,
        },
    )
    return V2TrainingResult(
        output_dir=output_dir,
        paths=paths,
        selected_model=selected,
        accepted_as_reference=accepted,
        metrics=metrics,
        feature_names=feature_names,
        input_sha256=input_sha256,
        split_sha256=split_sha256,
        row_counts=row_counts,
    )


def _splits_are_disjoint(split: V2TemporalSplit) -> bool:
    sets = [set(pd.to_datetime(part[DATE_COLUMN])) for part in (split.train, split.validation, split.test)]
    return not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def _splits_are_ordered(split: V2TemporalSplit) -> bool:
    train = pd.to_datetime(split.train[DATE_COLUMN])
    validation = pd.to_datetime(split.validation[DATE_COLUMN])
    test = pd.to_datetime(split.test[DATE_COLUMN])
    return bool(train.max() < validation.min() < validation.max() < test.min())


def _prediction_frame(dates: pd.Series, actual: pd.Series, predictions: Mapping[str, np.ndarray]) -> pd.DataFrame:
    base = pd.DataFrame({DATE_COLUMN: pd.to_datetime(dates).dt.strftime("%Y-%m-%d"), "Actual_Wind_Production": np.asarray(actual, dtype=float)})
    frames = []
    for model_name, values in predictions.items():
        item = base.copy()
        item["model"] = model_name
        item["Predicted_Wind_Production"] = np.asarray(values, dtype=float)
        frames.append(item)
    return pd.concat(frames, ignore_index=True)


def _period(frame: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(frame[DATE_COLUMN])
    return {"start": dates.min().strftime("%Y-%m-%d"), "end": dates.max().strftime("%Y-%m-%d"), "row_count": len(frame)}


def _write_comparison_plot(metrics: Mapping[str, Any], path: Path) -> None:
    names = list(metrics["validation"])
    values = [metrics["validation"][name]["MAE"] for name in names]
    figure = Figure(figsize=(8, 4.5))
    axis = figure.subplots()
    axis.bar(names, values)
    axis.set_ylabel("Validation MAE")
    axis.set_title("V2 candidate comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=120, metadata={"Software": "wind-energy-forecast"})


def _environment_manifest(lineage: Mapping[str, Any]) -> dict[str, Any]:
    packages = {}
    for package in ("joblib", "matplotlib", "mlflow", "numpy", "pandas", "scikit-learn"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {"schema_version": "wind_forecast.v2_environment.v1", "python": platform.python_version(), "python_implementation": platform.python_implementation(), "platform": sys.platform, "packages": packages, **lineage}


def _validate_output_dir(output_dir: Path) -> None:
    root = project_root().resolve()
    resolved = output_dir.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return
    if not relative.parts or relative.parts[0] != "outputs":
        raise ValueError("Project-local v2 training outputs must be written under outputs/.")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _stable_float(value: Any) -> float:
    """Normalize insignificant parallel floating-point reduction noise."""
    return float(round(float(value), 9))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
