"""Governed ANN v2 training and composite artifact loading.

This module deliberately keeps TensorFlow lazy.  The existing tree-based v2
reference and the legacy v1 serving path therefore retain their import and
artifact contracts while this opt-in module handles the scaled Keras bundle.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler

from .manifests import sha256_file
from .paths import project_root
from .schemas import DATE_COLUMN, TARGET_COLUMN
from .tracking import git_state
from .training import build_xy, load_training_table
from .v2_scaling import SCALER_FILENAMES
from .v2_training import (
    DEFAULT_TEST_END,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VALIDATION_END,
    DEFAULT_VALIDATION_START,
)


ANN_BUNDLE_SCHEMA = "wind_forecast.v2_ann_candidate_bundle.v1"
ANN_MODEL_MANIFEST_SCHEMA = "wind_forecast.v2_ann_model_manifest.v1"
ANN_DATASET_MANIFEST_SCHEMA = "wind_forecast.v2_ann_dataset_manifest.v1"
ANN_TRAINING_SCHEMA = "wind_forecast.v2_ann_training_run.v1"
ANN_PREDICTION_CONTRACT = "raw_features_to_original_target"
DEFAULT_DATASET_PATH = (
    project_root()
    / "data/processed/v2/ml_features/feature_ready_ren_era5_land_v2/feature_ready_daily.csv"
)
DEFAULT_SCALER_DIR = (
    project_root() / "models/v2/scalers/feature_ready_ren_era5_land_v2"
)

ANN_OUTPUT_FILENAMES = {
    "model": "model.keras",
    "scaler_x": "scaler_x.joblib",
    "scaler_y": "scaler_y.joblib",
    "scaler_manifest": "scaler_manifest.json",
    "validation_predictions": "validation_predictions.csv",
    "test_predictions": "test_predictions.csv",
    "training_evidence": "training_evidence.csv",
    "metrics": "metrics.json",
    "variant_comparison": "variant_comparison.json",
    "history": "training_history.json",
    "dataset_manifest": "dataset_manifest.json",
    "model_manifest": "model_manifest.json",
    "environment": "environment.json",
    "lineage": "lineage.json",
    "transformation_contract": "transformation_contract.json",
    "reload_sample": "reload_sample.csv",
    "summary": "run_summary.json",
    "bundle_manifest": "bundle_manifest.json",
}


@dataclass(frozen=True)
class ANNTrainingConfig:
    """Explicit date and reproducibility contract for one ANN run."""

    input_path: Path = DEFAULT_DATASET_PATH
    scaler_dir: Path = DEFAULT_SCALER_DIR
    output_dir: Path = project_root() / "outputs/training/v2_ann_candidate"
    seed: int = 42
    train_start: str = DEFAULT_TRAIN_START
    train_end: str = DEFAULT_TRAIN_END
    validation_start: str = DEFAULT_VALIDATION_START
    validation_end: str = DEFAULT_VALIDATION_END
    test_start: str = DEFAULT_TEST_START
    test_end: str = DEFAULT_TEST_END
    max_epochs: int = 200
    batch_size: int = 32
    patience: int = 50

    def __post_init__(self) -> None:
        for name in ("input_path", "scaler_dir", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.max_epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("Training limits must be positive.")


@dataclass(frozen=True)
class ANNTrainingResult:
    """Paths and decisions from one sealed ANN candidate training run."""

    output_dir: Path
    selected_variant: str
    selected_epochs: int
    feature_names: tuple[str, ...]
    metrics: Mapping[str, Mapping[str, float]]
    paths: Mapping[str, Path]
    input_sha256: str
    scaler_manifest_sha256: str
    split_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "output_dir": _display_path(self.output_dir),
            "selected_variant": self.selected_variant,
            "selected_epochs": self.selected_epochs,
            "feature_names": list(self.feature_names),
            "metrics": dict(self.metrics),
            "paths": {key: _display_path(value) for key, value in self.paths.items()},
            "input_sha256": self.input_sha256,
            "scaler_manifest_sha256": self.scaler_manifest_sha256,
            "split_sha256": self.split_sha256,
        }


class V2ANNError(RuntimeError):
    """Raised when the ANN v2 artifact contract cannot be trusted."""


class V2ANNPredictor:
    """Composite predictor that accepts raw v2 features and returns raw target."""

    def __init__(
        self,
        *,
        model: Any,
        scaler_x: MinMaxScaler,
        scaler_y: MinMaxScaler,
        feature_names: tuple[str, ...],
        target_variant: str,
    ) -> None:
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.feature_names = feature_names
        self.target_variant = target_variant

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict original-scale production from an ordered raw feature frame."""
        if list(frame.columns) != list(self.feature_names):
            raise V2ANNError("ANN predictor feature order differs from its manifest.")
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise V2ANNError("ANN predictor inputs must be finite numeric values.")
        scaled = self.scaler_x.transform(numeric)
        output = np.asarray(self.model.predict(scaled, verbose=0), dtype=float).reshape(-1, 1)
        if not np.isfinite(output).all():
            raise V2ANNError("ANN predictor returned non-finite values.")
        transformed = self.scaler_y.inverse_transform(output).reshape(-1)
        if self.target_variant == "log1p":
            transformed = np.expm1(transformed)
        elif self.target_variant != "original":
            raise V2ANNError(f"Unsupported ANN target variant: {self.target_variant!r}.")
        if not np.isfinite(transformed).all() or (transformed < 0).any():
            raise V2ANNError("ANN predictor returned invalid non-negative target values.")
        return transformed


def fit_v2_ann_candidate(config: ANNTrainingConfig) -> ANNTrainingResult:
    """Train, compare, and seal one scaled ANN v2 candidate."""
    if config.output_dir.exists():
        raise FileExistsError(f"ANN output directory already exists: {config.output_dir}")
    frame = load_training_table(config.input_path)
    features, target, dates = build_xy(frame)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise V2ANNError("V2 ANN features must be finite.")
    target_values = target.to_numpy(dtype=float)
    if not np.isfinite(target_values).all() or (target_values < 0).any():
        raise V2ANNError("V2 ANN target must be finite and non-negative.")
    if dates.duplicated().any():
        raise V2ANNError("V2 ANN dates must be unique.")
    if config.input_path.resolve() == DEFAULT_DATASET_PATH.resolve() and len(features.columns) != 56:
        raise V2ANNError("The accepted v2 ANN dataset must contain exactly 56 features.")

    split = _date_split(frame, config)
    scaler_bundle = _load_scaler_bundle(config.scaler_dir, config.input_path, features.columns)
    _set_determinism(config.seed)

    variant_results: dict[str, dict[str, Any]] = {}
    for offset, variant in enumerate(("original", "log1p")):
        _set_determinism(config.seed + offset)
        variant_results[variant] = _fit_selection_variant(
            split,
            variant=variant,
            seed=config.seed + offset,
            max_epochs=config.max_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
        )
    selected_variant = (
        "original"
        if variant_results["original"]["metrics"]["MAE"]
        <= variant_results["log1p"]["metrics"]["MAE"]
        else "log1p"
    )
    selected_epochs = int(variant_results[selected_variant]["best_epoch"])

    _set_determinism(config.seed)
    selected_scaler_x = scaler_bundle["x_original" if selected_variant == "original" else "x_log"]
    selected_scaler_y = scaler_bundle["y_original" if selected_variant == "original" else "y_log"]
    refit = pd.concat([split.train, split.validation], ignore_index=True)
    refit_x = refit[list(features.columns)]
    refit_y = pd.to_numeric(refit[TARGET_COLUMN], errors="coerce").to_numpy(float)
    x_scaled = selected_scaler_x.transform(refit_x)
    y_transformed = refit_y if selected_variant == "original" else np.log1p(refit_y)
    y_scaled = selected_scaler_y.transform(y_transformed.reshape(-1, 1))
    model = _build_model(len(features.columns), seed=config.seed)
    model.fit(
        x_scaled,
        y_scaled,
        epochs=selected_epochs,
        batch_size=config.batch_size,
        shuffle=True,
        verbose=0,
    )
    predictor = V2ANNPredictor(
        model=model,
        scaler_x=selected_scaler_x,
        scaler_y=selected_scaler_y,
        feature_names=tuple(str(column) for column in features.columns),
        target_variant=selected_variant,
    )
    validation_predictions = predictor.predict(split.validation[list(features.columns)])
    test_predictions = predictor.predict(split.test[list(features.columns)])
    for values in (validation_predictions, test_predictions):
        if not np.isfinite(values).all() or (values < 0).any():
            raise V2ANNError("ANN candidate produced invalid predictions.")

    metrics = {
        "validation": _metrics(split.validation[TARGET_COLUMN], validation_predictions),
        "test": _metrics(split.test[TARGET_COLUMN], test_predictions),
    }
    input_sha256 = sha256_file(config.input_path)
    scaler_manifest_sha256 = sha256_file(config.scaler_dir / SCALER_FILENAMES["manifest"])
    feature_names = tuple(str(column) for column in features.columns)
    split_sha256 = _split_sha256(split)
    paths = {key: config.output_dir / value for key, value in ANN_OUTPUT_FILENAMES.items()}
    config.output_dir.mkdir(parents=True)
    model.save(paths["model"], include_optimizer=True)
    shutil.copy2(
        scaler_bundle["paths"]["x_original" if selected_variant == "original" else "x_log"],
        paths["scaler_x"],
    )
    shutil.copy2(
        scaler_bundle["paths"]["y_original" if selected_variant == "original" else "y_log"],
        paths["scaler_y"],
    )
    shutil.copy2(scaler_bundle["manifest_path"], paths["scaler_manifest"])

    training_evidence = refit[[DATE_COLUMN, TARGET_COLUMN, *feature_names]].copy()
    training_evidence["Expected_Prediction"] = predictor.predict(refit[list(feature_names)])
    training_evidence.to_csv(paths["training_evidence"], index=False, lineterminator="\n")
    _prediction_frame(split.validation, validation_predictions).to_csv(
        paths["validation_predictions"], index=False, lineterminator="\n"
    )
    _prediction_frame(split.test, test_predictions).to_csv(
        paths["test_predictions"], index=False, lineterminator="\n"
    )
    training_evidence.head(5).to_csv(paths["reload_sample"], index=False, lineterminator="\n")

    model_manifest = {
        "schema_version": ANN_MODEL_MANIFEST_SCHEMA,
        "bundle_schema_version": ANN_BUNDLE_SCHEMA,
        "artifact_type": "keras_scaled_v2",
        "task": "daily_wind_production_historical_hindcast",
        "model_family": "ANN",
        "model_type": "keras.Sequential",
        "model_path": paths["model"].name,
        "model_sha256": sha256_file(paths["model"]),
        "target_variant": selected_variant,
        "target_transform": "identity" if selected_variant == "original" else "log1p",
        "prediction_contract": ANN_PREDICTION_CONTRACT,
        "scaler_required": True,
        "scaler_paths": {"x": paths["scaler_x"].name, "y": paths["scaler_y"].name},
        "scaler_manifest_path": paths["scaler_manifest"].name,
        "scaler_manifest_sha256": sha256_file(paths["scaler_manifest"]),
        "feature_names": list(feature_names),
        "feature_schema_sha256": _feature_schema_sha256(feature_names),
        "dataset_sha256": input_sha256,
        "seed": config.seed,
        "parameters": {
            "hidden_units": 32,
            "activation": "relu",
            "dropout": 0.2,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "mean_squared_error",
            "batch_size": config.batch_size,
            "epochs": selected_epochs,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "shuffle": True,
        },
        "metrics": metrics,
        "status": "candidate_not_promoted",
    }
    dataset_manifest = {
        "schema_version": ANN_DATASET_MANIFEST_SCHEMA,
        "dataset_version": "v2",
        "transformation_version": "feature_ready_ren_era5_land_v2_2A.18",
        "path": _display_path(config.input_path),
        "sha256": input_sha256,
        "row_count": len(frame),
        "feature_names": list(feature_names),
        "feature_schema_sha256": _feature_schema_sha256(feature_names),
        "target": TARGET_COLUMN,
        "splits": {
            name: {
                "start": _date_text(part[DATE_COLUMN].iloc[0]),
                "end": _date_text(part[DATE_COLUMN].iloc[-1]),
                "row_count": len(part),
            }
            for name, part in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
        },
        "fit_scaler_scope": "train_plus_validation_excluding_sealed_test",
        "scaler_manifest_sha256": scaler_manifest_sha256,
    }
    history = {
        variant: variant_results[variant]["history"] for variant in variant_results
    }
    _write_json(paths["metrics"], {"schema_version": ANN_TRAINING_SCHEMA, **metrics})
    _write_json(
        paths["variant_comparison"],
        {
            "schema_version": "wind_forecast.v2_ann_variant_comparison.v1",
            "selection_metric": "validation.MAE_original_scale",
            "tie_breaker": "original",
            "selected_variant": selected_variant,
            "variants": {
                name: {
                    "metrics": value["metrics"],
                    "best_epoch": value["best_epoch"],
                }
                for name, value in variant_results.items()
            },
            "test_used_for_selection": False,
        },
    )
    _write_json(paths["history"], history)
    _write_json(paths["dataset_manifest"], dataset_manifest)
    _write_json(paths["model_manifest"], model_manifest)
    _write_json(paths["environment"], _environment_manifest())
    _write_json(
        paths["lineage"],
        {
            "schema_version": "wind_forecast.v2_ann_lineage.v1",
            "dataset_sha256": input_sha256,
            "scaler_manifest_sha256": scaler_manifest_sha256,
            "split_sha256": split_sha256,
            "feature_schema_sha256": _feature_schema_sha256(feature_names),
            "git": _git_state(),
            "safeguards": {
                "test_used_for_selection": False,
                "automatic_registry_write": False,
                "automatic_promotion": False,
                "network_requests": False,
            },
        },
    )
    _write_json(
        paths["transformation_contract"],
        {
            "schema_version": "wind_forecast.v2_ann_transformation_contract.v1",
            "input_contract": "56 raw ordered features without scaling",
            "feature_names": list(feature_names),
            "feature_schema_sha256": _feature_schema_sha256(feature_names),
            "scaler_manifest": paths["scaler_manifest"].name,
            "scaler_manifest_sha256": sha256_file(paths["scaler_manifest"]),
            "x_scaler": paths["scaler_x"].name,
            "y_scaler": paths["scaler_y"].name,
            "target_variant": selected_variant,
            "target_transform": "identity" if selected_variant == "original" else "log1p",
            "inverse_transform": "identity" if selected_variant == "original" else "expm1",
            "output_contract": "Wind_Production in original units; finite and non-negative",
        },
    )
    _write_json(
        paths["summary"],
        {
            "schema_version": ANN_TRAINING_SCHEMA,
            "selected_variant": selected_variant,
            "selected_epochs": selected_epochs,
            "input_sha256": input_sha256,
            "scaler_manifest_sha256": scaler_manifest_sha256,
            "split_sha256": split_sha256,
            "metrics": metrics,
            "automatic_promotion": False,
            "registry_write": False,
            "network_requests": False,
        },
    )
    _write_json(
        paths["bundle_manifest"],
        {
            "schema_version": ANN_BUNDLE_SCHEMA,
            "artifact_type": "keras_scaled_v2",
            "files": {
                path.name: {"sha256": sha256_file(path)}
                for name, path in paths.items()
                if name != "bundle_manifest"
            },
            "model_sha256": sha256_file(paths["model"]),
            "scaler_manifest_sha256": sha256_file(paths["scaler_manifest"]),
        },
    )
    return ANNTrainingResult(
        output_dir=config.output_dir,
        selected_variant=selected_variant,
        selected_epochs=selected_epochs,
        feature_names=feature_names,
        metrics=metrics,
        paths=paths,
        input_sha256=input_sha256,
        scaler_manifest_sha256=scaler_manifest_sha256,
        split_sha256=split_sha256,
    )


def load_v2_ann_bundle(path: str | Path) -> V2ANNPredictor:
    """Validate and load an immutable ANN bundle for explicit callers."""
    root = Path(path)
    manifest = _read_json(root / ANN_OUTPUT_FILENAMES["model_manifest"])
    if manifest.get("schema_version") != ANN_MODEL_MANIFEST_SCHEMA:
        raise V2ANNError("Unsupported ANN model manifest schema.")
    if manifest.get("artifact_type") != "keras_scaled_v2":
        raise V2ANNError("Bundle is not a scaled ANN v2 artifact.")
    feature_names = tuple(str(value) for value in manifest.get("feature_names") or ())
    if not feature_names:
        raise V2ANNError("ANN model manifest has no features.")
    model_path = root / str(manifest.get("model_path") or ANN_OUTPUT_FILENAMES["model"])
    scaler_x_path = root / str(manifest.get("scaler_paths", {}).get("x") or ANN_OUTPUT_FILENAMES["scaler_x"])
    scaler_y_path = root / str(manifest.get("scaler_paths", {}).get("y") or ANN_OUTPUT_FILENAMES["scaler_y"])
    scaler_manifest_path = root / str(manifest.get("scaler_manifest_path") or ANN_OUTPUT_FILENAMES["scaler_manifest"])
    for artifact in (model_path, scaler_x_path, scaler_y_path, scaler_manifest_path):
        if not artifact.is_file():
            raise V2ANNError(f"ANN bundle artifact is missing: {artifact.name}")
    if manifest.get("model_sha256") != sha256_file(model_path):
        raise V2ANNError("ANN model checksum differs from its manifest.")
    if manifest.get("scaler_manifest_sha256") != sha256_file(scaler_manifest_path):
        raise V2ANNError("ANN scaler manifest checksum differs from its manifest.")
    scaler_manifest = _read_json(scaler_manifest_path)
    if scaler_manifest.get("schema_version") != "wind_forecast.v2_scaler_manifest.v1":
        raise V2ANNError("ANN bundle has an unsupported source scaler manifest.")
    if [str(value) for value in scaler_manifest.get("feature_names") or ()] != list(feature_names):
        raise V2ANNError("ANN source scaler feature order differs from its manifest.")
    variant = str(manifest.get("target_variant") or "")
    if variant not in {"original", "log1p"}:
        raise V2ANNError(f"Unsupported ANN target variant: {variant!r}.")
    source_x_name = "x_original" if variant == "original" else "x_log"
    source_y_name = "y_original" if variant == "original" else "y_log"
    for source_name, artifact in ((source_x_name, scaler_x_path), (source_y_name, scaler_y_path)):
        source_entry = (scaler_manifest.get("scalers") or {}).get(source_name, {})
        if source_entry.get("sha256") != sha256_file(artifact):
            raise V2ANNError(f"ANN bundle scaler checksum differs from source manifest: {source_name}.")
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    if not isinstance(scaler_x, MinMaxScaler) or not isinstance(scaler_y, MinMaxScaler):
        raise V2ANNError("ANN bundle scalers must be MinMaxScaler instances.")
    if getattr(scaler_x, "n_features_in_", None) != len(feature_names):
        raise V2ANNError("ANN X scaler feature count differs from its manifest.")
    if getattr(scaler_y, "n_features_in_", None) != 1:
        raise V2ANNError("ANN y scaler must have one target feature.")
    if list(getattr(scaler_x, "feature_names_in_", feature_names)) != list(feature_names):
        raise V2ANNError("ANN X scaler feature order differs from its manifest.")
    bundle_manifest_path = root / ANN_OUTPUT_FILENAMES["bundle_manifest"]
    if bundle_manifest_path.is_file():
        bundle_manifest = _read_json(bundle_manifest_path)
        if bundle_manifest.get("schema_version") == ANN_BUNDLE_SCHEMA:
            _validate_bundle_manifest(root, bundle_manifest)
    else:
        _validate_bundle_file_set(root, {value for value in ANN_OUTPUT_FILENAMES.values()})
    tf = _load_tensorflow()
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # Keras exposes format-specific exceptions.
        raise V2ANNError("Could not reload the ANN Keras model.") from exc
    return V2ANNPredictor(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        feature_names=feature_names,
        target_variant=str(manifest.get("target_variant") or ""),
    )


def _fit_selection_variant(
    split: Any,
    *,
    variant: str,
    seed: int,
    max_epochs: int,
    batch_size: int,
    patience: int,
) -> dict[str, Any]:
    features = [column for column in split.train.columns if column not in {DATE_COLUMN, TARGET_COLUMN}]
    x_train = split.train[features].apply(pd.to_numeric, errors="coerce")
    x_validation = split.validation[features].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(split.train[TARGET_COLUMN], errors="coerce").to_numpy(float)
    y_validation = pd.to_numeric(split.validation[TARGET_COLUMN], errors="coerce").to_numpy(float)
    scaler_x = MinMaxScaler().fit(x_train)
    scaler_y = MinMaxScaler().fit(
        (y_train if variant == "original" else np.log1p(y_train)).reshape(-1, 1)
    )
    x_train_scaled = scaler_x.transform(x_train)
    x_validation_scaled = scaler_x.transform(x_validation)
    y_train_scaled = scaler_y.transform(
        (y_train if variant == "original" else np.log1p(y_train)).reshape(-1, 1)
    )
    y_validation_scaled = scaler_y.transform(
        (y_validation if variant == "original" else np.log1p(y_validation)).reshape(-1, 1)
    )
    model = _build_model(len(features), seed=seed)
    tf = _load_tensorflow()
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        x_train_scaled,
        y_train_scaled,
        validation_data=(x_validation_scaled, y_validation_scaled),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[callback],
        verbose=0,
    )
    raw = np.asarray(model.predict(x_validation_scaled, verbose=0), dtype=float).reshape(-1, 1)
    inverse = scaler_y.inverse_transform(raw).reshape(-1)
    predictions = np.expm1(inverse) if variant == "log1p" else inverse
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise V2ANNError(f"ANN {variant} validation predictions are invalid.")
    val_history = [float(value) for value in history.history.get("val_mae", [])]
    best_epoch = int(np.argmin(val_history) + 1) if val_history else max_epochs
    return {
        "metrics": _metrics(split.validation[TARGET_COLUMN], predictions),
        "best_epoch": best_epoch,
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }


def _date_split(frame: pd.DataFrame, config: ANNTrainingConfig) -> Any:
    dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    bounds = [
        pd.Timestamp(config.train_start),
        pd.Timestamp(config.train_end),
        pd.Timestamp(config.validation_start),
        pd.Timestamp(config.validation_end),
        pd.Timestamp(config.test_start),
        pd.Timestamp(config.test_end),
    ]
    if not (bounds[0] <= bounds[1] < bounds[2] <= bounds[3] < bounds[4] <= bounds[5]):
        raise V2ANNError("ANN split dates must be ordered and disjoint.")
    masks = tuple(dates.between(bounds[index], bounds[index + 1]) for index in (0, 2, 4))
    assigned = sum(mask.astype(int) for mask in masks)
    if not assigned.eq(1).all():
        raise V2ANNError("Every ANN input row must belong to exactly one split.")
    parts = tuple(frame.loc[mask].copy().reset_index(drop=True) for mask in masks)
    if any(part.empty for part in parts):
        raise V2ANNError("ANN train, validation, and test splits must be non-empty.")
    return type("ANNDateSplit", (), {"train": parts[0], "validation": parts[1], "test": parts[2]})()


def _load_scaler_bundle(
    scaler_dir: Path, input_path: Path, feature_columns: Any
) -> dict[str, Any]:
    manifest_path = scaler_dir / SCALER_FILENAMES["manifest"]
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "wind_forecast.v2_scaler_manifest.v1":
        raise V2ANNError("Unsupported v2 scaler manifest schema.")
    if manifest.get("input_sha256") != sha256_file(input_path):
        raise V2ANNError("Versioned scalers were fit from a different dataset.")
    feature_names = [str(value) for value in manifest.get("feature_names") or ()]
    if feature_names != [str(value) for value in feature_columns]:
        raise V2ANNError("Versioned scaler feature order differs from the dataset.")
    if manifest.get("feature_count") != len(feature_names):
        raise V2ANNError("Versioned scaler feature count is inconsistent.")
    transformations = manifest.get("target_transformations") or {}
    if transformations.get("original") != "identity" or transformations.get("log") != "log1p":
        raise V2ANNError("Versioned scaler target transformations are incompatible.")
    paths: dict[str, Path] = {}
    scalers: dict[str, Any] = {}
    for name in ("x_original", "x_log", "y_original", "y_log"):
        path = scaler_dir / SCALER_FILENAMES[name]
        paths[name] = path
        expected = manifest.get("scalers", {}).get(name, {})
        if not path.is_file() or expected.get("sha256") != sha256_file(path):
            raise V2ANNError(f"Versioned scaler checksum is invalid: {name}.")
        scaler = joblib.load(path)
        if not isinstance(scaler, MinMaxScaler):
            raise V2ANNError(f"Versioned scaler is not MinMaxScaler: {name}.")
        scalers[name] = scaler
    for name in ("x_original", "x_log"):
        if getattr(scalers[name], "n_features_in_", None) != len(feature_names):
            raise V2ANNError("Versioned X scaler feature count is invalid.")
    for name in ("y_original", "y_log"):
        if getattr(scalers[name], "n_features_in_", None) != 1:
            raise V2ANNError("Versioned y scaler feature count is invalid.")
    return {**scalers, "paths": paths, "manifest_path": manifest_path, "manifest": manifest}


def _build_model(feature_count: int, *, seed: int) -> Any:
    tf = _load_tensorflow()
    _set_determinism(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(feature_count,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def _load_tensorflow() -> Any:
    # The governed recipe is CPU-only.  Keep this opt-in and lazy so existing
    # v1 imports are unaffected.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment-specific.
        raise V2ANNError("TensorFlow is required for the ANN v2 path.") from exc
    try:
        tf.config.set_visible_devices([], "GPU")
    except (AttributeError, RuntimeError):
        pass
    return tf


def _set_determinism(seed: int) -> None:
    tf = _load_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _metrics(actual: Any, predicted: Any) -> dict[str, float]:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    safe = y_true.copy()
    safe[safe == 0] = 1e-6
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE (%)": float(mean_absolute_percentage_error(safe, y_pred) * 100),
        "bias": float(np.mean(y_pred - y_true)),
    }


def _prediction_frame(part: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE_COLUMN: pd.to_datetime(part[DATE_COLUMN]).dt.strftime("%Y-%m-%d"),
            "Actual_Wind_Production": part[TARGET_COLUMN].to_numpy(dtype=float),
            "Predicted_Wind_Production": predicted,
        }
    )


def _split_sha256(split: Any) -> str:
    rows = []
    for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        rows.extend({"date": _date_text(value), "split": name} for value in part[DATE_COLUMN])
    return sha256(_canonical(rows)).hexdigest()


def _feature_schema_sha256(feature_names: tuple[str, ...]) -> str:
    return sha256(_canonical(list(feature_names))).hexdigest()


def _environment_manifest() -> dict[str, Any]:
    packages = {}
    for package in ("keras", "mlflow", "numpy", "pandas", "scikit-learn", "tensorflow"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "schema_version": "wind_forecast.v2_ann_environment.v1",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": sys.platform,
        "packages": packages,
        "tensorflow_deterministic_ops": True,
    }


def _validate_bundle_file_set(root: Path, expected_names: set[str]) -> None:
    actual = {path.name for path in root.iterdir() if path.is_file()}
    missing = expected_names.difference(actual)
    if missing:
        raise V2ANNError(f"ANN bundle is missing files: {sorted(missing)}")
    unexpected = actual.difference(expected_names)
    if unexpected:
        raise V2ANNError(f"ANN bundle contains undeclared files: {sorted(unexpected)}")


def _validate_bundle_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise V2ANNError("ANN bundle manifest has no file checksums.")
    expected = {str(name) for name in files}
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected | {ANN_OUTPUT_FILENAMES["bundle_manifest"]}:
        raise V2ANNError("ANN bundle file set differs from its manifest.")
    for name, entry in files.items():
        path = root / str(name)
        if not isinstance(entry, Mapping) or entry.get("sha256") != sha256_file(path):
            raise V2ANNError(f"ANN bundle checksum is invalid: {name}.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V2ANNError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise V2ANNError(f"JSON artifact must contain an object: {path}")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_ready(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _git_state() -> dict[str, Any]:
    return dict(git_state())


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return str(resolved)
