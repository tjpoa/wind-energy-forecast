from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from wind_forecast.manifests import sha256_file
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN
from wind_forecast.v2_ann import (
    ANN_OUTPUT_FILENAMES,
    ANN_MODEL_MANIFEST_SCHEMA,
    ANNTrainingConfig,
    V2ANNError,
    fit_v2_ann_candidate,
    load_v2_ann_bundle,
)
from wind_forecast.v2_scaling import SCALER_FILENAMES


def _frame(rows: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    target = np.linspace(100.0, 200.0, rows) + np.sin(np.arange(rows))
    return pd.DataFrame(
        {
            DATE_COLUMN: dates,
            TARGET_COLUMN: target,
            "Feature_A": np.linspace(1.0, 4.0, rows),
            "Wind_Production_Lag1": np.concatenate(([90.0], target[:-1])),
        }
    )


def _scaler_bundle(root: Path, input_path: Path, frame: pd.DataFrame) -> Path:
    scaler_dir = root / "scalers"
    scaler_dir.mkdir()
    features = frame[["Feature_A", "Wind_Production_Lag1"]]
    target = frame[TARGET_COLUMN].to_numpy(float)
    scalers = {
        "x_original": MinMaxScaler().fit(features.iloc[:22]),
        "x_log": MinMaxScaler().fit(features.iloc[:22]),
        "y_original": MinMaxScaler().fit(target[:22].reshape(-1, 1)),
        "y_log": MinMaxScaler().fit(np.log1p(target[:22]).reshape(-1, 1)),
    }
    entries = {}
    for name, filename in SCALER_FILENAMES.items():
        if name == "manifest":
            continue
        path = scaler_dir / filename
        joblib.dump(scalers[name], path)
        entries[name] = {
            "sha256": sha256_file(path),
            "type": "sklearn.preprocessing.MinMaxScaler",
            "n_features_in": int(scalers[name].n_features_in_),
        }
    feature_names = list(features.columns)
    manifest = {
        "schema_version": "wind_forecast.v2_scaler_manifest.v1",
        "input_sha256": sha256_file(input_path),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "target_transformations": {"original": "identity", "log": "log1p"},
        "scalers": entries,
    }
    (scaler_dir / SCALER_FILENAMES["manifest"]).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return scaler_dir


def test_ann_candidate_seals_scaled_bundle_and_reloads(tmp_path: Path) -> None:
    frame = _frame()
    input_path = tmp_path / "features.csv"
    frame.to_csv(input_path, index=False)
    scaler_dir = _scaler_bundle(tmp_path, input_path, frame)
    result = fit_v2_ann_candidate(
        ANNTrainingConfig(
            input_path=input_path,
            scaler_dir=scaler_dir,
            output_dir=tmp_path / "candidate",
            max_epochs=3,
            patience=2,
            batch_size=4,
            train_start="2024-01-01",
            train_end="2024-01-15",
            validation_start="2024-01-16",
            validation_end="2024-01-22",
            test_start="2024-01-23",
            test_end="2024-01-30",
        )
    )

    assert result.selected_variant in {"original", "log1p"}
    assert result.metrics["test"]["MAE"] >= 0
    manifest = json.loads(
        result.paths["model_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == ANN_MODEL_MANIFEST_SCHEMA
    assert manifest["artifact_type"] == "keras_scaled_v2"
    assert manifest["scaler_required"] is True
    predictor = load_v2_ann_bundle(result.output_dir)
    reloaded = predictor.predict(frame[["Feature_A", "Wind_Production_Lag1"]])
    expected = pd.read_csv(result.paths["training_evidence"])["Expected_Prediction"]
    np.testing.assert_allclose(
        reloaded[: len(expected)],
        expected.to_numpy(float),
        rtol=1e-7,
        atol=1e-5,
    )
    assert set(path.name for path in result.output_dir.iterdir()) == set(
        ANN_OUTPUT_FILENAMES.values()
    )


def test_ann_candidate_rejects_scaler_dataset_mismatch(tmp_path: Path) -> None:
    frame = _frame()
    input_path = tmp_path / "features.csv"
    frame.to_csv(input_path, index=False)
    scaler_dir = _scaler_bundle(tmp_path, input_path, frame)
    changed = frame.copy()
    changed.loc[0, "Feature_A"] += 1
    changed.to_csv(input_path, index=False)

    with pytest.raises(V2ANNError, match="different dataset"):
        fit_v2_ann_candidate(
            ANNTrainingConfig(
                input_path=input_path,
                scaler_dir=scaler_dir,
                output_dir=tmp_path / "candidate",
                max_epochs=2,
                patience=1,
                train_start="2024-01-01",
                train_end="2024-01-15",
                validation_start="2024-01-16",
                validation_end="2024-01-22",
                test_start="2024-01-23",
                test_end="2024-01-30",
            )
        )


def test_ann_candidate_rejects_existing_output(tmp_path: Path) -> None:
    frame = _frame()
    input_path = tmp_path / "features.csv"
    frame.to_csv(input_path, index=False)
    scaler_dir = _scaler_bundle(tmp_path, input_path, frame)
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        fit_v2_ann_candidate(
            ANNTrainingConfig(
                input_path=input_path,
                scaler_dir=scaler_dir,
                output_dir=output_dir,
            )
        )
