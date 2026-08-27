"""Versioned scaler fitting for the accepted v2 feature-ready dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .manifests import sha256_file
from .paths import project_root
from .schemas import DATE_COLUMN, TARGET_COLUMN
from .training import build_xy, load_training_table


DEFAULT_FIT_START = "2010-01-15"
DEFAULT_FIT_END = "2024-12-31"
DEFAULT_DATASET_VERSION = "v2"
DEFAULT_TRANSFORMATION_VERSION = "feature_ready_ren_era5_land_v2_2A.18"

SCALER_FILENAMES = {
    "x_original": "scaler_X_original_ann.joblib",
    "x_log": "scaler_X_log_ann.joblib",
    "y_original": "scaler_y_original_ann.joblib",
    "y_log": "scaler_y_log_ann.joblib",
    "manifest": "scaler_manifest.json",
}


@dataclass(frozen=True)
class V2ScalerResult:
    """Paths and lineage produced by one immutable v2 scaler fit."""

    output_dir: Path
    paths: Mapping[str, Path]
    input_path: Path
    input_sha256: str
    dataset_version: str
    transformation_version: str
    fit_scope: str
    fit_start: str
    fit_end: str
    fit_row_count: int
    total_row_count: int
    feature_names: tuple[str, ...]


def fit_v2_scalers(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    fit_start: str = DEFAULT_FIT_START,
    fit_end: str = DEFAULT_FIT_END,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    transformation_version: str = DEFAULT_TRANSFORMATION_VERSION,
) -> V2ScalerResult:
    """Fit separate X and y MinMax scalers without touching v1 artifacts.

    The default fit window covers the v2 train and validation periods and
    deliberately excludes the sealed test period.  The output directory must
    be a new child of ``models/v2``; an existing directory is rejected so that
    an accepted scaler fit cannot be silently replaced.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    _validate_output_dir(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"V2 feature table is missing: {input_path}")
    if output_dir.exists():
        raise FileExistsError(f"V2 scaler output directory already exists: {output_dir}")

    start = _parse_date(fit_start, "fit_start")
    end = _parse_date(fit_end, "fit_end")
    if start > end:
        raise ValueError("fit_start must be on or before fit_end.")
    sealed_test_start = _parse_date("2025-01-01", "sealed_test_start")
    if end >= sealed_test_start:
        raise ValueError("fit_end must exclude the sealed v2 test period.")
    fit_scope = (
        "train_plus_validation"
        if start == _parse_date(DEFAULT_FIT_START, "default_fit_start")
        and end == _parse_date(DEFAULT_FIT_END, "default_fit_end")
        else "explicit_date_window"
    )

    frame = load_training_table(input_path)
    if frame[DATE_COLUMN].duplicated().any():
        raise ValueError("V2 feature dates must be unique before scaler fitting.")
    features, target, dates = build_xy(frame)
    if features.empty or features.shape[1] == 0:
        raise ValueError("V2 feature table must contain at least one feature column.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("V2 features must be finite before scaler fitting.")

    target_values = target.to_numpy(dtype=float)
    if not np.isfinite(target_values).all() or (target_values < 0).any():
        raise ValueError("V2 target must be finite and non-negative for log1p scaling.")

    fit_mask = dates.between(start, end)
    if not fit_mask.any():
        raise ValueError("The requested scaler fit window contains no v2 rows.")
    fit_features = features.loc[fit_mask]
    fit_target = target.loc[fit_mask].to_numpy(dtype=float).reshape(-1, 1)

    scalers = {
        "x_original": MinMaxScaler().fit(fit_features),
        "x_log": MinMaxScaler().fit(fit_features),
        "y_original": MinMaxScaler().fit(fit_target),
        "y_log": MinMaxScaler().fit(np.log1p(fit_target)),
    }

    output_dir.mkdir(parents=True)
    paths = {name: output_dir / filename for name, filename in SCALER_FILENAMES.items()}
    for name in ("x_original", "x_log", "y_original", "y_log"):
        joblib.dump(scalers[name], paths[name])

    input_sha256 = sha256_file(input_path)
    feature_names = tuple(str(column) for column in features.columns)
    manifest = {
        "schema_version": "wind_forecast.v2_scaler_manifest.v1",
        "dataset_version": dataset_version,
        "transformation_version": transformation_version,
        "input_path": _display_path(input_path),
        "input_sha256": input_sha256,
        "fit_scope": fit_scope,
        "fit_start": start.strftime("%Y-%m-%d"),
        "fit_end": end.strftime("%Y-%m-%d"),
        "fit_row_count": int(fit_mask.sum()),
        "total_row_count": int(len(frame)),
        "target": TARGET_COLUMN,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "feature_schema_sha256": _json_sha256(list(feature_names)),
        "target_transformations": {
            "original": "identity",
            "log": "log1p",
        },
        "scalers": {
            name: {
                "path": _display_path(paths[name]),
                "sha256": sha256_file(paths[name]),
                "type": "sklearn.preprocessing.MinMaxScaler",
                "n_features_in": int(scalers[name].n_features_in_),
            }
            for name in ("x_original", "x_log", "y_original", "y_log")
        },
        "v1_artifacts_untouched": True,
    }
    _write_json(paths["manifest"], manifest)

    return V2ScalerResult(
        output_dir=output_dir,
        paths=paths,
        input_path=input_path,
        input_sha256=input_sha256,
        dataset_version=dataset_version,
        transformation_version=transformation_version,
        fit_scope=fit_scope,
        fit_start=start.strftime("%Y-%m-%d"),
        fit_end=end.strftime("%Y-%m-%d"),
        fit_row_count=int(fit_mask.sum()),
        total_row_count=len(frame),
        feature_names=feature_names,
    )


def _validate_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    root = project_root().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("V2 scaler outputs must be inside the project root.") from exc
    if len(relative.parts) < 3 or relative.parts[:2] != ("models", "v2"):
        raise ValueError("V2 scaler outputs must be written under models/v2/.")


def _parse_date(value: str, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"{name} must be a valid date.")
    return parsed.normalize()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
