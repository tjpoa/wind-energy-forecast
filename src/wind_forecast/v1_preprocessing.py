"""Reproducible raw-to-features reconstruction for the supported v1 model."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import os
import sys
import tempfile

import numpy as np
import pandas as pd

from .features import apply_feature_engineering, handle_final_nans
from .manifest_validation import ManifestValidationError, validate_v1_source_contract
from .manifests import sha256_file
from .paths import manifests_dir, processed_data_dir, project_root, raw_data_dir
from .schemas import (
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    DATE_COLUMN,
    DATE_TIME_COLUMN,
    RAW_DAY_COLUMN,
    RAW_MONTH_COLUMN,
    RAW_PRODUCTION_FILENAME,
    RAW_TEMPERATURE_FILENAME,
    RAW_WIND_DIRECTION_FILENAME,
    RAW_WIND_SPEED_FILENAME,
    RAW_YEAR_COLUMN,
    TARGET_COLUMN,
    rename_legacy_columns_to_english,
)
from .v1_contracts import V1ContractError, load_processed_contract


# Keep these paths and station ordering explicit: they are part of the v1 input contract.
DEFAULT_RAW_FILES = {
    "production": RAW_PRODUCTION_FILENAME,
    "wind_speed": RAW_WIND_SPEED_FILENAME,
    "temperature": RAW_TEMPERATURE_FILENAME,
    "wind_direction": RAW_WIND_DIRECTION_FILENAME,
}
DEFAULT_OUTPUT = processed_data_dir() / "agg_data_ml.csv"


class V1PreprocessingError(ValueError):
    """Raised when a v1 source snapshot cannot be reconstructed safely."""


def build_v1_dataset(
    *,
    raw_files: Mapping[str, str | Path] | None = None,
    output_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
    processed_contract_path: str | Path | None = None,
    repository_root: str | Path | None = None,
    mode: str = "integrity",
    overwrite: bool = False,
) -> Path:
    """Build the immutable 58-column v1 table from one authorized snapshot.

    Source validation is deliberately the first operation that can inspect an
    input.  Output is written through a temporary file and replaced only after
    the processed contract has validated the candidate bytes.
    """
    root = Path(repository_root or project_root()).resolve()
    files = _resolve_raw_files(raw_files, root)
    source_contract_path = _resolve_repo_path(
        source_manifest_path or root / "data" / "manifests" / "v1_source_contract.json",
        root,
        "source_manifest_path",
    )
    try:
        validate_v1_source_contract(
            mode=mode, required_paths=list(files.values()),
            manifest_path=source_contract_path, repository_root=root
        )
    except ManifestValidationError as exc:
        raise V1PreprocessingError(f"v1 source contract validation failed: {exc}") from exc

    output = _resolve_repo_path(
        output_path or root / "data" / "processed" / "agg_data_ml.csv",
        root,
        "output_path",
    )
    if output.exists() and not overwrite:
        raise V1PreprocessingError(
            f"Refusing to overwrite existing processed output: {output}. Use --overwrite."
        )
    contract = load_processed_contract(
        processed_contract_path
        or root / "data" / "manifests" / "v1_processed_contract.json",
        repository_root=root,
        verify_dataset=False,
    )
    if contract["source_contract_path"] != source_contract_path.relative_to(root).as_posix():
        raise V1PreprocessingError(
            "Processed contract points to a different source contract than the build."
        )

    try:
        production = _load_production(files["production"])
        weather = {
            "wind_speed": _load_weather(
                files["wind_speed"], strategy="interpolate_median"
            ),
            "temperature": _load_weather(files["temperature"], strategy="mean"),
            "wind_direction": _load_weather(
                files["wind_direction"], strategy="circular_median"
            ),
        }
        candidate = _assemble_features(production, weather)
    except (OSError, KeyError, TypeError, ValueError, FloatingPointError) as exc:
        raise V1PreprocessingError(f"Could not reconstruct v1 dataset: {exc}") from exc

    _validate_candidate(candidate, contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            candidate.to_csv(temporary, index=False)
        if sha256_file(temporary_path) != contract["dataset_sha256"]:
            raise V1PreprocessingError(
                "Reconstructed bytes do not match the immutable processed contract."
            )
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output


def _resolve_raw_files(
    raw_files: Mapping[str, str | Path] | None, root: Path
) -> dict[str, Path]:
    supplied = raw_files or {
        key: root / "data" / "raw" / filename
        for key, filename in DEFAULT_RAW_FILES.items()
    }
    if set(supplied) != set(DEFAULT_RAW_FILES):
        raise V1PreprocessingError(
            "raw_files must provide exactly production, wind_speed, temperature, "
            "and wind_direction."
        )
    return {
        key: _resolve_repo_path(value, root, f"raw_files.{key}")
        for key, value in supplied.items()
    }


def _resolve_repo_path(value: str | Path, root: Path, field: str) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise V1PreprocessingError(f"{field} escapes repository root: {value!r}.") from exc
    return resolved


def _load_production(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, na_values=-990, sep=";", skiprows=2)
    frame.columns = frame.columns.str.strip()
    frame = rename_legacy_columns_to_english(frame)
    required = {DATE_TIME_COLUMN, TARGET_COLUMN}
    if not required.issubset(frame.columns):
        raise V1PreprocessingError(
            f"Production file is missing columns: {sorted(required - set(frame.columns))}."
        )
    frame = frame[[DATE_TIME_COLUMN, TARGET_COLUMN]].copy()
    frame[DATE_TIME_COLUMN] = pd.to_datetime(frame[DATE_TIME_COLUMN], errors="raise")
    frame = frame.set_index(DATE_TIME_COLUMN)
    daily = frame.resample("D").sum().reset_index()
    daily.rename(
        columns={DATE_TIME_COLUMN: DATE_COLUMN, TARGET_COLUMN: TARGET_COLUMN},
        inplace=True,
    )
    return daily


def _load_weather(path: Path, *, strategy: str) -> pd.DataFrame:
    frame = pd.read_csv(path, na_values=-990, sep=";", decimal=".")
    frame = frame.rename(
        columns={RAW_YEAR_COLUMN: "year", RAW_MONTH_COLUMN: "month", RAW_DAY_COLUMN: "day"}
    )
    required = {"year", "month", "day"}
    if not required.issubset(frame.columns):
        raise V1PreprocessingError(
            f"Weather file {path.name} is missing date columns: {sorted(required - set(frame.columns))}."
        )
    frame[DATE_COLUMN] = pd.to_datetime(frame[["year", "month", "day"]], errors="raise")
    station_columns = [column for column in frame.columns if str(column).isdigit()]
    if not station_columns:
        raise V1PreprocessingError(f"Weather file {path.name} has no station columns.")
    for column in station_columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any():
            if strategy == "interpolate_median" and values.isna().mean() > 0.30:
                values = values.interpolate(method="linear", limit_direction="both")
            elif strategy == "circular_median":
                values = values.fillna(_circular_median(values.dropna().to_numpy()))
            else:
                values = values.fillna(values.mean() if strategy == "mean" else values.median())
            if values.isna().any():
                raise V1PreprocessingError(
                    f"Weather file {path.name} station {column} cannot be imputed."
                )
        frame[column] = values
    return frame[[DATE_COLUMN, *station_columns]]


def _circular_median(values: np.ndarray) -> float:
    if values.size == 0:
        raise V1PreprocessingError("A wind-direction station has no usable values.")
    radians = np.deg2rad(values)
    result = float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) % 360)
    return 0.0 if np.isclose(result, 360.0) else result


def _assemble_features(
    production: pd.DataFrame, weather: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    station_sets = {
        name: [column for column in frame.columns if str(column).isdigit()]
        for name, frame in weather.items()
    }
    expected = station_sets["wind_speed"]
    if any(stations != expected for stations in station_sets.values()):
        raise V1PreprocessingError(
            "Weather station sets differ across speed, temperature, and direction files."
        )

    speed = weather["wind_speed"]
    temperature = weather["temperature"]
    direction = weather["wind_direction"]
    speed_agg = pd.DataFrame(
        {DATE_COLUMN: speed[DATE_COLUMN], AVG_WIND_SPEED_COLUMN: speed[expected].mean(axis=1)}
    )
    temperature_agg = pd.DataFrame(
        {
            DATE_COLUMN: temperature[DATE_COLUMN],
            AVG_TEMPERATURE_COLUMN: temperature[expected].mean(axis=1),
        }
    )
    direction_radians = np.deg2rad(direction[expected])
    u_component = -speed[expected] * np.sin(direction_radians)
    v_component = -speed[expected] * np.cos(direction_radians)
    direction_agg = pd.DataFrame(
        {
            DATE_COLUMN: direction[DATE_COLUMN],
            AVG_WIND_DIRECTION_COLUMN: (
                np.rad2deg(np.arctan2(u_component.mean(axis=1), v_component.mean(axis=1)))
                + 180
            )
            % 360,
        }
    )
    merged = production.merge(speed_agg, on=DATE_COLUMN, how="inner")
    merged = merged.merge(temperature_agg, on=DATE_COLUMN, how="inner")
    merged = merged.merge(direction_agg, on=DATE_COLUMN, how="inner")
    merged = merged.sort_values(DATE_COLUMN).reset_index(drop=True)
    return handle_final_nans(apply_feature_engineering(merged))


def _validate_candidate(candidate: pd.DataFrame, contract: Mapping[str, object]) -> None:
    expected_columns = contract["columns"]
    if list(candidate.columns) != expected_columns:
        raise V1PreprocessingError(
            "Reconstructed feature order differs from v1_processed_contract."
        )
    if len(candidate) != contract["row_count"]:
        raise V1PreprocessingError("Reconstructed row count differs from v1_processed_contract.")
    if candidate.isna().any().any():
        raise V1PreprocessingError("Reconstructed dataset contains missing values.")
    coverage = contract["coverage"]
    observed_start = pd.Timestamp(candidate[DATE_COLUMN].iloc[0]).date().isoformat()
    observed_end = pd.Timestamp(candidate[DATE_COLUMN].iloc[-1]).date().isoformat()
    if (observed_start, observed_end) != (coverage["start"], coverage["end"]):
        raise V1PreprocessingError("Reconstructed coverage differs from v1_processed_contract.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the authorized v1 feature table.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument(
        "--processed-contract", type=Path,
        default=manifests_dir() / "v1_processed_contract.json",
    )
    parser.add_argument("--mode", choices=("integrity", "release"), default="integrity")
    parser.add_argument("--overwrite", action="store_true")
    for key, filename in DEFAULT_RAW_FILES.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path, default=raw_data_dir() / filename)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        build_v1_dataset(
            raw_files={
                "production": args.production,
                "wind_speed": args.wind_speed,
                "temperature": args.temperature,
                "wind_direction": args.wind_direction,
            },
            output_path=args.output,
            source_manifest_path=args.source_manifest,
            processed_contract_path=args.processed_contract,
            mode=args.mode,
            overwrite=args.overwrite,
        )
    except (V1PreprocessingError, V1ContractError) as exc:
        print(f"v1 preprocessing failed: {exc}", file=sys.stderr)
        return 1
    print(f"Built v1 dataset: {args.output}")
    return 0


__all__ = ["V1PreprocessingError", "build_v1_dataset", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
