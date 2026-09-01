from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wind_forecast.manifests import DatasetManifest, sha256_file
from wind_forecast.schemas import (
    RAW_WIND_DIRECTION_FILENAME,
    RAW_WIND_SPEED_FILENAME,
    RAW_PRODUCTION_FILENAME,
    RAW_TEMPERATURE_FILENAME,
)
from wind_forecast.v1_contracts import canonical_sha256
from wind_forecast.v1_preprocessing import (
    V1PreprocessingError,
    _assemble_features,
    _load_production,
    _load_weather,
    build_v1_dataset,
)


def _write_snapshot(root: Path, *, mismatch: bool = False) -> tuple[dict[str, Path], Path, Path]:
    raw = root / "data" / "raw"
    manifests = root / "data" / "manifests"
    raw.mkdir(parents=True)
    manifests.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    production = raw / RAW_PRODUCTION_FILENAME
    production.write_text(
        "ignored\nignored\nData e Hora;Eólica\n"
        + "\n".join(
            f"{date.strftime('%Y-%m-%d')} 00:00;{index + 1}" for index, date in enumerate(dates)
        )
        + "\n",
        encoding="utf-8",
    )

    def weather(path: Path, values: list[tuple[float, float]]) -> None:
        rows = [f"2024;1;{day};{a};{b}" for day, (a, b) in enumerate(values, 1)]
        path.write_text("ANO;MES;DIA;100;200\n" + "\n".join(rows) + "\n", encoding="utf-8")

    speed = raw / RAW_WIND_SPEED_FILENAME
    temperature = raw / RAW_TEMPERATURE_FILENAME
    direction = raw / RAW_WIND_DIRECTION_FILENAME
    weather(speed, [(4, 5), (-990, 6), (6, 7)])
    weather(temperature, [(10, 11), (12, 13), (14, 15)])
    weather(direction, [(359, 1), (-990, 1), (1, 359)])
    if mismatch:
        direction.write_text(
            direction.read_text(encoding="utf-8")
            .replace("ANO;MES;DIA;100;200", "ANO;MES;DIA;100;200;300")
            .replace(";1\n", ";1;1\n"),
            encoding="utf-8",
        )

    files = {
        "production": production,
        "wind_speed": speed,
        "temperature": temperature,
        "wind_direction": direction,
    }
    manifest_path = manifests / "v1_source_contract.json"
    manifest = DatasetManifest(
        dataset_version="v1",
        dataset_role="legacy_v1_source_contract",
        raw_file_paths=tuple(path.relative_to(root).as_posix() for path in files.values()),
        sha256_checksums={
            path.relative_to(root).as_posix(): sha256_file(path) for path in files.values()
        },
        status="provenance_incomplete",
    )
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    source_hash = sha256_file(manifest_path)
    output = root / "data" / "processed" / "agg_data_ml.csv"
    output.parent.mkdir(parents=True)
    contract_path = manifests / "v1_processed_contract.json"
    if mismatch:
        candidate = pd.DataFrame({"Date": ["2024-01-01"], "Wind_Production": [0.0]})
        dataset_hash = "0" * 64
        columns = candidate.columns.tolist()
        row_count = 1
        coverage = {"start": "2024-01-01", "end": "2024-01-01"}
    else:
        candidate = _assemble_features(
            _load_production(production),
            {
                "wind_speed": _load_weather(speed, strategy="interpolate_median"),
                "temperature": _load_weather(temperature, strategy="mean"),
                "wind_direction": _load_weather(direction, strategy="circular_median"),
            },
        )
        candidate.to_csv(output, index=False)
        dataset_hash = sha256_file(output)
        columns = candidate.columns.tolist()
        row_count = len(candidate)
        coverage = {
            "start": str(candidate.Date.iloc[0].date()),
            "end": str(candidate.Date.iloc[-1].date()),
        }
    contract = {
        "schema_version": "wind_forecast.v1_processed_contract.v1",
        "transformation_version": "v1_preprocessing.v1",
        "source_contract_path": "data/manifests/v1_source_contract.json",
        "source_contract_sha256": source_hash,
        "dataset_path": "data/processed/agg_data_ml.csv",
        "dataset_sha256": dataset_hash,
        "row_count": row_count,
        "column_count": len(columns),
        "coverage": coverage,
        "feature_schema_sha256": canonical_sha256(columns[2:]),
        "columns": columns,
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output.unlink(missing_ok=True)
    return files, manifest_path, contract_path


def test_v1_preprocessing_rebuilds_deterministically_and_refuses_overwrite(tmp_path: Path) -> None:
    files, manifest, contract = _write_snapshot(tmp_path)
    output = build_v1_dataset(
        raw_files=files,
        output_path=tmp_path / "data" / "processed" / "agg_data_ml.csv",
        source_manifest_path=manifest,
        processed_contract_path=contract,
        repository_root=tmp_path,
    )
    first = output.read_bytes()
    with pytest.raises(V1PreprocessingError, match="overwrite"):
        build_v1_dataset(
            raw_files=files,
            output_path=output,
            source_manifest_path=manifest,
            processed_contract_path=contract,
            repository_root=tmp_path,
        )
    assert output.read_bytes() == first


def test_v1_preprocessing_imputes_sentinels_and_uses_vector_direction(tmp_path: Path) -> None:
    files, manifest, contract = _write_snapshot(tmp_path)
    speed = _load_weather(files["wind_speed"], strategy="interpolate_median")
    direction = _load_weather(files["wind_direction"], strategy="circular_median")
    assert speed[["100", "200"]].isna().sum().sum() == 0
    assert np.isclose(direction.loc[1, "100"], 0.0)
    assert direction.loc[1, "100"] < 360
    assert manifest.is_file() and contract.is_file()


def test_v1_preprocessing_fails_on_station_mismatch_before_output(tmp_path: Path) -> None:
    files, manifest, contract = _write_snapshot(tmp_path, mismatch=True)
    with pytest.raises(V1PreprocessingError, match="hash|station"):
        build_v1_dataset(
            raw_files=files,
            output_path=tmp_path / "data" / "processed" / "agg_data_ml.csv",
            source_manifest_path=manifest,
            processed_contract_path=contract,
            repository_root=tmp_path,
        )
    assert not (tmp_path / "data" / "processed" / "agg_data_ml.csv").exists()
