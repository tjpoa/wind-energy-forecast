import io
import json
import zipfile
from pathlib import Path

import pytest

from wind_forecast.artifacts import (
    bundle_temp_extract,
    build_reproduction_bundle,
    extract_bundle,
    fetch_release_bundle,
    materialize_training_data,
    verify_bundle,
)
import wind_forecast.artifacts as artifact_module
from scripts import verify_reproduction
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN
from wind_forecast.training import run_baseline_training
import pandas as pd


def _build(tmp_path: Path, output_name: str):
    source = tmp_path / "input.csv"
    source.write_text("Date,value\n2026-01-01,1\n", encoding="utf-8")
    return build_reproduction_bundle(
        release="artifacts-v1.0.0",
        model_name="wind-energy-forecast-original",
        model_version="1",
        run_id="run-1",
        git_sha="a" * 40,
        files={"data/agg_data_ml.csv": source},
        output_dir=tmp_path / output_name,
        redistribution={"approved": False, "status": "blocked"},
    )


def test_bundle_is_deterministic_and_verifiable(tmp_path: Path):
    first = _build(tmp_path, "one")
    second = _build(tmp_path, "two")
    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    manifest = verify_bundle(first.archive_path, first.checksum_path)
    assert manifest["release"] == "artifacts-v1.0.0"


def test_verify_rejects_wrong_checksum(tmp_path: Path):
    result = _build(tmp_path, "bundle")
    result.checksum_path.write_text("0" * 64 + "  file.zip\n", encoding="ascii")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_bundle(result.archive_path, result.checksum_path)


def test_extract_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(archive, tmp_path / "extracted")


def test_materialize_refuses_overwrite(tmp_path: Path):
    extracted = tmp_path / "extracted"
    source = extracted / "data" / "agg_data_ml.csv"
    source.parent.mkdir(parents=True)
    source.write_text("x\n1\n", encoding="utf-8")
    destination = tmp_path / "data" / "processed" / "agg_data_ml.csv"
    destination.parent.mkdir(parents=True)
    destination.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_training_data(extracted, destination)


def test_release_name_rejects_path_components(tmp_path: Path):
    source = tmp_path / "input.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Release tags"):
        build_reproduction_bundle(
            release="artifacts-v1.0.0/escape",
            model_name="model",
            model_version="1",
            run_id="run",
            git_sha="a" * 40,
            files={"data/agg_data_ml.csv": source},
            output_dir=tmp_path / "out",
            redistribution={"approved": False},
        )


def test_verify_rejects_duplicate_members(tmp_path: Path):
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", "{}")
        with pytest.warns(UserWarning, match="Duplicate name: 'manifest.json'"):
            output.writestr("manifest.json", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        verify_bundle(archive)


def test_verify_rejects_safe_but_undeclared_member(tmp_path: Path):
    result = _build(tmp_path, "bundle")
    with zipfile.ZipFile(result.archive_path, "a") as output:
        output.writestr("extra.txt", "undeclared")
    with pytest.raises(ValueError, match="undeclared"):
        verify_bundle(result.archive_path)


def test_fetch_second_download_failure_leaves_no_destination(tmp_path: Path):
    built = _build(tmp_path, "source")
    payloads = [built.archive_path.read_bytes()]

    def opener(url):
        if payloads:
            return io.BytesIO(payloads.pop())
        raise OSError("second download failed")

    destination = tmp_path / "download"
    with pytest.raises(OSError, match="second download"):
        fetch_release_bundle(
            release="artifacts-v1.0.0",
            repository="owner/repo",
            destination=destination,
            opener=opener,
        )
    assert not destination.exists()


def test_materialize_copy_failure_is_atomic(tmp_path: Path, monkeypatch):
    extracted = tmp_path / "extracted"
    source = extracted / "data" / "agg_data_ml.csv"
    source.parent.mkdir(parents=True)
    source.write_text("x\n1\n", encoding="utf-8")
    destination = tmp_path / "data" / "processed" / "agg_data_ml.csv"

    def failing_copy(source_path, destination_path):
        Path(destination_path).write_text("partial", encoding="utf-8")
        raise OSError("copy failed")

    monkeypatch.setattr(artifact_module.shutil, "copy2", failing_copy)
    with pytest.raises(OSError, match="copy failed"):
        materialize_training_data(extracted, destination)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp"))


def test_late_unsafe_member_does_not_create_destination(tmp_path: Path):
    archive = tmp_path / "unsafe-late.zip"
    manifest = {
        "schema_version": "wind_forecast.reproduction_bundle.v1",
        "files": [],
    }
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", json.dumps(manifest))
        output.writestr("safe.txt", "safe")
        output.writestr("../escape.txt", "bad")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(archive, destination)
    assert not destination.exists()


def test_bundle_temp_extract_uses_nonexistent_child(tmp_path: Path):
    result = _build(tmp_path, "bundle")
    with bundle_temp_extract(result.archive_path) as extracted:
        assert (extracted / "manifest.json").is_file()
        assert (extracted / "data" / "agg_data_ml.csv").is_file()


def test_verify_reproduction_retrain_round_trip(tmp_path: Path, monkeypatch):
    rows = 20
    frame = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2026-01-01", periods=rows).strftime("%Y-%m-%d"),
            TARGET_COLUMN: range(100, 100 + rows),
            "Feature_A": range(rows),
            "Feature_B": [value % 3 for value in range(rows)],
        }
    )
    input_path = tmp_path / "features.csv"
    frame.to_csv(input_path, index=False)
    reference = run_baseline_training(
        input_path=input_path,
        output_dir=tmp_path / "reference",
        n_estimators=5,
        seed=42,
        test_fraction=0.25,
    )
    result = build_reproduction_bundle(
        release="artifacts-v1.0.0",
        model_name="wind-energy-forecast-original",
        model_version="1",
        run_id="run-1",
        git_sha="a" * 40,
        files={
            "data/agg_data_ml.csv": input_path,
            "baseline/metrics.json": reference.metrics_path,
            "baseline/predictions.csv": reference.predictions_path,
            "manifests/model_manifest.json": reference.model_manifest_path,
        },
        output_dir=tmp_path / "bundle",
        redistribution={"approved": False, "status": "blocked"},
    )
    monkeypatch.setattr(verify_reproduction, "project_root", lambda: tmp_path)
    verify_reproduction._retrain_and_compare(
        result.archive_path, "artifacts-v1.0.0", overwrite=False
    )
