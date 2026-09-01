from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from wind_forecast.v1_contracts import V1ContractError, load_processed_contract, load_serving_contract
from wind_forecast.inference import model_and_scaler_paths
from wind_forecast.paths import models_dir


def test_v1_processed_and_serving_contracts_are_consistent() -> None:
    processed = load_processed_contract(verify_dataset=False)
    dataset_path = Path(processed["dataset_path"])
    if not dataset_path.is_file():
        pytest.skip("authorized local processed snapshot is not present")
    processed = load_processed_contract(verify_dataset=True)
    serving = load_serving_contract(verify_files=True)
    assert serving["dataset_sha256"] == processed["dataset_sha256"]
    assert serving["feature_contract"]["columns_source"] == "processed_contract.columns[2:]"
    model_path, scaler_x, scaler_y = model_and_scaler_paths("ANN_Tuned", "original", models_dir())
    assert model_path == Path("models/best_model_original_target_ANN_Tuned.keras").resolve()
    assert scaler_x == Path("models/scaler_X_original_ann.joblib").resolve()
    assert scaler_y == Path("models/scaler_y_original_ann.joblib").resolve()


def test_serving_contract_rejects_tampered_artifact(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    shutil.copytree("models", tmp_path / "models", dirs_exist_ok=True)
    shutil.copy("data/manifests/v1_source_contract.json", tmp_path / "data/manifests")
    shutil.copy("data/manifests/v1_processed_contract.json", tmp_path / "data/manifests")
    shutil.copy("data/processed/agg_data_ml.csv", tmp_path / "data/processed")
    target = tmp_path / "models" / "v1_serving_contract.json"
    payload = target.read_text(encoding="utf-8")
    payload = payload.replace(
        "89fab64f576c517d267b83e426653ff3eb85e1445a42ad95d4c200ed5c89eacf",
        "0" * 64,
    )
    target.write_text(payload, encoding="utf-8")
    with pytest.raises(V1ContractError, match="hash mismatch"):
        load_serving_contract(target, repository_root=tmp_path, verify_files=True)
