import sys
import types
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from wind_forecast.registry import (
    PromotionReceipt,
    promote_candidate,
    register_candidate,
    rollback_promotion,
)
from wind_forecast.tracking import TrackingConfig


class _Info:
    status = "FINISHED"


class _Data:
    def __init__(self, *, dirty: str = "False"):
        self.params = {
            "dataset_version": "v1",
            "dataset_sha256": "a" * 64,
            "feature_schema_sha256": "b" * 64,
            "git_sha": "c" * 40,
            "git_dirty": dirty,
            "logged_model_uri": "runs:/run-1/model",
            "target_contract": "original",
        }
        self.metrics = {"R2": 0.5, "MAE": 1.0, "RMSE": 1.0, "MAPE_percent": 2.0}


class _Run:
    def __init__(self, *, dirty: str = "False"):
        self.info = _Info()
        self.data = _Data(dirty=dirty)


class _Version:
    def __init__(self, version="7", *, tags=None, run_id="run-1"):
        self.version = version
        self.tags = tags or {}
        self.run_id = run_id


class _Client:
    def __init__(self, run=None, *, candidate=None, champion=None):
        self.run = run or _Run()
        self.candidate = candidate
        self.champion = champion
        self.calls = []

    def get_run(self, run_id):
        assert run_id == "run-1"
        return self.run

    def set_model_version_tag(self, *args):
        self.calls.append(("tag", *args))

    def set_registered_model_alias(self, *args):
        self.calls.append(("set_alias", *args))

    def delete_registered_model_alias(self, *args):
        self.calls.append(("delete_alias", *args))

    def get_model_version_by_alias(self, model_name, alias):
        assert model_name == "wind-energy-forecast-original"
        value = self.candidate if alias == "candidate" else self.champion
        if value is None:
            raise LookupError(alias)
        return value


class _LoadedModel:
    def predict(self, frame):
        return frame["Feature_A"].to_numpy(dtype=float) + 1.0


def _fake_mlflow(tmp_path: Path):
    module = types.ModuleType("mlflow")
    module.calls = []
    module.set_tracking_uri = lambda uri: module.calls.append(("tracking", uri))
    module.set_experiment = lambda name: module.calls.append(("experiment", name))
    module.MlflowClient = lambda: None
    module.pyfunc = types.SimpleNamespace(load_model=lambda uri: _LoadedModel())
    module.register_model = lambda **kwargs: _Version("7", run_id="run-1")
    module.models = types.SimpleNamespace(
        get_model_info=lambda uri: types.SimpleNamespace(signature=object(), run_id="run-1")
    )

    validation = tmp_path / "validation_sample.csv"
    pd.DataFrame(
        {"Feature_A": [1.0, 2.0], "Expected_Prediction": [2.0, 3.0]}
    ).to_csv(validation, index=False)
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"model")
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_text(json.dumps({
        "schema_version": "wind_forecast.training_dataset.v1",
        "dataset_version": "v1",
        "sha256": "a" * 64,
        "feature_schema_sha256": "b" * 64,
        "target": "Wind_Production",
    }), encoding="utf-8")
    model_manifest = tmp_path / "model_manifest.json"
    model_manifest.write_text(json.dumps({
        "schema_version": "wind_forecast.model_manifest.v1",
        "dataset_version": "v1",
        "dataset_sha256": "a" * 64,
        "feature_schema_sha256": "b" * 64,
        "target_contract": "original",
        "model_sha256": sha256(b"model").hexdigest(),
    }), encoding="utf-8")
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({
        "schema_version": "wind_forecast.environment.v1",
        "python": "3.10",
        "packages": {"scikit-learn": "1.6.1"},
    }), encoding="utf-8")
    summary = tmp_path / "run_summary.json"
    summary.write_text(json.dumps({
        "dataset_version": "v1",
        "input_sha256": "a" * 64,
        "feature_schema_sha256": "b" * 64,
    }), encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({
        "R2": 0.5, "MAE": 1.0, "RMSE": 1.0, "MAPE (%)": 2.0
    }), encoding="utf-8")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("prediction\n2.0\n", encoding="utf-8")
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"png")
    paths = {
        "validation/validation_sample.csv": validation,
        "manifests/dataset_manifest.json": dataset_manifest,
        "manifests/model_manifest.json": model_manifest,
        "environment/environment.json": environment,
        "baseline/model.joblib": model_file,
        "baseline/metrics.json": metrics,
        "baseline/predictions.csv": predictions,
        "baseline/run_summary.json": summary,
        "evaluation/actual_vs_predicted.png": plot,
    }
    module.artifacts = types.SimpleNamespace(
        download_artifacts=lambda **kwargs: str(paths[kwargs["artifact_path"]])
    )
    return module


def _config():
    return TrackingConfig(tracking_uri="sqlite:///test.db")


def test_register_candidate_validates_and_moves_alias(tmp_path: Path, monkeypatch):
    mlflow = _fake_mlflow(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    client = _Client()

    receipt = register_candidate(
        "run-1", config=_config(), client=client, mlflow_module=mlflow
    )

    assert receipt.model_version == "7"
    assert receipt.dataset_version == "v1"
    assert (
        "set_alias",
        "wind-energy-forecast-original",
        "candidate",
        "7",
    ) in client.calls
    assert any(call[0] == "tag" and call[3] == "validation_status" for call in client.calls)


def test_register_candidate_rejects_dirty_run_before_registry_mutation(
    tmp_path: Path, monkeypatch
):
    mlflow = _fake_mlflow(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    client = _Client(run=_Run(dirty="True"))

    with pytest.raises(ValueError, match="Dirty Git"):
        register_candidate(
            "run-1", config=_config(), client=client, mlflow_module=mlflow
        )

    assert client.calls == []


def test_register_candidate_requires_all_four_metrics(tmp_path: Path, monkeypatch):
    mlflow = _fake_mlflow(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    run = _Run()
    run.data.metrics.pop("MAPE_percent")
    client = _Client(run=run)
    with pytest.raises(ValueError, match="missing required metrics"):
        register_candidate(
            "run-1", config=_config(), client=client, mlflow_module=mlflow
        )
    assert client.calls == []


def test_promote_candidate_checks_versions_and_removes_candidate(tmp_path, monkeypatch):
    mlflow = _fake_mlflow(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    comparable = {
        "validation_status": "passed",
        "dataset_version": "v1",
        "dataset_sha256": "a" * 64,
        "feature_schema_sha256": "b" * 64,
        "target_contract": "original",
    }
    client = _Client(
        candidate=_Version("7", tags=comparable),
        champion=_Version("5", tags=comparable),
    )

    receipt = promote_candidate(
        config=_config(),
        expected_candidate_version="7",
        expected_champion_version="5",
        approval_note="manual review passed",
        client=client,
        mlflow_module=mlflow,
    )

    assert receipt.previous_champion_version == "5"
    champion_call = (
        "set_alias",
        "wind-energy-forecast-original",
        "champion",
        "7",
    )
    delete_call = (
        "delete_alias",
        "wind-energy-forecast-original",
        "candidate",
    )
    assert client.calls.index(champion_call) < client.calls.index(delete_call)


def test_promote_candidate_version_mismatch_is_non_mutating(tmp_path, monkeypatch):
    mlflow = _fake_mlflow(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    client = _Client(candidate=_Version("8", tags={"validation_status": "passed"}))

    with pytest.raises(ValueError, match="Candidate changed"):
        promote_candidate(
            config=_config(),
            expected_candidate_version="7",
            expected_champion_version=None,
            approval_note="reviewed",
            client=client,
            mlflow_module=mlflow,
        )
    assert client.calls == []


def test_promotion_delete_failure_restores_previous_aliases(tmp_path, monkeypatch):
    mlflow = _fake_mlflow(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    comparable = {
        "validation_status": "passed",
        "dataset_version": "v1",
        "dataset_sha256": "a" * 64,
        "feature_schema_sha256": "b" * 64,
        "target_contract": "original",
    }

    class FailingClient(_Client):
        def delete_registered_model_alias(self, *args):
            self.calls.append(("delete_alias", *args))
            if args[1] == "candidate":
                raise OSError("delete failed")

    client = FailingClient(
        candidate=_Version("7", tags=comparable),
        champion=_Version("5", tags=comparable),
    )
    with pytest.raises(RuntimeError, match="aliases were restored"):
        promote_candidate(
            config=_config(),
            expected_candidate_version="7",
            expected_champion_version="5",
            approval_note="reviewed",
            client=client,
            mlflow_module=mlflow,
        )
    assert client.calls[-2:] == [
        ("set_alias", "wind-energy-forecast-original", "candidate", "7"),
        ("set_alias", "wind-energy-forecast-original", "champion", "5"),
    ]


def test_existing_receipt_blocks_promotion_before_mutation(tmp_path, monkeypatch):
    mlflow = _fake_mlflow(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    candidate = _Version("7", tags={"validation_status": "passed"})
    client = _Client(candidate=candidate)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Receipt already exists"):
        promote_candidate(
            config=_config(),
            expected_candidate_version="7",
            expected_champion_version=None,
            approval_note="reviewed",
            client=client,
            mlflow_module=mlflow,
            receipt_path=receipt,
        )
    assert client.calls == []


def test_rollback_restores_previous_champion_and_candidate(tmp_path, monkeypatch):
    mlflow = _fake_mlflow(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    client = _Client(champion=_Version("7"))
    receipt = PromotionReceipt(
        registered_model_name="wind-energy-forecast-original",
        promoted_version="7",
        previous_champion_version="5",
        approval_note="reviewed",
        candidate_alias_removed=True,
    )

    rollback_promotion(
        receipt, config=_config(), client=client, mlflow_module=mlflow
    )

    assert (
        "set_alias",
        "wind-energy-forecast-original",
        "champion",
        "5",
    ) in client.calls
    assert (
        "set_alias",
        "wind-energy-forecast-original",
        "candidate",
        "7",
    ) in client.calls


def test_rollback_rejects_receipt_for_another_model(tmp_path, monkeypatch):
    mlflow = _fake_mlflow(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    client = _Client(champion=_Version("7"))
    receipt = PromotionReceipt(
        registered_model_name="other-model",
        promoted_version="7",
        previous_champion_version=None,
        approval_note="reviewed",
        candidate_alias_removed=True,
    )
    with pytest.raises(ValueError, match="different registered model"):
        rollback_promotion(receipt, config=_config(), client=client, mlflow_module=mlflow)
    assert client.calls == []
