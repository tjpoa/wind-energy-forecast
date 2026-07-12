import sys
import types
from pathlib import Path

import pytest

from wind_forecast.tracking import (
    ArtifactReference,
    flatten_metric_groups,
    local_tracking_uri,
    log_dataset_input,
    log_run_data,
    normalize_tracking_key,
    start_local_run,
)


class _FakeRunContext:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        self.calls.append(("enter_run", None))
        return object()

    def __exit__(self, exc_type, exc_value, traceback):
        self.calls.append(("exit_run", exc_type))
        return False


def _fake_mlflow_module():
    module = types.ModuleType("mlflow")
    calls = []
    module.calls = calls

    def set_tracking_uri(uri):
        calls.append(("set_tracking_uri", uri))

    def set_experiment(experiment_name):
        calls.append(("set_experiment", experiment_name))
        return object()

    def start_run(**kwargs):
        calls.append(("start_run", kwargs))
        return _FakeRunContext(calls)

    def log_params(params):
        calls.append(("log_params", params))

    def log_metrics(metrics):
        calls.append(("log_metrics", metrics))

    def set_tags(tags):
        calls.append(("set_tags", tags))

    def log_artifact(local_path, artifact_path=None):
        calls.append(("log_artifact", local_path, artifact_path))

    def from_pandas(frame, **kwargs):
        dataset = object()
        calls.append(("from_pandas", frame, kwargs, dataset))
        return dataset

    def log_input(dataset, context):
        calls.append(("log_input", dataset, context))

    module.set_tracking_uri = set_tracking_uri
    module.set_experiment = set_experiment
    module.start_run = start_run
    module.log_params = log_params
    module.log_metrics = log_metrics
    module.set_tags = set_tags
    module.log_artifact = log_artifact
    module.data = types.SimpleNamespace(from_pandas=from_pandas)
    module.log_input = log_input
    return module


def test_local_tracking_uri_uses_file_uri(tmp_path: Path):
    tracking_dir = tmp_path / "mlruns"

    assert local_tracking_uri(tracking_dir) == tracking_dir.resolve().as_uri()


def test_normalize_tracking_key_handles_metric_punctuation():
    assert normalize_tracking_key("MAPE (%)") == "MAPE_percent"
    assert normalize_tracking_key("Best model / original") == "Best_model_/_original"


def test_flatten_metric_groups_uses_mlflow_friendly_metric_names():
    metrics = flatten_metric_groups(
        {
            "Best Original ANN": {
                "R2": 0.5,
                "MAPE (%)": 12,
            }
        }
    )

    assert metrics == {
        "Best_Original_ANN.R2": 0.5,
        "Best_Original_ANN.MAPE_percent": 12.0,
    }


def test_flatten_metric_groups_rejects_non_finite_metrics():
    with pytest.raises(ValueError, match="finite"):
        flatten_metric_groups({"model": {"RMSE": float("nan")}})


def test_start_local_run_and_log_run_data_use_mocked_mlflow(
    tmp_path: Path,
    monkeypatch,
):
    fake_mlflow = _fake_mlflow_module()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    artifact = tmp_path / "predictions.csv"
    artifact.write_text("Date,Prediction\n2026-01-01,1.0\n", encoding="utf-8")
    tracking_dir = tmp_path / "mlruns"

    with start_local_run(
        "local-evaluation",
        tracking_dir=tracking_dir,
        experiment_name="wind-test",
        tags={"phase": "4A"},
    ):
        log_run_data(
            params={"Input Path": Path("data/processed/example.csv"), "skip": None},
            metrics={"MAPE (%)": 12.5},
            tags={"kind": "evaluation"},
            artifact_paths=[ArtifactReference(artifact, "predictions")],
        )

    assert ("set_tracking_uri", tracking_dir.resolve().as_uri()) in fake_mlflow.calls
    assert ("set_experiment", "wind-test") in fake_mlflow.calls
    assert (
        "start_run",
        {
            "run_name": "local-evaluation",
            "tags": {"phase": "4A"},
            "nested": False,
        },
    ) in fake_mlflow.calls
    assert ("log_params", {"Input_Path": "data/processed/example.csv"}) in fake_mlflow.calls
    assert ("log_metrics", {"MAPE_percent": 12.5}) in fake_mlflow.calls
    assert ("set_tags", {"kind": "evaluation"}) in fake_mlflow.calls
    assert ("log_artifact", str(artifact), "predictions") in fake_mlflow.calls


@pytest.mark.parametrize(
    ("digest", "expected_digest"),
    [
        ("a" * 64, "a" * 36),
        (None, None),
    ],
)
def test_log_dataset_input_limits_mlflow_lineage_digest(
    digest: str | None,
    expected_digest: str | None,
    monkeypatch,
):
    fake_mlflow = _fake_mlflow_module()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    frame = object()

    dataset = log_dataset_input(
        frame,
        source="data/processed/agg_data_ml.csv",
        name="wind-production-v1",
        target="Wind_Production",
        context="training",
        digest=digest,
    )

    assert (
        "from_pandas",
        frame,
        {
            "source": "data/processed/agg_data_ml.csv",
            "name": "wind-production-v1",
            "targets": "Wind_Production",
            "digest": expected_digest,
        },
        dataset,
    ) in fake_mlflow.calls
    assert ("log_input", dataset, "training") in fake_mlflow.calls
