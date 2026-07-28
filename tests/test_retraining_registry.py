from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import wind_forecast.retraining_registry as registry
from wind_forecast.manifests import sha256_file
from wind_forecast.retraining_registry import (
    RetrainingRegistrationConfig,
    RetrainingRegistryError,
    RetrainingRegistryReconciliationError,
    load_retraining_registration_receipt,
    register_retraining_candidate,
)
from wind_forecast.tracking import DEFAULT_REGISTERED_MODEL_NAME


SHA = "a" * 64


def _bundle(path: Path) -> dict:
    candidate_metrics = {
        "MAE": 1.0,
        "RMSE": 2.0,
        "MAPE_percent": 3.0,
        "R2": 0.9,
        "bias": -0.5,
    }
    return {
        "backtest_id": "backtest",
        "backtest": {
            "backtest_id": "backtest",
            "outcome": "accepted",
            "evaluation_id": "evaluation",
            "evaluation_period": "2026-08",
            "identities": {
                "policy_sha256": SHA,
                "calibration_id": "calibration",
                "reference_id": "reference",
                "feature_schema_sha256": SHA,
                "incumbent_model_sha256": SHA,
            },
            "cutoffs": {
                "incumbent_fit_cutoff": "2024-12-31",
                "data_snapshot_cutoff": "2026-07-31",
                "candidate_fit_cutoff": "2026-07-31",
            },
            "aggregate_metrics": {"candidate": candidate_metrics},
            "git": {"git_sha": "b" * 40, "git_dirty": False},
            "final_training": {
                "candidate_model_sha256": sha256_file(path / "model.joblib"),
                "dataset_sha256": "c" * 64,
                "identity_sha256": "d" * 64,
            },
        },
        "git": {"git_sha": "b" * 40, "git_dirty": False},
    }


class Client:
    def __init__(self, candidate: str | None = None) -> None:
        self.aliases = {"candidate": candidate, "champion": "3", "stable": "2"}
        self.tags = {}
        self.alias_sets = []

    def get_run(self, run_id: str):
        return SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"),
            data=SimpleNamespace(
                params={
                    "logged_model_uri": f"runs:/{run_id}/candidate",
                    "backtest_id": "backtest",
                    "git_sha": "b" * 40,
                    "git_dirty": "false",
                    "candidate_model_artifact_path": "candidate/model.joblib",
                }
            ),
        )

    def get_model_version_by_alias(self, model_name: str, alias: str):
        version = self.aliases.get(alias)
        if version is None:
            raise LookupError(alias)
        return SimpleNamespace(version=version)

    def set_model_version_tag(
        self, model_name: str, version: str, key: str, value: str
    ) -> None:
        self.tags[key] = value

    def set_registered_model_alias(
        self, model_name: str, alias: str, version: str
    ) -> None:
        self.aliases[alias] = str(version)
        self.alias_sets.append((alias, str(version)))

    def delete_registered_model_alias(self, model_name: str, alias: str) -> None:
        self.aliases[alias] = None


class Mlflow:
    def __init__(self, client: Client, bundle: Path) -> None:
        self.client = client
        model = joblib.load(bundle / "model.joblib")
        self.pyfunc = SimpleNamespace(load_model=lambda uri: model)
        def numeric(name: str) -> SimpleNamespace:
            return SimpleNamespace(name=name, type="double")

        signature = SimpleNamespace(
            inputs=SimpleNamespace(
                inputs=[numeric("feature")]
            ),
            outputs=SimpleNamespace(inputs=[numeric("prediction")]),
        )
        self.models = SimpleNamespace(
            get_model_info=lambda uri: SimpleNamespace(
                signature=signature, run_id="run-1"
            )
        )
        self.artifacts = SimpleNamespace(
            download_artifacts=lambda **kwargs: str(bundle / "model.joblib")
        )

    def MlflowClient(self) -> Client:
        return self.client

    def register_model(self, *, model_uri: str, name: str):
        return SimpleNamespace(version="7", run_id="run-1")


def _config(
    tmp_path: Path, *, expected: str | None = None, name: str = "wind-v2"
) -> RetrainingRegistrationConfig:
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    features = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    model = RandomForestRegressor(n_estimators=2, random_state=1).fit(
        features, np.asarray([2.0, 4.0, 6.0, 8.0])
    )
    joblib.dump(model, bundle / "model.joblib")
    expected_predictions = model.predict(features)
    pd.DataFrame(
        {
            "feature": features["feature"],
            "Expected_Prediction": expected_predictions,
        }
    ).to_csv(bundle / "reload_sample.csv", index=False)
    pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=4).strftime("%Y-%m-%d"),
            "Wind_Production": [2.0, 4.0, 6.0, 8.0],
            "feature": features["feature"],
            "Expected_Prediction": expected_predictions,
        }
    ).to_csv(bundle / "training_evidence.csv", index=False)
    (bundle / "model_manifest.json").write_text(
        registry.json.dumps(
            {
                "model_type": "RandomForestRegressor",
                "parameters": model.get_params(deep=True),
                "feature_names": ["feature"],
            }
        ),
        encoding="utf-8",
    )
    (bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    return RetrainingRegistrationConfig(
        backtest_bundle=bundle,
        run_id="run-1",
        registered_model_name=name,
        expected_current_candidate_version=expected,
        output_root=tmp_path / "receipts",
    )


def _patch_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry,
        "load_retraining_backtest",
        lambda path: _bundle(Path(path)),
    )


def test_registered_model_name_is_required_and_legacy_default_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(RetrainingRegistryError, match="required"):
        _config(tmp_path, name="")
    with pytest.raises(RetrainingRegistryError, match="legacy"):
        _config(tmp_path, name=DEFAULT_REGISTERED_MODEL_NAME)


def test_registration_moves_only_candidate_and_writes_immutable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bundle(monkeypatch)
    client = Client()
    config = _config(tmp_path)
    receipt = register_retraining_candidate(
        config,
        client=client,
        mlflow_module=Mlflow(client, config.backtest_bundle),
        git_lineage={"git_sha": "b" * 40, "git_dirty": False},
    )
    assert client.aliases == {"candidate": "7", "champion": "3", "stable": "2"}
    assert receipt.champion_before == receipt.champion_after == "3"
    assert receipt.stable_before == receipt.stable_after == "2"
    path = tmp_path / "receipts" / receipt.registration_id / "receipt.json"
    assert load_retraining_registration_receipt(path) == receipt.to_dict()
    assert client.tags["backtest_id"] == "backtest"
    assert receipt.candidate_model_sha256 == sha256_file(
        config.backtest_bundle / "model.joblib"
    )
    assert receipt.final_training_dataset_sha256 == "c" * 64


def test_expected_candidate_mismatch_fails_before_version_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bundle(monkeypatch)
    client = Client(candidate="4")
    config = _config(tmp_path, expected="5")
    mlflow = Mlflow(client, config.backtest_bundle)
    calls = []
    mlflow.register_model = lambda **kwargs: calls.append(kwargs)
    with pytest.raises(RetrainingRegistryError, match="changed before"):
        register_retraining_candidate(
            config,
            client=client,
            mlflow_module=mlflow,
            git_lineage={"git_sha": "b" * 40, "git_dirty": False},
        )
    assert calls == []
    assert client.aliases["champion"] == "3"
    assert client.aliases["stable"] == "2"


def test_alias_race_after_version_creation_fails_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bundle(monkeypatch)

    class RacingClient(Client):
        candidate_reads = 0

        def get_model_version_by_alias(self, model_name: str, alias: str):
            if alias == "candidate":
                self.candidate_reads += 1
                if self.candidate_reads >= 2:
                    return SimpleNamespace(version="99")
            return super().get_model_version_by_alias(model_name, alias)

    client = RacingClient()
    config = _config(tmp_path)
    with pytest.raises(
        RetrainingRegistryReconciliationError, match="compensation failed"
    ):
        register_retraining_candidate(
                config,
                client=client,
                mlflow_module=Mlflow(client, config.backtest_bundle),
            git_lineage={"git_sha": "b" * 40, "git_dirty": False},
        )
    assert client.aliases["champion"] == "3"
    assert client.aliases["stable"] == "2"


def test_receipt_failure_restores_prior_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bundle(monkeypatch)
    monkeypatch.setattr(
        registry,
        "_seal_receipt",
        lambda *args: (_ for _ in ()).throw(OSError("disk")),
    )
    client = Client(candidate="4")
    config = _config(tmp_path, expected="4")
    with pytest.raises(RetrainingRegistryError, match="was restored"):
        register_retraining_candidate(
            config,
            client=client,
            mlflow_module=Mlflow(client, config.backtest_bundle),
            git_lineage={"git_sha": "b" * 40, "git_dirty": False},
        )
    assert client.aliases == {"candidate": "4", "champion": "3", "stable": "2"}


@pytest.mark.parametrize(
    ("status", "dirty"),
    [("RUNNING", False), ("FINISHED", True)],
)
def test_run_must_be_finished_and_git_must_be_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    dirty: bool,
) -> None:
    _patch_bundle(monkeypatch)
    client = Client()
    config = _config(tmp_path)
    if status != "FINISHED":
        client.get_run = lambda run_id: SimpleNamespace(
            info=SimpleNamespace(status=status),
            data=SimpleNamespace(params={}),
        )
    with pytest.raises(RetrainingRegistryError):
        register_retraining_candidate(
            config,
            client=client,
            mlflow_module=Mlflow(client, config.backtest_bundle),
            git_lineage={"git_sha": "b" * 40, "git_dirty": dirty},
        )


def test_lock_contention_fails_before_alias_read_or_version_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bundle(monkeypatch)
    client = Client()
    config = _config(tmp_path)
    lock = registry._registry_lock_path(config)
    lock.parent.mkdir(parents=True)
    lock.write_text("occupied", encoding="utf-8")
    alias_reads = []
    client.get_model_version_by_alias = lambda *args: alias_reads.append(args)
    mlflow = Mlflow(client, config.backtest_bundle)
    versions = []
    mlflow.register_model = lambda **kwargs: versions.append(kwargs)
    with pytest.raises(RetrainingRegistryError, match="locked"):
        register_retraining_candidate(
            config,
            client=client,
            mlflow_module=mlflow,
            git_lineage={"git_sha": "b" * 40, "git_dirty": False},
        )
    assert alias_reads == []
    assert versions == []


def test_model_artifact_hash_mismatch_fails_before_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bundle(monkeypatch)
    client = Client()
    config = _config(tmp_path)
    other = tmp_path / "other.joblib"
    other.write_bytes(b"wrong")
    mlflow = Mlflow(client, config.backtest_bundle)
    mlflow.artifacts.download_artifacts = lambda **kwargs: str(other)
    versions = []
    mlflow.register_model = lambda **kwargs: versions.append(kwargs)
    with pytest.raises(RetrainingRegistryError, match="differs"):
        register_retraining_candidate(
            config,
            client=client,
            mlflow_module=mlflow,
            git_lineage={"git_sha": "b" * 40, "git_dirty": False},
        )
    assert versions == []


def test_raw_artifact_predictions_must_match_logged_model_and_full_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bundle(monkeypatch)
    client = Client()
    config = _config(tmp_path)
    evidence_path = config.backtest_bundle / "training_evidence.csv"
    evidence = pd.read_csv(evidence_path)
    evidence["Expected_Prediction"] = evidence["feature"] * 2.0
    evidence.to_csv(evidence_path, index=False)
    mlflow = Mlflow(client, config.backtest_bundle)

    class LoggedOnlyModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return frame["feature"].to_numpy(dtype=float) * 2.0

    mlflow.pyfunc.load_model = lambda uri: LoggedOnlyModel()
    versions = []
    mlflow.register_model = lambda **kwargs: versions.append(kwargs)
    with pytest.raises(RetrainingRegistryError, match="raw model artifact"):
        register_retraining_candidate(
            config,
            client=client,
            mlflow_module=mlflow,
            git_lineage={"git_sha": "b" * 40, "git_dirty": False},
        )
    assert versions == []


def test_compensation_and_recovery_write_failure_still_reconciles_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bundle(monkeypatch)

    class RacingClient(Client):
        candidate_reads = 0

        def get_model_version_by_alias(self, model_name: str, alias: str):
            if alias == "candidate":
                self.candidate_reads += 1
                if self.candidate_reads >= 2:
                    return SimpleNamespace(version="99")
            return super().get_model_version_by_alias(model_name, alias)

    client = RacingClient()
    config = _config(tmp_path)
    monkeypatch.setattr(
        registry,
        "_write_recovery_evidence",
        lambda *args: (_ for _ in ()).throw(OSError("recovery disk failed")),
    )
    with pytest.raises(
        RetrainingRegistryReconciliationError,
        match="recovery evidence could not be written",
    ) as raised:
        register_retraining_candidate(
            config,
            client=client,
            mlflow_module=Mlflow(client, config.backtest_bundle),
            git_lineage={"git_sha": "b" * 40, "git_dirty": False},
        )
    assert "Compensation failure" in str(raised.value)
    assert "Recovery-write failure" in str(raised.value)
