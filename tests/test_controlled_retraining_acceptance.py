from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import wind_forecast.deployment_runtime as deployment_runtime
import wind_forecast.monitoring as monitoring
import wind_forecast.monitoring_reporting as monitoring_reporting
import wind_forecast.retraining_backtesting as backtesting
import wind_forecast.retraining_deployment as deployment
import wind_forecast.retraining_lifecycle as lifecycle
import wind_forecast.retraining_registry as registry
from wind_forecast.manifest_validation import validate_v1_source_contract
from wind_forecast.manifests import sha256_file
from wind_forecast.v1_contracts import load_serving_contract
from wind_forecast.monitoring import MonitoringConfig, run_historical_monitoring
from wind_forecast.orchestration import (
    BatchConfig,
    BatchOrchestrationError,
    run_batch,
)
from wind_forecast.retraining_deployment import (
    DeploymentBootstrapConfig,
    RetrainingDeploymentError,
    bootstrap_v2_deployment,
    load_verified_deployment_pointer,
)
from wind_forecast.retraining_lifecycle import (
    ExpectedDeploymentState,
    LifecycleConfig,
    RetrainingLifecycleError,
    execute_lifecycle_transition,
    plan_lifecycle_transition,
)
from wind_forecast.retraining_registry import (
    RetrainingRegistrationConfig,
    register_retraining_candidate,
)
from wind_forecast.v2_features import TRANSFORMATION_VERSION


REGISTERED_MODEL = "wind-v2-acceptance"
REFERENCE_BUNDLE_SHA = "1" * 64
REFERENCE_MODEL_SHA = "2" * 64
DATASET_SHA = "4" * 64
FEATURE_SCHEMA_SHA = sha256(b'[\"Month\"]').hexdigest()
BOOTSTRAP_SNAPSHOT_ID = "5" * 64
BOOTSTRAP_DAY = "2026-03-29"
PROBATION_DAY = "2026-06-01"
BOOTSTRAP_NOW = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
PROMOTION_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
GIT_SHA = "b" * 40

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class _RegistryClient:
    def __init__(self) -> None:
        self.model_exists = False
        self.aliases: dict[str, str] = {}
        self.versions: dict[str, dict[str, Any]] = {}
        self.alias_mutations: list[tuple[str, str, str | None]] = []

    def get_registered_model(self, name: str) -> SimpleNamespace:
        if not self.model_exists:
            raise LookupError(name)
        return SimpleNamespace(name=name)

    def search_model_versions(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(name=REGISTERED_MODEL, version=version)
            for version in sorted(self.versions, key=int)
        ]

    def get_model_version_by_alias(
        self, _name: str, alias: str
    ) -> SimpleNamespace:
        if alias not in self.aliases:
            raise LookupError(f"{alias} missing")
        return SimpleNamespace(version=self.aliases[alias])

    def get_model_version(
        self, name: str, version: str
    ) -> SimpleNamespace:
        stored = self.versions[str(version)]
        return SimpleNamespace(
            name=name,
            version=str(version),
            run_id=stored["run_id"],
            tags=dict(stored["tags"]),
        )

    def set_model_version_tag(
        self,
        _name: str,
        version: str,
        key: str,
        value: str,
    ) -> None:
        self.versions[str(version)]["tags"][key] = value

    def set_registered_model_alias(
        self, _name: str, alias: str, version: str
    ) -> None:
        self.aliases[alias] = str(version)
        self.alias_mutations.append(("set", alias, str(version)))

    def delete_registered_model_alias(self, _name: str, alias: str) -> None:
        self.aliases.pop(alias, None)
        self.alias_mutations.append(("delete", alias, None))

    def get_run(self, run_id: str) -> SimpleNamespace:
        params = {
            "logged_model_uri": f"runs:/{run_id}/candidate",
            "backtest_id": "accepted-backtest",
            "git_sha": GIT_SHA,
            "git_dirty": "false",
            "candidate_model_artifact_path": "candidate/model.joblib",
        }
        return SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"),
            data=SimpleNamespace(params=params),
        )


class _Mlflow:
    def __init__(self, client: _RegistryClient) -> None:
        self.client = client
        self.candidate_bundle: Path | None = None
        self.tracking_uri: str | None = None
        self.pyfunc = SimpleNamespace(load_model=self._load_model)
        self.models = SimpleNamespace(get_model_info=self._model_info)
        self.artifacts = SimpleNamespace(
            download_artifacts=self._download_artifact
        )

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def MlflowClient(self) -> _RegistryClient:
        return self.client

    def register_model(self, *, model_uri: str, name: str) -> SimpleNamespace:
        version = str(len(self.client.versions) + 1)
        run_id = "bootstrap-run" if version == "1" else "candidate-run"
        self.client.model_exists = True
        self.client.versions[version] = {"run_id": run_id, "tags": {}}
        return SimpleNamespace(
            name=name,
            version=version,
            run_id=run_id,
            source=model_uri,
        )

    def _load_model(self, _uri: str) -> Any:
        return joblib.load(self._candidate_model_path())

    def _model_info(self, _uri: str) -> SimpleNamespace:
        def numeric(name: str) -> SimpleNamespace:
            return SimpleNamespace(name=name, type="double")

        signature = SimpleNamespace(
            inputs=SimpleNamespace(inputs=[numeric("Month")]),
            outputs=SimpleNamespace(inputs=[numeric("prediction")]),
        )
        return SimpleNamespace(signature=signature, run_id="candidate-run")

    def _download_artifact(self, **_kwargs: Any) -> str:
        return str(self._candidate_model_path())

    def _candidate_model_path(self) -> Path:
        assert self.candidate_bundle is not None
        return self.candidate_bundle / "model.joblib"


@dataclass
class _Harness:
    root: Path
    client: _RegistryClient
    mlflow: _Mlflow
    reference_bundle: Path
    bootstrap_calibration: Path
    bootstrap_monitoring: Path
    deployment_root: Path
    registry_lock_root: Path
    runtime_monitoring: Path
    source_state: dict[str, Any]
    source_manifest: Path
    candidate_bundle: Path
    candidate_backtest: dict[str, Any]
    candidate_calibration: Path
    bootstrap_result: Any | None = None
    registration_receipt_path: Path | None = None
    promotion_result: Any | None = None
    bootstrap_era_id: str | None = None
    probation_era: dict[str, Any] | None = None


def _make_reference_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    features = pd.DataFrame({"Month": [1.0, 2.0, 3.0, 4.0]})
    model = RandomForestRegressor(n_estimators=2, random_state=11).fit(
        features, np.asarray([10.0, 20.0, 30.0, 40.0])
    )
    joblib.dump(model, root / "model.joblib")
    schema_hash = _json_hash(["Month"])
    documents: dict[str, dict[str, Any]] = {
        "model_manifest.json": {
            "schema_version": "wind_forecast.v2_model_manifest.v1",
            "task": "daily_wind_production_historical_hindcast",
            "model_type": type(model).__name__,
            "model_sha256": sha256_file(root / "model.joblib"),
            "dataset_version": "v2",
            "dataset_sha256": DATASET_SHA,
            "feature_names": ["Month"],
            "feature_schema_sha256": schema_hash,
            "scaler_required": False,
            "scaler": None,
            "reference_status": "selected_not_promoted",
        },
        "dataset_manifest.json": {
            "schema_version": "wind_forecast.v2_training_dataset.v1",
            "dataset_version": "v2",
            "transformation_version": TRANSFORMATION_VERSION,
            "sha256": DATASET_SHA,
            "target": "Wind_Production",
            "feature_names": ["Month"],
            "feature_schema_sha256": schema_hash,
        },
        "reference_decision.json": {
            "schema_version": "wind_forecast.v2_reference_decision.v1",
            "accepted_as_reference": True,
            "status": "selected_not_promoted",
            "automatic_promotion": False,
        },
        "run_summary.json": {
            "schema_version": "wind_forecast.v2_training_run.v1",
            "accepted_as_reference": True,
            "dataset_version": "v2",
            "dataset_sha256": DATASET_SHA,
            "scaler_required": False,
            "artifact_sha256": {},
        },
        "environment.json": {
            "schema_version": "wind_forecast.v2_environment.v1",
            "git_sha": GIT_SHA,
            "git_dirty": False,
        },
        "leakage_audit.json": {
            "schema_version": "wind_forecast.v2_leakage_audit.v1",
            "forecast_contract": "historical_daily_hindcast",
            "passed": True,
        },
    }
    for name, payload in documents.items():
        _write_json(root / name, payload)
    documents["run_summary.json"]["artifact_sha256"] = {
        name: sha256_file(root / name)
        for name in (
            "model.joblib",
            "model_manifest.json",
            "dataset_manifest.json",
            "reference_decision.json",
            "environment.json",
            "leakage_audit.json",
        )
    }
    _write_json(root / "run_summary.json", documents["run_summary.json"])
    return root


def _make_candidate_bundle(root: Path) -> tuple[Path, dict[str, Any]]:
    root.mkdir(parents=True)
    features = pd.DataFrame({"Month": [1.0, 2.0, 3.0, 4.0]})
    target = np.asarray([12.0, 22.0, 32.0, 42.0])
    model = RandomForestRegressor(n_estimators=3, random_state=17).fit(
        features, target
    )
    joblib.dump(model, root / "model.joblib")
    expected = model.predict(features)
    pd.DataFrame(
        {"Month": features["Month"], "Expected_Prediction": expected}
    ).to_csv(root / "reload_sample.csv", index=False)
    pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=4).strftime(
                "%Y-%m-%d"
            ),
            "Wind_Production": target,
            "Month": features["Month"],
            "Expected_Prediction": expected,
        }
    ).to_csv(root / "training_evidence.csv", index=False)
    model_sha = sha256_file(root / "model.joblib")
    schema_hash = _json_hash(["Month"])
    _write_json(
        root / "model_manifest.json",
        {
            "model_type": type(model).__name__,
            "parameters": model.get_params(deep=True),
            "feature_names": ["Month"],
            "feature_schema_sha256": schema_hash,
            "model_sha256": model_sha,
        },
    )
    _write_json(
        root / "dataset_manifest.json",
        {
            "final_training_dataset_sha256": "c" * 64,
            "candidate_fit_cutoff": "2026-04-30",
            "row_count": 4,
        },
    )
    _write_json(
        root / "environment.json",
        {"git": {"git_sha": GIT_SHA, "git_dirty": False}},
    )
    _write_json(root / "bundle_manifest.json", {"accepted": True})
    metrics = {
        "MAE": 1.0,
        "RMSE": 2.0,
        "MAPE_percent": 3.0,
        "R2": 0.9,
        "bias": 0.0,
    }
    backtest = {
        "backtest_id": "accepted-backtest",
        "outcome": "accepted",
        "evaluation_id": "evaluation",
        "evaluation_period": "2026-05",
        "identities": {
            "policy_sha256": "a" * 64,
            "calibration_id": "bootstrap-calibration",
            "reference_id": "bootstrap-reference",
            "feature_schema_sha256": schema_hash,
            "incumbent_model_sha256": REFERENCE_MODEL_SHA,
        },
        "cutoffs": {
            "incumbent_fit_cutoff": "2024-12-31",
            "data_snapshot_cutoff": "2026-04-30",
            "candidate_fit_cutoff": "2026-04-30",
            "monitoring_evaluation_cutoff": "2026-05-31",
        },
        "aggregate_metrics": {"candidate": metrics},
        "git": {"git_sha": GIT_SHA, "git_dirty": False},
        "final_training": {
            "candidate_model_sha256": model_sha,
            "dataset_sha256": "c" * 64,
            "identity_sha256": "d" * 64,
        },
    }
    return root, {
        "backtest_id": backtest["backtest_id"],
        "backtest": backtest,
        "git": {"git_sha": GIT_SHA, "git_dirty": False},
    }


def _make_calibration(
    root: Path,
    *,
    calibration_id: str,
    reference_id: str,
    model_sha256: str,
    backtest_id: str | None,
) -> Path:
    reference_root = root.parent / f"{reference_id}-files"
    reference_path = reference_root / "reference.csv"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text("Month\n1\n", encoding="utf-8")
    _write_json(
        root / "calibration.json",
        {
            "calibration_id": calibration_id,
            "reference_id": reference_id,
            "model_sha256": model_sha256,
            "backtest_id": backtest_id,
            "reference_path": str(reference_path),
        },
    )
    _write_json(root / "backtest_summary.json", {"accepted": True})
    return root


def _load_synthetic_calibration(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    payload = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    return {
        **payload,
        "_reference_manifest": {
            "reference_id": payload["reference_id"],
            "model_sha256": payload["model_sha256"],
            "calibration_subject": {
                "backtest_id": payload.get("backtest_id")
            },
        },
        "_reference_path": payload["reference_path"],
    }


def _write_source_day(root: Path, day: str, month: float) -> dict[str, Any]:
    day_root = root / day
    ren_path = day_root / "ren.csv"
    era_path = day_root / "era.csv"
    feature_path = day_root / "features.csv"
    ren_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [f"{day}T00:00:00Z", f"{day}T00:15:00Z"],
            "wind_production_mw": [20.0, 20.0],
            "unit": ["MW", "MW"],
            "source_date": [day, day],
            "retrieval_timestamp_utc": [
                f"{day}T12:00:00Z",
                f"{day}T12:00:00Z",
            ],
        }
    ).to_csv(ren_path, index=False)
    era_path.write_text(
        "timestamp_utc,wind_speed_m_s,temperature_2m_c\n"
        f"{day}T00:00:00Z,5,12\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {"Date": [day], "Wind_Production": [40.0], "Month": [month]}
    ).to_csv(feature_path, index=False)
    ren_sha = sha256_file(ren_path)
    era_sha = sha256_file(era_path)
    return {
        "ren": {
            "logical_key": day,
            "status": "complete",
            "provider_finality": "final",
            "physical_sha256": ren_sha,
            "semantic_sha256": ren_sha,
            "primary_path": str(ren_path),
            "local_dates": [day],
            "revision": 1,
            "revision_id": f"ren-{day}",
            "history": [],
            "supporting_observations": [],
        },
        "era": {
            "logical_key": f"station=1/month={day[:7]}",
            "status": "complete",
            "physical_sha256": era_sha,
            "semantic_sha256": era_sha,
            "primary_path": str(era_path),
            "local_dates": [day],
            "revision": 1,
            "revision_id": f"era-{day}",
        },
        "feature": {
            "partition_key": f"features-{day}",
            "feature_ready": True,
            "files": {
                "feature_ready": {
                    "path": str(feature_path),
                    "sha256": sha256_file(feature_path),
                }
            },
        },
    }


def _make_source_state(root: Path) -> tuple[dict[str, Any], Path]:
    bootstrap = _write_source_day(root, BOOTSTRAP_DAY, 3.0)
    probation = _write_source_day(root, PROBATION_DAY, 6.0)
    manifest = root / "source-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "wind_forecast.synthetic_source.v1",
            "days": [BOOTSTRAP_DAY, PROBATION_DAY],
        },
    )
    state = {
        "schema_version": "wind_forecast.v2_incremental_state.v1",
        "generation": 1,
        "release_id": "synthetic-release",
        "manifest_path": str(manifest),
        "versions": {"features": TRANSFORMATION_VERSION},
        "git_commit": GIT_SHA,
        "sources": {
            "ren": {
                BOOTSTRAP_DAY: bootstrap["ren"],
                PROBATION_DAY: probation["ren"],
            },
            "era5_land": {
                "bootstrap": bootstrap["era"],
                "probation": probation["era"],
            },
        },
        "partitions": {
            "features": {
                BOOTSTRAP_DAY: bootstrap["feature"],
                PROBATION_DAY: probation["feature"],
            }
        },
    }
    return state, manifest


def _new_harness(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> _Harness:
    reference_bundle = _make_reference_bundle(root / "reference-bundle")
    bootstrap_calibration = _make_calibration(
        root / "bootstrap-calibration",
        calibration_id="bootstrap-calibration",
        reference_id="bootstrap-reference",
        model_sha256=REFERENCE_MODEL_SHA,
        backtest_id=None,
    )
    bootstrap_monitoring = root / "bootstrap-monitoring"
    _write_json(bootstrap_monitoring / "state" / "current.json", {})
    source_state, source_manifest = _make_source_state(root / "source")
    candidate_bundle, candidate_backtest = _make_candidate_bundle(
        root / "candidate-bundle"
    )
    candidate_model_sha = candidate_backtest["backtest"]["final_training"][
        "candidate_model_sha256"
    ]
    candidate_calibration = _make_calibration(
        root / "candidate-calibration",
        calibration_id="candidate-calibration",
        reference_id="candidate-reference",
        model_sha256=candidate_model_sha,
        backtest_id="accepted-backtest",
    )
    client = _RegistryClient()
    mlflow = _Mlflow(client)
    mlflow.candidate_bundle = candidate_bundle
    harness = _Harness(
        root=root,
        client=client,
        mlflow=mlflow,
        reference_bundle=reference_bundle,
        bootstrap_calibration=bootstrap_calibration,
        bootstrap_monitoring=bootstrap_monitoring,
        deployment_root=root / "deployment",
        registry_lock_root=root / "registry-lock",
        runtime_monitoring=root / "runtime-monitoring",
        source_state=source_state,
        source_manifest=source_manifest,
        candidate_bundle=candidate_bundle,
        candidate_backtest=candidate_backtest,
        candidate_calibration=candidate_calibration,
    )
    _patch_synthetic_boundaries(harness, monkeypatch)
    return harness


def _patch_synthetic_boundaries(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_evidence = {
        "bundle_sha256": REFERENCE_BUNDLE_SHA,
        "model_manifest": {
            "model_sha256": REFERENCE_MODEL_SHA,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA,
        },
        "dataset_manifest": {
            "sha256": DATASET_SHA,
            "splits": {"validation": {"end": "2024-12-31"}},
        },
    }
    monkeypatch.setattr(
        deployment, "_load_exact_bundle", lambda _path: reference_evidence
    )
    monkeypatch.setattr(
        deployment,
        "_load_calibration",
        lambda path: _load_synthetic_calibration(path),
    )
    monkeypatch.setattr(
        deployment,
        "_load_ledger",
        lambda _path: {
            "model_snapshot_id": BOOTSTRAP_SNAPSHOT_ID,
            "activation_date": BOOTSTRAP_DAY,
        },
    )
    monkeypatch.setattr(
        deployment,
        "_load_ledger_snapshot",
        lambda _root, _ledger: {
            "model_snapshot_id": BOOTSTRAP_SNAPSHOT_ID,
            "model": {"model_sha256": REFERENCE_MODEL_SHA},
            "feature_schema_sha256": FEATURE_SCHEMA_SHA,
            "dataset": {"dataset_sha256": DATASET_SHA},
        },
    )
    monkeypatch.setattr(
        deployment,
        "_verify_mlflow",
        lambda _config, **_kwargs: {
            "run_id": "bootstrap-run",
            "model_uri": "models:/accepted-v2",
        },
    )
    monkeypatch.setattr(
        deployment,
        "load_exact_v2_bundle",
        lambda _path: reference_evidence,
    )
    monkeypatch.setattr(
        lifecycle,
        "load_exact_v2_bundle",
        lambda _path: reference_evidence,
    )
    monkeypatch.setattr(
        lifecycle,
        "load_monitoring_calibration",
        _load_synthetic_calibration,
    )
    monkeypatch.setattr(
        monitoring_reporting,
        "load_monitoring_calibration",
        _load_synthetic_calibration,
    )
    monkeypatch.setattr(
        registry,
        "load_retraining_backtest",
        lambda _path: harness.candidate_backtest,
    )
    monkeypatch.setattr(
        lifecycle,
        "load_retraining_backtest",
        lambda _path: harness.candidate_backtest,
    )
    monkeypatch.setattr(
        backtesting,
        "load_retraining_backtest",
        lambda _path: harness.candidate_backtest,
    )
    monkeypatch.setattr(
        monitoring,
        "load_verified_current_state",
        lambda _path: json.loads(json.dumps(harness.source_state)),
    )
    original_verify = deployment_runtime.verify_active_model_era

    def verify_for_monitoring(
        deployment_root: str | Path,
        model_bundle: str | Path | None = None,
        *,
        calibration_dir: str | Path | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return original_verify(
            deployment_root,
            model_bundle,
            calibration_dir=calibration_dir,
            client=harness.client,
            mlflow_module=harness.mlflow,
        )

    monkeypatch.setattr(
        monitoring, "verify_active_model_era", verify_for_monitoring
    )


def _bootstrap_config(
    harness: _Harness, **changes: Any
) -> DeploymentBootstrapConfig:
    values = {
        "model_bundle": harness.reference_bundle,
        "calibration_dir": harness.bootstrap_calibration,
        "monitoring_store_root": harness.bootstrap_monitoring,
        "deployment_root": harness.deployment_root,
        "registry_lock_root": harness.registry_lock_root,
        "registered_model_name": REGISTERED_MODEL,
        "tracking_uri": "file:synthetic-mlruns",
        "expect_no_deployment_pointer": True,
        "expect_no_v2_registry_state": True,
        "dry_run": True,
        "now_utc": BOOTSTRAP_NOW,
    }
    values.update(changes)
    return DeploymentBootstrapConfig(**values)


def _approval(
    harness: _Harness,
    template: Mapping[str, Any],
    name: str,
) -> tuple[Path, str]:
    payload = dict(template)
    payload.update(
        {
            "approved_by": "operator@example.test",
            "approved_at_utc": "2026-07-28T11:00:00Z",
            "note": f"Manual acceptance {name}.",
        }
    )
    path = harness.root / "approvals" / f"{name}.json"
    _write_json(path, payload)
    return path, sha256_file(path)


def _execute_bootstrap(harness: _Harness) -> None:
    config = _bootstrap_config(harness)
    planned = bootstrap_v2_deployment(
        config,
        client=harness.client,
        mlflow_module=harness.mlflow,
    )
    assert planned.status == "planned"
    assert not harness.deployment_root.exists()
    approval_path, approval_sha = _approval(
        harness, planned.plan.approval_template, "bootstrap"
    )
    result = bootstrap_v2_deployment(
        replace(
            config,
            dry_run=False,
            approval_path=approval_path,
            approval_sha256=approval_sha,
        ),
        client=harness.client,
        mlflow_module=harness.mlflow,
    )
    harness.bootstrap_result = result
    verified = load_verified_deployment_pointer(
        harness.deployment_root, client=harness.client
    )
    assert result.status == "bootstrapped"
    assert verified["pointer"]["generation"] == 1
    assert verified["receipt"]["bootstrap_exception"] is True
    assert harness.client.aliases == {"stable": "1", "champion": "1"}


def _run_synthetic_batch(
    harness: _Harness,
    *,
    suffix: str,
    now_utc: datetime,
) -> Any:
    monitoring_config = MonitoringConfig(
        source_store_root=harness.root / "source-store",
        monitoring_store_root=harness.runtime_monitoring,
        model_bundle=harness.reference_bundle,
        deployment_root=harness.deployment_root,
        through_date=BOOTSTRAP_DAY,
        activation_date=BOOTSTRAP_DAY,
        now_utc=now_utc,
    )
    mutations_before = list(harness.client.alias_mutations)
    monitoring_runs = 0

    def runner(command: Sequence[str], _timeout: int) -> Mapping[str, Any]:
        nonlocal monitoring_runs
        joined = " ".join(str(item) for item in command)
        if "verify_active_deployment.py" in joined:
            era = deployment_runtime.verify_active_model_era(
                harness.deployment_root,
                harness.reference_bundle,
                calibration_dir=harness.bootstrap_calibration,
                client=harness.client,
            )
            return {"status": "verified", **era}
        if "update_v2_dataset.py" in joined:
            if "--dry-run" in command:
                return {"status": "planned"}
            return {
                "status": "succeeded",
                "manifest_path": str(harness.source_manifest),
            }
        if "run_historical_monitoring.py" in joined:
            monitoring_runs += 1
            return run_historical_monitoring(monitoring_config).summary()
        if "run_monitoring_report.py" in joined:
            return {
                "status": "succeeded",
                "report_id": f"report-{suffix}",
                "active_alert_count": 0,
            }
        raise AssertionError(f"Unexpected batch command: {joined}")

    config = BatchConfig(
        model_bundle=harness.reference_bundle,
        calibration_dir=harness.bootstrap_calibration,
        deployment_root=harness.deployment_root,
        through_date=BOOTSTRAP_DAY,
        source_store_root=harness.root / "source-store",
        monitoring_store_root=harness.runtime_monitoring,
        orchestration_root=harness.root / f"orchestration-{suffix}",
        no_source_refresh=True,
        now_utc=now_utc,
    )
    result = run_batch(config, runner=runner)
    assert result.status == "succeeded"
    assert monitoring_runs == 1
    assert harness.client.alias_mutations == mutations_before
    if suffix == "bootstrap":
        harness.bootstrap_era_id = result.summary()["model_era_id"]
    return result


def _register_candidate(harness: _Harness) -> None:
    config = RetrainingRegistrationConfig(
        backtest_bundle=harness.candidate_bundle,
        run_id="candidate-run",
        registered_model_name=REGISTERED_MODEL,
        expected_current_candidate_version=None,
        output_root=harness.root / "registration-receipts",
        registry_lock_root=harness.registry_lock_root,
    )
    receipt = register_retraining_candidate(
        config,
        client=harness.client,
        mlflow_module=harness.mlflow,
        git_lineage={"git_sha": GIT_SHA, "git_dirty": False},
    )
    receipt_path = (
        config.output_root / receipt.registration_id / "receipt.json"
    )
    harness.registration_receipt_path = receipt_path
    assert receipt.model_version == "2"
    assert receipt_path.is_file()
    assert harness.client.aliases == {
        "candidate": "2",
        "champion": "1",
        "stable": "1",
    }


def _current_expected(
    harness: _Harness,
    *,
    candidate: str | None,
    champion: str,
    stable: str,
) -> ExpectedDeploymentState:
    verified = load_verified_deployment_pointer(
        harness.deployment_root, client=harness.client
    )
    return ExpectedDeploymentState(
        generation=int(verified["pointer"]["generation"]),
        deployment_state_id=str(
            verified["pointer"]["deployment_state_id"]
        ),
        pointer_sha256=sha256_file(verified["pointer_path"]),
        candidate=candidate,
        champion=champion,
        stable=stable,
    )


def _promotion_config(harness: _Harness) -> LifecycleConfig:
    assert harness.registration_receipt_path is not None
    return LifecycleConfig(
        action="promote",
        deployment_root=harness.deployment_root,
        registry_lock_root=harness.registry_lock_root,
        registered_model_name=REGISTERED_MODEL,
        tracking_uri="file:synthetic-mlruns",
        expected=_current_expected(
            harness, candidate="2", champion="1", stable="1"
        ),
        dry_run=True,
        candidate_bundle=harness.candidate_bundle,
        candidate_calibration=harness.candidate_calibration,
        incumbent_bundle=harness.reference_bundle,
        incumbent_calibration=harness.bootstrap_calibration,
        registration_receipt=harness.registration_receipt_path,
        promotion_effective_date=PROBATION_DAY,
        now_utc=PROMOTION_NOW,
    )


def _execute_promotion(harness: _Harness) -> LifecycleConfig:
    config = _promotion_config(harness)
    plan = plan_lifecycle_transition(config, client=harness.client)
    approval_path, approval_sha = _approval(
        harness, plan.approval_template, "promote"
    )
    approved = replace(
        config,
        dry_run=False,
        approval_path=approval_path,
        approval_sha256=approval_sha,
    )
    result = execute_lifecycle_transition(approved, client=harness.client)
    harness.promotion_result = result
    verified = load_verified_deployment_pointer(
        harness.deployment_root, client=harness.client
    )
    assert result.status == "probationary"
    assert verified["state"]["lifecycle_status"] == "probationary"
    assert verified["pointer"]["generation"] == 2
    assert harness.client.aliases == {"champion": "2", "stable": "1"}
    return approved


def _run_probation_monitoring(harness: _Harness) -> None:
    verified = load_verified_deployment_pointer(
        harness.deployment_root, client=harness.client
    )
    bundle_path = (
        harness.deployment_root
        / verified["state"]["artifacts"]["bundle"]["path"]
    )
    era = deployment_runtime.verify_active_model_era(
        harness.deployment_root,
        bundle_path,
        client=harness.client,
    )
    assert era["deployment"]["generation"] == 2
    assert era["registry"]["model_version"] == "2"
    assert harness.bootstrap_era_id is not None
    assert era["model_era_id"] != harness.bootstrap_era_id
    mutations_before = list(harness.client.alias_mutations)
    config = MonitoringConfig(
        source_store_root=harness.root / "source-store",
        monitoring_store_root=harness.root / "probation-monitoring",
        model_bundle=bundle_path,
        deployment_root=harness.deployment_root,
        through_date=PROBATION_DAY,
        activation_date=PROBATION_DAY,
        now_utc=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
    )
    result = run_historical_monitoring(config)
    assert result.status == "succeeded"
    assert result.plan.model_era_id == era["model_era_id"]
    assert harness.client.alias_mutations == mutations_before
    harness.probation_era = era


def _prepare_probationary(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> _Harness:
    harness = _new_harness(root, monkeypatch)
    _execute_bootstrap(harness)
    _run_synthetic_batch(
        harness, suffix="bootstrap", now_utc=BOOTSTRAP_NOW
    )
    _register_candidate(harness)
    _run_synthetic_batch(
        harness,
        suffix="candidate-staged",
        now_utc=BOOTSTRAP_NOW + timedelta(minutes=1),
    )
    _execute_promotion(harness)
    _run_probation_monitoring(harness)
    return harness


def _stability_evidence(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
) -> tuple[LifecycleConfig, list[dict[str, str]]]:
    assert harness.probation_era is not None
    expected = _current_expected(
        harness, candidate=None, champion="2", stable="1"
    )
    policy_path = harness.root / "stability-policy.json"
    monitoring_policy_path = harness.root / "monitoring-policy.json"
    report_path = harness.root / "stability-report.json"
    recommendation_path = harness.root / "stability-recommendation.json"
    _write_json(
        policy_path,
        {
            "automation": {"automatic_stability": False},
            "stability": {
                "allowed_issuance_kinds": ["scheduled", "catch_up"],
                "minimum_eligible_observations": 90,
                "require_no_active_warning_or_critical": True,
                "require_second_manual_approval": True,
            },
        },
    )
    monitoring_policy = {
        "schema_version": "wind_forecast.monitoring_policy.v1",
        "windows_days": [30, 90],
    }
    _write_json(monitoring_policy_path, monitoring_policy)
    _write_json(report_path, {"sealed": True})
    _write_json(recommendation_path, {"sealed": True})
    observations = [
        {
            "target_date": (
                date.fromisoformat(PROBATION_DAY) + timedelta(days=index)
            ).isoformat(),
            "prediction_id": f"prediction-{index:03d}",
            "actual_revision_id": f"actual-{index:03d}",
        }
        for index in range(90)
    ]
    observation_cutoff = date.fromisoformat(
        observations[-1]["target_date"]
    )
    state = load_verified_deployment_pointer(
        harness.deployment_root, client=harness.client
    )["state"]
    report = {
        "report_id": "probation-health",
        "through_date": observation_cutoff.isoformat(),
        "active_alerts": {},
        "breaches": [],
        "quality": {"issues": []},
        "reference": {
            "policy_sha256": sha256_file(monitoring_policy_path)
        },
        "config": monitoring_policy,
        "model_era": {
            "model_era_id": harness.probation_era["model_era_id"],
            "deployment_id": state["deployment_id"],
            "model_version": "2",
        },
    }
    recommendation = {
        "recommendation_id": "stability-ready",
        "policy": {
            "sha256": sha256_file(policy_path),
            "monitoring_policy_sha256": sha256_file(
                monitoring_policy_path
            ),
        },
        "deployment": {
            "deployment_id": state["deployment_id"],
            "deployment_state_id": state["deployment_state_id"],
            "generation": 2,
            "expected_aliases": expected.aliases(),
            "pointer_sha256": expected.pointer_sha256,
        },
        "stability": {
            "decision": "ready_for_second_manual_approval",
            "second_manual_approval_required": True,
            "observation_cutoff": observation_cutoff.isoformat(),
            "fixed_observations": observations,
        },
    }
    selected = observations[:count]
    ledger = {
        "active_model_era_id": harness.probation_era["model_era_id"],
        "as_issued": {
            item["target_date"]: item["prediction_id"]
            for item in selected
        },
        "actuals": {
            item["target_date"]: item["actual_revision_id"]
            for item in selected
        },
    }
    evidence = {
        item["prediction_id"]: {
            "prediction": {
                "prediction_id": item["prediction_id"],
                "target_date": item["target_date"],
                "model_era_id": harness.probation_era["model_era_id"],
                "issuance_kind": "scheduled",
                "prediction": 25.0,
            },
            "actual_revisions": [
                {
                    "actual_revision_id": item["actual_revision_id"],
                    "target_date": item["target_date"],
                    "actual": 26.0,
                }
            ],
        }
        for item in selected
    }
    monkeypatch.setattr(
        lifecycle, "load_monitoring_report", lambda _path: report
    )
    monkeypatch.setattr(
        lifecycle,
        "load_monitoring_report_state",
        lambda _root: {
            "latest_report_id": report["report_id"],
            "latest_through_date": report["through_date"],
            "active_alerts": {},
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "load_monthly_governance_recommendation",
        lambda _path: recommendation,
    )
    monkeypatch.setattr(
        lifecycle, "load_verified_monitoring_state", lambda _root: ledger
    )
    monkeypatch.setattr(
        lifecycle,
        "load_prediction_evidence",
        lambda _root, prediction_id: evidence[prediction_id],
    )
    config = LifecycleConfig(
        action="stabilize",
        deployment_root=harness.deployment_root,
        registry_lock_root=harness.registry_lock_root,
        registered_model_name=REGISTERED_MODEL,
        tracking_uri="file:synthetic-mlruns",
        expected=expected,
        monitoring_store_root=harness.root / "probation-monitoring",
        monitoring_report=report_path,
        policy_path=policy_path,
        monitoring_policy_path=monitoring_policy_path,
        observation_cutoff=observation_cutoff,
        monthly_recommendation=recommendation_path,
        dry_run=True,
        now_utc=datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc),
    )
    return config, observations


def test_full_v2_lifecycle_stabilizes_only_after_90_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _prepare_probationary(tmp_path, monkeypatch)
    premature, _observations = _stability_evidence(
        harness, monkeypatch, count=89
    )
    pointer_before = (
        harness.deployment_root / "state" / "current.json"
    ).read_bytes()
    aliases_before = dict(harness.client.aliases)

    with pytest.raises(
        RetrainingLifecycleError,
        match="exactly 90 eligible observations; found 89",
    ):
        plan_lifecycle_transition(premature, client=harness.client)

    assert (
        harness.deployment_root / "state" / "current.json"
    ).read_bytes() == pointer_before
    assert harness.client.aliases == aliases_before

    ready, observations = _stability_evidence(
        harness, monkeypatch, count=90
    )
    plan = plan_lifecycle_transition(ready, client=harness.client)
    assert plan.evidence["eligible_observation_count"] == 90
    assert len(observations) == 90
    approval_path, approval_sha = _approval(
        harness, plan.approval_template, "stabilize"
    )
    result = execute_lifecycle_transition(
        replace(
            ready,
            dry_run=False,
            approval_path=approval_path,
            approval_sha256=approval_sha,
        ),
        client=harness.client,
    )
    verified = load_verified_deployment_pointer(
        harness.deployment_root, client=harness.client
    )
    assert result.status == "stable"
    assert verified["pointer"]["generation"] == 3
    assert verified["state"]["lifecycle_status"] == "stable"
    assert harness.client.aliases == {"champion": "2", "stable": "2"}


def test_full_v2_lifecycle_rolls_back_only_to_promotion_fixed_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _prepare_probationary(tmp_path, monkeypatch)
    assert harness.promotion_result is not None
    assert harness.bootstrap_result is not None
    expected = _current_expected(
        harness, candidate=None, champion="2", stable="1"
    )
    config = LifecycleConfig(
        action="rollback",
        deployment_root=harness.deployment_root,
        registry_lock_root=harness.registry_lock_root,
        registered_model_name=REGISTERED_MODEL,
        tracking_uri="file:synthetic-mlruns",
        expected=expected,
        promotion_receipt=harness.promotion_result.receipt_path,
        expected_rollback_state_id=(
            harness.bootstrap_result.state_manifest_path.parent.name
        ),
        dry_run=True,
        now_utc=datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc),
    )
    plan = plan_lifecycle_transition(config, client=harness.client)
    assert (
        plan.evidence["rollback_target_state_id"]
        == harness.bootstrap_result.state_manifest_path.parent.name
    )
    approval_path, approval_sha = _approval(
        harness, plan.approval_template, "rollback"
    )
    result = execute_lifecycle_transition(
        replace(
            config,
            dry_run=False,
            approval_path=approval_path,
            approval_sha256=approval_sha,
        ),
        client=harness.client,
    )
    verified = load_verified_deployment_pointer(
        harness.deployment_root, client=harness.client
    )
    assert result.status == "rolled_back"
    assert verified["pointer"]["generation"] == 3
    assert verified["state"]["registry"]["model_version"] == "1"
    assert harness.client.aliases == {"champion": "1", "stable": "1"}


def test_bootstrap_rejects_missing_or_modified_manual_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _new_harness(tmp_path, monkeypatch)
    dry = _bootstrap_config(harness)
    plan = bootstrap_v2_deployment(
        dry,
        client=harness.client,
        mlflow_module=harness.mlflow,
    ).plan

    with pytest.raises(RetrainingDeploymentError, match="approval"):
        bootstrap_v2_deployment(
            replace(dry, dry_run=False),
            client=harness.client,
            mlflow_module=harness.mlflow,
        )

    approval_path, approval_sha = _approval(
        harness, plan.approval_template, "invalid-bootstrap"
    )
    approval_path.write_text(
        approval_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(RetrainingDeploymentError, match="checksum"):
        bootstrap_v2_deployment(
            replace(
                dry,
                dry_run=False,
                approval_path=approval_path,
                approval_sha256=approval_sha,
            ),
            client=harness.client,
            mlflow_module=harness.mlflow,
        )

    assert harness.client.versions == {}
    assert not harness.deployment_root.exists()


@pytest.mark.parametrize("divergence", ["state_hash", "champion_alias"])
def test_alias_or_hash_divergence_blocks_batch_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    divergence: str,
) -> None:
    harness = _new_harness(tmp_path, monkeypatch)
    _execute_bootstrap(harness)
    if divergence == "state_hash":
        assert harness.bootstrap_result is not None
        harness.bootstrap_result.state_manifest_path.write_text(
            harness.bootstrap_result.state_manifest_path.read_text(
                encoding="utf-8"
            )
            + " ",
            encoding="utf-8",
        )
    else:
        harness.client.aliases["champion"] = "99"
    orchestration_root = tmp_path / "blocked-orchestration"
    stage_calls = 0

    def runner(command: Sequence[str], _timeout: int) -> Mapping[str, Any]:
        nonlocal stage_calls
        stage_calls += 1
        try:
            deployment_runtime.verify_active_model_era(
                harness.deployment_root,
                harness.reference_bundle,
                calibration_dir=harness.bootstrap_calibration,
                client=harness.client,
            )
        except Exception as exc:
            raise BatchOrchestrationError(str(exc)) from exc
        raise AssertionError(f"Unexpected command after divergence: {command}")

    with pytest.raises(BatchOrchestrationError):
        run_batch(
            BatchConfig(
                model_bundle=harness.reference_bundle,
                calibration_dir=harness.bootstrap_calibration,
                deployment_root=harness.deployment_root,
                through_date=BOOTSTRAP_DAY,
                source_store_root=tmp_path / "source-store",
                monitoring_store_root=tmp_path / "monitoring",
                orchestration_root=orchestration_root,
                no_source_refresh=True,
                now_utc=BOOTSTRAP_NOW,
            ),
            runner=runner,
        )

    assert stage_calls == 1
    assert not orchestration_root.exists()


def test_failure_before_pointer_commit_restores_aliases_and_records_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _new_harness(tmp_path, monkeypatch)
    _execute_bootstrap(harness)
    _register_candidate(harness)
    config = _promotion_config(harness)
    plan = plan_lifecycle_transition(config, client=harness.client)
    approval_path, approval_sha = _approval(
        harness, plan.approval_template, "failed-promote"
    )
    approved = replace(
        config,
        dry_run=False,
        approval_path=approval_path,
        approval_sha256=approval_sha,
    )
    pointer_path = harness.deployment_root / "state" / "current.json"
    pointer_before = pointer_path.read_bytes()
    aliases_before = dict(harness.client.aliases)
    monkeypatch.setattr(
        lifecycle,
        "_publish_pointer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic pre-commit failure")
        ),
    )

    with pytest.raises(
        RetrainingLifecycleError, match="before pointer publication"
    ):
        execute_lifecycle_transition(approved, client=harness.client)

    reconciliation_paths = list(
        (harness.deployment_root / "reconciliation").glob("*.json")
    )
    assert pointer_path.read_bytes() == pointer_before
    assert harness.client.aliases == aliases_before
    assert len(reconciliation_paths) == 1
    reconciliation = json.loads(
        reconciliation_paths[0].read_text(encoding="utf-8")
    )
    assert reconciliation["pointer_published"] is False
    assert reconciliation["automatic_rollback_attempted"] is False


def test_v1_artifacts_keep_the_serving_contract_sha256() -> None:
    project_root = Path(__file__).resolve().parents[1]
    validate_v1_source_contract(mode="metadata")
    contract = load_serving_contract(verify_files=True)
    for record in contract["targets"].values():
        for artifact in (record["model"], record["scaler_x"], record["scaler_y"]):
            path = project_root / artifact["path"]
            assert sha256_file(path) == artifact["sha256"]
