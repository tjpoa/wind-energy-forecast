from __future__ import annotations

from pathlib import Path

import pytest

import wind_forecast.retraining_deployment as deployment
from wind_forecast.deployment_runtime import (
    DeploymentRuntimeConflictError,
    DeploymentRuntimeError,
    DeploymentRuntimeNotInitializedError,
    DeploymentRuntimeUnavailableError,
    verify_active_model_era,
)
from wind_forecast.retraining_deployment import (
    RetrainingDeploymentConflictError,
    RetrainingDeploymentNotInitializedError,
    RetrainingDeploymentUnavailableError,
)


def _verified(tmp_path: Path) -> dict:
    paths = {}
    for name in ("pointer", "state", "receipt"):
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"name":"{name}"}}', encoding="utf-8")
        paths[name] = path
    state = {
        "deployment_id": "d" * 64,
        "deployment_state_id": "s" * 64,
        "generation": 1,
        "registry": {
            "registered_model_name": "wind-v2",
            "model_version": "7",
            "run_id": "run-7",
            "model_uri": "models:/wind-v2/7",
        },
        "expected_aliases": {
            "candidate": None,
            "champion": "7",
            "stable": "7",
        },
        "pins": {
            "bundle_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "ledger_sha256": "c" * 64,
            "model_sha256": "1" * 64,
            "dataset_sha256": "2" * 64,
            "feature_schema_sha256": "3" * 64,
        },
        "calibration": {"calibration_id": "cal-7", "reference_id": "ref-7"},
        "monitoring": {
            "ledger_model_snapshot_id": "4" * 64,
            "ledger_state_sha256": "c" * 64,
        },
        "cutoffs": {
            "fit_cutoff": "2024-12-31",
            "activation_cutoff": "2026-01-01",
        },
    }
    return {
        "state": state,
        "pointer_path": str(paths["pointer"]),
        "state_manifest_path": str(paths["state"]),
        "receipt_path": str(paths["receipt"]),
    }


def _bundle() -> dict:
    return {
        "bundle_sha256": "a" * 64,
        "model_manifest": {
            "model_sha256": "1" * 64,
            "feature_schema_sha256": "3" * 64,
        },
        "dataset_manifest": {"sha256": "2" * 64},
    }


def test_runtime_snapshot_is_deterministic_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        deployment,
        "load_verified_deployment_pointer",
        lambda *_args, **_kwargs: _verified(tmp_path),
    )
    monkeypatch.setattr(
        deployment,
        "load_exact_v2_bundle",
        lambda *_args, **_kwargs: _bundle(),
    )

    first = verify_active_model_era(tmp_path / "deployment", tmp_path / "bundle")
    second = verify_active_model_era(tmp_path / "deployment", tmp_path / "bundle")

    assert first == second
    assert len(first["model_era_id"]) == 64
    assert first["deployment"]["deployment_id"] == "d" * 64
    assert first["registry"]["model_version"] == "7"
    assert first["cutoffs"]["fit_cutoff"] == "2024-12-31"
    assert set(first["pins"]) == {
        "bundle_sha256",
        "calibration_sha256",
        "ledger_sha256",
        "model_sha256",
        "dataset_sha256",
        "feature_schema_sha256",
    }


def test_runtime_metadata_is_opt_in_and_does_not_change_model_era_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        deployment,
        "load_verified_deployment_pointer",
        lambda *_args, **_kwargs: _verified(tmp_path),
    )
    bundle = _bundle()
    bundle["model_manifest"]["model_type"] = "RandomForestRegressor"
    bundle["dataset_manifest"].update(
        {
            "dataset_version": "v2",
            "transformation_version": "transform-v2",
        }
    )
    monkeypatch.setattr(
        deployment,
        "load_exact_v2_bundle",
        lambda *_args, **_kwargs: bundle,
    )

    regular = verify_active_model_era(tmp_path / "deployment", tmp_path / "bundle")
    metadata = verify_active_model_era(
        tmp_path / "deployment",
        tmp_path / "bundle",
        include_runtime_metadata=True,
    )

    assert "_runtime_metadata" not in regular
    assert metadata["model_era_id"] == regular["model_era_id"]
    assert metadata["_runtime_metadata"] == {
        "model_type": "RandomForestRegressor",
        "dataset_version": "v2",
        "transformation_version": "transform-v2",
    }


def test_runtime_rejects_explicit_bundle_divergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        deployment,
        "load_verified_deployment_pointer",
        lambda *_args, **_kwargs: _verified(tmp_path),
    )
    mismatched = _bundle()
    mismatched["bundle_sha256"] = "f" * 64
    monkeypatch.setattr(
        deployment,
        "load_exact_v2_bundle",
        lambda *_args, **_kwargs: mismatched,
    )

    with pytest.raises(DeploymentRuntimeError, match="bundle_sha256"):
        verify_active_model_era(tmp_path / "deployment", tmp_path / "bundle")


@pytest.mark.parametrize(
    ("source_error", "runtime_error"),
    (
        (
            RetrainingDeploymentNotInitializedError("not initialized"),
            DeploymentRuntimeNotInitializedError,
        ),
        (
            RetrainingDeploymentUnavailableError("unavailable"),
            DeploymentRuntimeUnavailableError,
        ),
        (
            RetrainingDeploymentConflictError("conflict"),
            DeploymentRuntimeConflictError,
        ),
    ),
)
def test_runtime_preserves_deployment_error_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_error: Exception,
    runtime_error: type[Exception],
) -> None:
    monkeypatch.setattr(
        deployment,
        "load_verified_deployment_pointer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(source_error),
    )

    with pytest.raises(runtime_error):
        verify_active_model_era(tmp_path / "deployment")
