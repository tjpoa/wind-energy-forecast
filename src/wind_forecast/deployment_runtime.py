"""Fail-closed runtime binding between monitoring and the active deployment."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from wind_forecast.manifests import sha256_file


MODEL_ERA_SCHEMA = "wind_forecast.monitoring_model_era.v1"


class DeploymentRuntimeError(RuntimeError):
    """Raised when runtime artifacts do not match the active deployment."""


def verify_active_model_era(
    deployment_root: str | Path,
    model_bundle: str | Path | None = None,
    *,
    calibration_dir: str | Path | None = None,
    client: Any | None = None,
    mlflow_module: Any | None = None,
) -> dict[str, Any]:
    """Verify pointer, aliases and explicit artifacts, then describe one era."""
    from wind_forecast.retraining_deployment import (
        RetrainingDeploymentError,
        load_exact_v2_bundle,
        load_verified_deployment_pointer,
    )

    try:
        verified = load_verified_deployment_pointer(
            deployment_root,
            client=client,
            mlflow_module=mlflow_module,
        )
    except RetrainingDeploymentError as exc:
        raise DeploymentRuntimeError(str(exc)) from exc

    state = verified["state"]
    lifecycle_artifacts = state.get("artifacts") or {}
    if model_bundle is None:
        bundle_ref = lifecycle_artifacts.get("bundle") or {}
        relative = str(bundle_ref.get("path") or "")
        if not relative:
            raise DeploymentRuntimeError(
                "Generation-one deployment requires an explicit model bundle."
            )
        model_bundle = Path(deployment_root) / relative
    try:
        if (Path(model_bundle) / "bundle_manifest.json").is_file():
            from wind_forecast.monitoring import (
                MonitoringError,
                validate_monitoring_model_bundle,
            )

            try:
                bundle = validate_monitoring_model_bundle(model_bundle)
            except MonitoringError as exc:
                raise DeploymentRuntimeError(str(exc)) from exc
        else:
            bundle = load_exact_v2_bundle(model_bundle)
    except RetrainingDeploymentError as exc:
        raise DeploymentRuntimeError(str(exc)) from exc
    pins = dict(state["pins"])
    observed = {
        "bundle_sha256": bundle["bundle_sha256"],
        "model_sha256": str(bundle["model_manifest"]["model_sha256"]),
        "dataset_sha256": str(bundle["dataset_manifest"]["sha256"]),
        "feature_schema_sha256": str(
            bundle["model_manifest"]["feature_schema_sha256"]
        ),
    }
    for name, value in observed.items():
        if pins.get(name) != value:
            raise DeploymentRuntimeError(
                f"Explicit model bundle {name} differs from active deployment."
            )

    if calibration_dir is None:
        calibration_ref = lifecycle_artifacts.get("calibration") or {}
        relative = str(calibration_ref.get("path") or "")
        if relative:
            calibration_dir = Path(deployment_root) / relative
    calibration: Mapping[str, Any] | None = None
    if calibration_dir is not None:
        from wind_forecast.monitoring_reporting import (
            MonitoringReportingError,
            load_monitoring_calibration,
        )

        calibration_path = Path(calibration_dir)
        try:
            calibration = load_monitoring_calibration(calibration_path)
        except MonitoringReportingError as exc:
            raise DeploymentRuntimeError(str(exc)) from exc
        calibration_sha = sha256_file(calibration_path / "calibration.json")
        if pins.get("calibration_sha256") != calibration_sha:
            raise DeploymentRuntimeError(
                "Explicit monitoring calibration differs from active deployment."
            )
        expected_calibration = state["calibration"]
        if (
            calibration.get("calibration_id")
            != expected_calibration["calibration_id"]
            or calibration.get("reference_id")
            != expected_calibration["reference_id"]
        ):
            raise DeploymentRuntimeError(
                "Calibration/reference identities differ from active deployment."
            )

    pointer_path = Path(verified["pointer_path"])
    state_path = Path(verified["state_manifest_path"])
    receipt_path = Path(verified["receipt_path"])
    body = {
        "schema_version": MODEL_ERA_SCHEMA,
        "association_kind": "active_deployment",
        "deployment": {
            "deployment_id": state["deployment_id"],
            "deployment_state_id": state["deployment_state_id"],
            "generation": state["generation"],
            "pointer_sha256": sha256_file(pointer_path),
            "state_manifest_sha256": sha256_file(state_path),
            "authorizing_receipt_sha256": sha256_file(receipt_path),
        },
        "registry": {
            "registered_model_name": state["registry"]["registered_model_name"],
            "model_version": state["registry"]["model_version"],
            "run_id": state["registry"]["run_id"],
            "model_uri": state["registry"]["model_uri"],
        },
        "expected_aliases": dict(state["expected_aliases"]),
        "cutoffs": dict(state["cutoffs"]),
        "pins": pins,
        "calibration": dict(state["calibration"]),
        "monitoring": dict(state["monitoring"]),
    }
    era_id = _identifier("monitoring_model_era", body)
    return {"model_era_id": era_id, **body}


def same_model_era(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two independently verified runtime snapshots are identical."""
    return (
        left.get("model_era_id") == right.get("model_era_id")
        and left.get("deployment") == right.get("deployment")
        and left.get("registry") == right.get("registry")
        and left.get("pins") == right.get("pins")
    )


def _identifier(kind: str, body: Mapping[str, Any]) -> str:
    return sha256(
        _canonical({"record_type": kind, "payload": body})
    ).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "DeploymentRuntimeError",
    "MODEL_ERA_SCHEMA",
    "same_model_era",
    "verify_active_model_era",
]
