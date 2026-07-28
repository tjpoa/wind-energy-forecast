"""One-time, fail-closed bootstrap of the accepted v2 deployment pointer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from wind_forecast.manifests import sha256_file
from wind_forecast.monitoring import (
    MonitoringError,
    load_verified_monitoring_state,
    validate_monitoring_model_bundle,
)
from wind_forecast.monitoring_reporting import (
    MonitoringReportingError,
    load_monitoring_calibration,
)
from wind_forecast.retraining_policy import (
    ActiveDeploymentPointer,
    RetrainingContractError,
)
from wind_forecast.retraining_registry import (
    RetrainingRegistryError,
    acquire_registry_lock,
    release_registry_lock,
)
from wind_forecast.tracking import DEFAULT_REGISTERED_MODEL_NAME, _load_mlflow


APPROVAL_SCHEMA = "wind_forecast.bootstrap_approval.v1"
RECEIPT_SCHEMA = "wind_forecast.bootstrap_receipt.v1"
DEPLOYMENT_STATE_SCHEMA = "wind_forecast.deployment_state.v1"
RECONCILIATION_SCHEMA = "wind_forecast.bootstrap_reconciliation.v1"
POINTER_RELATIVE_PATH = Path("state/current.json")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALIASES = ("candidate", "champion", "stable")
_BUNDLE_FILES = {
    "dataset_manifest.json",
    "environment.json",
    "leakage_audit.json",
    "metrics.json",
    "mlflow_receipt.json",
    "mlflow_reload_validation.json",
    "model.joblib",
    "model_comparison.png",
    "model_manifest.json",
    "reference_decision.json",
    "reload_sample.csv",
    "run_summary.json",
    "test_predictions.csv",
    "validation_predictions.csv",
}


class RetrainingDeploymentError(RuntimeError):
    """Raised when deployment bootstrap cannot safely proceed."""


class RetrainingDeploymentReconciliationError(RuntimeError):
    """Raised when a post-mutation failure cannot be safely hidden or undone."""


class _PointerPublishedCleanupError(RuntimeError):
    """Raised when the pointer exists but its prepared file could not be removed."""


@dataclass(frozen=True)
class DeploymentBootstrapConfig:
    """Explicit inputs and optimistic assertions for the one-time bootstrap."""

    model_bundle: Path
    calibration_dir: Path
    monitoring_store_root: Path
    deployment_root: Path
    registry_lock_root: Path
    registered_model_name: str
    tracking_uri: str
    approval_path: Path | None = None
    approval_sha256: str | None = None
    expect_no_deployment_pointer: bool = False
    expect_no_v2_registry_state: bool = False
    dry_run: bool = False
    now_utc: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "model_bundle",
            "calibration_dir",
            "monitoring_store_root",
            "deployment_root",
            "registry_lock_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.approval_path is not None:
            object.__setattr__(self, "approval_path", Path(self.approval_path))
        for name in ("registered_model_name", "tracking_uri"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise RetrainingDeploymentError(f"{name} must be explicit.")
        if self.registered_model_name == DEFAULT_REGISTERED_MODEL_NAME:
            raise RetrainingDeploymentError(
                "The legacy DEFAULT_REGISTERED_MODEL_NAME is forbidden for v2."
            )
        if self.approval_sha256 is not None and not SHA256_PATTERN.fullmatch(
            self.approval_sha256
        ):
            raise RetrainingDeploymentError(
                "approval_sha256 must be a lowercase SHA-256 digest."
            )
        for protected in (
            self.model_bundle,
            self.calibration_dir,
            self.monitoring_store_root,
        ):
            if _paths_overlap(self.deployment_root, protected):
                raise RetrainingDeploymentError(
                    "Deployment output must not overlap immutable input evidence."
                )
            if _paths_overlap(self.registry_lock_root, protected):
                raise RetrainingDeploymentError(
                    "Registry lock root must not overlap immutable input evidence."
                )
        if self.now_utc is not None:
            _utc_text(self.now_utc)


@dataclass(frozen=True)
class DeploymentBootstrapPlan:
    """Read-only, fully verified bootstrap plan and approval template."""

    status: str
    deployment_id: str
    run_id: str
    model_uri: str
    registered_model_name: str
    fit_cutoff: str
    activation_cutoff: str
    model_snapshot_id: str
    calibration_id: str
    reference_id: str
    pins: Mapping[str, str]
    approval_template: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeploymentBootstrapResult:
    """Result of a dry-run plan or successful generation-one bootstrap."""

    status: str
    plan: DeploymentBootstrapPlan
    model_version: str | None = None
    receipt_path: Path | None = None
    state_manifest_path: Path | None = None
    pointer_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plan"] = self.plan.to_dict()
        for name in ("receipt_path", "state_manifest_path", "pointer_path"):
            value = getattr(self, name)
            payload[name] = None if value is None else str(value)
        return payload


def plan_v2_deployment_bootstrap(
    config: DeploymentBootstrapConfig,
    *,
    client: Any | None = None,
    mlflow_module: Any | None = None,
) -> DeploymentBootstrapPlan:
    """Verify every local/MLflow precondition without locks or writes."""
    _require_expectations(config)
    _assert_pristine_deployment_root(config.deployment_root)
    bundle = _load_exact_bundle(config.model_bundle)
    calibration = _load_calibration(config.calibration_dir)
    ledger = _load_ledger(config.monitoring_store_root)
    snapshot = _load_ledger_snapshot(config.monitoring_store_root, ledger)
    model_manifest = bundle["model_manifest"]
    dataset_manifest = bundle["dataset_manifest"]
    if snapshot.get("model_snapshot_id") != ledger.get("model_snapshot_id"):
        raise RetrainingDeploymentError(
            "Phase 9 model snapshot and current ledger identity differ."
        )
    if (
        (snapshot.get("model") or {}).get("model_sha256")
        != model_manifest["model_sha256"]
        or snapshot.get("feature_schema_sha256")
        != model_manifest["feature_schema_sha256"]
        or (snapshot.get("dataset") or {}).get("dataset_sha256")
        != dataset_manifest["sha256"]
    ):
        raise RetrainingDeploymentError(
            "Phase 9 model snapshot does not correspond to the accepted bundle."
        )
    if (
        calibration.get("reference_id")
        != (calibration.get("_reference_manifest") or {}).get("reference_id")
    ):
        raise RetrainingDeploymentError(
            "Calibration and reference identities differ."
        )

    mlflow, client = _mlflow_client(
        config,
        client=client,
        mlflow_module=mlflow_module,
    )
    _assert_no_registry_state(client, config.registered_model_name)
    mlflow_evidence = _verify_mlflow(
        config,
        bundle=bundle,
        client=client,
        mlflow=mlflow,
    )
    fit_cutoff = str(dataset_manifest["splits"]["validation"]["end"])
    activation_cutoff = _date_text(
        ledger.get("activation_date"), "Phase 9 activation cutoff"
    )
    pins = {
        "bundle_sha256": bundle["bundle_sha256"],
        "calibration_sha256": sha256_file(
            config.calibration_dir / "calibration.json"
        ),
        "ledger_sha256": sha256_file(
            config.monitoring_store_root / "state" / "current.json"
        ),
        "model_sha256": str(model_manifest["model_sha256"]),
        "dataset_sha256": str(dataset_manifest["sha256"]),
        "feature_schema_sha256": str(model_manifest["feature_schema_sha256"]),
    }
    deployment_id = _identifier(
        "bootstrap_deployment",
        {
            "registered_model_name": config.registered_model_name,
            "run_id": mlflow_evidence["run_id"],
            "model_uri": mlflow_evidence["model_uri"],
            "model_snapshot_id": ledger["model_snapshot_id"],
            "pins": pins,
        },
    )
    template = {
        "schema_version": APPROVAL_SCHEMA,
        "approved_by": "<operator>",
        "approved_at_utc": "<YYYY-MM-DDTHH:MM:SSZ>",
        "note": "<manual bootstrap reason>",
        "bootstrap_exception": True,
        "deployment_root": str(config.deployment_root.resolve()),
        "registered_model_name": config.registered_model_name,
        "run_id": mlflow_evidence["run_id"],
        "model_uri": mlflow_evidence["model_uri"],
        "expected_bundle_sha256": pins["bundle_sha256"],
        "expected_calibration_sha256": pins["calibration_sha256"],
        "expected_ledger_sha256": pins["ledger_sha256"],
    }
    if config.approval_path is not None or config.approval_sha256 is not None:
        _load_approval(config, template)
    return DeploymentBootstrapPlan(
        status="planned",
        deployment_id=deployment_id,
        run_id=mlflow_evidence["run_id"],
        model_uri=mlflow_evidence["model_uri"],
        registered_model_name=config.registered_model_name,
        fit_cutoff=fit_cutoff,
        activation_cutoff=activation_cutoff,
        model_snapshot_id=str(ledger["model_snapshot_id"]),
        calibration_id=str(calibration["calibration_id"]),
        reference_id=str(calibration["reference_id"]),
        pins=pins,
        approval_template=template,
    )


def bootstrap_v2_deployment(
    config: DeploymentBootstrapConfig,
    *,
    client: Any | None = None,
    mlflow_module: Any | None = None,
) -> DeploymentBootstrapResult:
    """Create generation one, initialize stable/champion, and publish pointer."""
    plan = plan_v2_deployment_bootstrap(
        config,
        client=client,
        mlflow_module=mlflow_module,
    )
    if config.dry_run:
        return DeploymentBootstrapResult(status="planned", plan=plan)
    approval, approval_sha = _load_approval(config, plan.approval_template)
    mlflow, client = _mlflow_client(
        config,
        client=client,
        mlflow_module=mlflow_module,
    )
    lock_path = acquire_registry_lock(
        config.registry_lock_root,
        config.registered_model_name,
        {
            "action": "bootstrap_v2_deployment",
            "deployment_id": plan.deployment_id,
            "run_id": plan.run_id,
        },
    )
    try:
        _assert_pristine_deployment_root(
            config.deployment_root,
            ignored=(lock_path.parent,)
            if _is_within(lock_path.parent, config.deployment_root)
            else (),
        )
        _assert_no_registry_state(client, config.registered_model_name)
        version = mlflow.register_model(
            model_uri=plan.model_uri,
            name=config.registered_model_name,
        )
        version_text = str(getattr(version, "version", "") or "")
        if not version_text:
            raise RetrainingDeploymentError(
                "MLflow did not return the created Registry version."
            )
        pointer_published = False
        try:
            if version_text != "1":
                raise RetrainingDeploymentError(
                    "Bootstrap created a non-initial Registry version."
                )
            _verify_created_version(
                client,
                config.registered_model_name,
                version_text,
                plan.run_id,
            )
            tags = _bootstrap_tags(plan)
            for key, value in tags.items():
                client.set_model_version_tag(
                    config.registered_model_name,
                    version_text,
                    key,
                    value,
                )
            _verify_version_tags(
                client,
                config.registered_model_name,
                version_text,
                tags,
            )
            receipt, receipt_path = _seal_bootstrap_receipt(
                config,
                plan,
                approval,
                approval_sha,
                version_text,
            )
            state, state_path = _seal_deployment_state(
                config,
                plan,
                receipt,
                receipt_path,
                version_text,
            )
            _assert_pointer_absent(config.deployment_root)
            _assert_aliases_absent(client, config.registered_model_name)
            _assert_only_created_registry_version(
                client,
                config.registered_model_name,
                version_text,
            )
            client.set_registered_model_alias(
                config.registered_model_name, "stable", version_text
            )
            _require_alias(
                client, config.registered_model_name, "stable", version_text
            )
            _assert_pointer_absent(config.deployment_root)
            _assert_only_created_registry_version(
                client,
                config.registered_model_name,
                version_text,
            )
            if _alias_version(
                client, config.registered_model_name, "candidate"
            ) is not None:
                raise RetrainingDeploymentError(
                    "Candidate appeared before champion initialization."
                )
            if _alias_version(
                client, config.registered_model_name, "champion"
            ) is not None:
                raise RetrainingDeploymentError(
                    "Champion appeared before bootstrap could initialize it."
                )
            client.set_registered_model_alias(
                config.registered_model_name, "champion", version_text
            )
            _require_alias(
                client, config.registered_model_name, "champion", version_text
            )
            if _alias_version(
                client, config.registered_model_name, "candidate"
            ) is not None:
                raise RetrainingDeploymentError(
                    "Bootstrap must not initialize candidate."
                )
            _assert_only_created_registry_version(
                client,
                config.registered_model_name,
                version_text,
            )
            pointer_path = _publish_pointer(config, plan, state, state_path)
            pointer_published = True
            load_verified_deployment_pointer(
                config.deployment_root,
                client=client,
            )
            return DeploymentBootstrapResult(
                status="bootstrapped",
                plan=plan,
                model_version=version_text,
                receipt_path=receipt_path,
                state_manifest_path=state_path,
                pointer_path=pointer_path,
            )
        except Exception as exc:
            if pointer_published or isinstance(
                exc, _PointerPublishedCleanupError
            ):
                _write_reconciliation_evidence(
                    config,
                    plan,
                    version_text,
                    exc,
                    pointer_published=True,
                    compensation_errors=(),
                )
                raise RetrainingDeploymentReconciliationError(
                    "Bootstrap pointer was published but post-verification failed; "
                    "the pointer and aliases were preserved for manual reconciliation."
                ) from exc
            _reconcile_failed_bootstrap(
                config,
                client,
                plan,
                version_text,
                exc,
            )
            raise RetrainingDeploymentError(
                "Bootstrap failed after version creation. Safe aliases were removed; "
                "the Registry version and immutable evidence were retained. Manual "
                "reconciliation is required before any retry."
            ) from exc
    finally:
        try:
            release_registry_lock(lock_path)
        except RetrainingRegistryError as exc:
            raise RetrainingDeploymentReconciliationError(str(exc)) from exc


def load_bootstrap_approval(path: str | Path) -> dict[str, Any]:
    """Load one strict declarative bootstrap approval."""
    payload = _read_json(Path(path))
    _validate_approval_payload(payload)
    return payload


def _validate_approval_payload(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "approved_by",
        "approved_at_utc",
        "note",
        "bootstrap_exception",
        "deployment_root",
        "registered_model_name",
        "run_id",
        "model_uri",
        "expected_bundle_sha256",
        "expected_calibration_sha256",
        "expected_ledger_sha256",
    }
    if set(payload) != expected or payload.get("schema_version") != APPROVAL_SCHEMA:
        raise RetrainingDeploymentError(
            "Bootstrap approval fields differ from strict v1."
        )
    for name in (
        "approved_by",
        "approved_at_utc",
        "note",
        "deployment_root",
        "registered_model_name",
        "run_id",
        "model_uri",
    ):
        _required_text(payload.get(name), f"approval {name}")
    _parse_utc(str(payload["approved_at_utc"]), "approval approved_at_utc")
    if payload.get("bootstrap_exception") is not True:
        raise RetrainingDeploymentError(
            "Bootstrap approval must explicitly set bootstrap_exception=true."
        )
    for name in (
        "expected_bundle_sha256",
        "expected_calibration_sha256",
        "expected_ledger_sha256",
    ):
        _sha256_text(payload.get(name), f"approval {name}")


def load_bootstrap_receipt(path: str | Path) -> dict[str, Any]:
    """Load and identity-check one immutable bootstrap receipt."""
    receipt_path = Path(path)
    payload = _read_json(receipt_path)
    expected = {
        "schema_version",
        "bootstrap_receipt_id",
        "bootstrap_exception",
        "deployment_id",
        "registered_model_name",
        "model_version",
        "run_id",
        "model_uri",
        "approval",
        "approval_sha256",
        "approval_payload_sha256",
        "pins",
        "expected_aliases",
        "executed_at_utc",
    }
    if set(payload) != expected or payload.get("schema_version") != RECEIPT_SCHEMA:
        raise RetrainingDeploymentError(
            "Bootstrap receipt fields differ from strict v1."
        )
    identifier = _required_identifier(
        payload.get("bootstrap_receipt_id"), "bootstrap_receipt_id"
    )
    body = {
        key: value
        for key, value in payload.items()
        if key != "bootstrap_receipt_id"
    }
    if identifier != _identifier("bootstrap_receipt", body):
        raise RetrainingDeploymentError("Bootstrap receipt identity is corrupt.")
    if receipt_path.parent.name != identifier:
        raise RetrainingDeploymentError(
            "Bootstrap receipt path and identity differ."
        )
    if payload.get("bootstrap_exception") is not True:
        raise RetrainingDeploymentError("Bootstrap receipt exception flag is invalid.")
    _required_identifier(payload.get("deployment_id"), "deployment_id")
    for name in (
        "registered_model_name",
        "model_version",
        "run_id",
        "model_uri",
    ):
        _required_text(payload.get(name), name)
    approval = payload.get("approval")
    if not isinstance(approval, dict):
        raise RetrainingDeploymentError("Embedded bootstrap approval is invalid.")
    _validate_approval_payload(approval)
    _parse_utc(str(payload.get("executed_at_utc") or ""), "executed_at_utc")
    _sha256_text(payload.get("approval_sha256"), "approval_sha256")
    approval_payload_sha = _sha256_text(
        payload.get("approval_payload_sha256"),
        "approval_payload_sha256",
    )
    if sha256(_canonical(approval)).hexdigest() != approval_payload_sha:
        raise RetrainingDeploymentError(
            "Embedded bootstrap approval payload checksum is invalid."
        )
    _validate_pins(payload.get("pins"))
    _validate_expected_aliases(payload.get("expected_aliases"))
    aliases = payload["expected_aliases"]
    if aliases["champion"] != payload["model_version"]:
        raise RetrainingDeploymentError(
            "Bootstrap receipt alias and version identities differ."
        )
    receipt_root = receipt_path.resolve().parents[2]
    approval_expected = {
        "deployment_root": str(receipt_root),
        "registered_model_name": payload["registered_model_name"],
        "run_id": payload["run_id"],
        "model_uri": payload["model_uri"],
        "expected_bundle_sha256": payload["pins"]["bundle_sha256"],
        "expected_calibration_sha256": payload["pins"][
            "calibration_sha256"
        ],
        "expected_ledger_sha256": payload["pins"]["ledger_sha256"],
    }
    if any(approval.get(key) != value for key, value in approval_expected.items()):
        raise RetrainingDeploymentError(
            "Bootstrap receipt and embedded approval pins differ."
        )
    return payload


def load_verified_deployment_pointer(
    deployment_root: str | Path,
    *,
    client: Any | None = None,
    mlflow_module: Any | None = None,
) -> dict[str, Any]:
    """Verify pointer, state, receipt checksums/IDs, and Registry aliases."""
    root = Path(deployment_root)
    pointer_path = root / POINTER_RELATIVE_PATH
    try:
        pointer = ActiveDeploymentPointer.from_dict(_read_json(pointer_path))
    except RetrainingContractError as exc:
        raise RetrainingDeploymentError(str(exc)) from exc
    state_path = _resolve_relative_file(root, pointer.state_manifest_path)
    if sha256_file(state_path) != pointer.state_manifest_sha256:
        raise RetrainingDeploymentError("Deployment state checksum is invalid.")
    state = _load_deployment_state(state_path)
    if (
        state["deployment_state_id"] != pointer.deployment_state_id
        or state["deployment_id"] != pointer.deployment_id
        or state["generation"] != pointer.generation
    ):
        raise RetrainingDeploymentError(
            "Deployment pointer and immutable state identities differ."
        )
    receipt_ref = state["authorizing_receipt"]
    receipt_path = _resolve_relative_file(root, receipt_ref["path"])
    if sha256_file(receipt_path) != receipt_ref["sha256"]:
        raise RetrainingDeploymentError("Authorizing receipt checksum is invalid.")
    receipt = load_bootstrap_receipt(receipt_path)
    if (
        receipt["bootstrap_receipt_id"] != receipt_ref["bootstrap_receipt_id"]
        or receipt["deployment_id"] != state["deployment_id"]
        or receipt["registered_model_name"]
        != state["registry"]["registered_model_name"]
        or receipt["model_version"] != state["registry"]["model_version"]
        or receipt["run_id"] != state["registry"]["run_id"]
        or receipt["model_uri"] != state["registry"]["model_uri"]
        or receipt["pins"] != state["pins"]
        or receipt["expected_aliases"] != state["expected_aliases"]
    ):
        raise RetrainingDeploymentError(
            "Deployment state and authorizing receipt differ."
        )
    if client is None:
        mlflow = mlflow_module or _load_mlflow()
        tracking_uri = state["registry"]["tracking_uri"]
        if hasattr(mlflow, "set_tracking_uri"):
            mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
    name = state["registry"]["registered_model_name"]
    for alias in ALIASES:
        expected = state["expected_aliases"][alias]
        actual = _alias_version(client, name, alias)
        if actual != expected:
            raise RetrainingDeploymentError(
                f"Registry alias {alias} differs from deployment state."
            )
    return {
        "pointer": pointer.to_dict(),
        "state": state,
        "receipt": receipt,
        "pointer_path": str(pointer_path),
        "state_manifest_path": str(state_path),
        "receipt_path": str(receipt_path),
    }


def load_exact_v2_bundle(root: str | Path) -> dict[str, Any]:
    """Public strict loader used to bind explicit runtime artifacts."""
    return _load_exact_bundle(Path(root))


def _load_exact_bundle(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RetrainingDeploymentError(
            "Accepted v2 bundle must be a real directory."
        )
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if any(path.is_dir() or path.is_symlink() for path in root.iterdir()):
        raise RetrainingDeploymentError(
            "Accepted v2 bundle contains a directory or symbolic link."
        )
    if actual != _BUNDLE_FILES:
        raise RetrainingDeploymentError(
            "Accepted v2 bundle file set differs from the bootstrap contract."
        )
    try:
        bundle = validate_monitoring_model_bundle(root)
    except MonitoringError as exc:
        raise RetrainingDeploymentError(str(exc)) from exc
    file_hashes = {
        name: sha256_file(root / name)
        for name in sorted(_BUNDLE_FILES)
    }
    summary_hashes = dict(bundle["summary"].get("artifact_sha256") or {})
    summary_pinned = _BUNDLE_FILES - {
        "run_summary.json",
        "mlflow_receipt.json",
        "mlflow_reload_validation.json",
    }
    if set(summary_hashes) != summary_pinned or any(
        summary_hashes[name] != file_hashes[name]
        for name in summary_pinned
    ):
        raise RetrainingDeploymentError(
            "V2 training summary does not pin every bundle artifact checksum."
        )
    reload_validation = _read_json(root / "mlflow_reload_validation.json")
    expected_reload_fields = {
        "schema_version",
        "model_uri",
        "row_count",
        "predictions_equivalent",
        "rtol",
        "atol",
        "max_absolute_difference",
    }
    if (
        set(reload_validation) != expected_reload_fields
        or reload_validation.get("schema_version")
        != "wind_forecast.mlflow_reload_validation.v1"
        or reload_validation.get("predictions_equivalent") is not True
    ):
        raise RetrainingDeploymentError(
            "MLflow reload validation evidence is invalid."
        )
    receipt = _read_json(root / "mlflow_receipt.json")
    if set(receipt) != {
        "experiment_id",
        "model_uri",
        "run_id",
        "tracking_uri",
    }:
        raise RetrainingDeploymentError("Local MLflow receipt fields are invalid.")
    _required_text(receipt.get("experiment_id"), "MLflow receipt experiment_id")
    _required_text(receipt.get("run_id"), "MLflow receipt run_id")
    _required_text(receipt.get("model_uri"), "MLflow receipt model_uri")
    _required_text(receipt.get("tracking_uri"), "MLflow receipt tracking_uri")
    bundle["mlflow_receipt"] = receipt
    bundle["mlflow_reload_validation"] = reload_validation
    bundle["bundle_files"] = file_hashes
    bundle["bundle_sha256"] = sha256(
        b"accepted_v2_bundle:" + _canonical(file_hashes)
    ).hexdigest()
    return bundle


def _load_calibration(root: Path) -> dict[str, Any]:
    try:
        return load_monitoring_calibration(root)
    except (MonitoringReportingError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RetrainingDeploymentError(
            "Monitoring calibration/reference evidence is invalid."
        ) from exc


def _load_ledger(root: Path) -> dict[str, Any]:
    pointer = root / "state" / "current.json"
    if not pointer.is_file():
        raise RetrainingDeploymentError(
            "Verified Phase 9 ledger state/current.json is required."
        )
    try:
        ledger = load_verified_monitoring_state(root)
    except (MonitoringError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RetrainingDeploymentError("Phase 9 ledger is invalid.") from exc
    if ledger is None:
        raise RetrainingDeploymentError("Verified Phase 9 ledger is required.")
    return ledger


def _load_ledger_snapshot(
    root: Path,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot_id = _required_identifier(
        ledger.get("model_snapshot_id"), "Phase 9 model_snapshot_id"
    )
    return _read_json(
        root / "model_snapshots" / snapshot_id / "snapshot.json"
    )


def _verify_mlflow(
    config: DeploymentBootstrapConfig,
    *,
    bundle: Mapping[str, Any],
    client: Any,
    mlflow: Any,
) -> dict[str, str]:
    receipt = bundle["mlflow_receipt"]
    run_id = _required_text(receipt.get("run_id"), "MLflow receipt run_id")
    model_uri = _required_text(
        receipt.get("model_uri"), "MLflow receipt model_uri"
    )
    if receipt.get("tracking_uri") != config.tracking_uri:
        raise RetrainingDeploymentError(
            "MLflow receipt tracking URI differs from the explicit URI."
        )
    validation = bundle["mlflow_reload_validation"]
    if validation.get("model_uri") != model_uri:
        raise RetrainingDeploymentError(
            "MLflow reload evidence and receipt model URIs differ."
        )
    run = client.get_run(run_id)
    if str(getattr(run.info, "status", "")) != "FINISHED":
        raise RetrainingDeploymentError("MLflow source run is not FINISHED.")
    if str(getattr(run.info, "experiment_id", "")) != str(
        receipt["experiment_id"]
    ):
        raise RetrainingDeploymentError(
            "MLflow source run and receipt experiment identities differ."
        )
    params = dict(getattr(run.data, "params", {}) or {})
    tags = dict(getattr(run.data, "tags", {}) or {})
    model_manifest = bundle["model_manifest"]
    dataset = bundle["dataset_manifest"]
    expected_params = {
        "workflow": "train_v2_reference",
        "dataset_version": "v2",
        "dataset_sha256": str(dataset["sha256"]),
        "split_assignment_sha256": str(dataset["split_assignment_sha256"]),
        "feature_count": str(len(bundle["feature_names"])),
        "scaler_required": "False",
        "logged_model_uri": model_uri,
        "selected_model": str(bundle["summary"]["selected_model"]),
        "seed": str(model_manifest["parameters"]["random_state"]),
        "n_estimators": str(model_manifest["parameters"]["n_estimators"]),
    }
    for key, expected in expected_params.items():
        if str(params.get(key)) != expected:
            raise RetrainingDeploymentError(
                f"MLflow run parameter {key} differs from the accepted bundle."
            )
    expected_tags = {
        "forecast_contract": "historical_daily_hindcast",
        "reference_gate_passed": "True",
        "reference_status": "selected_not_promoted",
        "registry_used": "False",
        "automatic_promotion": "False",
    }
    for key, expected in expected_tags.items():
        if str(tags.get(key)) != expected:
            raise RetrainingDeploymentError(
                f"MLflow run tag {key} differs from the accepted v2 state."
            )
    info = mlflow.models.get_model_info(model_uri)
    if str(getattr(info, "run_id", "")) != run_id:
        raise RetrainingDeploymentError(
            "MLflow logged model and receipt run identities differ."
        )
    _validate_signature(getattr(info, "signature", None), bundle["feature_names"])
    sample = pd.read_csv(config.model_bundle / "reload_sample.csv")
    features = list(bundle["feature_names"])
    if sample.columns.tolist() != [*features, "Expected_Prediction"]:
        raise RetrainingDeploymentError(
            "V2 reload sample columns/order differ from the model contract."
        )
    expected = sample["Expected_Prediction"].to_numpy(dtype=float)
    logged = mlflow.pyfunc.load_model(model_uri)
    actual = np.asarray(logged.predict(sample[features]), dtype=float).reshape(-1)
    raw_model = joblib.load(config.model_bundle / "model.joblib")
    if (
        type(raw_model).__name__.lower()
        .replace("regressor", "")
        .replace("_", "")
        != str(model_manifest["model_type"]).replace("_", "")
        or raw_model.get_params(deep=True) != model_manifest["parameters"]
        or list(getattr(raw_model, "feature_names_in_", ())) != features
    ):
        raise RetrainingDeploymentError(
            "V2 model class, parameters, or ordered features differ."
        )
    raw = np.asarray(raw_model.predict(sample[features]), dtype=float).reshape(-1)
    recorded_row_count = validation.get("row_count")
    recorded_rtol = validation.get("rtol")
    recorded_atol = validation.get("atol")
    recorded_max_difference = validation.get("max_absolute_difference")
    observed_max_difference = float(
        np.max(np.abs(actual - expected)) if len(expected) else 0.0
    )
    if (
        actual.shape != expected.shape
        or raw.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(raw).all()
        or not np.allclose(actual, expected, rtol=1e-12, atol=1e-9)
        or not np.allclose(raw, expected, rtol=1e-12, atol=1e-9)
        or not np.allclose(actual, raw, rtol=1e-12, atol=1e-9)
        or recorded_row_count != len(expected)
        or recorded_rtol != 1e-12
        or recorded_atol != 1e-9
        or not isinstance(recorded_max_difference, (int, float))
        or isinstance(recorded_max_difference, bool)
        or not np.isfinite(float(recorded_max_difference))
        or not np.isclose(
            float(recorded_max_difference),
            observed_max_difference,
            rtol=1e-12,
            atol=1e-15,
        )
    ):
        raise RetrainingDeploymentError(
            "MLflow, raw model, and reload-sample predictions differ."
        )
    return {"run_id": run_id, "model_uri": model_uri}


def _mlflow_client(
    config: DeploymentBootstrapConfig,
    *,
    client: Any | None,
    mlflow_module: Any | None,
) -> tuple[Any, Any]:
    mlflow = mlflow_module or _load_mlflow()
    if hasattr(mlflow, "set_tracking_uri"):
        mlflow.set_tracking_uri(config.tracking_uri)
    return mlflow, client or mlflow.MlflowClient()


def _assert_no_registry_state(client: Any, name: str) -> None:
    registered = None
    try:
        registered = client.get_registered_model(name)
    except Exception as exc:
        if not _is_missing_resource(exc):
            raise
    if registered is not None:
        raise RetrainingDeploymentError(
            "The v2 registered-model already exists; bootstrap is forbidden."
        )
    versions = []
    try:
        versions = list(client.search_model_versions())
    except TypeError:
        versions = list(client.search_model_versions(""))
    except Exception as exc:
        if not _is_missing_resource(exc):
            raise
    if any(str(getattr(item, "name", "")) == name for item in versions):
        raise RetrainingDeploymentError(
            "A v2 Registry version already exists; bootstrap is forbidden."
        )
    _assert_aliases_absent(client, name)


def _assert_aliases_absent(client: Any, name: str) -> None:
    existing = {
        alias: _alias_version(client, name, alias)
        for alias in ALIASES
    }
    if any(value is not None for value in existing.values()):
        raise RetrainingDeploymentError(
            "V2 candidate, champion, and stable aliases must all be absent."
        )


def _assert_only_created_registry_version(
    client: Any,
    name: str,
    version: str,
) -> None:
    versions = _search_model_versions(client)
    observed = sorted(
        str(getattr(item, "version", ""))
        for item in versions
        if str(getattr(item, "name", "")) == name
    )
    if observed != [version]:
        raise RetrainingDeploymentError(
            "Registry versions changed during bootstrap."
        )


def _search_model_versions(client: Any) -> list[Any]:
    try:
        return list(client.search_model_versions())
    except TypeError:
        return list(client.search_model_versions(""))
    except Exception as exc:
        if _is_missing_resource(exc):
            return []
        raise


def _alias_version(client: Any, name: str, alias: str) -> str | None:
    try:
        value = client.get_model_version_by_alias(name, alias)
    except Exception as exc:
        if _is_missing_resource(exc):
            return None
        raise
    return str(value.version)


def _is_missing_resource(exc: Exception) -> bool:
    return isinstance(exc, LookupError) or getattr(exc, "error_code", None) in {
        "RESOURCE_DOES_NOT_EXIST",
        "INVALID_PARAMETER_VALUE",
    }


def _verify_created_version(
    client: Any,
    name: str,
    version: str,
    run_id: str,
) -> None:
    created = client.get_model_version(name, version)
    if (
        str(getattr(created, "version", "")) != version
        or str(getattr(created, "run_id", "")) != run_id
    ):
        raise RetrainingDeploymentError(
            "Created Registry version identity differs from the approved run."
        )


def _bootstrap_tags(plan: DeploymentBootstrapPlan) -> dict[str, str]:
    return {
        "bootstrap_exception": "true",
        "deployment_id": plan.deployment_id,
        "lifecycle_role": "bootstrap_champion_stable",
        "source_run_id": plan.run_id,
        "bundle_sha256": plan.pins["bundle_sha256"],
        "model_sha256": plan.pins["model_sha256"],
        "dataset_sha256": plan.pins["dataset_sha256"],
        "feature_schema_sha256": plan.pins["feature_schema_sha256"],
        "calibration_id": plan.calibration_id,
        "calibration_sha256": plan.pins["calibration_sha256"],
        "reference_id": plan.reference_id,
        "ledger_sha256": plan.pins["ledger_sha256"],
        "model_snapshot_id": plan.model_snapshot_id,
        "fit_cutoff": plan.fit_cutoff,
        "activation_cutoff": plan.activation_cutoff,
    }


def _verify_version_tags(
    client: Any,
    name: str,
    version: str,
    expected: Mapping[str, str],
) -> None:
    created = client.get_model_version(name, version)
    tags = dict(getattr(created, "tags", {}) or {})
    if any(str(tags.get(key)) != value for key, value in expected.items()):
        raise RetrainingDeploymentError(
            "Registry version checksum tags did not persist."
        )


def _seal_bootstrap_receipt(
    config: DeploymentBootstrapConfig,
    plan: DeploymentBootstrapPlan,
    approval: Mapping[str, Any],
    approval_sha: str,
    version: str,
) -> tuple[dict[str, Any], Path]:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "bootstrap_exception": True,
        "deployment_id": plan.deployment_id,
        "registered_model_name": plan.registered_model_name,
        "model_version": version,
        "run_id": plan.run_id,
        "model_uri": plan.model_uri,
        "approval": dict(approval),
        "approval_sha256": approval_sha,
        "approval_payload_sha256": sha256(
            _canonical(approval)
        ).hexdigest(),
        "pins": dict(plan.pins),
        "expected_aliases": {
            "candidate": None,
            "champion": version,
            "stable": version,
        },
        "executed_at_utc": _execution_time(config),
    }
    receipt_id = _identifier("bootstrap_receipt", body)
    payload = {"bootstrap_receipt_id": receipt_id, **body}
    target = (
        config.deployment_root
        / "receipts"
        / receipt_id
        / "receipt.json"
    )
    _seal_content_addressed_json(target, payload)
    load_bootstrap_receipt(target)
    return payload, target


def _seal_deployment_state(
    config: DeploymentBootstrapConfig,
    plan: DeploymentBootstrapPlan,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    version: str,
) -> tuple[dict[str, Any], Path]:
    receipt_relative = receipt_path.resolve().relative_to(
        config.deployment_root.resolve()
    ).as_posix()
    body = {
        "schema_version": DEPLOYMENT_STATE_SCHEMA,
        "generation": 1,
        "deployment_id": plan.deployment_id,
        "registry": {
            "tracking_uri": config.tracking_uri,
            "registered_model_name": plan.registered_model_name,
            "model_version": version,
            "run_id": plan.run_id,
            "model_uri": plan.model_uri,
        },
        "expected_aliases": {
            "candidate": None,
            "champion": version,
            "stable": version,
        },
        "pins": dict(plan.pins),
        "calibration": {
            "calibration_id": plan.calibration_id,
            "reference_id": plan.reference_id,
        },
        "monitoring": {
            "ledger_model_snapshot_id": plan.model_snapshot_id,
            "ledger_state_sha256": plan.pins["ledger_sha256"],
        },
        "cutoffs": {
            "fit_cutoff": plan.fit_cutoff,
            "activation_cutoff": plan.activation_cutoff,
        },
        "predecessor": None,
        "authorizing_receipt": {
            "bootstrap_receipt_id": receipt["bootstrap_receipt_id"],
            "path": receipt_relative,
            "sha256": sha256_file(receipt_path),
        },
    }
    state_id = _identifier("deployment_state", body)
    payload = {"deployment_state_id": state_id, **body}
    target = (
        config.deployment_root
        / "states"
        / state_id
        / "state.json"
    )
    _seal_content_addressed_json(target, payload)
    _load_deployment_state(target)
    return payload, target


def _publish_pointer(
    config: DeploymentBootstrapConfig,
    plan: DeploymentBootstrapPlan,
    state: Mapping[str, Any],
    state_path: Path,
) -> Path:
    pointer_path = config.deployment_root / POINTER_RELATIVE_PATH
    relative = (
        state_path.resolve()
        .relative_to(config.deployment_root.resolve())
        .as_posix()
    )
    pointer = ActiveDeploymentPointer(
        generation=1,
        deployment_id=plan.deployment_id,
        deployment_state_id=str(state["deployment_state_id"]),
        state_manifest_path=relative,
        state_manifest_sha256=sha256_file(state_path),
        updated_at_utc=_execution_time(config),
    )
    data = _json_bytes(pointer.to_dict())
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer_path.parent / f".current.{uuid4().hex}.tmp"
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, pointer_path)
            published = True
        except FileExistsError as exc:
            raise RetrainingDeploymentError(
                "Deployment pointer appeared during atomic publication."
            ) from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                if published:
                    raise _PointerPublishedCleanupError(
                        "Deployment pointer was published but its prepared "
                        "temporary file could not be removed."
                    )
                raise
    return pointer_path


def _load_deployment_state(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    expected = {
        "schema_version",
        "deployment_state_id",
        "generation",
        "deployment_id",
        "registry",
        "expected_aliases",
        "pins",
        "calibration",
        "monitoring",
        "cutoffs",
        "predecessor",
        "authorizing_receipt",
    }
    if (
        set(payload) != expected
        or payload.get("schema_version") != DEPLOYMENT_STATE_SCHEMA
        or payload.get("generation") != 1
        or isinstance(payload.get("generation"), bool)
        or payload.get("predecessor") is not None
    ):
        raise RetrainingDeploymentError(
            "Deployment state fields differ from generation-one strict v1."
        )
    identifier = _required_identifier(
        payload.get("deployment_state_id"), "deployment_state_id"
    )
    body = {
        key: value
        for key, value in payload.items()
        if key != "deployment_state_id"
    }
    if identifier != _identifier("deployment_state", body):
        raise RetrainingDeploymentError("Deployment state identity is corrupt.")
    if path.parent.name != identifier:
        raise RetrainingDeploymentError(
            "Deployment state path and identity differ."
        )
    _required_identifier(payload.get("deployment_id"), "deployment_id")
    _validate_pins(payload.get("pins"))
    _validate_expected_aliases(payload.get("expected_aliases"))
    registry = payload.get("registry")
    if not isinstance(registry, dict) or set(registry) != {
        "tracking_uri",
        "registered_model_name",
        "model_version",
        "run_id",
        "model_uri",
    }:
        raise RetrainingDeploymentError("Deployment Registry state is invalid.")
    for key, value in registry.items():
        _required_text(value, f"registry {key}")
    if payload["expected_aliases"]["champion"] != registry["model_version"]:
        raise RetrainingDeploymentError(
            "Deployment Registry version and expected aliases differ."
        )
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict) or set(calibration) != {
        "calibration_id",
        "reference_id",
    }:
        raise RetrainingDeploymentError(
            "Deployment calibration/reference state is invalid."
        )
    _required_text(calibration.get("calibration_id"), "calibration_id")
    _required_text(calibration.get("reference_id"), "reference_id")
    monitoring = payload.get("monitoring")
    if not isinstance(monitoring, dict) or set(monitoring) != {
        "ledger_model_snapshot_id",
        "ledger_state_sha256",
    }:
        raise RetrainingDeploymentError("Deployment monitoring state is invalid.")
    _required_identifier(
        monitoring.get("ledger_model_snapshot_id"),
        "ledger_model_snapshot_id",
    )
    _sha256_text(monitoring.get("ledger_state_sha256"), "ledger_state_sha256")
    if monitoring["ledger_state_sha256"] != payload["pins"]["ledger_sha256"]:
        raise RetrainingDeploymentError(
            "Deployment monitoring and ledger checksum pins differ."
        )
    cutoffs = payload.get("cutoffs")
    if not isinstance(cutoffs, dict) or set(cutoffs) != {
        "fit_cutoff",
        "activation_cutoff",
    }:
        raise RetrainingDeploymentError("Deployment cutoffs are invalid.")
    _date_text(cutoffs.get("fit_cutoff"), "fit_cutoff")
    _date_text(cutoffs.get("activation_cutoff"), "activation_cutoff")
    receipt = payload.get("authorizing_receipt")
    if not isinstance(receipt, dict) or set(receipt) != {
        "bootstrap_receipt_id",
        "path",
        "sha256",
    }:
        raise RetrainingDeploymentError(
            "Deployment authorizing receipt reference is invalid."
        )
    _required_identifier(
        receipt.get("bootstrap_receipt_id"), "bootstrap_receipt_id"
    )
    _required_text(receipt.get("path"), "authorizing receipt path")
    _sha256_text(receipt.get("sha256"), "authorizing receipt sha256")
    return payload


def _load_approval(
    config: DeploymentBootstrapConfig,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if config.approval_path is None or config.approval_sha256 is None:
        raise RetrainingDeploymentError(
            "Non-dry bootstrap requires --approval-path and --approval-sha256."
        )
    if not config.approval_path.is_file() or config.approval_path.is_symlink():
        raise RetrainingDeploymentError(
            "Bootstrap approval must be a regular existing file."
        )
    actual_sha = sha256_file(config.approval_path)
    if actual_sha != config.approval_sha256:
        raise RetrainingDeploymentError("Bootstrap approval checksum differs.")
    approval = load_bootstrap_approval(config.approval_path)
    for key, value in expected.items():
        if key in {"approved_by", "approved_at_utc", "note"}:
            continue
        if approval.get(key) != value:
            raise RetrainingDeploymentError(
                f"Bootstrap approval pin {key} differs from verified evidence."
            )
    return approval, actual_sha


def _reconcile_failed_bootstrap(
    config: DeploymentBootstrapConfig,
    client: Any,
    plan: DeploymentBootstrapPlan,
    version: str,
    failure: Exception,
) -> None:
    compensation_errors = []
    for alias in ("champion", "stable"):
        try:
            current = _alias_version(
                client, config.registered_model_name, alias
            )
            if current == version:
                client.delete_registered_model_alias(
                    config.registered_model_name, alias
                )
                remaining = _alias_version(
                    client, config.registered_model_name, alias
                )
                if remaining is not None:
                    compensation_errors.append(
                        f"{alias} remained {remaining} after deletion"
                    )
            elif current is not None:
                compensation_errors.append(
                    f"{alias} changed to {current}; deletion was unsafe"
                )
        except Exception as exc:
            compensation_errors.append(
                f"{alias}: {type(exc).__name__}: {str(exc)[:300]}"
            )
    target = _write_reconciliation_evidence(
        config,
        plan,
        version,
        failure,
        pointer_published=False,
        compensation_errors=tuple(compensation_errors),
    )
    if compensation_errors:
        raise RetrainingDeploymentReconciliationError(
            "Bootstrap failed and alias compensation was incomplete; inspect "
            f"{target}."
        ) from failure


def _write_reconciliation_evidence(
    config: DeploymentBootstrapConfig,
    plan: DeploymentBootstrapPlan,
    version: str,
    failure: Exception,
    *,
    pointer_published: bool,
    compensation_errors: tuple[str, ...],
) -> Path:
    body = {
        "schema_version": RECONCILIATION_SCHEMA,
        "deployment_id": plan.deployment_id,
        "registered_model_name": config.registered_model_name,
        "orphaned_model_version": version,
        "pointer_published": pointer_published,
        "failure_type": type(failure).__name__,
        "failure": str(failure)[:1000],
        "alias_compensation_errors": list(compensation_errors),
        "recorded_at_utc": _execution_time(config),
    }
    identifier = _identifier("bootstrap_reconciliation", body)
    target = (
        config.deployment_root
        / "reconciliation"
        / f"{identifier}.json"
    )
    try:
        _immutable_json_file(target, body)
    except Exception as exc:
        raise RetrainingDeploymentReconciliationError(
            "Bootstrap failed and immutable reconciliation evidence could not "
            f"be written: {type(exc).__name__}: {str(exc)[:500]}."
        ) from exc
    return target


def _assert_pristine_deployment_root(
    root: Path,
    *,
    ignored: tuple[Path, ...] = (),
) -> None:
    _assert_pointer_absent(root)
    if not root.exists():
        return
    if not root.is_dir() or root.is_symlink():
        raise RetrainingDeploymentError(
            "Deployment root must be an absent or real directory."
        )
    ignored_resolved = {path.resolve() for path in ignored}
    for item in root.iterdir():
        if item.resolve() in ignored_resolved:
            continue
        raise RetrainingDeploymentError(
            "Prior or incompatible deployment evidence exists."
        )


def _assert_pointer_absent(root: Path) -> None:
    if (root / POINTER_RELATIVE_PATH).exists():
        raise RetrainingDeploymentError(
            "Deployment pointer already exists; bootstrap is forbidden."
        )


def _resolve_relative_file(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise RetrainingDeploymentError(
            "Deployment pointer contains an unsafe state path."
        )
    root_resolved = root.resolve()
    if root.is_symlink():
        raise RetrainingDeploymentError("Deployment root must not be a symlink.")
    current = root
    for part in value.parts:
        current = current / part
        if current.is_symlink():
            raise RetrainingDeploymentError(
                "Deployment evidence paths must not traverse symlinks."
            )
    resolved = current.resolve()
    if not _is_within(resolved, root_resolved) or not resolved.is_file():
        raise RetrainingDeploymentError(
            "Deployment evidence path is missing or outside its root."
        )
    return resolved


def _seal_content_addressed_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RetrainingDeploymentError(
            "Prior bootstrap evidence exists; automatic retry is forbidden."
        )
    prepared = path.parent.parent / f".{path.parent.name}.{uuid4().hex}.tmp"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.mkdir()
    try:
        _immutable_json_file(prepared / path.name, payload)
        prepared.rename(path.parent)
    finally:
        if prepared.exists():
            for child in prepared.iterdir():
                child.unlink()
            prepared.rmdir()


def _immutable_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _validate_signature(signature: Any, features: list[str]) -> None:
    if signature is None:
        raise RetrainingDeploymentError("MLflow logged model has no signature.")
    inputs = getattr(getattr(signature, "inputs", None), "inputs", None)
    outputs = getattr(getattr(signature, "outputs", None), "inputs", None)
    try:
        input_items = list(inputs)
        output_items = list(outputs)
    except TypeError as exc:
        raise RetrainingDeploymentError(
            "MLflow logged-model signature is not inspectable."
        ) from exc
    if [str(getattr(item, "name", "")) for item in input_items] != features:
        raise RetrainingDeploymentError(
            "MLflow signature input order differs from the accepted bundle."
        )
    if (
        not input_items
        or not all(_numeric_signature_item(item) for item in input_items)
        or len(output_items) != 1
        or not _numeric_signature_item(output_items[0])
    ):
        raise RetrainingDeploymentError(
            "MLflow signature must contain numeric inputs and one numeric output."
        )


def _numeric_signature_item(item: Any) -> bool:
    value = getattr(item, "type", None)
    text = str(getattr(value, "name", value)).lower()
    return text in {
        "double",
        "float",
        "integer",
        "long",
        "int",
        "float32",
        "float64",
        "int32",
        "int64",
    }


def _validate_pins(value: Any) -> None:
    expected = {
        "bundle_sha256",
        "calibration_sha256",
        "ledger_sha256",
        "model_sha256",
        "dataset_sha256",
        "feature_schema_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise RetrainingDeploymentError("Deployment checksum pins are invalid.")
    for key, digest in value.items():
        _sha256_text(digest, key)


def _validate_expected_aliases(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != set(ALIASES)
        or value.get("candidate") is not None
        or not value.get("champion")
        or value.get("champion") != value.get("stable")
    ):
        raise RetrainingDeploymentError(
            "Expected bootstrap aliases are invalid."
        )


def _require_alias(
    client: Any,
    name: str,
    alias: str,
    expected: str,
) -> None:
    actual = _alias_version(client, name, alias)
    if actual != expected:
        raise RetrainingDeploymentError(
            f"Registry alias {alias} differs from deployment state."
        )


def _require_expectations(config: DeploymentBootstrapConfig) -> None:
    if config.expect_no_deployment_pointer is not True:
        raise RetrainingDeploymentError(
            "Bootstrap requires --expect-no-deployment-pointer."
        )
    if config.expect_no_v2_registry_state is not True:
        raise RetrainingDeploymentError(
            "Bootstrap requires --expect-no-v2-registry-state."
        )


def _execution_time(config: DeploymentBootstrapConfig) -> str:
    return _utc_text(config.now_utc or datetime.now(timezone.utc))


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise RetrainingDeploymentError("now_utc must be timezone-aware UTC.")
    return value.isoformat().replace("+00:00", "Z")


def _parse_utc(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RetrainingDeploymentError(f"{name} must be valid UTC.") from exc
    if (
        not value.endswith("Z")
        or parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
    ):
        raise RetrainingDeploymentError(f"{name} must be timezone-aware UTC.")
    return parsed


def _date_text(value: Any, name: str) -> str:
    text = _required_text(value, name)
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise RetrainingDeploymentError(
            f"{name} must be an ISO calendar date."
        ) from exc


def _required_text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
    ):
        raise RetrainingDeploymentError(f"{name} must be a non-empty string.")
    return value


def _required_identifier(value: Any, name: str) -> str:
    text = _required_text(value, name)
    if not SHA256_PATTERN.fullmatch(text):
        raise RetrainingDeploymentError(f"{name} must be a SHA-256 identifier.")
    return text


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise RetrainingDeploymentError(f"{name} must be a SHA-256 digest.")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RetrainingDeploymentError(
            f"JSON evidence must be a regular non-symlink file: {path}."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrainingDeploymentError(f"Invalid JSON evidence: {path}.") from exc
    if not isinstance(payload, dict):
        raise RetrainingDeploymentError(f"JSON evidence must be an object: {path}.")
    return payload


def _identifier(kind: str, body: Mapping[str, Any]) -> str:
    return sha256(kind.encode("ascii") + b":" + _canonical(body)).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve()
    right = second.resolve()
    return left == right or left in right.parents or right in left.parents


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    root_resolved = root.resolve()
    return resolved == root_resolved or root_resolved in resolved.parents


__all__ = [
    "APPROVAL_SCHEMA",
    "DEPLOYMENT_STATE_SCHEMA",
    "POINTER_RELATIVE_PATH",
    "RECEIPT_SCHEMA",
    "DeploymentBootstrapConfig",
    "DeploymentBootstrapPlan",
    "DeploymentBootstrapResult",
    "RetrainingDeploymentError",
    "RetrainingDeploymentReconciliationError",
    "bootstrap_v2_deployment",
    "load_bootstrap_approval",
    "load_bootstrap_receipt",
    "load_verified_deployment_pointer",
    "load_exact_v2_bundle",
    "plan_v2_deployment_bootstrap",
]
