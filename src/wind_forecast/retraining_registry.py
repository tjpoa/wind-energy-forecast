"""Optimistic, compensating Registry action for sealed retraining candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from wind_forecast.manifests import sha256_file
from wind_forecast.retraining_backtesting import (
    RetrainingBacktestError,
    load_retraining_backtest,
)
from wind_forecast.tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    _load_mlflow,
    git_state,
)


REGISTRATION_RECEIPT_SCHEMA = "wind_forecast.retraining_registration_receipt.v1"


class RetrainingRegistryError(RuntimeError):
    """Raised before a Registry mutation or after successful compensation."""


class RetrainingRegistryReconciliationError(RuntimeError):
    """Raised when post-creation state cannot be reconciled automatically."""


@dataclass(frozen=True)
class RetrainingRegistrationConfig:
    """Explicit, optimistic inputs for one v2 candidate registration."""

    backtest_bundle: Path
    run_id: str
    registered_model_name: str
    expected_current_candidate_version: str | None
    output_root: Path
    registry_lock_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backtest_bundle", Path(self.backtest_bundle))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self,
            "registry_lock_root",
            (
                Path(self.registry_lock_root)
                if self.registry_lock_root is not None
                else self.output_root
            ),
        )
        if _paths_overlap(self.output_root, self.backtest_bundle):
            raise RetrainingRegistryError(
                "Registry output root must not overlap the backtest bundle."
            )
        if _paths_overlap(self.registry_lock_root, self.backtest_bundle):
            raise RetrainingRegistryError(
                "Registry lock root must not overlap the backtest bundle."
            )
        if (
            not isinstance(self.registered_model_name, str)
            or not self.registered_model_name.strip()
            or self.registered_model_name != self.registered_model_name.strip()
        ):
            raise RetrainingRegistryError(
                "registered_model_name is required and must be non-empty."
            )
        if self.registered_model_name == DEFAULT_REGISTERED_MODEL_NAME:
            raise RetrainingRegistryError(
                "The legacy DEFAULT_REGISTERED_MODEL_NAME is forbidden for v2."
            )
        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
            or self.run_id != self.run_id.strip()
        ):
            raise RetrainingRegistryError("run_id must be non-empty.")
        expected = self.expected_current_candidate_version
        if expected is not None and (
            not isinstance(expected, str) or not expected.strip()
        ):
            raise RetrainingRegistryError(
                "Expected candidate state must be none or an exact non-empty version."
            )


@dataclass(frozen=True)
class RetrainingRegistrationReceipt:
    """Immutable evidence for one successful candidate-only Registry mutation."""

    registration_id: str
    registered_model_name: str
    model_version: str
    previous_candidate_version: str | None
    run_id: str
    model_uri: str
    backtest_id: str
    tags: Mapping[str, str]
    champion_before: str | None
    champion_after: str | None
    stable_before: str | None
    stable_after: str | None
    candidate_model_sha256: str
    final_training_dataset_sha256: str
    final_training_identity_sha256: str
    schema_version: str = REGISTRATION_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def register_retraining_candidate(
    config: RetrainingRegistrationConfig,
    *,
    client: Any | None = None,
    mlflow_module: Any | None = None,
    git_lineage: Mapping[str, Any] | None = None,
) -> RetrainingRegistrationReceipt:
    """Validate locally, create one version, tag it, and move only candidate."""
    try:
        bundle = load_retraining_backtest(config.backtest_bundle)
    except RetrainingBacktestError as exc:
        raise RetrainingRegistryError(str(exc)) from exc
    backtest = bundle["backtest"]
    if backtest["outcome"] != "accepted":
        raise RetrainingRegistryError("Only a sealed accepted backtest may register.")
    lineage = dict(git_lineage or git_state())
    if lineage.get("git_dirty") is not False or not lineage.get("git_sha"):
        raise RetrainingRegistryError("Registration requires clean Git lineage.")
    recorded_git = backtest.get("git") or {}
    if (
        recorded_git.get("git_dirty") is not False
        or recorded_git.get("git_sha") != lineage["git_sha"]
    ):
        raise RetrainingRegistryError(
            "Current clean Git lineage differs from the sealed backtest."
        )
    mlflow = mlflow_module or _load_mlflow()
    client = client or mlflow.MlflowClient()
    run = client.get_run(config.run_id)
    if str(getattr(run.info, "status", "")) != "FINISHED":
        raise RetrainingRegistryError("MLflow source run is not FINISHED.")
    params = dict(getattr(run.data, "params", {}) or {})
    model_uri = str(params.get("logged_model_uri") or "")
    if not model_uri or str(params.get("backtest_id") or "") != backtest["backtest_id"]:
        raise RetrainingRegistryError(
            "MLflow run does not pin the sealed backtest and model URI."
        )
    if (
        str(params.get("git_sha") or "") != lineage["git_sha"]
        or str(params.get("git_dirty") or "").lower() not in {"false", "0"}
    ):
        raise RetrainingRegistryError(
            "MLflow run does not have matching clean Git lineage."
        )
    info = mlflow.models.get_model_info(model_uri)
    if getattr(info, "signature", None) is None:
        raise RetrainingRegistryError("Logged candidate model has no signature.")
    if str(getattr(info, "run_id", "")) != config.run_id:
        raise RetrainingRegistryError("Logged model and source run IDs differ.")
    root = Path(config.backtest_bundle)
    if root.is_file():
        root = root.parent
    model_manifest = json.loads(
        (root / "model_manifest.json").read_text(encoding="utf-8")
    )
    feature_names = list(model_manifest.get("feature_names") or [])
    _validate_signature(info.signature, feature_names)
    artifact_path = str(params.get("candidate_model_artifact_path") or "")
    _validate_run_artifact_path(artifact_path)
    try:
        downloaded_model = Path(
            mlflow.artifacts.download_artifacts(
                run_id=config.run_id,
                artifact_path=artifact_path,
            )
        )
    except Exception as exc:
        raise RetrainingRegistryError(
            "Exact candidate model run artifact is missing."
        ) from exc
    sealed_model = root / "model.joblib"
    if (
        sha256_file(downloaded_model) != sha256_file(sealed_model)
        or sha256_file(sealed_model)
        != backtest["final_training"]["candidate_model_sha256"]
    ):
        raise RetrainingRegistryError(
            "MLflow run artifact differs from sealed candidate model.joblib."
        )
    artifact_estimator = joblib.load(downloaded_model)
    if (
        type(artifact_estimator).__name__ != model_manifest.get("model_type")
        or artifact_estimator.get_params(deep=True)
        != model_manifest.get("parameters")
        or list(getattr(artifact_estimator, "feature_names_in_", ()))
        != feature_names
    ):
        raise RetrainingRegistryError(
            "Run-artifact estimator class, parameters, or features differ."
        )
    evidence = pd.read_csv(root / "training_evidence.csv")
    if evidence.columns.tolist() != [
        "Date",
        "Wind_Production",
        *feature_names,
        "Expected_Prediction",
    ]:
        raise RetrainingRegistryError(
            "Complete sealed candidate training evidence is invalid."
        )
    features = evidence[feature_names]
    expected = evidence["Expected_Prediction"].to_numpy(dtype=float)
    loaded = mlflow.pyfunc.load_model(model_uri)
    actual = np.asarray(loaded.predict(features), dtype=float)
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-9):
        raise RetrainingRegistryError(
            "Logged-model reload predictions differ from sealed evidence."
        )
    raw_actual = np.asarray(
        artifact_estimator.predict(features), dtype=float
    )
    if (
        raw_actual.shape != expected.shape
        or not np.allclose(raw_actual, expected, rtol=1e-12, atol=1e-9)
        or not np.allclose(raw_actual, actual, rtol=1e-12, atol=1e-9)
    ):
        raise RetrainingRegistryError(
            "Exact raw model artifact predictions differ from sealed evidence "
            "or the logged pyfunc model."
        )
    lock_path = _acquire_registry_lock(config)
    try:
        return _register_under_lock(
            config=config,
            client=client,
            mlflow=mlflow,
            model_uri=model_uri,
            bundle=bundle,
            backtest=backtest,
            lineage=lineage,
        )
    finally:
        _release_registry_lock(lock_path)


def _register_under_lock(
    *,
    config: RetrainingRegistrationConfig,
    client: Any,
    mlflow: Any,
    model_uri: str,
    bundle: Mapping[str, Any],
    backtest: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> RetrainingRegistrationReceipt:
    candidate_before = _optional_alias(
        client, config.registered_model_name, "candidate"
    )
    previous_candidate = _version(candidate_before)
    if previous_candidate != config.expected_current_candidate_version:
        raise RetrainingRegistryError(
            "Candidate alias changed before registration: "
            f"expected {config.expected_current_candidate_version or 'none'}, "
            f"found {previous_candidate or 'none'}."
        )
    champion_before = _version(
        _optional_alias(client, config.registered_model_name, "champion")
    )
    stable_before = _version(
        _optional_alias(client, config.registered_model_name, "stable")
    )
    version = mlflow.register_model(
        model_uri=model_uri, name=config.registered_model_name
    )
    version_text = str(version.version)
    tags = _registration_tags(config, backtest, bundle, lineage)
    try:
        if getattr(version, "run_id", None) not in {None, config.run_id}:
            raise RetrainingRegistryError(
                "Created version has an unexpected source run."
            )
        for key, value in tags.items():
            client.set_model_version_tag(
                config.registered_model_name, version_text, key, value
            )
        observed = _version(
            _optional_alias(client, config.registered_model_name, "candidate")
        )
        if observed != previous_candidate:
            raise RetrainingRegistryError(
                "Candidate alias raced after version creation."
            )
        if _version(
            _optional_alias(client, config.registered_model_name, "champion")
        ) != champion_before or _version(
            _optional_alias(client, config.registered_model_name, "stable")
        ) != stable_before:
            raise RetrainingRegistryError(
                "Champion or stable alias changed during registration."
            )
        # MLflow exposes no compare-and-set alias primitive. This immediate
        # re-read is protected against cooperating PR3 CLIs by the local lock.
        if _version(
            _optional_alias(client, config.registered_model_name, "candidate")
        ) != previous_candidate:
            raise RetrainingRegistryError(
                "Candidate alias changed immediately before update."
            )
        client.set_registered_model_alias(
            config.registered_model_name, "candidate", version_text
        )
        if _version(
            _optional_alias(client, config.registered_model_name, "candidate")
        ) != version_text:
            raise RetrainingRegistryError("Candidate alias update did not persist.")
        champion_after = _version(
            _optional_alias(client, config.registered_model_name, "champion")
        )
        stable_after = _version(
            _optional_alias(client, config.registered_model_name, "stable")
        )
        if champion_after != champion_before or stable_after != stable_before:
            raise RetrainingRegistryError(
                "Candidate registration mutated champion or stable."
            )
        receipt = _receipt(
            config=config,
            version=version_text,
            model_uri=model_uri,
            backtest=backtest,
            tags=tags,
            champion=champion_before,
            stable=stable_before,
        )
        _seal_receipt(config.output_root, receipt)
        return receipt
    except Exception as exc:
        try:
            _restore_candidate(
                client,
                config.registered_model_name,
                previous_candidate,
                only_if_current=version_text,
            )
        except Exception as compensation:
            try:
                recovery = _write_recovery_evidence(
                    config,
                    version_text,
                    previous_candidate,
                    exc,
                    compensation,
                )
            except Exception as recovery_failure:
                raise RetrainingRegistryReconciliationError(
                    "Candidate registration and alias compensation failed, and "
                    "recovery evidence could not be written. "
                    f"Compensation failure: {type(compensation).__name__}: "
                    f"{str(compensation)[:500]}. Recovery-write failure: "
                    f"{type(recovery_failure).__name__}: "
                    f"{str(recovery_failure)[:500]}."
                ) from recovery_failure
            raise RetrainingRegistryReconciliationError(
                "Candidate registration and alias compensation failed; "
                f"inspect recovery evidence {recovery}. Compensation failure: "
                f"{type(compensation).__name__}: {str(compensation)[:500]}."
            ) from compensation
        raise RetrainingRegistryError(
            "Candidate registration failed after version creation; candidate alias "
            "was restored when necessary. The unaliased version is retained for audit."
        ) from exc


def load_retraining_registration_receipt(path: str | Path) -> dict[str, Any]:
    """Load one strict immutable content-addressed Registry receipt."""
    receipt_path = Path(path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REGISTRATION_RECEIPT_SCHEMA
    ):
        raise RetrainingRegistryError("Unsupported registration receipt schema.")
    expected_fields = {
        "registration_id",
        "registered_model_name",
        "model_version",
        "previous_candidate_version",
        "run_id",
        "model_uri",
        "backtest_id",
        "tags",
        "champion_before",
        "champion_after",
        "stable_before",
        "stable_after",
        "candidate_model_sha256",
        "final_training_dataset_sha256",
        "final_training_identity_sha256",
        "schema_version",
    }
    if set(payload) != expected_fields:
        raise RetrainingRegistryError(
            "Registration receipt fields differ from strict v1."
        )
    identifier = payload.get("registration_id")
    body = {key: value for key, value in payload.items() if key != "registration_id"}
    if identifier != _registration_id(body) or receipt_path.parent.name != identifier:
        raise RetrainingRegistryError("Registration receipt identity is corrupt.")
    return payload


def _registration_tags(
    config: RetrainingRegistrationConfig,
    backtest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, str]:
    aggregate = backtest["aggregate_metrics"]["candidate"]
    return {
        "validation_status": "passed",
        "lifecycle_role": "candidate",
        "backtest_id": str(backtest["backtest_id"]),
        "evaluation_id": str(backtest["evaluation_id"]),
        "evaluation_period": str(backtest["evaluation_period"]),
        "policy_sha256": str(backtest["identities"]["policy_sha256"]),
        "calibration_id": str(backtest["identities"]["calibration_id"]),
        "reference_id": str(backtest["identities"]["reference_id"]),
        "feature_schema_sha256": str(
            backtest["identities"]["feature_schema_sha256"]
        ),
        "incumbent_model_sha256": str(
            backtest["identities"]["incumbent_model_sha256"]
        ),
        "incumbent_fit_cutoff": str(
            backtest["cutoffs"]["incumbent_fit_cutoff"]
        ),
        "data_snapshot_cutoff": str(backtest["cutoffs"]["data_snapshot_cutoff"]),
        "candidate_fit_cutoff": str(backtest["cutoffs"]["candidate_fit_cutoff"]),
        "candidate_MAE": str(aggregate["MAE"]),
        "candidate_RMSE": str(aggregate["RMSE"]),
        "candidate_MAPE_percent": str(aggregate["MAPE_percent"]),
        "candidate_R2": str(aggregate["R2"]),
        "candidate_absolute_bias": str(abs(float(aggregate["bias"]))),
        "bundle_manifest_sha256": sha256_file(
            (
                config.backtest_bundle.parent
                if config.backtest_bundle.is_file()
                else config.backtest_bundle
            )
            / "bundle_manifest.json"
        ),
        "git_sha": str(lineage["git_sha"]),
        "source_run_id": config.run_id,
        "candidate_model_sha256": str(
            backtest["final_training"]["candidate_model_sha256"]
        ),
        "final_training_dataset_sha256": str(
            backtest["final_training"]["dataset_sha256"]
        ),
        "final_training_identity_sha256": str(
            backtest["final_training"]["identity_sha256"]
        ),
    }


def _receipt(
    *,
    config: RetrainingRegistrationConfig,
    version: str,
    model_uri: str,
    backtest: Mapping[str, Any],
    tags: Mapping[str, str],
    champion: str | None,
    stable: str | None,
) -> RetrainingRegistrationReceipt:
    body = {
        "schema_version": REGISTRATION_RECEIPT_SCHEMA,
        "registered_model_name": config.registered_model_name,
        "model_version": version,
        "previous_candidate_version": config.expected_current_candidate_version,
        "run_id": config.run_id,
        "model_uri": model_uri,
        "backtest_id": backtest["backtest_id"],
        "tags": dict(tags),
        "champion_before": champion,
        "champion_after": champion,
        "stable_before": stable,
        "stable_after": stable,
        "candidate_model_sha256": backtest["final_training"][
            "candidate_model_sha256"
        ],
        "final_training_dataset_sha256": backtest["final_training"][
            "dataset_sha256"
        ],
        "final_training_identity_sha256": backtest["final_training"][
            "identity_sha256"
        ],
    }
    return RetrainingRegistrationReceipt(
        registration_id=_registration_id(body),
        **{key: value for key, value in body.items() if key != "schema_version"},
    )


def _seal_receipt(
    root: Path, receipt: RetrainingRegistrationReceipt
) -> Path:
    target = root / receipt.registration_id / "receipt.json"
    if target.is_file():
        loaded = load_retraining_registration_receipt(target)
        if loaded != receipt.to_dict():
            raise RetrainingRegistryError("Conflicting registration receipt exists.")
        return target
    if root.exists() and not root.is_dir():
        raise RetrainingRegistryError("Receipt output root is not a directory.")
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / f".{receipt.registration_id}.{uuid4().hex}.tmp"
    prepared.mkdir()
    try:
        data = (
            json.dumps(
                receipt.to_dict(), ensure_ascii=True, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8")
        path = prepared / "receipt.json"
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        prepared.rename(target.parent)
    finally:
        if prepared.exists():
            for child in prepared.iterdir():
                child.unlink()
            prepared.rmdir()
    return target


def _restore_candidate(
    client: Any,
    model_name: str,
    previous: str | None,
    *,
    only_if_current: str,
) -> None:
    current = _version(_optional_alias(client, model_name, "candidate"))
    if current not in {previous, only_if_current}:
        raise RetrainingRegistryReconciliationError(
            "Candidate changed again; compensation was not safe."
        )
    if current == previous:
        return
    if previous is None:
        client.delete_registered_model_alias(model_name, "candidate")
    else:
        client.set_registered_model_alias(model_name, "candidate", previous)


def _validate_run_artifact_path(value: str) -> None:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.name != "model.joblib"
    ):
        raise RetrainingRegistryError(
            "candidate_model_artifact_path must be a safe run-relative model.joblib."
        )


def _validate_signature(signature: Any, feature_names: list[str]) -> None:
    inputs = getattr(signature, "inputs", None)
    outputs = getattr(signature, "outputs", None)
    input_items = getattr(inputs, "inputs", inputs)
    output_items = getattr(outputs, "inputs", outputs)
    try:
        input_items = list(input_items)
        output_items = list(output_items)
    except TypeError as exc:
        raise RetrainingRegistryError("Logged-model signature is not inspectable.") from exc
    names = [str(getattr(item, "name", "")) for item in input_items]
    if names != feature_names:
        raise RetrainingRegistryError(
            "Logged-model signature input order differs from sealed features."
        )
    if not input_items or not all(_numeric_signature_type(item) for item in input_items):
        raise RetrainingRegistryError(
            "Logged-model signature inputs must all be numeric."
        )
    if len(output_items) != 1 or not _numeric_signature_type(output_items[0]):
        raise RetrainingRegistryError(
            "Logged-model signature must declare one numeric output."
        )


def _numeric_signature_type(item: Any) -> bool:
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


def registry_lock_path(
    lock_root: str | Path,
    registered_model_name: str,
) -> Path:
    """Return the shared local lock path for one Registry model name."""
    identity = sha256(registered_model_name.encode("utf-8")).hexdigest()
    return Path(lock_root).resolve() / ".registry-locks" / f"{identity}.lock"


def acquire_registry_lock(
    lock_root: str | Path,
    registered_model_name: str,
    owner: Mapping[str, Any],
) -> Path:
    """Acquire the cooperative Registry lock without replacing stale state."""
    path = registry_lock_path(lock_root, registered_model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "wind_forecast.retraining_registry_lock.v1",
        "registered_model_name": registered_model_name,
        "owner": dict(owner),
    }
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RetrainingRegistryError(
            f"Registry critical section is locked: {path}. "
            "Inspect the lock before manual recovery."
        ) from exc
    return path


def release_registry_lock(path: Path) -> None:
    """Release a lock created by :func:`acquire_registry_lock`."""
    try:
        path.unlink()
        path.parent.rmdir()
    except OSError as exc:
        if path.exists():
            raise RetrainingRegistryReconciliationError(
                f"Registry lock could not be released: {path}."
            ) from exc


def _registry_lock_path(config: RetrainingRegistrationConfig) -> Path:
    """Backward-compatible test/support helper for the candidate CLI."""
    return registry_lock_path(
        config.registry_lock_root or config.output_root,
        config.registered_model_name,
    )


def _acquire_registry_lock(config: RetrainingRegistrationConfig) -> Path:
    return acquire_registry_lock(
        config.registry_lock_root or config.output_root,
        config.registered_model_name,
        {
            "action": "register_retraining_candidate",
            "run_id": config.run_id,
            "backtest_bundle": str(config.backtest_bundle.resolve()),
        },
    )


def _release_registry_lock(path: Path) -> None:
    release_registry_lock(path)


def _write_recovery_evidence(
    config: RetrainingRegistrationConfig,
    created_version: str,
    previous_candidate: str | None,
    failure: Exception,
    compensation_failure: Exception,
) -> Path:
    root = config.output_root / "reconciliation"
    root.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex
    path = root / f"{identifier}.json"
    payload = {
        "schema_version": "wind_forecast.retraining_registry_recovery.v1",
        "registered_model_name": config.registered_model_name,
        "run_id": config.run_id,
        "created_version": created_version,
        "previous_candidate_version": previous_candidate,
        "failure_type": type(failure).__name__,
        "failure": str(failure)[:1000],
        "compensation_failure_type": type(compensation_failure).__name__,
        "compensation_failure": str(compensation_failure)[:1000],
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve()
    right = second.resolve()
    return left == right or left in right.parents or right in left.parents


def _optional_alias(client: Any, model_name: str, alias: str) -> Any | None:
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except LookupError:
        return None
    except Exception as exc:
        if getattr(exc, "error_code", None) in {
            "RESOURCE_DOES_NOT_EXIST",
            "INVALID_PARAMETER_VALUE",
        }:
            return None
        raise


def _version(value: Any | None) -> str | None:
    return None if value is None else str(value.version)


def _registration_id(body: Mapping[str, Any]) -> str:
    return sha256(b"retraining_registration:" + _canonical(body)).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


__all__ = [
    "REGISTRATION_RECEIPT_SCHEMA",
    "RetrainingRegistrationConfig",
    "RetrainingRegistrationReceipt",
    "RetrainingRegistryError",
    "RetrainingRegistryReconciliationError",
    "acquire_registry_lock",
    "load_retraining_registration_receipt",
    "registry_lock_path",
    "register_retraining_candidate",
    "release_registry_lock",
]
