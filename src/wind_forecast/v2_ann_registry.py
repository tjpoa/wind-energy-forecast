"""Candidate-only MLflow logging and Registry registration for ANN v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .manifests import sha256_file
from .retraining_registry import acquire_registry_lock, release_registry_lock
from .tracking import _load_mlflow, git_state
from .v2_ann import load_v2_ann_bundle
from .v2_ann_challenger import (
    ChallengerBacktestError,
    load_v2_ann_challenger_bundle,
)


REGISTRATION_SCHEMA = "wind_forecast.v2_ann_registration_receipt.v1"
RUN_SCHEMA = "wind_forecast.v2_ann_mlflow_run.v1"
EXPERIMENT_NAME = "wind-energy-forecast-v2-ann-challenger"


class ANNRegistryError(RuntimeError):
    """Raised when ANN candidate logging or registration fails closed."""


class ANNRegistryReconciliationError(ANNRegistryError):
    """Raised when a post-creation Registry state cannot be compensated safely."""


@dataclass(frozen=True)
class ANNRunConfig:
    """Inputs for one controlled local MLflow logging run."""

    candidate_bundle: Path
    backtest_bundle: Path
    calibration_dir: Path
    tracking_uri: str
    experiment_name: str = EXPERIMENT_NAME

    def __post_init__(self) -> None:
        for name in ("candidate_bundle", "backtest_bundle", "calibration_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if not self.tracking_uri.strip() or not self.experiment_name.strip():
            raise ValueError("tracking_uri and experiment_name must be non-empty.")


@dataclass(frozen=True)
class ANNRegistrationConfig:
    """Optimistic inputs for one candidate-only Registry mutation."""

    challenger_bundle: Path
    calibration_dir: Path
    run_id: str
    registered_model_name: str
    tracking_uri: str
    expected_candidate: str | None
    expected_champion: str | None
    expected_stable: str | None
    output_root: Path
    registry_lock_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "challenger_bundle", Path(self.challenger_bundle))
        object.__setattr__(self, "calibration_dir", Path(self.calibration_dir))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self,
            "registry_lock_root",
            Path(self.registry_lock_root or self.output_root),
        )
        if not self.run_id.strip() or not self.registered_model_name.strip() or not self.tracking_uri.strip():
            raise ValueError("run_id, registered_model_name, and tracking_uri are required.")
        if self.registered_model_name != self.registered_model_name.strip() or self.run_id != self.run_id.strip():
            raise ValueError("run_id and registered_model_name must not have surrounding whitespace.")
        for name, value in (
            ("expected_candidate", self.expected_candidate),
            ("expected_champion", self.expected_champion),
            ("expected_stable", self.expected_stable),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip() or value != value.strip()):
                raise ValueError(f"{name} must be None or an exact non-empty version.")
        if self.expected_candidate is not None:
            raise ANNRegistryError("ANN candidate registration requires expected_candidate=None.")
        if _overlap(self.output_root, self.challenger_bundle) or _overlap(self.registry_lock_root, self.challenger_bundle):
            raise ANNRegistryError("Registry output and lock roots must not overlap the challenger bundle.")


@dataclass(frozen=True)
class ANNRunReceipt:
    run_id: str
    model_uri: str
    tracking_uri: str
    experiment_name: str
    schema_version: str = RUN_SCHEMA

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
        }


@dataclass(frozen=True)
class ANNRegistrationReceipt:
    registration_id: str
    registered_model_name: str
    model_version: str
    run_id: str
    model_uri: str
    backtest_id: str
    candidate_model_sha256: str
    scaler_manifest_sha256: str
    dataset_sha256: str
    previous_candidate: str | None
    candidate_after: str | None
    champion_before: str | None
    champion_after: str | None
    stable_before: str | None
    stable_after: str | None
    tags: Mapping[str, str]
    schema_version: str = REGISTRATION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registration_id": self.registration_id,
            "registered_model_name": self.registered_model_name,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "backtest_id": self.backtest_id,
            "candidate_model_sha256": self.candidate_model_sha256,
            "scaler_manifest_sha256": self.scaler_manifest_sha256,
            "dataset_sha256": self.dataset_sha256,
            "previous_candidate": self.previous_candidate,
            "candidate_after": self.candidate_after,
            "champion_before": self.champion_before,
            "champion_after": self.champion_after,
            "stable_before": self.stable_before,
            "stable_after": self.stable_after,
            "tags": dict(self.tags),
        }


def log_ann_candidate_run(config: ANNRunConfig, *, mlflow_module: Any | None = None) -> ANNRunReceipt:
    """Log the composite candidate model and evidence, but never register it."""
    try:
        sealed = load_v2_ann_challenger_bundle(config.backtest_bundle)
    except ChallengerBacktestError as exc:
        raise ANNRegistryError(str(exc)) from exc
    if Path(config.candidate_bundle).resolve() != Path(sealed["root"]).resolve():
        raise ANNRegistryError("Candidate and accepted challenger bundle paths differ.")
    root = Path(config.candidate_bundle)
    predictor = load_v2_ann_bundle(root)
    calibration = _read_json(config.calibration_dir / "calibration.json")
    evidence = pd.read_csv(root / "training_evidence.csv")
    features = evidence[list(predictor.feature_names)]
    output = pd.DataFrame({"Wind_Production": predictor.predict(features)})
    mlflow = mlflow_module or _load_mlflow()
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name="v2-ann-challenger") as run:
        run_id = str(run.info.run_id)
        mlflow.log_params(
            {
                "model_family": "ANN",
                "artifact_type": "keras_scaled_v2",
                "target_variant": predictor.target_variant,
                "backtest_id": sealed["backtest"]["backtest_id"],
                "candidate_model_artifact_path": "candidate_bundle/model.keras",
                "calibration_id": str(calibration.get("calibration_id") or ""),
                "calibration_sha256": sha256_file(config.calibration_dir / "calibration.json"),
                "git_sha": str((sealed["backtest"].get("git") or {}).get("git_sha") or ""),
                "git_dirty": str((sealed["backtest"].get("git") or {}).get("git_dirty")),
            }
        )
        mlflow.log_artifacts(str(root), artifact_path="candidate_bundle")
        signature = mlflow.models.infer_signature(features, output)
        model_info = mlflow.pyfunc.log_model(
            artifact_path="ann_v2_model",
            python_model=_make_pyfunc_model(mlflow),
            artifacts={"bundle": str(root)},
            signature=signature,
            input_example=features.head(1),
        )
        model_uri = str(getattr(model_info, "model_uri", ""))
        if not model_uri:
            raise ANNRegistryError("MLflow did not return a logged model URI.")
        mlflow.log_param("logged_model_uri", model_uri)
        mlflow.log_metrics(
            {
                f"candidate_{key}": float(value)
                for key, value in sealed["backtest"]["aggregate_metrics"]["candidate"].items()
                if value is not None and key != "bias"
            }
        )
    return ANNRunReceipt(
        run_id=run_id,
        model_uri=model_uri,
        tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
    )


def register_ann_candidate(
    config: ANNRegistrationConfig,
    *,
    client: Any | None = None,
    mlflow_module: Any | None = None,
    git_lineage: Mapping[str, Any] | None = None,
) -> ANNRegistrationReceipt:
    """Validate a finished run and move only the Registry candidate alias."""
    try:
        sealed = load_v2_ann_challenger_bundle(config.challenger_bundle)
    except ChallengerBacktestError as exc:
        raise ANNRegistryError(str(exc)) from exc
    backtest = sealed["backtest"]
    lineage = dict(git_lineage or git_state())
    if lineage.get("git_dirty") is not False or not lineage.get("git_sha"):
        raise ANNRegistryError("ANN registration requires clean Git lineage.")
    recorded = backtest.get("git") or {}
    if recorded.get("git_dirty") is not False or recorded.get("git_sha") != lineage["git_sha"]:
        raise ANNRegistryError("Current clean Git lineage differs from the sealed backtest.")
    mlflow = mlflow_module or _load_mlflow()
    mlflow.set_tracking_uri(config.tracking_uri)
    client = client or mlflow.MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(config.run_id)
    if str(getattr(run.info, "status", "")) != "FINISHED":
        raise ANNRegistryError("MLflow source run is not FINISHED.")
    params = dict(getattr(run.data, "params", {}) or {})
    model_uri = str(params.get("logged_model_uri") or "")
    if not model_uri or str(params.get("backtest_id") or "") != backtest["backtest_id"]:
        raise ANNRegistryError("MLflow run does not pin the sealed challenger backtest.")
    calibration = _read_json(config.calibration_dir / "calibration.json")
    if (
        params.get("calibration_id") != calibration.get("calibration_id")
        or params.get("calibration_sha256") != sha256_file(config.calibration_dir / "calibration.json")
    ):
        raise ANNRegistryError("MLflow run does not pin the candidate calibration.")
    if params.get("git_sha") != lineage["git_sha"] or str(params.get("git_dirty", "")).lower() not in {"false", "0"}:
        raise ANNRegistryError("MLflow run does not have matching clean Git lineage.")
    artifact_path = str(params.get("candidate_model_artifact_path") or "")
    if not artifact_path or Path(artifact_path).is_absolute() or ".." in Path(artifact_path).parts or not artifact_path.endswith("model.keras"):
        raise ANNRegistryError("Candidate model artifact path is unsafe or not model.keras.")
    try:
        info = mlflow.models.get_model_info(model_uri)
        downloaded = Path(mlflow.artifacts.download_artifacts(run_id=config.run_id, artifact_path=artifact_path))
    except Exception as exc:
        raise ANNRegistryError("Logged ANN model or run artifact is unavailable.") from exc
    if getattr(info, "run_id", None) not in {None, config.run_id}:
        raise ANNRegistryError("Logged ANN model and source run IDs differ.")
    root = Path(sealed["root"])
    model_path = root / "model.keras"
    if sha256_file(downloaded) != sha256_file(model_path):
        raise ANNRegistryError("MLflow model artifact differs from sealed ANN model.")
    _validate_signature(getattr(info, "signature", None), list(load_v2_ann_bundle(root).feature_names))
    evidence = pd.read_csv(root / "training_evidence.csv")
    expected = evidence["Expected_Prediction"].to_numpy(float)
    predictor = load_v2_ann_bundle(root)
    loaded = mlflow.pyfunc.load_model(model_uri)
    actual = _prediction_values(loaded.predict(evidence[list(predictor.feature_names)]))
    if not np.allclose(actual, expected, rtol=1e-7, atol=1e-5):
        raise ANNRegistryError("Logged pyfunc reload differs from sealed training evidence.")
    lock = acquire_registry_lock(
        config.registry_lock_root,
        config.registered_model_name,
        {"action": "register_v2_ann_candidate", "run_id": config.run_id},
    )
    try:
        return _register_locked(config, client, mlflow, model_uri, sealed, lineage)
    finally:
        release_registry_lock(lock)


def _register_locked(config: ANNRegistrationConfig, client: Any, mlflow: Any, model_uri: str, sealed: Mapping[str, Any], lineage: Mapping[str, Any]) -> ANNRegistrationReceipt:
    candidate_before = _alias(client, config.registered_model_name, "candidate")
    champion_before = _alias(client, config.registered_model_name, "champion")
    stable_before = _alias(client, config.registered_model_name, "stable")
    if (candidate_before, champion_before, stable_before) != (config.expected_candidate, config.expected_champion, config.expected_stable):
        raise ANNRegistryError("Registry aliases differ from explicit optimistic expectations.")
    version = mlflow.register_model(model_uri=model_uri, name=config.registered_model_name)
    version_text = str(version.version)
    root = Path(sealed["root"])
    backtest = sealed["backtest"]
    calibration = _read_json(config.calibration_dir / "calibration.json")
    tags = {
        "validation_status": "passed",
        "lifecycle_role": "candidate",
        "model_family": "ANN",
        "artifact_type": "keras_scaled_v2",
        "target_variant": str(json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))["target_variant"]),
        "backtest_id": str(backtest["backtest_id"]),
        "dataset_sha256": _dataset_sha(root),
        "candidate_model_sha256": sha256_file(root / "model.keras"),
        "scaler_manifest_sha256": sha256_file(root / "scaler_manifest.json"),
        "git_sha": str(lineage["git_sha"]),
        "calibration_id": str(calibration.get("calibration_id") or ""),
        "calibration_sha256": sha256_file(config.calibration_dir / "calibration.json"),
    }
    try:
        for key, value in tags.items():
            client.set_model_version_tag(config.registered_model_name, version_text, key, value)
        if _alias(client, config.registered_model_name, "candidate") != candidate_before or _alias(client, config.registered_model_name, "champion") != champion_before or _alias(client, config.registered_model_name, "stable") != stable_before:
            raise ANNRegistryError("Registry aliases changed during version creation.")
        client.set_registered_model_alias(config.registered_model_name, "candidate", version_text)
        if _alias(client, config.registered_model_name, "candidate") != version_text:
            raise ANNRegistryError("Candidate alias update did not persist.")
        if _alias(client, config.registered_model_name, "champion") != champion_before or _alias(client, config.registered_model_name, "stable") != stable_before:
            raise ANNRegistryError("Candidate registration mutated champion or stable.")
        body = {
            "schema_version": REGISTRATION_SCHEMA,
            "registered_model_name": config.registered_model_name,
            "model_version": version_text,
            "run_id": config.run_id,
            "model_uri": model_uri,
            "backtest_id": backtest["backtest_id"],
            "candidate_model_sha256": sha256_file(root / "model.keras"),
            "scaler_manifest_sha256": sha256_file(root / "scaler_manifest.json"),
            "dataset_sha256": _dataset_sha(root),
            "previous_candidate": candidate_before,
            "candidate_after": version_text,
            "champion_before": champion_before,
            "champion_after": champion_before,
            "stable_before": stable_before,
            "stable_after": stable_before,
            "tags": tags,
        }
        registration_id = sha256(_canonical(body)).hexdigest()
        receipt = ANNRegistrationReceipt(registration_id=registration_id, **{key: value for key, value in body.items() if key != "schema_version"})
        target = config.output_root / registration_id
        target.mkdir(parents=True, exist_ok=False)
        (target / "receipt.json").write_text(json.dumps(receipt.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return receipt
    except Exception as exc:
        try:
            current = _alias(client, config.registered_model_name, "candidate")
            if current == version_text:
                if candidate_before is None:
                    client.delete_registered_model_alias(config.registered_model_name, "candidate")
                else:
                    client.set_registered_model_alias(config.registered_model_name, "candidate", candidate_before)
        except Exception as compensation:
            raise ANNRegistryReconciliationError("ANN candidate registration failed and alias compensation was unsafe.") from compensation
        raise ANNRegistryError("ANN candidate registration failed; candidate alias was compensated when safe.") from exc


def _alias(client: Any, model_name: str, name: str) -> str | None:
    try:
        value = client.get_model_version_by_alias(model_name, name)
    except LookupError:
        return None
    except Exception as exc:
        if getattr(exc, "error_code", None) in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
            return None
        raise
    return None if value is None else str(value.version)


def _make_pyfunc_model(mlflow: Any) -> Any:
    """Build the custom PythonModel class without importing MLflow at module load."""
    base = mlflow.pyfunc.PythonModel

    class ANNModel(base):
        def load_context(self, context: Any) -> None:
            self._predictor = load_v2_ann_bundle(context.artifacts["bundle"])

        def predict(self, context: Any, model_input: Any) -> pd.DataFrame:
            frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
            return pd.DataFrame({"Wind_Production": self._predictor.predict(frame)})

    return ANNModel()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ANNRegistryError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ANNRegistryError(f"JSON artifact must contain an object: {path}")
    return value


def _validate_signature(signature: Any, feature_names: list[str]) -> None:
    if signature is None:
        raise ANNRegistryError("Logged ANN model has no signature.")
    inputs = list(getattr(getattr(signature, "inputs", signature), "inputs", getattr(signature, "inputs", ())))
    outputs = list(getattr(getattr(signature, "outputs", signature), "inputs", getattr(signature, "outputs", ())))
    if [str(getattr(item, "name", "")) for item in inputs] != feature_names:
        raise ANNRegistryError("Logged ANN signature input order differs from features.")
    if not inputs or not all(_numeric_signature_type(item) for item in inputs):
        raise ANNRegistryError("Logged ANN signature inputs must all be numeric.")
    if len(outputs) != 1 or not str(getattr(outputs[0], "name", "")) or not _numeric_signature_type(outputs[0]):
        raise ANNRegistryError("Logged ANN signature must contain one named numeric output.")


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


def _prediction_values(value: Any) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return np.asarray(value, dtype=float).reshape(-1)


def _dataset_sha(root: Path) -> str:
    payload = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    return str(payload["sha256"])


def _overlap(first: Path, second: Path) -> bool:
    left, right = first.resolve(), second.resolve()
    return left == right or left in right.parents or right in left.parents


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = [
    "ANNRegistryError",
    "ANNRegistryReconciliationError",
    "ANNRegistrationConfig",
    "ANNRunConfig",
    "ANNRegistrationReceipt",
    "ANNRunReceipt",
    "log_ann_candidate_run",
    "register_ann_candidate",
]
