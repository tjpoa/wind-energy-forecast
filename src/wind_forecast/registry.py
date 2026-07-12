"""Validated MLflow candidate registration and auditable alias promotion."""

from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .manifests import sha256_file
from .tracking import TrackingConfig, _load_mlflow, configure_tracking


EXPECTED_METRICS = {"R2", "MAE", "RMSE", "MAPE_percent"}


class RegistryReconciliationError(RuntimeError):
    """Raised when compensating alias operations could not restore prior state."""


@dataclass(frozen=True)
class CandidateReceipt:
    registered_model_name: str
    model_version: str
    run_id: str
    model_uri: str
    dataset_version: str
    dataset_sha256: str
    feature_schema_sha256: str
    alias: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionReceipt:
    registered_model_name: str
    promoted_version: str
    previous_champion_version: str | None
    approval_note: str
    candidate_alias_removed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path) -> "PromotionReceipt":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def register_candidate(
    run_id: str,
    *,
    config: TrackingConfig,
    client: Any | None = None,
    mlflow_module: Any | None = None,
) -> CandidateReceipt:
    """Validate one finished run, register its model, and move candidate."""
    if config.mode == "off":
        raise ValueError("Candidate registration requires tracking mode local.")
    mlflow = mlflow_module or _load_mlflow()
    configure_tracking(config)
    client = client or mlflow.MlflowClient()
    run = client.get_run(run_id)
    status = str(getattr(run.info, "status", ""))
    if status != "FINISHED":
        raise ValueError(f"Run {run_id} is not FINISHED: {status or 'unknown'}.")

    params = dict(run.data.params)
    required = {
        "dataset_version",
        "dataset_sha256",
        "feature_schema_sha256",
        "git_sha",
        "git_dirty",
        "logged_model_uri",
        "target_contract",
    }
    missing = sorted(required.difference(params))
    if missing:
        raise ValueError(f"Run {run_id} is missing required lineage params: {missing}.")
    if params["git_dirty"].lower() not in {"false", "0"}:
        raise ValueError("Dirty Git runs cannot become model candidates.")
    if params["target_contract"] != "original":
        raise ValueError("Only the original-target contract is approved for this model.")
    if params["dataset_version"] != config.dataset_version:
        raise ValueError(
            f"Run dataset version {params['dataset_version']!r} does not match "
            f"approved version {config.dataset_version!r}."
        )
    metric_names = set(run.data.metrics)
    if not EXPECTED_METRICS.issubset(metric_names):
        raise ValueError(
            f"Candidate run is missing required metrics: {sorted(EXPECTED_METRICS - metric_names)}."
        )
    if any(
        not math.isfinite(float(value)) for value in run.data.metrics.values()
    ):
        raise ValueError("Candidate metrics must be present and finite.")

    required_artifacts = {
        "validation": "validation/validation_sample.csv",
        "dataset_manifest": "manifests/dataset_manifest.json",
        "model_manifest": "manifests/model_manifest.json",
        "environment": "environment/environment.json",
        "model": "baseline/model.joblib",
        "metrics": "baseline/metrics.json",
        "predictions": "baseline/predictions.csv",
        "summary": "baseline/run_summary.json",
        "plot": "evaluation/actual_vs_predicted.png",
    }
    evidence = {
        name: _download_run_artifact(mlflow, run_id, artifact_path)
        for name, artifact_path in required_artifacts.items()
    }
    dataset_manifest = _json_object(evidence["dataset_manifest"], "dataset manifest")
    model_manifest = _json_object(evidence["model_manifest"], "model manifest")
    environment = _json_object(evidence["environment"], "environment manifest")
    summary = _json_object(evidence["summary"], "run summary")
    metrics_artifact = _json_object(evidence["metrics"], "metrics artifact")
    _validate_evidence(
        run_id=run_id,
        params=params,
        run_metrics=dict(run.data.metrics),
        dataset_manifest=dataset_manifest,
        model_manifest=model_manifest,
        environment=environment,
        summary=summary,
        metrics_artifact=metrics_artifact,
        model_path=evidence["model"],
        predictions_path=evidence["predictions"],
        plot_path=evidence["plot"],
    )
    validation_path = evidence["validation"]
    validation = pd.read_csv(validation_path)
    if "Expected_Prediction" not in validation:
        raise ValueError("Validation sample is missing Expected_Prediction.")
    features = validation.drop(columns=["Expected_Prediction"])
    expected = validation["Expected_Prediction"].to_numpy(dtype=float)
    loaded_model = mlflow.pyfunc.load_model(params["logged_model_uri"])
    actual = np.asarray(loaded_model.predict(features), dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
    model_info = mlflow.models.get_model_info(params["logged_model_uri"])
    if getattr(model_info, "signature", None) is None:
        raise ValueError("Logged model is missing its MLflow signature.")
    if str(getattr(model_info, "run_id", "")) != run_id:
        raise ValueError("Logged model metadata does not point to the source run.")

    version = mlflow.register_model(
        model_uri=params["logged_model_uri"],
        name=config.registered_model_name,
    )
    version_text = str(version.version)
    version_run_id = getattr(version, "run_id", None)
    if version_run_id is not None and str(version_run_id) != run_id:
        raise ValueError(
            f"Registered version source run mismatch: {version_run_id} != {run_id}."
        )
    tags = {
        "validation_status": "passed",
        "dataset_version": params["dataset_version"],
        "dataset_sha256": params["dataset_sha256"],
        "feature_schema_sha256": params["feature_schema_sha256"],
        "git_sha": params["git_sha"],
        "source_run_id": run_id,
        "target_contract": "original",
        "model_sha256": model_manifest["model_sha256"],
    }
    for key, value in tags.items():
        client.set_model_version_tag(config.registered_model_name, version_text, key, value)
    client.set_registered_model_alias(
        config.registered_model_name, "candidate", version_text
    )
    return CandidateReceipt(
        registered_model_name=config.registered_model_name,
        model_version=version_text,
        run_id=run_id,
        model_uri=params["logged_model_uri"],
        dataset_version=params["dataset_version"],
        dataset_sha256=params["dataset_sha256"],
        feature_schema_sha256=params["feature_schema_sha256"],
    )


def promote_candidate(
    *,
    config: TrackingConfig,
    expected_candidate_version: str,
    expected_champion_version: str | None,
    approval_note: str,
    client: Any | None = None,
    mlflow_module: Any | None = None,
    receipt_path: str | Path | None = None,
) -> PromotionReceipt:
    """Promote the expected candidate using optimistic alias checks."""
    if not approval_note.strip():
        raise ValueError("approval_note must be non-empty.")
    mlflow = mlflow_module or _load_mlflow()
    configure_tracking(config)
    client = client or mlflow.MlflowClient()
    candidate = client.get_model_version_by_alias(
        config.registered_model_name, "candidate"
    )
    candidate_version = str(candidate.version)
    if candidate_version != str(expected_candidate_version):
        raise ValueError(
            f"Candidate changed: expected {expected_candidate_version}, found {candidate_version}."
        )
    candidate_tags = dict(getattr(candidate, "tags", {}))
    if candidate_tags.get("validation_status") != "passed":
        raise ValueError("Candidate has not passed registration validation.")

    champion = _optional_alias(client, config.registered_model_name, "champion")
    current_champion = None if champion is None else str(champion.version)
    expected = None if expected_champion_version in {None, "none"} else str(expected_champion_version)
    if current_champion != expected:
        raise ValueError(
            f"Champion changed: expected {expected or 'none'}, found {current_champion or 'none'}."
        )
    if champion is not None:
        champion_tags = dict(getattr(champion, "tags", {}))
        comparable_fields = (
            "dataset_version",
            "dataset_sha256",
            "feature_schema_sha256",
            "target_contract",
        )
        mismatches = [
            field
            for field in comparable_fields
            if candidate_tags.get(field) != champion_tags.get(field)
        ]
        if mismatches:
            raise ValueError(
                "Candidate and champion evidence is not comparable: "
                f"{mismatches}."
            )
    if receipt_path is not None:
        _prepare_receipt_destination(Path(receipt_path))

    client.set_model_version_tag(
        config.registered_model_name,
        candidate_version,
        "promotion_approval_note",
        approval_note.strip(),
    )
    client.set_model_version_tag(
        config.registered_model_name,
        candidate_version,
        "previous_champion_version",
        current_champion or "none",
    )
    client.set_registered_model_alias(config.registered_model_name, "champion", candidate_version)
    try:
        client.delete_registered_model_alias(config.registered_model_name, "candidate")
    except Exception as exc:
        _restore_aliases(
            client,
            config.registered_model_name,
            promoted_version=candidate_version,
            previous_champion_version=current_champion,
        )
        raise RuntimeError("Candidate removal failed; aliases were restored.") from exc
    receipt = PromotionReceipt(
        registered_model_name=config.registered_model_name,
        promoted_version=candidate_version,
        previous_champion_version=current_champion,
        approval_note=approval_note.strip(),
        candidate_alias_removed=True,
    )
    if receipt_path is not None:
        try:
            write_receipt(receipt, receipt_path)
        except Exception as exc:
            _restore_aliases(
                client,
                config.registered_model_name,
                promoted_version=candidate_version,
                previous_champion_version=current_champion,
            )
            raise RuntimeError("Receipt persistence failed; aliases were restored.") from exc
    return receipt


def rollback_promotion(
    receipt: PromotionReceipt,
    *,
    config: TrackingConfig,
    client: Any | None = None,
    mlflow_module: Any | None = None,
) -> None:
    """Restore aliases only if champion still matches the promotion receipt."""
    mlflow = mlflow_module or _load_mlflow()
    configure_tracking(config)
    client = client or mlflow.MlflowClient()
    if receipt.registered_model_name != config.registered_model_name:
        raise ValueError("Promotion receipt belongs to a different registered model.")
    champion = client.get_model_version_by_alias(
        config.registered_model_name, "champion"
    )
    if str(champion.version) != receipt.promoted_version:
        raise ValueError("Champion changed after this receipt; rollback was not applied.")
    client.set_registered_model_alias(
        config.registered_model_name, "candidate", receipt.promoted_version
    )
    if receipt.previous_champion_version is None:
        client.delete_registered_model_alias(config.registered_model_name, "champion")
    else:
        client.set_registered_model_alias(
            config.registered_model_name,
            "champion",
            receipt.previous_champion_version,
        )


def write_receipt(receipt: CandidateReceipt | PromotionReceipt, path: str | Path) -> Path:
    """Write a deterministic local audit receipt."""
    output = Path(path)
    _prepare_receipt_destination(output)
    text = json.dumps(receipt.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary_path = Path(temporary.name)
        if output.exists():
            raise FileExistsError(f"Receipt already exists: {output}")
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def _download_run_artifact(mlflow: Any, run_id: str, artifact_path: str) -> Path:
    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path
        )
    except Exception as exc:
        raise ValueError(f"Required run artifact is missing: {artifact_path}.") from exc
    return Path(path)


def _optional_alias(client: Any, model_name: str, alias: str) -> Any | None:
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except LookupError:
        return None
    except Exception as exc:
        if _is_missing_alias_error(exc, alias):
            return None
        raise


def _is_missing_alias_error(exc: Exception, alias: str) -> bool:
    """Recognize MLflow's backend-specific response for an absent alias."""
    error_code = getattr(exc, "error_code", None)
    if error_code == "RESOURCE_DOES_NOT_EXIST":
        return True
    return error_code == "INVALID_PARAMETER_VALUE" and str(exc) == (
        f"INVALID_PARAMETER_VALUE: Registered model alias {alias} not found."
    )


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description}: {path}.") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{description.capitalize()} must be a non-empty JSON object.")
    return value


def _validate_evidence(
    *,
    run_id: str,
    params: dict[str, str],
    run_metrics: dict[str, float],
    dataset_manifest: dict[str, Any],
    model_manifest: dict[str, Any],
    environment: dict[str, Any],
    summary: dict[str, Any],
    metrics_artifact: dict[str, Any],
    model_path: Path,
    predictions_path: Path,
    plot_path: Path,
) -> None:
    expected_dataset = {
        "schema_version": "wind_forecast.training_dataset.v1",
        "dataset_version": params["dataset_version"],
        "sha256": params["dataset_sha256"],
        "feature_schema_sha256": params["feature_schema_sha256"],
        "target": "Wind_Production",
    }
    for field, expected in expected_dataset.items():
        if dataset_manifest.get(field) != expected:
            raise ValueError(f"Dataset manifest {field} does not match run lineage.")
    expected_model = {
        "schema_version": "wind_forecast.model_manifest.v1",
        "dataset_version": params["dataset_version"],
        "dataset_sha256": params["dataset_sha256"],
        "feature_schema_sha256": params["feature_schema_sha256"],
        "target_contract": params["target_contract"],
    }
    for field, expected in expected_model.items():
        if model_manifest.get(field) != expected:
            raise ValueError(f"Model manifest {field} does not match run lineage.")
    if model_manifest.get("model_sha256") != sha256_file(model_path):
        raise ValueError("Serialized model checksum does not match model manifest.")
    if environment.get("schema_version") != "wind_forecast.environment.v1":
        raise ValueError("Environment manifest schema is missing or unsupported.")
    if not environment.get("python") or not isinstance(environment.get("packages"), dict):
        raise ValueError("Environment manifest is incomplete.")
    summary_checks = {
        "dataset_version": params["dataset_version"],
        "input_sha256": params["dataset_sha256"],
        "feature_schema_sha256": params["feature_schema_sha256"],
    }
    for field, expected in summary_checks.items():
        if summary.get(field) != expected:
            raise ValueError(f"Run summary {field} does not match run lineage.")
    for metric in EXPECTED_METRICS:
        artifact_name = "MAPE (%)" if metric == "MAPE_percent" else metric
        if artifact_name not in metrics_artifact:
            raise ValueError(f"Metrics artifact is missing {artifact_name}.")
        if not math.isclose(
            float(metrics_artifact[artifact_name]),
            float(run_metrics[metric]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Metric {metric} differs between run and artifact.")
    if predictions_path.stat().st_size == 0 or plot_path.stat().st_size == 0:
        raise ValueError("Prediction or plot evidence is empty.")
    if not params["logged_model_uri"].strip() or not run_id.strip():
        raise ValueError("Source run/model URI evidence is incomplete.")
    for field in ("dataset_sha256", "feature_schema_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", params[field]):
            raise ValueError(f"Run lineage {field} is not a SHA-256 digest.")
    if not re.fullmatch(r"[0-9a-f]{40,64}", params["git_sha"]):
        raise ValueError("Run lineage git_sha is not a commit digest.")


def _prepare_receipt_destination(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir():
        raise ValueError(f"Receipt parent is not a directory: {path.parent}")


def _restore_aliases(
    client: Any,
    model_name: str,
    *,
    promoted_version: str,
    previous_champion_version: str | None,
) -> None:
    errors = []
    try:
        client.set_registered_model_alias(model_name, "candidate", promoted_version)
    except Exception as exc:
        errors.append(exc)
    try:
        if previous_champion_version is None:
            client.delete_registered_model_alias(model_name, "champion")
        else:
            client.set_registered_model_alias(
                model_name, "champion", previous_champion_version
            )
    except Exception as exc:
        errors.append(exc)
    if errors:
        raise RegistryReconciliationError(
            "Alias compensation failed; inspect candidate/champion manually."
        ) from errors[0]


__all__ = [
    "CandidateReceipt",
    "PromotionReceipt",
    "RegistryReconciliationError",
    "promote_candidate",
    "register_candidate",
    "rollback_promotion",
    "write_receipt",
]
