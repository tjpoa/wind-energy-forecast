"""MLflow tracking helpers with explicit local-server configuration."""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .paths import project_root


TrackingMode = Literal["local", "off"]

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT_NAME = "wind-energy-forecast-baseline"
DEFAULT_REGISTERED_MODEL_NAME = "wind-energy-forecast-original"
DEFAULT_DATASET_VERSION = "v1"
DEFAULT_TRACKING_DIRNAME = "mlruns"  # legacy FileStore compatibility only

_TRACKING_KEY_PATTERN = re.compile(r"[^A-Za-z0-9_.\-/ ]+")


class MLflowNotInstalledError(RuntimeError):
    """Raised when MLflow tracking is requested but MLflow is unavailable."""


class MLflowTrackingError(RuntimeError):
    """Raised when the configured MLflow tracking service cannot be used."""


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration shared by training, evaluation, and registry commands."""

    mode: TrackingMode = "local"
    tracking_uri: str = DEFAULT_TRACKING_URI
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME
    dataset_version: str = DEFAULT_DATASET_VERSION

    def __post_init__(self) -> None:
        if self.mode not in {"local", "off"}:
            raise ValueError(f"Unsupported tracking mode: {self.mode!r}.")
        for name, value in (
            ("tracking_uri", self.tracking_uri),
            ("experiment_name", self.experiment_name),
            ("registered_model_name", self.registered_model_name),
            ("dataset_version", self.dataset_version),
        ):
            if not value or not value.strip():
                raise ValueError(f"{name} must be non-empty.")


@dataclass(frozen=True)
class ArtifactReference:
    """A local artifact file and optional MLflow artifact directory."""

    path: Path
    artifact_path: str | None = None


@dataclass(frozen=True)
class RunReceipt:
    """Stable identifiers emitted for a completed MLflow run."""

    run_id: str
    experiment_id: str | None
    tracking_uri: str
    model_uri: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "experiment_id": self.experiment_id,
            "model_uri": self.model_uri,
            "run_id": self.run_id,
            "tracking_uri": self.tracking_uri,
        }


def local_tracking_uri(tracking_dir: Path | None = None) -> str:
    """Return the legacy FileStore URI without configuring it as the default."""
    tracking_dir = tracking_dir or project_root() / DEFAULT_TRACKING_DIRNAME
    return tracking_dir.resolve().as_uri()


def configure_tracking(config: TrackingConfig) -> Any:
    """Configure and verify the selected MLflow experiment."""
    if config.mode == "off":
        return None
    mlflow = _load_mlflow()
    try:
        mlflow.set_tracking_uri(config.tracking_uri)
        experiment = mlflow.set_experiment(config.experiment_name)
    except Exception as exc:  # MLflow exposes backend-specific exception types
        raise MLflowTrackingError(
            "Could not connect to the MLflow tracking server at "
            f"{config.tracking_uri}. Start it with `python -m mlflow server "
            "--backend-store-uri sqlite:///var/mlflow/mlflow.db "
            "--artifacts-destination ./var/mlflow/artifacts --host 127.0.0.1 "
            "--port 5000`."
        ) from exc
    return experiment


def configure_local_tracking(
    *,
    tracking_dir: Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> Any:
    """Configure the legacy FileStore for callers using the Phase 4A API."""
    config = TrackingConfig(
        tracking_uri=local_tracking_uri(tracking_dir),
        experiment_name=experiment_name,
    )
    return configure_tracking(config)


@contextmanager
def start_tracking_run(
    run_name: str,
    *,
    config: TrackingConfig,
    tags: Mapping[str, Any] | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Start one configured MLflow run, or reject tracking-off misuse."""
    if config.mode == "off":
        raise ValueError("Cannot start an MLflow run when tracking mode is off.")
    mlflow = _load_mlflow()
    configure_tracking(config)
    normalized_tags = _stringify_mapping(tags)
    with mlflow.start_run(
        run_name=run_name,
        tags=normalized_tags or None,
        nested=nested,
    ) as active_run:
        yield active_run


@contextmanager
def start_local_run(
    run_name: str,
    *,
    tracking_dir: Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tags: Mapping[str, Any] | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Backward-compatible Phase 4A wrapper around a legacy FileStore run."""
    config = TrackingConfig(
        tracking_uri=local_tracking_uri(tracking_dir),
        experiment_name=experiment_name,
    )
    with start_tracking_run(
        run_name,
        config=config,
        tags=tags,
        nested=nested,
    ) as active_run:
        yield active_run


def run_receipt(active_run: Any, config: TrackingConfig, *, model_uri: str | None = None) -> RunReceipt:
    """Build a receipt from an MLflow ActiveRun-like object."""
    info = getattr(active_run, "info", active_run)
    run_id = getattr(info, "run_id", None)
    if not run_id:
        raise ValueError("The MLflow active run does not expose a run_id.")
    return RunReceipt(
        run_id=str(run_id),
        experiment_id=_optional_text(getattr(info, "experiment_id", None)),
        tracking_uri=config.tracking_uri,
        model_uri=model_uri,
    )


def log_run_data(
    *,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    artifact_paths: Iterable[Path | ArtifactReference] | None = None,
) -> None:
    """Log normalized run metadata and explicitly supplied artifact files."""
    mlflow = _load_mlflow()
    normalized_params = _stringify_mapping(params)
    if normalized_params:
        mlflow.log_params(normalized_params)
    normalized_metrics = _numeric_metric_mapping(metrics)
    if normalized_metrics:
        mlflow.log_metrics(normalized_metrics)
    normalized_tags = _stringify_mapping(tags)
    if normalized_tags:
        mlflow.set_tags(normalized_tags)
    for artifact in artifact_paths or ():
        reference = _artifact_reference(artifact)
        if not reference.path.is_file():
            raise FileNotFoundError(f"MLflow artifact not found: {reference.path}")
        mlflow.log_artifact(str(reference.path), artifact_path=reference.artifact_path)


def log_dataset_input(
    frame: Any,
    *,
    source: str,
    name: str,
    target: str,
    context: str,
    digest: str | None = None,
) -> Any:
    """Log dataset metadata/lineage without uploading the source CSV."""
    mlflow = _load_mlflow()
    dataset = mlflow.data.from_pandas(
        frame,
        source=source,
        name=name,
        targets=target,
        digest=digest,
    )
    mlflow.log_input(dataset, context=context)
    return dataset


def log_sklearn_model(
    model: Any,
    *,
    name: str,
    input_example: Any,
    predictions: Any,
) -> str:
    """Log a sklearn model with signature and return its model URI."""
    mlflow = _load_mlflow()
    signature = mlflow.models.infer_signature(input_example, predictions)
    result = mlflow.sklearn.log_model(
        model,
        name=name,
        signature=signature,
        input_example=input_example,
    )
    model_uri = getattr(result, "model_uri", None)
    if model_uri:
        return str(model_uri)
    run = mlflow.active_run()
    run_id = getattr(getattr(run, "info", None), "run_id", None)
    if not run_id:
        raise MLflowTrackingError("MLflow did not return a logged model URI.")
    return f"runs:/{run_id}/{name}"


def git_state(root: Path | None = None) -> dict[str, str | bool]:
    """Return the current commit and dirty flag without mutating Git state."""
    root = (root or project_root()).resolve()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not determine Git lineage for this run.") from exc
    return {"git_sha": sha, "git_dirty": bool(status.strip())}


def flatten_metric_groups(metric_groups: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Flatten grouped metrics into MLflow-friendly metric names."""
    return {
        f"{normalize_tracking_key(group)}.{normalize_tracking_key(metric)}": _finite_float(value, str(metric))
        for group, metrics in metric_groups.items()
        for metric, value in metrics.items()
    }


def normalize_tracking_key(key: str) -> str:
    """Normalize a parameter, metric, or tag key for MLflow tracking."""
    normalized = key.replace("%", "percent")
    normalized = _TRACKING_KEY_PATTERN.sub("_", normalized)
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_.-/ ")
    return normalized or "value"


def _load_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise MLflowNotInstalledError(
            "MLflow tracking was requested, but the 'mlflow' package is not installed. "
            "Install project requirements before enabling tracking."
        ) from exc
    return mlflow


def _stringify_mapping(values: Mapping[str, Any] | None) -> dict[str, str]:
    return {
        normalize_tracking_key(str(key)): _stringify_value(value)
        for key, value in (values or {}).items()
        if value is not None
    }


def _numeric_metric_mapping(values: Mapping[str, Any] | None) -> dict[str, float]:
    return {
        normalize_tracking_key(str(key)): _finite_float(value, str(key))
        for key, value in (values or {}).items()
    }


def _stringify_value(value: Any) -> str:
    return value.as_posix() if isinstance(value, Path) else str(value)


def _finite_float(value: Any, key: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"MLflow metric '{key}' must be finite.")
    return numeric


def _artifact_reference(artifact: Path | ArtifactReference) -> ArtifactReference:
    return artifact if isinstance(artifact, ArtifactReference) else ArtifactReference(Path(artifact))


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "DEFAULT_DATASET_VERSION",
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_REGISTERED_MODEL_NAME",
    "DEFAULT_TRACKING_DIRNAME",
    "DEFAULT_TRACKING_URI",
    "ArtifactReference",
    "MLflowNotInstalledError",
    "MLflowTrackingError",
    "RunReceipt",
    "TrackingConfig",
    "configure_tracking",
    "flatten_metric_groups",
    "git_state",
    "local_tracking_uri",
    "log_dataset_input",
    "log_run_data",
    "log_sklearn_model",
    "normalize_tracking_key",
    "run_receipt",
    "start_local_run",
    "start_tracking_run",
]
