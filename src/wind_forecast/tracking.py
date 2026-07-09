"""Small MLflow helpers for local experiment tracking."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import project_root


DEFAULT_EXPERIMENT_NAME = "wind-energy-forecast-local"
DEFAULT_TRACKING_DIRNAME = "mlruns"

_TRACKING_KEY_PATTERN = re.compile(r"[^A-Za-z0-9_.\-/ ]+")


class MLflowNotInstalledError(RuntimeError):
    """Raised when MLflow tracking is requested but MLflow is unavailable."""


@dataclass(frozen=True)
class ArtifactReference:
    """A local artifact file and optional MLflow artifact directory."""

    path: Path
    artifact_path: str | None = None


def local_tracking_uri(tracking_dir: Path | None = None) -> str:
    """Return the file URI for the local MLflow tracking directory."""
    tracking_dir = tracking_dir or project_root() / DEFAULT_TRACKING_DIRNAME
    return tracking_dir.resolve().as_uri()


def configure_local_tracking(
    *,
    tracking_dir: Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
):
    """Configure MLflow to use a local tracking directory and experiment."""
    mlflow = _load_mlflow()
    mlflow.set_tracking_uri(local_tracking_uri(tracking_dir))
    return mlflow.set_experiment(experiment_name)


@contextmanager
def start_local_run(
    run_name: str,
    *,
    tracking_dir: Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tags: Mapping[str, Any] | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Start a local MLflow run using the configured local tracking store."""
    mlflow = _load_mlflow()
    configure_local_tracking(
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
    )
    normalized_tags = _stringify_mapping(tags)
    with mlflow.start_run(
        run_name=run_name,
        tags=normalized_tags or None,
        nested=nested,
    ) as active_run:
        yield active_run


def log_run_data(
    *,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    artifact_paths: Iterable[Path | ArtifactReference] | None = None,
) -> None:
    """Log basic run metadata to the active MLflow run."""
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

    if artifact_paths:
        for artifact in artifact_paths:
            reference = _artifact_reference(artifact)
            if not reference.path.exists():
                raise FileNotFoundError(f"MLflow artifact not found: {reference.path}")
            mlflow.log_artifact(
                str(reference.path),
                artifact_path=reference.artifact_path,
            )


def flatten_metric_groups(
    metric_groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Flatten grouped metrics into MLflow-friendly metric names."""
    flattened = {}
    for group_name, metrics in metric_groups.items():
        group_key = normalize_tracking_key(group_name)
        for metric_name, value in metrics.items():
            key = f"{group_key}.{normalize_tracking_key(metric_name)}"
            flattened[key] = _finite_float(value, key)
    return flattened


def normalize_tracking_key(key: str) -> str:
    """Normalize a parameter, metric, or tag key for MLflow tracking."""
    normalized = key.replace("%", "percent")
    normalized = _TRACKING_KEY_PATTERN.sub("_", normalized)
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_.-/ ")
    return normalized or "value"


def _load_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise MLflowNotInstalledError(
            "MLflow tracking was requested, but the 'mlflow' package is not "
            "installed. Install project requirements, then rerun with MLflow "
            "enabled."
        ) from exc

    return mlflow


def _stringify_mapping(values: Mapping[str, Any] | None) -> dict[str, str]:
    if not values:
        return {}

    return {
        normalize_tracking_key(str(key)): _stringify_value(value)
        for key, value in values.items()
        if value is not None
    }


def _numeric_metric_mapping(values: Mapping[str, Any] | None) -> dict[str, float]:
    if not values:
        return {}

    return {
        normalize_tracking_key(str(key)): _finite_float(value, str(key))
        for key, value in values.items()
    }


def _stringify_value(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _finite_float(value: Any, key: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"MLflow metric '{key}' must be finite.")
    return numeric


def _artifact_reference(
    artifact: Path | ArtifactReference,
) -> ArtifactReference:
    if isinstance(artifact, ArtifactReference):
        return artifact
    return ArtifactReference(path=Path(artifact))
