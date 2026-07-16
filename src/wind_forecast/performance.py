"""Read-only domain service for historical prediction performance."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import DATE_COLUMN
from .training import calculate_regression_metrics


ACTUAL_COLUMN = "Actual_Wind_Production"
PREDICTED_COLUMN = "Predicted_Wind_Production"
PREDICTION_COLUMNS = (DATE_COLUMN, ACTUAL_COLUMN, PREDICTED_COLUMN)
METRIC_NAMES = ("R2", "MAE", "RMSE", "MAPE (%)")
METRIC_REL_TOLERANCE = 1e-12
METRIC_ABS_TOLERANCE = 1e-9


class InvalidPerformanceIntervalError(ValueError):
    """Raised when the requested date interval is inverted."""


class NoPerformanceObservationsError(LookupError):
    """Raised when a valid interval contains no performance observations."""


class PerformanceArtifactError(RuntimeError):
    """Raised when configured performance artifacts cannot be served safely."""


class PerformanceArtifactsNotConfiguredError(PerformanceArtifactError):
    """Raised when no performance-artifact directory was configured."""


@dataclass(frozen=True)
class PerformanceArtifactPaths:
    """Required artifacts resolved from one explicitly selected directory."""

    predictions: Path
    metrics: Path
    summary: Path

    @classmethod
    def from_directory(cls, artifact_dir: str | Path) -> "PerformanceArtifactPaths":
        directory = Path(artifact_dir)
        return cls(
            predictions=directory / "predictions.csv",
            metrics=directory / "metrics.json",
            summary=directory / "run_summary.json",
        )


@dataclass(frozen=True)
class PerformanceMetrics:
    """Regression metrics mapped to stable API-facing names."""

    r2: float | None
    mae: float
    rmse: float
    mape_percent: float


@dataclass(frozen=True)
class PerformanceObservation:
    """Actual, predicted, and signed error values for one date."""

    date: str
    actual: float
    predicted: float
    error: float


@dataclass(frozen=True)
class PerformanceInterval:
    """Requested, available, and returned inclusive date bounds."""

    requested_start_date: str | None
    requested_end_date: str | None
    available_start_date: str
    available_end_date: str
    returned_start_date: str
    returned_end_date: str


@dataclass(frozen=True)
class PerformanceResultInfo:
    """Minimum non-path metadata about the evaluated artifact set."""

    model_type: str
    seed: int
    test_fraction: float
    dataset_version: str | None
    evaluation_start_date: str
    evaluation_end_date: str
    artifact_metrics: PerformanceMetrics


@dataclass(frozen=True)
class PerformanceReport:
    """Domain result returned by the performance service."""

    interval: PerformanceInterval
    observation_count: int
    metrics: PerformanceMetrics
    observations: tuple[PerformanceObservation, ...]
    result: PerformanceResultInfo


@dataclass(frozen=True)
class PerformanceService:
    """Load, validate, and query one immutable performance-artifact set."""

    paths: PerformanceArtifactPaths | None

    @classmethod
    def from_directory(cls, artifact_dir: str | Path | None) -> "PerformanceService":
        paths = (
            None
            if artifact_dir is None
            else PerformanceArtifactPaths.from_directory(artifact_dir)
        )
        return cls(paths=paths)

    def get_performance(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> PerformanceReport:
        """Return validated performance observations for an inclusive interval."""
        if start_date is not None and end_date is not None and start_date > end_date:
            raise InvalidPerformanceIntervalError(
                "start_date must be on or before end_date."
            )
        if self.paths is None:
            raise PerformanceArtifactsNotConfiguredError(
                "Performance artifacts are not configured."
            )

        predictions = _read_predictions(self.paths.predictions)
        artifact_metrics = _read_metrics(self.paths.metrics)
        summary = _read_summary(self.paths.summary)
        _validate_artifact_consistency(predictions, artifact_metrics, summary)

        available_start = predictions.iloc[0][DATE_COLUMN].date()
        available_end = predictions.iloc[-1][DATE_COLUMN].date()
        filtered = predictions
        if start_date is not None:
            filtered = filtered.loc[filtered[DATE_COLUMN].dt.date >= start_date]
        if end_date is not None:
            filtered = filtered.loc[filtered[DATE_COLUMN].dt.date <= end_date]
        if filtered.empty:
            raise NoPerformanceObservationsError(
                "No performance observations exist for the requested interval."
            )

        filtered_metrics = _calculate_metrics(filtered)
        observations = tuple(
            PerformanceObservation(
                date=row[DATE_COLUMN].strftime("%Y-%m-%d"),
                actual=float(row[ACTUAL_COLUMN]),
                predicted=float(row[PREDICTED_COLUMN]),
                error=float(row[PREDICTED_COLUMN] - row[ACTUAL_COLUMN]),
            )
            for _, row in filtered.iterrows()
        )
        returned_start = filtered.iloc[0][DATE_COLUMN].date()
        returned_end = filtered.iloc[-1][DATE_COLUMN].date()

        return PerformanceReport(
            interval=PerformanceInterval(
                requested_start_date=_optional_date_text(start_date),
                requested_end_date=_optional_date_text(end_date),
                available_start_date=available_start.isoformat(),
                available_end_date=available_end.isoformat(),
                returned_start_date=returned_start.isoformat(),
                returned_end_date=returned_end.isoformat(),
            ),
            observation_count=len(observations),
            metrics=filtered_metrics,
            observations=observations,
            result=PerformanceResultInfo(
                model_type=summary["model_type"],
                seed=summary["seed"],
                test_fraction=summary["test_fraction"],
                dataset_version=summary.get("dataset_version"),
                evaluation_start_date=summary["test_start_date"],
                evaluation_end_date=summary["test_end_date"],
                artifact_metrics=artifact_metrics,
            ),
        )


def _read_predictions(path: Path) -> pd.DataFrame:
    _ensure_nonempty_file(path, "predictions.csv")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise PerformanceArtifactError("predictions.csv is malformed.") from exc

    if frame.empty:
        raise PerformanceArtifactError("predictions.csv is empty.")
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise PerformanceArtifactError(
            "predictions.csv must contain exactly Date, "
            "Actual_Wind_Production, and Predicted_Wind_Production."
        )

    date_text = frame[DATE_COLUMN]
    date_shape_valid = date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    parsed_dates = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
    if not date_shape_valid.all() or parsed_dates.isna().any():
        raise PerformanceArtifactError("predictions.csv contains invalid dates.")
    if parsed_dates.duplicated().any():
        raise PerformanceArtifactError("predictions.csv contains duplicate dates.")
    if not parsed_dates.is_monotonic_increasing:
        raise PerformanceArtifactError(
            "predictions.csv dates must be strictly chronological."
        )

    numeric_columns: dict[str, pd.Series] = {}
    for column in (ACTUAL_COLUMN, PREDICTED_COLUMN):
        if not frame[column].str.strip().equals(frame[column]):
            raise PerformanceArtifactError(
                f"predictions.csv contains invalid values in {column}."
            )
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise PerformanceArtifactError(
                f"predictions.csv contains invalid values in {column}."
            )
        numeric_columns[column] = values.astype(float)

    return pd.DataFrame(
        {
            DATE_COLUMN: parsed_dates,
            ACTUAL_COLUMN: numeric_columns[ACTUAL_COLUMN],
            PREDICTED_COLUMN: numeric_columns[PREDICTED_COLUMN],
        }
    )


def _read_metrics(path: Path) -> PerformanceMetrics:
    payload = _read_json_object(path, "metrics.json")
    return _metrics_from_payload(payload, "metrics.json")


def _read_summary(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path, "run_summary.json")
    required = {
        "model_type",
        "seed",
        "test_fraction",
        "test_start_date",
        "test_end_date",
    }
    if not required.issubset(payload):
        raise PerformanceArtifactError(
            "run_summary.json is missing required performance metadata."
        )

    model_type = payload["model_type"]
    if (
        not isinstance(model_type, str)
        or not model_type
        or model_type != model_type.strip()
    ):
        raise PerformanceArtifactError("run_summary.json model_type is invalid.")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PerformanceArtifactError("run_summary.json seed is invalid.")
    test_fraction = payload["test_fraction"]
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, int | float)
        or not math.isfinite(float(test_fraction))
        or not 0 < float(test_fraction) < 1
    ):
        raise PerformanceArtifactError("run_summary.json test_fraction is invalid.")

    test_start = _parse_iso_date(payload["test_start_date"], "test_start_date")
    test_end = _parse_iso_date(payload["test_end_date"], "test_end_date")
    if test_start > test_end:
        raise PerformanceArtifactError("run_summary.json test coverage is inverted.")

    dataset_version = payload.get("dataset_version")
    if dataset_version is not None and (
        not isinstance(dataset_version, str)
        or not dataset_version
        or dataset_version != dataset_version.strip()
    ):
        raise PerformanceArtifactError("run_summary.json dataset_version is invalid.")

    normalized = dict(payload)
    normalized["model_type"] = model_type
    normalized["test_fraction"] = float(test_fraction)
    normalized["test_start_date"] = test_start.isoformat()
    normalized["test_end_date"] = test_end.isoformat()
    normalized["dataset_version"] = dataset_version
    return normalized


def _validate_artifact_consistency(
    predictions: pd.DataFrame,
    artifact_metrics: PerformanceMetrics,
    summary: dict[str, Any],
) -> None:
    prediction_start = predictions.iloc[0][DATE_COLUMN].date().isoformat()
    prediction_end = predictions.iloc[-1][DATE_COLUMN].date().isoformat()
    if summary["test_start_date"] != prediction_start:
        raise PerformanceArtifactError(
            "run_summary.json test_start_date differs from predictions.csv."
        )
    if summary["test_end_date"] != prediction_end:
        raise PerformanceArtifactError(
            "run_summary.json test_end_date differs from predictions.csv."
        )

    if "test_row_count" in summary:
        test_row_count = summary["test_row_count"]
        if (
            isinstance(test_row_count, bool)
            or not isinstance(test_row_count, int)
            or test_row_count != len(predictions)
        ):
            raise PerformanceArtifactError(
                "run_summary.json test_row_count differs from predictions.csv."
            )

    if "metrics" in summary:
        summary_metrics = _metrics_from_payload(
            summary["metrics"], "run_summary.json metrics"
        )
        _require_matching_metrics(
            artifact_metrics,
            summary_metrics,
            "run_summary.json metrics differ from metrics.json.",
        )

    calculated_metrics = _calculate_metrics(predictions)
    if calculated_metrics.r2 is None:
        raise PerformanceArtifactError(
            "predictions.csv has too few rows to validate full-run R2."
        )
    _require_matching_metrics(
        artifact_metrics,
        calculated_metrics,
        "metrics.json differs from metrics recalculated from predictions.csv.",
    )


def _calculate_metrics(frame: pd.DataFrame) -> PerformanceMetrics:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = calculate_regression_metrics(
            frame[ACTUAL_COLUMN],
            frame[PREDICTED_COLUMN].to_numpy(dtype=float),
        )
    r2 = None if len(frame) == 1 else _finite_float(values["R2"], "R2")
    return PerformanceMetrics(
        r2=r2,
        mae=_finite_float(values["MAE"], "MAE"),
        rmse=_finite_float(values["RMSE"], "RMSE"),
        mape_percent=_finite_float(values["MAPE (%)"], "MAPE (%)"),
    )


def _read_json_object(path: Path, artifact_name: str) -> dict[str, Any]:
    _ensure_nonempty_file(path, artifact_name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PerformanceArtifactError(f"{artifact_name} is malformed.") from exc
    if not isinstance(payload, dict) or not payload:
        raise PerformanceArtifactError(
            f"{artifact_name} must be a non-empty JSON object."
        )
    return payload


def _ensure_nonempty_file(path: Path, artifact_name: str) -> None:
    try:
        if not path.is_file():
            raise PerformanceArtifactError(
                f"Required performance artifact is missing: {artifact_name}."
            )
        if path.stat().st_size == 0:
            raise PerformanceArtifactError(f"{artifact_name} is empty.")
    except OSError as exc:
        raise PerformanceArtifactError(
            f"Required performance artifact is unavailable: {artifact_name}."
        ) from exc


def _metrics_from_payload(payload: Any, artifact_name: str) -> PerformanceMetrics:
    if not isinstance(payload, dict) or set(payload) != set(METRIC_NAMES):
        raise PerformanceArtifactError(
            f"{artifact_name} must contain exactly R2, MAE, RMSE, and MAPE (%)."
        )
    return PerformanceMetrics(
        r2=_finite_float(payload["R2"], "R2", artifact_name),
        mae=_finite_float(payload["MAE"], "MAE", artifact_name),
        rmse=_finite_float(payload["RMSE"], "RMSE", artifact_name),
        mape_percent=_finite_float(payload["MAPE (%)"], "MAPE (%)", artifact_name),
    )


def _finite_float(
    value: Any,
    field_name: str,
    artifact_name: str = "calculated metrics",
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PerformanceArtifactError(
            f"{artifact_name} contains a non-numeric {field_name}."
        )
    result = float(value)
    if not math.isfinite(result):
        raise PerformanceArtifactError(
            f"{artifact_name} contains a non-finite {field_name}."
        )
    return result


def _require_matching_metrics(
    expected: PerformanceMetrics,
    actual: PerformanceMetrics,
    message: str,
) -> None:
    pairs = (
        (expected.r2, actual.r2),
        (expected.mae, actual.mae),
        (expected.rmse, actual.rmse),
        (expected.mape_percent, actual.mape_percent),
    )
    if any(
        left is None
        or right is None
        or not math.isclose(
            left,
            right,
            rel_tol=METRIC_REL_TOLERANCE,
            abs_tol=METRIC_ABS_TOLERANCE,
        )
        for left, right in pairs
    ):
        raise PerformanceArtifactError(message)


def _parse_iso_date(value: Any, field_name: str) -> date:
    if not isinstance(value, str):
        raise PerformanceArtifactError(
            f"run_summary.json {field_name} is invalid."
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PerformanceArtifactError(
            f"run_summary.json {field_name} is invalid."
        ) from exc
    if parsed.isoformat() != value:
        raise PerformanceArtifactError(
            f"run_summary.json {field_name} must use YYYY-MM-DD."
        )
    return parsed


def _optional_date_text(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = [
    "ACTUAL_COLUMN",
    "InvalidPerformanceIntervalError",
    "NoPerformanceObservationsError",
    "PerformanceArtifactError",
    "PerformanceArtifactPaths",
    "PerformanceArtifactsNotConfiguredError",
    "PerformanceInterval",
    "PerformanceMetrics",
    "PerformanceObservation",
    "PerformanceReport",
    "PerformanceResultInfo",
    "PerformanceService",
    "PREDICTED_COLUMN",
]
