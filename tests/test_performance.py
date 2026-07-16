import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wind_forecast.config as config_module
from wind_forecast.config import (
    PERFORMANCE_ARTIFACT_DIR_ENV,
    load_performance_artifacts_config,
)
from wind_forecast.performance import (
    ACTUAL_COLUMN,
    PREDICTED_COLUMN,
    InvalidPerformanceIntervalError,
    NoPerformanceObservationsError,
    PerformanceArtifactError,
    PerformanceArtifactPaths,
    PerformanceArtifactsNotConfiguredError,
    PerformanceService,
)
from wind_forecast.schemas import DATE_COLUMN
from wind_forecast.training import calculate_regression_metrics


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_valid_artifacts(
    artifact_dir: Path,
    *,
    dates: tuple[str, ...] = ("2026-01-01", "2026-01-02", "2026-01-03"),
    actual: tuple[float, ...] = (100.0, 120.0, 140.0),
    predicted: tuple[float, ...] = (90.0, 125.0, 150.0),
    dataset_version: str | None = None,
) -> PerformanceArtifactPaths:
    artifact_dir.mkdir()
    paths = PerformanceArtifactPaths.from_directory(artifact_dir)
    pd.DataFrame(
        {
            DATE_COLUMN: dates,
            ACTUAL_COLUMN: actual,
            PREDICTED_COLUMN: predicted,
        }
    ).to_csv(paths.predictions, index=False, lineterminator="\n")
    metrics = calculate_regression_metrics(
        pd.Series(actual), np.asarray(predicted, dtype=float)
    )
    _write_json(paths.metrics, metrics)
    summary = {
        "model_type": "extra_trees",
        "seed": 42,
        "test_fraction": 0.2,
        "test_row_count": len(dates),
        "test_start_date": dates[0],
        "test_end_date": dates[-1],
        "metrics": metrics,
    }
    if dataset_version is not None:
        summary["dataset_version"] = dataset_version
    _write_json(paths.summary, summary)
    return paths


def test_load_performance_config_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv(PERFORMANCE_ARTIFACT_DIR_ENV, raising=False)

    config = load_performance_artifacts_config()

    assert config.artifact_dir is None


def test_load_performance_config_resolves_relative_path(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(config_module, "project_root", lambda: tmp_path)
    monkeypatch.setenv(PERFORMANCE_ARTIFACT_DIR_ENV, "outputs/training/baseline")

    config = load_performance_artifacts_config()

    assert config.artifact_dir == (tmp_path / "outputs/training/baseline").resolve()
    assert config.artifact_dir.exists() is False


def test_load_performance_config_preserves_absolute_path(monkeypatch, tmp_path: Path):
    artifact_dir = tmp_path / "mounted-results"
    monkeypatch.setenv(PERFORMANCE_ARTIFACT_DIR_ENV, str(artifact_dir))

    config = load_performance_artifacts_config()

    assert config.artifact_dir == artifact_dir.resolve()


def test_artifact_paths_use_confirmed_filenames(tmp_path: Path):
    paths = PerformanceArtifactPaths.from_directory(tmp_path)

    assert paths.predictions == tmp_path / "predictions.csv"
    assert paths.metrics == tmp_path / "metrics.json"
    assert paths.summary == tmp_path / "run_summary.json"


def test_service_returns_valid_full_period_without_mutating_artifacts(tmp_path: Path):
    paths = _write_valid_artifacts(tmp_path / "artifacts", dataset_version="v1")
    before = {path: path.read_bytes() for path in vars(paths).values()}
    service = PerformanceService(paths)

    report = service.get_performance()

    assert report.observation_count == 3
    assert report.interval.requested_start_date is None
    assert report.interval.requested_end_date is None
    assert report.interval.available_start_date == "2026-01-01"
    assert report.interval.available_end_date == "2026-01-03"
    assert report.interval.returned_start_date == "2026-01-01"
    assert report.interval.returned_end_date == "2026-01-03"
    assert report.result.model_type == "extra_trees"
    assert report.result.seed == 42
    assert report.result.test_fraction == 0.2
    assert report.result.dataset_version == "v1"
    assert report.metrics == report.result.artifact_metrics
    assert [item.error for item in report.observations] == [-10.0, 5.0, 10.0]
    assert {path: path.read_bytes() for path in vars(paths).values()} == before


def test_service_filters_inclusive_interval_and_handles_single_row_r2(tmp_path: Path):
    paths = _write_valid_artifacts(tmp_path / "artifacts")
    service = PerformanceService(paths)

    report = service.get_performance(
        start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
    )

    assert report.observation_count == 1
    assert report.interval.requested_start_date == "2026-01-02"
    assert report.interval.requested_end_date == "2026-01-02"
    assert report.interval.returned_start_date == "2026-01-02"
    assert report.interval.returned_end_date == "2026-01-02"
    assert report.metrics.r2 is None
    assert report.metrics.mae == 5.0
    assert report.metrics.rmse == 5.0
    assert report.observations[0].error == 5.0
    assert report.result.dataset_version is None


def test_service_supports_one_sided_bounds(tmp_path: Path):
    service = PerformanceService(
        _write_valid_artifacts(tmp_path / "artifacts")
    )

    from_start = service.get_performance(start_date=date(2026, 1, 2))
    through_end = service.get_performance(end_date=date(2026, 1, 2))

    assert [item.date for item in from_start.observations] == [
        "2026-01-02",
        "2026-01-03",
    ]
    assert [item.date for item in through_end.observations] == [
        "2026-01-01",
        "2026-01-02",
    ]


def test_service_preserves_current_zero_mape_semantics(tmp_path: Path):
    service = PerformanceService(
        _write_valid_artifacts(
            tmp_path / "artifacts",
            actual=(0.0, 120.0, 140.0),
            predicted=(0.0, 125.0, 150.0),
        )
    )

    report = service.get_performance(start_date=date(2026, 1, 1))

    expected = calculate_regression_metrics(
        pd.Series([0.0, 120.0, 140.0]), np.asarray([0.0, 125.0, 150.0])
    )
    assert report.metrics.mape_percent == expected["MAPE (%)"]


def test_service_rejects_inverted_interval_before_reading_artifacts():
    service = PerformanceService.from_directory(None)

    with pytest.raises(InvalidPerformanceIntervalError, match="start_date"):
        service.get_performance(
            start_date=date(2026, 1, 2), end_date=date(2026, 1, 1)
        )


def test_service_rejects_interval_without_observations(tmp_path: Path):
    service = PerformanceService(
        _write_valid_artifacts(tmp_path / "artifacts")
    )

    with pytest.raises(NoPerformanceObservationsError, match="No performance"):
        service.get_performance(start_date=date(2027, 1, 1))


def test_service_rejects_unconfigured_artifacts():
    service = PerformanceService.from_directory(None)

    with pytest.raises(
        PerformanceArtifactsNotConfiguredError, match="not configured"
    ):
        service.get_performance()


def test_service_rejects_missing_artifact_without_exposing_base_path(tmp_path: Path):
    service = PerformanceService.from_directory(tmp_path / "private-results")

    with pytest.raises(PerformanceArtifactError) as exc_info:
        service.get_performance()

    assert "predictions.csv" in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_service_rejects_empty_predictions(tmp_path: Path):
    paths = _write_valid_artifacts(tmp_path / "artifacts")
    paths.predictions.write_text("", encoding="utf-8")

    with pytest.raises(PerformanceArtifactError, match="predictions.csv is empty"):
        PerformanceService(paths).get_performance()


def test_service_rejects_blank_predictions(tmp_path: Path):
    paths = _write_valid_artifacts(tmp_path / "artifacts")
    paths.predictions.write_text("\n", encoding="utf-8")

    with pytest.raises(PerformanceArtifactError, match="predictions.csv is malformed"):
        PerformanceService(paths).get_performance()


@pytest.mark.parametrize(
    "replacement, message",
    [
        (
            "Date,Actual_Wind_Production\n2026-01-01,100\n",
            "must contain exactly",
        ),
        (
            "Date,Actual_Wind_Production,Predicted_Wind_Production\n"
            "2026-01-01,100,90\n2026-01-01,120,125\n",
            "duplicate dates",
        ),
        (
            "Date,Actual_Wind_Production,Predicted_Wind_Production\n"
            "2026-01-02,100,90\n2026-01-01,120,125\n",
            "strictly chronological",
        ),
        (
            "Date,Actual_Wind_Production,Predicted_Wind_Production\n"
            "2026-01-01,100,NaN\n",
            "invalid values",
        ),
    ],
)
def test_service_rejects_invalid_prediction_schema_and_values(
    tmp_path: Path, replacement: str, message: str
):
    paths = _write_valid_artifacts(tmp_path / "artifacts")
    paths.predictions.write_text(replacement, encoding="utf-8")

    with pytest.raises(PerformanceArtifactError, match=message):
        PerformanceService(paths).get_performance()


def test_service_rejects_invalid_metric_schema(tmp_path: Path):
    paths = _write_valid_artifacts(tmp_path / "artifacts")
    _write_json(paths.metrics, {"MAE": 1.0})

    with pytest.raises(PerformanceArtifactError, match="contain exactly"):
        PerformanceService(paths).get_performance()


def test_service_rejects_non_finite_metric(tmp_path: Path):
    paths = _write_valid_artifacts(tmp_path / "artifacts")
    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
    metrics["RMSE"] = float("inf")
    _write_json(paths.metrics, metrics)

    with pytest.raises(PerformanceArtifactError, match="non-finite RMSE"):
        PerformanceService(paths).get_performance()


def test_service_rejects_invalid_summary_metadata(tmp_path: Path):
    paths = _write_valid_artifacts(tmp_path / "artifacts")
    _write_json(paths.summary, {"model_type": "extra_trees"})

    with pytest.raises(PerformanceArtifactError, match="missing required"):
        PerformanceService(paths).get_performance()


def test_service_rejects_inconsistent_summary_coverage(tmp_path: Path):
    paths = _write_valid_artifacts(tmp_path / "artifacts")
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    summary["test_end_date"] = "2026-01-04"
    _write_json(paths.summary, summary)

    with pytest.raises(PerformanceArtifactError, match="test_end_date differs"):
        PerformanceService(paths).get_performance()


def test_service_rejects_inconsistent_persisted_metrics(tmp_path: Path):
    paths = _write_valid_artifacts(tmp_path / "artifacts")
    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
    metrics["MAE"] += 1.0
    _write_json(paths.metrics, metrics)

    with pytest.raises(PerformanceArtifactError, match="differ"):
        PerformanceService(paths).get_performance()
