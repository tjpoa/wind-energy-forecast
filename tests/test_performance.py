import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wind_forecast.config as config_module
from wind_forecast.config import (
    PERFORMANCE_ARTIFACT_DIR_ENV,
    PerformanceArtifactsConfig,
    load_performance_artifacts_config,
)
from wind_forecast.performance import (
    ACTUAL_COLUMN,
    PREDICTED_COLUMN,
    InvalidPerformanceIntervalError,
    NoPerformanceObservationsError,
    PerformanceArtifactDateError,
    PerformanceArtifactEmptyError,
    PerformanceArtifactMissingError,
    PerformanceArtifactPaths,
    PerformanceArtifactSchemaError,
    PerformanceArtifactValueError,
    PerformanceArtifactsNotConfiguredError,
    PerformanceService,
)
from wind_forecast.schemas import DATE_COLUMN
from wind_forecast.training import calculate_regression_metrics


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_artifacts(
    artifact_dir: Path,
    *,
    dates: tuple[str, ...] = ("2026-01-01", "2026-01-02", "2026-01-03"),
    actual: tuple[float, ...] = (100.0, 120.0, 140.0),
    predicted: tuple[float, ...] = (90.0, 125.0, 150.0),
    include_metrics: bool = True,
    include_summary: bool = True,
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
    if include_metrics:
        _write_json(paths.metrics, metrics)
    if include_summary:
        _write_json(
            paths.summary,
            {
                "model_type": "extra_trees",
                "seed": 42,
                "test_fraction": 0.2,
                "test_row_count": len(dates),
                "test_start_date": min(dates),
                "test_end_date": max(dates),
                "metrics": metrics,
            },
        )
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


def test_service_reads_valid_results_and_calculates_signed_and_absolute_error(
    tmp_path: Path,
):
    paths = _write_artifacts(tmp_path / "artifacts")
    before = {path: path.read_bytes() for path in vars(paths).values()}

    report = PerformanceService(paths).get_performance()

    assert report.observation_count == 3
    assert report.interval.available_start_date == "2026-01-01"
    assert report.interval.available_end_date == "2026-01-03"
    assert [item.error for item in report.observations] == [-10.0, 5.0, 10.0]
    assert [item.absolute_error for item in report.observations] == [10.0, 5.0, 10.0]
    assert report.result is not None
    assert report.result.model_type == "extra_trees"
    assert report.result.artifact_metrics == report.metrics
    assert {path: path.read_bytes() for path in vars(paths).values()} == before


def test_service_sorts_valid_observations_by_date(tmp_path: Path):
    paths = _write_artifacts(
        tmp_path / "artifacts",
        dates=("2026-01-03", "2026-01-01", "2026-01-02"),
        actual=(140.0, 100.0, 120.0),
        predicted=(150.0, 90.0, 125.0),
    )

    report = PerformanceService(paths).get_performance()

    assert [item.date for item in report.observations] == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    ]
    assert [item.error for item in report.observations] == [-10.0, 5.0, 10.0]


def test_service_filters_inclusive_interval_and_handles_single_row_r2(
    tmp_path: Path,
):
    service = PerformanceService(_write_artifacts(tmp_path / "artifacts"))

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
    assert report.observations[0].absolute_error == 5.0


def test_service_supports_one_sided_bounds(tmp_path: Path):
    service = PerformanceService(_write_artifacts(tmp_path / "artifacts"))

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


def test_service_can_read_predictions_without_optional_artifacts(tmp_path: Path):
    paths = _write_artifacts(
        tmp_path / "artifacts", include_metrics=False, include_summary=False
    )

    report = PerformanceService(paths).get_performance()

    assert report.observation_count == 3
    assert report.result is None
    assert report.metrics.mae == pytest.approx(25 / 3)


def test_service_keeps_optional_metrics_nullable_when_summary_exists(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts", include_metrics=False)

    report = PerformanceService(paths).get_performance()

    assert report.result is not None
    assert report.result.artifact_metrics is None


def test_service_from_config_uses_explicit_artifact_directory(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")

    report = PerformanceService.from_config(
        PerformanceArtifactsConfig(artifact_dir=paths.predictions.parent)
    ).get_performance()

    assert report.observation_count == 3


def test_service_preserves_current_zero_mape_semantics(tmp_path: Path):
    service = PerformanceService(
        _write_artifacts(
            tmp_path / "artifacts",
            actual=(0.0, 120.0, 140.0),
            predicted=(0.0, 125.0, 150.0),
        )
    )

    report = service.get_performance()

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
    service = PerformanceService(_write_artifacts(tmp_path / "artifacts"))

    with pytest.raises(NoPerformanceObservationsError, match="No performance"):
        service.get_performance(start_date=date(2027, 1, 1))


def test_service_rejects_unconfigured_artifacts():
    with pytest.raises(
        PerformanceArtifactsNotConfiguredError, match="not configured"
    ):
        PerformanceService.from_directory(None).get_performance()


def test_service_rejects_missing_predictions_without_exposing_base_path(
    tmp_path: Path,
):
    service = PerformanceService.from_directory(tmp_path / "private-results")

    with pytest.raises(PerformanceArtifactMissingError) as exc_info:
        service.get_performance()

    assert "predictions.csv" in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_service_rejects_empty_predictions(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    paths.predictions.write_text("", encoding="utf-8")

    with pytest.raises(PerformanceArtifactEmptyError, match="predictions.csv is empty"):
        PerformanceService(paths).get_performance()


def test_service_rejects_missing_prediction_columns(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    paths.predictions.write_text(
        "Date,Actual_Wind_Production\n2026-01-01,100\n", encoding="utf-8"
    )

    with pytest.raises(PerformanceArtifactSchemaError, match="must contain exactly"):
        PerformanceService(paths).get_performance()


def test_service_rejects_invalid_prediction_dates(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    paths.predictions.write_text(
        "Date,Actual_Wind_Production,Predicted_Wind_Production\n"
        "2026-13-01,100,90\n",
        encoding="utf-8",
    )

    with pytest.raises(PerformanceArtifactDateError, match="invalid dates"):
        PerformanceService(paths).get_performance()


def test_service_rejects_duplicate_prediction_dates(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    paths.predictions.write_text(
        "Date,Actual_Wind_Production,Predicted_Wind_Production\n"
        "2026-01-01,100,90\n2026-01-01,120,125\n",
        encoding="utf-8",
    )

    with pytest.raises(PerformanceArtifactDateError, match="duplicate dates"):
        PerformanceService(paths).get_performance()


def test_service_rejects_non_finite_prediction_value(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    paths.predictions.write_text(
        "Date,Actual_Wind_Production,Predicted_Wind_Production\n"
        "2026-01-01,100,NaN\n",
        encoding="utf-8",
    )

    with pytest.raises(PerformanceArtifactValueError, match="invalid values"):
        PerformanceService(paths).get_performance()


def test_service_rejects_empty_optional_metrics_when_present(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    paths.metrics.write_text("", encoding="utf-8")

    with pytest.raises(PerformanceArtifactEmptyError, match="metrics.json is empty"):
        PerformanceService(paths).get_performance()


def test_service_rejects_invalid_optional_metric_schema_when_present(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    _write_json(paths.metrics, {"MAE": 1.0})

    with pytest.raises(PerformanceArtifactSchemaError, match="contain exactly"):
        PerformanceService(paths).get_performance()


def test_service_rejects_non_finite_optional_metric_when_present(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
    metrics["RMSE"] = float("inf")
    _write_json(paths.metrics, metrics)

    with pytest.raises(PerformanceArtifactValueError, match="non-finite RMSE"):
        PerformanceService(paths).get_performance()


def test_service_rejects_invalid_optional_summary_metadata(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    _write_json(paths.summary, {"model_type": "extra_trees"})

    with pytest.raises(PerformanceArtifactSchemaError, match="missing required"):
        PerformanceService(paths).get_performance()


def test_service_rejects_inconsistent_optional_summary_coverage(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    summary["test_end_date"] = "2026-01-04"
    _write_json(paths.summary, summary)

    with pytest.raises(PerformanceArtifactDateError, match="test_end_date differs"):
        PerformanceService(paths).get_performance()


def test_service_rejects_inconsistent_persisted_metrics(tmp_path: Path):
    paths = _write_artifacts(tmp_path / "artifacts")
    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
    metrics["MAE"] += 1.0
    _write_json(paths.metrics, metrics)

    with pytest.raises(PerformanceArtifactSchemaError, match="differ"):
        PerformanceService(paths).get_performance()
