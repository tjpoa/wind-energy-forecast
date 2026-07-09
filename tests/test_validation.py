import numpy as np
import pandas as pd
import pytest

from wind_forecast.schemas import (
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    DATE_COLUMN,
    RAW_DATE_TIME_COLUMN,
    RAW_WIND_PRODUCTION_COLUMN,
    TARGET_COLUMN,
)
from wind_forecast.validation.common import (
    ValidationError,
    ValidationReport,
    ValidationSeverity,
    check_duplicate_columns,
    create_report,
)
from wind_forecast.validation.feature_ready import run_synthetic_feature_ready_validation_checks
from wind_forecast.validation.historical import (
    validate_daily_production_data,
    validate_raw_production_data,
)
from wind_forecast.validation.weather import validate_parsed_weather_api_data


def test_validation_report_serializes_and_raises_for_errors():
    report = ValidationReport(dataset_name="synthetic", row_count=2, column_count=3)
    report.add_issue(
        ValidationSeverity.ERROR,
        "bad_value",
        "A bad value was found.",
        column="Example",
        count=1,
        sample=[pd.Timestamp("2026-01-01"), np.int64(3)],
    )

    payload = report.to_dict()

    assert payload["passed"] is False
    assert payload["issues"][0]["sample"] == ["2026-01-01T00:00:00", 3]
    assert "bad_value" in report.format_summary()
    with pytest.raises(ValidationError, match="bad_value"):
        report.raise_for_errors()


def test_common_duplicate_column_check_handles_duplicate_headers():
    frame = pd.DataFrame([[1, 2]], columns=[DATE_COLUMN, DATE_COLUMN])
    report = create_report(frame, "duplicate_columns")

    check_duplicate_columns(report, frame)

    assert report.has_errors
    assert report.errors[0].code == "duplicate_column"
    assert report.errors[0].sample == [DATE_COLUMN]


def test_validate_daily_production_data_passes_valid_data_without_mutating():
    frame = pd.DataFrame(
        {
            DATE_COLUMN: ["2026-01-01", "2026-01-02"],
            TARGET_COLUMN: [100.0, 110.0],
        }
    )
    original = frame.copy(deep=True)

    report = validate_daily_production_data(frame)

    assert report.passed
    assert not report.has_warnings
    assert report.stats["missing_daily_date_count"] == 0
    pd.testing.assert_frame_equal(frame, original)


def test_validate_daily_production_data_reports_invalid_values_and_dates():
    frame = pd.DataFrame(
        {
            DATE_COLUMN: ["2026-01-02", "not-a-date", "2026-01-02"],
            TARGET_COLUMN: [100.0, -1.0, np.inf],
        }
    )

    report = validate_daily_production_data(frame)
    codes = {issue.code for issue in report.issues}

    assert not report.passed
    assert {
        "invalid_date",
        "negative_production",
        "non_finite_numeric",
        "duplicate_daily_date",
    }.issubset(codes)


def test_validate_raw_production_data_warns_on_duplicate_timestamps():
    frame = pd.DataFrame(
        {
            RAW_DATE_TIME_COLUMN: [
                "2026-01-01 00:00",
                "2026-01-01 00:00",
                "2026-01-01 00:15",
            ],
            RAW_WIND_PRODUCTION_COLUMN: [1.0, 2.0, 3.0],
        }
    )

    report = validate_raw_production_data(frame)
    codes = {issue.code for issue in report.issues}

    assert report.passed
    assert "duplicate_raw_timestamp" in codes


def test_validate_parsed_weather_api_data_reports_domain_and_coverage_issues():
    frame = pd.DataFrame(
        {
            DATE_COLUMN: ["2026-01-01", "2026-01-02"],
            AVG_TEMPERATURE_COLUMN: [10.0, "bad"],
            AVG_WIND_SPEED_COLUMN: [4.0, -1.0],
            AVG_WIND_DIRECTION_COLUMN: [90.0, 361.0],
        }
    )

    report = validate_parsed_weather_api_data(
        frame,
        requested_dates=["2026-01-01", "2026-01-02", "2026-01-03"],
    )
    codes = {issue.code for issue in report.issues}

    assert not report.passed
    assert {
        "invalid_numeric_value",
        "negative_wind_speed",
        "invalid_wind_direction",
        "partial_requested_date_coverage",
    }.issubset(codes)


def test_synthetic_feature_ready_validation_checks_pass():
    result = run_synthetic_feature_ready_validation_checks()

    assert result["passed"] is True
    assert result["checks"]["temporary_file_fixture_passes"] is True
    assert result["checks"]["checksum_mismatch_fails"] is True
