"""Validators for historical production datasets."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from wind_forecast.schemas import DATE_COLUMN, RAW_DATE_TIME_COLUMN, RAW_WIND_PRODUCTION_COLUMN, TARGET_COLUMN

from .common import (
    ValidationReport,
    ValidationSeverity,
    check_empty_dataframe,
    check_required_columns,
    create_report,
)


def _parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _parse_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _sample(values: Iterable[object], limit: int = 5) -> list[object]:
    sample = []
    for value in list(values)[:limit]:
        if hasattr(value, "isoformat"):
            sample.append(value.isoformat())
        elif hasattr(value, "item"):
            sample.append(value.item())
        else:
            sample.append(value)
    return sample


def _add_date_stats(report: ValidationReport, dates: pd.Series) -> None:
    valid_dates = dates.dropna()
    if valid_dates.empty:
        report.date_range = (None, None)
        return
    report.date_range = (valid_dates.min(), valid_dates.max())
    report.stats["date_min"] = valid_dates.min()
    report.stats["date_max"] = valid_dates.max()


def _add_numeric_stats(report: ValidationReport, values: pd.Series, prefix: str) -> None:
    valid_values = values.dropna()
    if valid_values.empty:
        report.stats[f"{prefix}_min"] = None
        report.stats[f"{prefix}_max"] = None
        return
    report.stats[f"{prefix}_min"] = valid_values.min()
    report.stats[f"{prefix}_max"] = valid_values.max()


def _missing_intervals(dates: pd.Series, freq: str) -> pd.DatetimeIndex:
    valid_dates = pd.DatetimeIndex(dates.dropna().drop_duplicates().sort_values())
    if len(valid_dates) < 2:
        return pd.DatetimeIndex([])
    expected_dates = pd.date_range(valid_dates.min(), valid_dates.max(), freq=freq)
    return expected_dates.difference(valid_dates)


def validate_raw_production_data(
    df: pd.DataFrame,
    *,
    timestamp_column: str = RAW_DATE_TIME_COLUMN,
    target_column: str = RAW_WIND_PRODUCTION_COLUMN,
    dataset_name: str = "raw_production",
) -> ValidationReport:
    """Validate raw 15-minute E-REDES production data without mutating it."""
    report = create_report(df, dataset_name)
    report.stats["row_count"] = len(df)
    report.stats["column_count"] = len(df.columns)

    check_empty_dataframe(report, df)
    check_required_columns(report, df, [timestamp_column, target_column])
    if timestamp_column not in df.columns or target_column not in df.columns:
        return report

    timestamps = _parse_dates(df[timestamp_column])
    targets = _parse_numeric(df[target_column])
    _add_date_stats(report, timestamps)
    _add_numeric_stats(report, targets, "target")

    invalid_timestamp_mask = df[timestamp_column].isna() | timestamps.isna()
    invalid_timestamp_count = int(invalid_timestamp_mask.sum())
    report.stats["invalid_timestamp_count"] = invalid_timestamp_count
    if invalid_timestamp_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "invalid_timestamp",
            f"Column '{timestamp_column}' contains null or unparseable timestamps.",
            column=timestamp_column,
            count=invalid_timestamp_count,
            sample=_sample(df.loc[invalid_timestamp_mask, timestamp_column]),
        )

    invalid_target_mask = df[target_column].isna() | targets.isna()
    invalid_target_count = int(invalid_target_mask.sum())
    report.stats["invalid_target_count"] = invalid_target_count
    if invalid_target_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "invalid_numeric_value",
            f"Column '{target_column}' contains null or non-numeric values.",
            column=target_column,
            count=invalid_target_count,
            sample=_sample(df.loc[invalid_target_mask, target_column]),
        )

    valid_targets = targets.dropna()
    non_finite_mask = ~np.isfinite(valid_targets.to_numpy(dtype=float))
    non_finite_count = int(non_finite_mask.sum())
    report.stats["non_finite_target_count"] = non_finite_count
    if non_finite_count:
        non_finite_values = valid_targets[non_finite_mask]
        report.add_issue(
            ValidationSeverity.ERROR,
            "non_finite_numeric",
            f"Column '{target_column}' contains non-finite values.",
            column=target_column,
            count=non_finite_count,
            sample=_sample(non_finite_values),
        )

    negative_mask = targets < 0
    negative_count = int(negative_mask.sum())
    report.stats["negative_target_count"] = negative_count
    if negative_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "negative_production",
            f"Column '{target_column}' contains negative production values.",
            column=target_column,
            count=negative_count,
            sample=_sample(targets[negative_mask]),
        )

    valid_timestamps = timestamps.dropna()
    duplicate_mask = valid_timestamps.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    duplicate_extra_count = int(valid_timestamps.duplicated().sum())
    report.stats["duplicate_timestamp_count"] = duplicate_count
    report.stats["duplicate_timestamp_extra_count"] = duplicate_extra_count
    if duplicate_count:
        report.add_issue(
            ValidationSeverity.WARNING,
            "duplicate_raw_timestamp",
            f"Column '{timestamp_column}' contains duplicate raw timestamps.",
            column=timestamp_column,
            count=duplicate_count,
            sample=_sample(valid_timestamps[duplicate_mask]),
        )

    if len(valid_timestamps) > 1 and not valid_timestamps.is_monotonic_increasing:
        report.add_issue(
            ValidationSeverity.WARNING,
            "non_chronological_timestamp",
            f"Column '{timestamp_column}' is not sorted chronologically.",
            column=timestamp_column,
            count=len(valid_timestamps),
            sample=_sample(df[timestamp_column].head()),
        )

    missing_intervals = _missing_intervals(timestamps, "15min")
    report.stats["missing_15min_interval_count"] = len(missing_intervals)
    if len(missing_intervals):
        report.add_issue(
            ValidationSeverity.WARNING,
            "missing_15min_interval",
            "Raw production timestamps have missing 15-minute intervals.",
            column=timestamp_column,
            count=len(missing_intervals),
            sample=_sample(missing_intervals),
        )

    return report


def validate_daily_production_data(
    df: pd.DataFrame,
    *,
    date_column: str = DATE_COLUMN,
    target_column: str = TARGET_COLUMN,
    dataset_name: str = "daily_production",
) -> ValidationReport:
    """Validate daily production data without mutating it."""
    report = create_report(df, dataset_name)
    report.stats["row_count"] = len(df)
    report.stats["column_count"] = len(df.columns)

    check_empty_dataframe(report, df)
    check_required_columns(report, df, [date_column, target_column])
    if date_column not in df.columns or target_column not in df.columns:
        return report

    dates = _parse_dates(df[date_column])
    targets = _parse_numeric(df[target_column])
    _add_date_stats(report, dates)
    _add_numeric_stats(report, targets, "target")

    invalid_date_mask = df[date_column].isna() | dates.isna()
    invalid_date_count = int(invalid_date_mask.sum())
    report.stats["invalid_date_count"] = invalid_date_count
    if invalid_date_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "invalid_date",
            f"Column '{date_column}' contains null or unparseable dates.",
            column=date_column,
            count=invalid_date_count,
            sample=_sample(df.loc[invalid_date_mask, date_column]),
        )

    invalid_target_mask = df[target_column].isna() | targets.isna()
    invalid_target_count = int(invalid_target_mask.sum())
    report.stats["invalid_target_count"] = invalid_target_count
    if invalid_target_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "invalid_numeric_value",
            f"Column '{target_column}' contains null or non-numeric values.",
            column=target_column,
            count=invalid_target_count,
            sample=_sample(df.loc[invalid_target_mask, target_column]),
        )

    valid_targets = targets.dropna()
    non_finite_mask = ~np.isfinite(valid_targets.to_numpy(dtype=float))
    non_finite_count = int(non_finite_mask.sum())
    report.stats["non_finite_target_count"] = non_finite_count
    if non_finite_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "non_finite_numeric",
            f"Column '{target_column}' contains non-finite values.",
            column=target_column,
            count=non_finite_count,
            sample=_sample(valid_targets[non_finite_mask]),
        )

    negative_mask = targets < 0
    negative_count = int(negative_mask.sum())
    report.stats["negative_target_count"] = negative_count
    if negative_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "negative_production",
            f"Column '{target_column}' contains negative production values.",
            column=target_column,
            count=negative_count,
            sample=_sample(targets[negative_mask]),
        )

    valid_dates = dates.dropna()
    duplicate_mask = valid_dates.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    report.stats["duplicate_date_count"] = duplicate_count
    if duplicate_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "duplicate_daily_date",
            f"Column '{date_column}' contains duplicate daily dates.",
            column=date_column,
            count=duplicate_count,
            sample=_sample(valid_dates[duplicate_mask]),
        )

    if len(valid_dates) > 1 and not valid_dates.is_monotonic_increasing:
        report.add_issue(
            ValidationSeverity.ERROR,
            "non_chronological_date",
            f"Column '{date_column}' is not sorted chronologically.",
            column=date_column,
            count=len(valid_dates),
            sample=_sample(df[date_column].head()),
        )

    missing_dates = _missing_intervals(dates, "D")
    report.stats["missing_daily_date_count"] = len(missing_dates)
    if len(missing_dates):
        report.add_issue(
            ValidationSeverity.WARNING,
            "missing_daily_date",
            "Daily production data has missing daily dates.",
            column=date_column,
            count=len(missing_dates),
            sample=_sample(missing_dates),
        )

    return report
