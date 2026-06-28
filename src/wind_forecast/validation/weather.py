"""Validators for weather and pre-feature merged datasets."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from enum import Enum

import numpy as np
import pandas as pd

from wind_forecast.schemas import (
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    DATE_COLUMN,
    RAW_DAY_COLUMN,
    RAW_MONTH_COLUMN,
    RAW_YEAR_COLUMN,
    TARGET_COLUMN,
)

from .common import (
    ValidationReport,
    ValidationSeverity,
    check_empty_dataframe,
    check_required_columns,
    create_report,
)


class WeatherDatasetType(str, Enum):
    """Supported weather matrix types."""

    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    TEMPERATURE = "temperature"


WEATHER_API_COLUMNS = [
    DATE_COLUMN,
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
]
MERGED_BASE_COLUMNS = [
    DATE_COLUMN,
    TARGET_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
]


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


def _station_columns(df: pd.DataFrame, exclude: Collection[str] = ()) -> list[str]:
    excluded = set(exclude)
    return [column for column in df.columns if str(column).isdigit() and column not in excluded]


def _construct_dates(
    df: pd.DataFrame,
    *,
    year_column: str,
    month_column: str,
    day_column: str,
) -> pd.Series:
    if not {year_column, month_column, day_column}.issubset(df.columns):
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(
        {
            "year": _parse_numeric(df[year_column]),
            "month": _parse_numeric(df[month_column]),
            "day": _parse_numeric(df[day_column]),
        },
        errors="coerce",
    )


def _add_date_stats(report: ValidationReport, dates: pd.Series) -> None:
    valid_dates = dates.dropna()
    if valid_dates.empty:
        report.date_range = (None, None)
        return
    report.date_range = (valid_dates.min(), valid_dates.max())
    report.stats["date_min"] = valid_dates.min()
    report.stats["date_max"] = valid_dates.max()


def _add_numeric_stats(report: ValidationReport, values: pd.DataFrame | pd.Series, prefix: str) -> None:
    valid_values = values.stack().dropna() if isinstance(values, pd.DataFrame) else values.dropna()
    if valid_values.empty:
        report.stats[f"{prefix}_min"] = None
        report.stats[f"{prefix}_max"] = None
        return
    report.stats[f"{prefix}_min"] = valid_values.min()
    report.stats[f"{prefix}_max"] = valid_values.max()


def _missing_daily_dates(dates: pd.Series) -> pd.DatetimeIndex:
    valid_dates = pd.DatetimeIndex(dates.dropna().drop_duplicates().sort_values())
    if len(valid_dates) < 2:
        return pd.DatetimeIndex([])
    expected_dates = pd.date_range(valid_dates.min(), valid_dates.max(), freq="D")
    return expected_dates.difference(valid_dates)


def _normalize_dates(values: Collection[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce", format="mixed")).dropna()


def _count_severity(report: ValidationReport) -> None:
    report.stats["error_count"] = len(report.errors)
    report.stats["warning_count"] = len(report.warnings)
    report.stats["info_count"] = len(report.infos)


def _validate_date_series(
    report: ValidationReport,
    dates: pd.Series,
    *,
    column: str,
    duplicate_code: str,
    non_chronological_severity: ValidationSeverity,
) -> None:
    invalid_count = int(dates.isna().sum())
    report.stats["invalid_date_count"] = invalid_count
    if invalid_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "invalid_date",
            f"Column '{column}' contains null or unparseable dates.",
            column=column,
            count=invalid_count,
        )

    valid_dates = dates.dropna()
    duplicate_mask = valid_dates.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    report.stats["duplicate_date_count"] = duplicate_count
    if duplicate_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            duplicate_code,
            f"Column '{column}' contains duplicate dates.",
            column=column,
            count=duplicate_count,
            sample=_sample(valid_dates[duplicate_mask]),
        )

    if len(valid_dates) > 1 and not valid_dates.is_monotonic_increasing:
        report.add_issue(
            non_chronological_severity,
            "non_chronological_date",
            f"Column '{column}' is not sorted chronologically.",
            column=column,
            count=len(valid_dates),
            sample=_sample(valid_dates.head()),
        )

    missing_dates = _missing_daily_dates(dates)
    report.stats["missing_daily_date_count"] = len(missing_dates)
    if len(missing_dates):
        report.add_issue(
            ValidationSeverity.WARNING,
            "missing_daily_date",
            "Dataset has missing daily dates.",
            column=column,
            count=len(missing_dates),
            sample=_sample(missing_dates),
        )


def _validate_numeric_columns(
    report: ValidationReport,
    df: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, pd.Series]:
    parsed_columns = {}
    for column in columns:
        values = _parse_numeric(df[column])
        parsed_columns[column] = values
        invalid_mask = df[column].isna() | values.isna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "invalid_numeric_value",
                f"Column '{column}' contains null or non-numeric values.",
                column=column,
                count=invalid_count,
                sample=_sample(df.loc[invalid_mask, column]),
            )
            continue

        non_finite_mask = ~np.isfinite(values.to_numpy(dtype=float))
        non_finite_count = int(non_finite_mask.sum())
        if non_finite_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "non_finite_numeric",
                f"Column '{column}' contains non-finite values.",
                column=column,
                count=non_finite_count,
                sample=_sample(values[non_finite_mask]),
            )
    return parsed_columns


def validate_weather_matrix(
    df: pd.DataFrame,
    dataset_type: WeatherDatasetType,
    *,
    year_column: str = RAW_YEAR_COLUMN,
    month_column: str = RAW_MONTH_COLUMN,
    day_column: str = RAW_DAY_COLUMN,
    dataset_name: str | None = None,
) -> ValidationReport:
    """Validate a raw historical weather station matrix without mutating it."""
    dataset_type = WeatherDatasetType(dataset_type)
    report = create_report(df, dataset_name or f"{dataset_type.value}_matrix")
    report.stats["row_count"] = len(df)
    report.stats["column_count"] = len(df.columns)

    check_empty_dataframe(report, df)
    check_required_columns(report, df, [year_column, month_column, day_column])
    required_date_columns = {year_column, month_column, day_column}
    if not required_date_columns.issubset(df.columns):
        return report

    station_columns = _station_columns(df, exclude=required_date_columns)
    report.stats["station_count"] = len(station_columns)
    report.stats["station_columns"] = station_columns
    if not station_columns:
        report.add_issue(
            ValidationSeverity.ERROR,
            "no_station_columns",
            "Weather matrix does not contain station columns.",
            count=0,
        )
        return report

    dates = _construct_dates(df, year_column=year_column, month_column=month_column, day_column=day_column)
    _add_date_stats(report, dates)
    _validate_date_series(
        report,
        dates,
        column=DATE_COLUMN,
        duplicate_code="duplicate_daily_date",
        non_chronological_severity=ValidationSeverity.ERROR,
    )

    station_values = df[station_columns].apply(_parse_numeric)
    _add_numeric_stats(report, station_values, "station_value")

    invalid_numeric_mask = df[station_columns].notna() & station_values.isna()
    invalid_numeric_count = int(invalid_numeric_mask.sum().sum())
    report.stats["invalid_station_value_count"] = invalid_numeric_count
    if invalid_numeric_count:
        for column in station_columns:
            column_count = int(invalid_numeric_mask[column].sum())
            if column_count:
                report.add_issue(
                    ValidationSeverity.ERROR,
                    "invalid_numeric_value",
                    f"Station column '{column}' contains non-numeric values.",
                    column=column,
                    count=column_count,
                    sample=_sample(df.loc[invalid_numeric_mask[column], column]),
                )

    non_null_values = station_values.where(station_values.notna())
    finite_mask = np.isfinite(non_null_values.to_numpy(dtype=float))
    finite_mask = pd.DataFrame(finite_mask, index=station_values.index, columns=station_columns)
    non_finite_mask = station_values.notna() & ~finite_mask
    non_finite_count = int(non_finite_mask.sum().sum())
    report.stats["non_finite_station_value_count"] = non_finite_count
    if non_finite_count:
        for column in station_columns:
            column_count = int(non_finite_mask[column].sum())
            if column_count:
                report.add_issue(
                    ValidationSeverity.ERROR,
                    "non_finite_numeric",
                    f"Station column '{column}' contains non-finite values.",
                    column=column,
                    count=column_count,
                    sample=_sample(station_values.loc[non_finite_mask[column], column]),
                )

    missing_mask = df[station_columns].isna()
    missing_count = int(missing_mask.sum().sum())
    report.stats["missing_station_value_count"] = missing_count
    if missing_count:
        report.add_issue(
            ValidationSeverity.WARNING,
            "missing_station_value",
            "Weather matrix contains missing station values.",
            count=missing_count,
        )

    all_null_stations = [column for column in station_columns if df[column].isna().all()]
    report.stats["all_null_station_columns"] = all_null_stations
    for column in all_null_stations:
        report.add_issue(
            ValidationSeverity.WARNING,
            "all_null_station_column",
            f"Station column '{column}' contains only null values.",
            column=column,
            count=len(df),
        )

    all_null_row_mask = df[station_columns].isna().all(axis=1)
    all_null_row_count = int(all_null_row_mask.sum())
    report.stats["all_null_station_row_count"] = all_null_row_count
    if all_null_row_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "all_null_station_row",
            "Weather matrix has rows where all station values are null.",
            count=all_null_row_count,
            sample=_sample(df.index[all_null_row_mask]),
        )

    if dataset_type == WeatherDatasetType.WIND_SPEED:
        invalid_mask = station_values < 0
        invalid_count = int(invalid_mask.sum().sum())
        report.stats["negative_wind_speed_count"] = invalid_count
        if invalid_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "negative_wind_speed",
                "Wind-speed matrix contains negative values.",
                count=invalid_count,
            )
    elif dataset_type == WeatherDatasetType.WIND_DIRECTION:
        invalid_mask = (station_values < 0) | (station_values > 360)
        invalid_count = int(invalid_mask.sum().sum())
        report.stats["invalid_wind_direction_count"] = invalid_count
        if invalid_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "invalid_wind_direction",
                "Wind-direction matrix contains values outside 0 through 360.",
                count=invalid_count,
            )

    _count_severity(report)
    return report


def validate_weather_alignment(
    weather_frames: Mapping[str, pd.DataFrame],
    *,
    year_column: str = RAW_YEAR_COLUMN,
    month_column: str = RAW_MONTH_COLUMN,
    day_column: str = RAW_DAY_COLUMN,
    dataset_name: str = "weather_alignment",
) -> ValidationReport:
    """Validate date and station alignment across weather matrices."""
    report = ValidationReport(dataset_name=dataset_name, row_count=len(weather_frames), column_count=0)
    report.stats["dataset_count"] = len(weather_frames)
    if len(weather_frames) < 2:
        report.add_issue(
            ValidationSeverity.ERROR,
            "insufficient_weather_datasets",
            "At least two weather datasets are required for alignment validation.",
            count=len(weather_frames),
        )
        _count_severity(report)
        return report

    date_sets: dict[str, pd.DatetimeIndex] = {}
    date_orders: dict[str, pd.DatetimeIndex] = {}
    station_sets: dict[str, set[str]] = {}
    station_orders: dict[str, list[str]] = {}
    required_date_columns = {year_column, month_column, day_column}

    for name, frame in weather_frames.items():
        report.stats[f"{name}_row_count"] = len(frame)
        missing_columns = sorted(required_date_columns.difference(frame.columns))
        if missing_columns:
            for column in missing_columns:
                report.add_issue(
                    ValidationSeverity.ERROR,
                    "missing_required_column",
                    f"Dataset '{name}' is missing required column '{column}'.",
                    column=column,
                    count=1,
                )
            continue

        dates = _construct_dates(frame, year_column=year_column, month_column=month_column, day_column=day_column)
        valid_dates = pd.DatetimeIndex(dates.dropna())
        date_sets[name] = pd.DatetimeIndex(valid_dates.drop_duplicates().sort_values())
        date_orders[name] = valid_dates
        stations = _station_columns(frame, exclude=required_date_columns)
        station_sets[name] = set(stations)
        station_orders[name] = stations
        report.stats[f"{name}_date_count"] = len(date_sets[name])
        report.stats[f"{name}_station_count"] = len(stations)

    if not date_sets or not station_sets:
        _count_severity(report)
        return report

    reference_name = next(iter(date_sets))
    reference_dates = date_sets[reference_name]
    reference_order = date_orders[reference_name]
    reference_stations = station_sets[reference_name]
    reference_station_order = station_orders[reference_name]

    for name in date_sets:
        missing_dates = reference_dates.difference(date_sets[name])
        extra_dates = date_sets[name].difference(reference_dates)
        if len(missing_dates) or len(extra_dates):
            report.add_issue(
                ValidationSeverity.ERROR,
                "weather_date_mismatch",
                f"Dataset '{name}' date set does not match '{reference_name}'.",
                count=len(missing_dates) + len(extra_dates),
                sample={"missing": _sample(missing_dates), "extra": _sample(extra_dates)},
            )
        elif not reference_order.equals(date_orders[name]):
            report.add_issue(
                ValidationSeverity.ERROR,
                "weather_date_order_mismatch",
                f"Dataset '{name}' date order does not match '{reference_name}'.",
                count=len(date_orders[name]),
            )

        missing_stations = sorted(reference_stations - station_sets[name])
        extra_stations = sorted(station_sets[name] - reference_stations)
        if missing_stations or extra_stations:
            report.add_issue(
                ValidationSeverity.ERROR,
                "weather_station_mismatch",
                f"Dataset '{name}' station columns do not match '{reference_name}'.",
                count=len(missing_stations) + len(extra_stations),
                sample={"missing": missing_stations[:5], "extra": extra_stations[:5]},
            )
        elif reference_station_order != station_orders[name]:
            report.add_issue(
                ValidationSeverity.ERROR,
                "weather_station_order_mismatch",
                f"Dataset '{name}' station column order does not match '{reference_name}'.",
                count=len(station_orders[name]),
            )

    _count_severity(report)
    return report


def validate_parsed_weather_api_data(
    df: pd.DataFrame,
    *,
    requested_dates: Collection[object] | None = None,
    dataset_name: str = "parsed_weather_api",
) -> ValidationReport:
    """Validate parsed WeatherAPI data in the project schema."""
    report = create_report(df, dataset_name)
    report.stats["row_count"] = len(df)
    report.stats["column_count"] = len(df.columns)

    check_empty_dataframe(report, df)
    check_required_columns(report, df, WEATHER_API_COLUMNS)
    if list(df.columns) != WEATHER_API_COLUMNS:
        report.add_issue(
            ValidationSeverity.ERROR,
            "unexpected_column_order",
            "Parsed WeatherAPI columns do not match the expected order.",
            count=len(df.columns),
            sample=list(df.columns),
        )
    if not set(WEATHER_API_COLUMNS).issubset(df.columns):
        _count_severity(report)
        return report

    dates = _parse_dates(df[DATE_COLUMN])
    _add_date_stats(report, dates)
    _validate_date_series(
        report,
        dates,
        column=DATE_COLUMN,
        duplicate_code="duplicate_daily_date",
        non_chronological_severity=ValidationSeverity.ERROR,
    )

    parsed_numeric = _validate_numeric_columns(
        report,
        df,
        [AVG_TEMPERATURE_COLUMN, AVG_WIND_SPEED_COLUMN, AVG_WIND_DIRECTION_COLUMN],
    )
    for column, values in parsed_numeric.items():
        _add_numeric_stats(report, values, column)

    wind_speed = parsed_numeric.get(AVG_WIND_SPEED_COLUMN)
    if wind_speed is not None:
        negative_count = int((wind_speed < 0).sum())
        report.stats["negative_wind_speed_count"] = negative_count
        if negative_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "negative_wind_speed",
                "Parsed WeatherAPI wind speed contains negative values.",
                column=AVG_WIND_SPEED_COLUMN,
                count=negative_count,
                sample=_sample(wind_speed[wind_speed < 0]),
            )

    wind_direction = parsed_numeric.get(AVG_WIND_DIRECTION_COLUMN)
    if wind_direction is not None:
        invalid_mask = (wind_direction < 0) | (wind_direction > 360)
        invalid_count = int(invalid_mask.sum())
        report.stats["invalid_wind_direction_count"] = invalid_count
        if invalid_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "invalid_wind_direction",
                "Parsed WeatherAPI wind direction is outside 0 through 360.",
                column=AVG_WIND_DIRECTION_COLUMN,
                count=invalid_count,
                sample=_sample(wind_direction[invalid_mask]),
            )

    if requested_dates is not None:
        requested_index = _normalize_dates(requested_dates)
        returned_index = pd.DatetimeIndex(dates.dropna())
        missing_requested_dates = requested_index.difference(returned_index)
        report.stats["requested_date_count"] = len(requested_index)
        report.stats["returned_date_count"] = len(returned_index)
        report.stats["missing_requested_date_count"] = len(missing_requested_dates)
        if len(missing_requested_dates):
            report.add_issue(
                ValidationSeverity.WARNING,
                "partial_requested_date_coverage",
                "Parsed WeatherAPI data does not include every requested date.",
                column=DATE_COLUMN,
                count=len(missing_requested_dates),
                sample=_sample(missing_requested_dates),
            )

    _count_severity(report)
    return report


def validate_merged_base_data(
    df: pd.DataFrame,
    *,
    production_dates: Collection[object] | None = None,
    weather_dates: Collection[object] | None = None,
    dataset_name: str = "merged_base_data",
) -> ValidationReport:
    """Validate merged production/weather base data before feature engineering."""
    report = create_report(df, dataset_name)
    report.stats["row_count"] = len(df)
    report.stats["column_count"] = len(df.columns)

    check_empty_dataframe(report, df)
    check_required_columns(report, df, MERGED_BASE_COLUMNS)
    if not set(MERGED_BASE_COLUMNS).issubset(df.columns):
        _count_severity(report)
        return report

    dates = _parse_dates(df[DATE_COLUMN])
    _add_date_stats(report, dates)
    _validate_date_series(
        report,
        dates,
        column=DATE_COLUMN,
        duplicate_code="duplicate_daily_date",
        non_chronological_severity=ValidationSeverity.ERROR,
    )

    parsed_numeric = _validate_numeric_columns(
        report,
        df,
        [TARGET_COLUMN, AVG_WIND_SPEED_COLUMN, AVG_TEMPERATURE_COLUMN, AVG_WIND_DIRECTION_COLUMN],
    )
    for column, values in parsed_numeric.items():
        _add_numeric_stats(report, values, column)

    production = parsed_numeric.get(TARGET_COLUMN)
    if production is not None:
        negative_count = int((production < 0).sum())
        report.stats["negative_production_count"] = negative_count
        if negative_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "negative_production",
                "Merged base data contains negative production values.",
                column=TARGET_COLUMN,
                count=negative_count,
                sample=_sample(production[production < 0]),
            )

    wind_speed = parsed_numeric.get(AVG_WIND_SPEED_COLUMN)
    if wind_speed is not None:
        negative_count = int((wind_speed < 0).sum())
        report.stats["negative_wind_speed_count"] = negative_count
        if negative_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "negative_wind_speed",
                "Merged base data contains negative wind-speed values.",
                column=AVG_WIND_SPEED_COLUMN,
                count=negative_count,
                sample=_sample(wind_speed[wind_speed < 0]),
            )

    wind_direction = parsed_numeric.get(AVG_WIND_DIRECTION_COLUMN)
    if wind_direction is not None:
        invalid_mask = (wind_direction < 0) | (wind_direction > 360)
        invalid_count = int(invalid_mask.sum())
        report.stats["invalid_wind_direction_count"] = invalid_count
        if invalid_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "invalid_wind_direction",
                "Merged base data has wind direction outside 0 through 360.",
                column=AVG_WIND_DIRECTION_COLUMN,
                count=invalid_count,
                sample=_sample(wind_direction[invalid_mask]),
            )

    returned_dates = pd.DatetimeIndex(dates.dropna())
    if production_dates is not None:
        production_index = _normalize_dates(production_dates)
        unmatched_production_dates = production_index.difference(returned_dates)
        report.stats["unmatched_production_date_count"] = len(unmatched_production_dates)
        if len(unmatched_production_dates):
            report.add_issue(
                ValidationSeverity.WARNING,
                "unmatched_production_dates",
                "Some production dates are absent from merged base data.",
                column=DATE_COLUMN,
                count=len(unmatched_production_dates),
                sample=_sample(unmatched_production_dates),
            )
    if weather_dates is not None:
        weather_index = _normalize_dates(weather_dates)
        unmatched_weather_dates = weather_index.difference(returned_dates)
        report.stats["unmatched_weather_date_count"] = len(unmatched_weather_dates)
        if len(unmatched_weather_dates):
            report.add_issue(
                ValidationSeverity.WARNING,
                "unmatched_weather_dates",
                "Some weather dates are absent from merged base data.",
                column=DATE_COLUMN,
                count=len(unmatched_weather_dates),
                sample=_sample(unmatched_weather_dates),
            )

    _count_severity(report)
    return report
