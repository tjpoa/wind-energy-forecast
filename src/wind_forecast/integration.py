"""Local-day REN and ERA5-Land v2 integration helpers.

The functions in this module are import-safe: importing the module does not
read local datasets, write outputs, start network calls, or execute pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wind_forecast.data_sources.era5_land import (
    DEFAULT_CALM_THRESHOLD_M_S,
    EXPECTED_STATION_COUNT,
    load_station_mapping,
)


LOCAL_TIMEZONE = "Europe/Lisbon"
LOCAL_TZ = ZoneInfo(LOCAL_TIMEZONE)
REN_EXPECTED_UNIT = "MW"
REN_PRODUCTION_INPUT_COLUMN = "wind_production_mw"
REN_PRODUCTION_OUTPUT_COLUMN = "Wind_Production"
REN_TIMESTAMP_COLUMN = "timestamp"
REN_SOURCE_DATE_COLUMN = "source_date"
REN_UNIT_COLUMN = "unit"
DATE_LOCAL_COLUMN = "date_local"
TRANSFORMATION_VERSION = "integrated_ren_era5_land_v2_local_day_2A.17"

OUTPUT_FILENAMES = {
    "ren_daily": "ren_daily_production_local.csv",
    "era5_daily_points": "era5_land_daily_points_local.csv",
    "era5_daily_aggregate": "era5_land_daily_aggregate_local.csv",
    "daily_merged": "daily_merged.csv",
    "coverage": "coverage.csv",
    "validation": "validation.json",
    "manifest": "manifest.json",
}

REN_DAILY_COLUMNS = (
    DATE_LOCAL_COLUMN,
    REN_PRODUCTION_OUTPUT_COLUMN,
    "production_unit",
    "production_aggregation",
    "ren_interval_count",
    "ren_expected_interval_count",
    "ren_missing_interval_count",
    "ren_timestamp_identity",
    "ren_source_timezone",
    "ren_source_date",
    "ren_first_timestamp_local",
    "ren_last_timestamp_local",
    "ren_first_timestamp_utc",
    "ren_last_timestamp_utc",
)

ERA5_DAILY_POINT_COLUMNS = (
    DATE_LOCAL_COLUMN,
    "station_id",
    "station_name",
    "station_latitude",
    "station_longitude",
    "grid_latitude",
    "grid_longitude",
    "hourly_count",
    "expected_count",
    "missing_count",
    "temperature_2m_c_mean",
    "temperature_2m_c_min",
    "temperature_2m_c_max",
    "temperature_2m_k_mean",
    "temperature_2m_k_min",
    "temperature_2m_k_max",
    "wind_speed_m_s_mean",
    "wind_speed_m_s_max",
    "wind_speed_m_s_std",
    "u10_m_s_mean",
    "v10_m_s_mean",
    "vector_mean_wind_speed_m_s",
    "vector_mean_wind_direction_deg_from",
    "calm_or_near_calm_count",
    "calm_or_near_calm_share",
    "era5_point_status",
)

ERA5_DAILY_AGGREGATE_COLUMNS = (
    DATE_LOCAL_COLUMN,
    "point_count",
    "expected_point_count",
    "missing_point_count",
    "hourly_observation_count",
    "expected_hourly_observation_count",
    "temperature_2m_c_mean",
    "temperature_2m_k_mean",
    "wind_speed_m_s_mean",
    "u10_m_s_mean",
    "v10_m_s_mean",
    "vector_mean_wind_speed_m_s",
    "vector_mean_wind_direction_deg_from",
    "calm_or_near_calm_share",
    "era5_status",
)

COVERAGE_COLUMNS = (
    DATE_LOCAL_COLUMN,
    "ren_status",
    "ren_interval_count",
    "ren_expected_interval_count",
    "ren_missing_interval_count",
    "era5_status",
    "era5_point_count",
    "era5_expected_point_count",
    "era5_missing_point_count",
    "era5_hourly_observation_count",
    "era5_expected_hourly_observation_count",
    "timezone_source_date_mismatch",
    "excluded_downstream",
    "excluded_downstream_reason",
    "integration_ready",
    "coverage_status",
    "message",
)


class IntegrationBuildError(ValueError):
    """Raised when local integrated dataset inputs or outputs are invalid."""


@dataclass(frozen=True)
class IntegrationPaths:
    """Resolved input and output paths for one integrated dataset build."""

    ren_root: Path
    era5_root: Path
    station_mapping: Path
    output_root: Path
    v1_production: Path | None = None

    @property
    def output_files(self) -> dict[str, Path]:
        return {key: self.output_root / filename for key, filename in OUTPUT_FILENAMES.items()}


@dataclass(frozen=True)
class BuildResult:
    """In-memory result from a completed integrated dataset build."""

    paths: IntegrationPaths
    ren_daily: pd.DataFrame
    era5_daily_points: pd.DataFrame
    era5_daily_aggregate: pd.DataFrame
    daily_merged: pd.DataFrame
    coverage: pd.DataFrame
    validation: dict[str, Any]
    manifest: dict[str, Any]
    checksums: dict[str, str]

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-ready build summary."""
        return {
            "output_root": str(self.paths.output_root),
            "verdict": self.validation.get("verdict"),
            "passed": self.validation.get("passed"),
            "coverage_rows": int(len(self.coverage)),
            "daily_merged_rows": int(len(self.daily_merged)),
            "era5_daily_point_rows": int(len(self.era5_daily_points)),
            "era5_daily_aggregate_rows": int(len(self.era5_daily_aggregate)),
            "ren_daily_rows": int(len(self.ren_daily)),
            "coverage_status_counts": self.validation.get("coverage_status_counts", {}),
            "ren_status_counts": self.validation.get("ren_status_counts", {}),
            "era5_status_counts": self.validation.get("era5_status_counts", {}),
            "output_files": {key: str(path) for key, path in self.paths.output_files.items()},
        }


def parse_local_date(value: str | date, argument_name: str = "date") -> date:
    """Parse and validate a YYYY-MM-DD date."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{argument_name} must be formatted as YYYY-MM-DD.") from exc


def iter_local_dates(start_date: str | date, end_date: str | date) -> list[date]:
    """Return every local civil date in an inclusive range."""
    start = parse_local_date(start_date, "start_date")
    end = parse_local_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date.")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def resolve_integration_paths(
    *,
    ren_root: str | Path,
    era5_root: str | Path,
    station_mapping: str | Path,
    output_root: str | Path,
    v1_production: str | Path | None = None,
) -> IntegrationPaths:
    """Return normalized paths without reading or writing datasets."""
    return IntegrationPaths(
        ren_root=Path(ren_root),
        era5_root=Path(era5_root),
        station_mapping=Path(station_mapping),
        output_root=Path(output_root),
        v1_production=Path(v1_production) if v1_production is not None else None,
    )


def discover_ren_paths(ren_root: str | Path, source_date: str | date) -> dict[str, Path]:
    """Return expected REN normalized and status paths for one local date."""
    root = Path(ren_root)
    day = parse_local_date(source_date).isoformat()
    ren_base = root if root.name == "ren" else root / "ren"
    return {
        "normalized_csv": ren_base / "normalized" / f"date={day}" / "production_15min.csv",
        "status_json": ren_base / "metadata" / f"date={day}" / "status.json",
    }


def discover_era5_hourly_files(era5_root: str | Path) -> dict[str, list[Path]]:
    """Discover monthly-bbox hourly CSV partitions by station ID."""
    hourly_root = Path(era5_root) / "hourly"
    if not hourly_root.is_dir():
        raise IntegrationBuildError(f"ERA5 hourly root is missing: {hourly_root}.")

    by_station: dict[str, list[Path]] = {}
    for path in sorted(hourly_root.glob("station_id=*/period=*/hourly.csv")):
        station_part = path.parent.parent.name
        if not station_part.startswith("station_id="):
            continue
        station_id = station_part.split("=", 1)[1]
        by_station.setdefault(station_id, []).append(path)
    return {station_id: sorted(paths) for station_id, paths in sorted(by_station.items())}


def expected_ren_interval_count(local_day: str | date) -> int:
    """Return expected 15-minute physical intervals for a Europe/Lisbon day."""
    return int(_local_day_duration_minutes(local_day) // 15)


def expected_era5_hourly_count(local_day: str | date) -> int:
    """Return expected hourly physical observations for a Europe/Lisbon day."""
    return int(_local_day_duration_minutes(local_day) // 60)


def _local_day_duration_minutes(local_day: str | date) -> int:
    day = parse_local_date(local_day)
    local_start = datetime.combine(day, time.min, tzinfo=LOCAL_TZ)
    local_end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=LOCAL_TZ)
    delta = local_end.astimezone(timezone.utc) - local_start.astimezone(timezone.utc)
    return int(delta.total_seconds() // 60)


def _expected_utc_index_for_local_day(local_day: str | date, *, frequency: str) -> pd.DatetimeIndex:
    day = parse_local_date(local_day)
    local_start = datetime.combine(day, time.min, tzinfo=LOCAL_TZ)
    local_end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=LOCAL_TZ)
    utc_start = pd.Timestamp(local_start.astimezone(timezone.utc))
    utc_end = pd.Timestamp(local_end.astimezone(timezone.utc))
    return pd.date_range(utc_start, utc_end, inclusive="left", freq=frequency)


def _expected_local_naive_index_for_local_day(local_day: str | date, *, frequency: str) -> pd.DatetimeIndex:
    utc_index = _expected_utc_index_for_local_day(local_day, frequency=frequency)
    local_index = utc_index.tz_convert(LOCAL_TIMEZONE).tz_localize(None)
    return pd.DatetimeIndex(local_index)


def _has_timezone_marker(value: object) -> bool:
    text = str(value).strip()
    if text.endswith("Z"):
        return True
    if len(text) < 6:
        return False
    suffix = text[-6:]
    return (
        suffix[0] in {"+", "-"}
        and suffix[1:3].isdigit()
        and suffix[3] == ":"
        and suffix[4:6].isdigit()
    )


def aggregate_ren_daily_local(
    ren_root: str | Path,
    *,
    start_date: str | date,
    end_date: str | date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate REN 15-minute production to Europe/Lisbon local days."""
    dates = iter_local_dates(start_date, end_date)
    daily_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for local_day in dates:
        day_text = local_day.isoformat()
        paths = discover_ren_paths(ren_root, local_day)
        expected_count = expected_ren_interval_count(local_day)
        status = _read_optional_json(paths["status_json"])
        status_validation = status.get("validation") if isinstance(status, Mapping) else {}
        status_name = str((status_validation or {}).get("validation_status") or "").strip()

        if status_name == "unavailable":
            coverage_rows.append(
                _ren_coverage_row(
                    local_day=day_text,
                    status="unavailable",
                    expected_count=expected_count,
                    message="REN status partition records this source date as unavailable.",
                )
            )
            continue

        if not paths["normalized_csv"].is_file():
            coverage_rows.append(
                _ren_coverage_row(
                    local_day=day_text,
                    status="missing",
                    expected_count=expected_count,
                    message=f"REN normalized partition is missing: {paths['normalized_csv']}.",
                )
            )
            continue

        try:
            daily_row, coverage_row = _aggregate_one_ren_partition(paths["normalized_csv"], local_day)
        except IntegrationBuildError as exc:
            message = str(exc)
            status_for_failure = (
                "timezone_source_date_mismatch"
                if "source_date" in message or "timezone" in message or "local date" in message
                else "invalid"
            )
            coverage_row = _ren_coverage_row(
                local_day=day_text,
                status=status_for_failure,
                expected_count=expected_count,
                message=message,
            )
        else:
            if coverage_row["ren_status"] == "complete":
                daily_rows.append(daily_row)
        coverage_rows.append(coverage_row)

    ren_daily = pd.DataFrame(daily_rows, columns=list(REN_DAILY_COLUMNS))
    ren_coverage = pd.DataFrame(coverage_rows)
    return (
        ren_daily.sort_values(DATE_LOCAL_COLUMN).reset_index(drop=True),
        ren_coverage.sort_values(DATE_LOCAL_COLUMN).reset_index(drop=True),
    )


def _ren_coverage_row(
    *,
    local_day: str,
    status: str,
    expected_count: int,
    interval_count: int = 0,
    missing_count: int | None = None,
    timestamp_identity: str | None = None,
    message: str = "",
) -> dict[str, Any]:
    return {
        DATE_LOCAL_COLUMN: local_day,
        "ren_status": status,
        "ren_interval_count": int(interval_count),
        "ren_expected_interval_count": int(expected_count),
        "ren_missing_interval_count": int(expected_count - interval_count if missing_count is None else missing_count),
        "ren_timestamp_identity": timestamp_identity,
        "message": message,
    }


def _aggregate_one_ren_partition(path: Path, local_day: date) -> tuple[dict[str, Any], dict[str, Any]]:
    day_text = local_day.isoformat()
    required_columns = {REN_TIMESTAMP_COLUMN, REN_PRODUCTION_INPUT_COLUMN, REN_UNIT_COLUMN, REN_SOURCE_DATE_COLUMN}
    frame = pd.read_csv(path, dtype={REN_TIMESTAMP_COLUMN: "string", REN_SOURCE_DATE_COLUMN: "string"})
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise IntegrationBuildError(f"REN normalized partition is missing columns: {missing_columns}.")

    expected_count = expected_ren_interval_count(local_day)
    markers = frame[REN_TIMESTAMP_COLUMN].map(_has_timezone_marker)
    if bool(markers.any()) and not bool(markers.all()):
        raise IntegrationBuildError("REN partition mixes timezone-aware and timezone-naive timestamps.")

    values = pd.to_numeric(frame[REN_PRODUCTION_INPUT_COLUMN], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise IntegrationBuildError("REN production contains missing or non-finite values.")
    if (values < 0).any():
        raise IntegrationBuildError("REN production contains negative values.")
    if set(frame[REN_UNIT_COLUMN].astype(str)) != {REN_EXPECTED_UNIT}:
        raise IntegrationBuildError("REN production unit is not consistently MW.")
    if set(frame[REN_SOURCE_DATE_COLUMN].astype(str)) != {day_text}:
        raise IntegrationBuildError("REN source_date values do not match the partition date.")

    if bool(markers.any()):
        timestamp_identity = "utc"
        timestamp_utc = pd.to_datetime(frame[REN_TIMESTAMP_COLUMN].astype(str), utc=True, errors="coerce")
        if timestamp_utc.isna().any():
            raise IntegrationBuildError("REN partition contains unparseable aware timestamps.")
        local_timestamps = timestamp_utc.dt.tz_convert(LOCAL_TIMEZONE)
        local_dates = local_timestamps.dt.strftime("%Y-%m-%d")
        expected_index = _expected_utc_index_for_local_day(local_day, frequency="15min")
        actual_index = pd.DatetimeIndex(timestamp_utc)
        first_utc = actual_index.min().isoformat().replace("+00:00", "Z") if len(actual_index) else None
        last_utc = actual_index.max().isoformat().replace("+00:00", "Z") if len(actual_index) else None
        first_local = local_timestamps.min().isoformat() if len(local_timestamps) else None
        last_local = local_timestamps.max().isoformat() if len(local_timestamps) else None
    else:
        timestamp_identity = "local_wall_clock"
        local_naive = pd.to_datetime(frame[REN_TIMESTAMP_COLUMN], errors="coerce")
        if local_naive.isna().any():
            raise IntegrationBuildError("REN partition contains unparseable naive timestamps.")
        if local_naive.duplicated().any():
            raise IntegrationBuildError(
                "REN naive local wall-clock timestamps are duplicated and cannot disambiguate DST identity."
            )
        local_dates = local_naive.dt.strftime("%Y-%m-%d")
        expected_index = _expected_local_naive_index_for_local_day(local_day, frequency="15min")
        actual_index = pd.DatetimeIndex(local_naive)
        first_utc = None
        last_utc = None
        first_local = actual_index.min().isoformat() if len(actual_index) else None
        last_local = actual_index.max().isoformat() if len(actual_index) else None

    if set(local_dates.astype(str)) != {day_text}:
        raise IntegrationBuildError("REN timestamp-derived Europe/Lisbon local date does not match source_date.")
    if actual_index.duplicated().any():
        raise IntegrationBuildError("REN timestamp identities contain duplicates.")
    if not actual_index.is_monotonic_increasing:
        raise IntegrationBuildError("REN timestamps are not sorted chronologically.")

    unexpected = actual_index.difference(expected_index)
    if len(unexpected):
        raise IntegrationBuildError("REN partition contains timestamps outside the expected local-day interval.")
    missing = expected_index.difference(actual_index)
    interval_count = int(len(frame))
    missing_count = int(len(missing))
    status = "complete" if interval_count == expected_count and missing_count == 0 else "incomplete"
    if interval_count > expected_count:
        raise IntegrationBuildError("REN partition has more observations than expected for the local day.")

    daily_row = {
        DATE_LOCAL_COLUMN: day_text,
        REN_PRODUCTION_OUTPUT_COLUMN: float(values.sum()),
        "production_unit": "sum_of_15_minute_MW_observations",
        "production_aggregation": "sum(wind_production_mw)",
        "ren_interval_count": interval_count,
        "ren_expected_interval_count": expected_count,
        "ren_missing_interval_count": missing_count,
        "ren_timestamp_identity": timestamp_identity,
        "ren_source_timezone": LOCAL_TIMEZONE,
        "ren_source_date": day_text,
        "ren_first_timestamp_local": first_local,
        "ren_last_timestamp_local": last_local,
        "ren_first_timestamp_utc": first_utc,
        "ren_last_timestamp_utc": last_utc,
    }
    coverage_row = _ren_coverage_row(
        local_day=day_text,
        status=status,
        expected_count=expected_count,
        interval_count=interval_count,
        missing_count=missing_count,
        timestamp_identity=timestamp_identity,
        message="" if status == "complete" else "REN partition is incomplete for the expected local-day intervals.",
    )
    return daily_row, coverage_row


def aggregate_era5_daily_local(
    era5_root: str | Path,
    *,
    station_mapping: str | Path,
    start_date: str | date,
    end_date: str | date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute ERA5-Land daily point and aggregate weather over local days."""
    dates = iter_local_dates(start_date, end_date)
    stations = load_station_mapping(station_mapping)
    if len(stations) != EXPECTED_STATION_COUNT:
        raise IntegrationBuildError(f"Expected {EXPECTED_STATION_COUNT} station mappings; found {len(stations)}.")

    hourly_files = discover_era5_hourly_files(era5_root)
    station_rows: list[pd.DataFrame] = []
    missing_station_ids = sorted({station.station_id for station in stations}.difference(hourly_files))
    if missing_station_ids:
        raise IntegrationBuildError(f"Missing ERA5 hourly partitions for station IDs: {missing_station_ids}.")

    for station in stations:
        station_rows.append(_aggregate_one_era5_station(station, hourly_files[station.station_id], dates))

    daily_points = pd.concat(station_rows, ignore_index=True)
    daily_points = daily_points.loc[:, list(ERA5_DAILY_POINT_COLUMNS)].sort_values(
        [DATE_LOCAL_COLUMN, "station_id"]
    )
    daily_aggregate = aggregate_era5_daily_points_local(daily_points, dates, expected_point_count=len(stations))
    return daily_points.reset_index(drop=True), daily_aggregate.reset_index(drop=True)


def _aggregate_one_era5_station(station: Any, hourly_paths: Sequence[Path], dates: Sequence[date]) -> pd.DataFrame:
    required_columns = {
        "timestamp_utc",
        "station_id",
        "station_name",
        "station_latitude",
        "station_longitude",
        "grid_latitude",
        "grid_longitude",
        "temperature_2m_k",
        "temperature_2m_c",
        "u10_m_s",
        "v10_m_s",
        "wind_speed_m_s",
        "is_calm_or_near_calm",
    }
    frames = []
    for path in hourly_paths:
        frame = pd.read_csv(path)
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise IntegrationBuildError(f"ERA5 hourly CSV {path} is missing columns: {missing}.")
        frames.append(frame)
    if not frames:
        raise IntegrationBuildError(f"No ERA5 hourly files were found for station {station.station_id}.")

    hourly = pd.concat(frames, ignore_index=True)
    timestamps_utc = pd.to_datetime(hourly["timestamp_utc"], utc=True, errors="coerce")
    if timestamps_utc.isna().any():
        raise IntegrationBuildError(f"ERA5 hourly timestamps are unparseable for station {station.station_id}.")
    hourly["_timestamp_utc"] = timestamps_utc
    hourly["_timestamp_local"] = timestamps_utc.dt.tz_convert(LOCAL_TIMEZONE)
    hourly[DATE_LOCAL_COLUMN] = hourly["_timestamp_local"].dt.strftime("%Y-%m-%d")
    hourly = hourly.loc[hourly[DATE_LOCAL_COLUMN].isin({item.isoformat() for item in dates})].copy()

    duplicate_count = int(hourly["_timestamp_utc"].duplicated().sum())
    if duplicate_count:
        raise IntegrationBuildError(f"ERA5 station {station.station_id} has {duplicate_count} duplicate UTC hours.")

    station_values = hourly.iloc[0] if not hourly.empty else None
    groups_by_date = {
        str(day): group
        for day, group in hourly.groupby(DATE_LOCAL_COLUMN, sort=False)
    }
    rows = []
    for local_day in dates:
        day_text = local_day.isoformat()
        group = groups_by_date.get(day_text, hourly.iloc[0:0])
        rows.append(_era5_daily_point_row(station, station_values, group, local_day))
    return pd.DataFrame(rows, columns=list(ERA5_DAILY_POINT_COLUMNS))


def _era5_daily_point_row(station: Any, station_values: Any, group: pd.DataFrame, local_day: date) -> dict[str, Any]:
    expected_count = expected_era5_hourly_count(local_day)
    hourly_count = int(group["_timestamp_utc"].nunique()) if not group.empty else 0
    required_numeric = ("temperature_2m_c", "temperature_2m_k", "wind_speed_m_s", "u10_m_s", "v10_m_s")
    required_ok = True
    for column in required_numeric:
        series = pd.to_numeric(group[column], errors="coerce") if column in group else pd.Series(dtype=float)
        if len(series) != hourly_count or series.isna().any() or not np.isfinite(series.to_numpy(dtype=float)).all():
            required_ok = False
    status = "complete" if hourly_count == expected_count and required_ok else ("missing" if hourly_count == 0 else "incomplete")

    mean_u = _finite_mean(group, "u10_m_s")
    mean_v = _finite_mean(group, "v10_m_s")
    vector_speed = (
        math.sqrt(mean_u * mean_u + mean_v * mean_v)
        if math.isfinite(mean_u) and math.isfinite(mean_v)
        else math.nan
    )
    if math.isfinite(vector_speed) and vector_speed >= DEFAULT_CALM_THRESHOLD_M_S:
        vector_direction = (180.0 + math.degrees(math.atan2(mean_u, mean_v))) % 360.0
    else:
        vector_direction = math.nan

    calm_count = _calm_count(group)
    station_name = getattr(station, "station_name", "")
    if station_values is not None:
        station_name = station_values.get("station_name", station_name)

    return {
        DATE_LOCAL_COLUMN: local_day.isoformat(),
        "station_id": station.station_id,
        "station_name": station_name,
        "station_latitude": _station_or_frame_value(station, station_values, "station_latitude", "latitude"),
        "station_longitude": _station_or_frame_value(station, station_values, "station_longitude", "longitude"),
        "grid_latitude": _frame_value(station_values, "grid_latitude"),
        "grid_longitude": _frame_value(station_values, "grid_longitude"),
        "hourly_count": hourly_count,
        "expected_count": expected_count,
        "missing_count": max(expected_count - hourly_count, 0),
        "temperature_2m_c_mean": _finite_mean(group, "temperature_2m_c"),
        "temperature_2m_c_min": _finite_min(group, "temperature_2m_c"),
        "temperature_2m_c_max": _finite_max(group, "temperature_2m_c"),
        "temperature_2m_k_mean": _finite_mean(group, "temperature_2m_k"),
        "temperature_2m_k_min": _finite_min(group, "temperature_2m_k"),
        "temperature_2m_k_max": _finite_max(group, "temperature_2m_k"),
        "wind_speed_m_s_mean": _finite_mean(group, "wind_speed_m_s"),
        "wind_speed_m_s_max": _finite_max(group, "wind_speed_m_s"),
        "wind_speed_m_s_std": _finite_std(group, "wind_speed_m_s"),
        "u10_m_s_mean": mean_u,
        "v10_m_s_mean": mean_v,
        "vector_mean_wind_speed_m_s": vector_speed,
        "vector_mean_wind_direction_deg_from": vector_direction,
        "calm_or_near_calm_count": calm_count,
        "calm_or_near_calm_share": calm_count / hourly_count if hourly_count else math.nan,
        "era5_point_status": status,
    }


def aggregate_era5_daily_points_local(
    daily_points: pd.DataFrame,
    dates: Sequence[date],
    *,
    expected_point_count: int,
) -> pd.DataFrame:
    """Equal-weight aggregate complete local station-day rows."""
    rows = []
    groups_by_date = {
        str(day): group
        for day, group in daily_points.groupby(DATE_LOCAL_COLUMN, sort=False)
    }
    for local_day in dates:
        day_text = local_day.isoformat()
        group = groups_by_date.get(day_text, daily_points.iloc[0:0])
        expected_hourly = expected_era5_hourly_count(local_day) * expected_point_count
        valid_points = group.loc[
            (group["era5_point_status"] == "complete")
            & group[["temperature_2m_c_mean", "temperature_2m_k_mean", "wind_speed_m_s_mean", "u10_m_s_mean", "v10_m_s_mean"]]
            .notna()
            .all(axis=1)
        ]
        point_count = int(len(valid_points))
        mean_u = _finite_mean(valid_points, "u10_m_s_mean")
        mean_v = _finite_mean(valid_points, "v10_m_s_mean")
        vector_speed = (
            math.sqrt(mean_u * mean_u + mean_v * mean_v)
            if math.isfinite(mean_u) and math.isfinite(mean_v)
            else math.nan
        )
        if math.isfinite(vector_speed) and vector_speed >= DEFAULT_CALM_THRESHOLD_M_S:
            vector_direction = (180.0 + math.degrees(math.atan2(mean_u, mean_v))) % 360.0
        else:
            vector_direction = math.nan
        calm_count = int(pd.to_numeric(valid_points["calm_or_near_calm_count"], errors="coerce").fillna(0).sum())
        hourly_count = int(pd.to_numeric(valid_points["hourly_count"], errors="coerce").fillna(0).sum())
        status = "complete" if point_count == expected_point_count and hourly_count == expected_hourly else "incomplete"
        rows.append(
            {
                DATE_LOCAL_COLUMN: day_text,
                "point_count": point_count,
                "expected_point_count": expected_point_count,
                "missing_point_count": expected_point_count - point_count,
                "hourly_observation_count": hourly_count,
                "expected_hourly_observation_count": expected_hourly,
                "temperature_2m_c_mean": _finite_mean(valid_points, "temperature_2m_c_mean"),
                "temperature_2m_k_mean": _finite_mean(valid_points, "temperature_2m_k_mean"),
                "wind_speed_m_s_mean": _finite_mean(valid_points, "wind_speed_m_s_mean"),
                "u10_m_s_mean": mean_u,
                "v10_m_s_mean": mean_v,
                "vector_mean_wind_speed_m_s": vector_speed,
                "vector_mean_wind_direction_deg_from": vector_direction,
                "calm_or_near_calm_share": calm_count / hourly_count if hourly_count else math.nan,
                "era5_status": status,
            }
        )
    return pd.DataFrame(rows, columns=list(ERA5_DAILY_AGGREGATE_COLUMNS))


def build_coverage_table(
    dates: Sequence[date],
    ren_coverage: pd.DataFrame,
    era5_daily_aggregate: pd.DataFrame,
) -> pd.DataFrame:
    """Build one explicit coverage row for every requested local date."""
    ren_by_date = ren_coverage.set_index(DATE_LOCAL_COLUMN).to_dict(orient="index")
    era5_by_date = era5_daily_aggregate.set_index(DATE_LOCAL_COLUMN).to_dict(orient="index")
    rows = []
    for local_day in dates:
        day_text = local_day.isoformat()
        ren = ren_by_date.get(day_text, {})
        era5 = era5_by_date.get(day_text, {})
        ren_status = str(ren.get("ren_status") or "missing")
        era5_status = str(era5.get("era5_status") or "missing")
        timezone_mismatch = ren_status == "timezone_source_date_mismatch"
        excluded = ren_status == "unavailable"
        integration_ready = ren_status == "complete" and era5_status == "complete"
        coverage_status = _coverage_status(
            ren_status=ren_status,
            era5_status=era5_status,
            timezone_mismatch=timezone_mismatch,
            excluded=excluded,
            integration_ready=integration_ready,
        )
        rows.append(
            {
                DATE_LOCAL_COLUMN: day_text,
                "ren_status": ren_status,
                "ren_interval_count": _int_value(ren.get("ren_interval_count")),
                "ren_expected_interval_count": _int_value(
                    ren.get("ren_expected_interval_count"),
                    expected_ren_interval_count(local_day),
                ),
                "ren_missing_interval_count": _int_value(ren.get("ren_missing_interval_count")),
                "era5_status": era5_status,
                "era5_point_count": _int_value(era5.get("point_count")),
                "era5_expected_point_count": _int_value(era5.get("expected_point_count"), EXPECTED_STATION_COUNT),
                "era5_missing_point_count": _int_value(era5.get("missing_point_count")),
                "era5_hourly_observation_count": _int_value(era5.get("hourly_observation_count")),
                "era5_expected_hourly_observation_count": _int_value(
                    era5.get("expected_hourly_observation_count"),
                    expected_era5_hourly_count(local_day) * EXPECTED_STATION_COUNT,
                ),
                "timezone_source_date_mismatch": bool(timezone_mismatch),
                "excluded_downstream": bool(excluded),
                "excluded_downstream_reason": "REN unavailable" if excluded else "",
                "integration_ready": bool(integration_ready),
                "coverage_status": coverage_status,
                "message": str(ren.get("message") or ""),
            }
        )
    return pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))


def join_integrated_daily(
    ren_daily: pd.DataFrame,
    era5_daily_aggregate: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Join integration-ready local-day production and weather rows."""
    ready_dates = coverage.loc[coverage["integration_ready"], DATE_LOCAL_COLUMN]
    ren_ready = ren_daily.loc[ren_daily[DATE_LOCAL_COLUMN].isin(set(ready_dates))]
    weather_ready = era5_daily_aggregate.loc[era5_daily_aggregate[DATE_LOCAL_COLUMN].isin(set(ready_dates))]
    merged = ren_ready.merge(
        weather_ready.drop(columns=["era5_status"]),
        on=DATE_LOCAL_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != int(coverage["integration_ready"].sum()):
        raise IntegrationBuildError("Daily merged row count does not match explicit integration-ready coverage.")
    return merged.sort_values(DATE_LOCAL_COLUMN).reset_index(drop=True)


def validate_integrated_outputs(
    *,
    start_date: str | date,
    end_date: str | date,
    ren_daily: pd.DataFrame,
    era5_daily_points: pd.DataFrame,
    era5_daily_aggregate: pd.DataFrame,
    daily_merged: pd.DataFrame,
    coverage: pd.DataFrame,
    v1_comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate output counts, local-day coverage, and integration readiness."""
    dates = iter_local_dates(start_date, end_date)
    expected_coverage_rows = len(dates)
    expected_point_rows = expected_coverage_rows * EXPECTED_STATION_COUNT
    failures: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(len(coverage) == expected_coverage_rows, f"Expected {expected_coverage_rows} coverage rows; found {len(coverage)}.")
    require(
        len(era5_daily_aggregate) == expected_coverage_rows,
        f"Expected {expected_coverage_rows} ERA5 aggregate rows; found {len(era5_daily_aggregate)}.",
    )
    require(
        len(era5_daily_points) == expected_point_rows,
        f"Expected {expected_point_rows} ERA5 daily point rows; found {len(era5_daily_points)}.",
    )
    require(
        int(coverage["integration_ready"].sum()) == len(daily_merged),
        "Merged rows must equal explicit integration-ready coverage rows.",
    )
    require(
        not coverage["timezone_source_date_mismatch"].any(),
        "Coverage contains timezone/source-date mismatch rows.",
    )

    ren_complete = coverage.loc[coverage["ren_status"] == "complete"]
    require(
        bool((ren_complete["ren_interval_count"] == ren_complete["ren_expected_interval_count"]).all()),
        "At least one complete REN date does not match the expected 96/92/100 interval count.",
    )
    require(
        bool((era5_daily_points["hourly_count"] == era5_daily_points["expected_count"]).all()),
        "At least one ERA5 station-day does not match the expected 24/23/25 local hourly count.",
    )
    require(
        bool((era5_daily_aggregate["point_count"] == era5_daily_aggregate["expected_point_count"]).all()),
        "At least one ERA5 aggregate day does not have all required station points.",
    )
    final_day = parse_local_date(end_date).isoformat()
    final_weather = era5_daily_aggregate.loc[era5_daily_aggregate[DATE_LOCAL_COLUMN] == final_day]
    require(not final_weather.empty, f"Final requested date {final_day} is missing from ERA5 aggregate output.")
    if not final_weather.empty:
        require(
            str(final_weather["era5_status"].iloc[0]) == "complete",
            f"Final requested date {final_day} does not have complete local-day weather coverage.",
        )

    ren_status_counts = _value_counts(coverage["ren_status"])
    era5_status_counts = _value_counts(coverage["era5_status"])
    coverage_status_counts = _value_counts(coverage["coverage_status"])
    unavailable_dates = coverage.loc[coverage["ren_status"] == "unavailable", DATE_LOCAL_COLUMN].tolist()
    if unavailable_dates:
        warnings.append(
            "REN unavailable dates are explicitly excluded downstream: "
            + ", ".join(str(item) for item in unavailable_dates)
            + "."
        )

    invalid_ren = coverage.loc[
        coverage["ren_status"].isin(["invalid", "incomplete", "missing", "timezone_source_date_mismatch"])
    ]
    require(invalid_ren.empty, "Coverage contains REN invalid, incomplete, missing, or timezone-mismatch rows.")
    invalid_era5 = coverage.loc[coverage["era5_status"] != "complete"]
    require(invalid_era5.empty, "Coverage contains ERA5 missing or incomplete rows.")

    if v1_comparison:
        differing = int(v1_comparison.get("differing_day_count", 0) or 0)
        if differing:
            warnings.append(
                f"V2 REN daily production differs from frozen v1 on {differing} overlapping local days."
            )

    passed = not failures
    verdict = "FAIL" if failures else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "passed": passed,
        "verdict": verdict,
        "start_date": parse_local_date(start_date, "start_date").isoformat(),
        "end_date": parse_local_date(end_date, "end_date").isoformat(),
        "canonical_daily_key": "Europe/Lisbon civil date",
        "local_timezone": LOCAL_TIMEZONE,
        "production_target": {
            "column": REN_PRODUCTION_OUTPUT_COLUMN,
            "aggregation": "sum of 15-minute REN MW observations",
            "unit_warning": "This is not MWh unless multiplied by 0.25h.",
        },
        "expected_counts": {
            "coverage_rows": expected_coverage_rows,
            "era5_daily_aggregate_rows": expected_coverage_rows,
            "era5_daily_point_rows": expected_point_rows,
            "station_count": EXPECTED_STATION_COUNT,
        },
        "actual_counts": {
            "coverage_rows": int(len(coverage)),
            "ren_daily_rows": int(len(ren_daily)),
            "era5_daily_aggregate_rows": int(len(era5_daily_aggregate)),
            "era5_daily_point_rows": int(len(era5_daily_points)),
            "daily_merged_rows": int(len(daily_merged)),
        },
        "ren_status_counts": ren_status_counts,
        "era5_status_counts": era5_status_counts,
        "coverage_status_counts": coverage_status_counts,
        "ren_unavailable_dates": unavailable_dates,
        "final_date_weather": final_weather.iloc[0].to_dict() if not final_weather.empty else None,
        "v1_production_comparison": dict(v1_comparison or {}),
        "warnings": warnings,
        "failures": failures,
        "checks": {
            "coverage_records_every_requested_date": len(coverage) == expected_coverage_rows,
            "ren_interval_counts_match_96_92_100": bool(
                (ren_complete["ren_interval_count"] == ren_complete["ren_expected_interval_count"]).all()
            ),
            "era5_hourly_counts_match_24_23_25": bool(
                (era5_daily_points["hourly_count"] == era5_daily_points["expected_count"]).all()
            ),
            "final_date_era5_complete": bool(not final_weather.empty and str(final_weather["era5_status"].iloc[0]) == "complete"),
            "merged_rows_match_integration_ready_coverage": int(coverage["integration_ready"].sum()) == len(daily_merged),
            "no_interpolation_or_ffill": True,
        },
    }


def compare_ren_daily_with_v1(ren_daily: pd.DataFrame, v1_production_csv: str | Path | None) -> dict[str, Any]:
    """Compare integrated v2 daily production against the frozen v1 production CSV."""
    if v1_production_csv is None:
        return {"available": False, "reason": "No v1 production path was supplied."}
    path = Path(v1_production_csv)
    if not path.is_file():
        return {"available": False, "reason": f"Frozen v1 production CSV is missing: {path}."}

    v1 = pd.read_csv(path, sep=";", skiprows=2, encoding="utf-8-sig")
    timestamp_column = v1.columns[0]
    wind_column = _find_normalized_column(v1.columns, "eolica")
    if wind_column is None:
        raise IntegrationBuildError("Frozen v1 production CSV does not contain an Eolica wind-production column.")
    timestamps = pd.to_datetime(v1[timestamp_column], errors="coerce")
    values = pd.to_numeric(v1[wind_column], errors="coerce")
    frame = pd.DataFrame(
        {
            DATE_LOCAL_COLUMN: timestamps.dt.strftime("%Y-%m-%d"),
            "v1_Wind_Production": values,
        }
    ).dropna(subset=[DATE_LOCAL_COLUMN, "v1_Wind_Production"])
    v1_daily = frame.groupby(DATE_LOCAL_COLUMN, as_index=False)["v1_Wind_Production"].sum()
    comparison = ren_daily[[DATE_LOCAL_COLUMN, REN_PRODUCTION_OUTPUT_COLUMN]].merge(
        v1_daily,
        on=DATE_LOCAL_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if comparison.empty:
        return {"available": True, "overlap_day_count": 0, "reason": "No overlapping local dates."}
    comparison["difference_v2_minus_v1"] = (
        comparison[REN_PRODUCTION_OUTPUT_COLUMN] - comparison["v1_Wind_Production"]
    )
    exact_mask = comparison["difference_v2_minus_v1"].abs() <= 1e-9
    differing = comparison.loc[~exact_mask]
    return {
        "available": True,
        "overlap_day_count": int(len(comparison)),
        "first_overlap_date": str(comparison[DATE_LOCAL_COLUMN].min()),
        "last_overlap_date": str(comparison[DATE_LOCAL_COLUMN].max()),
        "exact_match_day_count": int(exact_mask.sum()),
        "differing_day_count": int((~exact_mask).sum()),
        "mean_difference_v2_minus_v1": _json_float(comparison["difference_v2_minus_v1"].mean()),
        "mean_absolute_difference": _json_float(comparison["difference_v2_minus_v1"].abs().mean()),
        "max_absolute_difference": _json_float(comparison["difference_v2_minus_v1"].abs().max()),
        "sample_differing_dates": differing[DATE_LOCAL_COLUMN].head(10).astype(str).tolist(),
    }


def build_integrated_v2_dataset(
    *,
    start_date: str | date,
    end_date: str | date,
    ren_root: str | Path,
    era5_root: str | Path,
    station_mapping: str | Path,
    output_root: str | Path,
    v1_production: str | Path | None = None,
    overwrite: bool = False,
) -> BuildResult:
    """Build and write the integrated local-day REN + ERA5-Land v2 dataset."""
    paths = resolve_integration_paths(
        ren_root=ren_root,
        era5_root=era5_root,
        station_mapping=station_mapping,
        output_root=output_root,
        v1_production=v1_production,
    )
    if paths.output_root.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists; use --overwrite explicitly: {paths.output_root}.")

    dates = iter_local_dates(start_date, end_date)
    ren_daily, ren_coverage = aggregate_ren_daily_local(paths.ren_root, start_date=start_date, end_date=end_date)
    era5_daily_points, era5_daily_aggregate = aggregate_era5_daily_local(
        paths.era5_root,
        station_mapping=paths.station_mapping,
        start_date=start_date,
        end_date=end_date,
    )
    coverage = build_coverage_table(dates, ren_coverage, era5_daily_aggregate)
    daily_merged = join_integrated_daily(ren_daily, era5_daily_aggregate, coverage)
    v1_comparison = compare_ren_daily_with_v1(ren_daily, paths.v1_production)
    validation = validate_integrated_outputs(
        start_date=start_date,
        end_date=end_date,
        ren_daily=ren_daily,
        era5_daily_points=era5_daily_points,
        era5_daily_aggregate=era5_daily_aggregate,
        daily_merged=daily_merged,
        coverage=coverage,
        v1_comparison=v1_comparison,
    )

    paths.output_root.mkdir(parents=True, exist_ok=True)
    output_files = paths.output_files
    checksums = {
        "ren_daily": write_csv(output_files["ren_daily"], ren_daily),
        "era5_daily_points": write_csv(output_files["era5_daily_points"], era5_daily_points),
        "era5_daily_aggregate": write_csv(output_files["era5_daily_aggregate"], era5_daily_aggregate),
        "daily_merged": write_csv(output_files["daily_merged"], daily_merged),
        "coverage": write_csv(output_files["coverage"], coverage),
    }
    checksums["validation"] = write_json(output_files["validation"], validation)
    manifest = build_manifest_payload(paths=paths, validation=validation, checksums=checksums)
    checksums["manifest"] = write_json(output_files["manifest"], manifest)
    return BuildResult(
        paths=paths,
        ren_daily=ren_daily,
        era5_daily_points=era5_daily_points,
        era5_daily_aggregate=era5_daily_aggregate,
        daily_merged=daily_merged,
        coverage=coverage,
        validation=validation,
        manifest=manifest,
        checksums=checksums,
    )


def build_manifest_payload(
    *,
    paths: IntegrationPaths,
    validation: Mapping[str, Any],
    checksums: Mapping[str, str],
) -> dict[str, Any]:
    """Build deterministic manifest metadata for generated integrated outputs."""
    output_files = paths.output_files
    return {
        "schema_version": "wind_forecast.integrated_v2_manifest.v1",
        "dataset_version": "v2",
        "dataset_role": "integrated_daily_ren_era5_land",
        "transformation_version": TRANSFORMATION_VERSION,
        "canonical_daily_key": "Europe/Lisbon civil date",
        "coverage_start": validation.get("start_date"),
        "coverage_end": validation.get("end_date"),
        "timezone": LOCAL_TIMEZONE,
        "source_paths": {
            "ren_root": str(paths.ren_root),
            "era5_root": str(paths.era5_root),
            "station_mapping": str(paths.station_mapping),
            "v1_production": str(paths.v1_production) if paths.v1_production else None,
        },
        "output_files": {key: str(path) for key, path in output_files.items()},
        "sha256_checksums": {
            str(output_files[key]): checksum
            for key, checksum in checksums.items()
            if key in output_files
        },
        "row_counts": dict(validation.get("actual_counts") or {}),
        "status_counts": {
            "ren": dict(validation.get("ren_status_counts") or {}),
            "era5": dict(validation.get("era5_status_counts") or {}),
            "coverage": dict(validation.get("coverage_status_counts") or {}),
        },
        "production_target": dict(validation.get("production_target") or {}),
        "join_policy": {
            "join_key": DATE_LOCAL_COLUMN,
            "coverage_table_records_all_requested_dates": True,
            "interpolation": False,
            "forward_fill": False,
            "silent_inner_join": False,
        },
        "status": validation.get("verdict"),
        "warnings": list(validation.get("warnings") or []),
        "failures": list(validation.get("failures") or []),
    }


def run_synthetic_alignment_checks() -> dict[str, Any]:
    """Run small in-memory DST and timezone alignment checks."""
    import tempfile

    ordinary_day = date(2026, 1, 15)
    spring_day = date(2026, 3, 29)
    autumn_day = date(2025, 10, 26)
    final_day = date(2026, 6, 27)
    checks = {
        "ordinary_ren_intervals": expected_ren_interval_count(ordinary_day) == 96,
        "spring_ren_intervals": expected_ren_interval_count(spring_day) == 92,
        "autumn_ren_intervals": expected_ren_interval_count(autumn_day) == 100,
        "ordinary_era5_hours": expected_era5_hourly_count(ordinary_day) == 24,
        "spring_era5_hours": expected_era5_hourly_count(spring_day) == 23,
        "autumn_era5_hours": expected_era5_hourly_count(autumn_day) == 25,
        "spring_local_day_utc_hours": len(_expected_utc_index_for_local_day(spring_day, frequency="h")) == 23,
        "autumn_local_day_utc_hours": len(_expected_utc_index_for_local_day(autumn_day, frequency="h")) == 25,
        "final_day_local_weather_window": (
            _expected_utc_index_for_local_day(final_day, frequency="h")[0].isoformat().replace("+00:00", "Z")
            == "2026-06-26T23:00:00Z"
            and _expected_utc_index_for_local_day(final_day, frequency="h")[-1].isoformat().replace("+00:00", "Z")
            == "2026-06-27T22:00:00Z"
        ),
    }
    checks["aware_autumn_identity_preserves_repeated_wall_clock"] = _synthetic_autumn_aware_check(autumn_day)
    checks["utc_to_lisbon_conversion_assigns_adjacent_utc_hour"] = (
        pd.Timestamp("2026-06-26T23:00:00Z").tz_convert(LOCAL_TIMEZONE).strftime("%Y-%m-%d")
        == final_day.isoformat()
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        checks["ren_source_date_mismatch_fails"] = _synthetic_ren_failure_check(
            tmp_path,
            local_day=ordinary_day,
            timestamp_values=["2026-01-15 00:00:00"],
            source_date_values=["2026-01-16"],
            unit_values=[REN_EXPECTED_UNIT],
            production_values=[1.0],
            expected_fragment="source_date",
        )
        checks["duplicate_temporal_keys_fail"] = _synthetic_ren_failure_check(
            tmp_path,
            local_day=ordinary_day,
            timestamp_values=["2026-01-15 00:00:00", "2026-01-15 00:00:00"],
            source_date_values=["2026-01-15", "2026-01-15"],
            unit_values=[REN_EXPECTED_UNIT, REN_EXPECTED_UNIT],
            production_values=[1.0, 2.0],
            expected_fragment="duplicated",
        )
        checks["unexpected_units_fail"] = _synthetic_ren_failure_check(
            tmp_path,
            local_day=ordinary_day,
            timestamp_values=["2026-01-15 00:00:00"],
            source_date_values=["2026-01-15"],
            unit_values=["MWh"],
            production_values=[1.0],
            expected_fragment="unit",
        )
        checks["non_finite_values_fail"] = _synthetic_ren_failure_check(
            tmp_path,
            local_day=ordinary_day,
            timestamp_values=["2026-01-15 00:00:00"],
            source_date_values=["2026-01-15"],
            unit_values=[REN_EXPECTED_UNIT],
            production_values=[math.inf],
            expected_fragment="non-finite",
        )
        hash_path = tmp_path / "stable.csv"
        stable_frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        checks["checksum_stability"] = write_csv(hash_path, stable_frame) == write_csv(hash_path, stable_frame)

    incomplete_group = _synthetic_era5_hourly_group(local_day=ordinary_day, missing_last_hour=True)
    row = _era5_daily_point_row(_SyntheticStation(), incomplete_group.iloc[0], incomplete_group, ordinary_day)
    checks["missing_adjacent_or_hourly_data_marks_incomplete"] = (
        row["era5_point_status"] == "incomplete"
        and row["hourly_count"] == 23
        and row["expected_count"] == 24
    )

    daily_points = pd.DataFrame(
        [
            {
                DATE_LOCAL_COLUMN: ordinary_day.isoformat(),
                "station_id": "synthetic",
                "temperature_2m_c_mean": 10.0,
                "temperature_2m_k_mean": 283.15,
                "wind_speed_m_s_mean": 3.0,
                "u10_m_s_mean": 1.0,
                "v10_m_s_mean": -2.0,
                "calm_or_near_calm_count": 0,
                "hourly_count": 24,
                "era5_point_status": "complete",
            }
        ]
    )
    original_daily_points = daily_points.copy(deep=True)
    aggregate = aggregate_era5_daily_points_local(
        daily_points,
        [ordinary_day],
        expected_point_count=EXPECTED_STATION_COUNT,
    )
    checks["incomplete_station_coverage_marks_incomplete"] = (
        str(aggregate["era5_status"].iloc[0]) == "incomplete"
        and int(aggregate["point_count"].iloc[0]) == 1
    )
    checks["source_input_non_mutation"] = daily_points.equals(original_daily_points)
    checks["deterministic_ordering"] = iter_local_dates("2026-01-01", "2026-01-03") == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
    ]
    return {"passed": all(checks.values()), "checks": checks}


def write_csv(path: Path, frame: pd.DataFrame) -> str:
    """Write a deterministic CSV and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Write deterministic JSON and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sha256_file(path)


def sha256_file(path: str | Path) -> str:
    """Return a file SHA-256 checksum."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage_status(
    *,
    ren_status: str,
    era5_status: str,
    timezone_mismatch: bool,
    excluded: bool,
    integration_ready: bool,
) -> str:
    if integration_ready:
        return "integration-ready"
    if timezone_mismatch:
        return "timezone-source-date-mismatch"
    if excluded:
        return "excluded-downstream-ren-unavailable"
    if ren_status != "complete":
        return "ren-invalid-or-incomplete"
    if era5_status != "complete":
        return "era5-missing-or-incomplete"
    return "not-ready"


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _finite_mean(frame: pd.DataFrame, column: str) -> float:
    series = _finite_series(frame, column)
    return _json_float(series.mean()) if not series.empty else math.nan


def _finite_min(frame: pd.DataFrame, column: str) -> float:
    series = _finite_series(frame, column)
    return _json_float(series.min()) if not series.empty else math.nan


def _finite_max(frame: pd.DataFrame, column: str) -> float:
    series = _finite_series(frame, column)
    return _json_float(series.max()) if not series.empty else math.nan


def _finite_std(frame: pd.DataFrame, column: str) -> float:
    series = _finite_series(frame, column)
    return _json_float(series.std()) if len(series) > 1 else math.nan


def _calm_count(frame: pd.DataFrame) -> int:
    if frame.empty or "is_calm_or_near_calm" not in frame:
        return 0
    values = frame["is_calm_or_near_calm"]
    if values.dtype == bool:
        return int(values.sum())
    normalized = values.astype(str).str.strip().str.casefold()
    return int(normalized.isin({"true", "1", "yes"}).sum())


def _station_or_frame_value(station: Any, frame_row: Any, frame_key: str, station_key: str) -> float:
    value = _frame_value(frame_row, frame_key)
    if math.isfinite(value):
        return value
    return float(getattr(station, station_key))


def _frame_value(frame_row: Any, key: str) -> float:
    if frame_row is None:
        return math.nan
    try:
        return _json_float(frame_row.get(key))
    except (AttributeError, TypeError, ValueError):
        return math.nan


def _int_value(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def _find_normalized_column(columns: Sequence[str], normalized_name: str) -> str | None:
    for column in columns:
        if _normalize_text(column) == normalized_name:
            return str(column)
    return None


def _synthetic_autumn_aware_check(local_day: date) -> bool:
    utc_index = _expected_utc_index_for_local_day(local_day, frequency="15min")
    local = utc_index.tz_convert(LOCAL_TIMEZONE)
    local_dates = set(local.strftime("%Y-%m-%d"))
    labels = list(local.strftime("%H:%M"))
    return len(utc_index) == 100 and local_dates == {local_day.isoformat()} and len(set(labels)) < len(labels)


@dataclass(frozen=True)
class _SyntheticStation:
    station_id: str = "synthetic"
    station_name: str = "Synthetic Station"
    latitude: float = 40.0
    longitude: float = -8.0


def _synthetic_ren_failure_check(
    tmp_path: Path,
    *,
    local_day: date,
    timestamp_values: Sequence[str],
    source_date_values: Sequence[str],
    unit_values: Sequence[str],
    production_values: Sequence[float],
    expected_fragment: str,
) -> bool:
    path = tmp_path / f"ren_{expected_fragment}.csv"
    pd.DataFrame(
        {
            REN_TIMESTAMP_COLUMN: list(timestamp_values),
            REN_PRODUCTION_INPUT_COLUMN: list(production_values),
            REN_UNIT_COLUMN: list(unit_values),
            REN_SOURCE_DATE_COLUMN: list(source_date_values),
        }
    ).to_csv(path, index=False, lineterminator="\n")
    try:
        _aggregate_one_ren_partition(path, local_day)
    except IntegrationBuildError as exc:
        return expected_fragment.casefold() in str(exc).casefold()
    return False


def _synthetic_era5_hourly_group(*, local_day: date, missing_last_hour: bool) -> pd.DataFrame:
    utc_index = _expected_utc_index_for_local_day(local_day, frequency="h")
    if missing_last_hour:
        utc_index = utc_index[:-1]
    return pd.DataFrame(
        {
            "_timestamp_utc": utc_index,
            "timestamp_utc": [item.isoformat().replace("+00:00", "Z") for item in utc_index],
            "station_id": "synthetic",
            "station_name": "Synthetic Station",
            "station_latitude": 40.0,
            "station_longitude": -8.0,
            "grid_latitude": 40.0,
            "grid_longitude": -8.0,
            "temperature_2m_k": 283.15,
            "temperature_2m_c": 10.0,
            "u10_m_s": 1.0,
            "v10_m_s": -2.0,
            "wind_speed_m_s": math.sqrt(5.0),
            "is_calm_or_near_calm": False,
        }
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _json_float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes, bool, type(None))) else False:
        return None
    return value


__all__ = [
    "BuildResult",
    "IntegrationBuildError",
    "IntegrationPaths",
    "aggregate_era5_daily_local",
    "aggregate_era5_daily_points_local",
    "aggregate_ren_daily_local",
    "build_coverage_table",
    "build_integrated_v2_dataset",
    "compare_ren_daily_with_v1",
    "discover_era5_hourly_files",
    "discover_ren_paths",
    "expected_era5_hourly_count",
    "expected_ren_interval_count",
    "iter_local_dates",
    "join_integrated_daily",
    "parse_local_date",
    "resolve_integration_paths",
    "run_synthetic_alignment_checks",
    "validate_integrated_outputs",
]
