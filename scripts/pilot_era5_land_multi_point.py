"""Run the limited ERA5-Land multi-point seasonal technical pilot."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

try:
    from scripts.pilot_era5_land_one_point import (
        DATASET_ID,
        DEFAULT_CALM_THRESHOLD_M_S,
        OFFICIAL_URLS,
        REQUEST_TIMES,
        REQUEST_VARIABLES,
        build_cds_request,
        credential_presence,
        daily_aggregates,
        dependency_versions,
        expected_hourly_index,
        load_hourly_frame,
        output_units,
        retrieve_era5_land,
        sha256_file,
        utc_timestamp,
        write_json,
    )
except ModuleNotFoundError:
    from pilot_era5_land_one_point import (
        DATASET_ID,
        DEFAULT_CALM_THRESHOLD_M_S,
        OFFICIAL_URLS,
        REQUEST_TIMES,
        REQUEST_VARIABLES,
        build_cds_request,
        credential_presence,
        daily_aggregates,
        dependency_versions,
        expected_hourly_index,
        load_hourly_frame,
        output_units,
        retrieve_era5_land,
        sha256_file,
        utc_timestamp,
        write_json,
    )

from wind_forecast.manifest_validation import validate_v1_source_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "pilot" / "era5_land"
MAX_REQUESTS = 6
EXPECTED_HOURLY_ROWS = 1008
EXPECTED_DAILY_POINT_ROWS = 42
EXPECTED_AGGREGATE_DAILY_ROWS = 14
STATION_COUNT = 3
CALM_THRESHOLD_M_S = DEFAULT_CALM_THRESHOLD_M_S

RAW_WEATHER_MATRICES = {
    "v1_wind_speed_m_s": PROJECT_ROOT / "data" / "raw" / "IntensidadeMediaVento10m.csv",
    "v1_wind_direction_deg_from": PROJECT_ROOT / "data" / "raw" / "DirecaoMediaVento10m.csv",
    "v1_temperature_2m_c": PROJECT_ROOT / "data" / "raw" / "TemperaturaMedia.csv",
}


@dataclass(frozen=True)
class Station:
    """A fixed station used by the multi-point pilot."""

    station_id: str
    station_name: str
    latitude: float
    longitude: float
    expected_area: tuple[float, float, float, float]


@dataclass(frozen=True)
class SeasonPeriod:
    """A fixed seasonal one-week pilot period."""

    season: str
    start: date
    end: date


@dataclass(frozen=True)
class PlannedRequest:
    """One fixed ERA5-Land request and its approved raw output path."""

    season: str
    period: SeasonPeriod
    station: Station
    request: dict[str, Any]
    raw_path: Path


STATIONS = (
    Station("1210622", "Braga Merelim", 41.56678056, -8.45003056, (41.6, -8.5, 41.6, -8.5)),
    Station("1210683", "Guarda", 40.53333333, -7.26666667, (40.5, -7.3, 40.5, -7.3)),
    Station("1200562", "Beja", 38.02493056, -7.867275, (38.0, -7.9, 38.0, -7.9)),
)

PERIODS = (
    SeasonPeriod("winter", date(2023, 1, 1), date(2023, 1, 7)),
    SeasonPeriod("summer", date(2023, 7, 1), date(2023, 7, 7)),
)

COMBINED_OUTPUT_FILENAMES = {
    "hourly_csv": "era5_land_multi_point_2023_winter_summer_hourly.csv",
    "daily_points_csv": "era5_land_multi_point_2023_winter_summer_daily_points.csv",
    "daily_aggregate_csv": "era5_land_multi_point_2023_winter_summer_daily_aggregate.csv",
    "season_summary_csv": "era5_land_multi_point_2023_winter_summer_season_summary.csv",
    "v1_comparison_csv": "era5_land_multi_point_2023_winter_summer_v1_comparison.csv",
    "metadata_json": "era5_land_multi_point_2023_winter_summer_metadata.json",
    "validation_json": "era5_land_multi_point_2023_winter_summer_validation.json",
}

REQUIRED_HOURLY_COLUMNS = (
    "season",
    "station_id",
    "station_name",
    "station_latitude",
    "station_longitude",
    "grid_latitude",
    "grid_longitude",
    "timestamp_utc",
    "temperature_2m_k",
    "temperature_2m_c",
    "u10_m_s",
    "v10_m_s",
    "wind_speed_m_s",
    "is_calm_or_near_calm",
)

HOURLY_COLUMN_ORDER = (
    "season",
    "station_id",
    "station_name",
    "station_latitude",
    "station_longitude",
    "grid_latitude",
    "grid_longitude",
    "timestamp_utc",
    "temperature_2m_k",
    "temperature_2m_c",
    "u10_m_s",
    "v10_m_s",
    "wind_speed_m_s",
    "wind_direction_deg_from",
    "is_calm_or_near_calm",
)

def parse_args() -> argparse.Namespace:
    """Parse the narrow pilot CLI without network access or filesystem writes."""
    parser = argparse.ArgumentParser(
        description="Plan or run the fixed ERA5-Land multi-point seasonal pilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed six-request plan without contacting CDS or writing outputs.",
    )
    mode_group.add_argument(
        "--download",
        action="store_true",
        help="Run exactly six live CDS requests and write the fixed pilot outputs.",
    )
    mode_group.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Rebuild combined outputs from the fixed raw NetCDF files without contacting CDS.",
    )
    return parser.parse_args()


def output_path_strings(paths: dict[str, Path] | dict[str, dict[str, Path]]) -> dict[str, Any]:
    """Return stable project-relative output path strings."""
    rendered: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, dict):
            rendered[key] = {nested_key: relative_path(nested_path) for nested_key, nested_path in value.items()}
        else:
            rendered[key] = relative_path(value)
    return rendered


def relative_path(path: Path) -> str:
    """Render a path relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def raw_output_path(period: SeasonPeriod, station: Station) -> Path:
    """Return the approved raw NetCDF path for one season and station."""
    filename = (
        f"era5_land_multi_point_2023_{period.season}_{station.station_id}_"
        f"{period.start.isoformat()}_{period.end.isoformat()}_raw.nc"
    )
    return OUTPUT_DIR / filename


def combined_output_paths() -> dict[str, Path]:
    """Return the approved combined output paths."""
    return {key: OUTPUT_DIR / filename for key, filename in COMBINED_OUTPUT_FILENAMES.items()}


def validate_output_root(path: Path) -> None:
    """Fail if an output path escapes the approved pilot directory."""
    output_root = OUTPUT_DIR.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Output path escapes the approved pilot directory: {path}") from exc


def planned_requests() -> list[PlannedRequest]:
    """Build the exact fixed six-request ERA5-Land plan."""
    requests = []
    for period in PERIODS:
        for station in STATIONS:
            request = build_cds_request(
                period.start,
                period.end,
                latitude=station.latitude,
                longitude=station.longitude,
            )
            requests.append(
                PlannedRequest(
                    season=period.season,
                    period=period,
                    station=station,
                    request=request,
                    raw_path=raw_output_path(period, station),
                )
            )
    validate_request_plan(requests)
    return requests


def validate_request_plan(requests: list[PlannedRequest]) -> None:
    """Validate the fixed request plan before any optional download."""
    if len(requests) != MAX_REQUESTS:
        raise ValueError(f"Expected exactly {MAX_REQUESTS} requests; planned {len(requests)}.")
    if len(requests) > MAX_REQUESTS:
        raise ValueError(f"Request plan exceeds the hard maximum of {MAX_REQUESTS}.")

    seen = set()
    expected_times = [f"{hour:02d}:00" for hour in range(24)]
    for planned in requests:
        key = (planned.season, planned.station.station_id)
        if key in seen:
            raise ValueError(f"Duplicate request planned for {planned.season}/{planned.station.station_id}.")
        seen.add(key)

        request = planned.request
        if request.get("variable") != REQUEST_VARIABLES:
            raise ValueError(f"Unexpected variables for {planned.season}/{planned.station.station_id}.")
        if request.get("time") != REQUEST_TIMES or request.get("time") != expected_times:
            raise ValueError(f"Unexpected hourly times for {planned.season}/{planned.station.station_id}.")
        if request.get("area") != list(planned.station.expected_area):
            raise ValueError(
                f"Unexpected area for {planned.season}/{planned.station.station_id}: {request.get('area')}"
            )
        if request.get("data_format") != "netcdf" or request.get("download_format") != "unarchived":
            raise ValueError(f"Unexpected CDS output format for {planned.season}/{planned.station.station_id}.")
        validate_output_root(planned.raw_path)

    for path in combined_output_paths().values():
        validate_output_root(path)


def request_plan_payload(requests: list[PlannedRequest]) -> list[dict[str, Any]]:
    """Return a JSON-serializable view of the request plan."""
    rows = []
    for index, planned in enumerate(requests, start=1):
        rows.append(
            {
                "request_index": index,
                "dataset": DATASET_ID,
                "season": planned.season,
                "station_id": planned.station.station_id,
                "station_name": planned.station.station_name,
                "station_latitude": planned.station.latitude,
                "station_longitude": planned.station.longitude,
                "start_date": planned.period.start.isoformat(),
                "end_date": planned.period.end.isoformat(),
                "request_params": planned.request,
                "expected_area": list(planned.station.expected_area),
                "raw_output_path": relative_path(planned.raw_path),
            }
        )
    return rows


def dry_run_result(requests: list[PlannedRequest]) -> dict[str, Any]:
    """Return the fixed request plan without contacting CDS or writing files."""
    return {
        "mode": "dry_run",
        "passed": True,
        "dataset": DATASET_ID,
        "request_count": len(requests),
        "max_requests": MAX_REQUESTS,
        "planned_requests": request_plan_payload(requests),
        "combined_output_paths": output_path_strings(combined_output_paths()),
        "credential_presence": credential_presence(),
        "notes": [
            "Dry run only; no network request was made and no filesystem output was written.",
            "Use --download to run exactly the six planned CDS requests.",
        ],
    }


def retrieve_raw_files(requests: list[PlannedRequest]) -> dict[str, dict[str, Any]]:
    """Run exactly the approved CDS retrievals."""
    retrievals = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, planned in enumerate(requests, start=1):
        started_at = utc_timestamp()
        retrieve_era5_land(DATASET_ID, planned.request, planned.raw_path)
        finished_at = utc_timestamp()
        retrievals[request_key(planned)] = {
            "request_index": index,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "raw_output_path": relative_path(planned.raw_path),
            "sha256": sha256_file(planned.raw_path),
        }
    return retrievals


def existing_raw_files(requests: list[PlannedRequest]) -> dict[str, dict[str, Any]]:
    """Validate and describe existing raw files without contacting CDS."""
    retrievals = {}
    for index, planned in enumerate(requests, start=1):
        if not planned.raw_path.exists():
            raise FileNotFoundError(f"Required raw NetCDF file is missing: {relative_path(planned.raw_path)}")
        retrievals[request_key(planned)] = {
            "request_index": index,
            "started_at_utc": None,
            "finished_at_utc": None,
            "raw_output_path": relative_path(planned.raw_path),
            "sha256": sha256_file(planned.raw_path),
            "service_status": "not_contacted_reuse_raw",
        }
    return retrievals


def request_key(planned: PlannedRequest) -> str:
    """Return a stable request key for dictionaries."""
    return f"{planned.season}_{planned.station.station_id}"


def build_hourly_and_daily_frames(requests: list[PlannedRequest]) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Load raw NetCDF files and build combined hourly and station-day frames."""
    import pandas as pd

    hourly_frames = []
    daily_frames = []
    extraction_metadata = []
    for planned in requests:
        hourly, metadata = load_hourly_frame(
            planned.raw_path,
            station_id=planned.station.station_id,
            station_latitude=planned.station.latitude,
            station_longitude=planned.station.longitude,
            calm_threshold=CALM_THRESHOLD_M_S,
        )
        hourly.insert(0, "season", planned.season)
        hourly.insert(2, "station_name", planned.station.station_name)
        hourly = hourly.loc[:, list(HOURLY_COLUMN_ORDER)]
        hourly_frames.append(hourly)

        daily = daily_aggregates(hourly, planned.period.start, planned.period.end, CALM_THRESHOLD_M_S)
        for column, value in reversed(
            [
                ("season", planned.season),
                ("station_id", planned.station.station_id),
                ("station_name", planned.station.station_name),
                ("station_latitude", planned.station.latitude),
                ("station_longitude", planned.station.longitude),
                ("grid_latitude", float(hourly["grid_latitude"].iloc[0])),
                ("grid_longitude", float(hourly["grid_longitude"].iloc[0])),
            ]
        ):
            daily.insert(0, column, value)
        daily_frames.append(daily)

        extraction_metadata.append(
            {
                "season": planned.season,
                "station_id": planned.station.station_id,
                "station_name": planned.station.station_name,
                "selected_grid_coordinate": metadata["selected_grid_coordinate"],
                "coordinate_names": metadata["coordinate_names"],
                "source_variables": metadata["source_variables"],
                "source_netcdf_units": metadata["netcdf_variable_units"],
                "netcdf_dimensions_after_point_extraction": metadata["netcdf_dimensions_after_point_extraction"],
            }
        )

    hourly_combined = pd.concat(hourly_frames, ignore_index=True).sort_values(
        ["season", "station_id", "timestamp_utc"]
    )
    daily_points = pd.concat(daily_frames, ignore_index=True).sort_values(["season", "station_id", "date_utc"])
    return hourly_combined, daily_points, extraction_metadata


def aggregate_daily_points(daily_points: Any) -> Any:
    """Aggregate daily station rows across the fixed three stations."""
    import numpy as np
    import pandas as pd

    rows = []
    for (season, date_utc), group in daily_points.groupby(["season", "date_utc"], sort=True):
        valid_points = group.dropna(subset=["temperature_2m_c_mean", "wind_speed_m_s_mean", "u10_m_s_mean", "v10_m_s_mean"])
        point_count = int(len(valid_points))
        missing_point_count = STATION_COUNT - point_count
        mean_u = float(valid_points["u10_m_s_mean"].mean()) if point_count else math.nan
        mean_v = float(valid_points["v10_m_s_mean"].mean()) if point_count else math.nan
        vector_speed = math.sqrt(mean_u * mean_u + mean_v * mean_v) if math.isfinite(mean_u) and math.isfinite(mean_v) else math.nan
        if math.isfinite(vector_speed) and vector_speed >= CALM_THRESHOLD_M_S:
            vector_direction = (180.0 + math.degrees(math.atan2(mean_u, mean_v))) % 360.0
        else:
            vector_direction = math.nan

        calm_count = int(valid_points["calm_or_near_calm_count"].sum()) if point_count else 0
        hourly_count = int(valid_points["hourly_count"].sum()) if point_count else 0
        rows.append(
            {
                "season": season,
                "date_utc": date_utc,
                "point_count": point_count,
                "expected_point_count": STATION_COUNT,
                "missing_point_count": missing_point_count,
                "temperature_2m_c_mean": valid_points["temperature_2m_c_mean"].mean() if point_count else np.nan,
                "temperature_2m_k_mean": valid_points["temperature_2m_k_mean"].mean() if point_count else np.nan,
                "wind_speed_m_s_mean": valid_points["wind_speed_m_s_mean"].mean() if point_count else np.nan,
                "u10_m_s_mean": mean_u if math.isfinite(mean_u) else np.nan,
                "v10_m_s_mean": mean_v if math.isfinite(mean_v) else np.nan,
                "vector_mean_wind_speed_m_s": vector_speed if math.isfinite(vector_speed) else np.nan,
                "vector_mean_wind_direction_deg_from": vector_direction,
                "calm_or_near_calm_share": calm_count / hourly_count if hourly_count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def season_summary(hourly: Any, daily_points: Any, daily_aggregate: Any) -> Any:
    """Build per-station and per-season aggregate pilot summaries."""
    import pandas as pd

    rows = []
    for (season, station_id), group in hourly.groupby(["season", "station_id"], sort=True):
        station_days = daily_points[(daily_points["season"] == season) & (daily_points["station_id"] == station_id)]
        station_name = str(group["station_name"].iloc[0])
        rows.append(
            summary_row(
                scope="station",
                season=str(season),
                station_id=str(station_id),
                station_name=station_name,
                hourly_count=int(len(group)),
                expected_hourly_count=7 * 24,
                station_day_count=int(len(station_days)),
                expected_station_day_count=7,
                frame=group,
            )
        )

    for season, group in hourly.groupby("season", sort=True):
        aggregate_days = daily_aggregate[daily_aggregate["season"] == season]
        rows.append(
            summary_row(
                scope="season_aggregate",
                season=str(season),
                station_id="ALL",
                station_name="Three-station aggregate",
                hourly_count=int(len(group)),
                expected_hourly_count=STATION_COUNT * 7 * 24,
                station_day_count=int(len(aggregate_days)),
                expected_station_day_count=7,
                frame=group,
            )
        )
    return pd.DataFrame(rows)


def summary_row(
    *,
    scope: str,
    season: str,
    station_id: str,
    station_name: str,
    hourly_count: int,
    expected_hourly_count: int,
    station_day_count: int,
    expected_station_day_count: int,
    frame: Any,
) -> dict[str, Any]:
    """Create one coverage and descriptive-statistics summary row."""
    return {
        "scope": scope,
        "season": season,
        "station_id": station_id,
        "station_name": station_name,
        "hourly_count": hourly_count,
        "expected_hourly_count": expected_hourly_count,
        "hourly_coverage_share": hourly_count / expected_hourly_count if expected_hourly_count else math.nan,
        "station_day_count": station_day_count,
        "expected_station_day_count": expected_station_day_count,
        "station_day_coverage_share": station_day_count / expected_station_day_count if expected_station_day_count else math.nan,
        "temperature_2m_c_mean": frame["temperature_2m_c"].mean(),
        "temperature_2m_c_std": frame["temperature_2m_c"].std(),
        "temperature_2m_c_min": frame["temperature_2m_c"].min(),
        "temperature_2m_c_max": frame["temperature_2m_c"].max(),
        "wind_speed_m_s_mean": frame["wind_speed_m_s"].mean(),
        "wind_speed_m_s_std": frame["wind_speed_m_s"].std(),
        "wind_speed_m_s_min": frame["wind_speed_m_s"].min(),
        "wind_speed_m_s_max": frame["wind_speed_m_s"].max(),
        "u10_m_s_mean": frame["u10_m_s"].mean(),
        "v10_m_s_mean": frame["v10_m_s"].mean(),
        "calm_or_near_calm_share": frame["is_calm_or_near_calm"].mean(),
    }


def load_v1_weather_matrix(path: Path, value_name: str) -> Any:
    """Load one v1 weather matrix in read-only mode."""
    import pandas as pd

    frame = pd.read_csv(path, sep=";")
    frame["date_utc"] = pd.to_datetime(
        {
            "year": frame["ANO"],
            "month": frame["MES"],
            "day": frame["DIA"],
        },
        utc=True,
    ).dt.strftime("%Y-%m-%d")
    columns = ["date_utc"] + [station.station_id for station in STATIONS if station.station_id in frame.columns]
    return frame.loc[:, columns].melt(id_vars="date_utc", var_name="station_id", value_name=value_name)


def v1_comparison(daily_points: Any) -> Any:
    """Compare ERA5-Land station days against v1 matrices without mutating v1 data."""
    import pandas as pd

    comparison = daily_points.loc[
        :,
        [
            "season",
            "station_id",
            "station_name",
            "date_utc",
            "temperature_2m_c_mean",
            "wind_speed_m_s_mean",
            "vector_mean_wind_direction_deg_from",
        ],
    ].rename(
        columns={
            "temperature_2m_c_mean": "era5_temperature_2m_c_mean",
            "wind_speed_m_s_mean": "era5_wind_speed_m_s_mean",
            "vector_mean_wind_direction_deg_from": "era5_vector_mean_wind_direction_deg_from",
        }
    )

    for value_name, path in RAW_WEATHER_MATRICES.items():
        if path.exists():
            matrix = load_v1_weather_matrix(path, value_name)
            comparison = comparison.merge(matrix, on=["date_utc", "station_id"], how="left")
        else:
            comparison[value_name] = pd.NA

    comparison["v1_available"] = comparison[
        ["v1_wind_speed_m_s", "v1_wind_direction_deg_from", "v1_temperature_2m_c"]
    ].notna().all(axis=1)
    comparison["temperature_2m_c_difference_era5_minus_v1"] = (
        comparison["era5_temperature_2m_c_mean"] - comparison["v1_temperature_2m_c"]
    )
    comparison["wind_speed_m_s_difference_era5_minus_v1"] = (
        comparison["era5_wind_speed_m_s_mean"] - comparison["v1_wind_speed_m_s"]
    )
    comparison["wind_direction_signed_difference_deg_era5_minus_v1"] = comparison.apply(
        lambda row: circular_difference_degrees(
            row["era5_vector_mean_wind_direction_deg_from"],
            row["v1_wind_direction_deg_from"],
        ),
        axis=1,
    )
    comparison["wind_direction_abs_difference_deg"] = comparison[
        "wind_direction_signed_difference_deg_era5_minus_v1"
    ].abs()
    comparison["comparison_scope"] = "technical_non_equivalence"
    comparison["comparison_note"] = (
        "ERA5-Land grid-cell daily values and v1 station matrices are compared for diagnostics only; "
        "differences do not imply either source is equivalent or selected for v2."
    )
    return comparison.sort_values(["season", "station_id", "date_utc"])


def circular_difference_degrees(value: Any, reference: Any) -> float:
    """Return signed circular difference in degrees in the [-180, 180] interval."""
    try:
        if not math.isfinite(float(value)) or not math.isfinite(float(reference)):
            return math.nan
    except (TypeError, ValueError):
        return math.nan
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


def write_csv_outputs(
    *,
    hourly: Any,
    daily_points: Any,
    daily_aggregate: Any,
    summary: Any,
    comparison: Any,
    paths: dict[str, Path],
) -> None:
    """Write the combined CSV outputs under the approved pilot directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(paths["hourly_csv"], index=False)
    daily_points.to_csv(paths["daily_points_csv"], index=False)
    daily_aggregate.to_csv(paths["daily_aggregate_csv"], index=False)
    summary.to_csv(paths["season_summary_csv"], index=False)
    comparison.to_csv(paths["v1_comparison_csv"], index=False)


def file_checksums(paths: dict[str, Path]) -> dict[str, str]:
    """Calculate checksums for existing output files."""
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def raw_output_mapping(requests: list[PlannedRequest]) -> dict[str, Path]:
    """Return keyed raw output paths."""
    return {request_key(planned): planned.raw_path for planned in requests}


def validation_report(
    *,
    requests: list[PlannedRequest],
    hourly: Any,
    daily_points: Any,
    daily_aggregate: Any,
    summary: Any,
    comparison: Any,
    paths: dict[str, Path],
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Validate the generated multi-point pilot outputs."""
    import numpy as np
    import pandas as pd

    issues = []
    request_count = len(requests)
    if request_count != MAX_REQUESTS:
        issues.append(f"Expected exactly {MAX_REQUESTS} requests but found {request_count}.")
    if request_count > MAX_REQUESTS:
        issues.append(f"Request count {request_count} exceeds max_requests={MAX_REQUESTS}.")

    for planned in requests:
        area = planned.request.get("area")
        if area != list(planned.station.expected_area):
            issues.append(f"{request_key(planned)} area mismatch: expected {planned.station.expected_area}, found {area}.")

    if len(hourly) != EXPECTED_HOURLY_ROWS:
        issues.append(f"Expected {EXPECTED_HOURLY_ROWS} hourly rows but found {len(hourly)}.")
    if len(daily_points) != EXPECTED_DAILY_POINT_ROWS:
        issues.append(f"Expected {EXPECTED_DAILY_POINT_ROWS} station-day rows but found {len(daily_points)}.")
    if len(daily_aggregate) != EXPECTED_AGGREGATE_DAILY_ROWS:
        issues.append(f"Expected {EXPECTED_AGGREGATE_DAILY_ROWS} aggregate daily rows but found {len(daily_aggregate)}.")

    for column in REQUIRED_HOURLY_COLUMNS:
        if column not in hourly.columns:
            issues.append(f"Hourly output is missing required column {column}.")
        elif hourly[column].isna().any():
            issues.append(f"Hourly required column {column} has {int(hourly[column].isna().sum())} null values.")

    for planned in requests:
        group = hourly[(hourly["season"] == planned.season) & (hourly["station_id"] == planned.station.station_id)]
        expected = expected_hourly_index(planned.period.start, planned.period.end)
        actual = pd.DatetimeIndex(pd.to_datetime(group["timestamp_utc"], utc=True))
        missing = expected.difference(actual)
        unexpected = actual.difference(expected)
        duplicates = int(actual.duplicated().sum())
        if len(missing):
            issues.append(f"{request_key(planned)} is missing {len(missing)} expected hourly timestamps.")
        if len(unexpected):
            issues.append(f"{request_key(planned)} has {len(unexpected)} timestamps outside the fixed period.")
        if duplicates:
            issues.append(f"{request_key(planned)} has {duplicates} duplicate hourly timestamps.")

    required_output_columns = {
        "daily_points": [
            "temperature_2m_c_mean",
            "temperature_2m_k_mean",
            "wind_speed_m_s_mean",
            "u10_m_s_mean",
            "v10_m_s_mean",
            "vector_mean_wind_speed_m_s",
        ],
        "daily_aggregate": [
            "temperature_2m_c_mean",
            "temperature_2m_k_mean",
            "wind_speed_m_s_mean",
            "u10_m_s_mean",
            "v10_m_s_mean",
            "vector_mean_wind_speed_m_s",
            "point_count",
            "missing_point_count",
            "calm_or_near_calm_share",
        ],
    }
    for frame_name, columns in required_output_columns.items():
        frame = daily_points if frame_name == "daily_points" else daily_aggregate
        for column in columns:
            if column not in frame.columns:
                issues.append(f"{frame_name} output is missing required column {column}.")
            elif frame[column].isna().any():
                issues.append(f"{frame_name} required column {column} has {int(frame[column].isna().sum())} null values.")

    finite_checks = {}
    for frame_name, frame in [
        ("hourly", hourly),
        ("daily_points", daily_points),
        ("daily_aggregate", daily_aggregate),
        ("season_summary", summary),
        ("v1_comparison", comparison),
    ]:
        numeric_columns = list(frame.select_dtypes(include=["number", "bool"]).columns)
        finite_checks[frame_name] = {}
        for column in numeric_columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            finite = bool(np.isfinite(series.dropna()).all())
            finite_checks[frame_name][column] = finite
            if not finite:
                issues.append(f"{frame_name}.{column} contains non-finite numeric values.")

    all_paths = {**raw_output_mapping(requests), **paths}
    output_existence = {}
    for name, path in all_paths.items():
        exists = path.exists()
        if name == "validation_json":
            output_existence[name] = True
            continue
        output_existence[name] = exists
        if not exists:
            issues.append(f"Expected output path does not exist: {relative_path(path)}")
        elif name not in checksums:
            issues.append(f"Checksum missing for output path: {relative_path(path)}")

    return {
        "generated_at_utc": utc_timestamp(),
        "passed": not issues,
        "issues": issues,
        "request_count": request_count,
        "max_requests": MAX_REQUESTS,
        "expected_counts": {
            "hourly_rows": EXPECTED_HOURLY_ROWS,
            "station_day_rows": EXPECTED_DAILY_POINT_ROWS,
            "aggregate_daily_rows": EXPECTED_AGGREGATE_DAILY_ROWS,
        },
        "actual_counts": {
            "hourly_rows": int(len(hourly)),
            "station_day_rows": int(len(daily_points)),
            "aggregate_daily_rows": int(len(daily_aggregate)),
            "season_summary_rows": int(len(summary)),
            "v1_comparison_rows": int(len(comparison)),
        },
        "timestamp_expectation": {
            "per_station_season_hours": 7 * 24,
            "timezone": "UTC",
            "missing_expected_timestamps_issue_count": sum("expected hourly timestamps" in issue for issue in issues),
        },
        "required_hourly_columns": list(REQUIRED_HOURLY_COLUMNS),
        "finite_checks": finite_checks,
        "output_paths": output_path_strings(all_paths),
        "output_exists": output_existence,
        "checksums": checksums,
        "checksum_note": "The validation JSON cannot contain its own final SHA-256 without changing that SHA-256.",
    }


def metadata_payload(
    *,
    requests: list[PlannedRequest],
    retrievals: dict[str, dict[str, Any]],
    extraction_metadata: list[dict[str, Any]],
    validation: dict[str, Any],
    output_paths: dict[str, Path],
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Build metadata for the multi-point ERA5-Land pilot."""
    return {
        "generated_at_utc": utc_timestamp(),
        "source_dataset": DATASET_ID,
        "official_urls": OFFICIAL_URLS,
        "source": {
            "name": "Copernicus Climate Data Store ERA5-Land",
            "dataset": DATASET_ID,
            "technical_status": "limited_pilot_not_v2_source_selection",
        },
        "request_count": len(requests),
        "max_requests": MAX_REQUESTS,
        "request_plan": request_plan_payload(requests),
        "retrievals": retrievals,
        "stations": [
            {
                "station_id": station.station_id,
                "station_name": station.station_name,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "expected_area": list(station.expected_area),
            }
            for station in STATIONS
        ],
        "periods": [
            {
                "season": period.season,
                "start_date": period.start.isoformat(),
                "end_date": period.end.isoformat(),
                "timezone": "UTC",
            }
            for period in PERIODS
        ],
        "variables": REQUEST_VARIABLES,
        "times": REQUEST_TIMES,
        "units": output_units(),
        "calm_or_near_calm_threshold_m_s": CALM_THRESHOLD_M_S,
        "dependency_versions": dependency_versions(),
        "credential_presence": credential_presence(),
        "selected_grid_coordinates": extraction_metadata,
        "v1_comparison": {
            "status": "technical_comparison_non_equivalence",
            "matrices": {name: relative_path(path) for name, path in RAW_WEATHER_MATRICES.items()},
            "notes": [
                "The v1 files are read only for overlapping station/date diagnostics.",
                "This pilot does not mutate v1 data and does not claim ERA5-Land equivalence to v1 weather.",
            ],
        },
        "validation": {
            "passed": validation["passed"],
            "issues": validation["issues"],
            "actual_counts": validation["actual_counts"],
            "expected_counts": validation["expected_counts"],
        },
        "output_paths": output_path_strings(output_paths),
        "checksums": checksums,
        "notes": [
            "Generated for Checkpoint 2 limited multi-point seasonal ERA5-Land pilot.",
            "The pilot does not select a final v2 weather source or validate current model/scaler compatibility.",
            "Current v1 raw, processed, model, scaler, notebook, and validator assets are not modified.",
        ],
    }


def run_download(requests: list[PlannedRequest], *, reuse_raw: bool = False) -> dict[str, Any]:
    """Run the fixed live pilot or rebuild from existing raw files."""
    validate_v1_source_contract(required_paths=list(RAW_WEATHER_MATRICES.values()))
    combined_paths = combined_output_paths()
    retrievals = existing_raw_files(requests) if reuse_raw else retrieve_raw_files(requests)
    hourly, daily_points, extraction_metadata = build_hourly_and_daily_frames(requests)
    daily_aggregate = aggregate_daily_points(daily_points)
    summary = season_summary(hourly, daily_points, daily_aggregate)
    comparison = v1_comparison(daily_points)
    write_csv_outputs(
        hourly=hourly,
        daily_points=daily_points,
        daily_aggregate=daily_aggregate,
        summary=summary,
        comparison=comparison,
        paths=combined_paths,
    )

    raw_paths = raw_output_mapping(requests)
    csv_paths = {
        key: value
        for key, value in combined_paths.items()
        if key not in {"metadata_json", "validation_json"}
    }
    checksums = file_checksums({**raw_paths, **csv_paths})
    core_validation = validation_report(
        requests=requests,
        hourly=hourly,
        daily_points=daily_points,
        daily_aggregate=daily_aggregate,
        summary=summary,
        comparison=comparison,
        paths=csv_paths,
        checksums=checksums.copy(),
    )
    metadata = metadata_payload(
        requests=requests,
        retrievals=retrievals,
        extraction_metadata=extraction_metadata,
        validation=core_validation,
        output_paths={**raw_paths, **combined_paths},
        checksums=checksums.copy(),
    )
    checksums["metadata_json"] = write_json(combined_paths["metadata_json"], metadata)
    validation = validation_report(
        requests=requests,
        hourly=hourly,
        daily_points=daily_points,
        daily_aggregate=daily_aggregate,
        summary=summary,
        comparison=comparison,
        paths=combined_paths,
        checksums=checksums.copy(),
    )
    checksums["validation_json"] = write_json(combined_paths["validation_json"], validation)

    return {
        "mode": "reuse_raw" if reuse_raw else "download",
        "passed": validation["passed"],
        "issues": validation["issues"],
        "dataset": DATASET_ID,
        "request_count": len(requests),
        "planned_requests": request_plan_payload(requests),
        "output_paths": output_path_strings({**raw_paths, **combined_paths}),
        "checksums": checksums,
        "validation_json": relative_path(combined_paths["validation_json"]),
        "metadata_json": relative_path(combined_paths["metadata_json"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Plan or run the fixed pilot."""
    requests = planned_requests()
    if args.reuse_raw:
        return run_download(requests, reuse_raw=True)
    if not args.download:
        return dry_run_result(requests)
    return run_download(requests)


def main() -> None:
    """CLI entry point."""
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not result.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
