"""ERA5-Land v2 weather ingestion helpers.

The module is import-safe: importing it does not contact CDS, import cdsapi,
open NetCDF files, or create local outputs. Work happens only through explicit
functions such as :func:`run_ingestion`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from wind_forecast.manifests import DatasetManifest, manifest_to_json, sha256_file
from wind_forecast.paths import project_root


DATASET_ID = "reanalysis-era5-land"
DATASET_URL = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land"
PROVIDER = "Copernicus Climate Data Store"
REQUEST_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
)
REQUEST_TIMES = tuple(f"{hour:02d}:00" for hour in range(24))
DEFAULT_CALM_THRESHOLD_M_S = 0.5
EXPECTED_STATION_COUNT = 17
UNMATCHED_STATION_ID = "1200579"
TRANSFORMATION_VERSION = "era5_land_v2_weather_foundation_2A.12"

VARIABLE_ALIASES = {
    "temperature_2m_k": ("t2m", "2m_temperature"),
    "u10_m_s": ("u10", "10m_u_component_of_wind"),
    "v10_m_s": ("v10", "10m_v_component_of_wind"),
}
EXPECTED_NETCDF_UNITS = {
    "temperature_2m_k": {"k", "kelvin"},
    "u10_m_s": {"ms**-1", "ms-1", "m/s"},
    "v10_m_s": {"ms**-1", "ms-1", "m/s"},
}

HOURLY_COLUMNS = (
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
    "wind_direction_deg_from",
    "is_calm_or_near_calm",
)

DAILY_POINT_COLUMNS = (
    "date_utc",
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
)

DAILY_AGGREGATE_COLUMNS = (
    "date_utc",
    "point_count",
    "expected_point_count",
    "missing_point_count",
    "temperature_2m_c_mean",
    "temperature_2m_k_mean",
    "wind_speed_m_s_mean",
    "u10_m_s_mean",
    "v10_m_s_mean",
    "vector_mean_wind_speed_m_s",
    "vector_mean_wind_direction_deg_from",
    "calm_or_near_calm_share",
)


class Era5LandIngestionError(ValueError):
    """Raised when ERA5-Land ingestion inputs or outputs are invalid."""


@dataclass(frozen=True)
class StationMapping:
    """One approved exact-match IPMA station mapping."""

    station_id: str
    station_name: str
    latitude: float
    longitude: float
    matched_official_identifier: str
    source_endpoint: str
    match_method: str
    confidence: str


@dataclass(frozen=True)
class Era5LandChunk:
    """One inclusive same-month ERA5-Land date chunk."""

    start: date
    end: date

    @property
    def period_label(self) -> str:
        return f"{self.start.isoformat()}_{self.end.isoformat()}"


@dataclass(frozen=True)
class Era5LandPaths:
    """Deterministic output paths for one station/chunk partition."""

    raw_netcdf: Path
    hourly_csv: Path
    daily_points_csv: Path
    status_json: Path


@dataclass(frozen=True)
class Era5LandJob:
    """One station/chunk request and its deterministic paths."""

    station: StationMapping
    chunk: Era5LandChunk
    request: Mapping[str, Any]
    paths: Era5LandPaths


def utc_timestamp() -> str:
    """Return a second-resolution ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_source_date(value: str | date, argument_name: str = "date") -> date:
    """Parse and validate a YYYY-MM-DD date."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{argument_name} must be formatted as YYYY-MM-DD.") from exc


def iter_same_month_chunks(start_date: str | date, end_date: str | date) -> list[Era5LandChunk]:
    """Return inclusive chunks that never cross a calendar-month boundary."""
    start = parse_source_date(start_date, "start_date")
    end = parse_source_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date.")

    chunks = []
    current = start
    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        chunk_end = min(end, next_month - timedelta(days=1))
        chunks.append(Era5LandChunk(start=current, end=chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def requested_days(chunk: Era5LandChunk) -> list[str]:
    """Return day strings for one validated same-month chunk."""
    if chunk.start.year != chunk.end.year or chunk.start.month != chunk.end.month:
        raise ValueError("ERA5-Land request chunks must stay within one calendar month.")
    return [
        f"{(chunk.start + timedelta(days=offset)).day:02d}"
        for offset in range((chunk.end - chunk.start).days + 1)
    ]


def nearest_era5_land_grid_coordinate(value: float) -> float:
    """Round a coordinate to the nearest ERA5-Land 0.1-degree grid line."""
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def request_area_for_station(latitude: float, longitude: float) -> list[float]:
    """Build a single-grid-cell CDS area for a station coordinate."""
    grid_latitude = nearest_era5_land_grid_coordinate(latitude)
    grid_longitude = nearest_era5_land_grid_coordinate(longitude)
    return [grid_latitude, grid_longitude, grid_latitude, grid_longitude]


def build_cds_request(chunk: Era5LandChunk, station: StationMapping) -> dict[str, Any]:
    """Build one deterministic ERA5-Land CDS request."""
    return {
        "variable": list(REQUEST_VARIABLES),
        "year": f"{chunk.start.year:04d}",
        "month": f"{chunk.start.month:02d}",
        "day": requested_days(chunk),
        "time": list(REQUEST_TIMES),
        "area": request_area_for_station(station.latitude, station.longitude),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def load_station_mapping(
    station_mapping_csv: str | Path,
    *,
    station_ids: Sequence[str] | None = None,
) -> list[StationMapping]:
    """Load and validate the approved 17 exact-match IPMA station mappings."""
    path = Path(station_mapping_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise Era5LandIngestionError(f"Station mapping CSV is empty: {path}.")

    exact_rows = []
    unmatched_seen = False
    for row in rows:
        station_id = str(row.get("v1_identifier", "")).strip()
        status = str(row.get("status", "")).strip()
        if station_id == UNMATCHED_STATION_ID and status == "no_match":
            unmatched_seen = True
            continue
        if status == "exact_match":
            exact_rows.append(row)

    if not unmatched_seen:
        raise Era5LandIngestionError(f"Expected unmatched station {UNMATCHED_STATION_ID} was not recorded.")
    if len(exact_rows) != EXPECTED_STATION_COUNT:
        raise Era5LandIngestionError(
            f"Expected {EXPECTED_STATION_COUNT} exact-match station mappings; found {len(exact_rows)}."
        )

    stations = [_station_from_row(row) for row in exact_rows]
    stations = sorted(stations, key=lambda item: item.station_id)
    if station_ids is None:
        return stations

    requested = tuple(str(item) for item in station_ids)
    requested_set = set(requested)
    if len(requested_set) != len(requested):
        raise Era5LandIngestionError("--station-id values must be unique.")
    available = {station.station_id for station in stations}
    missing = sorted(requested_set.difference(available))
    if missing:
        raise Era5LandIngestionError(f"Requested station IDs are not approved exact matches: {missing}.")
    return [station for station in stations if station.station_id in requested_set]


def _station_from_row(row: Mapping[str, str]) -> StationMapping:
    station_id = str(row.get("v1_identifier", "")).strip()
    official_id = str(row.get("matched_official_identifier", "")).strip()
    status = str(row.get("status", "")).strip()
    match_method = str(row.get("match_method", "")).strip()
    confidence = str(row.get("confidence", "")).strip()
    if status != "exact_match" or station_id != official_id:
        raise Era5LandIngestionError(f"Station {station_id!r} is not an exact identifier match.")
    if match_method != "exact_string" or confidence != "high":
        raise Era5LandIngestionError(f"Station {station_id!r} does not have high-confidence exact matching.")
    try:
        latitude = float(str(row.get("latitude", "")).strip())
        longitude = float(str(row.get("longitude", "")).strip())
    except ValueError as exc:
        raise Era5LandIngestionError(f"Station {station_id!r} has invalid coordinates.") from exc
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise Era5LandIngestionError(f"Station {station_id!r} has non-finite coordinates.")
    return StationMapping(
        station_id=station_id,
        station_name=str(row.get("station_name", "")).strip(),
        latitude=latitude,
        longitude=longitude,
        matched_official_identifier=official_id,
        source_endpoint=str(row.get("source_endpoint", "")).strip(),
        match_method=match_method,
        confidence=confidence,
    )


def era5_land_paths(output_root: str | Path, station_id: str, chunk: Era5LandChunk) -> Era5LandPaths:
    """Return deterministic paths below ``output_root/era5_land``."""
    root = Path(output_root) / "era5_land"
    station_part = f"station_id={station_id}"
    period_part = f"period={chunk.period_label}"
    return Era5LandPaths(
        raw_netcdf=root / "raw" / station_part / period_part / "era5_land.nc",
        hourly_csv=root / "hourly" / station_part / period_part / "hourly.csv",
        daily_points_csv=root / "daily_points" / station_part / period_part / "daily_points.csv",
        status_json=root / "metadata" / station_part / period_part / "status.json",
    )


def aggregate_path(output_root: str | Path, start_date: str | date, end_date: str | date) -> Path:
    """Return the requested-period aggregate daily-weather CSV path."""
    start = parse_source_date(start_date, "start_date").isoformat()
    end = parse_source_date(end_date, "end_date").isoformat()
    return (
        Path(output_root)
        / "era5_land"
        / "daily_aggregate"
        / f"period={start}_{end}"
        / "daily_weather_aggregate.csv"
    )


def comparison_path(output_root: str | Path, start_date: str | date, end_date: str | date) -> Path:
    """Return the requested-period prior-pilot comparison CSV path."""
    start = parse_source_date(start_date, "start_date").isoformat()
    end = parse_source_date(end_date, "end_date").isoformat()
    return (
        Path(output_root)
        / "era5_land"
        / "comparisons"
        / f"period={start}_{end}"
        / "prior_era5_pilot_overlap.csv"
    )


def manifest_path(output_root: str | Path) -> Path:
    """Return the ERA5-Land v2 weather manifest path."""
    return Path(output_root) / "era5_land" / "manifests" / "era5_land_weather_manifest.json"


def retrieve_era5_land(dataset: str, request: Mapping[str, Any], target: str | Path) -> None:
    """Retrieve one ERA5-Land NetCDF file through cdsapi.

    The cdsapi import is intentionally lazy so module import remains safe in
    environments without CDS credentials or the client package installed.
    """
    import cdsapi

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve(dataset, dict(request), str(target_path))


def coordinate_name(dataset: Any, candidates: tuple[str, ...], description: str) -> str:
    """Find a coordinate name in an xarray dataset."""
    for candidate in candidates:
        if candidate in dataset.coords or candidate in dataset.dims:
            return candidate
    raise Era5LandIngestionError(f"NetCDF does not contain a {description} coordinate.")


def time_coordinate_name(dataset: Any) -> str:
    """Find the time coordinate used by an ERA5 NetCDF file."""
    for candidate in ("valid_time", "time"):
        if candidate in dataset.coords or candidate in dataset.dims:
            return candidate
    for name, coord in dataset.coords.items():
        if "datetime64" in str(coord.dtype):
            return str(name)
    raise Era5LandIngestionError("NetCDF does not contain a recognizable time coordinate.")


def source_variable(dataset: Any, aliases: Iterable[str], output_name: str) -> str:
    """Find the ERA5 variable name behind one output field."""
    for alias in aliases:
        if alias in dataset.data_vars:
            return alias
    available = ", ".join(str(name) for name in dataset.data_vars)
    raise Era5LandIngestionError(f"NetCDF is missing {output_name}; available variables: {available}.")


def normalized_unit(value: object) -> str:
    """Return a compact unit string for conservative comparisons."""
    return str(value).strip().casefold().replace(" ", "")


def validate_netcdf_units(dataset: Any, source_variables: Mapping[str, str]) -> dict[str, Any]:
    """Validate expected ERA5-Land source units and return recorded source units."""
    units = {}
    for output_name, source_name in source_variables.items():
        raw_unit = dataset[source_name].attrs.get("units")
        if raw_unit is None:
            raise Era5LandIngestionError(f"NetCDF variable {source_name!r} is missing a units attribute.")
        if normalized_unit(raw_unit) not in EXPECTED_NETCDF_UNITS[output_name]:
            raise Era5LandIngestionError(
                f"NetCDF variable {source_name!r} has unexpected units {raw_unit!r}; "
                f"expected one of {sorted(EXPECTED_NETCDF_UNITS[output_name])}."
            )
        units[str(source_name)] = raw_unit
    return units


def longitude_for_selection(dataset_longitudes: Any, station_longitude: float) -> float:
    """Adjust negative longitudes for 0..360 datasets when necessary."""
    values = [float(value) for value in dataset_longitudes.values.ravel()]
    if not values:
        raise Era5LandIngestionError("Longitude coordinate is empty.")
    if min(values) >= 0.0 and station_longitude < 0.0:
        return station_longitude % 360.0
    return station_longitude


def scalar_coordinate(point_dataset: Any, name: str, description: str) -> float:
    """Return a scalar selected coordinate."""
    coord = point_dataset[name]
    if coord.size != 1:
        raise Era5LandIngestionError(f"Expected exactly one {description}; found {coord.size}.")
    return float(coord.values.reshape(-1)[0])


def series_from_data_array(data_array: Any, time_name: str, output_name: str) -> Any:
    """Return a one-dimensional data array aligned to the time coordinate."""
    squeezed = data_array.squeeze(drop=True)
    extra_dims = [dim for dim in squeezed.dims if dim != time_name]
    if extra_dims:
        raise Era5LandIngestionError(f"{output_name} has unexpected dimensions after extraction: {extra_dims}.")
    if time_name not in squeezed.dims:
        raise Era5LandIngestionError(f"{output_name} is not aligned to the {time_name} coordinate.")
    return squeezed.transpose(time_name).values


def load_hourly_frame(
    raw_path: str | Path,
    *,
    station: StationMapping,
    calm_threshold: float = DEFAULT_CALM_THRESHOLD_M_S,
) -> tuple[Any, dict[str, Any]]:
    """Open one NetCDF file, extract one point, and build hourly rows."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(raw_path) as dataset:
        latitude_name = coordinate_name(dataset, ("latitude", "lat"), "latitude")
        longitude_name = coordinate_name(dataset, ("longitude", "lon"), "longitude")
        time_name = time_coordinate_name(dataset)
        selection_longitude = longitude_for_selection(dataset[longitude_name], station.longitude)
        point = dataset.sel(
            {latitude_name: station.latitude, longitude_name: selection_longitude},
            method="nearest",
        ).load()

    grid_latitude = scalar_coordinate(point, latitude_name, "latitude")
    grid_longitude = scalar_coordinate(point, longitude_name, "longitude")
    source_variables = {
        output_name: source_variable(point, aliases, output_name)
        for output_name, aliases in VARIABLE_ALIASES.items()
    }
    netcdf_variable_units = validate_netcdf_units(point, source_variables)
    timestamps = pd.to_datetime(point[time_name].values, utc=True)
    if timestamps.empty:
        raise Era5LandIngestionError("Extracted ERA5-Land point contains no hourly timestamps.")

    temperature_k = series_from_data_array(point[source_variables["temperature_2m_k"]], time_name, "temperature_2m_k")
    u10 = series_from_data_array(point[source_variables["u10_m_s"]], time_name, "u10_m_s")
    v10 = series_from_data_array(point[source_variables["v10_m_s"]], time_name, "v10_m_s")

    wind_speed = np.sqrt(np.square(u10) + np.square(v10))
    direction = (180.0 + np.degrees(np.arctan2(u10, v10))) % 360.0
    calm = np.isfinite(wind_speed) & (wind_speed < calm_threshold)
    direction = np.where(calm, np.nan, direction)

    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "station_id": station.station_id,
            "station_name": station.station_name,
            "station_latitude": station.latitude,
            "station_longitude": station.longitude,
            "grid_latitude": grid_latitude,
            "grid_longitude": grid_longitude,
            "temperature_2m_k": temperature_k,
            "temperature_2m_c": temperature_k - 273.15,
            "u10_m_s": u10,
            "v10_m_s": v10,
            "wind_speed_m_s": wind_speed,
            "wind_direction_deg_from": direction,
            "is_calm_or_near_calm": calm,
        },
        columns=list(HOURLY_COLUMNS),
    ).sort_values(["station_id", "timestamp_utc"])

    return frame, {
        "coordinate_names": {"latitude": latitude_name, "longitude": longitude_name, "time": time_name},
        "selected_grid_coordinate": {"latitude": grid_latitude, "longitude": grid_longitude},
        "source_variables": source_variables,
        "netcdf_dimensions_after_point_extraction": {str(key): int(value) for key, value in point.sizes.items()},
        "netcdf_variable_units": netcdf_variable_units,
    }


def expected_hourly_index(start: date, end: date) -> Any:
    """Return the expected hourly UTC index for the inclusive date range."""
    import pandas as pd

    start_timestamp = pd.Timestamp(start.isoformat(), tz="UTC")
    end_timestamp = pd.Timestamp(end.isoformat(), tz="UTC") + pd.Timedelta(hours=23)
    return pd.date_range(start_timestamp, end_timestamp, freq="h")


def daily_point_aggregates(
    hourly: Any,
    chunk: Era5LandChunk,
    *,
    calm_threshold: float = DEFAULT_CALM_THRESHOLD_M_S,
) -> Any:
    """Aggregate one station's hourly ERA5-Land values to daily UTC rows."""
    import numpy as np
    import pandas as pd

    frame = hourly.copy()
    frame["_timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["_date"] = frame["_timestamp"].dt.strftime("%Y-%m-%d")
    expected_dates = pd.date_range(chunk.start.isoformat(), chunk.end.isoformat(), freq="D").strftime("%Y-%m-%d")
    station_values = frame.iloc[0]
    rows = []
    for day in expected_dates:
        group = frame.loc[frame["_date"] == day]
        hourly_count = int(group["_timestamp"].nunique())
        mean_u = float(group["u10_m_s"].mean()) if not group.empty else math.nan
        mean_v = float(group["v10_m_s"].mean()) if not group.empty else math.nan
        vector_speed = math.sqrt(mean_u * mean_u + mean_v * mean_v) if math.isfinite(mean_u) and math.isfinite(mean_v) else math.nan
        if math.isfinite(vector_speed) and vector_speed >= calm_threshold:
            vector_direction = (180.0 + math.degrees(math.atan2(mean_u, mean_v))) % 360.0
        else:
            vector_direction = math.nan
        calm_count = int(group["is_calm_or_near_calm"].sum()) if not group.empty else 0
        rows.append(
            {
                "date_utc": day,
                "station_id": station_values["station_id"],
                "station_name": station_values["station_name"],
                "station_latitude": station_values["station_latitude"],
                "station_longitude": station_values["station_longitude"],
                "grid_latitude": station_values["grid_latitude"],
                "grid_longitude": station_values["grid_longitude"],
                "hourly_count": hourly_count,
                "expected_count": 24,
                "missing_count": max(24 - hourly_count, 0),
                "temperature_2m_c_mean": group["temperature_2m_c"].mean(),
                "temperature_2m_c_min": group["temperature_2m_c"].min(),
                "temperature_2m_c_max": group["temperature_2m_c"].max(),
                "temperature_2m_k_mean": group["temperature_2m_k"].mean(),
                "temperature_2m_k_min": group["temperature_2m_k"].min(),
                "temperature_2m_k_max": group["temperature_2m_k"].max(),
                "wind_speed_m_s_mean": group["wind_speed_m_s"].mean(),
                "wind_speed_m_s_max": group["wind_speed_m_s"].max(),
                "wind_speed_m_s_std": group["wind_speed_m_s"].std(),
                "u10_m_s_mean": mean_u if math.isfinite(mean_u) else np.nan,
                "v10_m_s_mean": mean_v if math.isfinite(mean_v) else np.nan,
                "vector_mean_wind_speed_m_s": vector_speed if math.isfinite(vector_speed) else np.nan,
                "vector_mean_wind_direction_deg_from": vector_direction,
                "calm_or_near_calm_count": calm_count,
                "calm_or_near_calm_share": calm_count / hourly_count if hourly_count else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=list(DAILY_POINT_COLUMNS))


def aggregate_daily_weather(daily_points: Any, *, expected_point_count: int) -> Any:
    """Equal-weight aggregate daily station rows across valid mapped points."""
    import numpy as np
    import pandas as pd

    rows = []
    for date_utc, group in daily_points.groupby("date_utc", sort=True):
        valid_points = group.dropna(
            subset=["temperature_2m_c_mean", "wind_speed_m_s_mean", "u10_m_s_mean", "v10_m_s_mean"]
        )
        point_count = int(len(valid_points))
        mean_u = float(valid_points["u10_m_s_mean"].mean()) if point_count else math.nan
        mean_v = float(valid_points["v10_m_s_mean"].mean()) if point_count else math.nan
        vector_speed = math.sqrt(mean_u * mean_u + mean_v * mean_v) if math.isfinite(mean_u) and math.isfinite(mean_v) else math.nan
        if math.isfinite(vector_speed) and vector_speed >= DEFAULT_CALM_THRESHOLD_M_S:
            vector_direction = (180.0 + math.degrees(math.atan2(mean_u, mean_v))) % 360.0
        else:
            vector_direction = math.nan
        calm_count = int(valid_points["calm_or_near_calm_count"].sum()) if point_count else 0
        hourly_count = int(valid_points["hourly_count"].sum()) if point_count else 0
        rows.append(
            {
                "date_utc": date_utc,
                "point_count": point_count,
                "expected_point_count": expected_point_count,
                "missing_point_count": expected_point_count - point_count,
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
    return pd.DataFrame(rows, columns=list(DAILY_AGGREGATE_COLUMNS))


def validate_partition_outputs(hourly: Any, daily_points: Any, chunk: Era5LandChunk) -> dict[str, Any]:
    """Validate one station/chunk normalized output pair."""
    import numpy as np
    import pandas as pd

    issues = []
    missing_hourly_columns = [column for column in HOURLY_COLUMNS if column not in hourly.columns]
    missing_daily_columns = [column for column in DAILY_POINT_COLUMNS if column not in daily_points.columns]
    if missing_hourly_columns:
        issues.append(f"Hourly output is missing columns: {missing_hourly_columns}.")
    if missing_daily_columns:
        issues.append(f"Daily-point output is missing columns: {missing_daily_columns}.")

    timestamps = pd.to_datetime(hourly["timestamp_utc"], utc=True) if "timestamp_utc" in hourly else pd.DatetimeIndex([])
    expected = expected_hourly_index(chunk.start, chunk.end)
    actual_index = pd.DatetimeIndex(timestamps)
    missing = expected.difference(actual_index)
    unexpected = actual_index.difference(expected)
    duplicate_count = int(actual_index.duplicated().sum())
    if duplicate_count:
        issues.append(f"Found {duplicate_count} duplicate hourly timestamps.")
    if len(missing):
        issues.append(f"Missing {len(missing)} expected hourly timestamps.")
    if len(unexpected):
        issues.append(f"Found {len(unexpected)} timestamps outside the requested UTC range.")

    nullable_hourly = {"wind_direction_deg_from"}
    null_counts = {column: int(hourly[column].isna().sum()) for column in hourly.columns}
    for column, null_count in null_counts.items():
        if null_count and column not in nullable_hourly:
            issues.append(f"Hourly column {column} has {null_count} null values.")

    finite_checks = {}
    numeric_columns = [
        "station_latitude",
        "station_longitude",
        "grid_latitude",
        "grid_longitude",
        "temperature_2m_k",
        "temperature_2m_c",
        "u10_m_s",
        "v10_m_s",
        "wind_speed_m_s",
        "wind_direction_deg_from",
    ]
    for column in numeric_columns:
        if column not in hourly:
            continue
        series = pd.to_numeric(hourly[column], errors="coerce")
        finite = bool(np.isfinite(series.dropna()).all())
        finite_checks[column] = finite
        if not finite:
            issues.append(f"Hourly column {column} contains non-finite numeric values.")

    directions = pd.to_numeric(hourly.get("wind_direction_deg_from"), errors="coerce")
    if not directions.dropna().between(0.0, 360.0, inclusive="left").all():
        issues.append("Hourly wind directions must be in [0, 360) when not null.")
    speeds = pd.to_numeric(hourly.get("wind_speed_m_s"), errors="coerce")
    if (speeds.dropna() < 0.0).any():
        issues.append("Hourly wind speeds must be non-negative.")

    expected_daily_rows = (chunk.end - chunk.start).days + 1
    if len(daily_points) != expected_daily_rows:
        issues.append(f"Expected {expected_daily_rows} daily rows but found {len(daily_points)}.")
    incomplete_days = int((daily_points["hourly_count"] != 24).sum()) if "hourly_count" in daily_points else 0
    if incomplete_days:
        issues.append(f"Found {incomplete_days} incomplete daily point rows.")

    return {
        "generated_at_utc": utc_timestamp(),
        "validation_status": "complete" if not issues else "invalid",
        "passed": not issues,
        "issues": issues,
        "hourly_rows": int(len(hourly)),
        "daily_point_rows": int(len(daily_points)),
        "timestamp_coverage": {
            "expected_start_utc": expected[0].isoformat().replace("+00:00", "Z"),
            "expected_end_utc": expected[-1].isoformat().replace("+00:00", "Z"),
            "actual_start_utc": actual_index.min().isoformat().replace("+00:00", "Z") if len(actual_index) else None,
            "actual_end_utc": actual_index.max().isoformat().replace("+00:00", "Z") if len(actual_index) else None,
            "missing_timestamp_count": int(len(missing)),
            "unexpected_timestamp_count": int(len(unexpected)),
            "duplicate_timestamp_count": duplicate_count,
        },
        "null_counts": null_counts,
        "finite_checks": finite_checks,
        "timezone": "UTC",
    }


def write_partition_outputs(
    *,
    output_root: Path,
    job: Era5LandJob,
    hourly: Any,
    daily_points: Any,
    validation: Mapping[str, Any],
    extraction_metadata: Mapping[str, Any],
    retrieval_started_at_utc: str,
    retrieval_finished_at_utc: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one station/chunk output partition and status JSON."""
    paths = job.paths
    existing = [paths.hourly_csv, paths.daily_points_csv, paths.status_json]
    if not overwrite and any(path.exists() for path in existing):
        raise FileExistsError("ERA5-Land output partition exists; use resume or overwrite explicitly.")

    raw_checksum = sha256_file(paths.raw_netcdf)
    hourly_checksum = write_csv(paths.hourly_csv, hourly)
    daily_checksum = write_csv(paths.daily_points_csv, daily_points)
    status_payload = {
        "source_dataset": DATASET_ID,
        "source_url": DATASET_URL,
        "provider": PROVIDER,
        "station": station_payload(job.station),
        "requested_period": {
            "start_date": job.chunk.start.isoformat(),
            "end_date": job.chunk.end.isoformat(),
            "timezone": "UTC",
        },
        "request": dict(job.request),
        "retrieval": {
            "started_at_utc": retrieval_started_at_utc,
            "finished_at_utc": retrieval_finished_at_utc,
            "service_status": "cds_retrieve_completed",
        },
        "paths": {
            "raw_netcdf": _manifest_path(paths.raw_netcdf, output_root=output_root),
            "hourly_csv": _manifest_path(paths.hourly_csv, output_root=output_root),
            "daily_points_csv": _manifest_path(paths.daily_points_csv, output_root=output_root),
            "status_json": _manifest_path(paths.status_json, output_root=output_root),
        },
        "checksums": {
            "raw_netcdf_sha256": raw_checksum,
            "hourly_csv_sha256": hourly_checksum,
            "daily_points_csv_sha256": daily_checksum,
        },
        "validation": dict(validation),
        "extraction": dict(extraction_metadata),
        "units": output_units(),
        "aggregation_contract": aggregation_contract(),
        "transformation_version": TRANSFORMATION_VERSION,
    }
    status_checksum = write_json(paths.status_json, status_payload)
    result = partition_summary(output_root=output_root, paths=paths, status_payload=status_payload)
    result["checksums"]["status_json_sha256"] = status_checksum
    return result


def write_invalid_status(
    *,
    output_root: Path,
    job: Era5LandJob,
    message: str,
    retrieval_started_at_utc: str | None,
    retrieval_finished_at_utc: str | None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write invalid metadata while preserving any raw NetCDF evidence."""
    paths = job.paths
    if paths.status_json.exists() and not overwrite:
        raise FileExistsError("ERA5-Land invalid status already exists; use --overwrite to replace it.")
    checksums = {}
    raw_path_value = None
    if paths.raw_netcdf.exists():
        checksums["raw_netcdf_sha256"] = sha256_file(paths.raw_netcdf)
        raw_path_value = _manifest_path(paths.raw_netcdf, output_root=output_root)
    status_payload = {
        "source_dataset": DATASET_ID,
        "provider": PROVIDER,
        "station": station_payload(job.station),
        "requested_period": {
            "start_date": job.chunk.start.isoformat(),
            "end_date": job.chunk.end.isoformat(),
            "timezone": "UTC",
        },
        "request": dict(job.request),
        "retrieval": {
            "started_at_utc": retrieval_started_at_utc,
            "finished_at_utc": retrieval_finished_at_utc,
            "service_status": "failed_or_invalid",
        },
        "paths": {
            "raw_netcdf": raw_path_value,
            "hourly_csv": None,
            "daily_points_csv": None,
            "status_json": _manifest_path(paths.status_json, output_root=output_root),
        },
        "checksums": checksums,
        "validation": {
            "validation_status": "invalid",
            "passed": False,
            "issues": [message],
        },
        "transformation_version": TRANSFORMATION_VERSION,
    }
    status_checksum = write_json(paths.status_json, status_payload)
    return {
        "station_id": job.station.station_id,
        "period_start": job.chunk.start.isoformat(),
        "period_end": job.chunk.end.isoformat(),
        "status": "invalid",
        "paths": {"status_json": paths.status_json, "raw_netcdf": paths.raw_netcdf if paths.raw_netcdf.exists() else None},
        "checksums": {"status_json_sha256": status_checksum, **checksums},
        "warnings": [message],
    }


def partition_is_verified(output_root: str | Path, station_id: str, chunk: Era5LandChunk) -> bool:
    """Return True when a station/chunk partition exists and checksums match."""
    paths = era5_land_paths(output_root, station_id, chunk)
    if not all(path.is_file() for path in (paths.raw_netcdf, paths.hourly_csv, paths.daily_points_csv, paths.status_json)):
        return False
    try:
        status = json.loads(paths.status_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    validation = status.get("validation")
    checksums = status.get("checksums")
    if not isinstance(validation, Mapping) or not isinstance(checksums, Mapping):
        return False
    if validation.get("validation_status") != "complete":
        return False
    return (
        checksums.get("raw_netcdf_sha256") == sha256_file(paths.raw_netcdf)
        and checksums.get("hourly_csv_sha256") == sha256_file(paths.hourly_csv)
        and checksums.get("daily_points_csv_sha256") == sha256_file(paths.daily_points_csv)
    )


def load_partition_summary(output_root: str | Path, station_id: str, chunk: Era5LandChunk) -> dict[str, Any]:
    """Load a verified station/chunk summary from status metadata."""
    paths = era5_land_paths(output_root, station_id, chunk)
    status = json.loads(paths.status_json.read_text(encoding="utf-8"))
    result = partition_summary(output_root=Path(output_root), paths=paths, status_payload=status)
    result["checksums"]["status_json_sha256"] = sha256_file(paths.status_json)
    return result


def partition_summary(
    *,
    output_root: Path,
    paths: Era5LandPaths,
    status_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a JSON-ready summary for one partition status payload."""
    validation = status_payload.get("validation", {})
    station = status_payload.get("station", {})
    period = status_payload.get("requested_period", {})
    checksums = dict(status_payload.get("checksums") or {})
    return {
        "station_id": str(station.get("station_id")),
        "period_start": str(period.get("start_date")),
        "period_end": str(period.get("end_date")),
        "status": str(validation.get("validation_status")),
        "hourly_rows": int(validation.get("hourly_rows", 0)),
        "daily_point_rows": int(validation.get("daily_point_rows", 0)),
        "paths": {
            "raw_netcdf": paths.raw_netcdf,
            "hourly_csv": paths.hourly_csv,
            "daily_points_csv": paths.daily_points_csv,
            "status_json": paths.status_json,
        },
        "manifest_paths": dict(status_payload.get("paths") or {}),
        "checksums": checksums,
        "warnings": list(validation.get("issues") or []),
        "retrieval_finished_at_utc": (status_payload.get("retrieval") or {}).get("finished_at_utc"),
        "selected_grid_coordinate": (status_payload.get("extraction") or {}).get("selected_grid_coordinate"),
    }


def write_csv(path: Path, frame: Any) -> str:
    """Write a deterministic CSV if content changed and return its SHA-256."""
    text = frame.to_csv(index=False, lineterminator="\n")
    write_text_if_changed(path, text)
    return sha256_file(path)


def write_json(path: Path, payload: Any) -> str:
    """Write deterministic JSON if content changed and return its SHA-256."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    write_text_if_changed(path, text)
    return sha256_file(path)


def write_text_if_changed(path: Path, text: str) -> None:
    """Write text only when the target content differs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8", newline="\n")


def output_units() -> dict[str, str]:
    """Return units used by normalized ERA5-Land weather outputs."""
    return {
        "timestamp_utc": "UTC",
        "date_utc": "UTC",
        "station_latitude": "degrees_north",
        "station_longitude": "degrees_east",
        "grid_latitude": "degrees_north",
        "grid_longitude": "degrees_east",
        "temperature_2m_k": "K",
        "temperature_2m_c": "degree_Celsius",
        "u10_m_s": "m s-1",
        "v10_m_s": "m s-1",
        "wind_speed_m_s": "m s-1",
        "wind_direction_deg_from": "degree_from",
        "is_calm_or_near_calm": "boolean",
    }


def aggregation_contract() -> dict[str, Any]:
    """Return the approved v2 weather aggregation contract."""
    return {
        "timezone": "UTC",
        "daily_boundaries": "00:00:00Z through 23:00:00Z inclusive per date_utc",
        "temperature": "Kelvin converted to Celsius; daily point mean; equal-weight aggregate across valid mapped points.",
        "hourly_wind_speed": "sqrt(u10^2 + v10^2)",
        "wind_direction_deg_from": "(180 + degrees(atan2(u10, v10))) % 360",
        "daily_direction": "computed from daily mean u10 and v10 components",
        "aggregate_strategy": "equal-weight mean across valid mapped station points",
        "calm_threshold_m_s": DEFAULT_CALM_THRESHOLD_M_S,
        "calm_direction_policy": "direction is null below the calm threshold",
    }


def station_payload(station: StationMapping) -> dict[str, Any]:
    """Return JSON-ready station metadata."""
    return {
        "station_id": station.station_id,
        "station_name": station.station_name,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "matched_official_identifier": station.matched_official_identifier,
        "source_endpoint": station.source_endpoint,
        "match_method": station.match_method,
        "confidence": station.confidence,
    }


def planned_jobs(
    *,
    output_root: str | Path,
    stations: Sequence[StationMapping],
    chunks: Sequence[Era5LandChunk],
) -> list[Era5LandJob]:
    """Build the deterministic station/chunk job list."""
    jobs = []
    for chunk in chunks:
        for station in stations:
            jobs.append(
                Era5LandJob(
                    station=station,
                    chunk=chunk,
                    request=build_cds_request(chunk, station),
                    paths=era5_land_paths(output_root, station.station_id, chunk),
                )
            )
    return jobs


def run_ingestion(
    *,
    start_date: str,
    end_date: str,
    output_root: str | Path,
    station_mapping: str | Path,
    station_ids: Sequence[str] | None = None,
    max_chunks: int = EXPECTED_STATION_COUNT,
    request_delay: float = 0.0,
    resume: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    prior_pilot_dir: str | Path | None = None,
    retrieve_func: Callable[[str, Mapping[str, Any], Path], None] | None = None,
) -> dict[str, Any]:
    """Run controlled ERA5-Land v2 weather ingestion."""
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive.")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than zero.")
    if request_delay < 0:
        raise ValueError("request_delay must be zero or greater.")

    chunks = iter_same_month_chunks(start_date, end_date)
    stations = load_station_mapping(station_mapping, station_ids=station_ids)
    jobs = planned_jobs(output_root=output_root, stations=stations, chunks=chunks)
    if len(jobs) > max_chunks:
        raise ValueError(f"Planned ERA5-Land chunks ({len(jobs)}) exceed max_chunks={max_chunks}.")

    output_root_path = Path(output_root)
    requested_start = parse_source_date(start_date, "start_date").isoformat()
    requested_end = parse_source_date(end_date, "end_date").isoformat()

    if dry_run:
        return {
            "dry_run": True,
            "dataset": DATASET_ID,
            "variables": list(REQUEST_VARIABLES),
            "network_requests_planned": len(jobs),
            "writes_planned": False,
            "requested_start_date": requested_start,
            "requested_end_date": requested_end,
            "station_count": len(stations),
            "chunk_count": len(jobs),
            "planned_requests": [_job_plan_payload(job, output_root_path) for job in jobs],
            "aggregate_path": str(aggregate_path(output_root_path, requested_start, requested_end)),
            "comparison_path": str(comparison_path(output_root_path, requested_start, requested_end)),
            "manifest_path": str(manifest_path(output_root_path)),
        }

    retriever = retrieve_func or retrieve_era5_land
    results: list[dict[str, Any]] = []
    requests_made = 0
    for index, job in enumerate(jobs):
        if resume and partition_is_verified(output_root_path, job.station.station_id, job.chunk):
            summary = load_partition_summary(output_root_path, job.station.station_id, job.chunk)
            summary["skipped_existing"] = True
            results.append(summary)
            continue

        if not overwrite:
            existing = [job.paths.raw_netcdf, job.paths.hourly_csv, job.paths.daily_points_csv, job.paths.status_json]
            if any(path.exists() for path in existing):
                raise FileExistsError(
                    "ERA5-Land partition already exists; use --resume to skip verified partitions "
                    "or --overwrite to replace explicitly."
                )

        started_at = utc_timestamp()
        finished_at = None
        try:
            job.paths.raw_netcdf.parent.mkdir(parents=True, exist_ok=True)
            requests_made += 1
            retriever(DATASET_ID, job.request, job.paths.raw_netcdf)
            finished_at = utc_timestamp()
            hourly, extraction = load_hourly_frame(job.paths.raw_netcdf, station=job.station)
            daily_points = daily_point_aggregates(hourly, job.chunk)
            validation = validate_partition_outputs(hourly, daily_points, job.chunk)
            if not validation["passed"]:
                write_invalid_status(
                    output_root=output_root_path,
                    job=job,
                    message=f"ERA5-Land validation failed: {validation['issues']}.",
                    retrieval_started_at_utc=started_at,
                    retrieval_finished_at_utc=finished_at,
                    overwrite=overwrite,
                )
                raise Era5LandIngestionError(
                    f"ERA5-Land validation failed for {job.station.station_id}: {validation['issues']}."
                )
            result = write_partition_outputs(
                output_root=output_root_path,
                job=job,
                hourly=hourly,
                daily_points=daily_points,
                validation=validation,
                extraction_metadata=extraction,
                retrieval_started_at_utc=started_at,
                retrieval_finished_at_utc=finished_at,
                overwrite=overwrite,
            )
            results.append(result)
        except Exception as exc:
            if finished_at is None:
                finished_at = utc_timestamp()
            if isinstance(exc, Era5LandIngestionError):
                message = str(exc)
            else:
                message = f"{type(exc).__name__}: {exc}"
            if not job.paths.status_json.exists():
                write_invalid_status(
                    output_root=output_root_path,
                    job=job,
                    message=message,
                    retrieval_started_at_utc=started_at,
                    retrieval_finished_at_utc=finished_at,
                    overwrite=overwrite,
                )
            raise

        if request_delay and index < len(jobs) - 1:
            import time

            time.sleep(request_delay)

    daily_points_all = load_daily_points_for_jobs(output_root_path, jobs)
    aggregate = aggregate_daily_weather(daily_points_all, expected_point_count=len(stations))
    aggregate_checksum = write_csv(aggregate_path(output_root_path, requested_start, requested_end), aggregate)
    comparison = compare_with_prior_pilot(daily_points_all, prior_pilot_dir=prior_pilot_dir)
    comparison_checksum = write_csv(comparison_path(output_root_path, requested_start, requested_end), comparison)
    summaries = [
        load_partition_summary(output_root_path, job.station.station_id, job.chunk)
        for job in jobs
        if partition_is_verified(output_root_path, job.station.station_id, job.chunk)
    ]
    manifest = build_manifest(
        output_root=output_root_path,
        requested_start_date=requested_start,
        requested_end_date=requested_end,
        stations=stations,
        partition_results=summaries,
        aggregate_csv=aggregate_path(output_root_path, requested_start, requested_end),
        aggregate_checksum=aggregate_checksum,
        comparison_csv=comparison_path(output_root_path, requested_start, requested_end),
        comparison_checksum=comparison_checksum,
        aggregate_row_count=len(aggregate),
        aggregate_column_count=len(aggregate.columns),
    )
    manifest_checksum = write_manifest(output_root_path, manifest)
    return {
        "dry_run": False,
        "dataset": DATASET_ID,
        "variables": list(REQUEST_VARIABLES),
        "requested_start_date": requested_start,
        "requested_end_date": requested_end,
        "station_count": len(stations),
        "chunk_count": len(jobs),
        "requests_made": requests_made,
        "partition_results": _json_ready(results),
        "aggregate_path": str(aggregate_path(output_root_path, requested_start, requested_end)),
        "aggregate_sha256": aggregate_checksum,
        "comparison_path": str(comparison_path(output_root_path, requested_start, requested_end)),
        "comparison_sha256": comparison_checksum,
        "manifest_path": str(manifest_path(output_root_path)),
        "manifest_sha256": manifest_checksum,
    }


def load_daily_points_for_jobs(output_root: Path, jobs: Sequence[Era5LandJob]) -> Any:
    """Load verified daily-point CSVs for the requested jobs."""
    import pandas as pd

    frames = []
    missing = []
    for job in jobs:
        if not partition_is_verified(output_root, job.station.station_id, job.chunk):
            missing.append(f"{job.station.station_id}/{job.chunk.period_label}")
            continue
        frames.append(pd.read_csv(job.paths.daily_points_csv))
    if missing:
        raise Era5LandIngestionError(f"Cannot aggregate unverified ERA5-Land partitions: {missing}.")
    if not frames:
        raise Era5LandIngestionError("No verified daily-point partitions are available for aggregation.")
    return pd.concat(frames, ignore_index=True).sort_values(["date_utc", "station_id"])


def compare_with_prior_pilot(daily_points: Any, *, prior_pilot_dir: str | Path | None) -> Any:
    """Compare station-day rows with existing ERA5-Land pilot outputs when present."""
    import pandas as pd

    columns = [
        "station_id",
        "date_utc",
        "new_temperature_2m_c_mean",
        "pilot_temperature_2m_c_mean",
        "temperature_2m_c_difference_new_minus_pilot",
        "new_wind_speed_m_s_mean",
        "pilot_wind_speed_m_s_mean",
        "wind_speed_m_s_difference_new_minus_pilot",
        "new_vector_mean_wind_direction_deg_from",
        "pilot_vector_mean_wind_direction_deg_from",
        "wind_direction_signed_difference_deg_new_minus_pilot",
        "overlap_status",
    ]
    if prior_pilot_dir is None:
        prior_pilot_dir = Path("data/pilot/era5_land")
    prior_path = Path(prior_pilot_dir) / "era5_land_multi_point_2023_winter_summer_daily_points.csv"
    if not prior_path.is_file():
        return pd.DataFrame(columns=columns)

    pilot = pd.read_csv(prior_path)
    required = {
        "station_id",
        "date_utc",
        "temperature_2m_c_mean",
        "wind_speed_m_s_mean",
        "vector_mean_wind_direction_deg_from",
    }
    if not required.issubset(set(pilot.columns)):
        return pd.DataFrame(columns=columns)
    current = daily_points.loc[:, list(required)].rename(
        columns={
            "temperature_2m_c_mean": "new_temperature_2m_c_mean",
            "wind_speed_m_s_mean": "new_wind_speed_m_s_mean",
            "vector_mean_wind_direction_deg_from": "new_vector_mean_wind_direction_deg_from",
        }
    )
    pilot = pilot.loc[:, list(required)].rename(
        columns={
            "temperature_2m_c_mean": "pilot_temperature_2m_c_mean",
            "wind_speed_m_s_mean": "pilot_wind_speed_m_s_mean",
            "vector_mean_wind_direction_deg_from": "pilot_vector_mean_wind_direction_deg_from",
        }
    )
    merged = current.merge(pilot, on=["station_id", "date_utc"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["temperature_2m_c_difference_new_minus_pilot"] = (
        merged["new_temperature_2m_c_mean"] - merged["pilot_temperature_2m_c_mean"]
    )
    merged["wind_speed_m_s_difference_new_minus_pilot"] = (
        merged["new_wind_speed_m_s_mean"] - merged["pilot_wind_speed_m_s_mean"]
    )
    merged["wind_direction_signed_difference_deg_new_minus_pilot"] = merged.apply(
        lambda row: circular_difference_degrees(
            row["new_vector_mean_wind_direction_deg_from"],
            row["pilot_vector_mean_wind_direction_deg_from"],
        ),
        axis=1,
    )
    merged["overlap_status"] = "prior_pilot_overlap"
    return merged.loc[:, columns].sort_values(["station_id", "date_utc"])


def circular_difference_degrees(value: Any, reference: Any) -> float:
    """Return signed circular difference in degrees in [-180, 180]."""
    try:
        if not math.isfinite(float(value)) or not math.isfinite(float(reference)):
            return math.nan
    except (TypeError, ValueError):
        return math.nan
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


def build_manifest(
    *,
    output_root: Path,
    requested_start_date: str,
    requested_end_date: str,
    stations: Sequence[StationMapping],
    partition_results: Sequence[Mapping[str, Any]],
    aggregate_csv: Path,
    aggregate_checksum: str,
    comparison_csv: Path,
    comparison_checksum: str,
    aggregate_row_count: int,
    aggregate_column_count: int,
) -> DatasetManifest:
    """Build a deterministic ERA5-Land v2 weather manifest."""
    path_checksum_map: dict[str, str] = {
        _manifest_path(aggregate_csv, output_root=output_root): aggregate_checksum,
        _manifest_path(comparison_csv, output_root=output_root): comparison_checksum,
    }
    raw_paths = []
    metadata_paths = []
    normalized_paths = []
    retrieval_timestamps = []
    selected_grid = {}
    warnings = []
    for item in partition_results:
        manifest_paths = item.get("manifest_paths") or {}
        checksums = item.get("checksums") or {}
        for path_key, checksum_key in (
            ("raw_netcdf", "raw_netcdf_sha256"),
            ("hourly_csv", "hourly_csv_sha256"),
            ("daily_points_csv", "daily_points_csv_sha256"),
            ("status_json", "status_json_sha256"),
        ):
            path = manifest_paths.get(path_key)
            checksum = checksums.get(checksum_key)
            if path and checksum:
                path_checksum_map[str(path)] = str(checksum)
                if path_key == "raw_netcdf":
                    raw_paths.append(str(path))
                elif path_key == "status_json":
                    metadata_paths.append(str(path))
                else:
                    normalized_paths.append(str(path))
        if item.get("retrieval_finished_at_utc"):
            retrieval_timestamps.append(str(item["retrieval_finished_at_utc"]))
        if item.get("selected_grid_coordinate"):
            selected_grid[str(item["station_id"])] = item["selected_grid_coordinate"]
        warnings.extend(str(warning) for warning in item.get("warnings", []))

    warnings.extend(
        [
            "ERA5-Land weather is a v2 source contract and is not compatible with v1 scalers or trained models without refitting/retraining.",
            f"Unmatched station {UNMATCHED_STATION_ID} is excluded by approved v2 spatial strategy.",
            "Manifest covers only requested ERA5-Land weather partitions; it does not claim full historical weather coverage.",
        ]
    )
    return DatasetManifest(
        dataset_version="v2",
        dataset_role="raw_weather",
        provider=PROVIDER,
        source_identifier=DATASET_ID,
        source_endpoint=DATASET_URL,
        retrieval_timestamp=max(retrieval_timestamps) if retrieval_timestamps else None,
        coverage_start=requested_start_date,
        coverage_end=requested_end_date,
        temporal_granularity="hourly_source_daily_aggregate",
        units=output_units(),
        timezone="UTC",
        geographic_coverage={
            "strategy": "17_exact_match_ipma_station_points_equal_weight_aggregate",
            "excluded_station_ids": [UNMATCHED_STATION_ID],
        },
        station_ids=tuple(station.station_id for station in stations),
        coordinates=tuple(station_payload(station) for station in stations),
        raw_file_paths=tuple(sorted(raw_paths)),
        sha256_checksums=dict(sorted(path_checksum_map.items())),
        row_count=int(aggregate_row_count),
        column_count=int(aggregate_column_count),
        preprocessing_version=TRANSFORMATION_VERSION,
        known_warnings=tuple(sorted(set(warnings))),
        license="Copernicus CDS terms apply; record exact accepted terms during operational retrieval.",
        attribution="Copernicus Climate Data Store ERA5-Land; IPMA station metadata mapping from local pilot evidence.",
        status="v2_weather_ingestion_foundation",
        extra_metadata={
            "requested_range": {
                "start_date": requested_start_date,
                "end_date": requested_end_date,
                "inclusive": True,
            },
            "dataset": DATASET_ID,
            "variables": list(REQUEST_VARIABLES),
            "times": list(REQUEST_TIMES),
            "aggregation_contract": aggregation_contract(),
            "normalized_file_paths": sorted(normalized_paths),
            "metadata_file_paths": sorted(metadata_paths),
            "aggregate_file_path": _manifest_path(aggregate_csv, output_root=output_root),
            "prior_pilot_comparison_file_path": _manifest_path(comparison_csv, output_root=output_root),
            "selected_grid_coordinates_by_station_id": selected_grid,
            "request_contract": {
                "data_format": "netcdf",
                "download_format": "unarchived",
                "one_request_per_station_per_same_month_chunk": True,
                "variables_are_fixed": True,
            },
            "model_compatibility": {
                "v1_scalers_valid_for_v2": False,
                "v1_models_valid_for_v2": False,
                "requires_v2_scaler_refit": True,
                "requires_v2_model_retraining": True,
                "requires_v2_metric_rebaseline": True,
            },
        },
    )


def write_manifest(output_root: str | Path, manifest: DatasetManifest) -> str:
    """Write the ERA5-Land weather manifest and return its checksum."""
    path = manifest_path(output_root)
    write_text_if_changed(path, manifest_to_json(manifest))
    return sha256_file(path)


def _job_plan_payload(job: Era5LandJob, output_root: Path) -> dict[str, Any]:
    return {
        "dataset": DATASET_ID,
        "station": station_payload(job.station),
        "period": {
            "start_date": job.chunk.start.isoformat(),
            "end_date": job.chunk.end.isoformat(),
            "timezone": "UTC",
        },
        "request": dict(job.request),
        "paths": {
            "raw_netcdf": str(job.paths.raw_netcdf),
            "hourly_csv": str(job.paths.hourly_csv),
            "daily_points_csv": str(job.paths.daily_points_csv),
            "status_json": str(job.paths.status_json),
        },
        "manifest_paths": {
            "raw_netcdf": _manifest_path(job.paths.raw_netcdf, output_root=output_root),
            "hourly_csv": _manifest_path(job.paths.hourly_csv, output_root=output_root),
            "daily_points_csv": _manifest_path(job.paths.daily_points_csv, output_root=output_root),
            "status_json": _manifest_path(job.paths.status_json, output_root=output_root),
        },
    }


def _manifest_path(path: str | Path, *, output_root: Path) -> str:
    """Return a stable, non-absolute path suitable for DatasetManifest."""
    raw_path = Path(path)
    try:
        return raw_path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        try:
            return raw_path.resolve().relative_to(output_root.resolve()).as_posix()
        except ValueError:
            return raw_path.name


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "DATASET_ID",
    "DEFAULT_CALM_THRESHOLD_M_S",
    "EXPECTED_STATION_COUNT",
    "REQUEST_TIMES",
    "REQUEST_VARIABLES",
    "DAILY_AGGREGATE_COLUMNS",
    "DAILY_POINT_COLUMNS",
    "HOURLY_COLUMNS",
    "Era5LandChunk",
    "Era5LandIngestionError",
    "Era5LandJob",
    "Era5LandPaths",
    "StationMapping",
    "aggregate_daily_weather",
    "aggregate_path",
    "build_cds_request",
    "build_manifest",
    "compare_with_prior_pilot",
    "comparison_path",
    "daily_point_aggregates",
    "era5_land_paths",
    "iter_same_month_chunks",
    "load_hourly_frame",
    "load_station_mapping",
    "manifest_path",
    "nearest_era5_land_grid_coordinate",
    "partition_is_verified",
    "request_area_for_station",
    "retrieve_era5_land",
    "run_ingestion",
    "validate_partition_outputs",
    "write_manifest",
]
