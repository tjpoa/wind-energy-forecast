"""Run the ERA5-Land one-point technical pilot for one station and week."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


DATASET_ID = "reanalysis-era5-land"
DEFAULT_STATION_ID = "1200551"
DEFAULT_LATITUDE = 41.648875
DEFAULT_LONGITUDE = -8.804606
DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2023-07-07"
DEFAULT_OUTPUT_DIR = Path("data/pilot/era5_land")
DEFAULT_CALM_THRESHOLD_M_S = 0.5
REQUEST_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
REQUEST_TIMES = [f"{hour:02d}:00" for hour in range(24)]
OFFICIAL_URLS = {
    "dataset": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview",
    "api": "https://cds.climate.copernicus.eu/how-to-api",
}
VARIABLE_ALIASES = {
    "temperature_2m_k": ("t2m", "2m_temperature"),
    "u10_m_s": ("u10", "10m_u_component_of_wind"),
    "v10_m_s": ("v10", "10m_v_component_of_wind"),
}
DEPENDENCIES_TO_RECORD = ("cdsapi", "xarray", "netCDF4", "numpy", "pandas")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments without touching CDS or the filesystem."""
    parser = argparse.ArgumentParser(
        description="Run the ERA5-Land one-point technical pilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--station-id", default=DEFAULT_STATION_ID)
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calm-threshold", type=float, default=DEFAULT_CALM_THRESHOLD_M_S)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an existing raw NetCDF file instead of making a CDS request.",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        help="Existing raw NetCDF path to validate when --skip-download is set.",
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso_date(value: str, argument_name: str) -> date:
    """Parse and validate a YYYY-MM-DD date."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{argument_name} must be formatted as YYYY-MM-DD.") from exc


def requested_days(start: date, end: date) -> list[str]:
    """Return one inclusive same-month day list for the CDS request."""
    if end < start:
        raise ValueError("--end-date must be on or after --start-date.")
    if start.year != end.year or start.month != end.month:
        raise ValueError("This one-request pilot only supports one calendar month per run.")
    day_count = (end - start).days + 1
    if day_count > 7:
        raise ValueError("This one-point pilot supports at most seven inclusive days per run.")
    return [f"{start.day + offset:02d}" for offset in range(day_count)]


def output_stem(station_id: str, start: date, end: date) -> str:
    """Build the approved output filename stem."""
    return f"era5_land_one_point_{station_id}_{start.isoformat()}_{end.isoformat()}"


def output_paths(output_dir: Path, station_id: str, start: date, end: date) -> dict[str, Path]:
    """Return all approved pilot output paths."""
    stem = output_stem(station_id, start, end)
    return {
        "raw": output_dir / f"{stem}_raw.nc",
        "hourly_csv": output_dir / f"{stem}_hourly.csv",
        "daily_csv": output_dir / f"{stem}_daily_utc.csv",
        "metadata_json": output_dir / f"{stem}_metadata.json",
        "validation_json": output_dir / f"{stem}_validation.json",
    }


def nearest_era5_land_grid_coordinate(value: float) -> float:
    """Round a station coordinate to the nearest ERA5-Land 0.1-degree grid line."""
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def request_area_for_station(latitude: float, longitude: float) -> list[float]:
    """Build a single-grid-cell CDS area from station coordinates."""
    grid_latitude = nearest_era5_land_grid_coordinate(latitude)
    grid_longitude = nearest_era5_land_grid_coordinate(longitude)
    return [grid_latitude, grid_longitude, grid_latitude, grid_longitude]


def build_cds_request(start: date, end: date, *, latitude: float, longitude: float) -> dict[str, Any]:
    """Build exactly one ERA5-Land CDS request for the requested station and dates."""
    days = requested_days(start, end)
    return {
        "variable": REQUEST_VARIABLES,
        "year": f"{start.year:04d}",
        "month": f"{start.month:02d}",
        "day": days,
        "time": REQUEST_TIMES,
        "area": request_area_for_station(latitude, longitude),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def retrieve_era5_land(dataset: str, request: dict[str, Any], target: Path) -> None:
    """Run the single approved CDS retrieval."""
    import cdsapi

    target.parent.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve(dataset, request, target)


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> str:
    """Write JSON and return the saved-file SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return sha256_file(path)


def dependency_versions() -> dict[str, str | None]:
    """Return installed dependency versions without importing the packages."""
    versions: dict[str, str | None] = {}
    for package in DEPENDENCIES_TO_RECORD:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def credential_presence() -> dict[str, bool]:
    """Record CDS credential presence booleans only."""
    return {
        "cdsapi_url_env_present": bool(os.environ.get("CDSAPI_URL")),
        "cdsapi_key_env_present": bool(os.environ.get("CDSAPI_KEY")),
        "cdsapirc_present": (Path.home() / ".cdsapirc").exists(),
    }


def coordinate_name(dataset: Any, candidates: tuple[str, ...], description: str) -> str:
    """Find a coordinate name in an xarray dataset."""
    for candidate in candidates:
        if candidate in dataset.coords or candidate in dataset.dims:
            return candidate
    raise ValueError(f"NetCDF does not contain a {description} coordinate.")


def time_coordinate_name(dataset: Any) -> str:
    """Find the time coordinate used by an ERA5 NetCDF file."""
    for candidate in ("valid_time", "time"):
        if candidate in dataset.coords or candidate in dataset.dims:
            return candidate
    for name, coord in dataset.coords.items():
        if "datetime64" in str(coord.dtype):
            return str(name)
    raise ValueError("NetCDF does not contain a recognizable time coordinate.")


def source_variable(dataset: Any, aliases: tuple[str, ...], output_name: str) -> str:
    """Find the ERA5 variable name behind one output field."""
    for alias in aliases:
        if alias in dataset.data_vars:
            return alias
    available = ", ".join(str(name) for name in dataset.data_vars)
    raise ValueError(f"NetCDF is missing {output_name}; available variables: {available}")


def longitude_for_selection(dataset_longitudes: Any, station_longitude: float) -> float:
    """Adjust negative longitudes for 0..360 datasets when necessary."""
    values = [float(value) for value in dataset_longitudes.values.ravel()]
    if not values:
        raise ValueError("Longitude coordinate is empty.")
    if min(values) >= 0.0 and station_longitude < 0.0:
        return station_longitude % 360.0
    return station_longitude


def scalar_coordinate(point_dataset: Any, name: str, description: str) -> float:
    """Return a scalar selected coordinate, failing clearly otherwise."""
    coord = point_dataset[name]
    if coord.size != 1:
        raise ValueError(f"Expected exactly one {description} after extraction; found {coord.size}.")
    return float(coord.values.reshape(-1)[0])


def validate_raw_coordinate_size(dataset: Any, name: str, description: str) -> None:
    """Fail unless the raw NetCDF coordinate already contains exactly one point."""
    coord = dataset[name]
    if coord.size != 1:
        raise ValueError(
            f"Expected raw NetCDF to contain exactly one {description} coordinate before extraction; "
            f"found {coord.size}."
        )


def series_from_data_array(data_array: Any, time_name: str, output_name: str) -> Any:
    """Return a one-dimensional data array aligned to the time coordinate."""
    squeezed = data_array.squeeze(drop=True)
    extra_dims = [dim for dim in squeezed.dims if dim != time_name]
    if extra_dims:
        raise ValueError(f"{output_name} has unexpected dimensions after point extraction: {extra_dims}.")
    if time_name not in squeezed.dims:
        raise ValueError(f"{output_name} is not aligned to the {time_name} coordinate.")
    return squeezed.transpose(time_name).values


def load_hourly_frame(
    raw_path: Path,
    *,
    station_id: str,
    station_latitude: float,
    station_longitude: float,
    calm_threshold: float,
) -> tuple[Any, dict[str, Any]]:
    """Open the NetCDF, extract one nearest point, and build the hourly frame."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(raw_path, engine="netcdf4") as dataset:
        latitude_name = coordinate_name(dataset, ("latitude", "lat"), "latitude")
        longitude_name = coordinate_name(dataset, ("longitude", "lon"), "longitude")
        time_name = time_coordinate_name(dataset)
        validate_raw_coordinate_size(dataset, latitude_name, "latitude")
        validate_raw_coordinate_size(dataset, longitude_name, "longitude")
        selection_longitude = longitude_for_selection(dataset[longitude_name], station_longitude)
        point = dataset.sel(
            {
                latitude_name: station_latitude,
                longitude_name: selection_longitude,
            },
            method="nearest",
        ).load()

    grid_latitude = scalar_coordinate(point, latitude_name, "latitude")
    grid_longitude = scalar_coordinate(point, longitude_name, "longitude")
    source_variables = {
        output_name: source_variable(point, aliases, output_name)
        for output_name, aliases in VARIABLE_ALIASES.items()
    }
    timestamps = pd.to_datetime(point[time_name].values, utc=True)
    if timestamps.empty:
        raise ValueError("Extracted ERA5-Land point contains no hourly timestamps.")

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
            "grid_latitude": grid_latitude,
            "grid_longitude": grid_longitude,
            "station_id": station_id,
            "station_latitude": station_latitude,
            "station_longitude": station_longitude,
            "temperature_2m_k": temperature_k,
            "temperature_2m_c": temperature_k - 273.15,
            "u10_m_s": u10,
            "v10_m_s": v10,
            "wind_speed_m_s": wind_speed,
            "wind_direction_deg_from": direction,
            "is_calm_or_near_calm": calm,
        }
    ).sort_values("timestamp_utc")

    extraction_metadata = {
        "coordinate_names": {
            "latitude": latitude_name,
            "longitude": longitude_name,
            "time": time_name,
        },
        "selected_grid_coordinate": {
            "latitude": grid_latitude,
            "longitude": grid_longitude,
        },
        "source_variables": source_variables,
        "netcdf_dimensions_after_point_extraction": {str(key): int(value) for key, value in point.sizes.items()},
        "netcdf_variable_units": {
            str(name): point[name].attrs.get("units")
            for name in source_variables.values()
            if name in point
        },
    }
    return frame, extraction_metadata


def expected_hourly_index(start: date, end: date) -> Any:
    """Return the expected hourly UTC index for the inclusive date range."""
    import pandas as pd

    start_timestamp = pd.Timestamp(start.isoformat(), tz="UTC")
    end_timestamp = pd.Timestamp(end.isoformat(), tz="UTC") + pd.Timedelta(hours=23)
    return pd.date_range(start_timestamp, end_timestamp, freq="h")


def daily_aggregates(hourly: Any, start: date, end: date, calm_threshold: float) -> Any:
    """Aggregate hourly ERA5-Land values to daily UTC rows."""
    import numpy as np
    import pandas as pd

    frame = hourly.copy()
    frame["_timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["_date"] = frame["_timestamp"].dt.strftime("%Y-%m-%d")
    expected_dates = pd.date_range(start.isoformat(), end.isoformat(), freq="D").strftime("%Y-%m-%d")
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
    return pd.DataFrame(rows)


def validation_report(
    hourly: Any,
    daily: Any,
    *,
    start: date,
    end: date,
    output_path_strings: dict[str, str],
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Build validation details for the generated ERA5-Land pilot outputs."""
    import numpy as np
    import pandas as pd

    issues = []
    timestamps = pd.to_datetime(hourly["timestamp_utc"], utc=True)
    expected = expected_hourly_index(start, end)
    actual_index = pd.DatetimeIndex(timestamps)
    duplicate_count = int(actual_index.duplicated().sum())
    missing = expected.difference(actual_index)
    unexpected = actual_index.difference(expected)
    if duplicate_count:
        issues.append(f"Found {duplicate_count} duplicate hourly timestamps.")
    if len(missing):
        issues.append(f"Missing {len(missing)} expected hourly timestamps.")
    if len(unexpected):
        issues.append(f"Found {len(unexpected)} timestamps outside the requested range.")
    if len(hourly) != len(expected):
        issues.append(f"Expected {len(expected)} hourly rows but found {len(hourly)}.")
    if len(daily) != (end - start).days + 1:
        issues.append(f"Expected {(end - start).days + 1} daily rows but found {len(daily)}.")

    nullable_columns = {"wind_direction_deg_from"}
    null_counts = {column: int(hourly[column].isna().sum()) for column in hourly.columns}
    for column, null_count in null_counts.items():
        if null_count and column not in nullable_columns:
            issues.append(f"Column {column} has {null_count} null values.")

    numeric_columns = [
        "grid_latitude",
        "grid_longitude",
        "station_latitude",
        "station_longitude",
        "temperature_2m_k",
        "temperature_2m_c",
        "u10_m_s",
        "v10_m_s",
        "wind_speed_m_s",
        "wind_direction_deg_from",
    ]
    finite_checks = {}
    min_max_stats = {}
    for column in numeric_columns:
        series = pd.to_numeric(hourly[column], errors="coerce")
        finite_or_null = bool(np.isfinite(series.dropna()).all())
        finite_checks[column] = finite_or_null
        if not finite_or_null:
            issues.append(f"Column {column} contains non-finite numeric values.")
        min_max_stats[column] = {
            "min": float(series.min()) if not series.dropna().empty else None,
            "max": float(series.max()) if not series.dropna().empty else None,
        }

    return {
        "generated_at_utc": utc_timestamp(),
        "passed": not issues,
        "issues": issues,
        "timestamp_coverage": {
            "expected_start_utc": expected[0].isoformat().replace("+00:00", "Z"),
            "expected_end_utc": expected[-1].isoformat().replace("+00:00", "Z"),
            "actual_start_utc": actual_index.min().isoformat().replace("+00:00", "Z") if len(actual_index) else None,
            "actual_end_utc": actual_index.max().isoformat().replace("+00:00", "Z") if len(actual_index) else None,
            "missing_timestamp_count": int(len(missing)),
            "unexpected_timestamp_count": int(len(unexpected)),
        },
        "duplicates": {
            "duplicate_timestamp_count": duplicate_count,
        },
        "expected_hours": {
            "expected_hourly_rows": int(len(expected)),
            "actual_hourly_rows": int(len(hourly)),
            "expected_daily_rows": int((end - start).days + 1),
            "actual_daily_rows": int(len(daily)),
        },
        "null_counts": null_counts,
        "finite_checks": finite_checks,
        "min_max_stats": min_max_stats,
        "units": output_units(),
        "output_paths": output_path_strings,
        "checksums": checksums,
    }


def output_units() -> dict[str, str]:
    """Return units used by the derived pilot outputs."""
    return {
        "timestamp_utc": "UTC",
        "grid_latitude": "degrees_north",
        "grid_longitude": "degrees_east",
        "station_latitude": "degrees_north",
        "station_longitude": "degrees_east",
        "temperature_2m_k": "K",
        "temperature_2m_c": "degree_Celsius",
        "u10_m_s": "m s-1",
        "v10_m_s": "m s-1",
        "wind_speed_m_s": "m s-1",
        "wind_direction_deg_from": "degree_from",
        "is_calm_or_near_calm": "boolean",
    }


def metadata_payload(
    *,
    station_id: str,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    request: dict[str, Any],
    retrieval_started_at: str | None,
    retrieval_finished_at: str | None,
    service_status: str,
    extraction_metadata: dict[str, Any],
    output_path_strings: dict[str, str],
    checksums: dict[str, str],
    calm_threshold: float,
) -> dict[str, Any]:
    """Build metadata for the ERA5-Land pilot."""
    return {
        "generated_at_utc": utc_timestamp(),
        "source_dataset": DATASET_ID,
        "official_urls": OFFICIAL_URLS,
        "request_params": request,
        "station": {
            "station_id": station_id,
            "latitude": latitude,
            "longitude": longitude,
        },
        "requested_period": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "UTC",
        },
        "selected_grid_coordinate": extraction_metadata["selected_grid_coordinate"],
        "retrieval_timestamps": {
            "started_at_utc": retrieval_started_at,
            "finished_at_utc": retrieval_finished_at,
        },
        "units": output_units(),
        "source_netcdf_units": extraction_metadata["netcdf_variable_units"],
        "service_status": service_status,
        "dependency_versions": dependency_versions(),
        "credential_presence": credential_presence(),
        "calm_or_near_calm_threshold_m_s": calm_threshold,
        "source_variables": extraction_metadata["source_variables"],
        "coordinate_names": extraction_metadata["coordinate_names"],
        "netcdf_dimensions_after_point_extraction": extraction_metadata["netcdf_dimensions_after_point_extraction"],
        "output_paths": output_path_strings,
        "checksums": checksums,
        "notes": [
            "Generated under Phase 2 Step 2A.8 ERA5-Land one-point technical pilot.",
            "The pilot does not select a final v2 weather source or validate current model/scaler compatibility.",
        ],
    }


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Run the one-request ERA5-Land pilot or validate an existing raw file."""
    if args.raw_path is not None and not args.skip_download:
        raise ValueError("--raw-path is only supported with --skip-download.")

    start = parse_iso_date(args.start_date, "--start-date")
    end = parse_iso_date(args.end_date, "--end-date")
    request = build_cds_request(start, end, latitude=args.latitude, longitude=args.longitude)
    paths = output_paths(args.output_dir, args.station_id, start, end)
    raw_path = args.raw_path if args.raw_path is not None else paths["raw"]

    retrieval_started_at = None
    retrieval_finished_at = None
    if args.skip_download:
        if not raw_path.exists():
            raise FileNotFoundError(f"--skip-download requested but raw NetCDF was not found: {raw_path}")
        service_status = "not_contacted_skip_download"
    else:
        retrieval_started_at = utc_timestamp()
        retrieve_era5_land(DATASET_ID, request, paths["raw"])
        retrieval_finished_at = utc_timestamp()
        raw_path = paths["raw"]
        service_status = "cds_retrieve_completed"

    hourly, extraction_metadata = load_hourly_frame(
        raw_path,
        station_id=args.station_id,
        station_latitude=args.latitude,
        station_longitude=args.longitude,
        calm_threshold=args.calm_threshold,
    )
    daily = daily_aggregates(hourly, start, end, args.calm_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(paths["hourly_csv"], index=False)
    daily.to_csv(paths["daily_csv"], index=False)

    output_path_strings = {name: str(path) for name, path in paths.items()}
    if raw_path != paths["raw"]:
        output_path_strings["raw"] = str(raw_path)

    checksums = {
        "raw": sha256_file(raw_path),
        "hourly_csv": sha256_file(paths["hourly_csv"]),
        "daily_csv": sha256_file(paths["daily_csv"]),
    }
    validation = validation_report(
        hourly,
        daily,
        start=start,
        end=end,
        output_path_strings=output_path_strings,
        checksums=checksums.copy(),
    )
    checksums["validation_json"] = write_json(paths["validation_json"], validation)

    metadata = metadata_payload(
        station_id=args.station_id,
        latitude=args.latitude,
        longitude=args.longitude,
        start=start,
        end=end,
        request=request,
        retrieval_started_at=retrieval_started_at,
        retrieval_finished_at=retrieval_finished_at,
        service_status=service_status,
        extraction_metadata=extraction_metadata,
        output_path_strings=output_path_strings,
        checksums=checksums.copy(),
        calm_threshold=args.calm_threshold,
    )
    checksums["metadata_json"] = write_json(paths["metadata_json"], metadata)

    return {
        "passed": validation["passed"],
        "issues": validation["issues"],
        "dataset": DATASET_ID,
        "request": request,
        "output_paths": output_path_strings,
        "checksums": checksums,
        "service_status": service_status,
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = run_pilot(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
