"""REN production-breakdown ingestion helpers for the v2 raw dataset.

The functions in this module are import-safe: they do not perform network I/O,
create directories, or read local datasets unless explicitly called.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from wind_forecast.manifests import DatasetManifest, manifest_to_json
from wind_forecast.paths import project_root
from wind_forecast.validation.historical import validate_raw_production_data


REN_PRODUCTION_ENDPOINT = (
    "https://servicebus.ren.pt/datahubapi/electricity/"
    "ElectricityProductionBreakdownDaily"
)
ENDPOINT_IDENTIFIER = "REN ElectricityProductionBreakdownDaily"
DEFAULT_CULTURE = "pt-PT"
DEFAULT_TIMEOUT_SECONDS = 20.0
EXPECTED_UNIT = "MW"
EXPECTED_ROWS_PER_COMPLETE_DAY = 96
EXPECTED_FREQUENCY = "15min"
UNRESOLVED_REN_TIMEZONE = "unresolved_ren_source_time"
REN_WALL_CLOCK_TIMEZONE = "Europe/Lisbon"
REN_TIMEZONE_STRATEGY = "ren_wall_clock_interpreted_as_europe_lisbon_for_dst_disambiguation"

TIMESTAMP_COLUMN = "timestamp"
PRODUCTION_COLUMN = "wind_production_mw"
UNIT_COLUMN = "unit"
SOURCE_DATE_COLUMN = "source_date"
RETRIEVAL_TIMESTAMP_COLUMN = "retrieval_timestamp_utc"
ENDPOINT_IDENTIFIER_COLUMN = "endpoint_identifier"
RAW_RESPONSE_SHA256_COLUMN = "raw_response_sha256"

NORMALIZED_COLUMNS = (
    TIMESTAMP_COLUMN,
    PRODUCTION_COLUMN,
    UNIT_COLUMN,
    SOURCE_DATE_COLUMN,
    RETRIEVAL_TIMESTAMP_COLUMN,
    ENDPOINT_IDENTIFIER_COLUMN,
    RAW_RESPONSE_SHA256_COLUMN,
)

RELEVANT_RESPONSE_HEADERS = {
    "cache-control",
    "content-type",
    "date",
    "etag",
    "expires",
    "last-modified",
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
}


class RenIngestionError(ValueError):
    """Raised when a REN response or normalized partition is invalid."""


class RenHTTPError(RenIngestionError):
    """Raised when a REN HTTP request returns an unsuccessful response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class RenResponseCapture:
    """A fetched REN response plus source metadata."""

    requested_date: str
    retrieval_timestamp_utc: str
    payload: Any
    status_code: int | None
    response_headers: Mapping[str, str]
    endpoint_identifier: str = ENDPOINT_IDENTIFIER
    endpoint_url: str = REN_PRODUCTION_ENDPOINT


@dataclass(frozen=True)
class RenPartitionPaths:
    """Deterministic paths for one REN daily partition."""

    raw_response: Path
    normalized_csv: Path
    status_json: Path


@dataclass(frozen=True)
class RenExpectedInterval:
    """One expected REN 15-minute interval under the explicit wall-clock strategy."""

    label: str
    local_naive: pd.Timestamp
    local_aware_text: str
    utc_timestamp: pd.Timestamp


@dataclass(frozen=True)
class RenTimestampComponents:
    """Parsed timestamp identities used for DST-aware validation."""

    local_naive: pd.Series
    identity: pd.Series
    identity_kind: str
    timezone_aware: bool


def parse_source_date(value: str | date) -> date:
    """Validate and normalize a YYYY-MM-DD source date."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("REN source dates must be formatted as YYYY-MM-DD.") from exc


def iter_inclusive_dates(start_date: str | date, end_date: str | date) -> list[date]:
    """Return an inclusive list of dates after validating the range."""
    start = parse_source_date(start_date)
    end = parse_source_date(end_date)
    if start > end:
        raise ValueError("--start-date must be on or before --end-date.")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def utc_timestamp() -> str:
    """Return a second-resolution ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_request_params(source_date: str | date) -> dict[str, str]:
    """Build REN request parameters for one source date."""
    return {"culture": DEFAULT_CULTURE, "date": parse_source_date(source_date).isoformat()}


def ren_partition_paths(output_root: Path, source_date: str | date) -> RenPartitionPaths:
    """Return deterministic partition paths below an output root."""
    day = parse_source_date(source_date).isoformat()
    return RenPartitionPaths(
        raw_response=output_root / "ren" / "raw" / f"date={day}" / "response.json",
        normalized_csv=output_root / "ren" / "normalized" / f"date={day}" / "production_15min.csv",
        status_json=output_root / "ren" / "metadata" / f"date={day}" / "status.json",
    )


def manifest_path(output_root: Path) -> Path:
    """Return the REN production manifest path below an output root."""
    return output_root / "ren" / "manifests" / "ren_production_manifest.json"


def deterministic_json_text(payload: Any) -> str:
    """Serialize JSON-compatible payloads deterministically."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"


def deterministic_json_sha256(payload: Any) -> str:
    """Return the SHA-256 digest of deterministic JSON serialization."""
    return sha256(deterministic_json_text(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 checksum."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_csv_sha256(df: pd.DataFrame) -> str:
    """Return a deterministic CSV checksum for a normalized DataFrame."""
    return sha256(df.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def relevant_headers(headers: Mapping[str, Any] | None) -> dict[str, str]:
    """Return non-sensitive response headers useful for data lineage."""
    if headers is None:
        return {}
    return {
        str(key): str(value)
        for key, value in headers.items()
        if str(key).lower() in RELEVANT_RESPONSE_HEADERS
    }


def fetch_ren_production_day(
    source_date: str | date,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    request_get: Callable[..., Any] | None = None,
) -> RenResponseCapture:
    """Fetch and parse one REN production-breakdown response."""
    requested_date = parse_source_date(source_date).isoformat()
    getter = request_get or requests.get
    try:
        response = getter(
            REN_PRODUCTION_ENDPOINT,
            params=build_request_params(requested_date),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RenHTTPError(f"REN request failed for {requested_date}: {exc}.") from exc

    status_code = getattr(response, "status_code", None)
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RenHTTPError(
            f"REN request failed for {requested_date}: {exc}.",
            status_code=status_code,
        ) from exc
    except Exception as exc:
        raise RenHTTPError(
            f"REN request failed for {requested_date} with HTTP status {status_code}.",
            status_code=status_code,
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RenIngestionError(f"REN response for {requested_date} was not valid JSON.") from exc

    return RenResponseCapture(
        requested_date=requested_date,
        retrieval_timestamp_utc=utc_timestamp(),
        payload=payload,
        status_code=status_code,
        response_headers=relevant_headers(getattr(response, "headers", None)),
    )


def normalize_text(value: object) -> str:
    """Normalize source labels for conservative matching."""
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def extract_categories(payload: Any) -> list[str]:
    """Extract REN x-axis categories without mutating the payload."""
    if not isinstance(payload, Mapping):
        return []
    x_axis = payload.get("xAxis")
    if not isinstance(x_axis, Mapping):
        return []
    categories = x_axis.get("categories")
    if not isinstance(categories, list):
        return []
    return [str(item) for item in categories]


def extract_series(payload: Any) -> list[Mapping[str, Any]]:
    """Extract REN series entries without mutating the payload."""
    if not isinstance(payload, Mapping):
        return []
    series = payload.get("series")
    if not isinstance(series, list):
        return []
    return [item for item in series if isinstance(item, Mapping)]


def find_unambiguous_wind_series(payload: Any) -> Mapping[str, Any]:
    """Return the single unambiguous wind production series."""
    wind_matches = []
    for item in extract_series(payload):
        normalized_name = normalize_text(item.get("name", ""))
        if "eolica" in normalized_name or "wind" in normalized_name:
            wind_matches.append(item)

    if not wind_matches:
        raise RenIngestionError("REN response did not contain an identifiable wind series.")
    if len(wind_matches) > 1:
        names = [str(item.get("name")) for item in wind_matches]
        raise RenIngestionError(f"REN response contained ambiguous wind series: {names}.")
    return wind_matches[0]


def extract_unit(payload: Any) -> str | None:
    """Extract the y-axis unit when present."""
    if not isinstance(payload, Mapping):
        return None
    y_axis = payload.get("yAxis")
    if not isinstance(y_axis, Mapping):
        return None
    title = y_axis.get("title")
    if isinstance(title, Mapping):
        text = title.get("text")
        return str(text).strip() if text is not None else None
    if title is not None:
        return str(title).strip()
    return None


def _category_to_timestamp(source_date: str, category: str) -> pd.Timestamp:
    text = str(category).strip()
    if ":" in text and len(text) <= 8:
        parsed = pd.to_datetime(f"{source_date} {text}", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise RenIngestionError(f"REN response contained an unparseable timestamp category: {category!r}.")
    return pd.Timestamp(parsed).tz_localize(None)


def _expected_source_intervals(source_date: str | date) -> list[RenExpectedInterval]:
    """Return expected physical intervals for one REN local source day."""
    day = parse_source_date(source_date)
    source_tz = ZoneInfo(REN_WALL_CLOCK_TIMEZONE)
    local_start = datetime.combine(day, time.min, tzinfo=source_tz)
    local_end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=source_tz)
    current_utc = local_start.astimezone(timezone.utc)
    end_utc = local_end.astimezone(timezone.utc)

    intervals: list[RenExpectedInterval] = []
    while current_utc < end_utc:
        local = current_utc.astimezone(source_tz)
        local_without_tz = local.replace(tzinfo=None)
        intervals.append(
            RenExpectedInterval(
                label=local.strftime("%H:%M"),
                local_naive=pd.Timestamp(local_without_tz),
                local_aware_text=local.isoformat(timespec="seconds"),
                utc_timestamp=pd.Timestamp(current_utc),
            )
        )
        current_utc += timedelta(minutes=15)
    return intervals


def _expected_source_labels(source_date: str | date) -> list[str]:
    return [item.label for item in _expected_source_intervals(source_date)]


def _is_dst_transition_interval_sequence(intervals: list[RenExpectedInterval]) -> bool:
    labels = [item.label for item in intervals]
    return len(intervals) != EXPECTED_ROWS_PER_COMPLETE_DAY or len(set(labels)) != len(labels)


def _timestamps_for_source_categories(source_date: str, categories: list[str]) -> list[pd.Timestamp | str]:
    """Map REN wall-clock categories to timestamp values without dropping DST repeats."""
    expected_intervals = _expected_source_intervals(source_date)
    expected_labels = [item.label for item in expected_intervals]
    if categories == expected_labels and _is_dst_transition_interval_sequence(expected_intervals):
        return [item.local_aware_text for item in expected_intervals]
    return [_category_to_timestamp(source_date, category) for category in categories]


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


def _timestamp_components(values: pd.Series) -> RenTimestampComponents:
    markers = values.map(_has_timezone_marker)
    if markers.any() and not markers.all():
        raise RenIngestionError("Normalized REN data mixes timezone-aware and timezone-naive timestamps.")
    if markers.any():
        identity = pd.to_datetime(values.astype(str), errors="coerce", utc=True)
        local_naive = identity.dt.tz_convert(REN_WALL_CLOCK_TIMEZONE).dt.tz_localize(None)
        return RenTimestampComponents(
            local_naive=local_naive,
            identity=identity,
            identity_kind="utc",
            timezone_aware=True,
        )

    local_naive = pd.to_datetime(values, errors="coerce")
    return RenTimestampComponents(
        local_naive=local_naive,
        identity=local_naive,
        identity_kind="local_wall_clock",
        timezone_aware=False,
    )


def _duplicate_timestamp_groups(frame: pd.DataFrame, timestamps: pd.Series) -> list[dict[str, Any]]:
    groups = []
    duplicate_values = timestamps[timestamps.duplicated(keep=False)].dropna().drop_duplicates()
    for timestamp in duplicate_values:
        mask = timestamps == timestamp
        values = pd.to_numeric(frame.loc[mask, PRODUCTION_COLUMN], errors="coerce").tolist()
        groups.append(
            {
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "count": int(mask.sum()),
                "values": values,
                "classification": "identical" if len(set(values)) <= 1 else "conflicting",
            }
        )
    return groups


def _comparison_local_timestamps(values: pd.Series) -> pd.Series:
    return _timestamp_components(values).local_naive


def normalize_ren_payload(
    source_date: str | date,
    payload: Any,
    *,
    retrieval_timestamp_utc: str,
    raw_response_sha256: str,
) -> pd.DataFrame:
    """Extract a normalized 15-minute wind-production DataFrame."""
    day = parse_source_date(source_date).isoformat()
    categories = extract_categories(payload)
    wind_series = find_unambiguous_wind_series(payload)
    values = wind_series.get("data")
    if not isinstance(values, list):
        raise RenIngestionError("REN wind series did not contain a list-valued 'data' field.")
    if len(categories) != len(values):
        raise RenIngestionError(
            "REN timestamp category count did not match wind production value count."
        )

    unit = extract_unit(payload)
    if unit is None:
        raise RenIngestionError("REN response did not expose a production unit.")
    records = []
    timestamps = _timestamps_for_source_categories(day, categories)
    for timestamp, value in zip(timestamps, values):
        records.append(
            {
                TIMESTAMP_COLUMN: timestamp,
                PRODUCTION_COLUMN: value,
                UNIT_COLUMN: unit,
                SOURCE_DATE_COLUMN: day,
                RETRIEVAL_TIMESTAMP_COLUMN: retrieval_timestamp_utc,
                ENDPOINT_IDENTIFIER_COLUMN: ENDPOINT_IDENTIFIER,
                RAW_RESPONSE_SHA256_COLUMN: raw_response_sha256,
            }
        )

    df = pd.DataFrame.from_records(records, columns=NORMALIZED_COLUMNS)
    df[PRODUCTION_COLUMN] = pd.to_numeric(df[PRODUCTION_COLUMN], errors="coerce")
    return df


def validate_ren_normalized_day(df: pd.DataFrame, source_date: str | date) -> dict[str, Any]:
    """Validate one normalized REN daily partition.

    Incomplete days are reported in the returned status. Invalid values, date
    leakage, duplicate timestamps, and ambiguous cadence raise errors.
    """
    day = parse_source_date(source_date).isoformat()
    missing_columns = [column for column in NORMALIZED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise RenIngestionError(f"Normalized REN data is missing required columns: {missing_columns}.")

    frame = df.loc[:, NORMALIZED_COLUMNS].copy()
    timestamp_components = _timestamp_components(frame[TIMESTAMP_COLUMN])
    timestamps = timestamp_components.identity
    local_timestamps = timestamp_components.local_naive
    values = pd.to_numeric(frame[PRODUCTION_COLUMN], errors="coerce")

    if timestamps.isna().any():
        raise RenIngestionError("Normalized REN data contains unparseable timestamps.")
    if values.isna().any():
        raise RenIngestionError("Normalized REN data contains missing or non-numeric production values.")
    if not all(math.isfinite(float(value)) for value in values):
        raise RenIngestionError("Normalized REN data contains non-finite production values.")
    if (values < 0).any():
        raise RenIngestionError("Normalized REN data contains negative production values.")
    if timestamps.duplicated().any():
        duplicate_groups = _duplicate_timestamp_groups(frame, timestamps)
        raise RenIngestionError(
            "Normalized REN data contains duplicate timestamp identities: "
            f"{duplicate_groups[:5]}."
        )
    if not timestamps.is_monotonic_increasing:
        raise RenIngestionError("Normalized REN data is not sorted chronologically.")

    if set(frame[UNIT_COLUMN].astype(str)) != {EXPECTED_UNIT}:
        raise RenIngestionError("Normalized REN data has units other than MW.")
    if set(frame[SOURCE_DATE_COLUMN].astype(str)) != {day}:
        raise RenIngestionError("Normalized REN data has inconsistent source_date values.")

    expected_intervals = _expected_source_intervals(day)
    expected_row_count = len(expected_intervals)
    if timestamp_components.timezone_aware:
        expected = pd.DatetimeIndex([item.utc_timestamp for item in expected_intervals])
        expected_labels = [item.label for item in expected_intervals]
        actual_labels = [pd.Timestamp(item).strftime("%H:%M") for item in local_timestamps]
        if actual_labels != expected_labels:
            raise RenIngestionError(
                "Timezone-aware REN timestamps do not match the expected "
                f"{REN_WALL_CLOCK_TIMEZONE} wall-clock interval sequence."
            )
    else:
        expected = pd.DatetimeIndex([item.local_naive for item in expected_intervals])
        local_duplicate_groups = _duplicate_timestamp_groups(frame, local_timestamps)
        if local_duplicate_groups:
            raise RenIngestionError(
                "Normalized REN data contains duplicate local wall-clock timestamps "
                "without timezone disambiguation: "
                f"{local_duplicate_groups[:5]}."
            )
    timestamp_index = pd.DatetimeIndex(timestamps)
    unexpected = timestamp_index.difference(expected)
    if len(unexpected):
        raise RenIngestionError("Normalized REN data contains timestamps outside the daily partition.")

    missing = expected.difference(timestamp_index)
    is_complete = len(frame) == expected_row_count and len(missing) == 0
    if len(frame) > expected_row_count:
        raise RenIngestionError("Normalized REN data has more rows than a complete 15-minute day.")

    reusable_timestamps = timestamps if timestamp_components.timezone_aware else local_timestamps
    reusable_report = validate_raw_production_data(
        pd.DataFrame({TIMESTAMP_COLUMN: reusable_timestamps, PRODUCTION_COLUMN: values}),
        timestamp_column=TIMESTAMP_COLUMN,
        target_column=PRODUCTION_COLUMN,
        dataset_name=f"ren_production_{day}",
    )
    issue_payloads = [issue.to_dict() for issue in reusable_report.issues]
    error_count = sum(1 for item in issue_payloads if item.get("severity") == "error")
    if error_count:
        raise RenIngestionError("Existing raw production validator rejected normalized REN data.")

    warnings = []
    if not is_complete:
        warnings.append(
            f"REN daily partition {day} is incomplete: {len(frame)} rows; "
            f"{len(missing)} expected timestamps missing."
        )

    return {
        "validation_status": "complete" if is_complete else "incomplete",
        "source_date": day,
        "row_count": int(len(frame)),
        "expected_complete_row_count": expected_row_count,
        "missing_timestamp_count": int(len(missing)),
        "missing_timestamps": [item.isoformat() for item in missing],
        "earliest_timestamp": timestamp_index.min().isoformat() if len(timestamp_index) else None,
        "latest_timestamp": timestamp_index.max().isoformat() if len(timestamp_index) else None,
        "temporal_granularity": EXPECTED_FREQUENCY,
        "unit": EXPECTED_UNIT,
        "timestamp_identity": timestamp_components.identity_kind,
        "source_timezone_strategy": REN_TIMEZONE_STRATEGY,
        "source_timezone": REN_WALL_CLOCK_TIMEZONE,
        "dst_transition_day": expected_row_count != EXPECTED_ROWS_PER_COMPLETE_DAY,
        "duplicate_local_wall_clock_timestamp_count": int(local_timestamps.duplicated(keep=False).sum()),
        "warnings": warnings,
        "reusable_validator": {
            "is_valid": reusable_report.passed,
            "issue_count": len(issue_payloads),
            "issues": issue_payloads,
            "stats": dict(reusable_report.stats),
        },
    }


def write_json(path: Path, payload: Any) -> str:
    """Write deterministic JSON and return the file checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(deterministic_json_text(payload), encoding="utf-8", newline="\n")
    return sha256_file(path)


def write_normalized_csv(path: Path, df: pd.DataFrame) -> str:
    """Write a normalized CSV and return the file checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def write_daily_partition(
    output_root: Path,
    capture: RenResponseCapture,
    normalized: pd.DataFrame,
    validation: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write raw, normalized, and status outputs for one daily REN partition."""
    paths = ren_partition_paths(output_root, capture.requested_date)
    existing = [path for path in (paths.raw_response, paths.normalized_csv, paths.status_json) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "REN partition already exists; use --resume to skip verified partitions "
            "or --overwrite to replace it explicitly."
        )

    raw_checksum = write_json(paths.raw_response, capture.payload)
    csv_checksum = write_normalized_csv(paths.normalized_csv, normalized)
    status_payload = {
        "endpoint_identifier": capture.endpoint_identifier,
        "endpoint_url": capture.endpoint_url,
        "http_status": capture.status_code,
        "request_params": build_request_params(capture.requested_date),
        "response_headers": dict(capture.response_headers),
        "retrieval_timestamp_utc": capture.retrieval_timestamp_utc,
        "source_date": capture.requested_date,
        "timezone": UNRESOLVED_REN_TIMEZONE,
        "validation": dict(validation),
        "paths": {
            "raw_response": _manifest_path(paths.raw_response, output_root=output_root),
            "normalized_csv": _manifest_path(paths.normalized_csv, output_root=output_root),
            "status_json": _manifest_path(paths.status_json, output_root=output_root),
        },
        "checksums": {
            "raw_response_sha256": raw_checksum,
            "normalized_csv_sha256": csv_checksum,
        },
        "source_status": "provisional_or_final_unknown",
    }
    status_checksum = write_json(paths.status_json, status_payload)
    return {
        "source_date": capture.requested_date,
        "status": validation["validation_status"],
        "row_count": int(validation["row_count"]),
        "paths": {
            "raw_response": paths.raw_response,
            "normalized_csv": paths.normalized_csv,
            "status_json": paths.status_json,
        },
        "checksums": {
            "raw_response_sha256": raw_checksum,
            "normalized_csv_sha256": csv_checksum,
            "status_json_sha256": status_checksum,
        },
        "warnings": list(validation.get("warnings", [])),
    }


def write_unavailable_status(
    output_root: Path,
    source_date: str | date,
    *,
    message: str,
    retrieval_timestamp_utc: str,
    status_code: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write metadata for an unavailable date without raw or normalized data."""
    day = parse_source_date(source_date).isoformat()
    paths = ren_partition_paths(output_root, day)
    if paths.status_json.exists() and not overwrite:
        raise FileExistsError("REN unavailable status already exists; use --overwrite to replace it.")
    status_payload = {
        "endpoint_identifier": ENDPOINT_IDENTIFIER,
        "endpoint_url": REN_PRODUCTION_ENDPOINT,
        "http_status": status_code,
        "request_params": build_request_params(day),
        "retrieval_timestamp_utc": retrieval_timestamp_utc,
        "source_date": day,
        "timezone": UNRESOLVED_REN_TIMEZONE,
        "validation": {
            "validation_status": "unavailable",
            "source_date": day,
            "row_count": 0,
            "warnings": [message],
        },
        "paths": {
            "raw_response": None,
            "normalized_csv": None,
            "status_json": _manifest_path(paths.status_json, output_root=output_root),
        },
        "checksums": {},
        "source_status": "provisional_or_final_unknown",
    }
    checksum = write_json(paths.status_json, status_payload)
    return {
        "source_date": day,
        "status": "unavailable",
        "row_count": 0,
        "paths": {"status_json": paths.status_json},
        "checksums": {"status_json_sha256": checksum},
        "warnings": [message],
    }


def partition_is_verified(output_root: Path, source_date: str | date) -> bool:
    """Return True when a successful daily partition exists and checksums match."""
    paths = ren_partition_paths(output_root, source_date)
    if not (paths.raw_response.is_file() and paths.normalized_csv.is_file() and paths.status_json.is_file()):
        return False
    try:
        status = json.loads(paths.status_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    validation = status.get("validation")
    checksums = status.get("checksums")
    if not isinstance(validation, Mapping) or not isinstance(checksums, Mapping):
        return False
    if validation.get("validation_status") not in {"complete", "incomplete"}:
        return False
    return (
        checksums.get("raw_response_sha256") == sha256_file(paths.raw_response)
        and checksums.get("normalized_csv_sha256") == sha256_file(paths.normalized_csv)
    )


def partition_is_unavailable_status_only(output_root: Path, source_date: str | date) -> bool:
    """Return True when a partition has only strict unavailable status metadata."""
    day = parse_source_date(source_date).isoformat()
    paths = ren_partition_paths(output_root, day)
    if not paths.status_json.is_file():
        return False
    if paths.raw_response.exists() or paths.normalized_csv.exists():
        return False

    try:
        status = json.loads(paths.status_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(status, Mapping):
        return False
    if status.get("source_date") != day:
        return False

    validation = status.get("validation")
    metadata_paths = status.get("paths")
    checksums = status.get("checksums")
    if not isinstance(validation, Mapping):
        return False
    if not isinstance(metadata_paths, Mapping) or not isinstance(checksums, Mapping):
        return False
    if validation.get("validation_status") != "unavailable":
        return False

    return (
        _metadata_value_is_absent(metadata_paths, "raw_response")
        and _metadata_value_is_absent(metadata_paths, "normalized_csv")
        and _metadata_value_is_absent(checksums, "raw_response_sha256")
        and _metadata_value_is_absent(checksums, "normalized_csv_sha256")
    )


def _metadata_value_is_absent(metadata: Mapping[str, Any], key: str) -> bool:
    value = metadata.get(key)
    return key not in metadata or value is None or value == ""


def load_partition_summary(output_root: Path, source_date: str | date) -> dict[str, Any]:
    """Load a verified daily partition summary from status metadata."""
    paths = ren_partition_paths(output_root, source_date)
    status = json.loads(paths.status_json.read_text(encoding="utf-8"))
    validation = status["validation"]
    checksums = dict(status.get("checksums") or {})
    checksums["status_json_sha256"] = sha256_file(paths.status_json)
    return {
        "source_date": parse_source_date(source_date).isoformat(),
        "status": validation["validation_status"],
        "row_count": int(validation["row_count"]),
        "paths": {
            "raw_response": paths.raw_response,
            "normalized_csv": paths.normalized_csv,
            "status_json": paths.status_json,
        },
        "checksums": checksums,
        "warnings": list(validation.get("warnings") or []),
    }


def load_all_partition_summaries(output_root: Path) -> list[dict[str, Any]]:
    """Load all verified daily partition summaries below an output root."""
    metadata_root = output_root / "ren" / "metadata"
    if not metadata_root.exists():
        return []

    summaries = []
    for status_path in sorted(metadata_root.glob("date=*/status.json")):
        date_part = status_path.parent.name
        if not date_part.startswith("date="):
            continue
        source_date = date_part.removeprefix("date=")
        if partition_is_verified(output_root, source_date):
            summaries.append(load_partition_summary(output_root, source_date))
    return summaries


def compare_normalized_with_v1(
    normalized: pd.DataFrame,
    v1_csv: Path,
    *,
    timestamp_column: str = "Data e Hora",
    production_column: str = "Eólica",
) -> dict[str, Any]:
    """Compare normalized REN rows with the frozen v1 production CSV."""
    local = pd.read_csv(v1_csv, sep=";", skiprows=2, na_values=-990)
    if timestamp_column not in local.columns or production_column not in local.columns:
        raise RenIngestionError(
            f"V1 CSV must contain '{timestamp_column}' and '{production_column}' columns."
        )
    local_frame = local[[timestamp_column, production_column]].copy()
    local_frame[timestamp_column] = pd.to_datetime(local_frame[timestamp_column], errors="coerce")
    local_frame[production_column] = pd.to_numeric(local_frame[production_column], errors="coerce")
    local_frame = local_frame.rename(
        columns={timestamp_column: TIMESTAMP_COLUMN, production_column: "v1_wind_production_mw"}
    )
    local_frame["_timestamp_occurrence"] = local_frame.groupby(TIMESTAMP_COLUMN).cumcount()

    ren_frame = normalized[[TIMESTAMP_COLUMN, PRODUCTION_COLUMN]].copy()
    ren_frame[TIMESTAMP_COLUMN] = _comparison_local_timestamps(ren_frame[TIMESTAMP_COLUMN])
    ren_frame["_timestamp_occurrence"] = ren_frame.groupby(TIMESTAMP_COLUMN).cumcount()
    aligned = pd.merge(
        local_frame,
        ren_frame,
        on=[TIMESTAMP_COLUMN, "_timestamp_occurrence"],
        how="inner",
    ).dropna()
    aligned = aligned.sort_values([TIMESTAMP_COLUMN, "_timestamp_occurrence"])
    result: dict[str, Any] = {
        "aligned_timestamp_count": int(len(aligned)),
        "exact_match_count": None,
        "mean_absolute_error_mw": None,
        "maximum_absolute_difference_mw": None,
        "pearson_correlation": None,
        "possible_revision_evidence": False,
    }
    if aligned.empty:
        return result

    diff = aligned[PRODUCTION_COLUMN] - aligned["v1_wind_production_mw"]
    exact_match_count = int((diff == 0).sum())
    max_abs = float(diff.abs().max())
    result.update(
        {
            "exact_match_count": exact_match_count,
            "mean_absolute_error_mw": float(diff.abs().mean()),
            "maximum_absolute_difference_mw": max_abs,
            "pearson_correlation": (
                float(aligned["v1_wind_production_mw"].corr(aligned[PRODUCTION_COLUMN]))
                if len(aligned) > 1
                else None
            ),
            "possible_revision_evidence": max_abs > 0,
        }
    )
    return result


def build_manifest(
    *,
    output_root: Path,
    requested_start_date: str,
    requested_end_date: str,
    retrieval_timestamp_utc: str,
    daily_results: list[Mapping[str, Any]],
    requested_ranges: list[Mapping[str, Any]] | None = None,
    compare_v1_csv: Path | None = None,
    comparison_results: Mapping[str, Any] | None = None,
) -> DatasetManifest:
    """Build a deterministic REN v2 production manifest."""
    actual_results = [item for item in daily_results if item.get("status") in {"complete", "incomplete"}]
    actual_dates = [str(item["source_date"]) for item in actual_results]
    incomplete_dates = [str(item["source_date"]) for item in daily_results if item.get("status") == "incomplete"]
    unavailable_dates = [str(item["source_date"]) for item in daily_results if item.get("status") == "unavailable"]
    skipped_dates = [str(item["source_date"]) for item in daily_results if item.get("skipped_existing")]
    warnings = []
    for item in daily_results:
        warnings.extend(str(warning) for warning in item.get("warnings", []))
    warnings.extend(
        [
            "REN source timezone semantics are unresolved.",
            "REN attribution and license are unknown.",
            "REN provisional/final source status is unknown.",
            "Manifest covers only the explicitly requested date range; it does not claim full historical coverage.",
        ]
    )

    path_checksum_map: dict[str, str] = {}
    raw_paths = []
    normalized_paths = []
    metadata_paths = []
    for item in daily_results:
        paths = item.get("paths") or {}
        checksums = item.get("checksums") or {}
        for key, checksum_key in (
            ("raw_response", "raw_response_sha256"),
            ("normalized_csv", "normalized_csv_sha256"),
            ("status_json", "status_json_sha256"),
        ):
            path = paths.get(key)
            checksum = checksums.get(checksum_key)
            if path is None or checksum is None:
                continue
            manifest_ready_path = _manifest_path(Path(path), output_root=output_root)
            path_checksum_map[manifest_ready_path] = str(checksum)
            if key == "raw_response":
                raw_paths.append(manifest_ready_path)
            elif key == "normalized_csv":
                normalized_paths.append(manifest_ready_path)
            elif key == "status_json":
                metadata_paths.append(manifest_ready_path)

    row_counts = {str(item["source_date"]): int(item.get("row_count", 0)) for item in daily_results}
    return DatasetManifest(
        dataset_version="v2",
        dataset_role="raw_production",
        provider="REN",
        source_identifier=ENDPOINT_IDENTIFIER,
        source_endpoint=REN_PRODUCTION_ENDPOINT,
        retrieval_timestamp=retrieval_timestamp_utc,
        coverage_start=min(actual_dates) if actual_dates else None,
        coverage_end=max(actual_dates) if actual_dates else None,
        temporal_granularity=EXPECTED_FREQUENCY,
        units={PRODUCTION_COLUMN: EXPECTED_UNIT},
        timezone=UNRESOLVED_REN_TIMEZONE,
        raw_file_paths=tuple(sorted(raw_paths)),
        sha256_checksums=dict(sorted(path_checksum_map.items())),
        row_count=sum(row_counts.values()),
        column_count=len(NORMALIZED_COLUMNS),
        known_warnings=tuple(sorted(set(warnings))),
        license="unknown",
        attribution="unknown",
        status="provisional_or_final_status_unknown",
        extra_metadata={
            "requested_ranges": requested_ranges
            or [
                {
                    "start_date": requested_start_date,
                    "end_date": requested_end_date,
                    "inclusive": True,
                }
            ],
            "actual_dates": actual_dates,
            "incomplete_dates": incomplete_dates,
            "unavailable_dates": unavailable_dates,
            "skipped_existing_dates": skipped_dates,
            "row_counts_by_date": row_counts,
            "normalized_file_paths": sorted(normalized_paths),
            "metadata_file_paths": sorted(metadata_paths),
            "column_contract": {
                "required_columns": list(NORMALIZED_COLUMNS),
                "timestamp_column": TIMESTAMP_COLUMN,
                "production_column": PRODUCTION_COLUMN,
                "unit": EXPECTED_UNIT,
                "complete_day_row_count": EXPECTED_ROWS_PER_COMPLETE_DAY,
                "granularity": EXPECTED_FREQUENCY,
            },
            "request_contract": {
                "culture": DEFAULT_CULTURE,
                "one_request_per_date": True,
                "request_params": ["culture", "date"],
            },
            "comparison_with_v1": {
                "v1_csv": _manifest_path(compare_v1_csv, output_root=output_root)
                if compare_v1_csv is not None
                else None,
                "results_by_date": dict(comparison_results or {}),
            },
            "source_status": "provisional_or_final_unknown",
            "timezone_status": "unresolved_ren_source_time",
            "historical_coverage_claim": "requested_range_only",
        },
    )


def write_manifest(output_root: Path, manifest: DatasetManifest) -> str:
    """Write the REN production manifest and return its checksum."""
    path = manifest_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_to_json(manifest), encoding="utf-8", newline="\n")
    return sha256_file(path)


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


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "ENDPOINT_IDENTIFIER",
    "EXPECTED_FREQUENCY",
    "EXPECTED_ROWS_PER_COMPLETE_DAY",
    "EXPECTED_UNIT",
    "NORMALIZED_COLUMNS",
    "PRODUCTION_COLUMN",
    "REN_PRODUCTION_ENDPOINT",
    "RETRIEVAL_TIMESTAMP_COLUMN",
    "RenHTTPError",
    "RenIngestionError",
    "RenPartitionPaths",
    "RenResponseCapture",
    "SOURCE_DATE_COLUMN",
    "TIMESTAMP_COLUMN",
    "UNRESOLVED_REN_TIMEZONE",
    "build_manifest",
    "build_request_params",
    "compare_normalized_with_v1",
    "dataframe_csv_sha256",
    "deterministic_json_sha256",
    "deterministic_json_text",
    "fetch_ren_production_day",
    "find_unambiguous_wind_series",
    "iter_inclusive_dates",
    "load_all_partition_summaries",
    "manifest_path",
    "normalize_ren_payload",
    "partition_is_unavailable_status_only",
    "partition_is_verified",
    "ren_partition_paths",
    "validate_ren_normalized_day",
    "write_daily_partition",
    "write_manifest",
    "write_unavailable_status",
]
