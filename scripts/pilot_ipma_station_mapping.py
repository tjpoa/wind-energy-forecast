"""Map v1 weather station identifiers against current IPMA metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


IPMA_STATIONS_URL = "https://api.ipma.pt/open-data/observation/meteorology/stations/stations.json"
IPMA_SURFACE_URL = "https://api.ipma.pt/open-data/observation/meteorology/stations/obs-surface.geojson"
DEFAULT_WEATHER_MATRIX = Path("data/raw/IntensidadeMediaVento10m.csv")
DEFAULT_OUTPUT_DIR = Path("data/pilot/ipma")
DEFAULT_TIMEOUT_SECONDS = 20.0
DATE_COLUMNS = {"ANO", "MES", "DIA"}
EXPECTED_V1_STATION_COUNT = 18
COORDINATE_CORROBORATION_TOLERANCE_DEGREES = 0.01


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Probe current IPMA station metadata and map v1 weather matrix station IDs."
    )
    parser.add_argument("--weather-matrix", type=Path, default=DEFAULT_WEATHER_MATRIX)
    parser.add_argument("--wind-speed-matrix", type=Path)
    parser.add_argument("--wind-direction-matrix", type=Path)
    parser.add_argument("--temperature-matrix", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def fetch_json(url: str, timeout: float) -> Any:
    """Fetch one official IPMA metadata resource."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError(f"IPMA response from {url} was not valid JSON.") from exc


def extract_station_ids(matrix_path: Path) -> list[str]:
    """Extract station ID columns from a v1 weather matrix header."""
    columns = pd.read_csv(matrix_path, sep=";", nrows=0).columns.tolist()
    return [str(column).strip() for column in columns if str(column).strip() not in DATE_COLUMNS]


def validate_optional_matrices(reference_ids: list[str], paths: list[Path]) -> list[str]:
    """Return warnings for optional matrices whose station IDs differ."""
    warnings = []
    for path in paths:
        ids = extract_station_ids(path)
        if ids != reference_ids:
            warnings.append(f"Station IDs in {path} do not match {paths[0] if paths else 'reference matrix'}.")
    return warnings


def normalize_identifier(value: object) -> str | None:
    """Normalize identifiers while preserving leading-zero awareness."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def numeric_equivalent(value: str) -> str | None:
    """Return a lossless numeric string equivalent when safe."""
    if "." in value:
        left, right = value.split(".", maxsplit=1)
        if not right or set(right) != {"0"}:
            return None
        value = left
    if not value.isdigit():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if str(parsed) == value or value == value.lstrip("0"):
        return str(parsed)
    return None


def identifier_values(feature: dict[str, Any]) -> dict[str, str]:
    """Collect plausible identifier fields from an IPMA feature."""
    values: dict[str, str] = {}
    feature_id = normalize_identifier(feature.get("id"))
    if feature_id is not None:
        values["feature.id"] = feature_id

    properties = feature.get("properties", {})
    if isinstance(properties, dict):
        for key, value in properties.items():
            lowered = str(key).casefold()
            if "id" in lowered or "wmo" in lowered:
                normalized = normalize_identifier(value)
                if normalized is not None:
                    values[f"properties.{key}"] = normalized
    return values


def station_name(feature: dict[str, Any]) -> str | None:
    """Extract the best available station name."""
    properties = feature.get("properties", {})
    if not isinstance(properties, dict):
        return None
    for key in ["localEstacao", "nome", "name"]:
        if key in properties and properties[key] is not None:
            return str(properties[key])
    return None


def altitude(feature: dict[str, Any]) -> object | None:
    """Extract altitude-like metadata when available."""
    properties = feature.get("properties", {})
    if not isinstance(properties, dict):
        return None
    for key, value in properties.items():
        if "alt" in str(key).casefold():
            return value
    return None


def coordinates(feature: dict[str, Any]) -> tuple[object | None, object | None]:
    """Extract latitude and longitude from GeoJSON geometry."""
    geometry = feature.get("geometry", {})
    if not isinstance(geometry, dict):
        return None, None
    coords = geometry.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return None, None
    longitude, latitude = coords[0], coords[1]
    return latitude, longitude


def features_from_payload(payload: Any, source_endpoint: str) -> list[dict[str, Any]]:
    """Normalize station metadata payloads into GeoJSON-like features."""
    if isinstance(payload, list):
        return [feature | {"_source_endpoint": source_endpoint} for feature in payload if isinstance(feature, dict)]
    if isinstance(payload, dict):
        features = payload.get("features")
        if isinstance(features, list):
            return [
                feature | {"_source_endpoint": source_endpoint}
                for feature in features
                if isinstance(feature, dict)
            ]
    raise ValueError(f"Malformed metadata from {source_endpoint}: expected feature list or FeatureCollection.")


def match_one(v1_id: str, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find exact or lossless numeric-equivalent identifier matches."""
    matches = []
    v1_numeric = numeric_equivalent(v1_id)
    for feature in features:
        ids = identifier_values(feature)
        for field, official_id in ids.items():
            method = None
            if official_id == v1_id:
                method = "exact_string"
            elif v1_numeric is not None and numeric_equivalent(official_id) == v1_numeric:
                method = "numeric_equivalent"
            if method is None:
                continue
            latitude, longitude = coordinates(feature)
            matches.append(
                {
                    "v1_identifier": v1_id,
                    "matched_official_identifier": official_id,
                    "source_endpoint": feature.get("_source_endpoint"),
                    "station_name": station_name(feature),
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude": altitude(feature),
                    "identifier_field": field,
                    "other_identifier_fields": json.dumps(ids, ensure_ascii=False, sort_keys=True),
                    "match_method": method,
                    "confidence": "high" if method == "exact_string" else "medium",
                    "ambiguity_notes": "",
                }
            )
    return matches


def canonical_identifier(value: object) -> str:
    """Return a stable identifier for grouping corroborating metadata records."""
    normalized = normalize_identifier(value)
    if normalized is None:
        return ""
    return numeric_equivalent(normalized) or normalized


def numeric_values(matches: list[dict[str, Any]], field: str) -> list[float]:
    """Collect numeric field values while ignoring unavailable coordinates."""
    values = []
    for item in matches:
        value = item.get(field)
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def span(values: list[float]) -> float:
    """Return the numeric span of a list, or zero for empty/singleton lists."""
    return max(values) - min(values) if len(values) > 1 else 0.0


def endpoint_rank(endpoint: object) -> int:
    """Prefer the station metadata endpoint when choosing representative fields."""
    if endpoint == IPMA_STATIONS_URL:
        return 0
    if endpoint == IPMA_SURFACE_URL:
        return 1
    return 2


def preferred_match(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose one deterministic representative row from corroborating matches."""
    return sorted(
        matches,
        key=lambda item: (
            endpoint_rank(item.get("source_endpoint")),
            item.get("latitude") in ("", None),
            item.get("longitude") in ("", None),
            str(item.get("station_name", "")),
            str(item.get("identifier_field", "")),
        ),
    )[0].copy()


def collapse_corroborated_matches(matches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    """Collapse repeated endpoint records when they corroborate one station ID."""
    if not matches:
        return [], "no_match", "No exact or numeric-equivalent metadata match."

    identifiers = {canonical_identifier(item.get("matched_official_identifier")) for item in matches}
    if len(identifiers) != 1:
        return matches, "multiple_exact_matches", "Multiple official identifiers matched this v1 identifier."

    latitudes = numeric_values(matches, "latitude")
    longitudes = numeric_values(matches, "longitude")
    latitude_span = span(latitudes)
    longitude_span = span(longitudes)
    if (
        latitude_span > COORDINATE_CORROBORATION_TOLERANCE_DEGREES
        or longitude_span > COORDINATE_CORROBORATION_TOLERANCE_DEGREES
    ):
        return (
            matches,
            "multiple_exact_matches",
            "Matched records share an identifier but have materially different coordinates.",
        )

    row = preferred_match(matches)
    source_endpoints = sorted({str(item.get("source_endpoint", "")) for item in matches if item.get("source_endpoint")})
    station_names = sorted({str(item.get("station_name", "")) for item in matches if item.get("station_name")})
    identifier_fields = sorted({str(item.get("identifier_field", "")) for item in matches if item.get("identifier_field")})
    methods = {str(item.get("match_method", "")) for item in matches}
    row["source_endpoint"] = ";".join(source_endpoints)
    if station_names:
        row["station_name"] = "; ".join(station_names)
    row["identifier_field"] = ";".join(identifier_fields)
    row["match_method"] = "exact_string" if "exact_string" in methods else "numeric_equivalent"
    row["confidence"] = "high" if row["match_method"] == "exact_string" else "medium"
    row["other_identifier_fields"] = json.dumps(
        sorted({str(item.get("other_identifier_fields", "")) for item in matches if item.get("other_identifier_fields")}),
        ensure_ascii=False,
        sort_keys=True,
    )
    note = ""
    if len(matches) > 1:
        note = (
            "Multiple official IPMA metadata records corroborate the same station identifier; "
            f"coordinate spans are {latitude_span:.6f} latitude and {longitude_span:.6f} longitude degrees."
        )
    return [row], "exact_match", note


def mapping_rows(v1_ids: list[str], features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create one or more deterministic mapping rows for each v1 identifier."""
    rows = []
    for v1_id in v1_ids:
        matches = match_one(v1_id, features)
        collapsed_matches, status, ambiguity_note = collapse_corroborated_matches(matches)

        if not collapsed_matches:
            rows.append(
                {
                    "v1_identifier": v1_id,
                    "status": status,
                    "matched_official_identifier": "",
                    "source_endpoint": "",
                    "station_name": "",
                    "latitude": "",
                    "longitude": "",
                    "altitude": "",
                    "identifier_field": "",
                    "other_identifier_fields": "",
                    "match_method": "",
                    "confidence": "none",
                    "ambiguity_notes": ambiguity_note,
                }
            )
            continue

        for item in sorted(collapsed_matches, key=lambda row: tuple(str(row.get(key, "")) for key in row)):
            row = {"status": status, **item}
            if status == "multiple_exact_matches":
                row["confidence"] = "low"
                row["ambiguity_notes"] = ambiguity_note
            else:
                row["ambiguity_notes"] = ambiguity_note
            rows.append(row)
    return rows


def write_mapping_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write mapping rows with deterministic columns."""
    fieldnames = [
        "v1_identifier",
        "status",
        "matched_official_identifier",
        "source_endpoint",
        "station_name",
        "latitude",
        "longitude",
        "altitude",
        "identifier_field",
        "other_identifier_fields",
        "match_method",
        "confidence",
        "ambiguity_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize_mapping(
    *,
    retrieved_at: str,
    v1_ids: list[str],
    rows: list[dict[str, Any]],
    raw_checksums: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    """Build a deterministic mapping summary."""
    per_id_status = {}
    for v1_id in v1_ids:
        statuses = {row["status"] for row in rows if row["v1_identifier"] == v1_id}
        per_id_status[v1_id] = sorted(statuses)[0] if statuses else "unresolved_metadata"

    exact_ids = [v1_id for v1_id, status in per_id_status.items() if status == "exact_match"]
    ambiguous_ids = [v1_id for v1_id, status in per_id_status.items() if status == "multiple_exact_matches"]
    unmatched_ids = [v1_id for v1_id, status in per_id_status.items() if status == "no_match"]
    coordinate_rows = [
        row
        for row in rows
        if row["status"] == "exact_match" and row.get("latitude") not in ("", None) and row.get("longitude") not in ("", None)
    ]
    return {
        "retrieval_timestamp_utc": retrieved_at,
        "source_identifiers": {
            "stations_json": IPMA_STATIONS_URL,
            "obs_surface_geojson": IPMA_SURFACE_URL,
        },
        "raw_response_sha256": raw_checksums,
        "v1_identifier_count": len(v1_ids),
        "expected_v1_identifier_count": EXPECTED_V1_STATION_COUNT,
        "exact_match_count": len(exact_ids),
        "ambiguous_match_count": len(ambiguous_ids),
        "unmatched_count": len(unmatched_ids),
        "exact_match_ids": exact_ids,
        "ambiguous_match_ids": ambiguous_ids,
        "unmatched_ids": unmatched_ids,
        "metadata_errors": [],
        "warnings": warnings,
        "enough_coordinates_for_meaningful_era5_pilot": len(coordinate_rows) >= 3,
        "interpretation_note": (
            "A matching IPMA identifier does not prove IPMA was the original source "
            "of the historical v1 matrices."
        ),
    }


def run_mapping_probe(
    *,
    weather_matrix: Path,
    output_dir: Path,
    timeout: float,
    optional_matrices: list[Path] | None = None,
) -> dict[str, Any]:
    """Run the IPMA station-mapping probe."""
    optional_matrices = optional_matrices or []
    retrieved_at = utc_timestamp()
    v1_ids = extract_station_ids(weather_matrix)
    warnings = validate_optional_matrices(v1_ids, optional_matrices)
    if len(v1_ids) != EXPECTED_V1_STATION_COUNT:
        warnings.append(
            f"Expected {EXPECTED_V1_STATION_COUNT} v1 station identifiers but found {len(v1_ids)}."
        )

    stations_payload = fetch_json(IPMA_STATIONS_URL, timeout)
    surface_payload = fetch_json(IPMA_SURFACE_URL, timeout)
    stations_path = output_dir / "ipma_stations_raw.json"
    surface_path = output_dir / "ipma_surface_raw.geojson"
    raw_checksums = {
        "ipma_stations_raw.json": write_json(stations_path, stations_payload),
        "ipma_surface_raw.geojson": write_json(surface_path, surface_payload),
    }

    features = features_from_payload(stations_payload, IPMA_STATIONS_URL) + features_from_payload(
        surface_payload,
        IPMA_SURFACE_URL,
    )
    rows = mapping_rows(v1_ids, features)
    mapping_path = output_dir / "ipma_station_mapping.csv"
    write_mapping_csv(mapping_path, rows)

    summary = summarize_mapping(
        retrieved_at=retrieved_at,
        v1_ids=v1_ids,
        rows=rows,
        raw_checksums=raw_checksums,
        warnings=warnings,
    )
    summary_path = output_dir / "ipma_station_mapping_summary.json"
    write_json(summary_path, summary)
    return {
        "mapping_path": str(mapping_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    optional_matrices = [
        path
        for path in [args.wind_speed_matrix, args.wind_direction_matrix, args.temperature_matrix]
        if path is not None
    ]
    result = run_mapping_probe(
        weather_matrix=args.weather_matrix,
        output_dir=args.output_dir,
        timeout=args.timeout,
        optional_matrices=optional_matrices,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
