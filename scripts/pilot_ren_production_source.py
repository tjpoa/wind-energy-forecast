"""Probe the official REN production-breakdown endpoint for one date."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


REN_PRODUCTION_ENDPOINT = (
    "https://servicebus.ren.pt/datahubapi/electricity/"
    "ElectricityProductionBreakdownDaily"
)
ENDPOINT_IDENTIFIER = "REN ElectricityProductionBreakdownDaily"
DEFAULT_OUTPUT_DIR = Path("data/pilot/ren")
DEFAULT_TIMEOUT_SECONDS = 20.0
RAW_TIMESTAMP_COLUMN = "Data e Hora"
RAW_WIND_COLUMN = "Eólica"
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Probe one REN production-breakdown date and optionally compare it with the local CSV."
    )
    parser.add_argument("--date", required=True, help="Single date to request, formatted as YYYY-MM-DD.")
    parser.add_argument("--local-csv", type=Path, help="Optional local ReparticaoProducao.csv path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for ignored pilot output.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds.")
    return parser.parse_args()


def validate_date(date_str: str) -> str:
    """Validate and normalize a YYYY-MM-DD date."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError("--date must be formatted as YYYY-MM-DD.") from exc


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def request_ren_production(date_str: str, timeout: float) -> requests.Response:
    """Request one REN production-breakdown date."""
    response = requests.get(
        REN_PRODUCTION_ENDPOINT,
        params={"culture": "pt-PT", "date": date_str},
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def write_json(path: Path, payload: Any) -> str:
    """Write JSON and return the saved-file SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relevant_headers(response: requests.Response) -> dict[str, str]:
    """Return non-sensitive response headers useful for source documentation."""
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() in RELEVANT_RESPONSE_HEADERS
    }


def safe_json(response: requests.Response) -> Any:
    """Parse response JSON after a successful HTTP response."""
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError("REN response was not valid JSON.") from exc


def normalize_text(value: object) -> str:
    """Normalize a field name for conservative matching."""
    text = str(value).strip().casefold()
    replacements = {
        "á": "a",
        "à": "a",
        "â": "a",
        "ã": "a",
        "ç": "c",
        "é": "e",
        "ê": "e",
        "í": "i",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ú": "u",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def extract_categories(payload: Any) -> list[str]:
    """Extract REN x-axis categories when present."""
    if not isinstance(payload, dict):
        return []
    x_axis = payload.get("xAxis")
    if not isinstance(x_axis, dict):
        return []
    categories = x_axis.get("categories")
    if not isinstance(categories, list):
        return []
    return [str(item) for item in categories]


def extract_series(payload: Any) -> list[dict[str, Any]]:
    """Extract REN series entries when present."""
    if not isinstance(payload, dict):
        return []
    series = payload.get("series")
    if not isinstance(series, list):
        return []
    return [item for item in series if isinstance(item, dict)]


def find_wind_series(series: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Find wind-related REN series without guessing between ambiguous candidates."""
    matches = []
    notes = []
    for item in series:
        name = item.get("name", "")
        normalized = normalize_text(name)
        if "eolica" in normalized or "wind" in normalized:
            matches.append(item)
    if not matches:
        notes.append("No wind-related series name was identified.")
    elif len(matches) > 1:
        notes.append("Multiple wind-related series were identified; no single series was selected automatically.")
    return matches, notes


def infer_interval(timestamps: list[pd.Timestamp]) -> str | None:
    """Infer the most common timestamp interval."""
    if len(timestamps) < 2:
        return None
    values = pd.Series(timestamps).sort_values().diff().dropna()
    if values.empty:
        return None
    return str(values.mode().iloc[0])


def category_timestamps(date_str: str, categories: list[str]) -> list[pd.Timestamp]:
    """Convert REN categories to timestamps for one requested date."""
    timestamps = []
    for category in categories:
        text = str(category).strip()
        if ":" in text and len(text) <= 8:
            parsed = pd.to_datetime(f"{date_str} {text}", errors="coerce")
        else:
            parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            continue
        timestamps.append(pd.Timestamp(parsed))
    return timestamps


def summarize_payload(
    *,
    requested_date: str,
    response: requests.Response,
    payload: Any,
    raw_sha256: str,
    retrieved_at: str,
) -> dict[str, Any]:
    """Build a normalized REN response summary."""
    categories = extract_categories(payload)
    series = extract_series(payload)
    timestamps = category_timestamps(requested_date, categories)
    wind_matches, notes = find_wind_series(series)
    y_axis = payload.get("yAxis") if isinstance(payload, dict) else None
    y_axis_title = y_axis.get("title", {}).get("text") if isinstance(y_axis, dict) else None

    summary = {
        "requested_date": requested_date,
        "retrieval_timestamp_utc": retrieved_at,
        "endpoint_identifier": ENDPOINT_IDENTIFIER,
        "endpoint_url": REN_PRODUCTION_ENDPOINT,
        "http_status": response.status_code,
        "response_content_type": response.headers.get("content-type"),
        "response_headers": relevant_headers(response),
        "top_level_response_keys": list(payload.keys()) if isinstance(payload, dict) else None,
        "apparent_record_count": len(categories) if categories else None,
        "earliest_returned_timestamp": min(timestamps).isoformat() if timestamps else None,
        "latest_returned_timestamp": max(timestamps).isoformat() if timestamps else None,
        "apparent_temporal_granularity": infer_interval(timestamps),
        "series_names": [item.get("name") for item in series],
        "wind_related_series_names": [item.get("name") for item in wind_matches],
        "units_explicitly_present": y_axis_title,
        "raw_response_sha256": raw_sha256,
        "unresolved_interpretation_notes": notes,
    }
    if len(wind_matches) == 1:
        data = wind_matches[0].get("data")
        summary["selected_wind_series_name"] = wind_matches[0].get("name")
        summary["selected_wind_series_record_count"] = len(data) if isinstance(data, list) else None
    return summary


def load_local_production(local_csv: Path, requested_date: str) -> pd.DataFrame:
    """Load current local production CSV for one date without modifying it."""
    df = pd.read_csv(local_csv, sep=";", skiprows=2, na_values=-990)
    if RAW_TIMESTAMP_COLUMN not in df.columns or RAW_WIND_COLUMN not in df.columns:
        raise ValueError(f"Local CSV must contain '{RAW_TIMESTAMP_COLUMN}' and '{RAW_WIND_COLUMN}'.")
    result = df[[RAW_TIMESTAMP_COLUMN, RAW_WIND_COLUMN]].copy()
    result[RAW_TIMESTAMP_COLUMN] = pd.to_datetime(result[RAW_TIMESTAMP_COLUMN], errors="coerce")
    result[RAW_WIND_COLUMN] = pd.to_numeric(result[RAW_WIND_COLUMN], errors="coerce")
    start = pd.Timestamp(requested_date)
    end = start + pd.Timedelta(days=1)
    mask = (result[RAW_TIMESTAMP_COLUMN] >= start) & (result[RAW_TIMESTAMP_COLUMN] < end)
    result = result.loc[mask].dropna(subset=[RAW_TIMESTAMP_COLUMN, RAW_WIND_COLUMN])
    return result.rename(columns={RAW_TIMESTAMP_COLUMN: "timestamp", RAW_WIND_COLUMN: "local_wind_mw"})


def ren_wind_frame(date_str: str, payload: Any) -> tuple[pd.DataFrame | None, str]:
    """Extract one unambiguous REN wind series into a DataFrame."""
    categories = extract_categories(payload)
    series = extract_series(payload)
    wind_matches, _ = find_wind_series(series)
    if len(wind_matches) != 1:
        return None, "unresolved_wind_series"
    data = wind_matches[0].get("data")
    if not isinstance(data, list):
        return None, "missing_wind_data"
    timestamps = category_timestamps(date_str, categories)
    if len(timestamps) != len(data):
        return None, "timestamp_value_length_mismatch"
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "ren_wind_mw": pd.to_numeric(pd.Series(data), errors="coerce"),
        }
    ), "ok"


def compare_with_local(date_str: str, payload: Any, local_csv: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Compare one REN wind series with the local production CSV."""
    local = load_local_production(local_csv, date_str)
    ren, status = ren_wind_frame(date_str, payload)
    summary: dict[str, Any] = {
        "comparison_status": status,
        "local_csv": str(local_csv),
        "local_row_count": int(len(local)),
        "ren_row_count": int(len(ren)) if ren is not None else None,
        "local_units": "MW",
        "ren_units": "MW when yAxis title is MW",
    }
    if ren is None:
        return None, summary

    aligned = pd.merge(local, ren, on="timestamp", how="inner").sort_values("timestamp")
    summary["aligned_timestamp_count"] = int(len(aligned))
    summary["inferred_interval"] = infer_interval(list(aligned["timestamp"]))
    if aligned.empty:
        summary["comparison_status"] = "no_aligned_timestamps"
        return aligned, summary

    diff = aligned["ren_wind_mw"] - aligned["local_wind_mw"]
    aligned["difference_mw"] = diff
    summary["comparison_status"] = "compared"
    summary["exact_value_match_count"] = int((diff == 0).sum())
    summary["mean_absolute_difference"] = float(diff.abs().mean())
    summary["maximum_absolute_difference"] = float(diff.abs().max())
    summary["pearson_correlation"] = (
        float(aligned["local_wind_mw"].corr(aligned["ren_wind_mw"])) if len(aligned) > 1 else None
    )
    non_zero = aligned[(aligned["local_wind_mw"] != 0) & aligned["ren_wind_mw"].notna()]
    ratios = non_zero["ren_wind_mw"] / non_zero["local_wind_mw"]
    finite_ratios = ratios[[math.isfinite(float(value)) for value in ratios]]
    summary["median_ren_to_local_ratio"] = float(finite_ratios.median()) if not finite_ratios.empty else None
    return aligned, summary


def run_probe(date_str: str, output_dir: Path, timeout: float, local_csv: Path | None = None) -> dict[str, Any]:
    """Run one REN probe and write deterministic outputs."""
    requested_date = validate_date(date_str)
    retrieved_at = utc_timestamp()
    response = request_ren_production(requested_date, timeout)
    payload = safe_json(response)

    raw_path = output_dir / f"ren_production_{requested_date}_raw.json"
    raw_sha256 = write_json(raw_path, payload)
    summary = summarize_payload(
        requested_date=requested_date,
        response=response,
        payload=payload,
        raw_sha256=raw_sha256,
        retrieved_at=retrieved_at,
    )

    if local_csv is not None:
        comparison, comparison_summary = compare_with_local(requested_date, payload, local_csv)
        summary["local_comparison"] = comparison_summary
        if comparison is not None and not comparison.empty and comparison_summary["comparison_status"] == "compared":
            comparison_path = output_dir / f"ren_production_{requested_date}_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            summary["local_comparison"]["comparison_csv"] = str(comparison_path)

    summary_path = output_dir / f"ren_production_{requested_date}_summary.json"
    write_json(summary_path, summary)
    return {"raw_path": str(raw_path), "summary_path": str(summary_path), "summary": summary}


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = run_probe(args.date, args.output_dir, args.timeout, args.local_csv)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
