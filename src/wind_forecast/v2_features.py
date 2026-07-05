"""Feature-ready v2 dataset builder.

The functions in this module are import-safe: importing the module does not
read local datasets, write outputs, start network calls, or execute pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from wind_forecast.features import (
    WEATHER_LAGS,
    WIND_PRODUCTION_LAGS,
    apply_feature_engineering,
)
from wind_forecast.schemas import (
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
)


TRANSFORMATION_VERSION = "feature_ready_ren_era5_land_v2_2A.18"
SOURCE_TRANSFORMATION_VERSION = "integrated_ren_era5_land_v2_local_day_2A.17"
DATE_LOCAL_COLUMN = "date_local"

SOURCE_OUTPUT_FILES = (
    "daily_merged.csv",
    "coverage.csv",
    "validation.json",
    "manifest.json",
)

OUTPUT_FILENAMES = {
    "feature_ready_daily": "feature_ready_daily.csv",
    "feature_schema": "feature_schema.json",
    "feature_coverage": "feature_coverage.csv",
    "v1_structure_comparison": "v1_structure_comparison.json",
    "validation": "validation.json",
    "manifest": "manifest.json",
}

BASE_COLUMN_MAPPING = {
    DATE_COLUMN: DATE_LOCAL_COLUMN,
    TARGET_COLUMN: TARGET_COLUMN,
    AVG_WIND_SPEED_COLUMN: "wind_speed_m_s_mean",
    AVG_TEMPERATURE_COLUMN: "temperature_2m_c_mean",
    AVG_WIND_DIRECTION_COLUMN: "vector_mean_wind_direction_deg_from",
}

FEATURE_READY_STATUS = "feature-ready"
EXCLUDED_REN_UNAVAILABLE_STATUS = "excluded-ren-unavailable"
EXCLUDED_CURRENT_NULL_STATUS = "excluded-current-weather-or-target-null"
EXCLUDED_WARMUP_STATUS = "excluded-warmup-insufficient-14-day-history"
EXCLUDED_GAP_PRIOR_14_STATUS = "excluded-gap-in-prior-14-day-history"
EXCLUDED_DIRECTION_PRIOR_7_STATUS = "excluded-direction-null-in-prior-7-day-history"
EXCLUDED_INTEGRATED_NOT_READY_STATUS = "excluded-integrated-not-ready"

EXPECTED_PROVISIONAL_COUNTS = {
    FEATURE_READY_STATUS: 5322,
    EXCLUDED_REN_UNAVAILABLE_STATUS: 6,
    EXCLUDED_CURRENT_NULL_STATUS: 134,
    EXCLUDED_WARMUP_STATUS: 13,
    EXCLUDED_GAP_PRIOR_14_STATUS: 67,
    EXCLUDED_DIRECTION_PRIOR_7_STATUS: 480,
}

FEATURE_COVERAGE_COLUMNS = (
    DATE_LOCAL_COLUMN,
    "integrated_coverage_status",
    "ren_status",
    "era5_status",
    "integration_ready",
    "current_target_finite",
    "current_speed_finite",
    "current_temperature_finite",
    "current_direction_finite",
    "current_base_complete",
    "prior_14_calendar_days_available",
    "prior_14_target_speed_temp_finite_count",
    "prior_14_target_speed_temp_complete",
    "prior_production_lag_dates_finite",
    "prior_weather_lag_dates_finite",
    "prior_direction_lag_dates_finite",
    "feature_ready",
    "feature_coverage_status",
    "exclusion_reason",
)


class FeatureBuildError(ValueError):
    """Raised when feature-ready v2 inputs or outputs are invalid."""


@dataclass(frozen=True)
class FeatureBuildPaths:
    """Resolved input and output paths for one feature-ready build."""

    input_root: Path
    v1_feature_table: Path
    output_root: Path

    @property
    def input_files(self) -> dict[str, Path]:
        """Return expected accepted integrated input files."""
        return {filename: self.input_root / filename for filename in SOURCE_OUTPUT_FILES}

    @property
    def output_files(self) -> dict[str, Path]:
        """Return generated output file paths."""
        return {key: self.output_root / filename for key, filename in OUTPUT_FILENAMES.items()}


@dataclass(frozen=True)
class IntegratedInputs:
    """Accepted Step 2A.17 integrated inputs loaded into memory."""

    daily_merged: pd.DataFrame
    coverage: pd.DataFrame
    validation: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class FeatureBuildResult:
    """In-memory result from a completed feature-ready build."""

    paths: FeatureBuildPaths
    feature_ready_daily: pd.DataFrame
    feature_schema: dict[str, Any]
    feature_coverage: pd.DataFrame
    v1_structure_comparison: dict[str, Any]
    validation: dict[str, Any]
    manifest: dict[str, Any]
    checksums: dict[str, str]

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-ready build summary."""
        return {
            "output_root": str(self.paths.output_root),
            "verdict": self.validation.get("verdict"),
            "passed": self.validation.get("passed"),
            "feature_ready_rows": int(len(self.feature_ready_daily)),
            "feature_coverage_rows": int(len(self.feature_coverage)),
            "feature_coverage_status_counts": self.validation.get(
                "feature_coverage_status_counts", {}
            ),
            "first_feature_ready_date": self.validation.get("first_feature_ready_date"),
            "last_feature_ready_date": self.validation.get("last_feature_ready_date"),
            "output_files": {
                key: str(path) for key, path in self.paths.output_files.items()
            },
            "sha256_checksums": dict(self.checksums),
        }


def resolve_feature_build_paths(
    *,
    input_root: str | Path,
    v1_feature_table: str | Path,
    output_root: str | Path,
) -> FeatureBuildPaths:
    """Return normalized paths without reading or writing datasets."""
    return FeatureBuildPaths(
        input_root=Path(input_root),
        v1_feature_table=Path(v1_feature_table),
        output_root=Path(output_root),
    )


def load_integrated_inputs(input_root: str | Path) -> IntegratedInputs:
    """Load accepted Step 2A.17 integrated inputs."""
    root = Path(input_root)
    missing_files = [filename for filename in SOURCE_OUTPUT_FILES if not (root / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Integrated input root is missing required files: {missing_files}."
        )

    daily_merged = pd.read_csv(root / "daily_merged.csv")
    coverage = pd.read_csv(root / "coverage.csv")
    validation = _read_json(root / "validation.json")
    manifest = _read_json(root / "manifest.json")
    return IntegratedInputs(
        daily_merged=daily_merged,
        coverage=coverage,
        validation=validation,
        manifest=manifest,
    )


def load_v1_feature_columns(v1_feature_table: str | Path) -> list[str]:
    """Load the v1 feature-table columns and order."""
    path = Path(v1_feature_table)
    if not path.is_file():
        raise FileNotFoundError(f"v1 feature table is missing: {path}.")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if not columns:
        raise FeatureBuildError(f"v1 feature table has no columns: {path}.")
    return [str(column) for column in columns]


def map_integrated_base_columns(daily_merged: pd.DataFrame) -> pd.DataFrame:
    """Map accepted integrated columns to the v1 feature-engineering base schema."""
    required_columns = set(BASE_COLUMN_MAPPING.values())
    missing_columns = sorted(required_columns.difference(daily_merged.columns))
    if missing_columns:
        raise FeatureBuildError(
            f"Integrated daily dataset is missing required columns: {missing_columns}."
        )
    if daily_merged[DATE_LOCAL_COLUMN].duplicated().any():
        duplicates = (
            daily_merged.loc[daily_merged[DATE_LOCAL_COLUMN].duplicated(), DATE_LOCAL_COLUMN]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise FeatureBuildError(
            f"Integrated daily dataset has duplicate local dates: {duplicates}."
        )

    mapped = pd.DataFrame(
        {
            output_column: daily_merged[input_column]
            for output_column, input_column in BASE_COLUMN_MAPPING.items()
        }
    )
    mapped[DATE_COLUMN] = _parse_date_series(mapped[DATE_COLUMN], DATE_COLUMN)
    for column in (
        TARGET_COLUMN,
        AVG_WIND_SPEED_COLUMN,
        AVG_TEMPERATURE_COLUMN,
        AVG_WIND_DIRECTION_COLUMN,
    ):
        mapped[column] = pd.to_numeric(mapped[column], errors="coerce")
    return mapped.sort_values(DATE_COLUMN).reset_index(drop=True)


def reindex_full_local_calendar(
    coverage: pd.DataFrame,
    mapped_base: pd.DataFrame,
) -> pd.DataFrame:
    """Reindex mapped base rows to the full requested local calendar."""
    required_coverage_columns = {
        DATE_LOCAL_COLUMN,
        "coverage_status",
        "ren_status",
        "era5_status",
        "integration_ready",
    }
    missing_columns = sorted(required_coverage_columns.difference(coverage.columns))
    if missing_columns:
        raise FeatureBuildError(
            f"Integrated coverage table is missing required columns: {missing_columns}."
        )
    if coverage[DATE_LOCAL_COLUMN].duplicated().any():
        duplicates = (
            coverage.loc[coverage[DATE_LOCAL_COLUMN].duplicated(), DATE_LOCAL_COLUMN]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise FeatureBuildError(f"Coverage table has duplicate local dates: {duplicates}.")

    coverage_copy = coverage.copy()
    coverage_copy[DATE_LOCAL_COLUMN] = _parse_date_series(
        coverage_copy[DATE_LOCAL_COLUMN], DATE_LOCAL_COLUMN
    )
    coverage_copy = coverage_copy.sort_values(DATE_LOCAL_COLUMN).reset_index(drop=True)
    start_date = coverage_copy[DATE_LOCAL_COLUMN].min()
    end_date = coverage_copy[DATE_LOCAL_COLUMN].max()
    full_dates = pd.date_range(start_date, end_date, freq="D")
    calendar = pd.DataFrame({DATE_LOCAL_COLUMN: full_dates})

    coverage_columns = [
        DATE_LOCAL_COLUMN,
        "coverage_status",
        "ren_status",
        "era5_status",
        "integration_ready",
    ]
    calendar = calendar.merge(
        coverage_copy[coverage_columns],
        on=DATE_LOCAL_COLUMN,
        how="left",
        validate="one_to_one",
    )
    calendar["coverage_status"] = calendar["coverage_status"].fillna("missing-coverage-row")
    calendar["ren_status"] = calendar["ren_status"].fillna("missing")
    calendar["era5_status"] = calendar["era5_status"].fillna("missing")
    calendar["integration_ready"] = _bool_series(calendar["integration_ready"])

    base = mapped_base.copy()
    base[DATE_LOCAL_COLUMN] = base[DATE_COLUMN]
    calendar = calendar.merge(
        base[
            [
                DATE_LOCAL_COLUMN,
                TARGET_COLUMN,
                AVG_WIND_SPEED_COLUMN,
                AVG_TEMPERATURE_COLUMN,
                AVG_WIND_DIRECTION_COLUMN,
            ]
        ],
        on=DATE_LOCAL_COLUMN,
        how="left",
        validate="one_to_one",
    )
    calendar[DATE_COLUMN] = calendar[DATE_LOCAL_COLUMN]
    return calendar[
        [
            DATE_COLUMN,
            DATE_LOCAL_COLUMN,
            "coverage_status",
            "ren_status",
            "era5_status",
            "integration_ready",
            TARGET_COLUMN,
            AVG_WIND_SPEED_COLUMN,
            AVG_TEMPERATURE_COLUMN,
            AVG_WIND_DIRECTION_COLUMN,
        ]
    ].reset_index(drop=True)


def generate_v2_features(full_calendar_base: pd.DataFrame) -> pd.DataFrame:
    """Generate v1 feature formulas over the full local calendar without filling NaNs."""
    base_columns = [
        DATE_COLUMN,
        TARGET_COLUMN,
        AVG_WIND_SPEED_COLUMN,
        AVG_TEMPERATURE_COLUMN,
        AVG_WIND_DIRECTION_COLUMN,
    ]
    missing_columns = sorted(set(base_columns).difference(full_calendar_base.columns))
    if missing_columns:
        raise FeatureBuildError(f"Calendar base table is missing columns: {missing_columns}.")
    with contextlib.redirect_stdout(io.StringIO()):
        features = apply_feature_engineering(full_calendar_base[base_columns])
    return features.reset_index(drop=True)


def build_feature_coverage(full_calendar_base: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic per-date feature eligibility and exclusion lineage."""
    frame = full_calendar_base.copy().reset_index(drop=True)
    frame[DATE_LOCAL_COLUMN] = _parse_date_series(frame[DATE_LOCAL_COLUMN], DATE_LOCAL_COLUMN)
    date_text = frame[DATE_LOCAL_COLUMN].dt.strftime("%Y-%m-%d")

    target_finite = _finite_column(frame, TARGET_COLUMN)
    speed_finite = _finite_column(frame, AVG_WIND_SPEED_COLUMN)
    temperature_finite = _finite_column(frame, AVG_TEMPERATURE_COLUMN)
    direction_finite = _finite_column(frame, AVG_WIND_DIRECTION_COLUMN)

    current_base_complete = (
        frame["integration_ready"].to_numpy(dtype=bool)
        & target_finite
        & speed_finite
        & temperature_finite
        & direction_finite
    )
    target_speed_temp_finite = target_finite & speed_finite & temperature_finite

    prior_14_count = _prior_complete_day_count(target_speed_temp_finite, window=14)
    prior_14_available = np.minimum(np.arange(len(frame), dtype=int), 14)
    prior_14_complete = prior_14_count == 14

    prior_production_lag_dates_finite = _lag_finite_mask(
        target_finite, WIND_PRODUCTION_LAGS
    )
    weather_speed_temp = speed_finite & temperature_finite
    prior_weather_lag_dates_finite = _lag_finite_mask(weather_speed_temp, WEATHER_LAGS)
    prior_direction_lag_dates_finite = _lag_finite_mask(direction_finite, WEATHER_LAGS)

    ren_unavailable = (
        frame["coverage_status"].astype(str).eq("excluded-downstream-ren-unavailable")
        | frame["ren_status"].astype(str).eq("unavailable")
    ).to_numpy()
    integration_ready = frame["integration_ready"].to_numpy(dtype=bool)
    warmup = np.arange(len(frame), dtype=int) < 14

    statuses: list[str] = []
    reasons: list[str] = []
    for index in range(len(frame)):
        if ren_unavailable[index]:
            status = EXCLUDED_REN_UNAVAILABLE_STATUS
            reason = "REN daily production is unavailable in the accepted integrated coverage."
        elif not integration_ready[index]:
            status = EXCLUDED_INTEGRATED_NOT_READY_STATUS
            reason = "The accepted integrated row is not integration-ready."
        elif not current_base_complete[index]:
            status = EXCLUDED_CURRENT_NULL_STATUS
            reason = "Current target, speed, temperature, or direction is missing or non-finite."
        elif warmup[index]:
            status = EXCLUDED_WARMUP_STATUS
            reason = "Fewer than 14 prior local calendar days are available."
        elif not prior_14_complete[index]:
            status = EXCLUDED_GAP_PRIOR_14_STATUS
            reason = "At least one prior 14-day target/speed/temperature value is missing or non-finite."
        elif not (
            prior_production_lag_dates_finite[index]
            and prior_weather_lag_dates_finite[index]
            and prior_direction_lag_dates_finite[index]
        ):
            status = EXCLUDED_DIRECTION_PRIOR_7_STATUS
            reason = "At least one prior direction lag date (1, 2, 3, or 7) is missing or non-finite."
        else:
            status = FEATURE_READY_STATUS
            reason = ""
        statuses.append(status)
        reasons.append(reason)

    coverage = pd.DataFrame(
        {
            DATE_LOCAL_COLUMN: date_text,
            "integrated_coverage_status": frame["coverage_status"].astype(str),
            "ren_status": frame["ren_status"].astype(str),
            "era5_status": frame["era5_status"].astype(str),
            "integration_ready": integration_ready,
            "current_target_finite": target_finite,
            "current_speed_finite": speed_finite,
            "current_temperature_finite": temperature_finite,
            "current_direction_finite": direction_finite,
            "current_base_complete": current_base_complete,
            "prior_14_calendar_days_available": prior_14_available,
            "prior_14_target_speed_temp_finite_count": prior_14_count,
            "prior_14_target_speed_temp_complete": prior_14_complete,
            "prior_production_lag_dates_finite": prior_production_lag_dates_finite,
            "prior_weather_lag_dates_finite": prior_weather_lag_dates_finite,
            "prior_direction_lag_dates_finite": prior_direction_lag_dates_finite,
            "feature_ready": np.array(statuses) == FEATURE_READY_STATUS,
            "feature_coverage_status": statuses,
            "exclusion_reason": reasons,
        },
        columns=list(FEATURE_COVERAGE_COLUMNS),
    )
    return coverage


def select_feature_ready_rows(
    features: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    v1_columns: Sequence[str],
) -> pd.DataFrame:
    """Return feature-ready rows aligned to the exact v1 feature columns."""
    missing_columns = [column for column in v1_columns if column not in features.columns]
    if missing_columns:
        raise FeatureBuildError(
            f"Generated v2 features are missing v1 columns: {missing_columns}."
        )
    if len(features) != len(feature_coverage):
        raise FeatureBuildError(
            "Feature rows and feature coverage rows must have identical lengths."
        )
    ready_mask = feature_coverage["feature_coverage_status"].eq(FEATURE_READY_STATUS)
    feature_ready = features.loc[ready_mask.to_numpy(), list(v1_columns)].copy()
    feature_ready[DATE_COLUMN] = pd.to_datetime(feature_ready[DATE_COLUMN]).dt.strftime("%Y-%m-%d")
    return feature_ready.reset_index(drop=True)


def compare_v1_structure(
    *,
    v1_feature_table: str | Path,
    v1_columns: Sequence[str],
    feature_ready_daily: pd.DataFrame,
) -> dict[str, Any]:
    """Compare v2 feature-ready output structure with the local v1 feature table."""
    path = Path(v1_feature_table)
    v1_sample = pd.read_csv(path, nrows=50)
    v2_sample = feature_ready_daily.head(50)
    v1_column_list = [str(column) for column in v1_columns]
    v2_column_list = [str(column) for column in feature_ready_daily.columns]
    v1_numeric_columns = [
        column for column in v1_column_list if column != DATE_COLUMN and column in v1_sample
    ]
    v2_numeric_columns = [
        column for column in v2_column_list if column != DATE_COLUMN and column in v2_sample
    ]
    numeric_columns_match = v1_numeric_columns == v2_numeric_columns
    v2_numeric_finite = True
    if v2_numeric_columns:
        v2_numeric_values = v2_sample[v2_numeric_columns].apply(pd.to_numeric, errors="coerce")
        v2_numeric_finite = bool(np.isfinite(v2_numeric_values.to_numpy(dtype=float)).all())

    return {
        "schema_version": "wind_forecast.v1_structure_comparison.v1",
        "v1_feature_table": str(path),
        "v1_row_count": int(len(pd.read_csv(path, usecols=[DATE_COLUMN]))),
        "v2_feature_ready_row_count": int(len(feature_ready_daily)),
        "v1_column_count": int(len(v1_column_list)),
        "v2_column_count": int(len(v2_column_list)),
        "exact_column_order_match": v1_column_list == v2_column_list,
        "missing_from_v2": [column for column in v1_column_list if column not in v2_column_list],
        "extra_in_v2": [column for column in v2_column_list if column not in v1_column_list],
        "different_order": (
            []
            if v1_column_list == v2_column_list
            else [
                {
                    "position": int(index),
                    "v1": v1_column_list[index] if index < len(v1_column_list) else None,
                    "v2": v2_column_list[index] if index < len(v2_column_list) else None,
                }
                for index in range(max(len(v1_column_list), len(v2_column_list)))
                if (
                    index >= len(v1_column_list)
                    or index >= len(v2_column_list)
                    or v1_column_list[index] != v2_column_list[index]
                )
            ]
        ),
        "date_column_name_match": (
            bool(v1_column_list)
            and bool(v2_column_list)
            and v1_column_list[0] == DATE_COLUMN
            and v2_column_list[0] == DATE_COLUMN
        ),
        "numeric_feature_columns_match": numeric_columns_match,
        "v2_numeric_sample_finite": v2_numeric_finite,
    }


def build_feature_schema(v1_columns: Sequence[str]) -> dict[str, Any]:
    """Build deterministic schema metadata for the generated feature table."""
    return {
        "schema_version": "wind_forecast.feature_ready_schema.v1",
        "dataset_version": "v2",
        "dataset_role": "feature_ready_daily_ren_era5_land",
        "transformation_version": TRANSFORMATION_VERSION,
        "formula_source": "wind_forecast.features.apply_feature_engineering",
        "fill_policy": {
            "handle_final_nans_called": False,
            "backfill": False,
            "forward_fill": False,
            "interpolation": False,
            "fill_zero": False,
        },
        "base_column_mapping": dict(BASE_COLUMN_MAPPING),
        "eligibility_policy": {
            "current_integrated_row_must_be_integration_ready": True,
            "current_base_columns_must_be_finite": [
                TARGET_COLUMN,
                AVG_WIND_SPEED_COLUMN,
                AVG_TEMPERATURE_COLUMN,
                AVG_WIND_DIRECTION_COLUMN,
            ],
            "wind_production_lags": list(WIND_PRODUCTION_LAGS),
            "weather_lags": list(WEATHER_LAGS),
            "rolling_windows_require_prior_14_calendar_days": True,
            "direction_lag_dates_required": list(WEATHER_LAGS),
        },
        "column_count": int(len(v1_columns)),
        "columns": [
            {
                "position": int(index),
                "name": str(column),
                "role": _feature_column_role(str(column)),
            }
            for index, column in enumerate(v1_columns)
        ],
    }


def validate_feature_ready_outputs(
    *,
    inputs: IntegratedInputs,
    full_calendar_base: pd.DataFrame,
    feature_ready_daily: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    v1_structure_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate generated feature-ready v2 outputs."""
    failures: list[str] = []
    warnings: list[str] = []

    integrated_validation_passed = bool(inputs.validation.get("passed"))
    if not integrated_validation_passed:
        failures.append("Integrated Step 2A.17 validation did not pass.")
    integrated_verdict = str(inputs.validation.get("verdict", ""))
    if integrated_verdict and integrated_verdict != "PASS":
        warnings.append(f"Integrated Step 2A.17 verdict was {integrated_verdict}.")
    warnings.extend(str(item) for item in inputs.validation.get("warnings", []))

    calendar_dates = pd.to_datetime(full_calendar_base[DATE_LOCAL_COLUMN])
    expected_calendar = pd.date_range(calendar_dates.min(), calendar_dates.max(), freq="D")
    coverage_records_full_calendar = bool(
        len(calendar_dates) == len(expected_calendar)
        and calendar_dates.reset_index(drop=True).equals(pd.Series(expected_calendar))
    )
    if not coverage_records_full_calendar:
        failures.append("Feature coverage does not record a contiguous local calendar.")

    status_counts = _value_counts(feature_coverage["feature_coverage_status"])
    if status_counts != EXPECTED_PROVISIONAL_COUNTS:
        warnings.append(
            "Feature coverage status counts differ from recovered provisional counts; "
            "inspect validation.json for actual evidence."
        )

    current_complete_rows = int(feature_coverage["current_base_complete"].sum())
    feature_ready_rows = int(len(feature_ready_daily))
    feature_ready_status_rows = int(
        feature_coverage["feature_coverage_status"].eq(FEATURE_READY_STATUS).sum()
    )
    if feature_ready_rows != feature_ready_status_rows:
        failures.append("Feature-ready output row count does not match coverage status count.")

    exact_v1_structure = bool(v1_structure_comparison.get("exact_column_order_match"))
    if not exact_v1_structure:
        failures.append("V2 feature columns/order do not exactly match the v1 feature table.")

    no_feature_ready_nans = not bool(feature_ready_daily.isna().any().any())
    if not no_feature_ready_nans:
        failures.append("Feature-ready output contains NaN values.")

    numeric_columns = [column for column in feature_ready_daily.columns if column != DATE_COLUMN]
    numeric_values = feature_ready_daily[numeric_columns].apply(pd.to_numeric, errors="coerce")
    feature_ready_numeric_finite = bool(np.isfinite(numeric_values.to_numpy(dtype=float)).all())
    if not feature_ready_numeric_finite:
        failures.append("Feature-ready numeric columns contain non-finite values.")

    unexpected_integrated_status_rows = int(
        feature_coverage["feature_coverage_status"].eq(EXCLUDED_INTEGRATED_NOT_READY_STATUS).sum()
    )
    if unexpected_integrated_status_rows:
        failures.append(
            f"{unexpected_integrated_status_rows} rows were not integration-ready for reasons other than REN unavailable."
        )

    date_series = pd.to_datetime(feature_ready_daily[DATE_COLUMN])
    first_ready = date_series.min().strftime("%Y-%m-%d") if len(date_series) else None
    last_ready = date_series.max().strftime("%Y-%m-%d") if len(date_series) else None
    if first_ready != "2010-01-15":
        warnings.append(f"First feature-ready date is {first_ready}, not provisional 2010-01-15.")
    if last_ready != "2026-06-27":
        warnings.append(f"Last feature-ready date is {last_ready}, not provisional 2026-06-27.")

    checks = {
        "integrated_validation_passed": integrated_validation_passed,
        "coverage_records_full_local_calendar": coverage_records_full_calendar,
        "feature_ready_rows_match_coverage": feature_ready_rows == feature_ready_status_rows,
        "v1_column_order_exact_match": exact_v1_structure,
        "no_handle_final_nans_fill_policy": True,
        "feature_ready_output_has_no_nans": no_feature_ready_nans,
        "feature_ready_numeric_values_are_finite": feature_ready_numeric_finite,
        "no_unexpected_integrated_not_ready_rows": unexpected_integrated_status_rows == 0,
    }
    passed = not failures
    verdict = "FAIL" if failures else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "schema_version": "wind_forecast.feature_ready_validation.v1",
        "dataset_version": "v2",
        "dataset_role": "feature_ready_daily_ren_era5_land",
        "transformation_version": TRANSFORMATION_VERSION,
        "source_transformation_version": SOURCE_TRANSFORMATION_VERSION,
        "passed": passed,
        "verdict": verdict,
        "failures": failures,
        "warnings": _dedupe_preserve_order(warnings),
        "checks": checks,
        "actual_counts": {
            "coverage_rows": int(len(feature_coverage)),
            "integrated_ready_rows": int(
                feature_coverage["integration_ready"].sum()
            ),
            "current_complete_base_rows": current_complete_rows,
            "feature_ready_rows": feature_ready_rows,
        },
        "expected_provisional_counts": dict(EXPECTED_PROVISIONAL_COUNTS),
        "feature_coverage_status_counts": status_counts,
        "first_feature_ready_date": first_ready,
        "last_feature_ready_date": last_ready,
        "source_integrated_validation": {
            "passed": inputs.validation.get("passed"),
            "verdict": inputs.validation.get("verdict"),
            "coverage_status_counts": dict(inputs.validation.get("coverage_status_counts") or {}),
            "ren_status_counts": dict(inputs.validation.get("ren_status_counts") or {}),
            "era5_status_counts": dict(inputs.validation.get("era5_status_counts") or {}),
        },
        "v1_structure_comparison": dict(v1_structure_comparison),
        "safeguards": {
            "interpolation": False,
            "forward_fill": False,
            "backfill": False,
            "fill_zero": False,
            "train_test_split": False,
            "scaler_refit": False,
            "model_training": False,
            "network_requests": False,
        },
    }


def build_feature_ready_v2_dataset(
    *,
    input_root: str | Path,
    v1_feature_table: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> FeatureBuildResult:
    """Build and write the feature-ready v2 daily dataset."""
    paths = resolve_feature_build_paths(
        input_root=input_root,
        v1_feature_table=v1_feature_table,
        output_root=output_root,
    )
    if paths.output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists; use --overwrite explicitly: {paths.output_root}."
        )

    inputs = load_integrated_inputs(paths.input_root)
    v1_columns = load_v1_feature_columns(paths.v1_feature_table)
    mapped_base = map_integrated_base_columns(inputs.daily_merged)
    full_calendar_base = reindex_full_local_calendar(inputs.coverage, mapped_base)
    features = generate_v2_features(full_calendar_base)
    feature_coverage = build_feature_coverage(full_calendar_base)
    feature_ready_daily = select_feature_ready_rows(features, feature_coverage, v1_columns)
    feature_schema = build_feature_schema(v1_columns)
    v1_structure_comparison = compare_v1_structure(
        v1_feature_table=paths.v1_feature_table,
        v1_columns=v1_columns,
        feature_ready_daily=feature_ready_daily,
    )
    validation = validate_feature_ready_outputs(
        inputs=inputs,
        full_calendar_base=full_calendar_base,
        feature_ready_daily=feature_ready_daily,
        feature_coverage=feature_coverage,
        v1_structure_comparison=v1_structure_comparison,
    )

    paths.output_root.mkdir(parents=True, exist_ok=True)
    output_files = paths.output_files
    checksums = {
        "feature_ready_daily": write_csv(
            output_files["feature_ready_daily"], feature_ready_daily
        ),
        "feature_coverage": write_csv(output_files["feature_coverage"], feature_coverage),
        "feature_schema": write_json(output_files["feature_schema"], feature_schema),
        "v1_structure_comparison": write_json(
            output_files["v1_structure_comparison"], v1_structure_comparison
        ),
        "validation": write_json(output_files["validation"], validation),
    }
    manifest = build_manifest_payload(
        paths=paths,
        validation=validation,
        checksums=checksums,
        source_manifest=inputs.manifest,
    )
    checksums["manifest"] = write_json(output_files["manifest"], manifest)
    return FeatureBuildResult(
        paths=paths,
        feature_ready_daily=feature_ready_daily,
        feature_schema=feature_schema,
        feature_coverage=feature_coverage,
        v1_structure_comparison=v1_structure_comparison,
        validation=validation,
        manifest=manifest,
        checksums=checksums,
    )


def build_manifest_payload(
    *,
    paths: FeatureBuildPaths,
    validation: Mapping[str, Any],
    checksums: Mapping[str, str],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deterministic manifest metadata for generated feature-ready outputs."""
    output_files = paths.output_files
    source_files = paths.input_files
    return {
        "schema_version": "wind_forecast.feature_ready_manifest.v1",
        "dataset_version": "v2",
        "dataset_role": "feature_ready_daily_ren_era5_land",
        "transformation_version": TRANSFORMATION_VERSION,
        "source_transformation_version": SOURCE_TRANSFORMATION_VERSION,
        "source_dataset_role": source_manifest.get("dataset_role"),
        "canonical_daily_key": "Europe/Lisbon civil date",
        "source_paths": {
            "input_root": str(paths.input_root),
            "v1_feature_table": str(paths.v1_feature_table),
        },
        "source_files": {key: str(path) for key, path in source_files.items()},
        "source_sha256_checksums": {
            str(path): sha256_file(path) for path in source_files.values()
        }
        | {str(paths.v1_feature_table): sha256_file(paths.v1_feature_table)},
        "output_files": {key: str(path) for key, path in output_files.items()},
        "sha256_checksums": {
            str(output_files[key]): checksum
            for key, checksum in checksums.items()
            if key in output_files
        },
        "row_counts": dict(validation.get("actual_counts") or {}),
        "feature_coverage_status_counts": dict(
            validation.get("feature_coverage_status_counts") or {}
        ),
        "first_feature_ready_date": validation.get("first_feature_ready_date"),
        "last_feature_ready_date": validation.get("last_feature_ready_date"),
        "status": validation.get("verdict"),
        "warnings": list(validation.get("warnings") or []),
        "failures": list(validation.get("failures") or []),
        "safeguards": dict(validation.get("safeguards") or {}),
    }


def run_synthetic_feature_checks() -> dict[str, Any]:
    """Run small in-memory feature eligibility checks."""
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    coverage = pd.DataFrame(
        {
            DATE_LOCAL_COLUMN: dates.strftime("%Y-%m-%d"),
            "coverage_status": "integration-ready",
            "ren_status": "complete",
            "era5_status": "complete",
            "integration_ready": True,
        }
    )
    daily = pd.DataFrame(
        {
            DATE_LOCAL_COLUMN: dates.strftime("%Y-%m-%d"),
            TARGET_COLUMN: np.arange(100.0, 125.0),
            "wind_speed_m_s_mean": np.linspace(4.0, 6.4, len(dates)),
            "temperature_2m_c_mean": np.linspace(10.0, 16.0, len(dates)),
            "vector_mean_wind_direction_deg_from": np.linspace(180.0, 240.0, len(dates)),
        }
    )
    complete_base = map_integrated_base_columns(daily)
    complete_calendar = reindex_full_local_calendar(coverage, complete_base)
    complete_features = generate_v2_features(complete_calendar)
    complete_feature_coverage = build_feature_coverage(complete_calendar)
    expected_columns = complete_features.columns.tolist()
    complete_ready = select_feature_ready_rows(
        complete_features, complete_feature_coverage, expected_columns
    )

    gapped_daily = daily.copy()
    gapped_daily.loc[gapped_daily[DATE_LOCAL_COLUMN].eq("2026-01-10"), TARGET_COLUMN] = math.nan
    gapped_base = map_integrated_base_columns(gapped_daily)
    gapped_coverage = build_feature_coverage(
        reindex_full_local_calendar(coverage, gapped_base)
    )

    direction_daily = daily.copy()
    direction_daily.loc[
        direction_daily[DATE_LOCAL_COLUMN].eq("2026-01-17"),
        "vector_mean_wind_direction_deg_from",
    ] = math.nan
    direction_coverage = build_feature_coverage(
        reindex_full_local_calendar(coverage, map_integrated_base_columns(direction_daily))
    )
    direction_status_by_date = dict(
        zip(
            direction_coverage[DATE_LOCAL_COLUMN],
            direction_coverage["feature_coverage_status"],
            strict=True,
        )
    )

    unavailable_coverage = coverage.copy()
    unavailable_coverage.loc[
        unavailable_coverage[DATE_LOCAL_COLUMN].eq("2026-01-20"),
        ["coverage_status", "ren_status", "integration_ready"],
    ] = ["excluded-downstream-ren-unavailable", "unavailable", False]
    unavailable_status = build_feature_coverage(
        reindex_full_local_calendar(unavailable_coverage, complete_base)
    )

    checks = {
        "base_mapping_preserves_v1_names": complete_base.columns.tolist()
        == [
            DATE_COLUMN,
            TARGET_COLUMN,
            AVG_WIND_SPEED_COLUMN,
            AVG_TEMPERATURE_COLUMN,
            AVG_WIND_DIRECTION_COLUMN,
        ],
        "complete_calendar_first_ready_after_14_prior_days": (
            complete_feature_coverage.loc[
                complete_feature_coverage["feature_coverage_status"].eq(
                    FEATURE_READY_STATUS
                ),
                DATE_LOCAL_COLUMN,
            ].iloc[0]
            == "2026-01-15"
        ),
        "complete_calendar_ready_row_count": int(len(complete_ready)) == 11,
        "complete_ready_has_no_nans_without_fill": not bool(complete_ready.isna().any().any()),
        "prior_gap_excludes_following_rows": (
            gapped_coverage.loc[
                gapped_coverage[DATE_LOCAL_COLUMN].eq("2026-01-24"),
                "feature_coverage_status",
            ].iloc[0]
            == EXCLUDED_GAP_PRIOR_14_STATUS
        ),
        "direction_lags_use_v1_lag_dates_1_2_3_7": (
            direction_status_by_date["2026-01-18"] == EXCLUDED_DIRECTION_PRIOR_7_STATUS
            and direction_status_by_date["2026-01-19"] == EXCLUDED_DIRECTION_PRIOR_7_STATUS
            and direction_status_by_date["2026-01-20"] == EXCLUDED_DIRECTION_PRIOR_7_STATUS
            and direction_status_by_date["2026-01-21"] == FEATURE_READY_STATUS
            and direction_status_by_date["2026-01-24"] == EXCLUDED_DIRECTION_PRIOR_7_STATUS
        ),
        "ren_unavailable_precedence": (
            unavailable_status.loc[
                unavailable_status[DATE_LOCAL_COLUMN].eq("2026-01-20"),
                "feature_coverage_status",
            ].iloc[0]
            == EXCLUDED_REN_UNAVAILABLE_STATUS
        ),
    }
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
        json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FeatureBuildError(f"JSON file is invalid: {path}.") from exc
    if not isinstance(payload, dict):
        raise FeatureBuildError(f"JSON file must contain an object: {path}.")
    return payload


def _parse_date_series(series: pd.Series, column_name: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if values.isna().any():
        examples = series[values.isna()].astype(str).head(10).tolist()
        raise FeatureBuildError(
            f"{column_name} contains unparseable dates: {examples}."
        )
    return values.dt.normalize()


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.casefold()
    return normalized.isin({"true", "1", "yes"})


def _finite_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.isfinite(values)


def _prior_complete_day_count(finite_mask: np.ndarray, *, window: int) -> np.ndarray:
    counts = np.zeros(len(finite_mask), dtype=int)
    for index in range(len(finite_mask)):
        start = max(0, index - window)
        counts[index] = int(finite_mask[start:index].sum())
    return counts


def _lag_finite_mask(finite_mask: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    result = np.ones(len(finite_mask), dtype=bool)
    for lag in lags:
        lagged = np.zeros(len(finite_mask), dtype=bool)
        if lag < len(finite_mask):
            lagged[lag:] = finite_mask[:-lag]
        result &= lagged
    return result


def _feature_column_role(column: str) -> str:
    if column == DATE_COLUMN:
        return "date_key"
    if column == TARGET_COLUMN:
        return "target"
    if column in {
        AVG_WIND_SPEED_COLUMN,
        AVG_TEMPERATURE_COLUMN,
        AVG_WIND_DIRECTION_COLUMN,
    }:
        return "current_weather_or_target_base"
    if "_Lag" in column:
        return "lag_feature"
    if "_Rolling_" in column:
        return "rolling_feature"
    if column.endswith("_Sin") or column.endswith("_Cos"):
        return "cyclical_feature"
    return "calendar_feature"


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


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
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes, bool, type(None))) else False:
        return None
    return value


__all__ = [
    "FeatureBuildError",
    "FeatureBuildPaths",
    "FeatureBuildResult",
    "IntegratedInputs",
    "build_feature_coverage",
    "build_feature_ready_v2_dataset",
    "build_feature_schema",
    "compare_v1_structure",
    "generate_v2_features",
    "load_integrated_inputs",
    "load_v1_feature_columns",
    "map_integrated_base_columns",
    "reindex_full_local_calendar",
    "resolve_feature_build_paths",
    "run_synthetic_feature_checks",
    "select_feature_ready_rows",
    "validate_feature_ready_outputs",
]
