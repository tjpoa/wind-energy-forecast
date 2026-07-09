"""Formal validation for the feature-ready REN + ERA5-Land v2 dataset."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wind_forecast.features import ROLLING_WINDOWS, WEATHER_LAGS, WIND_PRODUCTION_LAGS
from wind_forecast.schemas import (
    AVG_TEMPERATURE_COLUMN,
    AVG_WIND_DIRECTION_COLUMN,
    AVG_WIND_SPEED_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
)
from wind_forecast.v2_features import (
    BASE_COLUMN_MAPPING,
    DATE_LOCAL_COLUMN,
    EXCLUDED_REN_UNAVAILABLE_STATUS,
    FEATURE_COVERAGE_COLUMNS,
    FEATURE_READY_STATUS,
    SOURCE_TRANSFORMATION_VERSION,
    TRANSFORMATION_VERSION,
    build_feature_coverage,
    build_feature_schema,
    compare_v1_structure,
    generate_v2_features,
    load_v1_feature_columns,
    map_integrated_base_columns,
    reindex_full_local_calendar,
    select_feature_ready_rows,
)

from .common import ValidationReport, ValidationSeverity


VALIDATOR_NAME = "feature_ready_v2_dataset_validator"
VALIDATOR_VERSION = "feature_ready_v2_validation_2B.1"
DATASET_NAME = "feature_ready_ren_era5_land_v2"
DATASET_ROLE = "feature_ready_daily_ren_era5_land"
DATASET_VERSION = "v2"
SCHEMA_VERSION = "wind_forecast.feature_ready_validation_report.v1"
FEATURE_SCHEMA_VERSION = "wind_forecast.feature_ready_schema.v1"
FEATURE_VALIDATION_VERSION = "wind_forecast.feature_ready_validation.v1"
FEATURE_MANIFEST_VERSION = "wind_forecast.feature_ready_manifest.v1"
V1_COMPARISON_VERSION = "wind_forecast.v1_structure_comparison.v1"
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-9

REQUIRED_FEATURE_FILES = {
    "feature_ready_daily.csv",
    "feature_coverage.csv",
    "feature_schema.json",
    "manifest.json",
    "v1_structure_comparison.json",
    "validation.json",
}

EXPECTED_REN_UNAVAILABLE_DATES = (
    "2014-05-03",
    "2016-02-03",
    "2016-02-04",
    "2021-10-03",
    "2023-08-30",
    "2025-08-02",
)


@dataclass(frozen=True)
class FeatureReadyPayloads:
    """In-memory payloads used by the feature-ready validator."""

    feature_ready_daily: pd.DataFrame
    feature_coverage: pd.DataFrame
    feature_schema: Mapping[str, Any]
    manifest: Mapping[str, Any]
    validation_payload: Mapping[str, Any]
    v1_structure_comparison: Mapping[str, Any]
    integrated_daily: pd.DataFrame
    integrated_coverage: pd.DataFrame
    integrated_validation: Mapping[str, Any]
    integrated_manifest: Mapping[str, Any]
    v1_columns: Sequence[str]
    raw_feature_columns: Sequence[str] | None = None
    raw_coverage_columns: Sequence[str] | None = None


def validate_feature_ready_v2_dataset(
    *,
    feature_root: str | Path,
    integrated_root: str | Path,
    v1_feature_table: str | Path,
    expected_ren_unavailable_dates: Sequence[str] = EXPECTED_REN_UNAVAILABLE_DATES,
) -> ValidationReport:
    """Validate the accepted feature-ready v2 dataset from disk without writing files."""
    feature_root = Path(feature_root)
    integrated_root = Path(integrated_root)
    v1_feature_table = Path(v1_feature_table)
    report = ValidationReport(dataset_name=DATASET_NAME)
    _set_base_stats(
        report,
        feature_root=feature_root,
        integrated_root=integrated_root,
        v1_feature_table=v1_feature_table,
    )

    _validate_feature_root_files(report, feature_root)
    paths = {name: feature_root / name for name in REQUIRED_FEATURE_FILES}
    if report.has_errors:
        return _finalize_report(report)

    try:
        feature_daily, raw_feature_columns = _read_csv_with_raw_header(
            paths["feature_ready_daily.csv"]
        )
        feature_coverage, raw_coverage_columns = _read_csv_with_raw_header(
            paths["feature_coverage.csv"]
        )
        payloads = FeatureReadyPayloads(
            feature_ready_daily=feature_daily,
            feature_coverage=feature_coverage,
            feature_schema=_read_json(paths["feature_schema.json"]),
            manifest=_read_json(paths["manifest.json"]),
            validation_payload=_read_json(paths["validation.json"]),
            v1_structure_comparison=_read_json(paths["v1_structure_comparison.json"]),
            integrated_daily=pd.read_csv(integrated_root / "daily_merged.csv"),
            integrated_coverage=pd.read_csv(integrated_root / "coverage.csv"),
            integrated_validation=_read_json(integrated_root / "validation.json"),
            integrated_manifest=_read_json(integrated_root / "manifest.json"),
            v1_columns=load_v1_feature_columns(v1_feature_table),
            raw_feature_columns=raw_feature_columns,
            raw_coverage_columns=raw_coverage_columns,
        )
    except (FileNotFoundError, json.JSONDecodeError, pd.errors.ParserError, ValueError) as exc:
        report.add_issue(
            ValidationSeverity.ERROR,
            "load_failed",
            f"Could not load feature-ready validation inputs: {exc}",
        )
        return _finalize_report(report)

    return validate_feature_ready_frames(
        payloads=payloads,
        feature_root=feature_root,
        integrated_root=integrated_root,
        v1_feature_table=v1_feature_table,
        report=report,
        expected_ren_unavailable_dates=expected_ren_unavailable_dates,
    )


def validate_feature_ready_frames(
    *,
    payloads: FeatureReadyPayloads,
    feature_root: str | Path | None = None,
    integrated_root: str | Path | None = None,
    v1_feature_table: str | Path | None = None,
    report: ValidationReport | None = None,
    expected_ren_unavailable_dates: Sequence[str] = EXPECTED_REN_UNAVAILABLE_DATES,
) -> ValidationReport:
    """Validate loaded feature-ready payloads without mutating input DataFrames."""
    report = report or ValidationReport(dataset_name=DATASET_NAME)
    if feature_root is not None or integrated_root is not None or v1_feature_table is not None:
        _set_base_stats(
            report,
            feature_root=Path(feature_root) if feature_root is not None else None,
            integrated_root=Path(integrated_root) if integrated_root is not None else None,
            v1_feature_table=Path(v1_feature_table) if v1_feature_table is not None else None,
        )

    feature_daily = payloads.feature_ready_daily
    feature_coverage = payloads.feature_coverage
    report.row_count = int(len(feature_daily))
    report.column_count = int(len(feature_daily.columns))
    report.stats["feature_ready_row_count"] = int(len(feature_daily))
    report.stats["feature_ready_column_count"] = int(len(feature_daily.columns))
    report.stats["feature_coverage_row_count"] = int(len(feature_coverage))
    report.stats["v1_feature_column_count"] = int(len(payloads.v1_columns))
    report.stats["scaler_fitting_decision"] = (
        "not_performed; existing v1 scalers are not claimed valid for v2"
    )

    _validate_raw_headers(report, payloads.raw_feature_columns, "feature")
    _validate_raw_headers(report, payloads.raw_coverage_columns, "coverage")
    _validate_payload_metadata(report, payloads, feature_root, integrated_root, v1_feature_table)
    _validate_feature_schema_and_columns(report, payloads)
    feature_dates = _validate_feature_dates(report, feature_daily)
    coverage_dates = _validate_feature_coverage(report, payloads, feature_dates)
    _validate_required_ren_unavailable_dates(
        report,
        feature_dates,
        coverage_dates,
        feature_coverage,
        expected_ren_unavailable_dates,
    )
    _validate_numeric_values(report, feature_daily)
    _validate_domains(report, feature_daily)
    _validate_recomputed_features(report, payloads)
    _add_inherited_warnings(report, payloads)
    return _finalize_report(report)


def serialize_validation_report(report: ValidationReport) -> str:
    """Serialize a validation report as deterministic JSON."""
    return json.dumps(report.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def run_synthetic_feature_ready_validation_checks() -> dict[str, Any]:
    """Run focused synthetic checks for feature-ready validation behavior."""
    valid_payloads = _build_synthetic_payloads()
    originals = {
        "feature_ready_daily": valid_payloads.feature_ready_daily.copy(deep=True),
        "feature_coverage": valid_payloads.feature_coverage.copy(deep=True),
        "integrated_daily": valid_payloads.integrated_daily.copy(deep=True),
        "integrated_coverage": valid_payloads.integrated_coverage.copy(deep=True),
    }
    valid_report = validate_feature_ready_frames(
        payloads=valid_payloads,
        expected_ren_unavailable_dates=("2026-01-10",),
    )

    checks: dict[str, bool] = {
        "valid_dataset_passes_with_expected_warning": (
            valid_report.passed and valid_report.has_warnings
        ),
        "input_dataframes_are_not_mutated": (
            valid_payloads.feature_ready_daily.equals(originals["feature_ready_daily"])
            and valid_payloads.feature_coverage.equals(originals["feature_coverage"])
            and valid_payloads.integrated_daily.equals(originals["integrated_daily"])
            and valid_payloads.integrated_coverage.equals(originals["integrated_coverage"])
        ),
        "deterministic_serialization": (
            serialize_validation_report(valid_report) == serialize_validation_report(valid_report)
        ),
        "raise_for_errors_pass_path": _raise_for_errors_passes(valid_report),
    }

    scenarios = {
        "missing_feature_column": _mutate_payload(
            valid_payloads,
            feature_ready_daily=valid_payloads.feature_ready_daily.drop(
                columns=[AVG_WIND_SPEED_COLUMN]
            ),
        ),
        "additional_feature_column": _mutate_payload(
            valid_payloads,
            feature_ready_daily=valid_payloads.feature_ready_daily.assign(unexpected_feature=1),
        ),
        "reordered_feature_column": _mutate_payload(
            valid_payloads,
            feature_ready_daily=valid_payloads.feature_ready_daily[
                [TARGET_COLUMN, DATE_COLUMN]
                + [
                    column
                    for column in valid_payloads.feature_ready_daily.columns
                    if column not in {DATE_COLUMN, TARGET_COLUMN}
                ]
            ],
        ),
        "duplicate_feature_header": _mutate_payload(
            valid_payloads,
            raw_feature_columns=[DATE_COLUMN, DATE_COLUMN]
            + list(valid_payloads.feature_ready_daily.columns[2:]),
        ),
        "duplicate_date": _mutate_payload(
            valid_payloads,
            feature_ready_daily=_with_value(
                valid_payloads.feature_ready_daily, 1, DATE_COLUMN, "2026-01-15"
            ),
        ),
        "unsorted_date": _mutate_payload(
            valid_payloads,
            feature_ready_daily=valid_payloads.feature_ready_daily.iloc[::-1].reset_index(
                drop=True
            ),
        ),
        "non_finite_value": _mutate_payload(
            valid_payloads,
            feature_ready_daily=_with_value(
                valid_payloads.feature_ready_daily, 0, AVG_WIND_SPEED_COLUMN, np.inf
            ),
        ),
        "invalid_numeric_type": _mutate_payload(
            valid_payloads,
            feature_ready_daily=_with_value(
                valid_payloads.feature_ready_daily, 0, TARGET_COLUMN, "not-a-number"
            ),
        ),
        "row_count_mismatch": _mutate_payload(
            valid_payloads,
            validation_payload={
                **dict(valid_payloads.validation_payload),
                "actual_counts": {
                    **dict(valid_payloads.validation_payload["actual_counts"]),
                    "feature_ready_rows": 999,
                },
            },
        ),
        "source_lineage_mismatch": _mutate_payload(
            valid_payloads,
            manifest={
                **dict(valid_payloads.manifest),
                "source_transformation_version": "unexpected-source-version",
            },
        ),
        "extra_manifest_output_entry": _mutate_payload(
            valid_payloads,
            manifest={
                **dict(valid_payloads.manifest),
                "output_files": {
                    **dict(valid_payloads.manifest["output_files"]),
                    "unexpected_output": "synthetic_feature/unexpected.csv",
                },
            },
        ),
        "extra_manifest_output_checksum": _mutate_payload(
            valid_payloads,
            manifest={
                **dict(valid_payloads.manifest),
                "sha256_checksums": {
                    **dict(valid_payloads.manifest["sha256_checksums"]),
                    "synthetic_feature/unexpected.csv": "0" * 64,
                },
            },
        ),
        "extra_manifest_source_checksum": _mutate_payload(
            valid_payloads,
            manifest={
                **dict(valid_payloads.manifest),
                "source_sha256_checksums": {
                    **dict(valid_payloads.manifest["source_sha256_checksums"]),
                    "synthetic_integrated/unapproved_source.csv": "0" * 64,
                },
            },
        ),
        "undocumented_missing_date": _mutate_payload(
            valid_payloads,
            feature_ready_daily=valid_payloads.feature_ready_daily.iloc[1:].reset_index(
                drop=True
            ),
        ),
        "unavailable_date_included": _mutate_payload(
            valid_payloads,
            feature_ready_daily=pd.concat(
                [
                    valid_payloads.feature_ready_daily,
                    valid_payloads.feature_ready_daily.iloc[[0]].assign(
                        **{DATE_COLUMN: "2026-01-10"}
                    ),
                ],
                ignore_index=True,
            ).sort_values(DATE_COLUMN).reset_index(drop=True),
        ),
        "bad_lag_value": _mutate_payload(
            valid_payloads,
            feature_ready_daily=_with_value(
                valid_payloads.feature_ready_daily,
                0,
                "Wind_Production_Lag1",
                -999.0,
            ),
        ),
        "rolling_leakage": _mutate_payload(
            valid_payloads,
            feature_ready_daily=_with_value(
                valid_payloads.feature_ready_daily,
                0,
                "Wind_Production_Rolling_Mean_3",
                valid_payloads.feature_ready_daily.loc[0, TARGET_COLUMN],
            ),
        ),
        "malformed_exclusion_lineage": _mutate_payload(
            valid_payloads,
            feature_coverage=_with_value(
                valid_payloads.feature_coverage,
                int(
                    valid_payloads.feature_coverage[
                        valid_payloads.feature_coverage[DATE_LOCAL_COLUMN].eq("2026-01-10")
                    ].index[0]
                ),
                "feature_coverage_status",
                FEATURE_READY_STATUS,
            ),
        ),
    }

    for name, payloads in scenarios.items():
        scenario_report = validate_feature_ready_frames(
            payloads=payloads,
            expected_ren_unavailable_dates=("2026-01-10",),
        )
        checks[f"{name}_fails"] = scenario_report.has_errors

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_result = _write_synthetic_payloads(valid_payloads, Path(temp_dir))
        file_report = validate_feature_ready_v2_dataset(
            feature_root=temp_result["feature_root"],
            integrated_root=temp_result["integrated_root"],
            v1_feature_table=temp_result["v1_feature_table"],
            expected_ren_unavailable_dates=("2026-01-10",),
        )
        checks["temporary_file_fixture_passes"] = file_report.passed

        bad_manifest = Path(temp_result["feature_root"]) / "manifest.json"
        manifest = _read_json(bad_manifest)
        manifest["sha256_checksums"][
            str(Path(temp_result["feature_root"]) / "feature_ready_daily.csv")
        ] = "0" * 64
        bad_manifest.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_report = validate_feature_ready_v2_dataset(
            feature_root=temp_result["feature_root"],
            integrated_root=temp_result["integrated_root"],
            v1_feature_table=temp_result["v1_feature_table"],
            expected_ren_unavailable_dates=("2026-01-10",),
        )
        checks["checksum_mismatch_fails"] = checksum_report.has_errors

    error_report = validate_feature_ready_frames(
        payloads=scenarios["bad_lag_value"],
        expected_ren_unavailable_dates=("2026-01-10",),
    )
    checks["raise_for_errors_fail_path"] = _raise_for_errors_fails(error_report)

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "example_summary": valid_report.format_summary(),
        "example_error_message": _example_error_message(error_report),
    }


def _validate_feature_root_files(report: ValidationReport, feature_root: Path) -> None:
    if not feature_root.is_dir():
        report.add_issue(
            ValidationSeverity.ERROR,
            "missing_feature_root",
            f"Feature root does not exist or is not a directory: {feature_root}",
        )
        return
    actual = sorted(path.name for path in feature_root.iterdir() if path.is_file())
    actual_set = set(actual)
    missing = sorted(REQUIRED_FEATURE_FILES - actual_set)
    unexpected = sorted(actual_set - REQUIRED_FEATURE_FILES)
    report.stats["required_files"] = sorted(REQUIRED_FEATURE_FILES)
    report.stats["actual_files"] = actual
    if missing:
        report.add_issue(
            ValidationSeverity.ERROR,
            "missing_required_file",
            "Feature-ready root is missing required files.",
            count=len(missing),
            sample=missing,
        )
    if unexpected:
        report.add_issue(
            ValidationSeverity.ERROR,
            "unexpected_feature_file",
            "Feature-ready root contains unexpected files.",
            count=len(unexpected),
            sample=unexpected,
        )


def _validate_raw_headers(
    report: ValidationReport,
    raw_columns: Sequence[str] | None,
    label: str,
) -> None:
    if raw_columns is None:
        return
    duplicates = _duplicates(raw_columns)
    if duplicates:
        report.add_issue(
            ValidationSeverity.ERROR,
            f"duplicate_{label}_column_header",
            f"{label.capitalize()} CSV header contains duplicate column names.",
            count=len(duplicates),
            sample=duplicates[:5],
        )


def _validate_payload_metadata(
    report: ValidationReport,
    payloads: FeatureReadyPayloads,
    feature_root: str | Path | None,
    integrated_root: str | Path | None,
    v1_feature_table: str | Path | None,
) -> None:
    schema = payloads.feature_schema
    manifest = payloads.manifest
    validation = payloads.validation_payload
    comparison = payloads.v1_structure_comparison
    integrated_manifest = payloads.integrated_manifest
    integrated_validation = payloads.integrated_validation

    _expect_equal(report, schema.get("schema_version"), FEATURE_SCHEMA_VERSION, "schema_version")
    _expect_equal(report, schema.get("dataset_role"), DATASET_ROLE, "schema_dataset_role")
    _expect_equal(report, schema.get("dataset_version"), DATASET_VERSION, "schema_dataset_version")
    _expect_equal(
        report, schema.get("transformation_version"), TRANSFORMATION_VERSION, "schema_transformation_version"
    )
    _expect_equal(
        report,
        schema.get("formula_source"),
        "wind_forecast.features.apply_feature_engineering",
        "schema_formula_source",
    )
    fill_policy = dict(schema.get("fill_policy") or {})
    for key in ("handle_final_nans_called", "backfill", "forward_fill", "interpolation", "fill_zero"):
        if fill_policy.get(key) is not False:
            report.add_issue(
                ValidationSeverity.ERROR,
                "unsafe_fill_policy",
                f"Feature schema fill policy '{key}' must be false.",
                sample=fill_policy,
            )

    _expect_equal(report, manifest.get("schema_version"), FEATURE_MANIFEST_VERSION, "manifest_schema_version")
    _expect_equal(report, manifest.get("dataset_role"), DATASET_ROLE, "manifest_dataset_role")
    _expect_equal(report, manifest.get("dataset_version"), DATASET_VERSION, "manifest_dataset_version")
    _expect_equal(
        report,
        manifest.get("source_transformation_version"),
        SOURCE_TRANSFORMATION_VERSION,
        "manifest_source_transformation_version",
    )
    _expect_equal(
        report,
        manifest.get("transformation_version"),
        TRANSFORMATION_VERSION,
        "manifest_transformation_version",
    )

    _expect_equal(
        report, validation.get("schema_version"), FEATURE_VALIDATION_VERSION, "validation_schema_version"
    )
    _expect_equal(report, validation.get("dataset_role"), DATASET_ROLE, "validation_dataset_role")
    _expect_equal(report, validation.get("dataset_version"), DATASET_VERSION, "validation_dataset_version")
    _expect_equal(
        report,
        validation.get("source_transformation_version"),
        SOURCE_TRANSFORMATION_VERSION,
        "validation_source_transformation_version",
    )
    _expect_equal(
        report,
        validation.get("transformation_version"),
        TRANSFORMATION_VERSION,
        "validation_transformation_version",
    )
    if not bool(validation.get("passed")):
        report.add_issue(
            ValidationSeverity.ERROR,
            "embedded_validation_failed",
            "Embedded feature-ready validation payload did not pass.",
            sample=validation.get("failures"),
        )

    _expect_equal(
        report,
        comparison.get("schema_version"),
        V1_COMPARISON_VERSION,
        "v1_structure_comparison_schema_version",
    )
    if comparison.get("exact_column_order_match") is not True:
        report.add_issue(
            ValidationSeverity.ERROR,
            "embedded_v1_order_mismatch",
            "Embedded v1 structure comparison does not confirm exact column order.",
            sample=comparison,
        )

    row_counts = dict(validation.get("actual_counts") or {})
    manifest_row_counts = dict(manifest.get("row_counts") or {})
    _expect_equal(report, manifest_row_counts, row_counts, "manifest_validation_row_counts")
    _expect_equal(
        report,
        manifest.get("feature_coverage_status_counts"),
        validation.get("feature_coverage_status_counts"),
        "manifest_validation_status_counts",
    )
    _expect_equal(report, manifest.get("status"), validation.get("verdict"), "manifest_validation_status")

    _expect_equal(
        report,
        integrated_manifest.get("transformation_version"),
        SOURCE_TRANSFORMATION_VERSION,
        "integrated_manifest_transformation_version",
    )
    if not bool(integrated_validation.get("passed")):
        report.add_issue(
            ValidationSeverity.ERROR,
            "source_integrated_validation_failed",
            "Accepted integrated validation payload did not pass.",
            sample=integrated_validation.get("failures"),
        )

    _validate_manifest_paths(
        report,
        manifest,
        feature_root=Path(feature_root) if feature_root is not None else None,
        integrated_root=Path(integrated_root) if integrated_root is not None else None,
        v1_feature_table=Path(v1_feature_table) if v1_feature_table is not None else None,
    )
    _validate_manifest_checksums(
        report,
        manifest,
        feature_root=Path(feature_root) if feature_root is not None else None,
        integrated_root=Path(integrated_root) if integrated_root is not None else None,
        v1_feature_table=Path(v1_feature_table) if v1_feature_table is not None else None,
    )


def _validate_manifest_paths(
    report: ValidationReport,
    manifest: Mapping[str, Any],
    *,
    feature_root: Path | None,
    integrated_root: Path | None,
    v1_feature_table: Path | None,
) -> None:
    output_files = dict(manifest.get("output_files") or {})
    expected_names = {
        "feature_ready_daily": "feature_ready_daily.csv",
        "feature_coverage": "feature_coverage.csv",
        "feature_schema": "feature_schema.json",
        "v1_structure_comparison": "v1_structure_comparison.json",
        "validation": "validation.json",
        "manifest": "manifest.json",
    }
    unexpected_keys = sorted(set(output_files).difference(expected_names))
    if unexpected_keys:
        report.add_issue(
            ValidationSeverity.ERROR,
            "unexpected_manifest_output_file",
            "Manifest output_files contains entries outside the approved output allowlist.",
            count=len(unexpected_keys),
            sample=unexpected_keys,
        )
    if feature_root is not None:
        for key, filename in expected_names.items():
            path_text = output_files.get(key)
            if path_text is None or not _path_suffix_matches(path_text, feature_root / filename):
                report.add_issue(
                    ValidationSeverity.ERROR,
                    "manifest_output_path_mismatch",
                    f"Manifest output path for '{key}' does not match expected filename.",
                    sample={"actual": path_text, "expected": str(feature_root / filename)},
                )
    source_paths = dict(manifest.get("source_paths") or {})
    if integrated_root is not None and not _path_suffix_matches(
        source_paths.get("input_root"), integrated_root
    ):
        report.add_issue(
            ValidationSeverity.ERROR,
            "manifest_input_root_mismatch",
            "Manifest input_root does not match the validation integrated root.",
            sample={"actual": source_paths.get("input_root"), "expected": str(integrated_root)},
        )
    if v1_feature_table is not None and not _path_suffix_matches(
        source_paths.get("v1_feature_table"), v1_feature_table
    ):
        report.add_issue(
            ValidationSeverity.ERROR,
            "manifest_v1_feature_table_mismatch",
            "Manifest v1 feature-table path does not match the validation input.",
            sample={"actual": source_paths.get("v1_feature_table"), "expected": str(v1_feature_table)},
        )


def _validate_manifest_checksums(
    report: ValidationReport,
    manifest: Mapping[str, Any],
    *,
    feature_root: Path | None,
    integrated_root: Path | None,
    v1_feature_table: Path | None,
) -> None:
    checksum_paths: dict[str, Path] = {}
    if feature_root is not None:
        checksum_paths.update(
            {
                str(feature_root / "feature_ready_daily.csv"): feature_root / "feature_ready_daily.csv",
                str(feature_root / "feature_coverage.csv"): feature_root / "feature_coverage.csv",
                str(feature_root / "feature_schema.json"): feature_root / "feature_schema.json",
                str(feature_root / "v1_structure_comparison.json"): feature_root
                / "v1_structure_comparison.json",
                str(feature_root / "validation.json"): feature_root / "validation.json",
            }
        )
    _validate_checksum_group(
        report,
        dict(manifest.get("sha256_checksums") or {}),
        checksum_paths,
        code_prefix="output",
    )

    source_paths: dict[str, Path] = {}
    if integrated_root is not None:
        for filename in ("daily_merged.csv", "coverage.csv", "validation.json", "manifest.json"):
            source_paths[str(integrated_root / filename)] = integrated_root / filename
    if v1_feature_table is not None:
        source_paths[str(v1_feature_table)] = v1_feature_table
    _validate_checksum_group(
        report,
        dict(manifest.get("source_sha256_checksums") or {}),
        source_paths,
        code_prefix="source",
    )


def _validate_checksum_group(
    report: ValidationReport,
    recorded: Mapping[str, str],
    expected_paths: Mapping[str, Path],
    *,
    code_prefix: str,
) -> None:
    report.stats[f"{code_prefix}_checksum_record_count"] = len(recorded)
    unexpected_keys = [
        str(key)
        for key in sorted(recorded)
        if not any(_path_suffix_matches(key, expected_path) for expected_path in expected_paths.values())
    ]
    if unexpected_keys:
        report.add_issue(
            ValidationSeverity.ERROR,
            f"unexpected_{code_prefix}_checksum",
            f"Manifest contains {code_prefix} checksum records outside the approved file allowlist.",
            count=len(unexpected_keys),
            sample=unexpected_keys[:5],
        )
    for expected_text, path in expected_paths.items():
        recorded_checksum = _lookup_checksum(recorded, path)
        if recorded_checksum is None:
            report.add_issue(
                ValidationSeverity.ERROR,
                f"missing_{code_prefix}_checksum",
                f"Manifest does not record a checksum for {path}.",
                sample=str(path),
            )
            continue
        if not path.is_file():
            report.add_issue(
                ValidationSeverity.ERROR,
                f"missing_{code_prefix}_checksum_file",
                f"Checksum target file does not exist: {path}.",
                sample=str(path),
            )
            continue
        actual_checksum = sha256_file(path)
        if actual_checksum != recorded_checksum:
            report.add_issue(
                ValidationSeverity.ERROR,
                f"{code_prefix}_checksum_mismatch",
                f"Recorded checksum does not match {path.name}.",
                sample={
                    "path": str(path),
                    "recorded": recorded_checksum,
                    "actual": actual_checksum,
                },
            )


def _validate_feature_schema_and_columns(
    report: ValidationReport,
    payloads: FeatureReadyPayloads,
) -> None:
    feature_columns = [str(column) for column in payloads.feature_ready_daily.columns]
    v1_columns = [str(column) for column in payloads.v1_columns]
    schema_columns_payload = list(payloads.feature_schema.get("columns") or [])
    schema_columns = [str(item.get("name")) for item in schema_columns_payload]
    schema_positions = [item.get("position") for item in schema_columns_payload]

    report.stats["feature_columns"] = feature_columns
    report.stats["feature_column_count"] = len(feature_columns)
    report.stats["schema_column_count"] = int(payloads.feature_schema.get("column_count") or 0)
    report.stats["v1_feature_column_count"] = len(v1_columns)

    if _duplicates(feature_columns):
        report.add_issue(
            ValidationSeverity.ERROR,
            "duplicate_feature_column",
            "Feature-ready DataFrame contains duplicate columns after loading.",
            sample=_duplicates(feature_columns),
        )
    if feature_columns != v1_columns:
        report.add_issue(
            ValidationSeverity.ERROR,
            "feature_column_order_mismatch",
            "Feature-ready columns do not exactly match the v1 feature order.",
            sample=_column_diff(v1_columns, feature_columns),
        )
    if schema_columns != v1_columns:
        report.add_issue(
            ValidationSeverity.ERROR,
            "schema_column_order_mismatch",
            "Feature schema columns do not exactly match the v1 feature order.",
            sample=_column_diff(v1_columns, schema_columns),
        )
    if schema_positions != list(range(len(schema_columns))):
        report.add_issue(
            ValidationSeverity.ERROR,
            "schema_position_mismatch",
            "Feature schema positions are not the deterministic zero-based sequence.",
            sample=schema_positions[:10],
        )
    if int(payloads.feature_schema.get("column_count") or -1) != len(v1_columns):
        report.add_issue(
            ValidationSeverity.ERROR,
            "schema_column_count_mismatch",
            "Feature schema column count does not match v1 feature count.",
            sample={
                "schema": payloads.feature_schema.get("column_count"),
                "v1": len(v1_columns),
            },
        )
    if dict(payloads.feature_schema.get("base_column_mapping") or {}) != dict(BASE_COLUMN_MAPPING):
        report.add_issue(
            ValidationSeverity.ERROR,
            "base_column_mapping_mismatch",
            "Feature schema base-column mapping does not match the accepted v2 mapping.",
            sample=payloads.feature_schema.get("base_column_mapping"),
        )


def _validate_feature_dates(
    report: ValidationReport,
    feature_daily: pd.DataFrame,
) -> pd.Series:
    if DATE_COLUMN not in feature_daily.columns:
        report.add_issue(
            ValidationSeverity.ERROR,
            "missing_date_column",
            "Feature-ready table is missing the Date column.",
            column=DATE_COLUMN,
        )
        return pd.Series(dtype="datetime64[ns]")

    dates = pd.to_datetime(feature_daily[DATE_COLUMN], errors="coerce")
    invalid_count = int(dates.isna().sum())
    if invalid_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "invalid_feature_date",
            "Feature-ready Date contains null or unparseable values.",
            column=DATE_COLUMN,
            count=invalid_count,
        )
    valid_dates = dates.dropna()
    duplicate_mask = valid_dates.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    report.stats["feature_duplicate_date_count"] = duplicate_count
    if duplicate_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "duplicate_feature_date",
            "Feature-ready Date contains duplicate dates.",
            column=DATE_COLUMN,
            count=duplicate_count,
            sample=_sample(valid_dates[duplicate_mask]),
        )
    if len(valid_dates) > 1 and not valid_dates.is_monotonic_increasing:
        report.add_issue(
            ValidationSeverity.ERROR,
            "non_chronological_feature_date",
            "Feature-ready Date is not sorted chronologically.",
            column=DATE_COLUMN,
            sample=_sample(feature_daily[DATE_COLUMN].head()),
        )
    if not valid_dates.empty:
        report.date_range = (valid_dates.min(), valid_dates.max())
        report.stats["first_feature_ready_date"] = valid_dates.min().strftime("%Y-%m-%d")
        report.stats["last_feature_ready_date"] = valid_dates.max().strftime("%Y-%m-%d")
    return dates


def _validate_feature_coverage(
    report: ValidationReport,
    payloads: FeatureReadyPayloads,
    feature_dates: pd.Series,
) -> pd.Series:
    coverage = payloads.feature_coverage
    coverage_columns = [str(column) for column in coverage.columns]
    if coverage_columns != list(FEATURE_COVERAGE_COLUMNS):
        report.add_issue(
            ValidationSeverity.ERROR,
            "feature_coverage_column_order_mismatch",
            "Feature coverage columns do not match the accepted coverage schema.",
            sample=_column_diff(list(FEATURE_COVERAGE_COLUMNS), coverage_columns),
        )
    if DATE_LOCAL_COLUMN not in coverage:
        report.add_issue(
            ValidationSeverity.ERROR,
            "missing_coverage_date_column",
            "Feature coverage is missing date_local.",
            column=DATE_LOCAL_COLUMN,
        )
        return pd.Series(dtype="datetime64[ns]")

    coverage_dates = pd.to_datetime(coverage[DATE_LOCAL_COLUMN], errors="coerce")
    invalid_count = int(coverage_dates.isna().sum())
    if invalid_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "invalid_coverage_date",
            "Feature coverage contains null or unparseable dates.",
            column=DATE_LOCAL_COLUMN,
            count=invalid_count,
        )
    valid_dates = coverage_dates.dropna()
    if len(valid_dates) > 1 and not valid_dates.is_monotonic_increasing:
        report.add_issue(
            ValidationSeverity.ERROR,
            "non_chronological_coverage_date",
            "Feature coverage dates are not sorted chronologically.",
            column=DATE_LOCAL_COLUMN,
        )
    duplicate_mask = valid_dates.duplicated(keep=False)
    if duplicate_mask.any():
        report.add_issue(
            ValidationSeverity.ERROR,
            "duplicate_coverage_date",
            "Feature coverage contains duplicate local dates.",
            column=DATE_LOCAL_COLUMN,
            count=int(duplicate_mask.sum()),
            sample=_sample(valid_dates[duplicate_mask]),
        )
    if not valid_dates.empty:
        expected_calendar = pd.date_range(valid_dates.min(), valid_dates.max(), freq="D")
        missing_calendar = expected_calendar.difference(pd.DatetimeIndex(valid_dates))
        report.stats["coverage_start_date"] = valid_dates.min().strftime("%Y-%m-%d")
        report.stats["coverage_end_date"] = valid_dates.max().strftime("%Y-%m-%d")
        report.stats["coverage_missing_calendar_day_count"] = int(len(missing_calendar))
        if len(missing_calendar):
            report.add_issue(
                ValidationSeverity.ERROR,
                "coverage_calendar_gap",
                "Feature coverage does not record a contiguous daily calendar.",
                count=int(len(missing_calendar)),
                sample=_sample(missing_calendar),
            )

    status_counts = _value_counts(coverage["feature_coverage_status"])
    report.stats["feature_coverage_status_counts"] = status_counts
    payload_counts = dict(payloads.validation_payload.get("feature_coverage_status_counts") or {})
    manifest_counts = dict(payloads.manifest.get("feature_coverage_status_counts") or {})
    _expect_equal(report, payload_counts, status_counts, "validation_status_count_mismatch")
    _expect_equal(report, manifest_counts, status_counts, "manifest_status_count_mismatch")

    feature_ready_dates = _date_texts(feature_dates)
    coverage_ready_dates = _date_texts(
        coverage_dates[coverage["feature_coverage_status"].astype(str).eq(FEATURE_READY_STATUS)]
    )
    if feature_ready_dates != coverage_ready_dates:
        report.add_issue(
            ValidationSeverity.ERROR,
            "feature_ready_date_mismatch",
            "Feature-ready table dates do not exactly match coverage rows marked feature-ready.",
            sample={
                "missing_from_features": sorted(set(coverage_ready_dates) - set(feature_ready_dates))[:5],
                "unexpected_in_features": sorted(set(feature_ready_dates) - set(coverage_ready_dates))[:5],
            },
        )
    if "feature_ready" in coverage.columns:
        ready_flags = _bool_series(coverage["feature_ready"])
        expected_ready = coverage["feature_coverage_status"].astype(str).eq(FEATURE_READY_STATUS)
        if not ready_flags.equals(expected_ready):
            report.add_issue(
                ValidationSeverity.ERROR,
                "feature_ready_flag_mismatch",
                "Feature coverage boolean flag does not match feature-ready status.",
            )
    if "exclusion_reason" in coverage.columns:
        excluded = coverage["feature_coverage_status"].astype(str).ne(FEATURE_READY_STATUS)
        empty_reason = coverage["exclusion_reason"].fillna("").astype(str).str.len().eq(0)
        missing_reason_count = int((excluded & empty_reason).sum())
        if missing_reason_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "missing_exclusion_reason",
                "Excluded coverage rows must carry an exclusion reason.",
                count=missing_reason_count,
            )
    return coverage_dates


def _validate_required_ren_unavailable_dates(
    report: ValidationReport,
    feature_dates: pd.Series,
    coverage_dates: pd.Series,
    coverage: pd.DataFrame,
    expected_ren_unavailable_dates: Sequence[str],
) -> None:
    feature_date_texts = set(_date_texts(feature_dates))
    coverage_by_date = coverage.copy()
    coverage_by_date["_date_text"] = _date_texts(coverage_dates)
    unavailable_dates = sorted(
        coverage_by_date.loc[
            coverage_by_date["feature_coverage_status"].astype(str).eq(
                EXCLUDED_REN_UNAVAILABLE_STATUS
            ),
            "_date_text",
        ].tolist()
    )
    report.stats["ren_unavailable_dates"] = unavailable_dates
    expected = sorted(str(date) for date in expected_ren_unavailable_dates)
    if unavailable_dates != expected:
        report.add_issue(
            ValidationSeverity.ERROR,
            "ren_unavailable_dates_mismatch",
            "Coverage REN-unavailable dates do not match the accepted lineage.",
            sample={"expected": expected, "actual": unavailable_dates},
        )
    included = sorted(set(expected).intersection(feature_date_texts))
    if included:
        report.add_issue(
            ValidationSeverity.ERROR,
            "ren_unavailable_date_in_features",
            "REN-unavailable dates must be absent from feature-ready rows.",
            count=len(included),
            sample=included,
        )


def _validate_numeric_values(report: ValidationReport, feature_daily: pd.DataFrame) -> None:
    numeric_columns = [column for column in feature_daily.columns if column != DATE_COLUMN]
    non_numeric: list[str] = []
    null_counts: dict[str, int] = {}
    non_finite_counts: dict[str, int] = {}
    for column in numeric_columns:
        series = feature_daily[column]
        if not pd.api.types.is_numeric_dtype(series):
            non_numeric.append(str(column))
        values = pd.to_numeric(series, errors="coerce")
        invalid_count = int(series.isna().sum() + (series.notna() & values.isna()).sum())
        if invalid_count:
            null_counts[str(column)] = invalid_count
            continue
        non_finite_count = int((~np.isfinite(values.to_numpy(dtype=float))).sum())
        if non_finite_count:
            non_finite_counts[str(column)] = non_finite_count

    report.stats["non_numeric_feature_columns"] = non_numeric
    report.stats["null_or_invalid_numeric_counts"] = null_counts
    report.stats["non_finite_numeric_counts"] = non_finite_counts
    if non_numeric:
        report.add_issue(
            ValidationSeverity.ERROR,
            "non_numeric_feature_column",
            "All non-Date feature columns must load with numeric dtypes.",
            count=len(non_numeric),
            sample=non_numeric[:5],
        )
    if null_counts:
        report.add_issue(
            ValidationSeverity.ERROR,
            "null_or_invalid_numeric_feature",
            "Feature-ready numeric columns contain null or non-numeric values.",
            count=sum(null_counts.values()),
            sample=dict(list(null_counts.items())[:5]),
        )
    if non_finite_counts:
        report.add_issue(
            ValidationSeverity.ERROR,
            "non_finite_numeric_feature",
            "Feature-ready numeric columns contain non-finite values.",
            count=sum(non_finite_counts.values()),
            sample=dict(list(non_finite_counts.items())[:5]),
        )

    for column in [TARGET_COLUMN] + [
        column for column in numeric_columns if "Wind_Speed" in str(column)
    ]:
        if column not in feature_daily.columns:
            continue
        values = pd.to_numeric(feature_daily[column], errors="coerce")
        negative_count = int((values < 0).sum())
        if negative_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "negative_nonnegative_feature",
                f"Column '{column}' contains negative values.",
                column=str(column),
                count=negative_count,
            )


def _validate_domains(report: ValidationReport, feature_daily: pd.DataFrame) -> None:
    calendar_domains = {
        "Month": (1, 12),
        "Day_Of_Week": (0, 6),
        "Day_Of_Year": (1, 366),
        "ISO_Week": (1, 53),
        "Quarter": (1, 4),
        "Is_Weekend": (0, 1),
    }
    for column, (minimum, maximum) in calendar_domains.items():
        if column not in feature_daily.columns:
            continue
        values = pd.to_numeric(feature_daily[column], errors="coerce")
        invalid_mask = values.isna() | (values < minimum) | (values > maximum)
        if invalid_mask.any():
            report.add_issue(
                ValidationSeverity.ERROR,
                "calendar_domain_violation",
                f"Column '{column}' has values outside [{minimum}, {maximum}].",
                column=column,
                count=int(invalid_mask.sum()),
                sample=_sample(feature_daily.loc[invalid_mask, column]),
            )
    if AVG_WIND_DIRECTION_COLUMN in feature_daily:
        values = pd.to_numeric(feature_daily[AVG_WIND_DIRECTION_COLUMN], errors="coerce")
        invalid_mask = values.isna() | (values < 0) | (values >= 360)
        if invalid_mask.any():
            report.add_issue(
                ValidationSeverity.ERROR,
                "wind_direction_domain_violation",
                "Average wind direction must be in [0, 360).",
                column=AVG_WIND_DIRECTION_COLUMN,
                count=int(invalid_mask.sum()),
                sample=_sample(feature_daily.loc[invalid_mask, AVG_WIND_DIRECTION_COLUMN]),
            )
    cyclical_columns = [
        column
        for column in feature_daily.columns
        if str(column).endswith("_Sin") or str(column).endswith("_Cos")
    ]
    for column in cyclical_columns:
        values = pd.to_numeric(feature_daily[column], errors="coerce")
        invalid_mask = values.isna() | (values < -1.0 - FLOAT_ATOL) | (values > 1.0 + FLOAT_ATOL)
        if invalid_mask.any():
            report.add_issue(
                ValidationSeverity.ERROR,
                "cyclical_domain_violation",
                f"Column '{column}' has values outside [-1, 1].",
                column=str(column),
                count=int(invalid_mask.sum()),
                sample=_sample(feature_daily.loc[invalid_mask, column]),
            )


def _validate_recomputed_features(
    report: ValidationReport,
    payloads: FeatureReadyPayloads,
) -> None:
    try:
        mapped_base = map_integrated_base_columns(payloads.integrated_daily)
        full_calendar = reindex_full_local_calendar(payloads.integrated_coverage, mapped_base)
        recomputed_features = generate_v2_features(full_calendar)
        recomputed_coverage = build_feature_coverage(full_calendar)
        recomputed_ready = select_feature_ready_rows(
            recomputed_features, recomputed_coverage, payloads.v1_columns
        )
    except Exception as exc:  # noqa: BLE001 - validation must report instead of crashing.
        report.add_issue(
            ValidationSeverity.ERROR,
            "feature_recompute_failed",
            f"Could not recompute feature-ready rows from integrated inputs: {exc}",
        )
        return

    actual = payloads.feature_ready_daily.reset_index(drop=True)
    expected = recomputed_ready.reset_index(drop=True)
    report.stats["recomputed_feature_ready_row_count"] = int(len(expected))
    report.stats["recomputed_coverage_row_count"] = int(len(recomputed_coverage))

    actual_status_counts = _value_counts(payloads.feature_coverage["feature_coverage_status"])
    expected_status_counts = _value_counts(recomputed_coverage["feature_coverage_status"])
    if actual_status_counts != expected_status_counts:
        report.add_issue(
            ValidationSeverity.ERROR,
            "recomputed_status_counts_mismatch",
            "Feature coverage status counts differ from recomputation.",
            sample={"actual": actual_status_counts, "expected": expected_status_counts},
        )
    if payloads.feature_coverage["feature_coverage_status"].astype(str).tolist() != recomputed_coverage[
        "feature_coverage_status"
    ].astype(str).tolist():
        report.add_issue(
            ValidationSeverity.ERROR,
            "recomputed_coverage_status_mismatch",
            "Per-date feature coverage statuses differ from recomputation.",
        )

    if actual.columns.tolist() != expected.columns.tolist():
        report.add_issue(
            ValidationSeverity.ERROR,
            "recomputed_feature_column_mismatch",
            "Recomputed features do not have the expected columns.",
            sample=_column_diff(expected.columns.tolist(), actual.columns.tolist()),
        )
        return
    if len(actual) != len(expected):
        report.add_issue(
            ValidationSeverity.ERROR,
            "recomputed_feature_row_count_mismatch",
            "Feature-ready row count differs from recomputation.",
            sample={"actual": int(len(actual)), "expected": int(len(expected))},
        )
        return
    actual_dates = actual[DATE_COLUMN].astype(str).tolist()
    expected_dates = expected[DATE_COLUMN].astype(str).tolist()
    if actual_dates != expected_dates:
        report.add_issue(
            ValidationSeverity.ERROR,
            "recomputed_feature_dates_mismatch",
            "Feature-ready dates differ from recomputation.",
            sample={
                "missing_from_actual": sorted(set(expected_dates) - set(actual_dates))[:5],
                "unexpected_in_actual": sorted(set(actual_dates) - set(expected_dates))[:5],
            },
        )

    max_diffs: dict[str, float] = {}
    offending: dict[str, float] = {}
    for column in [column for column in actual.columns if column != DATE_COLUMN]:
        actual_values = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
        expected_values = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
        diff = np.abs(actual_values - expected_values)
        finite_diff = diff[np.isfinite(diff)]
        max_diff = float(finite_diff.max()) if len(finite_diff) else (math.inf if len(diff) else 0.0)
        max_diffs[str(column)] = max_diff
        if not np.allclose(
            actual_values,
            expected_values,
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
            equal_nan=False,
        ):
            offending[str(column)] = max_diff
    report.stats["feature_recompute_max_abs_difference"] = (
        max(max_diffs.values()) if max_diffs else 0.0
    )
    report.stats["feature_recompute_max_abs_difference_by_column"] = dict(
        sorted(max_diffs.items())
    )
    if offending:
        report.add_issue(
            ValidationSeverity.ERROR,
            "feature_recompute_value_mismatch",
            "Feature-ready values differ from date-based recomputation.",
            count=len(offending),
            sample=dict(list(sorted(offending.items()))[:5]),
        )


def _add_inherited_warnings(
    report: ValidationReport,
    payloads: FeatureReadyPayloads,
) -> None:
    for warning in payloads.validation_payload.get("warnings") or []:
        report.add_issue(
            ValidationSeverity.WARNING,
            "inherited_feature_ready_warning",
            str(warning),
        )
    if payloads.validation_payload.get("verdict") == "PASS WITH WARNINGS":
        report.add_issue(
            ValidationSeverity.WARNING,
            "feature_ready_pass_with_warnings",
            "Embedded feature-ready validation verdict is PASS WITH WARNINGS.",
        )


def _finalize_report(report: ValidationReport) -> ValidationReport:
    report.stats["error_count"] = len(report.errors)
    report.stats["warning_count"] = len(report.warnings)
    report.stats["info_count"] = len(report.infos)
    report.stats["verdict"] = "FAIL" if report.has_errors else (
        "PASS WITH WARNINGS" if report.has_warnings else "PASS"
    )
    return report


def _set_base_stats(
    report: ValidationReport,
    *,
    feature_root: Path | None = None,
    integrated_root: Path | None = None,
    v1_feature_table: Path | None = None,
) -> None:
    report.stats.setdefault("schema_version", SCHEMA_VERSION)
    report.stats.setdefault("validator_name", VALIDATOR_NAME)
    report.stats.setdefault("validator_version", VALIDATOR_VERSION)
    report.stats.setdefault("dataset_role", DATASET_ROLE)
    report.stats.setdefault("dataset_version", DATASET_VERSION)
    report.stats.setdefault("transformation_version", TRANSFORMATION_VERSION)
    report.stats.setdefault("source_transformation_version", SOURCE_TRANSFORMATION_VERSION)
    report.stats.setdefault("float_atol", FLOAT_ATOL)
    report.stats.setdefault("float_rtol", FLOAT_RTOL)
    if feature_root is not None:
        report.stats["feature_root"] = str(feature_root)
    if integrated_root is not None:
        report.stats["integrated_root"] = str(integrated_root)
    if v1_feature_table is not None:
        report.stats["v1_feature_table"] = str(v1_feature_table)


def _read_csv_with_raw_header(path: Path) -> tuple[pd.DataFrame, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        raw_columns = next(reader, [])
    return pd.read_csv(path), raw_columns


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def sha256_file(path: str | Path) -> str:
    """Return a file SHA-256 checksum."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup_checksum(recorded: Mapping[str, str], path: Path) -> str | None:
    normalized_path = _normalize_path(path)
    for key, value in recorded.items():
        if _normalize_path(key).endswith(normalized_path):
            return str(value)
        if normalized_path.endswith(_normalize_path(key)):
            return str(value)
    return None


def _path_suffix_matches(actual: object, expected: Path) -> bool:
    if actual is None:
        return False
    actual_text = _normalize_path(actual)
    expected_text = _normalize_path(expected)
    return actual_text == expected_text or actual_text.endswith(expected_text)


def _normalize_path(path: object) -> str:
    return str(path).replace("\\", "/").strip().rstrip("/")


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def _date_texts(values: pd.Series | Sequence[Any]) -> list[str]:
    dates = pd.to_datetime(values, errors="coerce")
    return [
        value.strftime("%Y-%m-%d") if not pd.isna(value) else "<invalid>"
        for value in dates
    ]


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.casefold()
    return normalized.isin({"true", "1", "yes"})


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        text = str(value)
        if text in seen and text not in duplicates:
            duplicates.append(text)
        seen.add(text)
    return duplicates


def _column_diff(expected: Sequence[str], actual: Sequence[str]) -> dict[str, Any]:
    return {
        "missing": [column for column in expected if column not in actual][:10],
        "extra": [column for column in actual if column not in expected][:10],
        "first_different_positions": [
            {
                "position": index,
                "expected": expected[index] if index < len(expected) else None,
                "actual": actual[index] if index < len(actual) else None,
            }
            for index in range(max(len(expected), len(actual)))
            if index >= len(expected) or index >= len(actual) or expected[index] != actual[index]
        ][:10],
    }


def _expect_equal(
    report: ValidationReport,
    actual: Any,
    expected: Any,
    code: str,
) -> None:
    if actual != expected:
        report.add_issue(
            ValidationSeverity.ERROR,
            code,
            "Metadata value does not match the accepted contract.",
            sample={"actual": actual, "expected": expected},
        )


def _sample(values: Sequence[Any] | pd.Series | pd.Index, limit: int = 5) -> list[Any]:
    result: list[Any] = []
    for value in list(values)[:limit]:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if hasattr(value, "isoformat"):
            try:
                value = value.isoformat()
            except (TypeError, ValueError):
                pass
        result.append(value)
    return result


def _build_synthetic_payloads() -> FeatureReadyPayloads:
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    integrated_coverage = pd.DataFrame(
        {
            DATE_LOCAL_COLUMN: dates.strftime("%Y-%m-%d"),
            "coverage_status": "integration-ready",
            "ren_status": "complete",
            "era5_status": "complete",
            "integration_ready": True,
        }
    )
    integrated_coverage.loc[
        integrated_coverage[DATE_LOCAL_COLUMN].eq("2026-01-10"),
        ["coverage_status", "ren_status", "integration_ready"],
    ] = ["excluded-downstream-ren-unavailable", "unavailable", False]
    integrated_daily = pd.DataFrame(
        {
            DATE_LOCAL_COLUMN: dates.strftime("%Y-%m-%d"),
            TARGET_COLUMN: np.linspace(100.0, 124.0, len(dates)),
            "wind_speed_m_s_mean": np.linspace(4.0, 6.4, len(dates)),
            "temperature_2m_c_mean": np.linspace(10.0, 16.0, len(dates)),
            "vector_mean_wind_direction_deg_from": np.linspace(180.0, 240.0, len(dates)),
        }
    )
    integrated_daily = integrated_daily[
        ~integrated_daily[DATE_LOCAL_COLUMN].eq("2026-01-10")
    ].reset_index(drop=True)
    mapped_base = map_integrated_base_columns(integrated_daily)
    full_calendar = reindex_full_local_calendar(integrated_coverage, mapped_base)
    features = generate_v2_features(full_calendar)
    feature_coverage = build_feature_coverage(full_calendar)
    v1_columns = features.columns.tolist()
    feature_ready_daily = select_feature_ready_rows(features, feature_coverage, v1_columns)
    feature_schema = build_feature_schema(v1_columns)
    v1_structure_comparison = {
        "schema_version": V1_COMPARISON_VERSION,
        "v1_feature_table": "synthetic_v1.csv",
        "v1_row_count": int(len(feature_ready_daily)),
        "v2_feature_ready_row_count": int(len(feature_ready_daily)),
        "v1_column_count": len(v1_columns),
        "v2_column_count": len(v1_columns),
        "exact_column_order_match": True,
        "missing_from_v2": [],
        "extra_in_v2": [],
        "different_order": [],
        "date_column_name_match": True,
        "numeric_feature_columns_match": True,
        "v2_numeric_sample_finite": True,
    }
    status_counts = _value_counts(feature_coverage["feature_coverage_status"])
    validation_payload = {
        "schema_version": FEATURE_VALIDATION_VERSION,
        "dataset_version": DATASET_VERSION,
        "dataset_role": DATASET_ROLE,
        "transformation_version": TRANSFORMATION_VERSION,
        "source_transformation_version": SOURCE_TRANSFORMATION_VERSION,
        "passed": True,
        "verdict": "PASS WITH WARNINGS",
        "failures": [],
        "warnings": ["Synthetic inherited warning."],
        "actual_counts": {
            "coverage_rows": len(feature_coverage),
            "integrated_ready_rows": int(feature_coverage["integration_ready"].sum()),
            "current_complete_base_rows": int(feature_coverage["current_base_complete"].sum()),
            "feature_ready_rows": len(feature_ready_daily),
        },
        "feature_coverage_status_counts": status_counts,
        "v1_structure_comparison": v1_structure_comparison,
    }
    manifest = {
        "schema_version": FEATURE_MANIFEST_VERSION,
        "dataset_version": DATASET_VERSION,
        "dataset_role": DATASET_ROLE,
        "transformation_version": TRANSFORMATION_VERSION,
        "source_transformation_version": SOURCE_TRANSFORMATION_VERSION,
        "source_dataset_role": "integrated_daily_ren_era5_land",
        "source_paths": {
            "input_root": "synthetic_integrated",
            "v1_feature_table": "synthetic_v1.csv",
        },
        "source_files": {},
        "source_sha256_checksums": {},
        "output_files": {},
        "sha256_checksums": {},
        "row_counts": dict(validation_payload["actual_counts"]),
        "feature_coverage_status_counts": status_counts,
        "status": "PASS WITH WARNINGS",
        "warnings": list(validation_payload["warnings"]),
        "failures": [],
    }
    integrated_validation = {
        "passed": True,
        "verdict": "PASS WITH WARNINGS",
        "warnings": ["Synthetic source warning."],
    }
    integrated_manifest = {
        "dataset_role": "integrated_daily_ren_era5_land",
        "transformation_version": SOURCE_TRANSFORMATION_VERSION,
    }
    return FeatureReadyPayloads(
        feature_ready_daily=feature_ready_daily,
        feature_coverage=feature_coverage,
        feature_schema=feature_schema,
        manifest=manifest,
        validation_payload=validation_payload,
        v1_structure_comparison=v1_structure_comparison,
        integrated_daily=integrated_daily,
        integrated_coverage=integrated_coverage,
        integrated_validation=integrated_validation,
        integrated_manifest=integrated_manifest,
        v1_columns=v1_columns,
        raw_feature_columns=feature_ready_daily.columns.tolist(),
        raw_coverage_columns=feature_coverage.columns.tolist(),
    )


def _mutate_payload(payloads: FeatureReadyPayloads, **changes: Any) -> FeatureReadyPayloads:
    values = {
        "feature_ready_daily": payloads.feature_ready_daily.copy(deep=True),
        "feature_coverage": payloads.feature_coverage.copy(deep=True),
        "feature_schema": dict(payloads.feature_schema),
        "manifest": dict(payloads.manifest),
        "validation_payload": dict(payloads.validation_payload),
        "v1_structure_comparison": dict(payloads.v1_structure_comparison),
        "integrated_daily": payloads.integrated_daily.copy(deep=True),
        "integrated_coverage": payloads.integrated_coverage.copy(deep=True),
        "integrated_validation": dict(payloads.integrated_validation),
        "integrated_manifest": dict(payloads.integrated_manifest),
        "v1_columns": list(payloads.v1_columns),
        "raw_feature_columns": list(payloads.raw_feature_columns or []),
        "raw_coverage_columns": list(payloads.raw_coverage_columns or []),
    }
    values.update(changes)
    return FeatureReadyPayloads(**values)


def _with_value(frame: pd.DataFrame, row: int, column: str, value: Any) -> pd.DataFrame:
    changed = frame.copy(deep=True)
    if isinstance(value, str) and column in changed.columns:
        changed[column] = changed[column].astype(object)
    changed.loc[row, column] = value
    return changed


def _write_synthetic_payloads(payloads: FeatureReadyPayloads, root: Path) -> dict[str, Path]:
    feature_root = root / "feature"
    integrated_root = root / "integrated"
    feature_root.mkdir(parents=True)
    integrated_root.mkdir(parents=True)
    v1_feature_table = root / "synthetic_v1.csv"

    payloads.feature_ready_daily.to_csv(feature_root / "feature_ready_daily.csv", index=False)
    payloads.feature_coverage.to_csv(feature_root / "feature_coverage.csv", index=False)
    payloads.integrated_daily.to_csv(integrated_root / "daily_merged.csv", index=False)
    payloads.integrated_coverage.to_csv(integrated_root / "coverage.csv", index=False)
    payloads.feature_ready_daily.head(3).to_csv(v1_feature_table, index=False)

    feature_schema = dict(payloads.feature_schema)
    validation_payload = dict(payloads.validation_payload)
    comparison = compare_v1_structure(
        v1_feature_table=v1_feature_table,
        v1_columns=payloads.v1_columns,
        feature_ready_daily=payloads.feature_ready_daily,
    )
    integrated_validation = dict(payloads.integrated_validation)
    integrated_manifest = dict(payloads.integrated_manifest)

    (integrated_root / "validation.json").write_text(
        json.dumps(integrated_validation, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (integrated_root / "manifest.json").write_text(
        json.dumps(integrated_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (feature_root / "feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (feature_root / "validation.json").write_text(
        json.dumps(validation_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (feature_root / "v1_structure_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = dict(payloads.manifest)
    manifest["source_paths"] = {
        "input_root": str(integrated_root),
        "v1_feature_table": str(v1_feature_table),
    }
    manifest["output_files"] = {
        "feature_ready_daily": str(feature_root / "feature_ready_daily.csv"),
        "feature_coverage": str(feature_root / "feature_coverage.csv"),
        "feature_schema": str(feature_root / "feature_schema.json"),
        "v1_structure_comparison": str(feature_root / "v1_structure_comparison.json"),
        "validation": str(feature_root / "validation.json"),
        "manifest": str(feature_root / "manifest.json"),
    }
    manifest["source_sha256_checksums"] = {
        str(integrated_root / "daily_merged.csv"): sha256_file(integrated_root / "daily_merged.csv"),
        str(integrated_root / "coverage.csv"): sha256_file(integrated_root / "coverage.csv"),
        str(integrated_root / "validation.json"): sha256_file(integrated_root / "validation.json"),
        str(integrated_root / "manifest.json"): sha256_file(integrated_root / "manifest.json"),
        str(v1_feature_table): sha256_file(v1_feature_table),
    }
    manifest["sha256_checksums"] = {
        str(feature_root / "feature_ready_daily.csv"): sha256_file(
            feature_root / "feature_ready_daily.csv"
        ),
        str(feature_root / "feature_coverage.csv"): sha256_file(
            feature_root / "feature_coverage.csv"
        ),
        str(feature_root / "feature_schema.json"): sha256_file(feature_root / "feature_schema.json"),
        str(feature_root / "v1_structure_comparison.json"): sha256_file(
            feature_root / "v1_structure_comparison.json"
        ),
        str(feature_root / "validation.json"): sha256_file(feature_root / "validation.json"),
    }
    (feature_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "feature_root": feature_root,
        "integrated_root": integrated_root,
        "v1_feature_table": v1_feature_table,
    }


def _raise_for_errors_passes(report: ValidationReport) -> bool:
    try:
        report.raise_for_errors()
    except RuntimeError:
        return False
    return True


def _raise_for_errors_fails(report: ValidationReport) -> bool:
    try:
        report.raise_for_errors()
    except RuntimeError:
        return True
    return False


def _example_error_message(report: ValidationReport) -> str:
    try:
        report.raise_for_errors()
    except RuntimeError as exc:
        return str(exc)
    return ""
