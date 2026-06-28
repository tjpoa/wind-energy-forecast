"""Generic, non-mutating validation report helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class ValidationSeverity(str, Enum):
    """Severity levels used by validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _serialize_value(value: object) -> object:
    """Convert common pandas/numpy values into simple Python values."""
    if isinstance(value, ValidationSeverity):
        return value.value
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return value


def _small_sample(values: Iterable[object], limit: int = 5) -> list[object]:
    return [_serialize_value(value) for value in list(values)[:limit]]


def _column_occurrences(df: pd.DataFrame, column: str) -> int:
    return sum(existing == column for existing in df.columns)


def _get_unique_column(df: pd.DataFrame, column: str) -> pd.Series | None:
    if _column_occurrences(df, column) != 1:
        return None
    return df.loc[:, column]


def _parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed")


@dataclass(frozen=True)
class ValidationIssue:
    """A single deterministic validation finding."""

    severity: ValidationSeverity
    code: str
    message: str
    column: str | None = None
    count: int | None = None
    sample: object | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the issue to plain Python values."""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "count": self.count,
            "sample": _serialize_value(self.sample),
        }


@dataclass
class ValidationReport:
    """Validation result for one logical dataset."""

    dataset_name: str
    row_count: int = 0
    column_count: int = 0
    date_range: tuple[object | None, object | None] | None = None
    stats: dict[str, object] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Critical validation failures."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Non-critical validation warnings."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]

    @property
    def infos(self) -> list[ValidationIssue]:
        """Informational validation notes."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.INFO]

    @property
    def has_errors(self) -> bool:
        """Whether the report contains critical validation failures."""
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        """Whether the report contains warnings."""
        return bool(self.warnings)

    @property
    def passed(self) -> bool:
        """Whether the report contains no critical validation failures."""
        return not self.has_errors

    def add_issue(
        self,
        severity: ValidationSeverity | str,
        code: str,
        message: str,
        *,
        column: str | None = None,
        count: int | None = None,
        sample: object | None = None,
    ) -> ValidationIssue:
        """Append one issue and return it."""
        issue = ValidationIssue(
            severity=ValidationSeverity(severity),
            code=code,
            message=message,
            column=column,
            count=count,
            sample=sample,
        )
        self.issues.append(issue)
        return issue

    def raise_for_errors(self) -> None:
        """Raise ``ValidationError`` when critical issues are present."""
        if self.has_errors:
            raise ValidationError(self)

    def format_summary(self) -> str:
        """Return a deterministic, human-readable validation summary."""
        status = "passed" if self.passed else "failed"
        lines = [
            f"Validation report for {self.dataset_name}: {status}",
            f"Rows: {self.row_count}",
            f"Columns: {self.column_count}",
        ]
        if self.date_range is not None:
            start, end = self.date_range
            lines.append(f"Date range: {start} to {end}")
        lines.append(
            "Issues: "
            f"{len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s), "
            f"{len(self.infos)} info(s)"
        )
        for issue in self.issues:
            location = f" [{issue.column}]" if issue.column else ""
            count = f" count={issue.count}" if issue.count is not None else ""
            sample = f" sample={_serialize_value(issue.sample)}" if issue.sample is not None else ""
            lines.append(
                f"- {issue.severity.value}: {issue.code}{location}{count}: "
                f"{issue.message}{sample}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialize the report to plain Python values."""
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "date_range": _serialize_value(self.date_range),
            "stats": _serialize_value(self.stats),
            "issues": [issue.to_dict() for issue in self.issues],
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "passed": self.passed,
        }


class ValidationError(RuntimeError):
    """Raised when a validation report contains critical errors."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        error_summaries = [f"{issue.code}: {issue.message}" for issue in report.errors[:3]]
        if len(report.errors) > 3:
            error_summaries.append(f"... and {len(report.errors) - 3} more")
        joined_errors = "; ".join(error_summaries)
        super().__init__(
            f"Validation failed for {report.dataset_name} with "
            f"{len(report.errors)} error(s): {joined_errors}"
        )


def create_report(
    df: pd.DataFrame,
    dataset_name: str,
    date_column: str | None = None,
) -> ValidationReport:
    """Create a base report without mutating the input DataFrame."""
    date_range: tuple[object | None, object | None] | None = None
    if date_column is not None and _column_occurrences(df, date_column) == 1:
        parsed_dates = _parse_dates(df.loc[:, date_column])
        valid_dates = parsed_dates.dropna()
        if not valid_dates.empty:
            date_range = (valid_dates.min(), valid_dates.max())
        else:
            date_range = (None, None)

    return ValidationReport(
        dataset_name=dataset_name,
        row_count=len(df),
        column_count=len(df.columns),
        date_range=date_range,
    )


def check_empty_dataframe(report: ValidationReport, df: pd.DataFrame) -> ValidationReport:
    """Record an error when the DataFrame has no rows or columns."""
    if df.empty:
        report.add_issue(
            ValidationSeverity.ERROR,
            "empty_dataframe",
            "DataFrame contains no rows or columns.",
            count=len(df),
        )
    return report


def check_duplicate_columns(report: ValidationReport, df: pd.DataFrame) -> ValidationReport:
    """Record duplicate column names while preserving their first duplicate order."""
    seen: set[object] = set()
    duplicates: list[object] = []
    for column in df.columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)

    if duplicates:
        report.add_issue(
            ValidationSeverity.ERROR,
            "duplicate_column",
            "DataFrame contains duplicate column names.",
            count=len(duplicates),
            sample=_small_sample(duplicates),
        )
    return report


def check_required_columns(
    report: ValidationReport,
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> ValidationReport:
    """Record one issue for each missing required column."""
    for column in required_columns:
        if column not in df.columns:
            report.add_issue(
                ValidationSeverity.ERROR,
                "missing_required_column",
                f"Required column '{column}' is missing.",
                column=column,
                count=1,
            )
    return report


def check_date_parseable(
    report: ValidationReport,
    df: pd.DataFrame,
    column: str,
) -> ValidationReport:
    """Record unparseable non-null values in a date column."""
    if column not in df.columns:
        report.add_issue(
            ValidationSeverity.ERROR,
            "missing_required_column",
            f"Required column '{column}' is missing.",
            column=column,
            count=1,
        )
        return report

    if _column_occurrences(df, column) != 1:
        report.add_issue(
            ValidationSeverity.ERROR,
            "duplicate_column",
            f"Column '{column}' appears multiple times.",
            column=column,
            count=_column_occurrences(df, column),
        )
        return report

    series = _get_unique_column(df, column)
    assert series is not None
    parsed_dates = _parse_dates(series)
    invalid_mask = series.notna() & parsed_dates.isna()
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "invalid_date",
            f"Column '{column}' contains unparseable date values.",
            column=column,
            count=invalid_count,
            sample=_small_sample(series[invalid_mask]),
        )
    return report


def check_chronological_order(
    report: ValidationReport,
    df: pd.DataFrame,
    column: str,
) -> ValidationReport:
    """Record an error when parseable dates are not chronological."""
    series = _get_unique_column(df, column)
    if series is None:
        return report

    parsed_dates = _parse_dates(series)
    valid_dates = parsed_dates[parsed_dates.notna()]
    if len(valid_dates) > 1 and not valid_dates.is_monotonic_increasing:
        report.add_issue(
            ValidationSeverity.ERROR,
            "non_chronological_date",
            f"Column '{column}' is not sorted in chronological order.",
            column=column,
            count=len(valid_dates),
            sample=_small_sample(series),
        )
    return report


def check_duplicate_dates(
    report: ValidationReport,
    df: pd.DataFrame,
    column: str,
) -> ValidationReport:
    """Record duplicate parseable date values."""
    series = _get_unique_column(df, column)
    if series is None:
        return report

    parsed_dates = _parse_dates(series)
    duplicate_mask = parsed_dates.notna() & parsed_dates.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        report.add_issue(
            ValidationSeverity.ERROR,
            "duplicate_date",
            f"Column '{column}' contains duplicate date values.",
            column=column,
            count=duplicate_count,
            sample=_small_sample(series[duplicate_mask]),
        )
    return report


def check_null_values(
    report: ValidationReport,
    df: pd.DataFrame,
    columns: Iterable[str],
) -> ValidationReport:
    """Record null values in the selected columns."""
    for column in columns:
        series = _get_unique_column(df, column)
        if series is None:
            continue
        null_mask = series.isna()
        null_count = int(null_mask.sum())
        if null_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "null_values",
                f"Column '{column}' contains null values.",
                column=column,
                count=null_count,
                sample=_small_sample(series[null_mask].index),
            )
    return report


def check_finite_numeric_values(
    report: ValidationReport,
    df: pd.DataFrame,
    columns: Iterable[str],
) -> ValidationReport:
    """Record positive and negative infinity in selected numeric columns."""
    for column in columns:
        series = _get_unique_column(df, column)
        if series is None:
            continue
        if not pd.api.types.is_numeric_dtype(series):
            report.add_issue(
                ValidationSeverity.ERROR,
                "non_numeric_column",
                f"Column '{column}' is not numeric.",
                column=column,
                count=len(series),
                sample=str(series.dtype),
            )
            continue

        non_null_values = series[series.notna()]
        finite_mask = np.isfinite(non_null_values.to_numpy(dtype=float))
        non_finite_mask = pd.Series(~finite_mask, index=non_null_values.index)
        non_finite_count = int(non_finite_mask.sum())
        if non_finite_count:
            report.add_issue(
                ValidationSeverity.ERROR,
                "non_finite_numeric",
                f"Column '{column}' contains non-finite numeric values.",
                column=column,
                count=non_finite_count,
                sample=_small_sample(non_null_values[non_finite_mask]),
            )
    return report
