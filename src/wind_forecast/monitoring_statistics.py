"""Deterministic statistical helpers for Phase 9 batch monitoring.

The functions in this module are deliberately free of filesystem writes and
model side effects.  They operate on explicit arrays/data frames so calibration
and reporting can use the same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


POLICY_SCHEMA = "wind_forecast.monitoring_policy.v1"
WINDOWS = (30, 90)
DIRECTION_GROUPS: Mapping[str, tuple[str, str]] = {
    "wind_direction_current": ("Wind_Direction_Sin", "Wind_Direction_Cos"),
    "wind_direction_lag1": ("Wind_Direction_Sin_Lag1", "Wind_Direction_Cos_Lag1"),
    "wind_direction_lag2": ("Wind_Direction_Sin_Lag2", "Wind_Direction_Cos_Lag2"),
    "wind_direction_lag3": ("Wind_Direction_Sin_Lag3", "Wind_Direction_Cos_Lag3"),
    "wind_direction_lag7": ("Wind_Direction_Sin_Lag7", "Wind_Direction_Cos_Lag7"),
}
DIRECTION_COMPONENTS = frozenset(
    component for pair in DIRECTION_GROUPS.values() for component in pair
)


@dataclass(frozen=True)
class MonitoringPolicy:
    """Validated configuration shared by calibration and reporting."""

    reference_start: str
    reference_end: str
    windows_days: tuple[int, ...]
    warning_quantile: float
    critical_quantile: float
    minimum_samples: Mapping[str, int]
    r2_minimum_samples: Mapping[str, int]
    mape_epsilon_quantile: float
    alert_persistence_distinct_dates: int
    source_objective_days: int
    source_late_days: int
    hard_quality_tolerance: int
    overrides: Mapping[str, Mapping[str, float]]
    schema_version: str = POLICY_SCHEMA

    @classmethod
    def load(cls, path: str | Path) -> "MonitoringPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != POLICY_SCHEMA:
            raise ValueError("Unsupported monitoring policy schema.")
        policy = cls(
            reference_start=str(payload["reference_start"]),
            reference_end=str(payload["reference_end"]),
            windows_days=tuple(int(value) for value in payload["windows_days"]),
            warning_quantile=float(payload["warning_quantile"]),
            critical_quantile=float(payload["critical_quantile"]),
            minimum_samples={str(k): int(v) for k, v in payload["minimum_samples"].items()},
            r2_minimum_samples={
                str(k): int(v) for k, v in payload["r2_minimum_samples"].items()
            },
            mape_epsilon_quantile=float(payload["mape_epsilon_quantile"]),
            alert_persistence_distinct_dates=int(
                payload["alert_persistence_distinct_dates"]
            ),
            source_objective_days=int(payload["source_objective_days"]),
            source_late_days=int(payload["source_late_days"]),
            hard_quality_tolerance=int(payload["hard_quality_tolerance"]),
            overrides={
                str(path): {str(key): float(value) for key, value in limits.items()}
                for path, limits in (payload.get("overrides") or {}).items()
            },
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.windows_days != WINDOWS:
            raise ValueError("Phase 9 requires exactly the 30-day and 90-day windows.")
        if not 0 < self.warning_quantile < self.critical_quantile < 1:
            raise ValueError("Monitoring quantiles must satisfy 0 < warning < critical < 1.")
        if not 0 < self.mape_epsilon_quantile < 1:
            raise ValueError("mape_epsilon_quantile must be between zero and one.")
        if self.alert_persistence_distinct_dates < 1:
            raise ValueError("Alert persistence must be at least one distinct date.")
        if not 0 <= self.source_objective_days < self.source_late_days:
            raise ValueError("Source objective must precede the source-late deadline.")
        if self.hard_quality_tolerance < 0:
            raise ValueError("Hard quality tolerance cannot be negative.")
        for path, limits in self.overrides.items():
            if not path or set(limits) != {"warning", "critical"}:
                raise ValueError(
                    "Each threshold override requires a path and warning/critical values."
                )
            if not all(math.isfinite(float(value)) for value in limits.values()):
                raise ValueError("Threshold overrides must be finite.")
        for window in self.windows_days:
            minimum = self.minimum_samples.get(str(window), 0)
            r2_minimum = self.r2_minimum_samples.get(str(window), 0)
            if not 2 <= minimum <= window:
                raise ValueError(f"Invalid minimum sample count for {window} days.")
            if not minimum <= r2_minimum <= window:
                raise ValueError(f"Invalid R2 sample count for {window} days.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_start": self.reference_start,
            "reference_end": self.reference_end,
            "windows_days": list(self.windows_days),
            "warning_quantile": self.warning_quantile,
            "critical_quantile": self.critical_quantile,
            "minimum_samples": dict(self.minimum_samples),
            "r2_minimum_samples": dict(self.r2_minimum_samples),
            "mape_epsilon_quantile": self.mape_epsilon_quantile,
            "alert_persistence_distinct_dates": self.alert_persistence_distinct_dates,
            "source_objective_days": self.source_objective_days,
            "source_late_days": self.source_late_days,
            "hard_quality_tolerance": self.hard_quality_tolerance,
            "overrides": {
                path: dict(limits) for path, limits in sorted(self.overrides.items())
            },
        }


def finite_values(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    """Return finite one-dimensional values or raise on invalid evidence."""
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0:
        raise ValueError("Monitoring samples must not be empty.")
    if not np.isfinite(result).all():
        raise ValueError("Monitoring samples must contain only finite values.")
    return result


def robust_scale(reference: Sequence[float] | np.ndarray) -> tuple[float, str]:
    """Return IQR, then standard deviation, with explicit constant handling."""
    values = finite_values(reference)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > 0:
        return iqr, "iqr"
    standard_deviation = float(np.std(values))
    if standard_deviation > 0:
        return standard_deviation, "standard_deviation"
    return 0.0, "constant"


def drift_statistics(
    current: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Calculate normalized Wasserstein and two-sample KS statistics."""
    current_values = finite_values(current)
    reference_values = finite_values(reference)
    scale, scale_method = robust_scale(reference_values)
    raw_wasserstein = float(wasserstein_distance(reference_values, current_values))
    if scale == 0:
        normalized = 0.0 if raw_wasserstein == 0 else math.inf
    else:
        normalized = raw_wasserstein / scale
    # The asymptotic path is deterministic for every supported sample size and
    # avoids SciPy's exact-to-asymptotic fallback warning on tied values.
    ks = ks_2samp(reference_values, current_values, method="asymp")
    return {
        "sample_count": int(current_values.size),
        "reference_count": int(reference_values.size),
        "wasserstein": raw_wasserstein,
        "normalized_wasserstein": normalized,
        "normalization_scale": scale,
        "normalization_method": scale_method,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue_informational": float(ks.pvalue),
        "current_mean": float(np.mean(current_values)),
        "reference_mean": float(np.mean(reference_values)),
    }


def circular_context(sine: Sequence[float], cosine: Sequence[float]) -> dict[str, float]:
    """Describe a direction distribution represented on the unit circle."""
    sine_values = finite_values(sine)
    cosine_values = finite_values(cosine)
    if sine_values.shape != cosine_values.shape:
        raise ValueError("Circular sine and cosine samples must have equal shapes.")
    mean_sine = float(np.mean(sine_values))
    mean_cosine = float(np.mean(cosine_values))
    angle = math.degrees(math.atan2(mean_sine, mean_cosine)) % 360.0
    return {
        "mean_angle_degrees": angle,
        "resultant_length": float(math.hypot(mean_sine, mean_cosine)),
    }


def circular_drift_statistics(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    sine_column: str,
    cosine_column: str,
) -> dict[str, Any]:
    """Evaluate wind direction without treating 0/360 degrees as a discontinuity."""
    components = {
        "sine": drift_statistics(current[sine_column], reference[sine_column]),
        "cosine": drift_statistics(current[cosine_column], reference[cosine_column]),
    }
    return {
        "components": components,
        "current_circular": circular_context(
            current[sine_column], current[cosine_column]
        ),
        "reference_circular": circular_context(
            reference[sine_column], reference[cosine_column]
        ),
        "normalized_wasserstein": max(
            item["normalized_wasserstein"] for item in components.values()
        ),
        "ks_statistic": max(item["ks_statistic"] for item in components.values()),
    }


def seasonal_reference(
    reference: pd.DataFrame,
    current_dates: Sequence[Any] | pd.Series,
    *,
    date_column: str = "Date",
    exclude_dates: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Select reference rows sharing the exact month/day keys of a window."""
    dates = pd.to_datetime(pd.Series(current_dates), errors="coerce")
    if dates.isna().any() or dates.empty:
        raise ValueError("Seasonal comparison requires valid current dates.")
    reference_dates = pd.to_datetime(reference[date_column], errors="coerce")
    if reference_dates.isna().any():
        raise ValueError("Reference data contains invalid dates.")
    wanted = set(zip(dates.dt.month, dates.dt.day, strict=True))
    mask = pd.Series(
        [key in wanted for key in zip(reference_dates.dt.month, reference_dates.dt.day, strict=True)],
        index=reference.index,
    )
    if exclude_dates is not None:
        excluded = set(pd.to_datetime(pd.Series(exclude_dates)).dt.normalize())
        mask &= ~reference_dates.dt.normalize().isin(excluded)
    selected = reference.loc[mask].copy()
    if selected.empty:
        raise ValueError("Seasonal reference selection is empty.")
    return selected


def calendar_window(
    frame: pd.DataFrame,
    through_date: str | pd.Timestamp,
    window_days: int,
    *,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Return an inclusive civil-calendar window ending on through_date."""
    end = pd.Timestamp(through_date).normalize()
    start = end - pd.Timedelta(days=window_days - 1)
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("Window input contains invalid dates.")
    return frame.loc[dates.between(start, end)].copy()


def regression_metrics(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    mape_epsilon: float,
    r2_minimum_samples: int,
) -> dict[str, Any]:
    """Calculate protected original-scale performance metrics."""
    actual_values = finite_values(actual)
    predicted_values = finite_values(predicted)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted samples must have equal shapes.")
    if mape_epsilon <= 0:
        raise ValueError("MAPE epsilon must be greater than zero.")
    error = predicted_values - actual_values
    denominator = np.maximum(np.abs(actual_values), mape_epsilon)
    r2: float | None = None
    r2_status = "insufficient_data"
    if len(actual_values) >= r2_minimum_samples:
        if float(np.var(actual_values)) > 0:
            r2 = float(1.0 - np.sum(error**2) / np.sum((actual_values - actual_values.mean()) ** 2))
            r2_status = "available"
        else:
            r2_status = "constant_target"
    return {
        "sample_count": int(len(actual_values)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "R2": r2,
        "R2_status": r2_status,
        "MAPE_percent": float(np.mean(np.abs(error) / denominator) * 100.0),
        "mape_epsilon": float(mape_epsilon),
        "mape_protected_sample_count": int((np.abs(actual_values) < mape_epsilon).sum()),
    }


def calibrated_limits(
    values: Sequence[float], policy: MonitoringPolicy, *, lower_is_bad: bool = False
) -> dict[str, Any]:
    """Resolve warning/critical empirical thresholds from backtest values."""
    samples = finite_values(values)
    if lower_is_bad:
        return {
            "warning": float(np.quantile(samples, 1.0 - policy.warning_quantile)),
            "critical": float(np.quantile(samples, 1.0 - policy.critical_quantile)),
            "direction": "lower",
        }
    return {
        "warning": float(np.quantile(samples, policy.warning_quantile)),
        "critical": float(np.quantile(samples, policy.critical_quantile)),
        "direction": "upper",
    }


def threshold_severity(value: float | None, limits: Mapping[str, Any]) -> str:
    """Classify a metric using an explicitly directional threshold contract."""
    if value is None or not math.isfinite(float(value)):
        return "critical" if value is not None else "not_available"
    direction = str(limits.get("direction", "upper"))
    warning = float(limits["warning"])
    critical = float(limits["critical"])
    if direction == "lower":
        if value < critical:
            return "critical"
        if value < warning:
            return "warning"
    else:
        if value > critical:
            return "critical"
        if value > warning:
            return "warning"
    return "ok"


__all__ = [
    "DIRECTION_COMPONENTS",
    "DIRECTION_GROUPS",
    "MonitoringPolicy",
    "calendar_window",
    "calibrated_limits",
    "circular_context",
    "circular_drift_statistics",
    "drift_statistics",
    "regression_metrics",
    "seasonal_reference",
    "threshold_severity",
]
