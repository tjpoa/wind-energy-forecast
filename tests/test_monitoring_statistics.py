from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_forecast.monitoring_statistics import (
    circular_drift_statistics,
    drift_statistics,
    regression_metrics,
    seasonal_reference,
)


def _directions(values: list[float]) -> pd.DataFrame:
    radians = np.radians(values)
    return pd.DataFrame({"sin": np.sin(radians), "cos": np.cos(radians)})


def test_circular_direction_handles_wraparound_and_detects_rotation() -> None:
    reference = _directions([358, 359, 0, 1, 2] * 10)
    wrapped = _directions([359, 0, 1, 2, 3] * 10)
    rotated = _directions([178, 179, 180, 181, 182] * 10)

    wrapped_stats = circular_drift_statistics(wrapped, reference, "sin", "cos")
    rotated_stats = circular_drift_statistics(rotated, reference, "sin", "cos")

    assert wrapped_stats["normalized_wasserstein"] < 1
    assert rotated_stats["normalized_wasserstein"] > wrapped_stats["normalized_wasserstein"]
    assert wrapped_stats["current_circular"]["mean_angle_degrees"] == pytest.approx(1.0)


def test_drift_statistics_known_shift_is_larger_than_no_shift() -> None:
    reference = np.linspace(0, 10, 100)
    same = drift_statistics(reference.copy(), reference)
    shifted = drift_statistics(reference + 10, reference)

    assert same["normalized_wasserstein"] == pytest.approx(0)
    assert same["ks_statistic"] == pytest.approx(0)
    assert shifted["normalized_wasserstein"] > 1
    assert shifted["ks_statistic"] > 0.5


def test_seasonal_reference_uses_exact_month_day_keys_and_excludes_window() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2023-02-28", "2023-03-01", "2024-02-28", "2024-02-29", "2024-03-01"]
            ),
            "value": range(5),
        }
    )
    result = seasonal_reference(
        frame,
        pd.to_datetime(["2024-02-28", "2024-02-29"]),
        exclude_dates=pd.to_datetime(["2024-02-28", "2024-02-29"]),
    )

    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2023-02-28"]


def test_performance_metrics_protect_mape_and_gate_r2() -> None:
    metrics = regression_metrics(
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        mape_epsilon=0.5,
        r2_minimum_samples=4,
    )

    assert metrics["MAE"] == pytest.approx(1.0)
    assert metrics["RMSE"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(1.0)
    assert metrics["MAPE_percent"] == pytest.approx((2 + 1 + 0.5) / 3 * 100)
    assert metrics["mape_protected_sample_count"] == 1
    assert metrics["R2"] is None
    assert metrics["R2_status"] == "insufficient_data"

    constant = regression_metrics(
        np.ones(20),
        np.ones(20),
        mape_epsilon=0.1,
        r2_minimum_samples=20,
    )
    assert constant["R2"] is None
    assert constant["R2_status"] == "constant_target"
