import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from wind_forecast import v2_scaling


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "Wind_Production": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Feature_B": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def test_fit_v2_scalers_uses_only_train_plus_validation_and_writes_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v2_scaling, "project_root", lambda: tmp_path)
    input_path = tmp_path / "feature_ready_daily.csv"
    _frame().to_csv(input_path, index=False)
    output_dir = tmp_path / "models" / "v2" / "scalers" / "run"

    result = v2_scaling.fit_v2_scalers(
        input_path=input_path,
        output_dir=output_dir,
        fit_start="2024-01-01",
        fit_end="2024-01-04",
    )

    assert result.fit_row_count == 4
    assert result.total_row_count == 6
    assert result.fit_scope == "explicit_date_window"
    assert result.feature_names == ("Feature_A", "Feature_B")
    x_scaler = joblib.load(result.paths["x_original"])
    y_log_scaler = joblib.load(result.paths["y_log"])
    assert x_scaler.n_features_in_ == 2
    assert list(x_scaler.feature_names_in_) == ["Feature_A", "Feature_B"]
    np.testing.assert_allclose(x_scaler.data_max_, [4.0, 40.0])
    np.testing.assert_allclose(y_log_scaler.data_max_, [np.log1p(30.0)])

    manifest = json.loads(result.paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["fit_scope"] == "explicit_date_window"
    assert manifest["fit_row_count"] == 4
    assert manifest["total_row_count"] == 6
    assert manifest["target_transformations"] == {
        "original": "identity",
        "log": "log1p",
    }
    assert manifest["v1_artifacts_untouched"] is True


def test_fit_v2_scalers_rejects_existing_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v2_scaling, "project_root", lambda: tmp_path)
    input_path = tmp_path / "feature_ready_daily.csv"
    _frame().to_csv(input_path, index=False)
    output_dir = tmp_path / "models" / "v2" / "scalers" / "run"
    output_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        v2_scaling.fit_v2_scalers(input_path=input_path, output_dir=output_dir)


def test_fit_v2_scalers_rejects_negative_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v2_scaling, "project_root", lambda: tmp_path)
    input_path = tmp_path / "feature_ready_daily.csv"
    frame = _frame()
    frame.loc[0, "Wind_Production"] = -1.0
    frame.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="non-negative"):
        v2_scaling.fit_v2_scalers(
            input_path=input_path,
            output_dir=tmp_path / "models" / "v2" / "scalers" / "run",
        )


def test_fit_v2_scalers_rejects_sealed_test_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v2_scaling, "project_root", lambda: tmp_path)
    input_path = tmp_path / "feature_ready_daily.csv"
    _frame().to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="sealed v2 test period"):
        v2_scaling.fit_v2_scalers(
            input_path=input_path,
            output_dir=tmp_path / "models" / "v2" / "scalers" / "run",
            fit_end="2025-01-01",
        )
