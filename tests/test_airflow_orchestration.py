from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from wind_forecast import airflow_orchestration
from wind_forecast.airflow_orchestration import (
    AirflowBatchConfig,
    AirflowStageError,
    run_dataset_update,
    run_predict_reconcile,
    through_date_from_interval_end,
    validate_three_day_backfill,
)


def _config(tmp_path: Path) -> AirflowBatchConfig:
    return AirflowBatchConfig(
        model_bundle=tmp_path / "model",
        calibration_dir=tmp_path / "calibration",
        source_store_root=tmp_path / "source",
        monitoring_store_root=tmp_path / "monitoring",
        activation_date="2026-07-01",
        no_source_refresh=True,
    )


def test_through_date_uses_lisbon_calendar() -> None:
    value = datetime(2026, 7, 26, 23, 30, tzinfo=timezone.utc)
    assert through_date_from_interval_end(value) == "2026-07-27"


def test_validation_backfill_is_exactly_three_consecutive_dates() -> None:
    assert validate_three_day_backfill("2026-07-01", "2026-07-03") == (
        "2026-07-01",
        "2026-07-03",
    )
    with pytest.raises(ValueError, match="exactly 3 days"):
        validate_three_day_backfill("2026-07-01", "2026-07-04")


def test_dataset_xcom_is_small_and_checksum_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
    monkeypatch.setattr(
        airflow_orchestration,
        "_run_cli_json",
        lambda command, timeout: {
            "status": "succeeded",
            "manifest_path": str(manifest),
            "large_internal_events": list(range(100)),
        },
    )
    result = run_dataset_update(_config(tmp_path), "2026-07-26")
    assert set(result) == {
        "schema_version",
        "stage",
        "status",
        "manifest_path",
        "manifest_sha256",
    }


def test_downstream_rejects_modified_upstream_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(AirflowStageError, match="checksum"):
        run_predict_reconcile(
            _config(tmp_path),
            "2026-07-26",
            source_manifest_path=str(manifest),
            source_manifest_sha256="0" * 64,
        )


def test_alert_exit_two_fails_only_when_cli_requests_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        returncode = 2
        stdout = '{"status":"completed_with_alerts"}'
        stderr = ""

    monkeypatch.setattr(
        airflow_orchestration.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )
    with pytest.raises(AirflowStageError, match="code 2"):
        airflow_orchestration._run_cli_json(["report"], 10)


def test_dag_source_has_no_provider_or_dataset_access_at_import() -> None:
    source = (
        Path(__file__).parents[1]
        / "airflow"
        / "dags"
        / "wind_forecast_historical_batch_v1.py"
    ).read_text(encoding="utf-8")
    compile(source, "wind_forecast_historical_batch_v1.py", "exec")
    assert "requests." not in source
    assert "read_csv(" not in source
    assert "open(" not in source
