"""Daily delayed-hindcast workflow for the local Airflow 3.3 stack."""

from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from airflow.timetables.trigger import CronTriggerTimetable

from wind_forecast.airflow_orchestration import (
    AirflowBatchConfig,
    run_availability_plan,
    run_dataset_update,
    run_drift_publish,
    run_predict_reconcile,
)


DAG_ID = "wind_forecast_historical_batch_v1"
TIMEZONE = "Europe/Lisbon"


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be configured explicitly.")
    return value


def _boolean_environment(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false.")
    return normalized == "true"


def _optional_path_environment(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _config() -> AirflowBatchConfig:
    return AirflowBatchConfig(
        model_bundle=Path(_required_environment("WIND_FORECAST_BATCH_MODEL_BUNDLE")),
        calibration_dir=Path(
            _required_environment("WIND_FORECAST_BATCH_CALIBRATION_DIR")
        ),
        source_store_root=Path(
            os.environ.get(
                "WIND_FORECAST_SOURCE_STORE_ROOT",
                "/opt/wind-energy-forecast/data/processed/v2/incremental_update",
            )
        ),
        monitoring_store_root=Path(
            os.environ.get(
                "WIND_FORECAST_MONITORING_STORE_ROOT",
                "/opt/wind-energy-forecast/data/processed/v2/monitoring",
            )
        ),
        activation_date=_required_environment(
            "WIND_FORECAST_AIRFLOW_ACTIVATION_DATE"
        ),
        raw_store_root=_optional_path_environment("WIND_FORECAST_RAW_STORE_ROOT"),
        ren_root=_optional_path_environment("WIND_FORECAST_REN_ROOT"),
        era5_root=_optional_path_environment("WIND_FORECAST_ERA5_ROOT"),
        station_mapping=_optional_path_environment("WIND_FORECAST_STATION_MAPPING"),
        v1_feature_table=_optional_path_environment("WIND_FORECAST_V1_FEATURE_TABLE"),
        baseline_integrated_root=_optional_path_environment(
            "WIND_FORECAST_BASELINE_INTEGRATED_ROOT"
        ),
        baseline_feature_root=_optional_path_environment(
            "WIND_FORECAST_BASELINE_FEATURE_ROOT"
        ),
        bootstrap_start=os.environ.get("WIND_FORECAST_BOOTSTRAP_START"),
        bootstrap_end=os.environ.get("WIND_FORECAST_BOOTSTRAP_END"),
        no_source_refresh=_boolean_environment(
            "WIND_FORECAST_NO_SOURCE_REFRESH"
        ),
        fail_on_active_alert=_boolean_environment(
            "WIND_FORECAST_FAIL_ON_ACTIVE_ALERT"
        ),
    )


def _run_availability(*, through_date: str) -> dict:
    return run_availability_plan(_config(), through_date)


def _run_update(*, through_date: str) -> dict:
    return run_dataset_update(_config(), through_date)


def _run_predict(
    *,
    through_date: str,
    source_manifest_path: str,
    source_manifest_sha256: str,
) -> dict:
    return run_predict_reconcile(
        _config(),
        through_date,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
    )


def _run_report(
    *,
    through_date: str,
    source_manifest_path: str,
    source_manifest_sha256: str,
) -> dict:
    return run_drift_publish(
        _config(),
        through_date,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
    )


activation_date = pendulum.parse(
    _required_environment("WIND_FORECAST_AIRFLOW_ACTIVATION_DATE"),
    tz=TIMEZONE,
)
through_date = "{{ data_interval_end.in_timezone('Europe/Lisbon').date() }}"

with DAG(
    dag_id=DAG_ID,
    description="Local delayed historical wind-forecast batch.",
    start_date=activation_date,
    schedule=CronTriggerTimetable(
        "0 12 * * *",
        timezone=TIMEZONE,
        interval=timedelta(days=1),
    ),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=6),
    tags=["wind-forecast", "historical-batch", "local"],
) as dag:
    availability_plan = PythonOperator(
        task_id="availability_plan",
        python_callable=_run_availability,
        op_kwargs={"through_date": through_date},
        retries=0,
        execution_timeout=timedelta(minutes=10),
    )
    dataset_update = PythonOperator(
        task_id="dataset_update",
        python_callable=_run_update,
        op_kwargs={"through_date": through_date},
        multiple_outputs=True,
        retries=2,
        retry_delay=timedelta(minutes=30),
        retry_exponential_backoff=True,
        execution_timeout=timedelta(hours=4),
    )
    predict_reconcile = PythonOperator(
        task_id="predict_reconcile",
        python_callable=_run_predict,
        op_kwargs={
            "through_date": through_date,
            "source_manifest_path": dataset_update.output["manifest_path"],
            "source_manifest_sha256": dataset_update.output["manifest_sha256"],
        },
        retries=2,
        retry_delay=timedelta(minutes=10),
        execution_timeout=timedelta(hours=1),
    )
    drift_publish = PythonOperator(
        task_id="drift_publish",
        python_callable=_run_report,
        op_kwargs={
            "through_date": through_date,
            "source_manifest_path": dataset_update.output["manifest_path"],
            "source_manifest_sha256": dataset_update.output["manifest_sha256"],
        },
        retries=1,
        retry_delay=timedelta(minutes=10),
        execution_timeout=timedelta(minutes=30),
    )

    availability_plan >> dataset_update >> predict_reconcile >> drift_publish
