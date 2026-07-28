"""Monthly recommendation-only controlled-retraining workflow."""

from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from airflow.task.trigger_rule import TriggerRule
from airflow.timetables.trigger import CronTriggerTimetable

from wind_forecast.monthly_governance import (
    MonthlyGovernanceConfig,
    canonical_monthly_logical_time,
    run_monthly_governance,
)
from wind_forecast.retraining_policy import RetrainingPolicy
from wind_forecast.scheduler_ownership import (
    acquire_scheduler_lease,
    release_scheduler_lease,
)


DAG_ID = "wind_forecast_monthly_governance_v1"
TIMEZONE = "Europe/Lisbon"


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be configured explicitly.")
    return value


def _acquire(*, run_id: str) -> dict:
    lease = acquire_scheduler_lease(
        Path(_required_environment("WIND_FORECAST_SCHEDULER_STATE_ROOT")),
        _required_environment("WIND_FORECAST_ENVIRONMENT_ID"),
        "airflow",
        workflow=DAG_ID,
        run_id=run_id,
    )
    return lease.to_dict()


def _recommend(*, evaluation_period: str) -> dict:
    policy_path = Path(
        _required_environment("WIND_FORECAST_RETRAINING_POLICY_PATH")
    )
    logical = canonical_monthly_logical_time(
        RetrainingPolicy.load(policy_path),
        evaluation_period,
    )
    return run_monthly_governance(
        MonthlyGovernanceConfig(
            policy_path=policy_path,
            monitoring_policy_path=Path(
                _required_environment("WIND_FORECAST_MONITORING_POLICY_PATH")
            ),
            monitoring_store_root=Path(
                _required_environment("WIND_FORECAST_MONITORING_STORE_ROOT")
            ),
            deployment_root=Path(
                _required_environment("WIND_FORECAST_DEPLOYMENT_ROOT")
            ),
            logical_at_utc=logical,
            output_root=Path(
                _required_environment(
                    "WIND_FORECAST_MONTHLY_RECOMMENDATION_ROOT"
                )
            ),
            evaluation_output_root=Path(
                _required_environment(
                    "WIND_FORECAST_RETRAINING_EVALUATION_ROOT"
                )
            ),
        )
    ).summary()


def _release(*, lease_id: str) -> dict:
    release_scheduler_lease(
        Path(_required_environment("WIND_FORECAST_SCHEDULER_STATE_ROOT")),
        _required_environment("WIND_FORECAST_ENVIRONMENT_ID"),
        lease_id,
    )
    return {"status": "released", "lease_id": lease_id}


activation_date = pendulum.parse(
    _required_environment("WIND_FORECAST_AIRFLOW_ACTIVATION_DATE"),
    tz=TIMEZONE,
)
evaluation_period = (
    "{{ data_interval_end.in_timezone('Europe/Lisbon').strftime('%Y-%m') }}"
)

with DAG(
    dag_id=DAG_ID,
    description="Monthly retraining and stability recommendations only.",
    start_date=activation_date,
    schedule=CronTriggerTimetable(
        "0 13 8 * *",
        timezone=TIMEZONE,
        interval=timedelta(days=1),
    ),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
    tags=["wind-forecast", "controlled-retraining", "recommendation-only"],
) as dag:
    scheduler_lease = PythonOperator(
        task_id="scheduler_lease",
        python_callable=_acquire,
        op_kwargs={"run_id": "{{ run_id }}"},
        multiple_outputs=True,
        retries=0,
        execution_timeout=timedelta(minutes=5),
    )
    monthly_recommendation = PythonOperator(
        task_id="monthly_recommendation",
        python_callable=_recommend,
        op_kwargs={"evaluation_period": evaluation_period},
        retries=2,
        retry_delay=timedelta(minutes=15),
        execution_timeout=timedelta(hours=1),
    )
    scheduler_release = PythonOperator(
        task_id="scheduler_release",
        python_callable=_release,
        op_kwargs={"lease_id": scheduler_lease.output["lease_id"]},
        retries=0,
        trigger_rule=TriggerRule.ALL_DONE,
        execution_timeout=timedelta(minutes=5),
    )

    scheduler_lease >> monthly_recommendation >> scheduler_release
