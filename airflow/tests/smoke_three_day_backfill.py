"""Offline Airflow DAG smoke for exactly three consecutive intervals."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from datetime import timedelta
from pathlib import Path
import tempfile

import pendulum


def _load_dag_module():
    path = Path("/opt/airflow/dags/wind_forecast_historical_batch_v1.py")
    spec = importlib.util.spec_from_file_location("wind_forecast_airflow_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_dag_module()
    root = Path(tempfile.mkdtemp(prefix="wind-airflow-smoke-"))
    created = {"source": set(), "prediction": set(), "report": set()}
    deployment_checks: list[str | None] = []
    model_era_id = "smoke-model-era"

    def deployment(*, expected_model_era_id: str | None = None) -> dict:
        if expected_model_era_id is not None:
            assert expected_model_era_id == model_era_id
        deployment_checks.append(expected_model_era_id)
        return {
            "status": "verified",
            "model_era_id": model_era_id,
            "deployment_id": "smoke-deployment",
            "model_version": "1",
        }

    def availability(*, through_date: str) -> dict:
        return {"status": "planned", "through_date": str(through_date)}

    def update(*, through_date: str) -> dict:
        value = str(through_date)
        path = root / "source" / f"{value}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps({"through_date": value}), encoding="utf-8")
            created["source"].add(value)
        digest = sha256(path.read_bytes()).hexdigest()
        return {
            "status": "succeeded",
            "manifest_path": str(path),
            "manifest_sha256": digest,
        }

    def predict(
        *,
        through_date: str,
        source_manifest_path: str,
        source_manifest_sha256: str,
    ) -> dict:
        value = str(through_date)
        assert sha256(Path(source_manifest_path).read_bytes()).hexdigest() == (
            source_manifest_sha256
        )
        created["prediction"].add(value)
        return {"status": "succeeded", "current_state_path": source_manifest_path}

    def report(
        *,
        through_date: str,
        source_manifest_path: str,
        source_manifest_sha256: str,
    ) -> dict:
        value = str(through_date)
        assert sha256(Path(source_manifest_path).read_bytes()).hexdigest() == (
            source_manifest_sha256
        )
        created["report"].add(value)
        return {"status": "succeeded", "report_path": source_manifest_path}

    dag = module.dag
    dag.get_task("deployment_preflight").python_callable = deployment
    dag.get_task("availability_plan").python_callable = availability
    dag.get_task("dataset_update").python_callable = update
    dag.get_task("predict_reconcile").python_callable = predict
    dag.get_task("drift_publish").python_callable = report
    dag.get_task("deployment_postcheck").python_callable = deployment
    for task in dag.tasks:
        task.retries = 0
        task.retry_delay = timedelta(0)

    dates = (
        pendulum.datetime(2026, 7, 1, 12, tz="Europe/Lisbon"),
        pendulum.datetime(2026, 7, 2, 12, tz="Europe/Lisbon"),
        pendulum.datetime(2026, 7, 3, 12, tz="Europe/Lisbon"),
    )
    for logical_date in dates:
        run = dag.test(logical_date=logical_date)
        assert run.state.value == "success"

    expected = {str(item.date()) for item in dates}
    assert created == {
        "source": expected,
        "prediction": expected,
        "report": expected,
    }
    assert deployment_checks == [
        None,
        model_era_id,
        None,
        model_era_id,
        None,
        model_era_id,
    ]

    # The synthetic boundaries are idempotent: repeating the same selections
    # does not create another immutable source record.
    before = len(list((root / "source").glob("*.json")))
    for value in sorted(expected):
        update(through_date=value)
    assert len(list((root / "source").glob("*.json"))) == before == 3
    print(json.dumps({"status": "succeeded", "intervals": sorted(expected)}))


if __name__ == "__main__":
    main()
