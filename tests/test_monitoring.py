from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import wind_forecast.monitoring as monitoring
from wind_forecast.manifests import sha256_file
from wind_forecast.incremental import (
    IncrementalUpdateError,
    load_verified_current_state,
)
from wind_forecast.monitoring import (
    ConcurrentMonitoringError,
    MonitoringConfig,
    MonitoringError,
    load_prediction_evidence,
    plan_historical_monitoring,
    replay_prediction,
    run_historical_monitoring,
)
from wind_forecast.v2_features import TRANSFORMATION_VERSION
from scripts.run_historical_monitoring import parse_args


TARGET = "2026-03-29"
LAG_DAY = "2026-03-28"
LAG2_DAY = "2026-03-27"
LAG3_DAY = "2026-03-26"
NOW = datetime(2026, 4, 3, 11, 0, tzinfo=timezone.utc)
FEATURES = [
    "Month",
    "Wind_Production_Lag1",
    "Wind_Production_Rolling_Mean_3",
    "Average_Wind_Speed",
    "Average_Wind_Speed_Lag1",
    "Average_Temperature_Rolling_Mean_3",
]
FEATURE_VALUES = [3.0, 8.0, 10.0, 5.0, 4.0, 11.0]


def _json_hash(value: object) -> str:
    data = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(data).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_ren(path: Path, value: float, day: str = TARGET) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [f"{day}T00:00:00Z", f"{day}T00:15:00Z"],
            "wind_production_mw": [value, value],
            "unit": ["MW", "MW"],
            "source_date": [day, day],
            "retrieval_timestamp_utc": [
                "2026-04-02T10:00:00Z",
                "2026-04-02T10:00:00Z",
            ],
        }
    ).to_csv(path, index=False, lineterminator="\n")


def _ren_ref(
    path: Path,
    day: str,
    revision: int,
    *,
    supersedes_id: str | None = None,
) -> dict[str, object]:
    physical = sha256_file(path)
    semantic = sha256(f"{day}:{physical}".encode()).hexdigest()
    return {
        "logical_key": day,
        "status": "complete",
        "provider_finality": "unknown",
        "physical_sha256": physical,
        "semantic_sha256": semantic,
        "primary_path": str(path),
        "local_dates": [day],
        "revision": revision,
        "revision_id": f"ren-{day}-{semantic[:16]}",
        "supersedes_id": supersedes_id,
        "history": [],
        "supporting_observations": [],
    }


def _make_bundle(root: Path) -> Path:
    root.mkdir()
    x = pd.DataFrame([FEATURE_VALUES, FEATURE_VALUES], columns=FEATURES)
    model = DummyRegressor(strategy="constant", constant=50.0).fit(x, [1.0, 2.0])
    joblib.dump(model, root / "model.joblib")
    schema_hash = _json_hash(FEATURES)
    dataset_hash = "d" * 64
    documents: dict[str, dict[str, object]] = {
        "model_manifest.json": {
            "schema_version": "wind_forecast.v2_model_manifest.v1",
            "task": "daily_wind_production_historical_hindcast",
            "model_type": "dummy",
            "model_sha256": sha256_file(root / "model.joblib"),
            "dataset_version": "v2",
            "dataset_sha256": dataset_hash,
            "feature_names": FEATURES,
            "feature_schema_sha256": schema_hash,
            "scaler_required": False,
            "scaler": None,
            "reference_status": "selected_not_promoted",
        },
        "dataset_manifest.json": {
            "schema_version": "wind_forecast.v2_training_dataset.v1",
            "dataset_version": "v2",
            "transformation_version": TRANSFORMATION_VERSION,
            "sha256": dataset_hash,
            "target": "Wind_Production",
            "feature_names": FEATURES,
            "feature_schema_sha256": schema_hash,
        },
        "reference_decision.json": {
            "schema_version": "wind_forecast.v2_reference_decision.v1",
            "accepted_as_reference": True,
            "status": "selected_not_promoted",
            "automatic_promotion": False,
        },
        "run_summary.json": {
            "schema_version": "wind_forecast.v2_training_run.v1",
            "accepted_as_reference": True,
            "dataset_version": "v2",
            "dataset_sha256": dataset_hash,
            "scaler_required": False,
            "artifact_sha256": {},
        },
        "environment.json": {
            "schema_version": "wind_forecast.v2_environment.v1",
            "git_sha": "abc123",
            "git_dirty": True,
        },
        "leakage_audit.json": {
            "schema_version": "wind_forecast.v2_leakage_audit.v1",
            "forecast_contract": "historical_daily_hindcast",
            "passed": True,
        },
    }
    for name, document in documents.items():
        _write_json(root / name, document)
    summary = documents["run_summary.json"]
    summary["artifact_sha256"] = {
        name: sha256_file(root / name)
        for name in (
            "model.joblib",
            "model_manifest.json",
            "dataset_manifest.json",
            "reference_decision.json",
            "environment.json",
            "leakage_audit.json",
        )
    }
    _write_json(root / "run_summary.json", summary)
    return root


def _make_source(root: Path) -> dict[str, object]:
    source_root = root / "source"
    target_path = source_root / "ren-target.csv"
    lag_path = source_root / "ren-lag.csv"
    lag2_path = source_root / "ren-lag2.csv"
    lag3_path = source_root / "ren-lag3.csv"
    era_path = source_root / "era.csv"
    _write_ren(target_path, 10.0)
    _write_ren(lag_path, 4.0, LAG_DAY)
    _write_ren(lag2_path, 5.0, LAG2_DAY)
    _write_ren(lag3_path, 6.0, LAG3_DAY)
    era_path.write_text(
        "timestamp_utc,wind_speed_m_s,temperature_2m_c\n"
        "2026-03-26T00:00:00Z,2,8\n"
        "2026-03-27T00:00:00Z,3,9\n"
        "2026-03-28T00:00:00Z,4,10\n"
        "2026-03-29T00:00:00Z,5,12\n"
    )
    feature_path = source_root / "features.csv"
    pd.DataFrame(
        {
            "Date": [TARGET],
            "Wind_Production": [20.0],
            "Month": [3.0],
            "Wind_Production_Lag1": [8.0],
            "Wind_Production_Rolling_Mean_3": [10.0],
            "Average_Wind_Speed": [5.0],
            "Average_Wind_Speed_Lag1": [4.0],
            "Average_Temperature_Rolling_Mean_3": [11.0],
        }
    ).to_csv(feature_path, index=False, lineterminator="\n")
    manifest_path = source_root / "run-manifest.json"
    _write_json(
        manifest_path,
        {
            "versions": {"features": TRANSFORMATION_VERSION},
            "git_commit": "source-commit",
        },
    )
    era_hash = sha256_file(era_path)
    return {
        "schema_version": "wind_forecast.v2_incremental_state.v1",
        "generation": 1,
        "release_id": "source-release-1",
        "manifest_path": str(manifest_path),
        "sources": {
            "ren": {
                TARGET: _ren_ref(target_path, TARGET, 1),
                LAG_DAY: _ren_ref(lag_path, LAG_DAY, 1),
                LAG2_DAY: _ren_ref(lag2_path, LAG2_DAY, 1),
                LAG3_DAY: _ren_ref(lag3_path, LAG3_DAY, 1),
            },
            "era5_land": {
                "station_id=1/month=2026-03": {
                    "logical_key": "station_id=1/month=2026-03",
                    "status": "complete",
                    "physical_sha256": era_hash,
                    "semantic_sha256": era_hash,
                    "primary_path": str(era_path),
                    "local_dates": [LAG3_DAY, LAG2_DAY, LAG_DAY, TARGET],
                    "revision": 1,
                    "revision_id": "era-r1",
                }
            },
        },
        "partitions": {
            "features": {
                TARGET: {
                    "partition_key": "features-r1",
                    "feature_ready": True,
                    "files": {
                        "feature_ready": {
                            "path": str(feature_path),
                            "sha256": sha256_file(feature_path),
                        }
                    },
                }
            }
        },
    }


@pytest.fixture
def environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[MonitoringConfig, dict[str, object]]:
    bundle = _make_bundle(tmp_path / "bundle")
    source = _make_source(tmp_path)
    monkeypatch.setattr(
        monitoring,
        "load_verified_current_state",
        lambda _root: json.loads(json.dumps(source)),
    )
    config = MonitoringConfig(
        source_store_root=tmp_path / "incremental",
        monitoring_store_root=tmp_path / "monitoring",
        model_bundle=bundle,
        through_date=TARGET,
        activation_date=TARGET,
        now_utc=NOW,
    )
    return config, source


def test_first_run_is_append_only_target_free_and_replayable(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    plan = plan_historical_monitoring(config)
    assert plan.eligible_dates == (TARGET,)
    assert not config.monitoring_store_root.exists()
    first = run_historical_monitoring(config)
    assert first.status == "succeeded"
    assert len(first.prediction_ids) == 1
    assert len(first.actual_revision_ids) == 1
    assert len(first.metric_revision_ids) == 1

    evidence = load_prediction_evidence(config.monitoring_store_root, first.prediction_ids[0])
    prediction = evidence["prediction"]
    snapshot = evidence["model_input_snapshot"]
    assert prediction["prediction_mode"] == "historical_hindcast"
    assert prediction["issuance_kind"] == "scheduled"
    assert prediction["scheduled_at_local"] == "2026-04-03T12:00:00+01:00"
    assert prediction["forecast_horizon"] is None
    assert prediction["target_scale"] == "sum_of_15_minute_MW_observations"
    assert snapshot["feature_names"] == FEATURES
    assert snapshot["feature_values"] == FEATURE_VALUES
    assert "Date" not in snapshot["feature_names"]
    assert "Wind_Production" not in snapshot["feature_names"]
    assert snapshot["dependencies"]["Month"]["derivation"] == "calendar_only"
    lag_dependencies = snapshot["dependencies"]["Wind_Production_Lag1"]
    assert lag_dependencies["source_revisions"][0]["source_date"] == LAG_DAY
    ren_rolling = snapshot["dependencies"]["Wind_Production_Rolling_Mean_3"]
    assert {item["source_date"] for item in ren_rolling["source_revisions"]} == {
        LAG_DAY,
        LAG2_DAY,
        LAG3_DAY,
    }
    era_direct = snapshot["dependencies"]["Average_Wind_Speed"]
    assert {item["source_date"] for item in era_direct["source_revisions"]} == {
        TARGET
    }
    era_lag = snapshot["dependencies"]["Average_Wind_Speed_Lag1"]
    assert {item["source_date"] for item in era_lag["source_revisions"]} == {
        LAG_DAY
    }
    era_rolling = snapshot["dependencies"]["Average_Temperature_Rolling_Mean_3"]
    assert {item["source_date"] for item in era_rolling["source_revisions"]} == {
        LAG_DAY,
        LAG2_DAY,
        LAG3_DAY,
    }
    assert replay_prediction(config.monitoring_store_root, first.prediction_ids[0])[
        "equivalent"
    ]

    prediction_bytes = (
        config.monitoring_store_root
        / "predictions"
        / f"{first.prediction_ids[0]}.json"
    ).read_bytes()
    second = run_historical_monitoring(config)
    assert second.status == "no_op"
    assert second.prediction_ids == ()
    assert len(list((config.monitoring_store_root / "predictions").glob("*.json"))) == 1
    assert (
        config.monitoring_store_root
        / "predictions"
        / f"{first.prediction_ids[0]}.json"
    ).read_bytes() == prediction_bytes


def test_actual_revision_adds_metric_without_changing_as_issued(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, source = environment
    first = run_historical_monitoring(config)
    pointer_path = config.monitoring_store_root / "state" / "current.json"
    before = json.loads(pointer_path.read_text())
    target_ref = source["sources"]["ren"][TARGET]
    target_path = Path(target_ref["primary_path"])
    _write_ren(target_path, 20.0)
    source["sources"]["ren"][TARGET] = _ren_ref(
        target_path,
        TARGET,
        2,
        supersedes_id=target_ref["revision_id"],
    )

    revised = run_historical_monitoring(config)
    after = json.loads(pointer_path.read_text())
    assert revised.prediction_ids == ()
    assert len(revised.actual_revision_ids) == 1
    assert len(revised.metric_revision_ids) == 1
    assert after["as_issued"] == before["as_issued"]
    assert len(list((config.monitoring_store_root / "actuals").glob("*.json"))) == 2
    metric = json.loads(
        (
            config.monitoring_store_root
            / "metrics"
            / f"{revised.metric_revision_ids[0]}.json"
        ).read_text()
    )
    assert metric["supersedes_id"] == first.metric_revision_ids[0]

    revised_source = source["sources"]["ren"][TARGET]
    _write_ren(target_path, 10.0)
    source["sources"]["ren"][TARGET] = _ren_ref(
        target_path,
        TARGET,
        3,
        supersedes_id=revised_source["revision_id"],
    )
    returned = run_historical_monitoring(config)
    final_pointer = json.loads(pointer_path.read_text())
    assert len(returned.actual_revision_ids) == 1
    assert len(returned.metric_revision_ids) == 1
    assert len(list((config.monitoring_store_root / "actuals").glob("*.json"))) == 3
    final_actual = json.loads(
        (
            config.monitoring_store_root
            / "actuals"
            / f"{returned.actual_revision_ids[0]}.json"
        ).read_text()
    )
    assert final_actual["source_revision"] == 3
    assert final_actual["source_revision_id"] == target_ref["revision_id"]
    assert final_actual["supersedes_id"] == revised.actual_revision_ids[0]
    assert final_pointer["actuals"][TARGET] == returned.actual_revision_ids[0]
    final_metric = json.loads(
        (
            config.monitoring_store_root
            / "metrics"
            / f"{returned.metric_revision_ids[0]}.json"
        ).read_text()
    )
    assert final_metric["supersedes_id"] == revised.metric_revision_ids[0]


def test_feature_revision_creates_restatement_and_preserves_as_issued(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, source = environment
    first = run_historical_monitoring(config)
    initial_pointer = json.loads(
        (config.monitoring_store_root / "state" / "current.json").read_text()
    )
    lag_ref = source["sources"]["ren"][LAG_DAY]
    lag_path = Path(lag_ref["primary_path"])
    _write_ren(lag_path, 6.0)
    source["sources"]["ren"][LAG_DAY] = _ren_ref(
        lag_path,
        LAG_DAY,
        2,
        supersedes_id=lag_ref["revision_id"],
    )
    feature_ref = source["partitions"]["features"][TARGET]["files"]["feature_ready"]
    feature_path = Path(feature_ref["path"])
    frame = pd.read_csv(feature_path)
    frame["Wind_Production_Lag1"] = 12.0
    frame["Wind_Production_Rolling_Mean_3"] = 34.0 / 3.0
    frame.to_csv(feature_path, index=False, lineterminator="\n")
    feature_ref["sha256"] = sha256_file(feature_path)
    source["partitions"]["features"][TARGET]["partition_key"] = "features-r2"

    revised = run_historical_monitoring(config)
    pointer = json.loads(
        (config.monitoring_store_root / "state" / "current.json").read_text()
    )
    assert len(revised.prediction_ids) == 1
    restated_id = revised.prediction_ids[0]
    assert pointer["as_issued"] == initial_pointer["as_issued"]
    assert pointer["restated"][TARGET] == restated_id
    restated = load_prediction_evidence(config.monitoring_store_root, restated_id)
    assert restated["prediction"]["view"] == "restated"
    assert restated["prediction"]["issuance_kind"] == "restatement"
    assert restated["prediction"]["restates_prediction_id"] == first.prediction_ids[0]
    assert restated["model_input_snapshot"]["feature_values"] == [
        3.0,
        12.0,
        34.0 / 3.0,
        5.0,
        4.0,
        11.0,
    ]
    assert replay_prediction(config.monitoring_store_root, restated_id)["equivalent"]

    revised_lag_ref = source["sources"]["ren"][LAG_DAY]
    _write_ren(lag_path, 4.0, LAG_DAY)
    source["sources"]["ren"][LAG_DAY] = _ren_ref(
        lag_path,
        LAG_DAY,
        3,
        supersedes_id=revised_lag_ref["revision_id"],
    )
    frame["Wind_Production_Lag1"] = 8.0
    frame["Wind_Production_Rolling_Mean_3"] = 10.0
    frame.to_csv(feature_path, index=False, lineterminator="\n")
    feature_ref["sha256"] = sha256_file(feature_path)
    source["partitions"]["features"][TARGET]["partition_key"] = "features-r3"
    returned = run_historical_monitoring(config)
    returned_pointer = json.loads(
        (config.monitoring_store_root / "state" / "current.json").read_text()
    )
    assert len(returned.prediction_ids) == 1
    returned_id = returned.prediction_ids[0]
    assert returned_pointer["restated"][TARGET] == returned_id
    returned_evidence = load_prediction_evidence(
        config.monitoring_store_root, returned_id
    )
    assert returned_evidence["prediction"]["supersedes_id"] == restated_id
    assert returned_evidence["prediction"]["restates_prediction_id"] == (
        first.prediction_ids[0]
    )
    assert returned_evidence["model_input_snapshot"]["feature_values"] == (
        FEATURE_VALUES
    )
    lag_occurrence = returned_evidence["model_input_snapshot"]["dependencies"][
        "Wind_Production_Lag1"
    ]["source_revisions"][0]
    assert lag_occurrence["revision"] == 3

    repeated = run_historical_monitoring(config)
    assert repeated.status == "no_op"
    assert len(list((config.monitoring_store_root / "predictions").glob("*.json"))) == 3


def test_failure_after_prediction_is_reconciled_without_duplicate(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, source = environment

    def fail(stage: str) -> None:
        if stage == "after_prediction":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        run_historical_monitoring(config, failure_hook=fail)
    predictions = list((config.monitoring_store_root / "predictions").glob("*.json"))
    assert len(predictions) == 1
    assert not (config.monitoring_store_root / "state" / "current.json").exists()

    target_source = source["sources"]["ren"].pop(TARGET)
    waiting = run_historical_monitoring(config)
    assert waiting.metric_revision_ids == ()
    assert not list((config.monitoring_store_root / "metrics").glob("*.json"))
    source["sources"]["ren"][TARGET] = target_source
    recovered = run_historical_monitoring(config)
    assert recovered.prediction_ids == ()
    assert len(list((config.monitoring_store_root / "predictions").glob("*.json"))) == 1
    pointer = json.loads(
        (config.monitoring_store_root / "state" / "current.json").read_text()
    )
    assert pointer["as_issued"][TARGET] == predictions[0].stem


def test_activation_backfill_and_contract_validation(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    with pytest.raises(MonitoringError, match="activation_date"):
        plan_historical_monitoring(
            replace(
                config,
                monitoring_store_root=config.monitoring_store_root.parent
                / "missing-activation",
                activation_date=None,
            )
        )
    too_early = replace(config, now_utc=datetime(2026, 4, 3, 10, 59, tzinfo=timezone.utc))
    assert plan_historical_monitoring(too_early).eligible_dates == ()
    with pytest.raises(ValueError, match="historical_hindcast"):
        replace(config, prediction_mode="ex_ante_forecast")
    with pytest.raises(ValueError, match="forecast_horizon"):
        replace(config, forecast_horizon=1)
    with pytest.raises(ValueError, match="target scale"):
        replace(config, target_scale="MWh")
    run_historical_monitoring(config)
    with pytest.raises(MonitoringError, match="immutable"):
        plan_historical_monitoring(replace(config, activation_date="2026-03-30"))
    with pytest.raises(MonitoringError, match="precede"):
        plan_historical_monitoring(
            replace(config, backfill_start=TARGET, backfill_end=TARGET)
        )


def test_d7_gap_does_not_block_a_later_eligible_date(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    config = replace(
        config,
        activation_date="2026-03-27",
        now_utc=datetime(2026, 4, 5, 11, 0, tzinfo=timezone.utc),
    )
    plan = plan_historical_monitoring(config)
    assert plan.date_states["2026-03-27"] == "source_late"
    assert TARGET in plan.eligible_dates
    result = run_historical_monitoring(config)
    assert len(result.prediction_ids) == 1
    pointer = json.loads(
        (config.monitoring_store_root / "state" / "current.json").read_text()
    )
    assert pointer["date_states"]["2026-03-27"] == "source_late"
    assert pointer["date_states"][TARGET] == "issued"


def test_explicit_pre_activation_backfill_is_labelled(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    config = replace(
        config,
        monitoring_store_root=config.monitoring_store_root.parent / "backfill-ledger",
        activation_date="2026-03-30",
        through_date="2026-03-30",
        backfill_start=TARGET,
        backfill_end=TARGET,
    )
    result = run_historical_monitoring(config)
    assert len(result.prediction_ids) == 1
    evidence = load_prediction_evidence(
        config.monitoring_store_root, result.prediction_ids[0]
    )
    assert evidence["prediction"]["issuance_kind"] == "explicit_backfill"


def test_semantically_equivalent_physical_revision_does_not_restate(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, source = environment
    run_historical_monitoring(config)
    lag_ref = source["sources"]["ren"][LAG_DAY]
    lag_path = Path(lag_ref["primary_path"])
    frame = pd.read_csv(lag_path)
    frame.to_csv(lag_path, index=False, lineterminator="\r\n")
    lag_ref["physical_sha256"] = sha256_file(lag_path)
    repeated = run_historical_monitoring(config)
    assert repeated.status == "no_op"
    assert len(list((config.monitoring_store_root / "predictions").glob("*.json"))) == 1


def test_corrupt_current_pointer_fails_closed(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    run_historical_monitoring(config)
    pointer_path = config.monitoring_store_root / "state" / "current.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["as_issued"][TARGET] = "0" * 64
    _write_json(pointer_path, pointer)
    with pytest.raises(MonitoringError, match="Invalid JSON|corrupt"):
        plan_historical_monitoring(config)


def test_corrupt_snapshot_fails_closed(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    result = run_historical_monitoring(config)
    evidence = load_prediction_evidence(config.monitoring_store_root, result.prediction_ids[0])
    model_path = Path(evidence["model_snapshot"]["files"]["model.joblib"]["path"])
    model_path.write_bytes(model_path.read_bytes() + b"corrupt")
    with pytest.raises(MonitoringError, match="corrupt"):
        load_prediction_evidence(config.monitoring_store_root, result.prediction_ids[0])


def test_rejected_bundle_fails_before_ledger_writes(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    decision_path = config.model_bundle / "reference_decision.json"
    decision = json.loads(decision_path.read_text())
    decision["accepted_as_reference"] = False
    _write_json(decision_path, decision)
    with pytest.raises(MonitoringError, match="not accepted"):
        run_historical_monitoring(config)
    assert not config.monitoring_store_root.exists()


def test_content_addressed_path_id_and_input_corruption_fail_closed(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    result = run_historical_monitoring(config)
    prediction_id = result.prediction_ids[0]
    prediction_path = (
        config.monitoring_store_root / "predictions" / f"{prediction_id}.json"
    )
    wrong_id = "f" * 64
    wrong_path = config.monitoring_store_root / "predictions" / f"{wrong_id}.json"
    wrong_path.write_bytes(prediction_path.read_bytes())
    with pytest.raises(MonitoringError, match="corrupt"):
        load_prediction_evidence(config.monitoring_store_root, wrong_id)

    prediction = json.loads(prediction_path.read_text())
    input_path = (
        config.monitoring_store_root
        / "input_snapshots"
        / f"{prediction['model_input_snapshot_id']}.json"
    )
    input_snapshot = json.loads(input_path.read_text())
    input_snapshot["feature_values"][0] = 99.0
    _write_json(input_path, input_snapshot)
    with pytest.raises(MonitoringError, match="corrupt"):
        load_prediction_evidence(config.monitoring_store_root, prediction_id)


def test_bundle_manifest_checksum_and_feature_order_corruption_fail_closed(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    manifest_path = config.model_bundle / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_type"] = "tampered"
    _write_json(manifest_path, manifest)
    with pytest.raises(MonitoringError, match="checksum"):
        plan_historical_monitoring(config)


def test_bundle_feature_order_mismatch_fails_closed(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    dataset_path = config.model_bundle / "dataset_manifest.json"
    dataset = json.loads(dataset_path.read_text())
    dataset["feature_names"] = list(reversed(dataset["feature_names"]))
    _write_json(dataset_path, dataset)
    with pytest.raises(MonitoringError, match="feature order"):
        plan_historical_monitoring(config)


def test_exclusive_lock_is_enforced(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, _ = environment
    lock = config.monitoring_store_root / "state" / "monitoring.lock"
    _write_json(lock, {"run_id": "other", "pid": 1, "host": "test"})
    with pytest.raises(ConcurrentMonitoringError, match="owns the lock"):
        run_historical_monitoring(config)


def test_autumn_dst_schedule_keeps_lisbon_offset() -> None:
    scheduled = monitoring._scheduled(date(2026, 10, 25))
    assert scheduled.isoformat() == "2026-10-30T12:00:00+00:00"
    assert scheduled.astimezone(timezone.utc).isoformat() == (
        "2026-10-30T12:00:00+00:00"
    )


def test_incompatible_restatement_transformation_is_blocked(
    environment: tuple[MonitoringConfig, dict[str, object]],
) -> None:
    config, source = environment
    run_historical_monitoring(config)
    lag_ref = source["sources"]["ren"][LAG_DAY]
    lag_path = Path(lag_ref["primary_path"])
    _write_ren(lag_path, 6.0, LAG_DAY)
    source["sources"]["ren"][LAG_DAY] = _ren_ref(
        lag_path,
        LAG_DAY,
        2,
        supersedes_id=lag_ref["revision_id"],
    )
    feature_ref = source["partitions"]["features"][TARGET]["files"][
        "feature_ready"
    ]
    feature_path = Path(feature_ref["path"])
    frame = pd.read_csv(feature_path)
    frame["Wind_Production_Lag1"] = 12.0
    frame.to_csv(feature_path, index=False, lineterminator="\n")
    feature_ref["sha256"] = sha256_file(feature_path)
    manifest_path = Path(source["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["versions"]["features"] = "incompatible-v3"
    _write_json(manifest_path, manifest)

    result = run_historical_monitoring(config)
    assert result.prediction_ids == ()
    assert result.blocked_dates[TARGET].startswith("blocked_prerequisite")
    pointer = json.loads(
        (config.monitoring_store_root / "state" / "current.json").read_text()
    )
    assert pointer["date_states"][TARGET] == "blocked_prerequisite"


def test_monitoring_plan_uses_real_verified_phase8_loader(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "bundle")
    source = _make_source(tmp_path)
    source_store = tmp_path / "incremental"
    manifest_path = Path(source["manifest_path"])
    source["manifest_sha256"] = sha256_file(manifest_path)
    _write_json(source_store / "state" / "current.json", source)
    loaded = load_verified_current_state(source_store)
    assert loaded["release_id"] == "source-release-1"
    config = MonitoringConfig(
        source_store_root=source_store,
        monitoring_store_root=tmp_path / "monitoring",
        model_bundle=bundle,
        through_date=TARGET,
        activation_date=TARGET,
        now_utc=NOW,
    )
    assert plan_historical_monitoring(config).eligible_dates == (TARGET,)
    feature_path = Path(
        source["partitions"]["features"][TARGET]["files"]["feature_ready"][
            "path"
        ]
    )
    feature_path.write_bytes(feature_path.read_bytes() + b"\n")
    with pytest.raises(IncrementalUpdateError, match="corrupt"):
        load_verified_current_state(source_store)


def test_cli_requires_paired_backfill_arguments() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--through-date",
                TARGET,
                "--model-bundle",
                "bundle",
                "--backfill-start",
                LAG_DAY,
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(["--through-date", TARGET])
    args = parse_args(
        [
            "--through-date",
            TARGET,
            "--model-bundle",
            "bundle",
            "--activation-date",
            TARGET,
            "--dry-run",
        ]
    )
    assert args.through_date == TARGET
    assert args.activation_date == TARGET
    assert args.dry_run is True
