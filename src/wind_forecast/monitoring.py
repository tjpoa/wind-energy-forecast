"""Append-only evidence ledger for the accepted historical hindcast contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from wind_forecast.incremental import load_verified_current_state
from wind_forecast.manifests import sha256_file
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN
from wind_forecast.v2_features import TRANSFORMATION_VERSION


CONTRACT_VERSION = "historical_batch_monitoring_v1"
PREDICTION_SCHEMA = "wind_forecast.monitoring_prediction.v1"
INPUT_SCHEMA = "wind_forecast.monitoring_input_snapshot.v1"
ACTUAL_SCHEMA = "wind_forecast.actual_revision.v1"
METRIC_SCHEMA = "wind_forecast.metric_revision.v1"
MODEL_SCHEMA = "wind_forecast.monitoring_model_snapshot.v1"
STATE_SCHEMA = "wind_forecast.monitoring_state.v1"
ACTIVATION_SCHEMA = "wind_forecast.monitoring_activation.v1"
TARGET_CONTRACT = "ren_wind_production_15min_mw_sum_v1"
TARGET_SCALE = "sum_of_15_minute_MW_observations"
LISBON = ZoneInfo("Europe/Lisbon")

_MODEL_FILES = (
    "model.joblib",
    "model_manifest.json",
    "dataset_manifest.json",
    "reference_decision.json",
    "run_summary.json",
    "environment.json",
    "leakage_audit.json",
)
_CALENDAR_FEATURES = {
    "Month",
    "Day_Of_Week",
    "Day_Of_Year",
    "ISO_Week",
    "Quarter",
    "Is_Weekend",
    "Day_Of_Week_Sin",
    "Day_Of_Week_Cos",
    "Month_Sin",
    "Month_Cos",
    "Day_Of_Year_Sin",
    "Day_Of_Year_Cos",
}


class MonitoringError(RuntimeError):
    """Raised when monitoring evidence cannot be produced safely."""


class ConcurrentMonitoringError(MonitoringError):
    """Raised when another monitoring run owns the ledger lock."""


@dataclass(frozen=True)
class MonitoringConfig:
    """Configuration for a local historical-monitoring run."""

    source_store_root: Path
    monitoring_store_root: Path
    model_bundle: Path
    through_date: str | date
    activation_date: str | date | None = None
    backfill_start: str | date | None = None
    backfill_end: str | date | None = None
    dry_run: bool = False
    now_utc: datetime | None = None
    prediction_mode: str = "historical_hindcast"
    forecast_horizon: int | None = None
    target_scale: str = TARGET_SCALE

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_store_root", Path(self.source_store_root))
        object.__setattr__(
            self, "monitoring_store_root", Path(self.monitoring_store_root)
        )
        object.__setattr__(self, "model_bundle", Path(self.model_bundle))
        object.__setattr__(self, "through_date", _parse_date(self.through_date))
        for name in ("activation_date", "backfill_start", "backfill_end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _parse_date(value))
        now = self.now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now_utc must be timezone-aware.")
        object.__setattr__(self, "now_utc", now.astimezone(timezone.utc))
        if self.prediction_mode != "historical_hindcast":
            raise ValueError("Only historical_hindcast is approved for this contract.")
        if self.forecast_horizon is not None:
            raise ValueError("historical_hindcast requires forecast_horizon=null.")
        if self.target_scale != TARGET_SCALE:
            raise ValueError(f"The approved target scale is {TARGET_SCALE!r}.")
        if (self.backfill_start is None) != (self.backfill_end is None):
            raise ValueError("backfill_start and backfill_end must be supplied together.")
        if self.backfill_start and self.backfill_start > self.backfill_end:
            raise ValueError("backfill_start must be on or before backfill_end.")


@dataclass(frozen=True)
class MonitoringPlan:
    """Read-only description of eligible and unresolved target dates."""

    status: str
    activation_date: str
    through_date: str
    eligible_dates: tuple[str, ...]
    pending_dates: tuple[str, ...]
    restatement_dates: tuple[str, ...]
    backfill_dates: tuple[str, ...]
    date_states: Mapping[str, str]

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MonitoringResult:
    """Outcome of one monitoring execution or dry run."""

    status: str
    run_id: str | None
    plan: MonitoringPlan
    prediction_ids: tuple[str, ...] = ()
    actual_revision_ids: tuple[str, ...] = ()
    metric_revision_ids: tuple[str, ...] = ()
    current_state_path: Path | None = None
    blocked_dates: Mapping[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        result = asdict(self)
        result["current_state_path"] = (
            str(self.current_state_path) if self.current_state_path else None
        )
        result["plan"] = self.plan.summary()
        return result


FailureHook = Callable[[str], None]


def plan_historical_monitoring(config: MonitoringConfig) -> MonitoringPlan:
    """Plan monitoring without creating locks, directories, or files."""
    bundle = _validate_model_bundle(config.model_bundle)
    source = load_verified_current_state(config.source_store_root)
    activation = _resolve_activation(config, write=False)
    current = _load_current(config.monitoring_store_root, verify=True)
    expected_model = _model_snapshot_record(bundle)
    if current is not None and current.get("model_snapshot_id") != expected_model.get(
        "model_snapshot_id"
    ):
        raise MonitoringError(
            "The activated model snapshot is immutable for this Stage 1 ledger."
        )
    targets, backfills = _requested_dates(config, activation["activation_date"])
    eligible: list[str] = []
    pending: list[str] = []
    date_states: dict[str, str] = {}
    for day in targets:
        issued = day in ((current or {}).get("as_issued") or {})
        sources_ready = _sources_eligible(day, source, bundle)
        due = config.now_utc >= _scheduled(_parse_date(day)).astimezone(timezone.utc)
        if sources_ready:
            if due:
                eligible.append(day)
            else:
                pending.append(day)
            date_states[day] = "issued" if issued else "eligible"
        else:
            pending.append(day)
            late_at = (_scheduled(_parse_date(day)) + timedelta(days=2)).astimezone(
                timezone.utc
            )
            date_states[day] = (
                "issued"
                if issued
                else "source_late"
                if config.now_utc >= late_at
                else "pending_source"
            )
    restatements = _restatement_dates(current, source, bundle)
    return MonitoringPlan(
        status="planned",
        activation_date=activation["activation_date"],
        through_date=config.through_date.isoformat(),
        eligible_dates=tuple(eligible),
        pending_dates=tuple(pending),
        restatement_dates=tuple(restatements),
        backfill_dates=tuple(backfills),
        date_states=date_states,
    )


def run_historical_monitoring(
    config: MonitoringConfig,
    *,
    failure_hook: FailureHook | None = None,
) -> MonitoringResult:
    """Append predictions first, then actuals/metrics, and publish one pointer."""
    plan = plan_historical_monitoring(config)
    if config.dry_run:
        return MonitoringResult(status="planned", run_id=None, plan=plan)
    run_id = _run_id(config.now_utc)
    lock = _acquire_lock(config.monitoring_store_root, run_id)
    run_root = config.monitoring_store_root / "runs" / run_id
    prediction_ids: list[str] = []
    actual_ids: list[str] = []
    metric_ids: list[str] = []
    blocked: dict[str, str] = {}
    try:
        run_root.mkdir(parents=True, exist_ok=False)
        _immutable_json(
            run_root / "request.json",
            {
                "schema_version": "wind_forecast.monitoring_request.v1",
                "run_id": run_id,
                "requested_at_utc": _utc_text(config.now_utc),
                "config": _config_payload(config),
                "plan": plan.summary(),
            },
        )
        source = load_verified_current_state(config.source_store_root)
        bundle = _validate_model_bundle(config.model_bundle)
        activation = _resolve_activation(config, write=True)
        model = _persist_model_snapshot(config.monitoring_store_root, bundle)
        previous = _load_current(config.monitoring_store_root, verify=True)
        if previous is not None and previous.get("model_snapshot_id") != model.get(
            "model_snapshot_id"
        ):
            raise MonitoringError(
                "The activated model snapshot is immutable for this Stage 1 ledger."
            )
        state = _empty_state(activation, model) if previous is None else _copy(previous)
        state["model_snapshot_id"] = model["model_snapshot_id"]

        target_set = set(plan.eligible_dates)
        target_set.update((state.get("as_issued") or {}).keys())
        for day in sorted(target_set):
            if not _is_eligible(day, config.now_utc, source, bundle):
                continue
            try:
                prediction, created = _ensure_prediction(
                    config=config,
                    run_id=run_id,
                    target_date=day,
                    source=source,
                    bundle=bundle,
                    model_snapshot=model,
                    state=state,
                    backfill=day in plan.backfill_dates,
                )
                if created:
                    prediction_ids.append(prediction["prediction_id"])
                    _call_hook(failure_hook, "after_prediction")
                _update_prediction_view(state, prediction)
                actual, actual_created = _ensure_actual(
                    config.monitoring_store_root,
                    day,
                    source,
                    state,
                    observed_at=config.now_utc,
                )
                if actual_created:
                    actual_ids.append(actual["actual_revision_id"])
                state.setdefault("actuals", {})[day] = actual["actual_revision_id"]
                for prediction_id in _current_predictions_for_day(state, day):
                    stored_prediction = _load_prediction(
                        config.monitoring_store_root, prediction_id
                    )
                    metric, metric_created = _ensure_metric(
                        config.monitoring_store_root,
                        stored_prediction,
                        actual,
                        state,
                        observed_at=config.now_utc,
                    )
                    if metric_created:
                        metric_ids.append(metric["metric_revision_id"])
                    state.setdefault("metrics", {})[prediction_id] = metric[
                        "metric_revision_id"
                    ]
            except MonitoringError as exc:
                blocked[day] = f"blocked_prerequisite: {exc}"

        state.pop("_store_root", None)
        state["schema_version"] = STATE_SCHEMA
        state["generation"] = int((previous or {}).get("generation", 0)) + 1
        state["updated_at_utc"] = _utc_text(config.now_utc)
        state["source_generation"] = source.get("generation")
        state["source_release_id"] = source.get("release_id")
        date_states = dict((previous or {}).get("date_states") or {})
        date_states.update(plan.date_states)
        date_states.update({day: "issued" for day in state.get("as_issued", {})})
        date_states.update({day: "blocked_prerequisite" for day in blocked})
        state["date_states"] = date_states
        _verify_state(config.monitoring_store_root, state)
        _atomic_json(config.monitoring_store_root / "state" / "current.json", state)
        status = "succeeded" if prediction_ids or actual_ids or metric_ids else "no_op"
        result = MonitoringResult(
            status=status,
            run_id=run_id,
            plan=plan,
            prediction_ids=tuple(prediction_ids),
            actual_revision_ids=tuple(actual_ids),
            metric_revision_ids=tuple(metric_ids),
            current_state_path=config.monitoring_store_root / "state" / "current.json",
            blocked_dates=blocked,
        )
        _immutable_json(run_root / "result.json", result.summary())
        return result
    except Exception as exc:
        if run_root.is_dir():
            _immutable_json(
                run_root / "failure.json",
                {
                    "schema_version": "wind_forecast.monitoring_failure.v1",
                    "run_id": run_id,
                    "failed_at_utc": _utc_text(config.now_utc),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                },
            )
        raise
    finally:
        _release_lock(lock, run_id)


def load_prediction_evidence(
    store_root: str | Path, prediction_id: str
) -> dict[str, Any]:
    """Resolve and verify the complete immutable evidence chain for a prediction."""
    root = Path(store_root)
    prediction = _load_prediction(root, prediction_id)
    snapshot = _load_typed_record(
        root / "input_snapshots" / f"{prediction['model_input_snapshot_id']}.json",
        "model_input_snapshot_id",
        "model_input_snapshot",
    )
    model = _load_model_snapshot(root, prediction["model_snapshot_id"])
    if snapshot.get("target_date") != prediction.get("target_date"):
        raise MonitoringError("Prediction and input snapshot target dates differ.")
    if snapshot.get("feature_names") != model.get("feature_names"):
        raise MonitoringError("Input snapshot and model feature orders differ.")
    if prediction.get("feature_schema_sha256") != model.get(
        "feature_schema_sha256"
    ):
        raise MonitoringError("Prediction and model feature schemas differ.")
    if prediction.get("dataset_sha256") != model.get("dataset", {}).get(
        "dataset_sha256"
    ):
        raise MonitoringError("Prediction and model dataset identities differ.")
    metrics: list[dict[str, Any]] = []
    actuals: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "metrics").glob("*.json")):
        item = _load_typed_record(
            path, "metric_revision_id", "metric_revision"
        )
        if item.get("prediction_id") != prediction_id:
            continue
        metrics.append(item)
        actual_id = str(item["actual_revision_id"])
        actuals[actual_id] = _load_typed_record(
            root / "actuals" / f"{actual_id}.json",
            "actual_revision_id",
            "actual_revision",
        )
        actual = actuals[actual_id]
        if actual.get("target_date") != prediction.get("target_date"):
            raise MonitoringError("Prediction and actual target dates differ.")
        expected_error = float(prediction["prediction"]) - float(actual["actual"])
        if not (
            np.isclose(item.get("signed_error"), expected_error)
            and np.isclose(item.get("absolute_error"), abs(expected_error))
            and np.isclose(item.get("squared_error"), expected_error**2)
        ):
            raise MonitoringError("Stored metric values do not match their evidence.")
    return {
        "prediction": prediction,
        "model_input_snapshot": snapshot,
        "model_snapshot": model,
        "actual_revisions": list(actuals.values()),
        "metric_revisions": metrics,
    }


def replay_prediction(store_root: str | Path, prediction_id: str) -> dict[str, Any]:
    """Reload model and target-free features and verify numerical equivalence."""
    evidence = load_prediction_evidence(store_root, prediction_id)
    prediction = evidence["prediction"]
    snapshot = evidence["model_input_snapshot"]
    model = evidence["model_snapshot"]
    model_path = Path(model["files"]["model.joblib"]["path"])
    loaded = joblib.load(model_path)
    frame = pd.DataFrame(
        [snapshot["feature_values"]], columns=snapshot["feature_names"]
    )
    replayed = float(np.asarray(loaded.predict(frame), dtype=float).reshape(-1)[0])
    expected = float(prediction["prediction"])
    equivalent = bool(np.isclose(replayed, expected, rtol=1e-12, atol=1e-9))
    if not np.isfinite(replayed) or not equivalent:
        raise MonitoringError("Replayed prediction differs from immutable evidence.")
    return {
        "prediction_id": prediction_id,
        "prediction": expected,
        "replayed_prediction": replayed,
        "equivalent": True,
        "rtol": 1e-12,
        "atol": 1e-9,
    }


def _validate_model_bundle(root: Path) -> dict[str, Any]:
    missing = [name for name in _MODEL_FILES if not (root / name).is_file()]
    if missing:
        raise MonitoringError(f"Model bundle is missing required files: {missing}.")
    payloads = {
        name: _read_json(root / name)
        for name in _MODEL_FILES
        if name.endswith(".json")
    }
    model_manifest = payloads["model_manifest.json"]
    dataset = payloads["dataset_manifest.json"]
    decision = payloads["reference_decision.json"]
    summary = payloads["run_summary.json"]
    if model_manifest.get("schema_version") != "wind_forecast.v2_model_manifest.v1":
        raise MonitoringError("Unsupported v2 model manifest schema.")
    if dataset.get("schema_version") != "wind_forecast.v2_training_dataset.v1":
        raise MonitoringError("Unsupported v2 dataset manifest schema.")
    if decision.get("schema_version") != "wind_forecast.v2_reference_decision.v1":
        raise MonitoringError("Unsupported v2 reference-decision schema.")
    if summary.get("schema_version") != "wind_forecast.v2_training_run.v1":
        raise MonitoringError("Unsupported v2 training-summary schema.")
    if payloads["environment.json"].get("schema_version") != (
        "wind_forecast.v2_environment.v1"
    ):
        raise MonitoringError("Unsupported v2 environment manifest schema.")
    leakage = payloads["leakage_audit.json"]
    if (
        leakage.get("schema_version") != "wind_forecast.v2_leakage_audit.v1"
        or leakage.get("forecast_contract") != "historical_daily_hindcast"
        or leakage.get("passed") is not True
    ):
        raise MonitoringError("The v2 temporal leakage audit is not accepted.")
    if model_manifest.get("task") != "daily_wind_production_historical_hindcast":
        raise MonitoringError("The model is not approved for historical hindcast.")
    if decision.get("accepted_as_reference") is not True:
        raise MonitoringError("The v2 reference decision was not accepted.")
    if decision.get("status") != "selected_not_promoted":
        raise MonitoringError("The model must remain selected_not_promoted.")
    if decision.get("automatic_promotion") is not False:
        raise MonitoringError("Automatic promotion must remain disabled.")
    if summary.get("accepted_as_reference") is not True:
        raise MonitoringError("The training summary did not accept this reference.")
    if model_manifest.get("reference_status") != "selected_not_promoted":
        raise MonitoringError("The model manifest reference status is invalid.")
    if model_manifest.get("scaler_required") is not False:
        raise MonitoringError("The accepted v2 model must not require a scaler.")
    if model_manifest.get("scaler") is not None:
        raise MonitoringError("The accepted v2 model manifest must not name a scaler.")
    if any(root.rglob("*scaler*")):
        raise MonitoringError("Unexpected scaler artifact found in the v2 bundle.")
    features = list(model_manifest.get("feature_names") or [])
    if not features or features != list(dataset.get("feature_names") or []):
        raise MonitoringError("Model and dataset feature order differs.")
    if DATE_COLUMN in features or TARGET_COLUMN in features:
        raise MonitoringError("Model features contain Date or the actual target.")
    schema_hash = _hash_json(features)
    if model_manifest.get("feature_schema_sha256") != schema_hash:
        raise MonitoringError("Model feature-schema checksum is invalid.")
    if dataset.get("feature_schema_sha256") != schema_hash:
        raise MonitoringError("Dataset feature-schema checksum is invalid.")
    if model_manifest.get("dataset_sha256") != dataset.get("sha256"):
        raise MonitoringError("Model and dataset identities differ.")
    if model_manifest.get("dataset_version") != dataset.get("dataset_version"):
        raise MonitoringError("Model and dataset versions differ.")
    if dataset.get("target") != TARGET_COLUMN:
        raise MonitoringError("The v2 dataset target contract differs.")
    if summary.get("dataset_version") != dataset.get("dataset_version"):
        raise MonitoringError("Training summary and dataset versions differ.")
    if summary.get("dataset_sha256") != dataset.get("sha256"):
        raise MonitoringError("Training summary and dataset identities differ.")
    if summary.get("scaler_required") is not False:
        raise MonitoringError("Training summary unexpectedly requires a scaler.")
    model_hash = sha256_file(root / "model.joblib")
    if model_manifest.get("model_sha256") != model_hash:
        raise MonitoringError("Model checksum is invalid.")
    artifact_hashes = dict(summary.get("artifact_sha256") or {})
    for name in _MODEL_FILES:
        if name == "run_summary.json":
            continue
        expected = artifact_hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise MonitoringError(f"Training summary checksum is missing for {name}.")
        if expected != sha256_file(root / name):
            raise MonitoringError(f"Training summary checksum is invalid for {name}.")
    if dataset.get("transformation_version") != TRANSFORMATION_VERSION:
        raise MonitoringError("Unsupported feature transformation version.")
    return {
        "root": root,
        "files": {
            name: {"sha256": sha256_file(root / name)} for name in _MODEL_FILES
        },
        "model_manifest": model_manifest,
        "dataset_manifest": dataset,
        "decision": decision,
        "summary": summary,
        "environment": payloads["environment.json"],
        "feature_names": features,
    }


def _resolve_activation(config: MonitoringConfig, *, write: bool) -> dict[str, Any]:
    root = config.monitoring_store_root / "activations"
    existing = []
    if root.is_dir():
        existing = [
            _load_typed_record(path, "activation_id", "activation")
            for path in sorted(root.glob("*.json"))
        ]
    dates = {str(item.get("activation_date")) for item in existing}
    if len(dates) > 1:
        raise MonitoringError("Multiple incompatible activation dates exist.")
    requested = config.activation_date.isoformat() if config.activation_date else None
    if dates:
        active = existing[0]
        if requested is not None and requested != active["activation_date"]:
            raise MonitoringError("activation_date is immutable after the first run.")
        activation = active
    else:
        if requested is None:
            raise MonitoringError("activation_date is required for the first run.")
        body = {
            "schema_version": ACTIVATION_SCHEMA,
            "activation_date": requested,
            "contract_version": CONTRACT_VERSION,
            "prediction_mode": "historical_hindcast",
        }
        activation = _with_id("activation", "activation_id", body)
        if write:
            _immutable_json(root / f"{activation['activation_id']}.json", activation)
    if config.backfill_start and config.backfill_end:
        active_date = _parse_date(activation["activation_date"])
        if config.backfill_end >= active_date:
            raise MonitoringError("Explicit backfill dates must precede activation_date.")
    return activation


def _requested_dates(
    config: MonitoringConfig, activation_text: str
) -> tuple[list[str], list[str]]:
    activation = _parse_date(activation_text)
    normal = _date_range(activation, config.through_date)
    backfills = (
        _date_range(config.backfill_start, config.backfill_end)
        if config.backfill_start and config.backfill_end
        else []
    )
    return sorted(set(normal) | set(backfills)), backfills


def _is_eligible(
    day: str,
    now_utc: datetime,
    source: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> bool:
    if now_utc < _scheduled(_parse_date(day)).astimezone(timezone.utc):
        return False
    return _sources_eligible(day, source, bundle)


def _sources_eligible(
    day: str,
    source: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> bool:
    features = dict((source.get("partitions") or {}).get("features") or {})
    ref = dict(features.get(day) or {})
    if ref.get("feature_ready") is not True:
        return False
    ren = dict((source.get("sources") or {}).get("ren") or {}).get(day) or {}
    if ren.get("status") != "complete":
        return False
    era = dict((source.get("sources") or {}).get("era5_land") or {})
    if not any(
        item.get("status") == "complete" and day in (item.get("local_dates") or [])
        for item in era.values()
    ):
        return False
    try:
        row = _read_feature_row(ref, day)
    except MonitoringError:
        return False
    return all(name in row for name in bundle["feature_names"])


def _persist_model_snapshot(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    record = _model_snapshot_record(bundle)
    snapshot_root = root / "model_snapshots" / record["model_snapshot_id"]
    for name in _MODEL_FILES:
        destination = snapshot_root / name
        _immutable_copy(bundle["root"] / name, destination)
    _immutable_json(snapshot_root / "snapshot.json", record)
    return _load_model_snapshot(root, record["model_snapshot_id"])


def _model_snapshot_record(bundle: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": MODEL_SCHEMA,
        "model": {
            "task": bundle["model_manifest"]["task"],
            "model_type": bundle["model_manifest"].get("model_type"),
            "model_sha256": bundle["model_manifest"]["model_sha256"],
            "reference_status": "selected_not_promoted",
            "scaler": None,
        },
        "dataset": {
            "dataset_version": bundle["dataset_manifest"]["dataset_version"],
            "dataset_sha256": bundle["dataset_manifest"]["sha256"],
            "dataset_manifest": bundle["dataset_manifest"],
        },
        "feature_names": bundle["feature_names"],
        "feature_schema_sha256": bundle["model_manifest"][
            "feature_schema_sha256"
        ],
        "transformation": {
            "version": bundle["dataset_manifest"]["transformation_version"],
            "training_git_commit": bundle["environment"].get("git_sha"),
            "training_git_dirty": bundle["environment"].get("git_dirty"),
        },
        "files": bundle["files"],
    }
    return _with_id("model_snapshot", "model_snapshot_id", body)


def _ensure_prediction(
    *,
    config: MonitoringConfig,
    run_id: str,
    target_date: str,
    source: Mapping[str, Any],
    bundle: Mapping[str, Any],
    model_snapshot: Mapping[str, Any],
    state: Mapping[str, Any],
    backfill: bool,
) -> tuple[dict[str, Any], bool]:
    current_id = (state.get("as_issued") or {}).get(target_date)
    view = "as_issued"
    restates: dict[str, Any] = {}
    if current_id:
        current = _load_prediction(config.monitoring_store_root, str(current_id))
        as_issued_snapshot = _load_typed_record(
            config.monitoring_store_root
            / "input_snapshots"
            / f"{current['model_input_snapshot_id']}.json",
            "model_input_snapshot_id",
            "model_input_snapshot",
        )
        current_dependencies = _dependency_map(
            bundle["feature_names"], target_date, source
        )
        latest_id = (state.get("restated") or {}).get(target_date)
        latest = None
        comparison = current
        comparison_snapshot = as_issued_snapshot
        if latest_id:
            latest = _load_prediction(config.monitoring_store_root, str(latest_id))
            comparison = latest
            comparison_snapshot = _load_typed_record(
                config.monitoring_store_root
                / "input_snapshots"
                / f"{latest['model_input_snapshot_id']}.json",
                "model_input_snapshot_id",
                "model_input_snapshot",
            )
        if _dependency_identity(
            comparison_snapshot["dependencies"]
        ) == _dependency_identity(current_dependencies):
            return comparison, False
        if as_issued_snapshot["transformation"] != _transformation_evidence(source):
            raise MonitoringError(
                "restatement transformation version or code hash differs"
            )
        view = "restated"
        restates = {
            "restates_prediction_id": current["prediction_id"],
            "restates_run_id": current["run_id"],
            "supersedes_id": latest_id,
        }
    ref = dict(
        ((source.get("partitions") or {}).get("features") or {}).get(target_date)
        or {}
    )
    row = _read_feature_row(ref, target_date)
    names = list(bundle["feature_names"])
    values = [float(row[name]) for name in names]
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise MonitoringError("Model input contains non-finite values.")
    dependencies = _dependency_map(names, target_date, source)
    transformation = _transformation_evidence(source)
    if transformation["version"] != bundle["dataset_manifest"][
        "transformation_version"
    ]:
        raise MonitoringError("Source and model feature transformations differ.")
    input_body = {
        "schema_version": INPUT_SCHEMA,
        "target_date": target_date,
        "feature_names": names,
        "feature_values": values,
        "feature_schema_sha256": _hash_json(names),
        "feature_partition": {
            "partition_key": ref.get("partition_key"),
            "sha256": ((ref.get("files") or {}).get("feature_ready") or {}).get(
                "sha256"
            ),
        },
        "dependencies": dependencies,
        "transformation": transformation,
        "target_excluded": TARGET_COLUMN not in names and DATE_COLUMN not in names,
    }
    input_snapshot = _with_id(
        "model_input_snapshot", "model_input_snapshot_id", input_body
    )
    _immutable_json(
        config.monitoring_store_root
        / "input_snapshots"
        / f"{input_snapshot['model_input_snapshot_id']}.json",
        input_snapshot,
    )
    model_path = Path(model_snapshot["files"]["model.joblib"]["path"])
    model = joblib.load(model_path)
    prediction_value = float(
        np.asarray(
            model.predict(pd.DataFrame([values], columns=names)), dtype=float
        ).reshape(-1)[0]
    )
    if not np.isfinite(prediction_value):
        raise MonitoringError("Model produced a non-finite prediction.")
    scheduled = _scheduled(_parse_date(target_date))
    issuance = "explicit_backfill" if backfill else (
        "scheduled"
        if config.now_utc.astimezone(LISBON).date() == scheduled.date()
        else "catch_up"
    )
    if view == "restated":
        issuance = "restatement"
    body = {
        "schema_version": PREDICTION_SCHEMA,
        "run_id": run_id,
        "target_date": target_date,
        "scheduled_at_local": scheduled.isoformat(),
        "scheduled_at_utc": _utc_text(scheduled.astimezone(timezone.utc)),
        "issued_at_utc": _utc_text(config.now_utc),
        "prediction_mode": "historical_hindcast",
        "issuance_kind": issuance,
        "target_day_offset": 0,
        "forecast_horizon": None,
        "actual_available_to_system_at_issue": True,
        "contract_version": CONTRACT_VERSION,
        "target_contract_id": TARGET_CONTRACT,
        "target_scale": TARGET_SCALE,
        "physical_unit": None,
        "model_snapshot_id": model_snapshot["model_snapshot_id"],
        "dataset_version": bundle["dataset_manifest"]["dataset_version"],
        "dataset_sha256": bundle["dataset_manifest"]["sha256"],
        "feature_schema_sha256": _hash_json(names),
        "model_input_snapshot_id": input_snapshot["model_input_snapshot_id"],
        "prediction": prediction_value,
        "view": view,
        **restates,
    }
    prediction = _with_id("prediction", "prediction_id", body)
    orphan = _find_prediction(
        config.monitoring_store_root,
        target_date=target_date,
        view=view,
        input_id=input_snapshot["model_input_snapshot_id"],
        model_id=model_snapshot["model_snapshot_id"],
        supersedes_id=restates.get("supersedes_id"),
    )
    if orphan is not None:
        return orphan, False
    _immutable_json(
        config.monitoring_store_root
        / "predictions"
        / f"{prediction['prediction_id']}.json",
        prediction,
    )
    return prediction, True


def _ensure_actual(
    root: Path,
    day: str,
    source: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    observed_at: datetime,
) -> tuple[dict[str, Any], bool]:
    ren = dict((source.get("sources") or {}).get("ren") or {}).get(day) or {}
    if ren.get("status") != "complete":
        raise MonitoringError("REN actual is not complete.")
    current_id = (state.get("actuals") or {}).get(day)
    if current_id:
        current = _load_typed_record(
            root / "actuals" / f"{current_id}.json",
            "actual_revision_id",
            "actual_revision",
        )
        if (
            current.get("source_revision_id") == ren.get("revision_id")
            and current.get("source_revision") == ren.get("revision")
        ):
            return current, False
    path = Path(str(ren.get("primary_path") or ""))
    if not path.is_file() or sha256_file(path) != ren.get("physical_sha256"):
        raise MonitoringError("REN actual source is missing or corrupt.")
    frame = pd.read_csv(path)
    if "wind_production_mw" not in frame:
        raise MonitoringError("REN actual source lacks wind_production_mw.")
    values = pd.to_numeric(frame["wind_production_mw"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise MonitoringError("REN actual source contains invalid values.")
    actual = float(values.sum())
    retrieval = None
    if "retrieval_timestamp_utc" in frame:
        timestamps = frame["retrieval_timestamp_utc"].dropna().astype(str)
        retrieval = max(timestamps) if not timestamps.empty else None
    if retrieval is None:
        retrieval = _supporting_retrieval_timestamp(ren)
    body = {
        "schema_version": ACTUAL_SCHEMA,
        "target_date": day,
        "target_contract_id": TARGET_CONTRACT,
        "target_scale": TARGET_SCALE,
        "physical_unit": None,
        "actual": actual,
        "source": "REN",
        "source_revision_id": ren.get("revision_id"),
        "source_revision": ren.get("revision"),
        "source_supersedes_id": ren.get("supersedes_id"),
        "source_history": ren.get("history", []),
        "source_retrieved_at_utc": retrieval,
        "observed_by_monitoring_at_utc": _utc_text(observed_at),
        "normalized_path": str(path.resolve()),
        "normalized_sha256": ren.get("physical_sha256"),
        "semantic_sha256": ren.get("semantic_sha256"),
        "raw_sha256": _supporting_raw_sha256(ren),
        "supporting_observations": ren.get("supporting_observations", []),
        "validation_status": "complete",
        "provider_finality": "unknown",
        "supersedes_id": current_id,
    }
    record = _with_id("actual_revision", "actual_revision_id", body)
    orphan = _find_record(
        root / "actuals",
        "actual_revision_id",
        "actual_revision",
        lambda item: item.get("target_date") == day
        and item.get("source_revision_id") == ren.get("revision_id")
        and item.get("source_revision") == ren.get("revision"),
    )
    if orphan is not None:
        return orphan, False
    _immutable_json(root / "actuals" / f"{record['actual_revision_id']}.json", record)
    return record, True


def _ensure_metric(
    root: Path,
    prediction: Mapping[str, Any],
    actual: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    observed_at: datetime,
) -> tuple[dict[str, Any], bool]:
    prediction_id = str(prediction["prediction_id"])
    current_id = (state.get("metrics") or {}).get(prediction_id)
    if current_id:
        current = _load_typed_record(
            root / "metrics" / f"{current_id}.json",
            "metric_revision_id",
            "metric_revision",
        )
        if current.get("actual_revision_id") == actual.get("actual_revision_id"):
            return current, False
    error = float(prediction["prediction"]) - float(actual["actual"])
    body = {
        "schema_version": METRIC_SCHEMA,
        "target_date": prediction["target_date"],
        "prediction_id": prediction_id,
        "actual_revision_id": actual["actual_revision_id"],
        "calculated_at_utc": _utc_text(observed_at),
        "signed_error": error,
        "absolute_error": abs(error),
        "squared_error": error**2,
        "target_scale": TARGET_SCALE,
        "supersedes_id": current_id,
    }
    record = _with_id("metric_revision", "metric_revision_id", body)
    orphan = _find_record(
        root / "metrics",
        "metric_revision_id",
        "metric_revision",
        lambda item: item.get("prediction_id") == prediction_id
        and item.get("actual_revision_id") == actual.get("actual_revision_id"),
    )
    if orphan is not None:
        return orphan, False
    _immutable_json(root / "metrics" / f"{record['metric_revision_id']}.json", record)
    return record, True


def _dependency_map(
    feature_names: Sequence[str], day: str, source: Mapping[str, Any]
) -> dict[str, Any]:
    target = _parse_date(day)
    result: dict[str, Any] = {}
    for name in feature_names:
        source_name, dates = _feature_dependency_dates(name, target)
        refs: list[dict[str, Any]] = []
        for dependency_date in dates:
            text = dependency_date.isoformat()
            if source_name == "ren":
                source_ref = dict(
                    ((source.get("sources") or {}).get("ren") or {}).get(text)
                    or {}
                )
                if source_ref.get("status") != "complete":
                    raise MonitoringError(f"REN dependency {text} is not complete.")
                refs.append(_source_dependency("ren", text, source_ref))
            elif source_name == "era5_land":
                matches = [
                    _source_dependency("era5_land", text, ref)
                    for ref in (
                        ((source.get("sources") or {}).get("era5_land") or {})
                    ).values()
                    if ref.get("status") == "complete"
                    and text in (ref.get("local_dates") or [])
                ]
                if not matches:
                    raise MonitoringError(f"ERA5-Land dependency {text} is not complete.")
                refs.extend(matches)
        result[name] = {
            "derivation": "calendar_only" if source_name is None else source_name,
            "source_revisions": sorted(
                refs, key=lambda item: (item["source"], item["logical_key"])
            ),
        }
    return result


def _feature_dependency_dates(
    feature: str, target: date
) -> tuple[str | None, list[date]]:
    if feature in _CALENDAR_FEATURES:
        return None, []
    lag = re.search(r"_Lag(\d+)$", feature)
    rolling = re.search(r"_Rolling_(?:Mean|Std)_(\d+)$", feature)
    if feature.startswith("Wind_Production_"):
        if lag:
            return "ren", [target - timedelta(days=int(lag.group(1)))]
        if rolling:
            return "ren", [
                target - timedelta(days=value)
                for value in range(1, int(rolling.group(1)) + 1)
            ]
    weather_prefixes = (
        "Average_Wind_Speed",
        "Average_Temperature",
        "Average_Wind_Direction",
        "Wind_Direction_Sin",
        "Wind_Direction_Cos",
    )
    if feature.startswith(weather_prefixes):
        if lag:
            return "era5_land", [target - timedelta(days=int(lag.group(1)))]
        if rolling:
            return "era5_land", [
                target - timedelta(days=value)
                for value in range(1, int(rolling.group(1)) + 1)
            ]
        return "era5_land", [target]
    raise MonitoringError(f"No approved dependency rule exists for feature {feature!r}.")


def _source_dependency(source: str, day: str, ref: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "source_date": day,
        "logical_key": ref.get("logical_key", day),
        "revision_id": ref.get("revision_id"),
        "revision": ref.get("revision"),
        "semantic_sha256": ref.get("semantic_sha256"),
        "physical_sha256": ref.get("physical_sha256"),
        "path": ref.get("primary_path"),
        "provider_finality": ref.get("provider_finality"),
        "supporting_observations": ref.get("supporting_observations", []),
    }


def _supporting_raw_sha256(ref: Mapping[str, Any]) -> str | None:
    for item in ref.get("supporting_observations", []):
        if str(item.get("filename") or "").casefold() == "response.json":
            return str(item.get("sha256") or "") or None
    return None


def _supporting_retrieval_timestamp(ref: Mapping[str, Any]) -> str | None:
    candidates: list[str] = []
    for item in ref.get("supporting_observations", []):
        path = Path(str(item.get("path") or ""))
        if path.suffix.casefold() != ".json" or not path.is_file():
            continue
        payload = _read_json(path)
        _collect_timestamp_values(payload, candidates)
    return max(candidates) if candidates else None


def _collect_timestamp_values(value: Any, candidates: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {
                "retrieval_timestamp_utc",
                "retrieval_finished_at_utc",
            } and isinstance(item, str):
                candidates.append(item)
            else:
                _collect_timestamp_values(item, candidates)
    elif isinstance(value, list):
        for item in value:
            _collect_timestamp_values(item, candidates)


def _dependency_identity(dependencies: Mapping[str, Any]) -> str:
    logical = {
        name: [
            {
                "source": ref.get("source"),
                "source_date": ref.get("source_date"),
                "logical_key": ref.get("logical_key"),
                "revision_id": ref.get("revision_id"),
                "semantic_sha256": ref.get("semantic_sha256"),
            }
            for ref in item.get("source_revisions", [])
        ]
        for name, item in dependencies.items()
    }
    return _hash_json(logical)


def _transformation_evidence(source: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(source.get("manifest_path") or ""))
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    module_path = Path(__file__).with_name("v2_features.py")
    return {
        "version": (manifest.get("versions") or {}).get(
            "features", TRANSFORMATION_VERSION
        ),
        "source_code_commit_sha": manifest.get("git_commit"),
        "source_code_sha256": sha256_file(module_path),
    }


def _read_feature_row(ref: Mapping[str, Any], day: str) -> dict[str, Any]:
    file_ref = dict((ref.get("files") or {}).get("feature_ready") or {})
    path = Path(str(file_ref.get("path") or ""))
    checksum = str(file_ref.get("sha256") or "")
    if not path.is_file() or not checksum or sha256_file(path) != checksum:
        raise MonitoringError("Feature partition is missing or corrupt.")
    frame = pd.read_csv(path)
    if DATE_COLUMN not in frame:
        raise MonitoringError("Feature partition lacks Date.")
    rows = frame.loc[frame[DATE_COLUMN].astype(str).eq(day)]
    if len(rows) != 1:
        raise MonitoringError("Feature partition must contain exactly one target row.")
    return rows.iloc[0].to_dict()


def _restatement_dates(
    current: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> list[str]:
    if not current:
        return []
    result = []
    root_text = str(current.get("_store_root") or "")
    if not root_text:
        return result
    root = Path(root_text)
    for day, prediction_id in (current.get("as_issued") or {}).items():
        comparison_id = (current.get("restated") or {}).get(day) or prediction_id
        prediction = _load_prediction(root, str(comparison_id))
        snapshot = _load_typed_record(
            root / "input_snapshots" / f"{prediction['model_input_snapshot_id']}.json",
            "model_input_snapshot_id",
            "model_input_snapshot",
        )
        dependencies = _dependency_map(bundle["feature_names"], day, source)
        if _dependency_identity(snapshot["dependencies"]) != _dependency_identity(
            dependencies
        ):
            result.append(day)
    return sorted(result)


def _empty_state(
    activation: Mapping[str, Any], model: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "generation": 0,
        "activation_id": activation["activation_id"],
        "activation_date": activation["activation_date"],
        "model_snapshot_id": model["model_snapshot_id"],
        "as_issued": {},
        "restated": {},
        "actuals": {},
        "metrics": {},
    }


def _load_current(root: Path, *, verify: bool) -> dict[str, Any] | None:
    path = root / "state" / "current.json"
    if not path.is_file():
        return None
    state = _read_json(path)
    if state.get("schema_version") != STATE_SCHEMA:
        raise MonitoringError("Unsupported monitoring current-state schema.")
    if verify:
        _verify_state(root, state)
    state["_store_root"] = str(root)
    return state


def _verify_state(root: Path, state: Mapping[str, Any]) -> None:
    activation_id = str(state.get("activation_id") or "")
    _load_typed_record(
        root / "activations" / f"{activation_id}.json",
        "activation_id",
        "activation",
    )
    _load_model_snapshot(root, str(state.get("model_snapshot_id") or ""))
    for prediction_id in {
        *(state.get("as_issued") or {}).values(),
        *(state.get("restated") or {}).values(),
    }:
        prediction = _load_prediction(root, str(prediction_id))
        _load_typed_record(
            root / "input_snapshots" / f"{prediction['model_input_snapshot_id']}.json",
            "model_input_snapshot_id",
            "model_input_snapshot",
        )
    for actual_id in (state.get("actuals") or {}).values():
        _load_typed_record(
            root / "actuals" / f"{actual_id}.json",
            "actual_revision_id",
            "actual_revision",
        )
    for prediction_id, metric_id in (state.get("metrics") or {}).items():
        metric = _load_typed_record(
            root / "metrics" / f"{metric_id}.json",
            "metric_revision_id",
            "metric_revision",
        )
        if metric.get("prediction_id") != prediction_id:
            raise MonitoringError("Current metric key does not match its prediction.")
        _load_prediction(root, str(prediction_id))
        actual_id = str(metric.get("actual_revision_id") or "")
        _load_typed_record(
            root / "actuals" / f"{actual_id}.json",
            "actual_revision_id",
            "actual_revision",
        )


def _load_model_snapshot(root: Path, snapshot_id: str) -> dict[str, Any]:
    record = _load_typed_record(
        root / "model_snapshots" / snapshot_id / "snapshot.json",
        "model_snapshot_id",
        "model_snapshot",
    )
    enriched = _copy(record)
    for name, ref in (record.get("files") or {}).items():
        path = root / "model_snapshots" / snapshot_id / name
        if not path.is_file() or sha256_file(path) != ref.get("sha256"):
            raise MonitoringError(f"Model snapshot file is corrupt: {name}.")
        enriched["files"][name]["path"] = str(path.resolve())
    return enriched


def _load_prediction(root: Path, prediction_id: str) -> dict[str, Any]:
    return _load_typed_record(
        root / "predictions" / f"{prediction_id}.json",
        "prediction_id",
        "prediction",
    )


def _load_typed_record(
    path: Path,
    id_field: str,
    kind: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    identifier = str(payload.get(id_field) or "")
    body = {key: value for key, value in payload.items() if key != id_field}
    expected = _record_id(kind, body)
    path_identifier = path.parent.name if path.name == "snapshot.json" else path.stem
    if identifier != expected or identifier != path_identifier:
        raise MonitoringError(f"Content-addressed record is corrupt: {path}.")
    return payload


def _find_prediction(
    root: Path,
    *,
    target_date: str,
    view: str,
    input_id: str,
    model_id: str,
    supersedes_id: str | None,
) -> dict[str, Any] | None:
    return _find_record(
        root / "predictions",
        "prediction_id",
        "prediction",
        lambda item: item.get("target_date") == target_date
        and item.get("view") == view
        and item.get("model_input_snapshot_id") == input_id
        and item.get("model_snapshot_id") == model_id
        and item.get("supersedes_id") == supersedes_id,
    )


def _find_record(
    directory: Path,
    id_field: str,
    kind: str,
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any] | None:
    if not directory.is_dir():
        return None
    matches = []
    for path in sorted(directory.glob("*.json")):
        item = _load_typed_record(path, id_field, kind)
        if predicate(item):
            matches.append(item)
    if len(matches) > 1:
        raise MonitoringError(f"Multiple immutable {kind} records match one identity.")
    return matches[0] if matches else None


def _update_prediction_view(state: dict[str, Any], prediction: Mapping[str, Any]) -> None:
    view = str(prediction["view"])
    state.setdefault(view, {})[prediction["target_date"]] = prediction["prediction_id"]


def _current_predictions_for_day(state: Mapping[str, Any], day: str) -> list[str]:
    return [
        str(value)
        for value in (
            (state.get("as_issued") or {}).get(day),
            (state.get("restated") or {}).get(day),
        )
        if value
    ]


def _with_id(kind: str, id_field: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return {id_field: _record_id(kind, body), **_copy(body)}


def _record_id(kind: str, body: Mapping[str, Any]) -> str:
    return _hash_json({"record_type": kind, "payload": body})


def _hash_json(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise MonitoringError(f"Immutable JSON collision at {path}.")
        return
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise MonitoringError(f"Immutable JSON collision at {path}.")


def _immutable_copy(source: Path, target: Path) -> None:
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise MonitoringError(f"Immutable file collision at {target}.")
        return
    try:
        with target.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if target.read_bytes() != data:
            raise MonitoringError(f"Immutable file collision at {target}.")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    _immutable_json(temporary, payload)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitoringError(f"Invalid JSON file: {path}.") from exc
    if not isinstance(value, dict):
        raise MonitoringError(f"JSON file must contain an object: {path}.")
    return value


def _acquire_lock(root: Path, run_id: str) -> Path:
    path = root / "state" / "monitoring.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
    except FileExistsError as exc:
        raise ConcurrentMonitoringError("Another monitoring run owns the lock.") from exc
    return path


def _release_lock(path: Path, run_id: str) -> None:
    if not path.exists():
        return
    try:
        payload = _read_json(path)
    except MonitoringError:
        return
    if payload.get("run_id") == run_id:
        path.unlink()


def _scheduled(target: date) -> datetime:
    return datetime.combine(target + timedelta(days=5), time(12), tzinfo=LISBON)


def _parse_date(value: str | date) -> date:
    if isinstance(value, datetime):
        raise ValueError("A civil date, not a datetime, is required.")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("Dates must be formatted as YYYY-MM-DD.") from exc


def _date_range(start: date, end: date) -> list[str]:
    if start > end:
        return []
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    ]


def _run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:12]}"


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _config_payload(config: MonitoringConfig) -> dict[str, Any]:
    return {
        "source_store_root": str(config.source_store_root),
        "monitoring_store_root": str(config.monitoring_store_root),
        "model_bundle": str(config.model_bundle),
        "through_date": config.through_date.isoformat(),
        "activation_date": (
            config.activation_date.isoformat() if config.activation_date else None
        ),
        "backfill_start": (
            config.backfill_start.isoformat() if config.backfill_start else None
        ),
        "backfill_end": config.backfill_end.isoformat() if config.backfill_end else None,
        "prediction_mode": config.prediction_mode,
        "forecast_horizon": config.forecast_horizon,
        "target_scale": config.target_scale,
    }


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _call_hook(hook: FailureHook | None, stage: str) -> None:
    if hook is not None:
        hook(stage)


__all__ = [
    "ConcurrentMonitoringError",
    "MonitoringConfig",
    "MonitoringError",
    "MonitoringPlan",
    "MonitoringResult",
    "load_prediction_evidence",
    "plan_historical_monitoring",
    "replay_prediction",
    "run_historical_monitoring",
]
