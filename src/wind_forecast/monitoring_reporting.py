"""Immutable Phase 9 monitoring references, calibration, reports, and alerts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from wind_forecast.manifests import sha256_file
from wind_forecast.deployment_runtime import (
    DeploymentRuntimeError,
    same_model_era,
    verify_active_model_era,
)
from wind_forecast.monitoring import (
    LEGACY_STATE_SCHEMA,
    MonitoringError,
    list_model_eras,
    load_model_era,
    load_prediction_evidence,
    load_verified_monitoring_state,
    validate_monitoring_model_bundle,
)
from wind_forecast.monitoring_statistics import (
    DIRECTION_COMPONENTS,
    DIRECTION_GROUPS,
    MonitoringPolicy,
    calendar_window,
    calibrated_limits,
    circular_drift_statistics,
    drift_statistics,
    regression_metrics,
    seasonal_reference,
    threshold_severity,
)
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN


REFERENCE_SCHEMA = "wind_forecast.monitoring_reference.v1"
CALIBRATION_SCHEMA = "wind_forecast.monitoring_calibration.v1"
REPORT_SCHEMA = "wind_forecast.monitoring_report.v2"
LEGACY_REPORT_SCHEMA = "wind_forecast.monitoring_report.v1"
REPORT_STATE_SCHEMA = "wind_forecast.monitoring_report_state.v2"
LEGACY_REPORT_STATE_SCHEMA = "wind_forecast.monitoring_report_state.v1"
ALERT_SCHEMA = "wind_forecast.monitoring_alert_event.v2"
SOURCE_RUN_SCHEMAS = {
    "wind_forecast.v2_incremental_run.v1",
    "wind_forecast.v2_incremental_run.v2",
}
RAW_DIRECTION_COLUMN = "Average_Wind_Direction"
PREDICTION_COLUMN = "Reference_Prediction"
ACTUAL_COLUMN = "Actual"
LEDGER_PREDICTION_COLUMN = "Prediction"
SEVERITY_ORDER = {"not_available": -1, "ok": 0, "warning": 1, "critical": 2}
_PARALLEL_LOADER_MIN_ITEMS = 32
_PARALLEL_LOADER_MAX_WORKERS = 8
_LoadedRecord = TypeVar("_LoadedRecord")


class MonitoringReportingError(RuntimeError):
    """Raised when reporting evidence is absent, incompatible, or corrupt."""


class MonitoringReportingUnavailableError(MonitoringReportingError):
    """Raised when required reporting evidence cannot be read."""


class MonitoringReportingConflictError(MonitoringReportingError):
    """Raised when individually readable reporting records disagree."""


@dataclass(frozen=True)
class CalibrationConfig:
    """Inputs for one explicit immutable calibration."""

    dataset_path: Path
    model_bundle: Path
    policy_path: Path
    output_root: Path
    backtest_stride_days: int = 7
    retraining_candidate: Path | None = None

    def __post_init__(self) -> None:
        for name in ("dataset_path", "model_bundle", "policy_path", "output_root"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.retraining_candidate is not None:
            object.__setattr__(
                self, "retraining_candidate", Path(self.retraining_candidate)
            )
        if self.backtest_stride_days < 1:
            raise ValueError("backtest_stride_days must be at least one.")


@dataclass(frozen=True)
class CalibrationResult:
    reference_id: str
    calibration_id: str
    reference_dir: Path
    calibration_dir: Path

    def summary(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


@dataclass(frozen=True)
class MonitoringReportConfig:
    """Explicit inputs for one read-only plan or append-only report execution."""

    source_run_manifest: Path
    monitoring_store_root: Path
    model_bundle: Path
    calibration_dir: Path
    deployment_root: Path
    through_date: str | date
    dry_run: bool = False
    now_utc: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_run_manifest", Path(self.source_run_manifest))
        object.__setattr__(self, "monitoring_store_root", Path(self.monitoring_store_root))
        object.__setattr__(self, "calibration_dir", Path(self.calibration_dir))
        object.__setattr__(self, "model_bundle", Path(self.model_bundle))
        object.__setattr__(self, "deployment_root", Path(self.deployment_root))
        value = self.through_date
        object.__setattr__(self, "through_date", value if isinstance(value, date) else date.fromisoformat(value))
        now = self.now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now_utc must be timezone-aware.")
        object.__setattr__(self, "now_utc", now.astimezone(timezone.utc))


@dataclass(frozen=True)
class MonitoringReportPlan:
    status: str
    source_run_id: str
    source_status: str
    through_date: str
    calibration_id: str
    ledger_available: bool
    quality_available: bool
    model_era_id: str
    deployment_id: str
    model_version: str

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MonitoringReportResult:
    status: str
    run_id: str | None
    report_id: str | None
    report_path: Path | None
    markdown_path: Path | None
    active_alert_count: int
    plan: MonitoringReportPlan

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["report_path"] = str(self.report_path) if self.report_path else None
        payload["markdown_path"] = str(self.markdown_path) if self.markdown_path else None
        payload["plan"] = self.plan.summary()
        payload["model_era_id"] = self.plan.model_era_id
        payload["deployment_id"] = self.plan.deployment_id
        payload["model_version"] = self.plan.model_version
        return payload


def calibrate_monitoring_reference(config: CalibrationConfig) -> CalibrationResult:
    """Build the model-fit reference and resolve thresholds by historical backtest."""
    policy = MonitoringPolicy.load(config.policy_path)
    candidate = None
    if config.retraining_candidate is not None:
        from wind_forecast.retraining_backtesting import (
            RetrainingBacktestError,
            load_retraining_backtest,
        )

        try:
            candidate = load_retraining_backtest(config.retraining_candidate)
        except RetrainingBacktestError as exc:
            raise MonitoringReportingError(str(exc)) from exc
        if candidate["backtest"]["outcome"] != "accepted":
            raise MonitoringReportingError(
                "Candidate-specific calibration requires an accepted backtest."
            )
        candidate_root = Path(config.retraining_candidate)
        if candidate_root.is_file():
            candidate_root = candidate_root.parent
        model_manifest = _read_json(candidate_root / "model_manifest.json")
        candidate_dataset = _read_json(candidate_root / "dataset_manifest.json")
        frame = pd.read_csv(candidate_root / "training_evidence.csv").drop(
            columns=["Expected_Prediction"]
        )
        bundle = {
            "root": candidate_root,
            "feature_names": list(model_manifest["feature_names"]),
            "model_manifest": {
                **model_manifest,
                "task": "daily_wind_production_historical_hindcast",
            },
            "dataset_manifest": {
                **candidate_dataset,
                "sha256": candidate_dataset["final_training_dataset_sha256"],
                "transformation_version": (
                    candidate["backtest"]["identities"].get(
                        "transformation_version"
                    )
                    or "ren_era5_land_daily_features_v2"
                ),
                "splits": {
                    "train": {"start": str(frame[DATE_COLUMN].iloc[0])},
                    "validation": {
                        "end": candidate_dataset["candidate_fit_cutoff"]
                    },
                    "row_counts": {
                        "refit_train_validation": len(frame)
                    },
                },
            },
        }
    else:
        bundle = validate_monitoring_model_bundle(config.model_bundle)
        if sha256_file(config.dataset_path) != bundle["dataset_manifest"]["sha256"]:
            raise MonitoringReportingError("Reference dataset checksum differs from the model bundle.")
        frame = pd.read_csv(config.dataset_path)
    feature_names = list(bundle["feature_names"])
    expected_columns = [DATE_COLUMN, TARGET_COLUMN, *feature_names]
    if frame.columns.tolist() != expected_columns:
        raise MonitoringReportingError("Reference dataset columns/order differ from the accepted model schema.")
    dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise MonitoringReportingError("Reference dates must be valid, unique, and chronological.")
    splits = bundle["dataset_manifest"]["splits"]
    expected_start = str(splits["train"]["start"])
    expected_end = str(splits["validation"]["end"])
    if candidate is None and (
        policy.reference_start != expected_start
        or policy.reference_end != expected_end
    ):
        raise MonitoringReportingError(
            "Monitoring reference boundaries must exactly match train.start and validation.end."
        )
    reference_start = expected_start if candidate is not None else policy.reference_start
    reference_end = expected_end if candidate is not None else policy.reference_end
    mask = dates.between(reference_start, reference_end)
    reference = frame.loc[mask].copy().reset_index(drop=True)
    expected_fit_rows = int(bundle["dataset_manifest"]["splits"]["row_counts"]["refit_train_validation"])
    if len(reference) != expected_fit_rows:
        raise MonitoringReportingError("Reference period row count differs from the model-fit contract.")
    numeric = reference[[TARGET_COLUMN, *feature_names]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise MonitoringReportingError("Reference monitoring values are not finite numeric values.")
    model = joblib.load(config.model_bundle / "model.joblib")
    predictions = np.asarray(model.predict(reference[feature_names]), dtype=float).reshape(-1)
    if not np.isfinite(predictions).all() or len(predictions) != len(reference):
        raise MonitoringReportingError("Reference prediction generation failed validation.")
    reference[PREDICTION_COLUMN] = predictions

    reference_csv = reference.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
    reference_csv_sha = sha256(reference_csv).hexdigest()
    reference_body = {
        "schema_version": REFERENCE_SCHEMA,
        "dataset_sha256": bundle["dataset_manifest"]["sha256"],
        "model_sha256": bundle["model_manifest"]["model_sha256"],
        "feature_schema_sha256": bundle["model_manifest"]["feature_schema_sha256"],
        "transformation_version": bundle["dataset_manifest"]["transformation_version"],
        "period": {"start": reference_start, "end": reference_end},
        "row_count": len(reference),
        "feature_names": feature_names,
        "target": TARGET_COLUMN,
        "reference_prediction_column": PREDICTION_COLUMN,
        "reference_csv_sha256": reference_csv_sha,
        "prediction_role": "in_sample_distribution_reference_only",
        "performance_claim": False,
        "calibration_subject": (
            {
                "kind": "accepted_retraining_candidate",
                "backtest_id": candidate["backtest_id"],
            }
            if candidate is not None
            else {"kind": "accepted_v2_reference"}
        ),
    }
    reference_record = _with_id("monitoring_reference", "reference_id", reference_body)
    reference_dir = config.output_root / "references" / reference_record["reference_id"]
    _immutable_bytes(reference_dir / "reference.csv", reference_csv)
    reference_record["reference_path"] = str((reference_dir / "reference.csv").resolve())
    _immutable_json(reference_dir / "manifest.json", reference_record)

    positive_target = numeric.loc[numeric[TARGET_COLUMN] > 0, TARGET_COLUMN]
    if positive_target.empty:
        raise MonitoringReportingError("MAPE calibration requires positive reference targets.")
    mape_epsilon = float(positive_target.quantile(policy.mape_epsilon_quantile))
    thresholds, backtest_summary = _calibrate_drift_thresholds(
        reference,
        feature_names,
        policy,
        stride=config.backtest_stride_days,
    )
    if candidate is None:
        performance_path = config.model_bundle / "test_predictions.csv"
        _verify_bundle_artifact(config.model_bundle, performance_path.name)
        performance_thresholds, performance_summary = _calibrate_performance_thresholds(
            performance_path,
            str(bundle["model_manifest"]["model_type"]),
            policy,
            mape_epsilon,
        )
    else:
        candidate_predictions = Path(bundle["root"]) / "predictions.csv"
        performance_thresholds, performance_summary = _calibrate_performance_thresholds(
            candidate_predictions,
            "candidate",
            policy,
            mape_epsilon,
        )
    thresholds["performance"] = performance_thresholds
    _apply_threshold_overrides(thresholds, policy.overrides)
    backtest_record = {
        "schema_version": "wind_forecast.monitoring_backtest_summary.v1",
        **backtest_summary,
        "performance": performance_summary,
    }
    backtest_bytes = _json_bytes(backtest_record)
    calibration_body = {
        "schema_version": CALIBRATION_SCHEMA,
        "reference_id": reference_record["reference_id"],
        "reference_manifest_sha256": sha256_file(reference_dir / "manifest.json"),
        "policy": policy.to_dict(),
        "policy_sha256": sha256_file(config.policy_path),
        "backtest_stride_days": config.backtest_stride_days,
        "mape_epsilon": mape_epsilon,
        "mape_epsilon_role": "reference_positive_target_quantile",
        "thresholds": thresholds,
        "backtest_summary": {**backtest_summary, "performance": performance_summary},
        "backtest_summary_sha256": sha256(backtest_bytes).hexdigest(),
        "safeguards": {
            "ledger_prediction_write": False,
            "model_write": False,
            "training": False,
            "network_requests": False,
        },
    }
    calibration_record = _with_id("monitoring_calibration", "calibration_id", calibration_body)
    calibration_dir = config.output_root / "calibrations" / calibration_record["calibration_id"]
    calibration_record["reference_dir"] = str(reference_dir.resolve())
    _immutable_json(calibration_dir / "calibration.json", calibration_record)
    _immutable_bytes(calibration_dir / "backtest_summary.json", backtest_bytes)
    return CalibrationResult(
        reference_id=reference_record["reference_id"],
        calibration_id=calibration_record["calibration_id"],
        reference_dir=reference_dir,
        calibration_dir=calibration_dir,
    )


def load_monitoring_calibration(calibration_dir: str | Path) -> dict[str, Any]:
    """Load a calibration after verifying its content-addressed identity and reference."""
    root = Path(calibration_dir)
    record = _read_json(root / "calibration.json")
    if record.get("schema_version") != CALIBRATION_SCHEMA:
        raise MonitoringReportingError("Unsupported monitoring calibration schema.")
    _verify_record_id(
        record,
        "monitoring_calibration",
        "calibration_id",
        ignored=("reference_dir",),
    )
    reference_id = str(record.get("reference_id") or "")
    recorded_reference_dir = Path(str(record.get("reference_dir") or ""))
    relocatable_reference_dir = root.parent.parent / "references" / reference_id
    reference_dir = (
        recorded_reference_dir
        if recorded_reference_dir.is_dir()
        else relocatable_reference_dir
    )
    manifest_path = reference_dir / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != record.get("reference_manifest_sha256"):
        raise MonitoringReportingError("Monitoring reference manifest is missing or corrupt.")
    reference = _read_json(manifest_path)
    if reference.get("schema_version") != REFERENCE_SCHEMA:
        raise MonitoringReportingError("Unsupported monitoring reference schema.")
    _verify_record_id(reference, "monitoring_reference", "reference_id", ignored=("reference_path",))
    if reference.get("reference_id") != record.get("reference_id"):
        raise MonitoringReportingError("Calibration and reference identities differ.")
    csv_path = reference_dir / "reference.csv"
    if not csv_path.is_file() or sha256_file(csv_path) != reference.get("reference_csv_sha256"):
        raise MonitoringReportingError("Monitoring reference table is missing or corrupt.")
    backtest_path = root / "backtest_summary.json"
    if (
        not backtest_path.is_file()
        or sha256_file(backtest_path) != record.get("backtest_summary_sha256")
    ):
        raise MonitoringReportingError("Monitoring backtest summary is missing or corrupt.")
    record["_reference_manifest"] = reference
    record["_reference_path"] = str(csv_path)
    return record


def plan_monitoring_report(config: MonitoringReportConfig) -> MonitoringReportPlan:
    """Verify report inputs without creating a lock, directory, report, or alert."""
    source_manifest = _load_source_manifest(config.source_run_manifest)
    _validate_source_manifest_date(source_manifest, config.through_date)
    calibration = load_monitoring_calibration(config.calibration_dir)
    era = _verify_reporting_era(config)
    quality_available = _load_quality(source_manifest) is not None
    ledger = load_verified_monitoring_state(config.monitoring_store_root)
    _require_ledger_era(ledger, era)
    return MonitoringReportPlan(
        status="planned",
        source_run_id=str(source_manifest["run_id"]),
        source_status=str(source_manifest["status"]),
        through_date=config.through_date.isoformat(),
        calibration_id=str(calibration["calibration_id"]),
        ledger_available=ledger is not None,
        quality_available=quality_available,
        model_era_id=str(era["model_era_id"]),
        deployment_id=str(era["deployment"]["deployment_id"]),
        model_version=str(era["registry"]["model_version"]),
    )


def run_monitoring_report(config: MonitoringReportConfig) -> MonitoringReportResult:
    """Generate one immutable report and advance only the derived alert pointer."""
    plan = plan_monitoring_report(config)
    if config.dry_run:
        return MonitoringReportResult(
            status="planned",
            run_id=None,
            report_id=None,
            report_path=None,
            markdown_path=None,
            active_alert_count=0,
            plan=plan,
        )
    reporting_root = config.monitoring_store_root / "reporting"
    run_id = _new_run_id(config.now_utc)
    lock = _acquire_lock(reporting_root, run_id)
    run_dir = reporting_root / "runs" / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        era = _verify_reporting_era(config)
        _require_reporting_plan_era(plan, era)
        _immutable_json(
            run_dir / "request.json",
            {
                "schema_version": "wind_forecast.monitoring_report_request.v2",
                "run_id": run_id,
                "requested_at_utc": _utc_text(config.now_utc),
                "config": {
                    "source_run_manifest": str(config.source_run_manifest.resolve()),
                    "monitoring_store_root": str(config.monitoring_store_root.resolve()),
                    "model_bundle": str(config.model_bundle.resolve()),
                    "calibration_dir": str(config.calibration_dir.resolve()),
                    "deployment_root": str(config.deployment_root.resolve()),
                    "through_date": config.through_date.isoformat(),
                },
                "plan": plan.summary(),
                "model_era": era,
            },
        )
        calibration = load_monitoring_calibration(config.calibration_dir)
        policy = _policy_from_payload(calibration["policy"])
        policy.validate()
        reference = pd.read_csv(calibration["_reference_path"])
        source_manifest = _load_source_manifest(config.source_run_manifest)
        _validate_source_manifest_date(source_manifest, config.through_date)
        quality = _load_quality(source_manifest)
        ledger = load_verified_monitoring_state(config.monitoring_store_root)
        observations, ledger_lineage = _load_ledger_observations(
            config.monitoring_store_root,
            ledger,
            config.through_date,
            max(policy.windows_days),
            calibration["_reference_manifest"]["feature_names"],
            era,
        )
        windows, statistical_breaches = _build_window_reports(
            observations,
            reference,
            calibration,
            policy,
            config.through_date,
        )
        quality_breaches = _quality_breaches(quality)
        breaches = quality_breaches + statistical_breaches
        previous_state = _load_report_state(reporting_root)
        if (
            previous_state is not None
            and (
                (
                    previous_state.get("schema_version") == REPORT_STATE_SCHEMA
                    and previous_state.get("model_era_id") != era["model_era_id"]
                )
                or (
                    previous_state.get("schema_version")
                    == LEGACY_REPORT_STATE_SCHEMA
                    and not _legacy_state_is_adoptable(
                        config.monitoring_store_root,
                        era,
                    )
                )
            )
        ):
            previous_state = None
        next_state, alert_events = _evaluate_alerts(
            previous_state,
            config.through_date,
            breaches,
            policy.alert_persistence_distinct_dates,
            era,
        )
        for event in alert_events:
            _immutable_json(
                reporting_root / "alerts" / f"{event['alert_event_id']}.json",
                event,
            )
        report_body = {
            "schema_version": REPORT_SCHEMA,
            "run_id": run_id,
            "created_at_utc": _utc_text(config.now_utc),
            "through_date": config.through_date.isoformat(),
            "model_era": {
                "model_era_id": era["model_era_id"],
                "association_kind": "active_deployment",
                "deployment_id": era["deployment"]["deployment_id"],
                "deployment_state_id": era["deployment"]["deployment_state_id"],
                "deployment_generation": era["deployment"]["generation"],
                "registered_model_name": era["registry"]["registered_model_name"],
                "model_version": era["registry"]["model_version"],
                "cutoffs": era["cutoffs"],
                "pins": era["pins"],
            },
            "source_batch": {
                "run_id": source_manifest["run_id"],
                "status": source_manifest["status"],
                "manifest_path": str(config.source_run_manifest.resolve()),
                "manifest_sha256": sha256_file(config.source_run_manifest),
            },
            "reference": {
                "reference_id": calibration["reference_id"],
                "calibration_id": calibration["calibration_id"],
                "policy_sha256": calibration["policy_sha256"],
            },
            "config": policy.to_dict(),
            "quality": quality or {"status": "not_available", "reason": "legacy_or_missing_quality_sidecar"},
            "windows": windows,
            "breaches": breaches,
            "persistence": next_state["rules"],
            "alert_events": [event["alert_event_id"] for event in alert_events],
            "active_alerts": next_state["active_alerts"],
            "lineage": ledger_lineage,
            "safeguards": {
                "predictions_unchanged": True,
                "models_unchanged": True,
                "as_issued_primary": True,
                "restatements_alerting": False,
                "training": False,
                "network_requests": False,
            },
        }
        report = _with_id("monitoring_report", "report_id", report_body)
        report_dir = reporting_root / "reports" / report["report_id"]
        report_path = report_dir / "report.json"
        markdown_path = report_dir / "report.md"
        _immutable_json(report_path, report)
        _immutable_bytes(markdown_path, _render_markdown(report).encode("utf-8"))
        next_state["schema_version"] = REPORT_STATE_SCHEMA
        next_state["generation"] = int((previous_state or {}).get("generation", 0)) + 1
        next_state["updated_at_utc"] = _utc_text(config.now_utc)
        next_state["latest_report_id"] = report["report_id"]
        next_state["model_era_id"] = era["model_era_id"]
        next_state["deployment_id"] = era["deployment"]["deployment_id"]
        verified_again = _verify_reporting_era(config)
        if not same_model_era(era, verified_again):
            raise MonitoringReportingError(
                "Active deployment changed before report pointer publication."
            )
        _atomic_json(reporting_root / "state" / "current.json", next_state)
        result = MonitoringReportResult(
            status="succeeded",
            run_id=run_id,
            report_id=report["report_id"],
            report_path=report_path,
            markdown_path=markdown_path,
            active_alert_count=len(next_state["active_alerts"]),
            plan=plan,
        )
        _immutable_json(run_dir / "result.json", result.summary())
        return result
    except Exception as exc:
        if run_dir.is_dir():
            _immutable_json(
                run_dir / "failure.json",
                {
                    "schema_version": "wind_forecast.monitoring_report_failure.v1",
                    "run_id": run_id,
                    "failed_at_utc": _utc_text(config.now_utc),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                },
            )
        raise
    finally:
        _release_lock(lock, run_id)


def load_monitoring_report(report_path: str | Path) -> dict[str, Any]:
    """Load and verify one immutable report and all referenced alert events."""
    path = Path(report_path)
    report = _read_json(path)
    if report.get("schema_version") not in {REPORT_SCHEMA, LEGACY_REPORT_SCHEMA}:
        raise MonitoringReportingError("Unsupported monitoring report schema.")
    if not isinstance(report.get("active_alerts"), dict):
        raise MonitoringReportingError("Monitoring report active alerts are invalid.")
    if not isinstance(report.get("alert_events"), list):
        raise MonitoringReportingError("Monitoring report alert events are invalid.")
    _verify_record_id(report, "monitoring_report", "report_id")
    reporting_root = path.parent.parent.parent
    alert_ids = set(report.get("alert_events") or []) | set(
        (report.get("active_alerts") or {}).values()
    ) | {
        item.get("last_event_id")
        for item in (report.get("persistence") or {}).values()
        if item.get("last_event_id")
    }
    for alert_id in alert_ids:
        alert = _read_json(reporting_root / "alerts" / f"{alert_id}.json")
        _verify_record_id(alert, "monitoring_alert", "alert_event_id")
    return report


def resolve_report_model_era(
    monitoring_store_root: str | Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve persisted V2 identity or a strictly evidenced bootstrap adoption."""
    embedded = report.get("model_era")
    if report.get("schema_version") == REPORT_SCHEMA and isinstance(embedded, dict):
        return dict(embedded)
    prediction_ids = {
        str(value)
        for value in ((report.get("lineage") or {}).get("prediction_ids") or [])
    }
    reference = report.get("reference") or {}
    matches: list[dict[str, Any]] = []
    for era in list_model_eras(monitoring_store_root):
        adopted = era.get("_adopted_state") or {}
        adopted_ids = {
            str(value)
            for value in (adopted.get("as_issued") or {}).values()
        }
        if (
            era.get("calibration", {}).get("calibration_id")
            == reference.get("calibration_id")
            and prediction_ids.issubset(adopted_ids)
        ):
            matches.append(era)
    if len(matches) != 1:
        return {"association_kind": "legacy_unassociated"}
    era = matches[0]
    return {
        "model_era_id": era["model_era_id"],
        "association_kind": "bootstrap_adopted",
        "deployment_id": era["deployment"]["deployment_id"],
        "deployment_state_id": era["deployment"]["deployment_state_id"],
        "deployment_generation": era["deployment"]["generation"],
        "registered_model_name": era["registry"]["registered_model_name"],
        "model_version": era["registry"]["model_version"],
        "cutoffs": era["cutoffs"],
        "pins": era["pins"],
    }


def load_monitoring_report_state(
    monitoring_store_root: str | Path,
) -> dict[str, Any] | None:
    """Return a verified copy of the derived reporting-state pointer."""
    state = _load_report_state(Path(monitoring_store_root) / "reporting")
    return None if state is None else json.loads(json.dumps(state))


def load_reporting_attempts(
    monitoring_store_root: str | Path,
) -> list[dict[str, Any]]:
    """Return verified reporting attempts in deterministic newest-first order."""
    store_root = Path(monitoring_store_root)
    if store_root.exists() and not store_root.is_dir():
        raise MonitoringReportingError(
            "The configured monitoring store root is not a directory."
        )
    runs_root = store_root / "reporting" / "runs"
    if not runs_root.is_dir():
        return []
    run_dirs = [path for path in sorted(runs_root.iterdir()) if path.is_dir()]
    attempts = _load_paths_in_order(
        run_dirs,
        lambda path: _load_reporting_attempt(store_root, path),
    )
    return sorted(
        attempts,
        key=lambda item: (
            str(item.get("attempted_at_utc") or ""),
            str(item.get("run_id") or ""),
        ),
        reverse=True,
    )


def load_reporting_attempt(
    monitoring_store_root: str | Path,
    *,
    reporting_run_id: str | None = None,
    report_id: str | None = None,
) -> dict[str, Any] | None:
    """Return one verified reporting attempt selected by one exact identity."""
    if (reporting_run_id is None) == (report_id is None):
        raise ValueError("Exactly one reporting-run or report identifier is required.")
    attempts = load_reporting_attempts(monitoring_store_root)
    key = "run_id" if reporting_run_id is not None else "report_id"
    expected = reporting_run_id if reporting_run_id is not None else report_id
    matches = [item for item in attempts if item.get(key) == expected]
    if len(matches) > 1:
        raise MonitoringReportingError(
            "Reporting attempt identities are not unique."
        )
    return matches[0] if matches else None


def _load_reporting_attempt(
    monitoring_store_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    request = _read_json(run_dir / "request.json")
    run_id = run_dir.name
    plan = request.get("plan")
    if (
        request.get("schema_version")
        not in {
            "wind_forecast.monitoring_report_request.v1",
            "wind_forecast.monitoring_report_request.v2",
        }
        or request.get("run_id") != run_id
        or not isinstance(plan, dict)
        or plan.get("status") != "planned"
    ):
        raise MonitoringReportingError("Invalid monitoring reporting request.")
    attempted_at = _reporting_utc_text(request, "requested_at_utc")
    through_date = _reporting_date_text(plan, "through_date")
    source_run_id = _reporting_text(plan, "source_run_id")
    source_status = _reporting_text(plan, "source_status")
    calibration_id = _reporting_text(plan, "calibration_id")
    result_path = run_dir / "result.json"
    failure_path = run_dir / "failure.json"
    if result_path.is_file() and failure_path.is_file():
        raise MonitoringReportingConflictError(
            "Reporting run has conflicting outcomes."
        )
    base = {
        "run_id": run_id,
        "attempted_at_utc": attempted_at,
        "through_date": through_date,
        "source_pipeline_run_id": source_run_id,
        "source_pipeline_status": source_status,
    }
    if result_path.is_file():
        result = _read_json(result_path)
        loaded_report_id = str(result.get("report_id") or "")
        active_alert_count = result.get("active_alert_count")
        if (
            result.get("run_id") != run_id
            or result.get("status") != "succeeded"
            or not loaded_report_id
            or result.get("plan") != plan
            or not isinstance(active_alert_count, int)
            or isinstance(active_alert_count, bool)
            or active_alert_count < 0
        ):
            raise MonitoringReportingError("Invalid monitoring reporting result.")
        report = load_monitoring_report(
            monitoring_store_root
            / "reporting"
            / "reports"
            / loaded_report_id
            / "report.json"
        )
        report_reference = report.get("reference") or {}
        report_source = report.get("source_batch") or {}
        if (
            report.get("run_id") != run_id
            or report.get("through_date") != through_date
            or report_source.get("run_id") != source_run_id
            or report_source.get("status") != source_status
            or report_reference.get("calibration_id") != calibration_id
            or len(report.get("active_alerts") or {}) != active_alert_count
        ):
            raise MonitoringReportingConflictError(
                "Reporting request, result, and report lineage differ."
            )
        return {
            **base,
            "status": "succeeded",
            "report_id": loaded_report_id,
            "active_alert_count": active_alert_count,
            "failure": None,
        }
    if failure_path.is_file():
        failure = _read_json(failure_path)
        if (
            failure.get("schema_version")
            != "wind_forecast.monitoring_report_failure.v1"
            or failure.get("run_id") != run_id
        ):
            raise MonitoringReportingError("Invalid monitoring reporting failure.")
        failed_at = _reporting_utc_text(failure, "failed_at_utc")
        error_type = _reporting_text(failure, "error_type")
        return {
            **base,
            "status": "failed",
            "report_id": None,
            "active_alert_count": 0,
            "failure": {
                "failed_at_utc": failed_at,
                "error_type": error_type,
                "message": (
                    "The reporting attempt failed. Inspect local operator logs."
                ),
            },
        }
    return {
        **base,
        "status": "in_progress",
        "report_id": None,
        "active_alert_count": 0,
        "failure": None,
    }


def _reporting_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise MonitoringReportingError(
            f"Monitoring metadata field is invalid: {field}."
        )
    return value


def _reporting_date_text(payload: Mapping[str, Any], field: str) -> str:
    value = _reporting_text(payload, field)
    date.fromisoformat(value)
    return value


def _reporting_utc_text(payload: Mapping[str, Any], field: str) -> str:
    value = _reporting_text(payload, field)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise MonitoringReportingError(
            f"Monitoring timestamp is not timezone-aware: {field}."
        )
    return value


def load_active_alerts(monitoring_store_root: str | Path) -> dict[str, Any]:
    """Return the verified derived active-alert view."""
    root = Path(monitoring_store_root) / "reporting"
    state = _load_report_state(root)
    if state is None:
        return {}
    for alert_id in state.get("active_alerts", {}).values():
        alert = _read_json(root / "alerts" / f"{alert_id}.json")
        _verify_record_id(alert, "monitoring_alert", "alert_event_id")
    return dict(state.get("active_alerts") or {})


def load_alert_history(
    monitoring_store_root: str | Path, *, rule_id: str | None = None
) -> list[dict[str, Any]]:
    """Return immutable, identity-verified alert events in deterministic order."""
    root = Path(monitoring_store_root) / "reporting" / "alerts"
    if not root.is_dir():
        return []
    paths = sorted(root.glob("*.json"))
    loaded_events = _load_paths_in_order(paths, _load_alert_event)
    events: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for event in loaded_events:
        by_id[str(event["alert_event_id"])] = event
        if rule_id is None or event.get("rule_id") == rule_id:
            events.append(event)
    for event in events:
        previous_id = event.get("previous_alert_event_id")
        if previous_id is not None and str(previous_id) not in by_id:
            raise MonitoringReportingError("Alert history contains a broken event chain.")
        if previous_id is not None and by_id[str(previous_id)].get("rule_id") != event.get("rule_id"):
            raise MonitoringReportingError("Alert history crosses rule identities.")
    depth_cache: dict[str, int] = {}

    def causal_depth(event_id: str, visiting: set[str] | None = None) -> int:
        if event_id in depth_cache:
            return depth_cache[event_id]
        active = set(visiting or ())
        if event_id in active:
            raise MonitoringReportingError("Alert history contains a causal cycle.")
        active.add(event_id)
        previous_id = by_id[event_id].get("previous_alert_event_id")
        depth = 0 if previous_id is None else causal_depth(str(previous_id), active) + 1
        depth_cache[event_id] = depth
        return depth

    return sorted(
        events,
        key=lambda item: (
            str(item.get("through_date")),
            str(item.get("rule_id")),
            causal_depth(str(item.get("alert_event_id"))),
            str(item.get("alert_event_id")),
        ),
    )


def _load_alert_event(path: Path) -> dict[str, Any]:
    event = _read_json(path)
    _verify_record_id(event, "monitoring_alert", "alert_event_id")
    return event


def _load_paths_in_order(
    paths: Sequence[Path],
    loader: Callable[[Path], _LoadedRecord],
) -> list[_LoadedRecord]:
    """Load independent files concurrently while observing results in path order."""
    if len(paths) < _PARALLEL_LOADER_MIN_ITEMS:
        return [loader(path) for path in paths]
    # Eight workers bound file-descriptor pressure while overlapping local JSON reads.
    try:
        executor = ThreadPoolExecutor(max_workers=_PARALLEL_LOADER_MAX_WORKERS)
    except (OSError, RuntimeError) as exc:
        raise MonitoringReportingUnavailableError(
            "Parallel monitoring artifact loading is unavailable."
        ) from exc
    with executor:
        try:
            loaded = executor.map(loader, paths)
        except (OSError, RuntimeError) as exc:
            raise MonitoringReportingUnavailableError(
                "Parallel monitoring artifact loading is unavailable."
            ) from exc
        return list(loaded)


def _calibrate_drift_thresholds(
    reference: pd.DataFrame,
    feature_names: Sequence[str],
    policy: MonitoringPolicy,
    *,
    stride: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scalar = [
        name
        for name in feature_names
        if name != RAW_DIRECTION_COLUMN and name not in DIRECTION_COMPONENTS
    ]
    direction_groups = [
        name
        for name, pair in DIRECTION_GROUPS.items()
        if set(pair).issubset(feature_names)
    ]
    entities = [*scalar, *direction_groups]
    thresholds: dict[str, Any] = {"feature_drift": {}, "prediction_drift": {}, "target_drift": {}, "coverage": {}}
    summary: dict[str, Any] = {"feature_entities": len(entities), "windows": {}}
    endpoints = pd.to_datetime(reference[DATE_COLUMN]).iloc[::stride].tolist()
    if pd.Timestamp(reference[DATE_COLUMN].iloc[-1]) not in endpoints:
        endpoints.append(pd.Timestamp(reference[DATE_COLUMN].iloc[-1]))
    for window in policy.windows_days:
        samples: dict[str, dict[str, dict[str, list[float]]]] = {
            entity: {
                comparator: {"normalized_wasserstein": [], "ks_statistic": []}
                for comparator in ("global", "seasonal")
            }
            for entity in entities
        }
        special = {
            PREDICTION_COLUMN: {c: {"normalized_wasserstein": [], "ks_statistic": []} for c in ("global", "seasonal")},
            TARGET_COLUMN: {c: {"normalized_wasserstein": [], "ks_statistic": []} for c in ("global", "seasonal")},
        }
        coverage_values: list[float] = []
        accepted_windows = 0
        for endpoint in endpoints:
            current = calendar_window(reference, endpoint, window)
            minimum = policy.minimum_samples[str(window)]
            if len(current) < minimum:
                continue
            accepted_windows += 1
            coverage_values.append(len(current) / window)
            current_dates = pd.to_datetime(current[DATE_COLUMN])
            global_ref = reference.loc[~pd.to_datetime(reference[DATE_COLUMN]).isin(current_dates)]
            seasonal_ref = seasonal_reference(
                reference,
                current_dates,
                exclude_dates=current_dates,
            )
            for comparator, comparison in (("global", global_ref), ("seasonal", seasonal_ref)):
                for entity in entities:
                    stats = _entity_statistics(entity, current, comparison)
                    for detector in ("normalized_wasserstein", "ks_statistic"):
                        samples[entity][comparator][detector].append(float(stats[detector]))
                for column in (PREDICTION_COLUMN, TARGET_COLUMN):
                    stats = drift_statistics(current[column], comparison[column])
                    for detector in ("normalized_wasserstein", "ks_statistic"):
                        special[column][comparator][detector].append(float(stats[detector]))
        if accepted_windows == 0:
            raise MonitoringReportingError(f"No {window}-day backtest window met the sample contract.")
        for entity in entities:
            entity_root = thresholds["feature_drift"].setdefault(entity, {}).setdefault(str(window), {})
            for comparator in ("global", "seasonal"):
                entity_root[comparator] = {
                    detector: calibrated_limits(values, policy)
                    for detector, values in samples[entity][comparator].items()
                }
        for column, section in ((PREDICTION_COLUMN, "prediction_drift"), (TARGET_COLUMN, "target_drift")):
            window_root = thresholds[section].setdefault(str(window), {})
            for comparator in ("global", "seasonal"):
                window_root[comparator] = {
                    detector: calibrated_limits(values, policy)
                    for detector, values in special[column][comparator].items()
                }
        thresholds["coverage"][str(window)] = calibrated_limits(
            coverage_values, policy, lower_is_bad=True
        )
        summary["windows"][str(window)] = {
            "accepted_backtest_windows": accepted_windows,
            "minimum_samples": policy.minimum_samples[str(window)],
            "coverage_min": min(coverage_values),
            "coverage_max": max(coverage_values),
        }
    return thresholds, summary


def _calibrate_performance_thresholds(
    path: Path,
    selected_model: str,
    policy: MonitoringPolicy,
    mape_epsilon: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {DATE_COLUMN, "Actual_Wind_Production", "model", "Predicted_Wind_Production"}
    if not required.issubset(frame.columns):
        raise MonitoringReportingError("Sealed test prediction artifact has an incompatible schema.")
    frame = frame.loc[frame["model"].astype(str).eq(selected_model)].copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if frame[DATE_COLUMN].isna().any() or frame[DATE_COLUMN].duplicated().any():
        raise MonitoringReportingError("Sealed test prediction dates are invalid or duplicated.")
    thresholds: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for window in policy.windows_days:
        values = {key: [] for key in ("MAE", "RMSE", "absolute_bias", "R2", "MAPE_percent")}
        for endpoint in frame[DATE_COLUMN]:
            current = calendar_window(frame, endpoint, window)
            if len(current) < policy.minimum_samples[str(window)]:
                continue
            metrics = regression_metrics(
                current["Actual_Wind_Production"],
                current["Predicted_Wind_Production"],
                mape_epsilon=mape_epsilon,
                r2_minimum_samples=policy.r2_minimum_samples[str(window)],
            )
            for key in ("MAE", "RMSE", "MAPE_percent"):
                values[key].append(float(metrics[key]))
            values["absolute_bias"].append(abs(float(metrics["bias"])))
            if metrics["R2"] is not None:
                values["R2"].append(float(metrics["R2"]))
        thresholds[str(window)] = {
            key: calibrated_limits(sample, policy, lower_is_bad=key == "R2")
            for key, sample in values.items()
            if sample
        }
        summary[str(window)] = {key: len(sample) for key, sample in values.items()}
    return thresholds, summary


def _apply_threshold_overrides(
    thresholds: dict[str, Any],
    overrides: Mapping[str, Mapping[str, float]],
) -> None:
    for path, values in sorted(overrides.items()):
        parts = path.split(".")
        node: Any = thresholds
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise MonitoringReportingError(
                    f"Monitoring threshold override path does not exist: {path}."
                )
            node = node[part]
        if not isinstance(node, dict) or not {"warning", "critical", "direction"}.issubset(node):
            raise MonitoringReportingError(
                f"Monitoring threshold override is not a limit leaf: {path}."
            )
        warning = float(values["warning"])
        critical = float(values["critical"])
        direction = str(node["direction"])
        valid_order = warning <= critical if direction == "upper" else warning >= critical
        if not valid_order:
            raise MonitoringReportingError(
                f"Monitoring threshold override severity order is invalid: {path}."
            )
        node.update(
            {
                "warning": warning,
                "critical": critical,
                "override": True,
                "calibrated_warning": node["warning"],
                "calibrated_critical": node["critical"],
            }
        )


def _entity_statistics(entity: str, current: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    if entity in DIRECTION_GROUPS:
        sine, cosine = DIRECTION_GROUPS[entity]
        return circular_drift_statistics(current, reference, sine, cosine)
    return drift_statistics(current[entity], reference[entity])


def _build_window_reports(
    observations: pd.DataFrame,
    reference: pd.DataFrame,
    calibration: Mapping[str, Any],
    policy: MonitoringPolicy,
    through_date: date,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if observations.empty:
        return {
            str(window): {"status": "not_available", "reason": "prediction_ledger_unavailable"}
            for window in policy.windows_days
        }, []
    thresholds = calibration["thresholds"]
    feature_names = calibration["_reference_manifest"]["feature_names"]
    entities = [
        name
        for name in feature_names
        if name != RAW_DIRECTION_COLUMN and name not in DIRECTION_COMPONENTS
    ] + [
        name
        for name, pair in DIRECTION_GROUPS.items()
        if set(pair).issubset(feature_names)
    ]
    windows: dict[str, Any] = {}
    breaches: list[dict[str, Any]] = []
    for window in policy.windows_days:
        current = calendar_window(observations, through_date.isoformat(), window)
        sample_count = len(current)
        minimum = policy.minimum_samples[str(window)]
        if sample_count < minimum:
            windows[str(window)] = {
                "status": "insufficient_data",
                "sample_count": sample_count,
                "minimum_samples": minimum,
            }
            continue
        seasonal = seasonal_reference(reference, current[DATE_COLUMN])
        feature_results: dict[str, Any] = {}
        for entity in entities:
            comparisons: dict[str, Any] = {}
            for comparator, comparison in (("global", reference), ("seasonal", seasonal)):
                stats = _entity_statistics(entity, current, comparison)
                severity = _statistics_severity(
                    stats,
                    thresholds["feature_drift"][entity][str(window)][comparator],
                )
                stats["severity"] = severity
                comparisons[comparator] = stats
                if severity in {"warning", "critical"}:
                    breaches.append(
                        _breach(
                            f"feature_drift:{entity}:{window}:{comparator}",
                            severity,
                            category="feature_drift",
                            immediate=False,
                        )
                    )
            feature_results[entity] = comparisons
        prediction = _drift_section(
            current[LEDGER_PREDICTION_COLUMN],
            current[DATE_COLUMN],
            reference,
            PREDICTION_COLUMN,
            thresholds["prediction_drift"][str(window)],
            f"prediction_drift:{window}",
            breaches,
        )
        actual_rows = current.dropna(subset=[ACTUAL_COLUMN])
        target: dict[str, Any]
        performance: dict[str, Any]
        if len(actual_rows) < minimum:
            target = {"status": "insufficient_data", "sample_count": len(actual_rows)}
            performance = {"status": "insufficient_data", "sample_count": len(actual_rows)}
        else:
            target = _drift_section(
                actual_rows[ACTUAL_COLUMN],
                actual_rows[DATE_COLUMN],
                reference,
                TARGET_COLUMN,
                thresholds["target_drift"][str(window)],
                f"target_drift:{window}",
                breaches,
            )
            metrics = regression_metrics(
                actual_rows[ACTUAL_COLUMN],
                actual_rows[LEDGER_PREDICTION_COLUMN],
                mape_epsilon=float(calibration["mape_epsilon"]),
                r2_minimum_samples=policy.r2_minimum_samples[str(window)],
            )
            metric_severity: dict[str, str] = {}
            for key in ("MAE", "RMSE", "MAPE_percent", "R2"):
                if key not in thresholds["performance"][str(window)]:
                    continue
                severity = threshold_severity(
                    metrics.get(key), thresholds["performance"][str(window)][key]
                )
                metric_severity[key] = severity
                if severity in {"warning", "critical"}:
                    breaches.append(
                        _breach(f"performance:{key}:{window}", severity, category="performance", immediate=False)
                    )
            bias_severity = threshold_severity(
                abs(float(metrics["bias"])),
                thresholds["performance"][str(window)]["absolute_bias"],
            )
            metric_severity["bias"] = bias_severity
            if bias_severity in {"warning", "critical"}:
                breaches.append(
                    _breach(f"performance:bias:{window}", bias_severity, category="performance", immediate=False)
                )
            performance = {"status": "available", "metrics": metrics, "severity": metric_severity}
        coverage_ratio = sample_count / window
        coverage_severity = threshold_severity(
            coverage_ratio, thresholds["coverage"][str(window)]
        )
        if coverage_severity in {"warning", "critical"}:
            breaches.append(
                _breach(f"coverage:prediction_samples:{window}", coverage_severity, category="quality", immediate=True)
            )
        windows[str(window)] = {
            "status": "available",
            "calendar_start": (pd.Timestamp(through_date) - pd.Timedelta(days=window - 1)).date().isoformat(),
            "calendar_end": through_date.isoformat(),
            "sample_count": sample_count,
            "coverage_ratio": coverage_ratio,
            "coverage_severity": coverage_severity,
            "feature_drift": feature_results,
            "prediction_drift": prediction,
            "target_drift": target,
            "performance": performance,
        }
    return windows, _dedupe_breaches(breaches)


def _drift_section(
    current_values: pd.Series,
    current_dates: pd.Series,
    reference: pd.DataFrame,
    reference_column: str,
    limits: Mapping[str, Any],
    rule_prefix: str,
    breaches: list[dict[str, Any]],
) -> dict[str, Any]:
    seasonal = seasonal_reference(reference, current_dates)
    result: dict[str, Any] = {}
    for comparator, comparison in (("global", reference), ("seasonal", seasonal)):
        stats = drift_statistics(current_values, comparison[reference_column])
        severity = _statistics_severity(stats, limits[comparator])
        stats["severity"] = severity
        result[comparator] = stats
        if severity in {"warning", "critical"}:
            breaches.append(
                _breach(f"{rule_prefix}:{comparator}", severity, category=rule_prefix.split(":", 1)[0], immediate=False)
            )
    return result


def _statistics_severity(stats: Mapping[str, Any], limits: Mapping[str, Any]) -> str:
    severities = [
        threshold_severity(float(stats[detector]), limits[detector])
        for detector in ("normalized_wasserstein", "ks_statistic")
    ]
    return max(severities, key=SEVERITY_ORDER.get)


def _load_ledger_observations(
    root: Path,
    state: Mapping[str, Any] | None,
    through_date: date,
    window_days: int,
    feature_names: Sequence[str],
    model_era: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [DATE_COLUMN, *feature_names, LEDGER_PREDICTION_COLUMN, ACTUAL_COLUMN]
    if state is None:
        return pd.DataFrame(columns=columns), {"ledger_status": "not_available"}
    start = through_date - pd.Timedelta(days=window_days - 1).to_pytimedelta()
    rows: list[dict[str, Any]] = []
    prediction_ids: list[str] = []
    actual_ids: list[str] = []
    input_ids: list[str] = []
    adopted_ids = _adopted_prediction_ids(root, state, model_era)
    for day, prediction_id in sorted((state.get("as_issued") or {}).items()):
        day_date = date.fromisoformat(day)
        if not start <= day_date <= through_date:
            continue
        evidence = load_prediction_evidence(root, str(prediction_id))
        prediction = evidence["prediction"]
        if (
            prediction.get("model_era_id") != model_era["model_era_id"]
            and str(prediction_id) not in adopted_ids
        ):
            continue
        snapshot = evidence["model_input_snapshot"]
        if snapshot["feature_names"] != list(feature_names):
            raise MonitoringReportingError("Ledger and calibration feature orders differ.")
        row = {DATE_COLUMN: day}
        row.update(dict(zip(snapshot["feature_names"], snapshot["feature_values"], strict=True)))
        row[LEDGER_PREDICTION_COLUMN] = float(prediction["prediction"])
        row[ACTUAL_COLUMN] = math.nan
        current_metric_id = (state.get("metrics") or {}).get(str(prediction_id))
        if current_metric_id:
            metric = next(
                (item for item in evidence["metric_revisions"] if item["metric_revision_id"] == current_metric_id),
                None,
            )
            if metric is None:
                raise MonitoringReportingError("Current metric is absent from verified evidence.")
            actual = next(
                item for item in evidence["actual_revisions"] if item["actual_revision_id"] == metric["actual_revision_id"]
            )
            row[ACTUAL_COLUMN] = float(actual["actual"])
            actual_ids.append(actual["actual_revision_id"])
        rows.append(row)
        prediction_ids.append(prediction["prediction_id"])
        input_ids.append(snapshot["model_input_snapshot_id"])
    frame = pd.DataFrame(rows, columns=columns)
    return frame, {
        "ledger_status": "available",
        "ledger_generation": state.get("generation"),
        "source_generation": state.get("source_generation"),
        "prediction_ids": prediction_ids,
        "actual_revision_ids": actual_ids,
        "model_input_snapshot_ids": input_ids,
        "restated_prediction_count": len(state.get("restated") or {}),
        "primary_view": "as_issued",
        "model_era_id": model_era["model_era_id"],
        "deployment_id": model_era["deployment"]["deployment_id"],
        "model_version": model_era["registry"]["model_version"],
    }


def _adopted_prediction_ids(
    root: Path,
    state: Mapping[str, Any],
    model_era: Mapping[str, Any],
) -> set[str]:
    if (
        state.get("schema_version") in {None, LEGACY_STATE_SCHEMA}
        and (
            state.get("model_snapshot_id") is None
            or state.get("model_snapshot_id")
            == (model_era.get("monitoring") or {}).get(
                "ledger_model_snapshot_id"
            )
        )
    ):
        return {
            str(value)
            for value in (state.get("as_issued") or {}).values()
        }
    try:
        stored = load_model_era(root, str(model_era["model_era_id"]))
    except (MonitoringError, OSError):
        return set()
    adopted = stored.get("_adopted_state") or {}
    return {
        str(value)
        for value in (adopted.get("as_issued") or {}).values()
    }


def _load_source_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") not in SOURCE_RUN_SCHEMAS:
        raise MonitoringReportingError("Unsupported Phase 8 source-run manifest schema.")
    if not payload.get("run_id") or payload.get("status") not in {"succeeded", "no_op", "failed"}:
        raise MonitoringReportingError("Invalid Phase 8 source-run manifest.")
    return payload


def _validate_source_manifest_date(
    source_manifest: Mapping[str, Any], through_date: date
) -> None:
    recorded = str((source_manifest.get("command") or {}).get("through_date") or "")
    if recorded != through_date.isoformat():
        raise MonitoringReportingError(
            "Phase 8 source-run through_date differs from the report through_date."
        )


def _load_quality(source_manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    ref = source_manifest.get("quality_evidence") or {}
    path = Path(str(ref.get("path") or ""))
    required = source_manifest.get("schema_version") == "wind_forecast.v2_incremental_run.v2"
    if not path.is_file():
        if required:
            raise MonitoringReportingError("Batch quality sidecar is missing.")
        return None
    if sha256_file(path) != ref.get("sha256"):
        raise MonitoringReportingError("Batch quality sidecar checksum is invalid.")
    payload = _read_json(path)
    if payload.get("schema_version") != "wind_forecast.batch_quality.v1":
        raise MonitoringReportingError("Unsupported batch quality schema.")
    expected_through = str((source_manifest.get("command") or {}).get("through_date") or "")
    if (
        payload.get("run_id") != source_manifest.get("run_id")
        or payload.get("batch_status") != source_manifest.get("status")
        or payload.get("through_date") != expected_through
    ):
        raise MonitoringReportingError(
            "Batch quality sidecar does not belong to the source-run manifest."
        )
    return payload


def _quality_breaches(quality: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if quality is None:
        return []
    breaches = []
    immediate_codes = {
        "schema_validation_failed",
        "missing_required_schema_columns",
        "schema_column_order_changed",
        "schema_column_type_changed",
        "duplicate_validation_failed",
        "null_validation_failed",
        "finiteness_validation_failed",
        "interval_validation_failed",
        "invalid_complete_interval_count",
        "source_checksum_mismatch",
        "source_late",
    }
    alertable_codes = immediate_codes
    for issue in quality.get("issues") or []:
        severity = str(issue.get("severity"))
        if severity not in {"warning", "critical"}:
            continue
        code = str(issue.get("code"))
        if code not in alertable_codes:
            continue
        breaches.append(
            _breach(
                f"quality:{code}",
                severity,
                category="quality",
                immediate=code in immediate_codes,
            )
        )
    return _dedupe_breaches(breaches)


def _breach(rule_id: str, severity: str, *, category: str, immediate: bool) -> dict[str, Any]:
    return {"rule_id": rule_id, "severity": severity, "category": category, "immediate": immediate}


def _dedupe_breaches(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_rule: dict[str, dict[str, Any]] = {}
    for item in items:
        rule = str(item["rule_id"])
        current = by_rule.get(rule)
        if current is None or SEVERITY_ORDER[str(item["severity"])] > SEVERITY_ORDER[str(current["severity"])]:
            by_rule[rule] = dict(item)
    return [by_rule[key] for key in sorted(by_rule)]


def _evaluate_alerts(
    previous: Mapping[str, Any] | None,
    through_date: date,
    breaches: Sequence[Mapping[str, Any]],
    persistence: int,
    model_era: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rules = json.loads(json.dumps(dict((previous or {}).get("rules") or {})))
    active = dict((previous or {}).get("active_alerts") or {})
    events: list[dict[str, Any]] = []
    current = {str(item["rule_id"]): dict(item) for item in breaches}
    day = through_date.isoformat()
    previous_day = (previous or {}).get("latest_through_date")
    if previous_day is None:
        prior_dates = [str(item.get("last_date")) for item in rules.values() if item.get("last_date")]
        previous_day = max(prior_dates, default=None)
    if previous_day is not None and through_date < date.fromisoformat(str(previous_day)):
        raise MonitoringReportingError(
            "Monitoring report through_date cannot precede the alert-state date."
        )
    for rule_id, breach in current.items():
        prior = dict(rules.get(rule_id) or {})
        same_date = prior.get("last_date") == day
        prior_day = date.fromisoformat(str(prior["last_date"])) if prior.get("last_date") else None
        if same_date:
            consecutive = max(int(prior.get("consecutive", 0)), 1)
        elif prior_day is not None and prior_day + timedelta(days=1) == through_date:
            consecutive = int(prior.get("consecutive", 0)) + 1
        else:
            consecutive = 1
        required = 1 if breach.get("immediate") else persistence
        was_active = bool(prior.get("active"))
        severity = str(breach["severity"])
        event_type = None
        if not was_active and consecutive >= required:
            event_type = "opened"
        elif was_active and SEVERITY_ORDER[severity] > SEVERITY_ORDER.get(str(prior.get("severity")), 0):
            event_type = "escalated"
        if event_type:
            event = _alert_event(
                rule_id,
                day,
                event_type,
                severity,
                prior.get("last_event_id") or active.get(rule_id),
                model_era,
            )
            events.append(event)
            active[rule_id] = event["alert_event_id"]
        rules[rule_id] = {
            "consecutive": consecutive,
            "last_date": day,
            "active": was_active or event_type in {"opened", "escalated"},
            "severity": (
                max(
                    (str(prior.get("severity")), severity),
                    key=lambda value: SEVERITY_ORDER.get(value, 0),
                )
                if was_active
                else severity
            ),
            "required": required,
            "last_event_id": (
                event["alert_event_id"] if event_type else prior.get("last_event_id")
            ),
        }
    for rule_id, prior_value in list(rules.items()):
        if rule_id in current:
            continue
        prior = dict(prior_value)
        if prior.get("active"):
            event = _alert_event(
                rule_id,
                day,
                "resolved",
                "ok",
                prior.get("last_event_id") or active.get(rule_id),
                model_era,
            )
            events.append(event)
        active.pop(rule_id, None)
        rules[rule_id] = {
            "consecutive": 0,
            "last_date": day,
            "active": False,
            "severity": "ok",
            "required": prior.get("required", persistence),
            "last_event_id": (
                event["alert_event_id"] if prior.get("active") else prior.get("last_event_id")
            ),
        }
    return {
        "schema_version": REPORT_STATE_SCHEMA,
        "latest_through_date": day,
        "rules": rules,
        "active_alerts": active,
        "model_era_id": model_era["model_era_id"],
        "deployment_id": model_era["deployment"]["deployment_id"],
    }, events


def _alert_event(
    rule_id: str,
    through_date: str,
    event_type: str,
    severity: str,
    previous_alert_event_id: str | None,
    model_era: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": ALERT_SCHEMA,
        "rule_id": rule_id,
        "through_date": through_date,
        "event_type": event_type,
        "severity": severity,
        "previous_alert_event_id": previous_alert_event_id,
        "model_era_id": model_era["model_era_id"],
        "deployment_id": model_era["deployment"]["deployment_id"],
        "model_version": model_era["registry"]["model_version"],
        "delivery": "local_immutable_record",
    }
    return _with_id("monitoring_alert", "alert_event_id", body)


def _load_report_state(reporting_root: Path) -> dict[str, Any] | None:
    path = reporting_root / "state" / "current.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    if payload.get("schema_version") not in {
        REPORT_STATE_SCHEMA,
        LEGACY_REPORT_STATE_SCHEMA,
    }:
        raise MonitoringReportingError("Unsupported monitoring report-state schema.")
    active_alerts = payload.get("active_alerts")
    rules = payload.get("rules")
    if not isinstance(active_alerts, dict) or not isinstance(rules, dict):
        raise MonitoringReportingError("Monitoring report state is invalid.")
    if not all(
        isinstance(rule_id, str) and isinstance(alert_id, str)
        for rule_id, alert_id in active_alerts.items()
    ) or not all(
        isinstance(rule_id, str) and isinstance(rule, dict)
        for rule_id, rule in rules.items()
    ):
        raise MonitoringReportingError("Monitoring report state is invalid.")
    alert_ids = set(active_alerts.values()) | {
        rule.get("last_event_id")
        for rule in rules.values()
        if rule.get("last_event_id")
    }
    for alert_id in alert_ids:
        alert = _read_json(reporting_root / "alerts" / f"{alert_id}.json")
        _verify_record_id(alert, "monitoring_alert", "alert_event_id")
    return payload


def _verify_bundle_artifact(bundle_root: Path, filename: str) -> None:
    summary = _read_json(bundle_root / "run_summary.json")
    expected = (summary.get("artifact_sha256") or {}).get(filename)
    path = bundle_root / filename
    if not path.is_file() or expected != sha256_file(path):
        raise MonitoringReportingError(f"Model-bundle artifact is missing or corrupt: {filename}.")


def _policy_from_payload(payload: Mapping[str, Any]) -> MonitoringPolicy:
    policy = MonitoringPolicy(
        reference_start=str(payload["reference_start"]),
        reference_end=str(payload["reference_end"]),
        windows_days=tuple(int(value) for value in payload["windows_days"]),
        warning_quantile=float(payload["warning_quantile"]),
        critical_quantile=float(payload["critical_quantile"]),
        minimum_samples={str(key): int(value) for key, value in payload["minimum_samples"].items()},
        r2_minimum_samples={
            str(key): int(value) for key, value in payload["r2_minimum_samples"].items()
        },
        mape_epsilon_quantile=float(payload["mape_epsilon_quantile"]),
        alert_persistence_distinct_dates=int(payload["alert_persistence_distinct_dates"]),
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    quality = report.get("quality") or {}
    config = report.get("config") or {}
    lineage = report.get("lineage") or {}
    lines = [
        "# Historical batch monitoring report",
        "",
        f"- Report: `{report['report_id']}`",
        f"- Through date: `{report['through_date']}`",
        f"- Source batch: `{report['source_batch']['run_id']}` ({report['source_batch']['status']})",
        f"- Calibration: `{report['reference']['calibration_id']}`",
        f"- Reference: `{report['reference']['reference_id']}`",
        f"- Active alerts: `{len(report.get('active_alerts') or {})}`",
        "",
        "## Configuration",
        "",
        f"- Windows: `{config.get('windows_days')}`",
        f"- Warning/critical quantiles: `{config.get('warning_quantile')}` / `{config.get('critical_quantile')}`",
        f"- Minimum samples: `{config.get('minimum_samples')}`",
        f"- R2 minimum samples: `{config.get('r2_minimum_samples')}`",
        f"- Persistence dates: `{config.get('alert_persistence_distinct_dates')}`",
        "",
        "## Lineage",
        "",
        f"- Ledger: `{lineage.get('ledger_status', 'not_available')}`; generation `{lineage.get('ledger_generation')}`",
        f"- Primary view: `{lineage.get('primary_view', 'not_available')}`",
        f"- Prediction/input/actual IDs: `{len(lineage.get('prediction_ids') or [])}` / "
        f"`{len(lineage.get('model_input_snapshot_ids') or [])}` / "
        f"`{len(lineage.get('actual_revision_ids') or [])}`",
        f"- Restated predictions (diagnostic only): `{lineage.get('restated_prediction_count', 0)}`",
        "",
        "## Quality",
        "",
        f"- Status/verdict: `{quality.get('status', quality.get('batch_status', 'not_available'))}` / "
        f"`{quality.get('verdict', 'not_available')}`",
        f"- Issues: `{len(quality.get('issues') or [])}`",
        f"- Source checksum files: `{(quality.get('checksums') or {}).get('count', 0)}`",
        "",
        "## Windows",
        "",
    ]
    for window, payload in report.get("windows", {}).items():
        lines.extend(
            [
                f"### {window} days",
                "",
                f"- Status: `{payload.get('status')}`; samples: `{payload.get('sample_count', 0)}`",
            ]
        )
        if payload.get("status") != "available":
            continue
        prediction = payload.get("prediction_drift") or {}
        target = payload.get("target_drift") or {}
        feature_severities = [
            comparison.get("severity")
            for entity in (payload.get("feature_drift") or {}).values()
            for comparison in entity.values()
        ]
        lines.extend(
            [
                f"- Feature drift comparisons: `{len(feature_severities)}`; warning/critical: "
                f"`{sum(value == 'warning' for value in feature_severities)}` / "
                f"`{sum(value == 'critical' for value in feature_severities)}`",
                "- Prediction drift (global/seasonal): "
                f"`{(prediction.get('global') or {}).get('severity', 'not_available')}` / "
                f"`{(prediction.get('seasonal') or {}).get('severity', 'not_available')}`",
                "- Target drift (global/seasonal): "
                f"`{(target.get('global') or {}).get('severity', target.get('status', 'not_available'))}` / "
                f"`{(target.get('seasonal') or {}).get('severity', target.get('status', 'not_available'))}`",
            ]
        )
        performance = payload.get("performance") or {}
        lines.append(
            f"- Performance: `{performance.get('status', 'not_available')}`; metrics: "
            f"`{json.dumps(performance.get('metrics') or {}, sort_keys=True)}`"
        )
    lines.extend(["", "## Breaches", ""])
    breaches = report.get("breaches") or []
    if breaches:
        lines.extend(
            f"- `{item['severity']}` - `{item['rule_id']}`" for item in breaches
        )
    else:
        lines.append("- None.")
    lines.extend(["", "## Persistence", ""])
    persistence = report.get("persistence") or {}
    if persistence:
        for rule_id, state in sorted(persistence.items()):
            lines.append(
                f"- `{rule_id}`: `{state.get('consecutive', 0)}/{state.get('required')}`; "
                f"active `{state.get('active')}`; last date `{state.get('last_date')}`"
            )
    else:
        lines.append("- No tracked rules.")
    lines.extend(
        [
            "",
            "## Alert events",
            "",
            f"- Immutable events emitted by this run: `{len(report.get('alert_events') or [])}`",
            "",
            "This is retrospective historical-hindcast monitoring, not an ex-ante forecast.",
            "",
        ]
    )
    return "\n".join(lines)


def _with_id(kind: str, id_field: str, body: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_ready(body)
    return {id_field: _record_id(kind, payload), **payload}


def _record_id(kind: str, body: Mapping[str, Any]) -> str:
    return sha256(kind.encode("utf-8") + b":" + _canonical(body)).hexdigest()


def _verify_record_id(
    payload: Mapping[str, Any],
    kind: str,
    id_field: str,
    *,
    ignored: Sequence[str] = (),
) -> None:
    body = {key: value for key, value in payload.items() if key != id_field and key not in ignored and not key.startswith("_")}
    if payload.get(id_field) != _record_id(kind, body):
        raise MonitoringReportingError(f"Corrupt {kind} content-addressed identity.")


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_ready(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    _immutable_bytes(path, _json_bytes(payload))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _immutable_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise MonitoringReportingError(f"Immutable path already contains different bytes: {path}.")
        return
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != data:
            raise MonitoringReportingError(f"Immutable path collision at {path}.")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MonitoringReportingError(f"Invalid JSON artifact: {path}.") from exc
    except OSError as exc:
        raise MonitoringReportingUnavailableError(
            f"JSON artifact is unavailable: {path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise MonitoringReportingError(f"Invalid JSON artifact: {path}.") from exc
    if not isinstance(payload, dict):
        raise MonitoringReportingError(f"JSON artifact must contain an object: {path}.")
    return payload


def _new_run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:12]}"


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _verify_reporting_era(config: MonitoringReportConfig) -> dict[str, Any]:
    try:
        return verify_active_model_era(
            config.deployment_root,
            config.model_bundle,
            calibration_dir=config.calibration_dir,
        )
    except DeploymentRuntimeError as exc:
        raise MonitoringReportingError(str(exc)) from exc


def _require_ledger_era(
    ledger: Mapping[str, Any] | None,
    era: Mapping[str, Any],
) -> None:
    if ledger is None:
        return
    if (
        ledger.get("schema_version") == LEGACY_STATE_SCHEMA
        and ledger.get("model_snapshot_id")
        != (era.get("monitoring") or {}).get("ledger_model_snapshot_id")
    ):
        raise MonitoringReportingError(
            "Legacy monitoring ledger differs from active deployment era."
        )
    if (
        ledger.get("schema_version") not in {None, LEGACY_STATE_SCHEMA}
        and ledger.get("active_model_era_id") != era.get("model_era_id")
    ):
        raise MonitoringReportingError(
            "Monitoring ledger and active deployment eras differ."
        )


def _legacy_state_is_adoptable(
    monitoring_store_root: Path,
    era: Mapping[str, Any],
) -> bool:
    try:
        stored = load_model_era(
            monitoring_store_root,
            str(era["model_era_id"]),
        )
    except (MonitoringError, OSError):
        return False
    return bool(stored.get("_adopted_state"))


def _require_reporting_plan_era(
    plan: MonitoringReportPlan,
    era: Mapping[str, Any],
) -> None:
    if (
        plan.model_era_id != era.get("model_era_id")
        or plan.deployment_id
        != (era.get("deployment") or {}).get("deployment_id")
        or plan.model_version != (era.get("registry") or {}).get("model_version")
    ):
        raise MonitoringReportingError(
            "Active deployment changed after report planning."
        )


def _acquire_lock(root: Path, run_id: str) -> Path:
    path = root / "state" / "report.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise MonitoringReportingError("Another monitoring report execution owns the lock.") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id}, handle, sort_keys=True)
    return path


def _release_lock(path: Path, run_id: str) -> None:
    if not path.is_file():
        return
    try:
        payload = _read_json(path)
    except MonitoringReportingError:
        return
    if payload.get("run_id") == run_id:
        path.unlink()


__all__ = [
    "CalibrationConfig",
    "CalibrationResult",
    "MonitoringReportConfig",
    "MonitoringReportPlan",
    "MonitoringReportResult",
    "MonitoringReportingConflictError",
    "MonitoringReportingError",
    "MonitoringReportingUnavailableError",
    "calibrate_monitoring_reference",
    "load_active_alerts",
    "load_alert_history",
    "load_monitoring_calibration",
    "load_monitoring_report",
    "load_reporting_attempt",
    "load_reporting_attempts",
    "resolve_report_model_era",
    "load_monitoring_report_state",
    "plan_monitoring_report",
    "run_monitoring_report",
]
