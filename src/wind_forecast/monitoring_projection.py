"""Read-only API projections over verified Phase 9 monitoring evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from wind_forecast.config import (
    MONITORING_STORE_ROOT_ENV,
    load_monitoring_store_config,
)
from wind_forecast.monitoring import (
    MonitoringError,
    load_prediction_evidence,
)
from wind_forecast.monitoring_reporting import (
    MonitoringReportingError,
    load_alert_history,
    load_monitoring_calibration,
    load_monitoring_report,
    load_monitoring_report_state,
)


MONITORING_MODE = "retrospective_historical_batch_not_real_time"
LISBON = ZoneInfo("Europe/Lisbon")
SEVERITY_ORDER = {"not_available": -1, "ok": 0, "warning": 1, "critical": 2}


class MonitoringProjectionError(RuntimeError):
    """Raised when stored monitoring evidence cannot be projected safely."""


class MonitoringRunNotFoundError(LookupError):
    """Raised when a requested reporting run does not exist."""


@dataclass(frozen=True)
class MonitoringProjectionService:
    """Build sanitized dashboard views from immutable local monitoring evidence."""

    store_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_root", Path(self.store_root))

    @classmethod
    def from_config(cls) -> "MonitoringProjectionService":
        return cls(load_monitoring_store_config().store_root)

    def latest(self, *, now_utc: datetime | None = None) -> dict[str, Any]:
        """Return the latest verified report and latest reporting attempt."""
        now = _aware_utc(now_utc)
        try:
            runs = self._runs()
            state = load_monitoring_report_state(self.store_root)
            if state is None:
                if any(run["status"] == "succeeded" for run in runs):
                    raise MonitoringReportingError(
                        "Succeeded reporting runs require a current report pointer."
                    )
                return {
                    "state": "empty",
                    "mode": MONITORING_MODE,
                    "served_at_utc": _utc_text(now),
                    "message": (
                        "No historical monitoring reports or runs are available."
                        if not runs
                        else "Reporting attempts exist, but no successful report is available."
                    ),
                    "latest_attempt": runs[0] if runs else None,
                    "report": None,
                }
            report_id = _required_text(state, "latest_report_id")
            report = load_monitoring_report(
                self.store_root
                / "reporting"
                / "reports"
                / report_id
                / "report.json"
            )
            run_id = _required_text(report, "run_id")
            run = next((item for item in runs if item["run_id"] == run_id), None)
            if (
                run is None
                or run["status"] != "succeeded"
                or run.get("report_id") != report_id
                or report.get("through_date") != state.get("latest_through_date")
            ):
                raise MonitoringReportingError(
                    "Current report pointer, run, and report identities differ."
                )
            return {
                "state": "available",
                "mode": MONITORING_MODE,
                "served_at_utc": _utc_text(now),
                "message": None,
                "latest_attempt": runs[0] if runs else None,
                "report": self._project_report(report, now),
            }
        except (
            MonitoringError,
            MonitoringReportingError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as exc:
            raise MonitoringProjectionError(
                "Stored monitoring evidence is unavailable or corrupt."
            ) from exc

    def history(
        self,
        *,
        run_limit: int = 20,
        run_offset: int = 0,
        alert_limit: int = 50,
        alert_offset: int = 0,
    ) -> dict[str, Any]:
        """Return paginated reporting attempts and causally ordered alerts."""
        try:
            runs = self._runs()
            alerts = [self._sanitize_alert(item) for item in load_alert_history(self.store_root)]
            return {
                "state": "available" if runs or alerts else "empty",
                "mode": MONITORING_MODE,
                "runs": {
                    "items": runs[run_offset : run_offset + run_limit],
                    "total": len(runs),
                    "limit": run_limit,
                    "offset": run_offset,
                },
                "alerts": {
                    "items": alerts[alert_offset : alert_offset + alert_limit],
                    "total": len(alerts),
                    "limit": alert_limit,
                    "offset": alert_offset,
                },
            }
        except (
            MonitoringError,
            MonitoringReportingError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as exc:
            raise MonitoringProjectionError(
                "Stored monitoring evidence is unavailable or corrupt."
            ) from exc

    def run(self, run_id: str, *, now_utc: datetime | None = None) -> dict[str, Any]:
        """Return one sanitized reporting attempt and its report when successful."""
        if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
            raise MonitoringRunNotFoundError(run_id)
        try:
            run = next((item for item in self._runs() if item["run_id"] == run_id), None)
            if run is None:
                raise MonitoringRunNotFoundError(run_id)
            report = None
            if run["status"] == "succeeded" and run.get("report_id"):
                path = (
                    self.store_root
                    / "reporting"
                    / "reports"
                    / str(run["report_id"])
                    / "report.json"
                )
                loaded = load_monitoring_report(path)
                if loaded.get("run_id") != run_id:
                    raise MonitoringReportingError(
                        "Reporting result and report run identities differ."
                    )
                report = self._project_report(loaded, _aware_utc(now_utc))
            return {
                "state": "available",
                "mode": MONITORING_MODE,
                "run": run,
                "report": report,
            }
        except MonitoringRunNotFoundError:
            raise
        except (
            MonitoringError,
            MonitoringReportingError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as exc:
            raise MonitoringProjectionError(
                "Stored monitoring evidence is unavailable or corrupt."
            ) from exc

    def _runs(self) -> list[dict[str, Any]]:
        self._validate_store_root()
        root = self.store_root / "reporting" / "runs"
        if not root.is_dir():
            return []
        runs = [self._load_run(path) for path in sorted(root.iterdir()) if path.is_dir()]
        return sorted(
            runs,
            key=lambda item: (
                str(item.get("attempted_at_utc") or ""),
                str(item.get("run_id") or ""),
            ),
            reverse=True,
        )

    def _validate_store_root(self) -> None:
        if self.store_root.exists() and not self.store_root.is_dir():
            raise MonitoringReportingError(
                "The configured monitoring store root is not a directory."
            )

    def _load_run(self, run_dir: Path) -> dict[str, Any]:
        request = _read_object(run_dir / "request.json")
        run_id = run_dir.name
        plan = request.get("plan")
        if (
            request.get("schema_version")
            != "wind_forecast.monitoring_report_request.v1"
            or request.get("run_id") != run_id
            or not isinstance(plan, dict)
            or plan.get("status") != "planned"
        ):
            raise MonitoringReportingError("Invalid monitoring reporting request.")
        attempted_at = _required_utc_text(request, "requested_at_utc")
        through_date = _required_date_text(plan, "through_date")
        source_run_id = _required_text(plan, "source_run_id")
        source_status = _required_text(plan, "source_status")
        calibration_id = _required_text(plan, "calibration_id")
        result_path = run_dir / "result.json"
        failure_path = run_dir / "failure.json"
        if result_path.is_file() and failure_path.is_file():
            raise MonitoringReportingError("Reporting run has conflicting outcomes.")
        base = {
            "run_id": run_id,
            "attempted_at_utc": attempted_at,
            "through_date": through_date,
            "source_pipeline_run_id": source_run_id,
            "source_pipeline_status": source_status,
        }
        if result_path.is_file():
            result = _read_object(result_path)
            report_id = str(result.get("report_id") or "")
            active_alert_count = result.get("active_alert_count")
            if (
                result.get("run_id") != run_id
                or result.get("status") != "succeeded"
                or not report_id
                or result.get("plan") != plan
                or not isinstance(active_alert_count, int)
                or active_alert_count < 0
            ):
                raise MonitoringReportingError("Invalid monitoring reporting result.")
            report = load_monitoring_report(
                self.store_root
                / "reporting"
                / "reports"
                / report_id
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
                raise MonitoringReportingError(
                    "Reporting request, result, and report lineage differ."
                )
            return {
                **base,
                "status": "succeeded",
                "report_id": report_id,
                "active_alert_count": active_alert_count,
                "failure": None,
            }
        if failure_path.is_file():
            failure = _read_object(failure_path)
            if (
                failure.get("schema_version")
                != "wind_forecast.monitoring_report_failure.v1"
                or failure.get("run_id") != run_id
            ):
                raise MonitoringReportingError("Invalid monitoring reporting failure.")
            failed_at = _required_utc_text(failure, "failed_at_utc")
            error_type = _required_text(failure, "error_type")
            return {
                **base,
                "status": "failed",
                "report_id": None,
                "active_alert_count": 0,
                "failure": {
                    "failed_at_utc": failed_at,
                    "error_type": error_type,
                    "message": "The reporting attempt failed. Inspect local operator logs.",
                },
            }
        return {
            **base,
            "status": "in_progress",
            "report_id": None,
            "active_alert_count": 0,
            "failure": None,
        }

    def _project_report(
        self, report: Mapping[str, Any], now: datetime
    ) -> dict[str, Any]:
        report_reference = report.get("reference")
        if not isinstance(report_reference, Mapping):
            raise MonitoringReportingError(
                "Monitoring report reference metadata is invalid."
            )
        calibration = load_monitoring_calibration(
            self.store_root
            / "reporting"
            / "calibrations"
            / str(report_reference.get("calibration_id") or "")
        )
        reference = calibration.get("_reference_manifest")
        if (
            not isinstance(reference, Mapping)
            or report_reference.get("calibration_id") != calibration.get("calibration_id")
            or report_reference.get("reference_id") != calibration.get("reference_id")
            or report_reference.get("reference_id") != reference.get("reference_id")
            or report_reference.get("policy_sha256") != calibration.get("policy_sha256")
        ):
            raise MonitoringReportingError(
                "Report and calibration identities differ."
            )
        active_ids = set((report.get("active_alerts") or {}).values())
        alert_by_id = {
            str(item["alert_event_id"]): item
            for item in load_alert_history(self.store_root)
        }
        if not active_ids.issubset(alert_by_id):
            raise MonitoringReportingError(
                "Report active-alert references are incomplete."
            )
        active_alerts = [
            self._sanitize_alert(alert_by_id[alert_id])
            for alert_id in sorted(active_ids)
        ]
        return {
            "report_id": report.get("report_id"),
            "reporting_run_id": report.get("run_id"),
            "created_at_utc": report.get("created_at_utc"),
            "as_of_date": report.get("through_date"),
            "source_pipeline": {
                "run_id": (report.get("source_batch") or {}).get("run_id"),
                "status": (report.get("source_batch") or {}).get("status"),
            },
            "freshness": self._freshness(report, now),
            "model": self._model(report, calibration),
            "windows": {
                window: self._project_window(
                    window,
                    (report.get("windows") or {}).get(window) or {},
                    calibration,
                )
                for window in ("30", "90")
            },
            "active_alerts": active_alerts,
            "target_scale": "sum_of_15_minute_MW_observations",
        }

    def _model(
        self,
        report: Mapping[str, Any],
        calibration: Mapping[str, Any],
    ) -> dict[str, Any]:
        reference = calibration.get("_reference_manifest")
        if not isinstance(reference, Mapping):
            raise MonitoringReportingError(
                "Monitoring calibration reference metadata is unavailable."
            )
        reference_checksum = _required_text(reference, "model_sha256")
        projected = {
            "snapshot_id": None,
            "checksum": reference_checksum,
            "model_type": None,
            "dataset_version": None,
            "dataset_checksum": _required_text(reference, "dataset_sha256"),
            "transformation_version": _required_text(
                reference, "transformation_version"
            ),
            "status": "selected_not_promoted",
        }
        prediction_ids = list((report.get("lineage") or {}).get("prediction_ids") or [])
        snapshots: dict[str, Mapping[str, Any]] = {}
        for prediction_id in prediction_ids:
            evidence = load_prediction_evidence(self.store_root, str(prediction_id))
            snapshot = evidence["model_snapshot"]
            snapshots[str(snapshot.get("model_snapshot_id") or "")] = snapshot
        if not snapshots:
            return projected
        if len(snapshots) != 1 or "" in snapshots:
            raise MonitoringReportingError(
                "Report lineage references multiple or invalid model snapshots."
            )
        snapshot = next(iter(snapshots.values()))
        model = snapshot.get("model") or {}
        dataset = snapshot.get("dataset") or {}
        transformation = snapshot.get("transformation") or {}
        if (
            model.get("model_sha256") != reference_checksum
            or dataset.get("dataset_sha256") != projected["dataset_checksum"]
            or transformation.get("version") != projected["transformation_version"]
            or model.get("reference_status") != "selected_not_promoted"
        ):
            raise MonitoringReportingError(
                "Report lineage and calibration model identities differ."
            )
        return {
            **projected,
            "snapshot_id": snapshot.get("model_snapshot_id"),
            "model_type": model.get("model_type"),
            "dataset_version": dataset.get("dataset_version"),
            "dataset_checksum": dataset.get("dataset_sha256"),
            "transformation_version": transformation.get("version"),
        }

    def _freshness(
        self, report: Mapping[str, Any], now_utc: datetime
    ) -> dict[str, Any]:
        config = report.get("config") or {}
        freshness = (report.get("quality") or {}).get("freshness") or {}
        watermark_text = freshness.get("common_validated_watermark")
        objective_value = config.get("source_objective_days")
        late_value = config.get("source_late_days")
        if watermark_text is None or objective_value is None or late_value is None:
            return {
                "status": "unknown",
                "watermark_date": (
                    str(watermark_text) if watermark_text is not None else None
                ),
                "objective_at": None,
                "late_at": None,
                "timezone": "Europe/Lisbon",
                "objective_days": (
                    int(objective_value) if objective_value is not None else 5
                ),
                "late_days": int(late_value) if late_value is not None else 7,
            }
        objective_days = int(objective_value)
        late_days = int(late_value)
        watermark = date.fromisoformat(str(watermark_text))
        objective_at = datetime.combine(
            watermark + timedelta(days=objective_days),
            time(12),
            tzinfo=LISBON,
        )
        late_at = datetime.combine(
            watermark + timedelta(days=late_days),
            time(12),
            tzinfo=LISBON,
        )
        source_late = any(
            str(item.get("code")) == "source_late"
            for item in ((report.get("quality") or {}).get("issues") or [])
        ) or bool(freshness.get("unresolved_late_dates"))
        local_now = now_utc.astimezone(LISBON)
        status = (
            "late"
            if source_late or local_now >= late_at
            else "behind_objective"
            if local_now >= objective_at
            else "within_objective"
        )
        return {
            "status": status,
            "watermark_date": watermark.isoformat(),
            "objective_at": objective_at.isoformat(),
            "late_at": late_at.isoformat(),
            "timezone": "Europe/Lisbon",
            "objective_days": objective_days,
            "late_days": late_days,
        }

    def _project_window(
        self,
        window: str,
        payload: Mapping[str, Any],
        calibration: Mapping[str, Any],
    ) -> dict[str, Any]:
        status = str(payload.get("status") or "not_available")
        result: dict[str, Any] = {
            "window_days": int(window),
            "status": status,
            "sample_count": int(payload.get("sample_count") or 0),
            "minimum_samples": payload.get("minimum_samples"),
            "calendar_start": payload.get("calendar_start"),
            "calendar_end": payload.get("calendar_end"),
            "coverage_ratio": payload.get("coverage_ratio"),
            "coverage_severity": payload.get("coverage_severity"),
            "performance": [],
            "top_drift": [],
        }
        if status != "available":
            return result
        performance = payload.get("performance") or {}
        if performance.get("status") == "available":
            thresholds = calibration["thresholds"]["performance"][window]
            metrics = performance.get("metrics") or {}
            severities = performance.get("severity") or {}
            for key, label in (
                ("MAE", "MAE"),
                ("RMSE", "RMSE"),
                ("bias", "Bias"),
                ("MAPE_percent", "MAPE"),
                ("R2", "R²"),
            ):
                limit_key = "absolute_bias" if key == "bias" else key
                limits = thresholds.get(limit_key)
                if limits is None:
                    continue
                result["performance"].append(
                    {
                        "metric": key,
                        "label": label,
                        "value": metrics.get(key),
                        "status": (
                            metrics.get("R2_status")
                            if key == "R2" and metrics.get(key) is None
                            else "available"
                        ),
                        "severity": severities.get(key, "not_available"),
                        "warning": limits.get("warning"),
                        "critical": limits.get("critical"),
                        "direction": limits.get("direction", "upper"),
                    }
                )
        result["top_drift"] = self._top_drift(
            payload.get("feature_drift") or {},
            calibration["thresholds"]["feature_drift"],
            window,
        )
        return result

    @staticmethod
    def _top_drift(
        feature_drift: Mapping[str, Any],
        thresholds: Mapping[str, Any],
        window: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for feature, comparisons in feature_drift.items():
            candidates = []
            for comparator, stats in (comparisons or {}).items():
                limits_by_detector = (
                    thresholds.get(feature, {}).get(window, {}).get(comparator, {})
                )
                for detector in ("normalized_wasserstein", "ks_statistic"):
                    value = stats.get(detector)
                    limits = limits_by_detector.get(detector)
                    if not isinstance(value, (int, float)) or not limits:
                        continue
                    severity = _threshold_severity(float(value), limits)
                    threshold = (
                        float(limits["critical"])
                        if severity == "critical"
                        else float(limits["warning"])
                    )
                    ratio = float(value) / threshold if threshold > 0 else 0.0
                    candidates.append(
                        {
                            "feature": feature,
                            "comparator": comparator,
                            "detector": detector,
                            "value": float(value),
                            "severity": severity,
                            "threshold": threshold,
                            "threshold_ratio": ratio,
                        }
                    )
            if candidates:
                entries.append(
                    max(
                        candidates,
                        key=lambda item: (
                            SEVERITY_ORDER.get(str(item["severity"]), -1),
                            float(item["threshold_ratio"]),
                            str(item["comparator"]),
                            str(item["detector"]),
                        ),
                    )
                )
        return sorted(
            entries,
            key=lambda item: (
                -SEVERITY_ORDER.get(str(item["severity"]), -1),
                -float(item["threshold_ratio"]),
                str(item["feature"]),
            ),
        )[:5]

    @staticmethod
    def _sanitize_alert(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "alert_event_id": item.get("alert_event_id"),
            "rule_id": item.get("rule_id"),
            "through_date": item.get("through_date"),
            "event_type": item.get("event_type"),
            "severity": item.get("severity"),
            "previous_alert_event_id": item.get("previous_alert_event_id"),
        }


def _threshold_severity(value: float, limits: Mapping[str, Any]) -> str:
    warning = float(limits["warning"])
    critical = float(limits["critical"])
    direction = str(limits.get("direction") or "upper")
    if direction == "upper":
        return "critical" if value >= critical else "warning" if value >= warning else "ok"
    if direction == "lower":
        return "critical" if value <= critical else "warning" if value <= warning else "ok"
    raise MonitoringReportingError("Monitoring threshold direction is invalid.")


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise MonitoringReportingError(f"Monitoring metadata field is invalid: {field}.")
    return value


def _required_date_text(payload: Mapping[str, Any], field: str) -> str:
    value = _required_text(payload, field)
    date.fromisoformat(value)
    return value


def _required_utc_text(payload: Mapping[str, Any], field: str) -> str:
    value = _required_text(payload, field)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise MonitoringReportingError(
            f"Monitoring timestamp is not timezone-aware: {field}."
        )
    return value


def _aware_utc(value: datetime | None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware.")
    return result.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitoringReportingError("Invalid monitoring run metadata.") from exc
    if not isinstance(value, dict):
        raise MonitoringReportingError("Invalid monitoring run metadata.")
    return value


__all__ = [
    "MONITORING_MODE",
    "MONITORING_STORE_ROOT_ENV",
    "MonitoringProjectionError",
    "MonitoringProjectionService",
    "MonitoringRunNotFoundError",
]
