"""Airflow-free task boundaries for the Phase 10 historical batch DAG."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from wind_forecast.manifests import sha256_file
from wind_forecast.paths import project_root


LISBON = ZoneInfo("Europe/Lisbon")
STAGE_RESULT_SCHEMA = "wind_forecast.airflow_stage_result.v1"


class AirflowStageError(RuntimeError):
    """Raised when an Airflow-owned CLI boundary fails or emits invalid output."""


@dataclass(frozen=True)
class AirflowBatchConfig:
    """Explicit immutable selections shared by every DAG task."""

    model_bundle: Path
    calibration_dir: Path
    deployment_root: Path
    source_store_root: Path
    monitoring_store_root: Path
    activation_date: str | date
    raw_store_root: Path | None = None
    ren_root: Path | None = None
    era5_root: Path | None = None
    station_mapping: Path | None = None
    v1_feature_table: Path | None = None
    baseline_integrated_root: Path | None = None
    baseline_feature_root: Path | None = None
    bootstrap_start: str | date | None = None
    bootstrap_end: str | date | None = None
    no_source_refresh: bool = False
    fail_on_active_alert: bool = False

    def __post_init__(self) -> None:
        root = project_root()
        for name in (
            "model_bundle",
            "calibration_dir",
            "source_store_root",
            "monitoring_store_root",
            "deployment_root",
        ):
            value = Path(getattr(self, name))
            if not value.is_absolute():
                value = root / value
            object.__setattr__(self, name, value.resolve())
        optional_paths = (
            "raw_store_root",
            "ren_root",
            "era5_root",
            "station_mapping",
            "v1_feature_table",
            "baseline_integrated_root",
            "baseline_feature_root",
        )
        for name in optional_paths:
            value = getattr(self, name)
            if value is not None:
                path = Path(value)
                if not path.is_absolute():
                    path = root / path
                object.__setattr__(self, name, path.resolve())
        activation = self.activation_date
        if isinstance(activation, str):
            activation = date.fromisoformat(activation)
        object.__setattr__(self, "activation_date", activation)


def through_date_from_interval_end(value: datetime) -> str:
    """Return the Lisbon calendar date represented by a data interval end."""

    if value.tzinfo is None:
        raise ValueError("data_interval_end must be timezone-aware.")
    return value.astimezone(LISBON).date().isoformat()


def validate_three_day_backfill(start: str | date, end: str | date) -> tuple[str, str]:
    """Validate the deliberately narrow offline backfill window."""

    first = date.fromisoformat(start) if isinstance(start, str) else start
    last = date.fromisoformat(end) if isinstance(end, str) else end
    if (last - first).days != 2:
        raise ValueError("The Phase 10 validation backfill must span exactly 3 days.")
    return first.isoformat(), last.isoformat()


def run_deployment_preflight(
    config: AirflowBatchConfig,
    *,
    expected_model_era_id: str | None = None,
    timeout_seconds: int = 10 * 60,
) -> dict[str, Any]:
    """Verify pointer, aliases and artifacts before or after an Airflow run."""
    command = [
        sys.executable,
        str(project_root() / "scripts" / "verify_active_deployment.py"),
        "--deployment-root",
        str(config.deployment_root),
        "--model-bundle",
        str(config.model_bundle),
        "--calibration-dir",
        str(config.calibration_dir),
    ]
    payload = _run_cli_json(command, timeout_seconds)
    if (
        expected_model_era_id is not None
        and payload.get("model_era_id") != expected_model_era_id
    ):
        raise AirflowStageError("Active deployment changed during Airflow batch.")
    return _small_result(
        "deployment_preflight",
        payload,
        model_era_id=payload.get("model_era_id"),
        deployment_id=(payload.get("deployment") or {}).get("deployment_id"),
        model_version=(payload.get("registry") or {}).get("model_version"),
    )


def run_availability_plan(
    config: AirflowBatchConfig,
    through_date: str,
    *,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Run the strictly read-only source availability boundary."""

    command = _update_command(config, through_date, dry_run=True)
    return _small_result(
        "availability_plan",
        _run_cli_json(command, timeout_seconds),
    )


def run_dataset_update(
    config: AirflowBatchConfig,
    through_date: str,
    *,
    timeout_seconds: int = 4 * 60 * 60,
) -> dict[str, Any]:
    """Run the Phase 8 transaction and return checksum-pinned evidence."""

    payload = _run_cli_json(
        _update_command(config, through_date, dry_run=False),
        timeout_seconds,
    )
    manifest = _required_file(payload, "manifest_path")
    return _small_result(
        "dataset_update",
        payload,
        manifest_path=str(manifest),
        manifest_sha256=sha256_file(manifest),
    )


def run_predict_reconcile(
    config: AirflowBatchConfig,
    through_date: str,
    *,
    source_manifest_path: str,
    source_manifest_sha256: str,
    timeout_seconds: int = 60 * 60,
) -> dict[str, Any]:
    """Verify the upstream transaction, then issue/reconcile hindcasts."""

    _verify_file(source_manifest_path, source_manifest_sha256)
    command = [
        sys.executable,
        str(project_root() / "scripts" / "run_historical_monitoring.py"),
        "--through-date",
        through_date,
        "--source-store-root",
        str(config.source_store_root),
        "--monitoring-store-root",
        str(config.monitoring_store_root),
        "--model-bundle",
        str(config.model_bundle),
        "--deployment-root",
        str(config.deployment_root),
        "--activation-date",
        config.activation_date.isoformat(),
    ]
    payload = _run_cli_json(command, timeout_seconds)
    return _small_result(
        "predict_reconcile",
        payload,
        run_id=payload.get("run_id"),
        current_state_path=payload.get("current_state_path"),
    )


def run_drift_publish(
    config: AirflowBatchConfig,
    through_date: str,
    *,
    source_manifest_path: str,
    source_manifest_sha256: str,
    timeout_seconds: int = 30 * 60,
) -> dict[str, Any]:
    """Verify upstream evidence and publish the immutable monitoring report."""

    _verify_file(source_manifest_path, source_manifest_sha256)
    command = [
        sys.executable,
        str(project_root() / "scripts" / "run_monitoring_report.py"),
        "--source-run-manifest",
        source_manifest_path,
        "--monitoring-store-root",
        str(config.monitoring_store_root),
        "--calibration-dir",
        str(config.calibration_dir),
        "--model-bundle",
        str(config.model_bundle),
        "--deployment-root",
        str(config.deployment_root),
        "--through-date",
        through_date,
    ]
    if config.fail_on_active_alert:
        command.append("--fail-on-active-alert")
    payload = _run_cli_json(command, timeout_seconds)
    alerts = int(payload.get("active_alert_count") or 0)
    report_path = _required_file(payload, "report_path")
    return _small_result(
        "drift_publish",
        payload,
        status="completed_with_alerts" if alerts else payload.get("status"),
        report_id=payload.get("report_id"),
        report_path=str(report_path),
        report_sha256=sha256_file(report_path),
        active_alert_count=alerts,
    )


def _update_command(
    config: AirflowBatchConfig,
    through_date: str,
    *,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(project_root() / "scripts" / "update_v2_dataset.py"),
        "--through-date",
        through_date,
        "--store-root",
        str(config.source_store_root),
    ]
    if dry_run:
        command.append("--dry-run")
    if config.no_source_refresh:
        command.append("--no-source-refresh")
    if config.raw_store_root is not None:
        command.extend(["--raw-store-root", str(config.raw_store_root)])
    for option, value in (
        ("--ren-root", config.ren_root),
        ("--era5-root", config.era5_root),
        ("--station-mapping", config.station_mapping),
        ("--v1-feature-table", config.v1_feature_table),
        ("--baseline-integrated-root", config.baseline_integrated_root),
        ("--baseline-feature-root", config.baseline_feature_root),
    ):
        if value is not None:
            command.extend([option, str(value)])
    if config.bootstrap_start and config.bootstrap_end:
        command.extend(
            ["--bootstrap-start", str(config.bootstrap_start), "--bootstrap-end", str(config.bootstrap_end)]
        )
    return command


def _run_cli_json(command: Sequence[str], timeout_seconds: int) -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=project_root(),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AirflowStageError(
            f"Stage exceeded its {timeout_seconds}-second timeout."
        ) from exc
    if completed.returncode:
        raise AirflowStageError(
            f"Stage exited with code {completed.returncode}; inspect Airflow task logs."
        )
    objects: list[tuple[int, dict[str, Any]]] = []
    decoder = json.JSONDecoder()
    text = completed.stdout
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append((offset + consumed, value))
        offset += consumed
    if not objects:
        raise AirflowStageError("Stage did not emit a JSON object.")
    preferred = [
        item
        for item in objects
        if any(
            key in item[1]
            for key in ("manifest_path", "report_path", "current_state_path", "active_alert_count")
        )
    ]
    payload = max(preferred or objects, key=lambda item: item[0])[1]
    return payload


def _required_file(payload: Mapping[str, Any], key: str) -> Path:
    value = str(payload.get(key) or "")
    path = Path(value)
    if not value or not path.is_file():
        raise AirflowStageError(f"Stage did not produce a verified {key}.")
    return path.resolve()


def _verify_file(path_value: str, expected_sha256: str) -> Path:
    path = Path(path_value)
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise AirflowStageError("Upstream manifest path or checksum is invalid.")
    return path.resolve()


def _small_result(
    stage: str,
    payload: Mapping[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    result = {
        "schema_version": STAGE_RESULT_SCHEMA,
        "stage": stage,
        "status": str(payload.get("status") or "succeeded"),
    }
    result.update({key: value for key, value in overrides.items() if value is not None})
    return result
