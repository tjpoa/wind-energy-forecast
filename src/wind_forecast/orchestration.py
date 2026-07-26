"""Local orchestration for the accepted historical batch workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

from wind_forecast.manifests import sha256_file
from wind_forecast.paths import project_root


BATCH_SCHEMA = "wind_forecast.batch_run.v1"
BATCH_STATE_SCHEMA = "wind_forecast.batch_state.v1"
LISBON = ZoneInfo("Europe/Lisbon")
_SECRET_PARTS = ("key", "secret", "password", "token", "credential")


class BatchOrchestrationError(RuntimeError):
    """Raised when the local batch cannot be completed or verified."""


class ConcurrentBatchError(BatchOrchestrationError):
    """Raised when another live coordinator owns the batch lock."""


@dataclass(frozen=True)
class BatchConfig:
    """Explicit inputs for one local historical batch."""

    model_bundle: Path
    calibration_dir: Path
    through_date: str | date | None = None
    activation_date: str | date | None = None
    backfill_start: str | date | None = None
    backfill_end: str | date | None = None
    source_store_root: Path = Path("data/processed/v2/incremental_update")
    monitoring_store_root: Path = Path("data/processed/v2/monitoring")
    orchestration_root: Path = Path("data/processed/v2/orchestration")
    no_source_refresh: bool = False
    fail_on_active_alert: bool = False
    now_utc: datetime | None = None
    stage_timeout_seconds: int = 6 * 60 * 60

    def __post_init__(self) -> None:
        root = project_root()
        for name in (
            "model_bundle",
            "calibration_dir",
            "source_store_root",
            "monitoring_store_root",
            "orchestration_root",
        ):
            value = Path(getattr(self, name))
            if not value.is_absolute():
                value = root / value
            object.__setattr__(self, name, value.resolve())
        through = self.through_date or _now(self.now_utc).astimezone(LISBON).date()
        object.__setattr__(self, "through_date", _parse_date(through, "through_date"))
        for name in ("activation_date", "backfill_start", "backfill_end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _parse_date(value, name))
        if (self.backfill_start is None) != (self.backfill_end is None):
            raise ValueError("backfill_start and backfill_end must be supplied together.")
        if self.backfill_start and self.backfill_start > self.backfill_end:
            raise ValueError("backfill_start must be on or before backfill_end.")
        if self.stage_timeout_seconds < 1:
            raise ValueError("stage_timeout_seconds must be positive.")
        object.__setattr__(self, "now_utc", _now(self.now_utc))


@dataclass(frozen=True)
class BatchStageResult:
    """Outcome of one stable CLI boundary."""

    name: str
    status: str
    started_at_utc: str
    finished_at_utc: str
    command: tuple[str, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class BatchPlan:
    """Read-only batch plan."""

    through_date: str
    stages: tuple[BatchStageResult, ...]
    status: str = "planned"

    def summary(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class BatchRunResult:
    """Final outcome of one coordinated batch attempt."""

    status: str
    run_id: str | None
    through_date: str
    stages: tuple[BatchStageResult, ...]
    manifest_path: Path | None = None
    active_alert_count: int = 0

    def summary(self) -> dict[str, Any]:
        payload = _json_ready(asdict(self))
        payload["manifest_path"] = (
            str(self.manifest_path) if self.manifest_path else None
        )
        return payload


Runner = Callable[[Sequence[str], int], Mapping[str, Any]]


def plan_batch(config: BatchConfig, *, runner: Runner | None = None) -> BatchPlan:
    """Plan source availability and monitoring without coordinator writes."""

    execute = runner or _run_json_command
    availability = _execute_stage(
        "availability_plan",
        _update_command(config, dry_run=True),
        config.stage_timeout_seconds,
        execute,
    )
    monitoring_command = _monitoring_command(config, dry_run=True)
    try:
        monitoring = _execute_stage(
            "monitoring_plan",
            monitoring_command,
            config.stage_timeout_seconds,
            execute,
        )
    except BatchOrchestrationError as exc:
        timestamp = _utc_text(datetime.now(timezone.utc))
        monitoring = BatchStageResult(
            name="monitoring_plan",
            status="deferred",
            started_at_utc=timestamp,
            finished_at_utc=timestamp,
            command=tuple(_safe_command(monitoring_command)),
            payload={
                "status": "deferred",
                "reason": _redact(str(exc)),
                "note": "Re-evaluated after the transactional dataset update.",
            },
        )
    return BatchPlan(
        through_date=config.through_date.isoformat(),
        stages=(availability, monitoring),
    )


def run_batch(config: BatchConfig, *, runner: Runner | None = None) -> BatchRunResult:
    """Run the stable batch CLIs sequentially and persist coordinator evidence."""

    execute = runner or _run_json_command
    run_id = _run_id(config.now_utc)
    root = config.orchestration_root
    lock = _acquire_lock(root, run_id, config.now_utc)
    run_root = root / "runs" / run_id
    manifest_path = run_root / "manifest.json"
    stages: list[BatchStageResult] = []
    run_root.mkdir(parents=True, exist_ok=False)
    request = _request_payload(config, run_id)
    _write_json(manifest_path, {**request, "status": "running", "stages": []})
    try:
        stages.append(
            _execute_stage(
                "availability_plan",
                _update_command(config, dry_run=True),
                config.stage_timeout_seconds,
                execute,
            )
        )
        update = _execute_stage(
            "dataset_update",
            _update_command(config, dry_run=False),
            config.stage_timeout_seconds,
            execute,
        )
        stages.append(update)
        source_manifest = str(update.payload.get("manifest_path") or "")
        if not source_manifest:
            raise BatchOrchestrationError(
                "Dataset update did not return a source run manifest."
            )
        stages.append(
            _execute_stage(
                "predict_reconcile",
                _monitoring_command(config, dry_run=False),
                config.stage_timeout_seconds,
                execute,
            )
        )
        report = _execute_stage(
            "drift_publish",
            _report_command(config, source_manifest),
            config.stage_timeout_seconds,
            execute,
        )
        stages.append(report)
        alerts = int(report.payload.get("active_alert_count") or 0)
        status = "completed_with_alerts" if alerts else "succeeded"
        manifest = {
            **request,
            "status": status,
            "finished_at_utc": _utc_text(datetime.now(timezone.utc)),
            "active_alert_count": alerts,
            "stages": [item.summary() for item in stages],
        }
        _write_json(manifest_path, manifest)
        _publish_pointer(root, manifest_path, run_id, status)
        return BatchRunResult(
            status=status,
            run_id=run_id,
            through_date=config.through_date.isoformat(),
            stages=tuple(stages),
            manifest_path=manifest_path,
            active_alert_count=alerts,
        )
    except Exception as exc:
        failure = {
            **request,
            "status": "failed",
            "finished_at_utc": _utc_text(datetime.now(timezone.utc)),
            "failed_stage": _next_stage(stages),
            "error": _redact(str(exc)),
            "stages": [item.summary() for item in stages],
        }
        _write_json(manifest_path, failure)
        _publish_pointer(root, manifest_path, run_id, "failed")
        raise
    finally:
        _release_lock(lock, run_id)


def load_verified_batch_run(
    orchestration_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the latest or an explicit batch manifest and verify its pointer."""

    root = Path(orchestration_root)
    if manifest_path is None:
        pointer_path = root / "state" / "current.json"
        pointer = _read_json(pointer_path)
        if pointer.get("schema_version") != BATCH_STATE_SCHEMA:
            raise BatchOrchestrationError("Unsupported batch state schema.")
        path = Path(str(pointer.get("manifest_path") or ""))
        expected = str(pointer.get("manifest_sha256") or "")
        if not path.is_file() or sha256_file(path) != expected:
            raise BatchOrchestrationError("Batch state references invalid evidence.")
    else:
        path = Path(manifest_path)
    payload = _read_json(path)
    if payload.get("schema_version") != BATCH_SCHEMA:
        raise BatchOrchestrationError("Unsupported batch manifest schema.")
    return payload


def _execute_stage(
    name: str,
    command: Sequence[str],
    timeout: int,
    runner: Runner,
) -> BatchStageResult:
    started = datetime.now(timezone.utc)
    payload = dict(runner(command, timeout))
    return BatchStageResult(
        name=name,
        status=str(payload.get("status") or "succeeded"),
        started_at_utc=_utc_text(started),
        finished_at_utc=_utc_text(datetime.now(timezone.utc)),
        command=tuple(_safe_command(command)),
        payload=payload,
    )


def _update_command(config: BatchConfig, *, dry_run: bool) -> list[str]:
    command = [
        sys.executable,
        str(project_root() / "scripts" / "update_v2_dataset.py"),
        "--through-date",
        config.through_date.isoformat(),
        "--store-root",
        str(config.source_store_root),
    ]
    if dry_run:
        command.append("--dry-run")
    if config.no_source_refresh:
        command.append("--no-source-refresh")
    return command


def _monitoring_command(config: BatchConfig, *, dry_run: bool) -> list[str]:
    command = [
        sys.executable,
        str(project_root() / "scripts" / "run_historical_monitoring.py"),
        "--through-date",
        config.through_date.isoformat(),
        "--source-store-root",
        str(config.source_store_root),
        "--monitoring-store-root",
        str(config.monitoring_store_root),
        "--model-bundle",
        str(config.model_bundle),
    ]
    if config.activation_date:
        command.extend(["--activation-date", config.activation_date.isoformat()])
    if config.backfill_start and config.backfill_end:
        command.extend(
            [
                "--backfill-start",
                config.backfill_start.isoformat(),
                "--backfill-end",
                config.backfill_end.isoformat(),
            ]
        )
    if dry_run:
        command.append("--dry-run")
    return command


def _report_command(config: BatchConfig, source_manifest: str) -> list[str]:
    command = [
        sys.executable,
        str(project_root() / "scripts" / "run_monitoring_report.py"),
        "--source-run-manifest",
        source_manifest,
        "--monitoring-store-root",
        str(config.monitoring_store_root),
        "--calibration-dir",
        str(config.calibration_dir),
        "--through-date",
        config.through_date.isoformat(),
    ]
    if config.fail_on_active_alert:
        command.append("--fail-on-active-alert")
    return command


def _run_json_command(command: Sequence[str], timeout: int) -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=project_root(),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BatchOrchestrationError(
            f"Stage exceeded its {timeout}-second timeout."
        ) from exc
    allowed_alert_exit = (
        completed.returncode == 2 and "--fail-on-active-alert" in command
    )
    if completed.returncode and not allowed_alert_exit:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BatchOrchestrationError(
            f"Stage exited with code {completed.returncode}: {_redact(detail)}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BatchOrchestrationError(
            "Stage did not emit one valid JSON document."
        ) from exc
    if not isinstance(payload, dict):
        raise BatchOrchestrationError("Stage JSON result must be an object.")
    return payload


def _request_payload(config: BatchConfig, run_id: str) -> dict[str, Any]:
    return {
        "schema_version": BATCH_SCHEMA,
        "run_id": run_id,
        "started_at_utc": _utc_text(config.now_utc),
        "through_date": config.through_date.isoformat(),
        "git_commit": _git_commit(),
        "configuration": {
            "model_bundle": str(config.model_bundle),
            "calibration_dir": str(config.calibration_dir),
            "source_store_root": str(config.source_store_root),
            "monitoring_store_root": str(config.monitoring_store_root),
            "activation_date": (
                config.activation_date.isoformat() if config.activation_date else None
            ),
            "backfill_start": (
                config.backfill_start.isoformat() if config.backfill_start else None
            ),
            "backfill_end": (
                config.backfill_end.isoformat() if config.backfill_end else None
            ),
            "no_source_refresh": config.no_source_refresh,
            "fail_on_active_alert": config.fail_on_active_alert,
        },
    }


def _acquire_lock(root: Path, run_id: str, now: datetime) -> Path:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    lock = state / "batch.lock"
    body = {
        "run_id": run_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "created_at_utc": _utc_text(now),
    }
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = _read_json(lock)
        if existing.get("host") != socket.gethostname():
            raise ConcurrentBatchError("Batch lock belongs to another host.")
        pid = int(existing.get("pid") or -1)
        if _pid_is_live(pid):
            raise ConcurrentBatchError(
                f"Batch lock belongs to live run {existing.get('run_id')}."
            )
        _write_json(
            root
            / "recoveries"
            / f"{str(existing.get('run_id') or 'unknown')}-abandoned.json",
            {
                "schema_version": "wind_forecast.batch_abandoned.v1",
                "recovered_at_utc": _utc_text(now),
                "recovered_by_run_id": run_id,
                "abandoned_lock": existing,
            },
        )
        lock.unlink()
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(body, handle, sort_keys=True)
    return lock


def _release_lock(lock: Path, run_id: str) -> None:
    if not lock.exists():
        return
    current = _read_json(lock)
    if current.get("run_id") == run_id:
        lock.unlink()


def _pid_is_live(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _publish_pointer(root: Path, manifest: Path, run_id: str, status: str) -> None:
    _write_json(
        root / "state" / "current.json",
        {
            "schema_version": BATCH_STATE_SCHEMA,
            "run_id": run_id,
            "status": status,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
        },
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        _json_ready(payload),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchOrchestrationError(f"Could not read verified JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BatchOrchestrationError(f"Expected a JSON object: {path}")
    return payload


def _safe_command(command: Sequence[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for value in command:
        lowered = value.lower()
        if redact_next:
            result.append("<redacted>")
            redact_next = False
        elif any(part in lowered for part in _SECRET_PARTS):
            result.append(value)
            redact_next = True
        else:
            result.append(value)
    return result


def _redact(value: str) -> str:
    lines = []
    for line in value.splitlines()[-20:]:
        lowered = line.lower()
        lines.append("<redacted>" if any(part in lowered for part in _SECRET_PARTS) else line)
    return "\n".join(lines)


def _next_stage(stages: Sequence[BatchStageResult]) -> str:
    names = ("availability_plan", "dataset_update", "predict_reconcile", "drift_publish")
    return names[min(len(stages), len(names) - 1)]


def _now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware.")
    return current.astimezone(timezone.utc)


def _parse_date(value: str | date, label: str) -> date:
    try:
        return value if isinstance(value, date) else date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD.") from exc


def _run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "BATCH_SCHEMA",
    "BatchConfig",
    "BatchOrchestrationError",
    "BatchPlan",
    "BatchRunResult",
    "BatchStageResult",
    "ConcurrentBatchError",
    "load_verified_batch_run",
    "plan_batch",
    "run_batch",
]
