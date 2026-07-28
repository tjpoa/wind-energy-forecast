"""Fail-closed ownership and execution leases for scheduled workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
from typing import Any, Literal, Mapping
from uuid import uuid4


OWNERSHIP_SCHEMA = "wind_forecast.scheduler_ownership.v1"
LEASE_SCHEMA = "wind_forecast.scheduler_lease.v1"
RECOVERY_SCHEMA = "wind_forecast.scheduler_lease_recovery.v1"
SCHEDULER_OWNERS = ("windows_task_scheduler", "airflow")


class SchedulerOwnershipError(RuntimeError):
    """Raised when scheduled execution ownership cannot be trusted."""


class SchedulerLeaseError(SchedulerOwnershipError):
    """Raised when another scheduled workflow owns the environment lease."""


@dataclass(frozen=True)
class SchedulerOwnership:
    environment_id: str
    generation: int
    active_scheduler: Literal["windows_task_scheduler", "airflow"]
    ownership_id: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OWNERSHIP_SCHEMA,
            **asdict(self),
        }


@dataclass(frozen=True)
class SchedulerLease:
    environment_id: str
    scheduler: Literal["windows_task_scheduler", "airflow"]
    workflow: str
    run_id: str
    ownership_id: str
    lease_id: str
    host: str
    pid: int
    acquired_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEASE_SCHEMA,
            **asdict(self),
        }


def configure_scheduler_owner(
    scheduler_root: str | Path,
    environment_id: str,
    active_scheduler: str,
    *,
    expected_generation: int,
    expected_owner: str | None,
    now_utc: datetime | None = None,
    dry_run: bool = False,
) -> SchedulerOwnership:
    """Create or atomically replace one environment ownership pointer."""
    root = _environment_root(scheduler_root, environment_id)
    owner = _owner(active_scheduler)
    current = load_scheduler_owner(scheduler_root, environment_id, required=False)
    observed_generation = current.generation if current else 0
    observed_owner = current.active_scheduler if current else None
    if (
        expected_generation != observed_generation
        or expected_owner != observed_owner
    ):
        raise SchedulerOwnershipError(
            "Scheduler ownership differs from the expected generation/owner."
        )
    body = {
        "schema_version": OWNERSHIP_SCHEMA,
        "environment_id": environment_id,
        "generation": observed_generation + 1,
        "active_scheduler": owner,
        "updated_at_utc": _utc_text(now_utc),
    }
    ownership_id = _identifier("scheduler_ownership", body)
    ownership = SchedulerOwnership(
        environment_id=environment_id,
        generation=body["generation"],
        active_scheduler=owner,
        ownership_id=ownership_id,
        updated_at_utc=body["updated_at_utc"],
    )
    if dry_run:
        return ownership
    change_lock = _acquire_ownership_change_lock(root)
    try:
        current = load_scheduler_owner(
            scheduler_root,
            environment_id,
            required=False,
        )
        locked_generation = current.generation if current else 0
        locked_owner = current.active_scheduler if current else None
        if (
            expected_generation != locked_generation
            or expected_owner != locked_owner
        ):
            raise SchedulerOwnershipError(
                "Scheduler ownership changed while configuration was starting."
            )
        if (root / "state" / "execution.lock").exists():
            raise SchedulerLeaseError(
                "Scheduler ownership cannot change while an execution lease exists."
            )
        history_path = root / "history" / ownership_id / "ownership.json"
        _immutable_json(history_path, ownership.to_dict())
        _atomic_json(root / "state" / "current.json", ownership.to_dict())
    finally:
        os.close(change_lock)
        (root / "state" / "ownership-change.lock").unlink(missing_ok=True)
    loaded = load_scheduler_owner(scheduler_root, environment_id)
    if loaded != ownership:
        raise SchedulerOwnershipError(
            "Published scheduler ownership did not verify after publication."
        )
    return ownership


def load_scheduler_owner(
    scheduler_root: str | Path,
    environment_id: str,
    *,
    required: bool = True,
) -> SchedulerOwnership | None:
    """Load and strictly verify one environment ownership pointer."""
    root = _environment_root(scheduler_root, environment_id)
    path = root / "state" / "current.json"
    if not path.is_file():
        if required:
            raise SchedulerOwnershipError(
                "Scheduler ownership is not configured for this environment."
            )
        return None
    payload = _read_json(path)
    required_fields = {
        "schema_version",
        "environment_id",
        "generation",
        "active_scheduler",
        "ownership_id",
        "updated_at_utc",
    }
    if set(payload) != required_fields or payload.get("schema_version") != (
        OWNERSHIP_SCHEMA
    ):
        raise SchedulerOwnershipError("Scheduler ownership fields are invalid.")
    if payload.get("environment_id") != environment_id:
        raise SchedulerOwnershipError(
            "Scheduler ownership belongs to a different environment."
        )
    generation = payload.get("generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        raise SchedulerOwnershipError("Scheduler ownership generation is invalid.")
    owner = _owner(payload.get("active_scheduler"))
    body = {key: value for key, value in payload.items() if key != "ownership_id"}
    ownership_id = payload.get("ownership_id")
    if ownership_id != _identifier("scheduler_ownership", body):
        raise SchedulerOwnershipError("Scheduler ownership identity is corrupt.")
    history_path = root / "history" / str(ownership_id) / "ownership.json"
    if not history_path.is_file() or _read_json(history_path) != payload:
        raise SchedulerOwnershipError(
            "Scheduler ownership history is missing or differs from the pointer."
        )
    _parse_utc(payload.get("updated_at_utc"), "updated_at_utc")
    return SchedulerOwnership(
        environment_id=environment_id,
        generation=generation,
        active_scheduler=owner,
        ownership_id=str(ownership_id),
        updated_at_utc=str(payload["updated_at_utc"]),
    )


def acquire_scheduler_lease(
    scheduler_root: str | Path,
    environment_id: str,
    scheduler: str,
    *,
    workflow: str,
    run_id: str,
    now_utc: datetime | None = None,
) -> SchedulerLease:
    """Acquire the shared environment lease after checking active ownership."""
    root = _environment_root(scheduler_root, environment_id)
    expected_owner = _owner(scheduler)
    if not workflow.strip() or not run_id.strip():
        raise SchedulerOwnershipError("workflow and run_id must be explicit.")
    if (root / "state" / "ownership-change.lock").exists():
        raise SchedulerOwnershipError(
            "Scheduler ownership is currently being changed."
        )
    ownership = load_scheduler_owner(scheduler_root, environment_id)
    if ownership is None or ownership.active_scheduler != expected_owner:
        observed = ownership.active_scheduler if ownership else None
        raise SchedulerOwnershipError(
            f"Scheduler owner is {observed!r}, not {expected_owner!r}."
        )
    body = {
        "schema_version": LEASE_SCHEMA,
        "environment_id": environment_id,
        "scheduler": expected_owner,
        "workflow": workflow,
        "run_id": run_id,
        "ownership_id": ownership.ownership_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "acquired_at_utc": _utc_text(now_utc),
    }
    lease_id = _identifier("scheduler_lease", body)
    payload = {**body, "lease_id": lease_id}
    lock = root / "state" / "execution.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        existing = _read_json(lock)
        raise SchedulerLeaseError(
            "A scheduled execution lease already exists"
            f" for run {existing.get('run_id')!r}; explicit recovery is required"
            " if it is abandoned."
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        lock.unlink(missing_ok=True)
        raise
    try:
        loaded = load_scheduler_lease(scheduler_root, environment_id)
        current_owner = load_scheduler_owner(scheduler_root, environment_id)
        if (
            loaded is None
            or loaded.lease_id != lease_id
            or current_owner is None
            or current_owner.ownership_id != ownership.ownership_id
            or (root / "state" / "ownership-change.lock").exists()
        ):
            raise SchedulerLeaseError(
                "Scheduler ownership changed while the lease was acquired."
            )
        return loaded
    except Exception:
        lock.unlink(missing_ok=True)
        raise


def load_scheduler_lease(
    scheduler_root: str | Path,
    environment_id: str,
) -> SchedulerLease | None:
    """Load and strictly verify the current execution lease."""
    path = (
        _environment_root(scheduler_root, environment_id)
        / "state"
        / "execution.lock"
    )
    if not path.is_file():
        return None
    payload = _read_json(path)
    required = {
        "schema_version",
        "environment_id",
        "scheduler",
        "workflow",
        "run_id",
        "ownership_id",
        "lease_id",
        "host",
        "pid",
        "acquired_at_utc",
    }
    if set(payload) != required or payload.get("schema_version") != LEASE_SCHEMA:
        raise SchedulerLeaseError("Scheduler lease fields are invalid.")
    if payload.get("environment_id") != environment_id:
        raise SchedulerLeaseError("Scheduler lease belongs to another environment.")
    scheduler = _owner(payload.get("scheduler"))
    for key in ("workflow", "run_id", "ownership_id", "host"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise SchedulerLeaseError(f"Scheduler lease {key} is invalid.")
    pid = payload.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise SchedulerLeaseError("Scheduler lease pid is invalid.")
    _parse_utc(payload.get("acquired_at_utc"), "acquired_at_utc")
    body = {key: value for key, value in payload.items() if key != "lease_id"}
    if payload.get("lease_id") != _identifier("scheduler_lease", body):
        raise SchedulerLeaseError("Scheduler lease identity is corrupt.")
    return SchedulerLease(
        environment_id=environment_id,
        scheduler=scheduler,
        workflow=str(payload["workflow"]),
        run_id=str(payload["run_id"]),
        ownership_id=str(payload["ownership_id"]),
        lease_id=str(payload["lease_id"]),
        host=str(payload["host"]),
        pid=pid,
        acquired_at_utc=str(payload["acquired_at_utc"]),
    )


def release_scheduler_lease(
    scheduler_root: str | Path,
    environment_id: str,
    lease_id: str,
) -> None:
    """Release only the exact lease acquired by the caller."""
    root = _environment_root(scheduler_root, environment_id)
    current = load_scheduler_lease(scheduler_root, environment_id)
    if current is None:
        raise SchedulerLeaseError("Scheduler lease is absent.")
    if current.lease_id != lease_id:
        raise SchedulerLeaseError("Scheduler lease identity changed before release.")
    current_owner = load_scheduler_owner(scheduler_root, environment_id)
    if current_owner is None or current_owner.ownership_id != current.ownership_id:
        raise SchedulerLeaseError(
            "Scheduler ownership changed while the execution lease was active."
        )
    (root / "state" / "execution.lock").unlink()


def recover_scheduler_lease(
    scheduler_root: str | Path,
    environment_id: str,
    lease_id: str,
    *,
    recovered_by: str,
    note: str,
    now_utc: datetime | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Explicitly recover one abandoned lease and retain immutable evidence."""
    if not recovered_by.strip() or not note.strip():
        raise SchedulerLeaseError("Lease recovery requires operator and note.")
    root = _environment_root(scheduler_root, environment_id)
    current = load_scheduler_lease(scheduler_root, environment_id)
    if current is None or current.lease_id != lease_id:
        raise SchedulerLeaseError("Exact abandoned lease is not active.")
    body = {
        "schema_version": RECOVERY_SCHEMA,
        "environment_id": environment_id,
        "recovered_at_utc": _utc_text(now_utc),
        "recovered_by": recovered_by,
        "note": note,
        "abandoned_lease": current.to_dict(),
    }
    recovery_id = _identifier("scheduler_lease_recovery", body)
    payload = {**body, "recovery_id": recovery_id}
    target = root / "recoveries" / recovery_id / "recovery.json"
    if dry_run:
        return None
    _immutable_json(target, payload)
    lock = root / "state" / "execution.lock"
    verified = load_scheduler_lease(scheduler_root, environment_id)
    if verified is None or verified.lease_id != lease_id:
        raise SchedulerLeaseError("Lease changed before explicit recovery.")
    lock.unlink()
    return target


def _environment_root(root: str | Path, environment_id: str) -> Path:
    if (
        not isinstance(environment_id, str)
        or not environment_id.strip()
        or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in environment_id)
        or environment_id in {".", ".."}
    ):
        raise SchedulerOwnershipError(
            "environment_id may contain only letters, digits, '.', '_' and '-'."
        )
    return Path(root).resolve() / environment_id


def _acquire_ownership_change_lock(root: Path) -> int:
    path = root / "state" / "ownership-change.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SchedulerOwnershipError(
            "Another scheduler ownership change is already active."
        ) from exc
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return descriptor


def _owner(value: Any) -> Literal["windows_task_scheduler", "airflow"]:
    if value not in SCHEDULER_OWNERS:
        raise SchedulerOwnershipError(
            f"active_scheduler must be one of: {', '.join(SCHEDULER_OWNERS)}."
        )
    return value


def _identifier(kind: str, payload: Mapping[str, Any]) -> str:
    return sha256(kind.encode("utf-8") + b":" + _canonical(payload)).hexdigest()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _utc_text(value: datetime | None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise SchedulerOwnershipError("Scheduler timestamps must be timezone-aware.")
    return (
        current.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _parse_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise SchedulerOwnershipError(f"{name} must be a UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchedulerOwnershipError(f"{name} must be a UTC timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SchedulerOwnershipError(f"{name} must be a UTC timestamp.")
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchedulerOwnershipError(f"Invalid scheduler JSON artifact: {path}.") from exc
    if not isinstance(value, dict):
        raise SchedulerOwnershipError(f"Scheduler JSON artifact is not an object: {path}.")
    return value


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise SchedulerOwnershipError(
                f"Immutable scheduler artifact already differs: {path}."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != data:
            raise SchedulerOwnershipError(
                f"Immutable scheduler artifact already differs: {path}."
            )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "LEASE_SCHEMA",
    "OWNERSHIP_SCHEMA",
    "RECOVERY_SCHEMA",
    "SCHEDULER_OWNERS",
    "SchedulerLease",
    "SchedulerLeaseError",
    "SchedulerOwnership",
    "SchedulerOwnershipError",
    "acquire_scheduler_lease",
    "configure_scheduler_owner",
    "load_scheduler_lease",
    "load_scheduler_owner",
    "recover_scheduler_lease",
    "release_scheduler_lease",
]
