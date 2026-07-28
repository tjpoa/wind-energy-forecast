from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from wind_forecast.scheduler_ownership import (
    SchedulerLeaseError,
    SchedulerOwnershipError,
    acquire_scheduler_lease,
    configure_scheduler_owner,
    load_scheduler_lease,
    load_scheduler_owner,
    recover_scheduler_lease,
    release_scheduler_lease,
)


NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)


def _configure(root: Path, owner: str = "windows_task_scheduler"):
    return configure_scheduler_owner(
        root,
        "local",
        owner,
        expected_generation=0,
        expected_owner=None,
        now_utc=NOW,
    )


def test_owner_is_versioned_and_compare_and_set(tmp_path: Path) -> None:
    first = _configure(tmp_path)
    assert first.generation == 1
    assert load_scheduler_owner(tmp_path, "local") == first

    with pytest.raises(SchedulerOwnershipError, match="expected"):
        configure_scheduler_owner(
            tmp_path,
            "local",
            "airflow",
            expected_generation=0,
            expected_owner=None,
            now_utc=NOW,
        )

    second = configure_scheduler_owner(
        tmp_path,
        "local",
        "airflow",
        expected_generation=1,
        expected_owner="windows_task_scheduler",
        now_utc=NOW,
    )
    assert second.generation == 2
    assert second.active_scheduler == "airflow"


def test_owner_mismatch_fails_before_creating_lease(tmp_path: Path) -> None:
    _configure(tmp_path)
    with pytest.raises(SchedulerOwnershipError, match="not 'airflow'"):
        acquire_scheduler_lease(
            tmp_path,
            "local",
            "airflow",
            workflow="monthly",
            run_id="run-1",
            now_utc=NOW,
        )
    assert load_scheduler_lease(tmp_path, "local") is None


def test_lease_refuses_concurrent_ownership_change(tmp_path: Path) -> None:
    _configure(tmp_path)
    marker = tmp_path / "local" / "state" / "ownership-change.lock"
    marker.write_text("operator\n", encoding="utf-8")
    with pytest.raises(SchedulerOwnershipError, match="being changed"):
        acquire_scheduler_lease(
            tmp_path,
            "local",
            "windows_task_scheduler",
            workflow="monthly",
            run_id="run-1",
            now_utc=NOW,
        )
    assert load_scheduler_lease(tmp_path, "local") is None


def test_lease_serializes_execution_and_blocks_owner_switch(tmp_path: Path) -> None:
    owner = _configure(tmp_path)
    lease = acquire_scheduler_lease(
        tmp_path,
        "local",
        "windows_task_scheduler",
        workflow="daily",
        run_id="run-1",
        now_utc=NOW,
    )
    assert lease.ownership_id == owner.ownership_id
    with pytest.raises(SchedulerLeaseError, match="already exists"):
        acquire_scheduler_lease(
            tmp_path,
            "local",
            "windows_task_scheduler",
            workflow="monthly",
            run_id="run-2",
            now_utc=NOW,
        )
    with pytest.raises(SchedulerLeaseError, match="cannot change"):
        configure_scheduler_owner(
            tmp_path,
            "local",
            "airflow",
            expected_generation=1,
            expected_owner="windows_task_scheduler",
            now_utc=NOW,
        )
    release_scheduler_lease(tmp_path, "local", lease.lease_id)
    assert load_scheduler_lease(tmp_path, "local") is None


def test_abandoned_lease_requires_explicit_audited_recovery(tmp_path: Path) -> None:
    _configure(tmp_path)
    lease = acquire_scheduler_lease(
        tmp_path,
        "local",
        "windows_task_scheduler",
        workflow="daily",
        run_id="run-1",
        now_utc=NOW,
    )
    with pytest.raises(SchedulerLeaseError, match="not active"):
        recover_scheduler_lease(
            tmp_path,
            "local",
            "wrong",
            recovered_by="operator",
            note="confirmed abandoned",
            now_utc=NOW,
        )
    recovery = recover_scheduler_lease(
        tmp_path,
        "local",
        lease.lease_id,
        recovered_by="operator",
        note="confirmed abandoned",
        now_utc=NOW,
    )
    assert recovery is not None
    payload = json.loads(recovery.read_text(encoding="utf-8"))
    assert payload["abandoned_lease"]["lease_id"] == lease.lease_id
    assert load_scheduler_lease(tmp_path, "local") is None
