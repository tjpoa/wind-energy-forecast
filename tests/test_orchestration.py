from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence

import pytest

from scripts.run_monitoring_report import parse_args as parse_report_args
from scripts.update_v2_dataset import parse_args as parse_update_args
from wind_forecast import batch_cli, orchestration
from wind_forecast.orchestration import (
    BATCH_SCHEMA,
    BatchConfig,
    BatchOrchestrationError,
    ConcurrentBatchError,
    load_verified_batch_run,
    plan_batch,
    run_batch,
)


NOW = datetime(2026, 7, 26, 11, 0, tzinfo=timezone.utc)


def _config(tmp_path: Path, **overrides: Any) -> BatchConfig:
    values: dict[str, Any] = {
        "through_date": "2026-07-26",
        "model_bundle": tmp_path / "model",
        "calibration_dir": tmp_path / "calibration",
        "source_store_root": tmp_path / "source",
        "monitoring_store_root": tmp_path / "monitoring",
        "orchestration_root": tmp_path / "orchestration",
        "now_utc": NOW,
        "no_source_refresh": True,
    }
    values.update(overrides)
    return BatchConfig(**values)


def test_plan_is_read_only_and_uses_dry_run_boundaries(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []

    def runner(command: Sequence[str], timeout: int) -> Mapping[str, Any]:
        commands.append(tuple(command))
        return {"status": "planned", "timeout": timeout}

    config = _config(tmp_path)
    result = plan_batch(config, runner=runner)

    assert result.status == "planned"
    assert [stage.name for stage in result.stages] == [
        "availability_plan",
        "monitoring_plan",
    ]
    assert all("--dry-run" in command for command in commands)
    assert not config.orchestration_root.exists()


def test_run_persists_verified_summary_and_completed_with_alerts(
    tmp_path: Path,
) -> None:
    calls = 0

    def runner(command: Sequence[str], timeout: int) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": "planned"}
        if calls == 2:
            return {
                "status": "succeeded",
                "manifest_path": str(tmp_path / "source-run.json"),
            }
        if calls == 3:
            return {"status": "succeeded", "prediction_ids": ["prediction"]}
        return {
            "status": "succeeded",
            "report_id": "report",
            "active_alert_count": 2,
        }

    config = _config(tmp_path)
    result = run_batch(config, runner=runner)
    verified = load_verified_batch_run(config.orchestration_root)

    assert result.status == "completed_with_alerts"
    assert result.active_alert_count == 2
    assert verified["schema_version"] == BATCH_SCHEMA
    assert verified["status"] == "completed_with_alerts"
    assert [stage["name"] for stage in verified["stages"]] == [
        "availability_plan",
        "dataset_update",
        "predict_reconcile",
        "drift_publish",
    ]
    assert not (config.orchestration_root / "state" / "batch.lock").exists()


def test_failed_stage_is_recorded_and_same_command_can_recover(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def failing(command: Sequence[str], timeout: int) -> Mapping[str, Any]:
        if "--dry-run" not in command:
            raise BatchOrchestrationError("provider token=do-not-record")
        return {"status": "planned"}

    with pytest.raises(BatchOrchestrationError):
        run_batch(config, runner=failing)

    failed = load_verified_batch_run(config.orchestration_root)
    assert failed["status"] == "failed"
    assert failed["failed_stage"] == "dataset_update"
    assert "do-not-record" not in failed["error"]

    calls = 0

    def recovered(command: Sequence[str], timeout: int) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            return {"status": "no_op", "manifest_path": str(tmp_path / "source.json")}
        if calls == 4:
            return {"status": "succeeded", "active_alert_count": 0}
        return {"status": "planned" if calls == 1 else "no_op"}

    result = run_batch(config, runner=recovered)
    assert result.status == "succeeded"
    assert load_verified_batch_run(config.orchestration_root)["run_id"] == result.run_id


def test_cli_requires_explicit_artifact_selections(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = batch_cli.main(["plan", "--through-date", "2026-07-26"])
    error = json.loads(capsys.readouterr().err)
    assert code == 1
    assert error["schema_version"] == "wind_forecast.batch_cli_error.v1"


def test_live_lock_rejects_concurrent_batch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lock = config.orchestration_root / "state" / "batch.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "run_id": "live-run",
                "host": socket.gethostname(),
                "pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConcurrentBatchError, match="live run"):
        run_batch(config, runner=lambda command, timeout: {"status": "planned"})


def test_current_process_liveness_does_not_signal_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_kill(pid: int, signal: int) -> None:
        raise AssertionError("Current-process liveness must not call os.kill.")

    monkeypatch.setattr(orchestration.os, "kill", unexpected_kill)
    assert orchestration._pid_is_live(os.getpid()) is True


def test_stale_lock_is_recorded_before_recovery(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lock = config.orchestration_root / "state" / "batch.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "run_id": "stale-run",
                "host": socket.gethostname(),
                "pid": 999_999_999,
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def runner(command: Sequence[str], timeout: int) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            return {"status": "no_op", "manifest_path": str(tmp_path / "source.json")}
        if calls == 4:
            return {"status": "succeeded", "active_alert_count": 0}
        return {"status": "planned" if calls == 1 else "no_op"}

    run_batch(config, runner=runner)
    recovery = (
        config.orchestration_root
        / "recoveries"
        / "stale-run-abandoned.json"
    )
    assert json.loads(recovery.read_text(encoding="utf-8"))["abandoned_lock"][
        "run_id"
    ] == "stale-run"


def test_latest_pointer_rejects_modified_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls = 0

    def runner(command: Sequence[str], timeout: int) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            return {"status": "no_op", "manifest_path": str(tmp_path / "source.json")}
        if calls == 4:
            return {"status": "succeeded", "active_alert_count": 0}
        return {"status": "planned" if calls == 1 else "no_op"}

    result = run_batch(config, runner=runner)
    assert result.manifest_path is not None
    result.manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(BatchOrchestrationError, match="invalid evidence"):
        load_verified_batch_run(config.orchestration_root)


def test_cli_status_reads_explicit_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    calls = 0

    def runner(command: Sequence[str], timeout: int) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            return {"status": "no_op", "manifest_path": str(tmp_path / "source.json")}
        if calls == 4:
            return {"status": "succeeded", "active_alert_count": 0}
        return {"status": "planned" if calls == 1 else "no_op"}

    result = run_batch(config, runner=runner)
    code = batch_cli.main(
        [
            "status",
            "--orchestration-root",
            str(config.orchestration_root),
            "--manifest",
            str(result.manifest_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["run_id"] == result.run_id


def test_backfill_arguments_must_be_paired() -> None:
    with pytest.raises(SystemExit):
        batch_cli.parse_args(
            [
                "run",
                "--model-bundle",
                "model",
                "--calibration-dir",
                "calibration",
                "--backfill-start",
                "2026-01-01",
            ]
        )


def test_operational_cli_parsers_accept_explicit_argv() -> None:
    update = parse_update_args(
        ["--through-date", "2026-07-26", "--dry-run", "--no-source-refresh"]
    )
    report = parse_report_args(
        [
            "--source-run-manifest",
            "source.json",
            "--calibration-dir",
            "calibration",
            "--through-date",
            "2026-07-26",
            "--dry-run",
        ]
    )
    assert update.through_date == "2026-07-26"
    assert update.dry_run is True
    assert report.source_run_manifest == Path("source.json")
    assert report.dry_run is True
