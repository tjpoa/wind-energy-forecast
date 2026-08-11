from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket

import numpy as np
import pandas as pd
import pytest

import wind_forecast.incremental as incremental_module
from wind_forecast.incremental import (
    ConcurrentUpdateError,
    IncrementalUpdateError,
    RefreshResult,
    UpdateConfig,
    materialize_current_features,
    materialize_current_integrated,
    load_verified_current_state,
    plan_v2_update,
    run_v2_update,
)
from wind_forecast.integration import (
    build_integrated_v2_dataset,
    expected_era5_hourly_count,
    expected_ren_interval_count,
    run_synthetic_alignment_checks,
    sha256_file,
)
from wind_forecast.v2_features import (
    build_feature_ready_v2_dataset,
    generate_v2_features,
    map_integrated_base_columns,
    reindex_full_local_calendar,
)


START = "2026-01-01"
END = "2026-01-20"
NOW = datetime(2026, 1, 27, 12, tzinfo=timezone.utc)


def _write_station_mapping(path: Path) -> list[str]:
    station_ids = [f"{1200000 + index}" for index in range(17)]
    rows = [
        {
            "v1_identifier": station_id,
            "status": "exact_match",
            "matched_official_identifier": station_id,
            "station_name": f"Station {index}",
            "latitude": 39.0 + index * 0.01,
            "longitude": -8.0 - index * 0.01,
            "source_endpoint": "synthetic",
            "match_method": "exact_string",
            "confidence": "high",
        }
        for index, station_id in enumerate(station_ids)
    ]
    rows.append(
        {
            "v1_identifier": "1200579",
            "status": "no_match",
            "matched_official_identifier": "",
            "station_name": "Unmatched",
            "latitude": "",
            "longitude": "",
            "source_endpoint": "synthetic",
            "match_method": "none",
            "confidence": "low",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return station_ids


def _write_ren_day(root: Path, day: str, *, value: float = 10.0, extra: bool = False) -> None:
    timestamps = pd.date_range(day, periods=96, freq="15min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "wind_production_mw": value + np.arange(96) / 100.0,
            "unit": "MW",
            "source_date": day,
        }
    )
    if extra:
        frame["retrieval_timestamp_utc"] = "2099-01-01T00:00:00Z"
    path = root / "ren" / "normalized" / f"date={day}" / "production_15min.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_era_month(root: Path, station_ids: list[str]) -> None:
    timestamps = pd.date_range(START, "2026-01-21", freq="h", inclusive="left", tz="UTC")
    period = f"period={START}_{END}"
    for index, station_id in enumerate(station_ids):
        radians = np.linspace(0.0, 1.0, len(timestamps))
        frame = pd.DataFrame(
            {
                "timestamp_utc": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "station_id": station_id,
                "station_name": f"Station {index}",
                "station_latitude": 39.0 + index * 0.01,
                "station_longitude": -8.0 - index * 0.01,
                "grid_latitude": 39.0 + index * 0.01,
                "grid_longitude": -8.0 - index * 0.01,
                "temperature_2m_k": 285.0 + np.sin(radians),
                "temperature_2m_c": 11.85 + np.sin(radians),
                "u10_m_s": 3.0 + np.sin(radians),
                "v10_m_s": 2.0 + np.cos(radians),
                "wind_speed_m_s": 4.0 + np.sin(radians),
                "is_calm_or_near_calm": False,
            }
        )
        path = root / "hourly" / f"station_id={station_id}" / period / "hourly.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)


def _build_environment(tmp_path: Path) -> UpdateConfig:
    ren_root = tmp_path / "raw" / "production"
    era_root = tmp_path / "raw" / "weather"
    mapping = tmp_path / "mapping.csv"
    station_ids = _write_station_mapping(mapping)
    for day in pd.date_range(START, END, freq="D").strftime("%Y-%m-%d"):
        _write_ren_day(ren_root, day)
    _write_era_month(era_root, station_ids)

    integrated_root = tmp_path / "baseline" / "integrated"
    integrated = build_integrated_v2_dataset(
        start_date=START,
        end_date=END,
        ren_root=ren_root,
        era5_root=era_root,
        station_mapping=mapping,
        output_root=integrated_root,
    )
    mapped = map_integrated_base_columns(integrated.daily_merged)
    calendar = reindex_full_local_calendar(integrated.coverage, mapped)
    all_features = generate_v2_features(calendar)
    v1_table = tmp_path / "v1_features.csv"
    all_features.to_csv(v1_table, index=False)
    feature_root = tmp_path / "baseline" / "features"
    build_feature_ready_v2_dataset(
        input_root=integrated_root,
        v1_feature_table=v1_table,
        output_root=feature_root,
    )
    return UpdateConfig(
        through_date=END,
        ren_root=ren_root,
        era5_root=era_root,
        station_mapping=mapping,
        v1_feature_table=v1_table,
        baseline_integrated_root=integrated_root,
        baseline_feature_root=feature_root,
        store_root=tmp_path / "store",
        raw_store_root=tmp_path / "raw_versions",
        revision_lookback_days=0,
        recheck_min_age_hours=0,
        bootstrap_start=START,
        bootstrap_end=END,
        now_utc=NOW,
    )


@pytest.fixture()
def environment(tmp_path: Path) -> UpdateConfig:
    return _build_environment(tmp_path)


def _ren_refresh(tmp_path: Path, day: str, *, value: float, extra: bool = False):
    root = tmp_path / f"refresh-{value}-{extra}"
    _write_ren_day(root, day, value=value, extra=extra)

    def refresh(_plan, _staging):
        return RefreshResult(ren_roots=(root,))

    return refresh


def _ren_unavailable_refresh(tmp_path: Path, day: str):
    root = tmp_path / f"unavailable-{day}"
    path = root / "ren" / "metadata" / f"date={day}" / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "source_date": day,
                "validation": {"validation_status": "unavailable"},
            }
        ),
        encoding="utf-8",
    )

    def refresh(_plan, _staging):
        return RefreshResult(ren_roots=(root,))

    return refresh


def _era_segment_refresh(
    tmp_path: Path,
    source_root: Path,
    *,
    start: str,
    end: str,
) -> Path:
    root = tmp_path / f"era-segment-{start}-{end}"
    for source in source_root.glob("hourly/station_id=*/period=*/hourly.csv"):
        station_part = source.parent.parent.name
        frame = pd.read_csv(source)
        dates = frame["timestamp_utc"].str.slice(0, 10)
        segment = frame.loc[dates.between(start, end)].copy()
        target = (
            root
            / "hourly"
            / station_part
            / f"period={start}_{end}"
            / "hourly.csv"
        )
        target.parent.mkdir(parents=True)
        segment.to_csv(target, index=False)
    return root


def _write_era_days(
    root: Path,
    station_ids: list[str],
    *,
    start: str,
    end: str,
) -> None:
    timestamps = pd.date_range(
        start,
        pd.Timestamp(end) + pd.Timedelta(days=1),
        freq="h",
        inclusive="left",
        tz="UTC",
    )
    for index, station_id in enumerate(station_ids):
        radians = np.linspace(0.0, 1.0, len(timestamps))
        frame = pd.DataFrame(
            {
                "timestamp_utc": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "station_id": station_id,
                "station_name": f"Station {index}",
                "station_latitude": 39.0 + index * 0.01,
                "station_longitude": -8.0 - index * 0.01,
                "grid_latitude": 39.0 + index * 0.01,
                "grid_longitude": -8.0 - index * 0.01,
                "temperature_2m_k": 285.0 + np.sin(radians),
                "temperature_2m_c": 11.85 + np.sin(radians),
                "u10_m_s": 3.0 + np.sin(radians),
                "v10_m_s": 2.0 + np.cos(radians),
                "wind_speed_m_s": 4.0 + np.sin(radians),
                "is_calm_or_near_calm": False,
            }
        )
        path = (
            root
            / "hourly"
            / f"station_id={station_id}"
            / f"period={start}_{end}"
            / "hourly.csv"
        )
        path.parent.mkdir(parents=True)
        frame.to_csv(path, index=False)


def test_dry_run_is_read_only_and_reports_plan(environment: UpdateConfig) -> None:
    config = replace(environment, dry_run=True)
    before = set(environment.store_root.parent.rglob("*"))
    result = run_v2_update(config, source_refresher=lambda *_: pytest.fail("network callback called"))
    after = set(environment.store_root.parent.rglob("*"))

    assert result.status == "planned"
    assert result.run_id is None
    assert result.plan.bootstrap_required is True
    assert result.plan.network_requests_planned == {"ren": 0, "era5_land": 0}
    assert before == after
    assert not environment.store_root.exists()


def test_two_runs_are_idempotent_and_current_readers_verify_data(environment: UpdateConfig) -> None:
    first = run_v2_update(environment)
    pointer_before = (environment.store_root / "state" / "current.json").read_bytes()
    second = run_v2_update(environment)
    pointer_after = (environment.store_root / "state" / "current.json").read_bytes()

    assert first.status == "succeeded"
    assert second.status == "no_op"
    assert second.generation == first.generation == 1
    assert pointer_before == pointer_after
    merged, coverage = materialize_current_integrated(environment.store_root)
    features, feature_coverage = materialize_current_features(environment.store_root)
    assert len(merged) == len(coverage) == 20
    assert len(features) == 6
    assert len(feature_coverage) == 20
    assert not coverage["date_local"].duplicated().any()
    for result in (first, second):
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        quality_ref = manifest["quality_evidence"]
        quality_path = Path(quality_ref["path"])
        assert manifest["schema_version"] == "wind_forecast.v2_incremental_run.v2"
        assert quality_path.is_file()
        assert sha256_file(quality_path) == quality_ref["sha256"]
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        assert quality["schema_version"] == "wind_forecast.batch_quality.v1"
        assert quality["batch_status"] == result.status


def test_ren_revision_rebuilds_only_date_and_forward_feature_window(
    environment: UpdateConfig,
) -> None:
    bootstrap = run_v2_update(environment)
    day = "2026-01-05"
    config = replace(environment, recheck_ren_dates=(day,))
    refresh = _ren_refresh(environment.store_root.parent, day, value=40.0)
    revised = run_v2_update(config, source_refresher=refresh)

    assert bootstrap.status == "succeeded"
    assert revised.status == "succeeded"
    assert revised.affected_dates == (day,)
    assert revised.feature_dates == tuple(
        item.strftime("%Y-%m-%d")
        for item in pd.date_range(day, "2026-01-19", freq="D")
    )
    state = json.loads((environment.store_root / "state" / "current.json").read_text())
    assert state["partitions"]["integrated"][day]["release_id"] == revised.run_id
    assert state["partitions"]["integrated"]["2026-01-04"]["storage"] == "baseline"
    assert state["sources"]["ren"][day]["supersedes_id"] is not None
    assert len(list(environment.raw_store_root.rglob("production_15min.csv"))) == 1

    repeated = run_v2_update(config, source_refresher=refresh)
    assert repeated.status == "no_op"
    assert repeated.affected_dates == ()


def test_ren_complete_to_unavailable_preserves_current_complete(
    environment: UpdateConfig,
) -> None:
    run_v2_update(environment)
    day = "2026-01-05"
    before = json.loads(
        (environment.store_root / "state" / "current.json").read_text()
    )
    partition_key = before["partitions"]["integrated"][day]["partition_key"]
    revision_id = before["sources"]["ren"][day]["revision_id"]
    config = replace(environment, recheck_ren_dates=(day,))
    refresh = _ren_unavailable_refresh(environment.store_root.parent, day)

    result = run_v2_update(config, source_refresher=refresh)
    after = json.loads(
        (environment.store_root / "state" / "current.json").read_text()
    )
    assert result.status == "succeeded"
    assert result.affected_dates == ()
    assert after["sources"]["ren"][day]["status"] == "complete"
    assert after["sources"]["ren"][day]["revision_id"] == revision_id
    assert after["sources"]["ren"][day]["failed_observations"]
    assert after["partitions"]["integrated"][day]["partition_key"] == partition_key
    assert run_v2_update(config, source_refresher=refresh).status == "no_op"


def test_physical_only_revision_is_recorded_without_downstream_rebuild(
    environment: UpdateConfig,
) -> None:
    run_v2_update(environment)
    day = "2026-01-06"
    config = replace(environment, recheck_ren_dates=(day,))
    refresh = _ren_refresh(environment.store_root.parent, day, value=10.0, extra=True)
    result = run_v2_update(config, source_refresher=refresh)

    assert result.status == "succeeded"
    assert result.affected_dates == ()
    assert result.feature_dates == ()
    state = json.loads((environment.store_root / "state" / "current.json").read_text())
    source = state["sources"]["ren"][day]
    assert len(source["observations"]) == 2
    assert source["semantic_equivalent_to"]
    assert state["partitions"]["integrated"][day]["storage"] == "baseline"


def test_era_revision_uses_exact_timestamp_to_local_date_impact(
    environment: UpdateConfig,
) -> None:
    run_v2_update(environment)
    source_path = next(
        environment.era5_root.glob("hourly/station_id=*/period=*/hourly.csv")
    )
    relative = source_path.relative_to(environment.era5_root)
    refresh_root = environment.store_root.parent / "era-refresh"
    target = refresh_root / relative
    target.parent.mkdir(parents=True)
    frame = pd.read_csv(source_path)
    changed_day = "2026-01-08"
    mask = frame["timestamp_utc"].str.startswith(changed_day)
    frame.loc[mask, "u10_m_s"] = frame.loc[mask, "u10_m_s"] + 1.0
    frame.loc[mask, "wind_speed_m_s"] = np.sqrt(
        frame.loc[mask, "u10_m_s"] ** 2 + frame.loc[mask, "v10_m_s"] ** 2
    )
    frame.to_csv(target, index=False)

    config = replace(environment, recheck_era5_months=("2026-01",))
    result = run_v2_update(
        config,
        source_refresher=lambda *_: RefreshResult(era5_roots=(refresh_root,)),
    )
    assert result.affected_dates == (changed_day,)
    assert result.feature_dates[0] == changed_day
    assert result.feature_dates[-1] == END


def test_partial_era_month_extension_merges_per_station_without_duplicates(
    environment: UpdateConfig,
) -> None:
    for path in environment.era5_root.glob(
        "hourly/station_id=*/period=*/hourly.csv"
    ):
        frame = pd.read_csv(path)
        frame.loc[frame["timestamp_utc"].str.slice(0, 10) <= "2026-01-10"].to_csv(
            path, index=False
        )
    first = run_v2_update(environment)
    assert first.plan.era5_missing_months == ("2026-01",)

    refresh_root = _era_segment_refresh(
        environment.store_root.parent,
        _build_environment(environment.store_root.parent / "complete-copy").era5_root,
        start="2026-01-11",
        end=END,
    )
    result = run_v2_update(
        environment,
        source_refresher=lambda *_: RefreshResult(era5_roots=(refresh_root,)),
    )
    state = json.loads(
        (environment.store_root / "state" / "current.json").read_text()
    )
    era = state["sources"]["era5_land"]
    assert result.affected_dates == tuple(
        pd.date_range("2026-01-11", END, freq="D").strftime("%Y-%m-%d")
    )
    assert len(era) == 17
    assert all("/month=2026-01" in key for key in era)
    assert all(ref["status"] == "complete" for ref in era.values())
    assert all(len(ref["complete_utc_dates"]) == 20 for ref in era.values())
    merged, coverage = materialize_current_integrated(environment.store_root)
    assert len(merged) == len(coverage) == 20
    assert not merged["date_local"].duplicated().any()


def test_ren_can_advance_while_era5_is_pending_then_integrates_on_catch_up(
    environment: UpdateConfig,
) -> None:
    bootstrap = run_v2_update(environment)
    ren_ahead_root = environment.store_root.parent / "ren-ahead"
    pending_dates = ("2026-01-21", "2026-01-22")
    for day in pending_dates:
        _write_ren_day(ren_ahead_root, day, value=30.0)

    ren_ahead = replace(
        environment,
        through_date=pending_dates[-1],
        now_utc=datetime(2026, 1, 27, 12, tzinfo=timezone.utc),
    )
    ren_result = run_v2_update(
        ren_ahead,
        source_refresher=lambda *_: RefreshResult(ren_roots=(ren_ahead_root,)),
    )
    state_after_ren = load_verified_current_state(environment.store_root)

    assert bootstrap.status == "succeeded"
    assert ren_result.status == "succeeded"
    assert ren_result.affected_dates == ()
    assert ren_result.feature_dates == ()
    assert set(pending_dates).issubset(state_after_ren["sources"]["ren"])
    assert all(
        state_after_ren["sources"]["ren"][day]["status"] == "complete"
        for day in pending_dates
    )
    assert all(
        day not in state_after_ren["partitions"]["integrated"]
        for day in pending_dates
    )
    assert state_after_ren["watermarks"]["ren"]["observed_through"] == pending_dates[-1]
    assert state_after_ren["watermarks"]["common_validated_watermark"] == END

    pointer_before_no_op = (
        environment.store_root / "state" / "current.json"
    ).read_bytes()
    repeated = run_v2_update(
        ren_ahead,
        source_refresher=lambda *_: RefreshResult(ren_roots=(ren_ahead_root,)),
    )
    assert repeated.status == "no_op"
    assert (
        environment.store_root / "state" / "current.json"
    ).read_bytes() == pointer_before_no_op

    era_catch_up_root = environment.store_root.parent / "era-catch-up"
    station_ids = [f"{1200000 + index}" for index in range(17)]
    _write_era_days(
        era_catch_up_root,
        station_ids,
        start=pending_dates[0],
        end=pending_dates[-1],
    )
    caught_up = replace(
        ren_ahead,
        now_utc=datetime(2026, 1, 28, 12, tzinfo=timezone.utc),
    )
    era_result = run_v2_update(
        caught_up,
        source_refresher=lambda *_: RefreshResult(era5_roots=(era_catch_up_root,)),
    )
    final_state = load_verified_current_state(environment.store_root)

    assert era_result.status == "succeeded"
    assert era_result.affected_dates == pending_dates
    assert all(
        final_state["partitions"]["integrated"][day]["integration_ready"]
        for day in pending_dates
    )
    assert final_state["watermarks"]["common_validated_watermark"] == pending_dates[-1]
    assert run_v2_update(caught_up).status == "no_op"


def test_era_preliminary_to_consolidated_is_versioned_without_recalculation(
    environment: UpdateConfig,
) -> None:
    run_v2_update(environment)
    later = replace(
        environment,
        recheck_era5_months=("2026-01",),
        now_utc=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
    )
    result = run_v2_update(later)
    state = json.loads((environment.store_root / "state" / "current.json").read_text())
    refs = list(state["sources"]["era5_land"].values())

    assert result.status == "succeeded"
    assert result.affected_dates == ()
    assert {item["provider_finality"] for item in refs} == {"consolidated_window"}
    assert all(item["history"] for item in refs)


def test_missing_source_date_is_detected_without_writing(environment: UpdateConfig) -> None:
    missing_day = "2026-01-09"
    path = (
        environment.ren_root
        / "ren"
        / "normalized"
        / f"date={missing_day}"
        / "production_15min.csv"
    )
    path.unlink()
    plan = plan_v2_update(environment)
    assert plan.ren_missing_dates == (missing_day,)
    assert not environment.store_root.exists()


def test_gap_older_than_published_watermark_is_filled_incrementally(
    environment: UpdateConfig,
) -> None:
    gap = "2026-01-04"
    path = (
        environment.ren_root
        / "ren"
        / "normalized"
        / f"date={gap}"
        / "production_15min.csv"
    )
    saved = pd.read_csv(path)
    path.unlink()
    first = run_v2_update(environment)
    assert first.watermarks["ren"]["published_watermark"] == END
    refresh_root = environment.store_root.parent / "gap-refresh"
    target = (
        refresh_root
        / "ren"
        / "normalized"
        / f"date={gap}"
        / "production_15min.csv"
    )
    target.parent.mkdir(parents=True)
    saved.to_csv(target, index=False)
    filled = run_v2_update(
        environment,
        source_refresher=lambda *_: RefreshResult(ren_roots=(refresh_root,)),
    )
    assert filled.affected_dates == (gap,)
    assert filled.watermarks["ren"]["published_watermark"] == END


def test_default_bootstrap_contract_rejects_truncated_baseline(
    environment: UpdateConfig,
) -> None:
    production_config = replace(
        environment,
        bootstrap_start=None,
        bootstrap_end=None,
    )
    with pytest.raises(IncrementalUpdateError, match="bootstrap calendar"):
        plan_v2_update(production_config)


def test_failure_before_publish_preserves_pointer_and_retry_converges(
    environment: UpdateConfig,
) -> None:
    run_v2_update(environment)
    pointer_before = (environment.store_root / "state" / "current.json").read_bytes()
    day = "2026-01-07"
    config = replace(environment, recheck_ren_dates=(day,))
    refresh = _ren_refresh(environment.store_root.parent, day, value=50.0)

    def fail(stage: str) -> None:
        if stage == "before_publish":
            raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        run_v2_update(config, source_refresher=refresh, failure_hook=fail)
    assert (environment.store_root / "state" / "current.json").read_bytes() == pointer_before
    assert list((environment.store_root / "quarantine").iterdir())
    failed_manifest_path = next(
        path
        for path in (environment.store_root / "runs").glob("*/manifest.json")
        if json.loads(path.read_text(encoding="utf-8"))["status"] == "failed"
    )
    failed_manifest = json.loads(failed_manifest_path.read_text(encoding="utf-8"))
    failed_quality = json.loads(
        Path(failed_manifest["quality_evidence"]["path"]).read_text(encoding="utf-8")
    )
    assert failed_quality["batch_status"] == "failed"
    assert failed_quality["verdict"] == "FAIL"

    recovered = run_v2_update(config, source_refresher=refresh)
    assert recovered.status == "succeeded"
    assert recovered.affected_dates == (day,)


@pytest.mark.parametrize(
    "failure_stage",
    ["after_download", "after_validation", "after_integration", "before_publish"],
)
def test_all_prepublication_failpoints_preserve_current_generation(
    environment: UpdateConfig,
    failure_stage: str,
) -> None:
    run_v2_update(environment)
    pointer_path = environment.store_root / "state" / "current.json"
    pointer_before = pointer_path.read_bytes()
    day = "2026-01-12"
    config = replace(environment, recheck_ren_dates=(day,))
    refresh = _ren_refresh(environment.store_root.parent, day, value=70.0)

    def fail(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError(f"injected {failure_stage}")

    with pytest.raises(RuntimeError, match=failure_stage):
        run_v2_update(config, source_refresher=refresh, failure_hook=fail)
    assert pointer_path.read_bytes() == pointer_before
    assert list((environment.store_root / "quarantine").iterdir())


def test_failure_after_publish_leaves_valid_state_and_next_run_is_noop(
    environment: UpdateConfig,
) -> None:
    run_v2_update(environment)
    day = "2026-01-10"
    config = replace(environment, recheck_ren_dates=(day,))
    refresh = _ren_refresh(environment.store_root.parent, day, value=60.0)

    def fail(stage: str) -> None:
        if stage == "after_publish":
            raise RuntimeError("crash after atomic pointer")

    with pytest.raises(RuntimeError, match="crash after atomic pointer"):
        run_v2_update(config, source_refresher=refresh, failure_hook=fail)
    merged, coverage = materialize_current_integrated(environment.store_root)
    assert len(merged) == len(coverage) == 20
    repeated = run_v2_update(config, source_refresher=refresh)
    assert repeated.status == "no_op"


def test_live_lock_is_rejected_and_stale_lock_is_recorded_abandoned(
    environment: UpdateConfig,
) -> None:
    state_root = environment.store_root / "state"
    state_root.mkdir(parents=True)
    lock = state_root / "update.lock"
    lock.write_text(
        json.dumps({"run_id": "live", "host": socket.gethostname(), "pid": os.getpid()})
    )
    with pytest.raises(ConcurrentUpdateError):
        run_v2_update(environment)

    lock.write_text(
        json.dumps({"run_id": "stale", "host": socket.gethostname(), "pid": 99999999})
    )
    stale_run = environment.store_root / "runs" / "stale"
    stale_run.mkdir(parents=True)
    original_manifest = stale_run / "manifest.json"
    original_manifest.write_text(
        json.dumps({"run_id": "stale", "status": "failed", "sentinel": True}),
        encoding="utf-8",
    )
    original_bytes = original_manifest.read_bytes()
    (environment.store_root / "staging" / "stale").mkdir(parents=True)
    stale_release = environment.store_root / "releases" / "stale"
    stale_release.mkdir(parents=True)
    (stale_release / "orphan.txt").write_text("orphan", encoding="utf-8")
    result = run_v2_update(environment)
    abandoned = json.loads(
        (environment.store_root / "runs" / "stale" / "abandoned.json").read_text()
    )
    assert result.status == "succeeded"
    assert abandoned["status"] == "abandoned"
    assert original_manifest.read_bytes() == original_bytes
    assert (environment.store_root / "quarantine" / "stale").is_dir()
    assert not stale_release.exists()
    assert not lock.exists()


def test_stale_lock_for_committed_run_is_not_marked_abandoned(
    environment: UpdateConfig,
) -> None:
    committed = run_v2_update(environment)
    lock = environment.store_root / "state" / "update.lock"
    lock.write_text(
        json.dumps(
            {
                "run_id": committed.run_id,
                "host": socket.gethostname(),
                "pid": 99999999,
            }
        )
    )
    result = run_v2_update(environment)
    assert result.status == "no_op"
    assert not (
        environment.store_root
        / "runs"
        / str(committed.run_id)
        / "abandoned.json"
    ).exists()


def test_current_pointer_race_fails_closed_and_releases_lock(
    environment: UpdateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = incremental_module._load_current_state
    calls = 0

    def fail_second_verification(
        store_root: Path,
        *,
        verify: bool,
    ) -> dict[str, object] | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise IncrementalUpdateError("simulated current pointer race")
        return original_load(store_root, verify=verify)

    monkeypatch.setattr(
        incremental_module,
        "_load_current_state",
        fail_second_verification,
    )
    with pytest.raises(IncrementalUpdateError, match="current pointer race"):
        run_v2_update(environment)

    assert not (environment.store_root / "state" / "update.lock").exists()
    manifests = list((environment.store_root / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["validations"]["current_pointer_verified"] is False


def test_invalid_duplicate_era_partition_fails_closed(environment: UpdateConfig) -> None:
    path = next(environment.era5_root.glob("hourly/station_id=*/period=*/hourly.csv"))
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)

    with pytest.raises(IncrementalUpdateError, match="duplicate"):
        plan_v2_update(environment)
    assert not environment.store_root.exists()

    with pytest.raises(IncrementalUpdateError, match="duplicate"):
        run_v2_update(environment)
    manifest_path = next((environment.store_root / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality_path = Path(manifest["quality_evidence"]["path"])
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert quality["batch_status"] == "failed"
    assert quality["duplicates"]["duplicate_timestamp_count"] == 1
    assert quality["checksums"]["count"] == 1


def test_invalid_era_source_schema_fails_closed(environment: UpdateConfig) -> None:
    path = next(environment.era5_root.glob("hourly/station_id=*/period=*/hourly.csv"))
    frame = pd.read_csv(path).drop(columns=["wind_speed_m_s"])
    frame.to_csv(path, index=False)

    with pytest.raises(IncrementalUpdateError, match="missing columns"):
        plan_v2_update(environment)
    assert not environment.store_root.exists()

    with pytest.raises(IncrementalUpdateError, match="missing columns"):
        run_v2_update(environment)
    manifest_path = next((environment.store_root / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = json.loads(Path(manifest["quality_evidence"]["path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert {item["code"] for item in quality["issues"]} >= {"schema_validation_failed"}
    assert quality["checksums"]["files"][0]["source"] == "rejected_input"


def test_invalid_monitoring_policy_produces_failed_batch_evidence(
    environment: UpdateConfig,
) -> None:
    policy_path = environment.store_root.parent / "invalid-policy.json"
    policy_path.write_text(json.dumps({"schema_version": "invalid"}), encoding="utf-8")
    config = replace(environment, monitoring_policy_path=policy_path)

    with pytest.raises(ValueError, match="Unsupported monitoring policy"):
        run_v2_update(config)

    manifest_path = next((environment.store_root / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = json.loads(Path(manifest["quality_evidence"]["path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert quality["policy"]["status"] == "invalid"
    assert quality["policy"]["sha256"] == incremental_module.sha256_file(policy_path)
    assert not (environment.store_root / "state" / "update.lock").exists()


def test_failed_run_falls_back_when_detailed_quality_scan_fails(
    environment: UpdateConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = incremental_module.build_batch_quality_evidence
    calls = 0

    def flaky_quality(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated quality scan failure")
        return original(**kwargs)

    monkeypatch.setattr(
        incremental_module, "build_batch_quality_evidence", flaky_quality
    )

    def fail(stage: str) -> None:
        if stage == "after_download":
            raise RuntimeError("simulated batch failure")

    with pytest.raises(RuntimeError, match="simulated batch failure"):
        run_v2_update(environment, failure_hook=fail)

    manifest_path = next((environment.store_root / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = json.loads(Path(manifest["quality_evidence"]["path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert quality["batch_status"] == "failed"
    assert calls == 2
    assert not (environment.store_root / "state" / "update.lock").exists()


def test_null_ren_revision_fails_closed_without_changing_current(
    environment: UpdateConfig,
) -> None:
    run_v2_update(environment)
    pointer = environment.store_root / "state" / "current.json"
    before = pointer.read_bytes()
    day = "2026-01-13"
    root = environment.store_root.parent / "null-ren"
    _write_ren_day(root, day, value=10.0)
    path = root / "ren" / "normalized" / f"date={day}" / "production_15min.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "wind_production_mw"] = np.nan
    frame.to_csv(path, index=False)
    config = replace(environment, recheck_ren_dates=(day,))
    with pytest.raises(IncrementalUpdateError, match="invalid"):
        run_v2_update(
            config,
            source_refresher=lambda *_: RefreshResult(ren_roots=(root,)),
        )
    assert pointer.read_bytes() == before


def test_dst_contract_counts() -> None:
    assert expected_ren_interval_count("2026-03-29") == 92
    assert expected_ren_interval_count("2026-10-25") == 100
    assert expected_era5_hourly_count("2026-03-29") == 23
    assert expected_era5_hourly_count("2026-10-25") == 25
    assert run_synthetic_alignment_checks()["passed"] is True


def test_incremental_revision_matches_clean_full_rebuild(
    environment: UpdateConfig,
) -> None:
    run_v2_update(environment)
    day = "2026-01-14"
    refresh = _ren_refresh(environment.store_root.parent, day, value=80.0)
    config = replace(environment, recheck_ren_dates=(day,))
    run_v2_update(config, source_refresher=refresh)
    current_merged, current_coverage = materialize_current_integrated(
        environment.store_root
    )
    current_features, current_feature_coverage = materialize_current_features(
        environment.store_root
    )

    clean_root = environment.store_root.parent / "clean-rebuild"
    clean_ren = clean_root / "production"
    clean_era = clean_root / "weather"
    shutil.copytree(environment.ren_root, clean_ren)
    shutil.copytree(environment.era5_root, clean_era)
    revised_source = next(
        (environment.store_root.parent / "refresh-80.0-False").glob(
            f"ren/normalized/date={day}/production_15min.csv"
        )
    )
    revised_target = (
        clean_ren / "ren" / "normalized" / f"date={day}" / "production_15min.csv"
    )
    shutil.copyfile(revised_source, revised_target)
    reference_integrated_root = clean_root / "integrated"
    build_integrated_v2_dataset(
        start_date=START,
        end_date=END,
        ren_root=clean_ren,
        era5_root=clean_era,
        station_mapping=environment.station_mapping,
        output_root=reference_integrated_root,
    )
    reference_feature_root = clean_root / "features"
    build_feature_ready_v2_dataset(
        input_root=reference_integrated_root,
        v1_feature_table=environment.v1_feature_table,
        output_root=reference_feature_root,
    )
    reference_features = pd.read_csv(
        reference_feature_root / "feature_ready_daily.csv"
    )
    reference_merged = pd.read_csv(reference_integrated_root / "daily_merged.csv")
    reference_coverage = pd.read_csv(reference_integrated_root / "coverage.csv")
    reference_feature_coverage = pd.read_csv(
        reference_feature_root / "feature_coverage.csv"
    )
    pd.testing.assert_frame_equal(
        current_merged.reset_index(drop=True),
        reference_merged.reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        current_coverage.reset_index(drop=True),
        reference_coverage.reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        current_features.reset_index(drop=True),
        reference_features.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
    )
    pd.testing.assert_frame_equal(
        current_feature_coverage.reset_index(drop=True),
        reference_feature_coverage.reset_index(drop=True),
        check_dtype=False,
    )


def test_current_reader_rejects_checksum_corruption(environment: UpdateConfig) -> None:
    run_v2_update(environment)
    pointer = json.loads((environment.store_root / "state" / "current.json").read_text())
    ref = pointer["partitions"]["integrated"][START]["files"]["coverage"]
    path = Path(ref["path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(IncrementalUpdateError, match="corrupt"):
            materialize_current_integrated(environment.store_root)
    finally:
        path.write_bytes(original)


def test_manifest_checksum_matches_current_pointer(
    environment: UpdateConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_v2_update(environment)
    captured = capsys.readouterr()
    pointer = json.loads((environment.store_root / "state" / "current.json").read_text())
    assert result.manifest_sha256 == pointer["manifest_sha256"]
    assert sha256_file(Path(pointer["manifest_path"])) == pointer["manifest_sha256"]
    manifest = json.loads(Path(pointer["manifest_path"]).read_text())
    started = datetime.fromisoformat(manifest["started_at_utc"].replace("Z", "+00:00"))
    finished = datetime.fromisoformat(manifest["finished_at_utc"].replace("Z", "+00:00"))
    assert finished >= started
    event_lines = (
        (Path(pointer["manifest_path"]).parent / "events.jsonl")
        .read_text()
        .splitlines()
    )
    events = [json.loads(line) for line in event_lines]
    assert captured.out == ""
    assert captured.err.splitlines() == event_lines
    assert events
    assert all(event["source"] for event in events)
    assert all(isinstance(event["duration_ms"], (int, float)) for event in events)
    assert events[-1]["stage"] == "run"
    assert events[-1]["result"] == "succeeded"


def test_public_current_state_loader_returns_verified_detached_copy(
    environment: UpdateConfig,
) -> None:
    result = run_v2_update(environment)
    first = load_verified_current_state(environment.store_root)
    assert first["release_id"] == result.run_id
    first["generation"] = -1
    second = load_verified_current_state(environment.store_root)
    assert second["generation"] == result.generation
