from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from wind_forecast.batch_quality import build_batch_quality_evidence


def _plan(day: str) -> dict[str, object]:
    return {
        "potentially_affected_dates": [day],
        "ren_missing_dates": [],
        "pending_availability_dates": {"ren": [], "era5_land": []},
    }


def test_batch_quality_records_dst_interval_contract_and_schema(tmp_path) -> None:
    day = "2026-03-29"
    ren_path = tmp_path / "ren.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-29", periods=92, freq="15min", tz="UTC"),
            "wind_production_mw": [1.0] * 92,
            "unit": ["MW"] * 92,
            "source_date": [day] * 92,
            "retrieval_timestamp_utc": ["2026-04-03T12:00:00Z"] * 92,
            "endpoint_identifier": ["test"] * 92,
            "raw_response_sha256": ["0" * 64] * 92,
        }
    ).to_csv(ren_path, index=False)
    era_path = tmp_path / "era.csv"
    pd.DataFrame(
        columns=[
            "timestamp_utc",
            "station_id",
            "station_name",
            "station_latitude",
            "station_longitude",
            "grid_latitude",
            "grid_longitude",
            "temperature_2m_k",
            "temperature_2m_c",
            "u10_m_s",
            "v10_m_s",
            "wind_speed_m_s",
            "wind_direction_deg_from",
            "is_calm_or_near_calm",
        ]
    ).to_csv(era_path, index=False)
    era = {
        f"station-{index}": {
            "status": "complete",
            "station_id": str(index),
            "primary_path": str(era_path),
            "local_dates": [day],
            "local_hour_counts": {day: 23},
        }
        for index in range(17)
    }
    state = {
        "watermarks": {"common_validated_watermark": day},
        "sources": {
            "ren": {day: {"status": "complete", "primary_path": str(ren_path)}},
            "era5_land": era,
        },
        "partitions": {
            "integrated": {day: {"integration_ready": True}},
            "features": {day: {"feature_ready": True, "files": {}}},
        },
    }

    quality = build_batch_quality_evidence(
        run_id="run",
        through_date=day,
        evaluated_at_utc=datetime(2026, 4, 3, 12, tzinfo=timezone.utc),
        plan=_plan(day),
        state=state,
        status="succeeded",
    )

    ren_check = next(item for item in quality["intervals"]["checks"] if item["source"] == "ren")
    era_checks = [item for item in quality["intervals"]["checks"] if item["source"] == "era5_land"]
    assert ren_check["expected"] == 92
    assert len(era_checks) == 17
    assert {item["expected"] for item in era_checks} == {23}
    assert quality["intervals"]["invalid_complete_count"] == 0
    assert quality["schemas"]["incompatible_schema_count"] == 0
    assert quality["checksums"]["count"] == 2

    era["station-0"]["status"] = "partial"
    era["station-0"]["local_hour_counts"][day] = 22
    incomplete_quality = build_batch_quality_evidence(
        run_id="incomplete-era",
        through_date=day,
        evaluated_at_utc=datetime(2026, 4, 5, 12, tzinfo=timezone.utc),
        plan=_plan(day),
        state=state,
        status="succeeded",
    )
    assert incomplete_quality["coverage"]["era5_complete_count"] == 0
    assert incomplete_quality["coverage"]["dates"][0]["era5_station_count"] == 16
    assert "source_late" in {item["code"] for item in incomplete_quality["issues"]}
    era["station-0"]["status"] = "complete"
    era["station-0"]["local_hour_counts"][day] = 23

    invalid_ren = pd.read_csv(ren_path).iloc[:-1]
    invalid_ren.to_csv(ren_path, index=False)
    invalid_quality = build_batch_quality_evidence(
        run_id="invalid-run",
        through_date=day,
        evaluated_at_utc=datetime(2026, 4, 3, 12, tzinfo=timezone.utc),
        plan=_plan(day),
        state=state,
        status="succeeded",
    )
    invalid_check = next(
        item for item in invalid_quality["intervals"]["checks"] if item["source"] == "ren"
    )
    assert invalid_check["actual"] == 91
    assert invalid_quality["intervals"]["invalid_complete_count"] == 1
    assert "invalid_complete_interval_count" in {
        item["code"] for item in invalid_quality["issues"]
    }


def test_failed_batch_has_structured_duplicate_issue() -> None:
    quality = build_batch_quality_evidence(
        run_id="failed",
        through_date="2026-01-01",
        evaluated_at_utc=datetime(2026, 1, 8, 12, tzinfo=timezone.utc),
        plan=_plan("2026-01-01"),
        state=None,
        status="failed",
        error="ERA5 partition contains duplicate timestamps",
    )

    codes = {item["code"] for item in quality["issues"]}
    assert "duplicate_validation_failed" in codes
    assert quality["verdict"] == "FAIL"

    tolerated = build_batch_quality_evidence(
        run_id="tolerated",
        through_date="2026-01-01",
        evaluated_at_utc=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        plan=_plan("2026-01-01"),
        state=None,
        status="failed",
        hard_quality_tolerance=1,
        error="ERA5 partition contains duplicate timestamps",
    )
    duplicate_issue = next(
        item for item in tolerated["issues"] if item["code"] == "duplicate_validation_failed"
    )
    assert duplicate_issue["severity"] == "informational"


def test_source_late_alert_starts_at_d7_noon_lisbon() -> None:
    state = {
        "watermarks": {"common_validated_watermark": "2026-03-28"},
        "sources": {},
        "partitions": {},
    }
    before = build_batch_quality_evidence(
        run_id="before-noon",
        through_date="2026-03-29",
        evaluated_at_utc=datetime(2026, 4, 5, 10, 59, tzinfo=timezone.utc),
        plan=_plan("2026-03-29"),
        state=state,
        status="failed",
        error="source unavailable",
    )
    at_deadline = build_batch_quality_evidence(
        run_id="at-noon",
        through_date="2026-03-29",
        evaluated_at_utc=datetime(2026, 4, 5, 11, 0, tzinfo=timezone.utc),
        plan=_plan("2026-03-29"),
        state=state,
        status="failed",
        error="source unavailable",
    )
    assert "source_late" not in {item["code"] for item in before["issues"]}
    assert "source_late" in {item["code"] for item in at_deadline["issues"]}


def test_d5_objective_uses_latest_due_date_not_watermark_age() -> None:
    day = "2026-03-29"
    base_state = {"sources": {}, "partitions": {}}
    complete = build_batch_quality_evidence(
        run_id="objective-complete",
        through_date=day,
        evaluated_at_utc=datetime(2026, 4, 3, 11, 0, tzinfo=timezone.utc),
        plan=_plan(day),
        state={**base_state, "watermarks": {"common_validated_watermark": day}},
        status="succeeded",
    )
    late = build_batch_quality_evidence(
        run_id="objective-late",
        through_date=day,
        evaluated_at_utc=datetime(2026, 4, 3, 11, 0, tzinfo=timezone.utc),
        plan=_plan(day),
        state={
            **base_state,
            "watermarks": {"common_validated_watermark": "2026-03-28"},
        },
        status="succeeded",
    )
    assert complete["freshness"]["objective_missed"] is False
    assert late["freshness"]["objective_missed"] is True


def test_freshness_checks_overdue_date_when_requested_through_is_newer() -> None:
    quality = build_batch_quality_evidence(
        run_id="newer-through",
        through_date="2026-04-02",
        evaluated_at_utc=datetime(2026, 4, 5, 11, 0, tzinfo=timezone.utc),
        plan=_plan("2026-04-02"),
        state=None,
        status="failed",
        error="source unavailable",
    )
    assert quality["freshness"]["unresolved_late_dates"] == [
        {"date": "2026-03-29", "status": "source_late"}
    ]


def test_old_explicit_gap_is_not_hidden_by_a_later_watermark() -> None:
    plan = _plan("2026-04-02")
    plan["ren_unavailable_dates"] = ["2026-03-29"]
    quality = build_batch_quality_evidence(
        run_id="old-gap",
        through_date="2026-04-02",
        evaluated_at_utc=datetime(2026, 4, 5, 11, 0, tzinfo=timezone.utc),
        plan=plan,
        state={
            "watermarks": {"common_validated_watermark": "2026-04-01"},
            "sources": {},
            "partitions": {},
        },
        status="succeeded",
    )
    assert {item["date"] for item in quality["freshness"]["unresolved_late_dates"]} == {
        "2026-03-29"
    }


def test_schema_added_removed_and_reordered_are_reported(tmp_path) -> None:
    day = "2026-01-15"
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range(day, periods=96, freq="15min", tz="UTC"),
            "wind_production_mw": [1.0] * 96,
            "unit": ["MW"] * 96,
            "source_date": [day] * 96,
            "retrieval_timestamp_utc": ["2026-01-16T12:00:00Z"] * 96,
            "endpoint_identifier": ["test"] * 96,
            "raw_response_sha256": ["0" * 64] * 96,
        }
    )
    path = tmp_path / "ren.csv"
    state = {
        "watermarks": {"common_validated_watermark": day},
        "sources": {"ren": {day: {"status": "complete", "primary_path": str(path)}}},
        "partitions": {},
    }

    variants = [
        (base.assign(extra=1), "additional_schema_columns"),
        (base.drop(columns=["unit"]), "missing_required_schema_columns"),
        (base[[*base.columns[1:], base.columns[0]]], "schema_column_order_changed"),
    ]
    for frame, expected_code in variants:
        frame.to_csv(path, index=False)
        quality = build_batch_quality_evidence(
            run_id=expected_code,
            through_date=day,
            evaluated_at_utc=datetime(2026, 1, 16, 12, tzinfo=timezone.utc),
            plan=_plan(day),
            state=state,
            status="succeeded",
        )
        assert expected_code in {item["code"] for item in quality["issues"]}
