import json
from pathlib import Path

import pytest

from wind_forecast.readiness import (
    READINESS_SCHEMA,
    ReadinessError,
    load_automation_readiness,
)
from wind_forecast.paths import project_root


def test_repository_readiness_is_explicitly_no_go() -> None:
    receipt = load_automation_readiness(
        project_root() / "config" / "local_automation_readiness_v1.json",
        environment_id="local",
    )

    assert receipt.status == "NO-GO"
    assert not receipt.allows("historical_daily_batch")


def test_go_receipt_requires_evidence_and_workflow(tmp_path: Path) -> None:
    path = tmp_path / "readiness.json"
    path.write_text(
        json.dumps(
            {
                "allowed_workflows": ["historical_daily_batch"],
                "environment_id": "local",
                "evidence_refs": ["receipt-1"],
                "reason_codes": [],
                "schema_version": READINESS_SCHEMA,
                "status": "GO",
                "updated_at_utc": "2026-08-31T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    receipt = load_automation_readiness(
        path,
        environment_id="local",
        workflow="historical_daily_batch",
    )
    assert receipt.allows("historical_daily_batch")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.update({"allowed_workflows": ["x"]}), "NO-GO"),
        (
            lambda payload: payload.update(
                {"status": "GO", "evidence_refs": []}
            ),
            "GO receipt",
        ),
        (lambda payload: payload.update({"updated_at_utc": "not-a-date"}), "ISO-8601"),
    ],
)
def test_readiness_rejects_invalid_receipts(tmp_path: Path, mutate, message: str) -> None:
    payload = {
        "allowed_workflows": [],
        "environment_id": "local",
        "evidence_refs": [],
        "reason_codes": ["blocked"],
        "schema_version": READINESS_SCHEMA,
        "status": "NO-GO",
        "updated_at_utc": "2026-08-31T00:00:00Z",
    }
    mutate(payload)
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReadinessError, match=message):
        load_automation_readiness(path, environment_id="local")
