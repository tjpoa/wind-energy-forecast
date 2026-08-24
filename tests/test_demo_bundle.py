from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from wind_forecast.api import (
    create_app,
    get_monitoring_service,
    get_performance_service,
)
from wind_forecast.monitoring_projection import MonitoringProjectionService
from wind_forecast.performance import PerformanceService


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "demo" / "v1"


def test_demo_manifest_covers_only_checksum_verified_bundle_files() -> None:
    manifest = json.loads((BUNDLE_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "wind_forecast.demo_bundle.v1"
    assert manifest["bundle_version"] == "demo-v1"
    assert manifest["evidence_type"] == "deterministic_synthetic"
    assert manifest["source"]["credentials_required"] is False
    assert manifest["source"]["network_requests"] is False

    declared = {entry["path"]: entry for entry in manifest["files"]}
    actual = {
        path.relative_to(BUNDLE_ROOT).as_posix(): path
        for path in BUNDLE_ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(BUNDLE_ROOT).as_posix() != "manifest.json"
    }
    assert set(declared) == set(actual)
    for name, entry in declared.items():
        path = actual[name]
        assert entry["size"] == path.stat().st_size
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    pipeline = json.loads((BUNDLE_ROOT / "pipeline" / "run.json").read_text(encoding="utf-8"))
    assert all((BUNDLE_ROOT / output).exists() for output in pipeline["outputs"])


def test_demo_bundle_drives_monitoring_and_performance_api() -> None:
    get_monitoring_service.cache_clear()
    get_performance_service.cache_clear()
    app = create_app()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_monitoring_service] = lambda: MonitoringProjectionService(
        BUNDLE_ROOT / "monitoring"
    )
    app.dependency_overrides[get_performance_service] = lambda: PerformanceService.from_directory(
        BUNDLE_ROOT / "performance"
    )

    with TestClient(app) as client:
        latest = client.get("/api/v1/monitoring/latest")
        performance = client.get("/api/v1/performance")

    assert latest.status_code == 200
    assert latest.json()["state"] == "available"
    assert latest.json()["report"]["model_era"]["deployment_id"] == "demo-deployment-v1"
    assert performance.status_code == 200
    assert performance.json()["observation_count"] == 14
    assert performance.json()["result"]["dataset_version"] == "demo-v1"
