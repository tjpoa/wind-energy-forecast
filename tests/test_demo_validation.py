from __future__ import annotations

from pathlib import Path

import pytest

from wind_forecast.demo_validation import (
    DemoBundleValidationError,
    validate_demo_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "demo" / "v1"


def test_demo_bundle_validation_checks_projection_contracts() -> None:
    result = validate_demo_bundle(BUNDLE_ROOT)

    assert result.bundle_version == "demo-v1"
    assert result.file_count == 15
    assert result.performance_observation_count == 14
    assert result.monitoring_state == "available"
    assert result.monitoring_report_id


def test_demo_bundle_validation_rejects_tampered_file(tmp_path: Path) -> None:
    destination = tmp_path / "demo"
    destination.mkdir()
    for source in BUNDLE_ROOT.rglob("*"):
        target = destination / source.relative_to(BUNDLE_ROOT)
        if source.is_dir():
            target.mkdir()
        else:
            target.write_bytes(source.read_bytes())

    path = destination / "performance" / "predictions.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DemoBundleValidationError, match="checksum"):
        validate_demo_bundle(destination)


def test_demo_bundle_validation_rejects_extra_file(tmp_path: Path) -> None:
    destination = tmp_path / "demo"
    destination.mkdir()
    for source in BUNDLE_ROOT.rglob("*"):
        target = destination / source.relative_to(BUNDLE_ROOT)
        if source.is_dir():
            target.mkdir()
        else:
            target.write_bytes(source.read_bytes())
    (destination / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(DemoBundleValidationError, match="contents"):
        validate_demo_bundle(destination)
