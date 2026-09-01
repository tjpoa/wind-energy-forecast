from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_reproduction_bundle, fetch_reproduction_bundle
from wind_forecast.release_catalog import (
    CATALOG_SCHEMA_V3,
    ReleaseCatalogError,
    load_release_catalog,
    require_release_approved,
    validate_release_catalog,
)


RELEASE = "artifacts-v1.0.0"


def _blocked_catalog() -> dict:
    return {
        "schema_version": CATALOG_SCHEMA_V3,
        "releases": {
            RELEASE: {
                "bundle_sha256": None,
                "source_contract": {
                    "path": "data/manifests/v1_source_contract.json",
                    "sha256": "b" * 64,
                },
                "processed_contract": {
                    "path": "data/manifests/v1_processed_contract.json",
                    "sha256": "c" * 64,
                },
                "redistribution": {
                    "approved": False,
                    "authorization_evidence": [],
                    "classification": "internal",
                    "required_components": ["production", "weather"],
                    "status": "blocked_provenance_incomplete",
                },
            }
        },
    }


def _approved_catalog() -> dict:
    return {
        "schema_version": CATALOG_SCHEMA_V3,
        "releases": {
            RELEASE: {
                "bundle_sha256": "a" * 64,
                "source_contract": {
                    "path": "data/manifests/v1_source_contract.json",
                    "sha256": "b" * 64,
                },
                "processed_contract": {
                    "path": "data/manifests/v1_processed_contract.json",
                    "sha256": "c" * 64,
                },
                "redistribution": {
                    "approved": True,
                    "authorization_evidence": [
                        {
                            "attribution": "Production Provider",
                            "authorization_kind": "public_license",
                            "authorization_reference": "https://example.test/production-license",
                            "authorization_scope": ["redistribution"],
                            "component": "production",
                            "source_contract_sha256": "b" * 64,
                            "license": "CC BY 4.0",
                            "provider": "Production Provider",
                            "redistribution_permitted": True,
                            "source_identifier": "production-v1",
                            "verified_at_utc": "2026-09-01T00:00:00Z",
                        },
                        {
                            "attribution": "Weather Provider",
                            "authorization_kind": "public_license",
                            "authorization_reference": "https://example.test/weather-license",
                            "authorization_scope": ["redistribution"],
                            "component": "weather",
                            "source_contract_sha256": "b" * 64,
                            "license": "CC BY 4.0",
                            "provider": "Weather Provider",
                            "redistribution_permitted": True,
                            "source_identifier": "weather-v1",
                            "verified_at_utc": "2026-09-01T00:00:00Z",
                        },
                    ],
                    "classification": "public",
                    "required_components": ["production", "weather"],
                    "status": "approved_for_redistribution",
                },
            }
        },
    }


def test_repository_catalog_is_blocked_and_internal() -> None:
    catalog_path = Path("artifacts/catalog.json")
    catalog = load_release_catalog(catalog_path)
    entry = catalog["releases"][RELEASE]

    assert catalog["schema_version"] == CATALOG_SCHEMA_V3
    assert entry["redistribution"]["approved"] is False
    assert entry["redistribution"]["classification"] == "internal"
    assert entry["redistribution"]["authorization_evidence"] == []


def test_blocked_release_cannot_be_required_as_approved() -> None:
    with pytest.raises(ReleaseCatalogError, match="not approved"):
        require_release_approved(_blocked_catalog(), RELEASE)


def test_legacy_approved_release_is_rejected() -> None:
    legacy = {
        "schema_version": "wind_forecast.release_catalog.v1",
        "releases": {
            RELEASE: {
                "bundle_sha256": "a" * 64,
                "redistribution": {"approved": True},
            }
        },
    }
    with pytest.raises(ReleaseCatalogError, match="Legacy catalog"):
        validate_release_catalog(legacy)


def test_approved_release_requires_every_component() -> None:
    invalid = _approved_catalog()
    invalid["releases"][RELEASE]["redistribution"]["authorization_evidence"] = []
    with pytest.raises(ReleaseCatalogError, match="one authorization record"):
        validate_release_catalog(invalid)


def test_written_permission_requires_receipt_hash() -> None:
    invalid = _approved_catalog()
    evidence = invalid["releases"][RELEASE]["redistribution"][
        "authorization_evidence"
    ][0]
    evidence["authorization_kind"] = "written_permission"
    with pytest.raises(ReleaseCatalogError, match="receipt_sha256"):
        validate_release_catalog(invalid)


def test_approved_release_requires_explicit_permission_flag() -> None:
    invalid = _approved_catalog()
    invalid["releases"][RELEASE]["redistribution"]["authorization_evidence"][0].pop(
        "redistribution_permitted"
    )
    with pytest.raises(ReleaseCatalogError, match="redistribution_permitted"):
        validate_release_catalog(invalid)


def test_complete_explicit_authorization_is_accepted() -> None:
    catalog = _approved_catalog()
    validate_release_catalog(catalog)
    assert require_release_approved(catalog, RELEASE)["redistribution"]["approved"]


def test_authorization_must_cover_exact_source_snapshot() -> None:
    invalid = _approved_catalog()
    invalid["releases"][RELEASE]["redistribution"]["authorization_evidence"][0][
        "source_contract_sha256"
    ] = "d" * 64
    with pytest.raises(ReleaseCatalogError, match="exact source contract hash"):
        validate_release_catalog(invalid)


def test_legacy_v2_approved_release_is_rejected() -> None:
    legacy = _approved_catalog()
    legacy["schema_version"] = "wind_forecast.release_catalog.v2"
    with pytest.raises(ReleaseCatalogError, match="Legacy catalog"):
        validate_release_catalog(legacy)


def test_build_script_blocks_before_mlflow_or_output(tmp_path: Path, monkeypatch) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_blocked_catalog()), encoding="utf-8")
    monkeypatch.setattr(
        build_reproduction_bundle,
        "_load_mlflow",
        lambda: pytest.fail("MLflow must not be loaded for a blocked release"),
    )
    output_dir = tmp_path / "output"

    with pytest.raises(SystemExit, match="publication is blocked"):
        build_reproduction_bundle.main(
            [
                "--release",
                RELEASE,
                "--model-version",
                "1",
                "--catalog",
                str(catalog_path),
                "--output-dir",
                str(output_dir),
            ]
        )
    assert not output_dir.exists()


def test_fetch_script_blocks_before_network_or_destination(tmp_path: Path, monkeypatch) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_blocked_catalog()), encoding="utf-8")
    monkeypatch.setattr(
        fetch_reproduction_bundle,
        "fetch_release_bundle",
        lambda **_: pytest.fail("network must not be used for a blocked release"),
    )
    destination = tmp_path / "releases"

    with pytest.raises(SystemExit, match="publication is blocked"):
        fetch_reproduction_bundle.main(
            [
                "--release",
                RELEASE,
                "--catalog",
                str(catalog_path),
                "--output-root",
                str(destination),
            ]
        )
    assert not destination.exists()
