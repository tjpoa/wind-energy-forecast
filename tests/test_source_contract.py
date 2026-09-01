import json
import pytest
import subprocess
from pathlib import Path

from wind_forecast.manifest_validation import (
    ManifestValidationError,
    validate_v1_source_contract,
)
from wind_forecast.paths import project_root


V1_RAW_PATHS = {
    "data/raw/DirecaoMediaVento10m.csv",
    "data/raw/IntensidadeMediaVento10m.csv",
    "data/raw/ReparticaoProducao.csv",
    "data/raw/TemperaturaMedia.csv",
}


def test_v1_source_contract_metadata_remains_available_without_public_raw_files() -> None:
    result = validate_v1_source_contract(mode="metadata")
    manifest = result.manifest

    assert manifest.dataset_version == "v1"
    assert manifest.status == "provenance_incomplete"
    assert manifest.license is None
    assert manifest.attribution is None
    assert manifest.known_warnings
    assert result.verified_paths


def test_v1_raw_files_are_not_tracked_in_the_public_head() -> None:
    root = project_root()
    tracked = {
        relative
        for relative in subprocess.check_output(
            ["git", "ls-files", "--", "data/raw"], cwd=root
        )
        .decode("utf-8")
        .splitlines()
    }
    assert tracked.isdisjoint(V1_RAW_PATHS)


def test_provenance_review_records_unresolved_fields_without_inference() -> None:
    review_path = (
        project_root()
        / "data"
        / "manifests"
        / "v1_provenance_review_2026-09-01.json"
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))

    assert review["schema_version"] == "wind_forecast.provenance_review.v1"
    assert review["decision"] == {
        "approved": False,
        "classification": "internal",
        "status": "provenance_incomplete",
        "reason": "No component has explicit, snapshot-scoped redistribution authorization.",
    }
    for component in review["components"].values():
        assert component["provider"] is None
        assert component["source_identifier"] is None
        assert component["timezone"] is None
        assert component["license"] is None
        assert component["attribution"] is None
        assert component["redistribution_authorized"] is False


def test_v1_source_contract_integrity_is_local_but_release_is_blocked() -> None:
    with pytest.raises(ManifestValidationError, match="provenance|missing file"):
        validate_v1_source_contract(mode="release")


def test_v1_data_hashes_are_not_duplicated_outside_the_manifest() -> None:
    root = project_root()
    result = validate_v1_source_contract(mode="metadata")
    manifest_path = root / "data" / "manifests" / "v1_source_contract.json"
    text_extensions = {"", ".json", ".md", ".py", ".toml", ".yml", ".yaml", ".ipynb"}
    tracked_paths = (
        root / relative
        for relative in subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=root
        ).decode("utf-8").split("\0")
        if relative
        and root / relative != manifest_path
        and Path(relative).suffix.lower() in text_extensions
    )
    text_by_path = {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for path in tracked_paths
    }
    for checksum in result.manifest.sha256_checksums.values():
        assert not any(checksum in text for text in text_by_path.values())
