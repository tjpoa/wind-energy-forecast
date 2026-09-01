import pytest
import subprocess
from pathlib import Path

from wind_forecast.manifest_validation import (
    ManifestValidationError,
    validate_v1_source_contract,
)
from wind_forecast.paths import project_root


def test_v1_source_contract_is_versioned_and_matches_tracked_raw_files() -> None:
    result = validate_v1_source_contract()
    manifest = result.manifest

    assert manifest.dataset_version == "v1"
    assert manifest.status == "provenance_incomplete"
    assert manifest.license is None
    assert manifest.attribution is None
    assert manifest.known_warnings
    assert result.verified_paths


def test_v1_source_contract_integrity_is_local_but_release_is_blocked() -> None:
    with pytest.raises(ManifestValidationError, match="provenance"):
        validate_v1_source_contract(mode="release")


def test_v1_data_hashes_are_not_duplicated_outside_the_manifest() -> None:
    root = project_root()
    result = validate_v1_source_contract()
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
