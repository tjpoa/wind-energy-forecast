from wind_forecast.manifests import manifest_from_json, sha256_file
from wind_forecast.paths import project_root


def test_v1_source_contract_is_versioned_and_matches_tracked_raw_files() -> None:
    root = project_root()
    manifest_path = root / "data" / "manifests" / "v1_source_contract.json"
    manifest = manifest_from_json(manifest_path.read_text(encoding="utf-8"))

    assert manifest.dataset_version == "v1"
    assert manifest.status == "provenance_incomplete"
    assert manifest.license is None
    assert manifest.attribution is None
    assert manifest.known_warnings
    for relative_path, expected_hash in manifest.sha256_checksums.items():
        assert sha256_file(root / relative_path) == expected_hash
