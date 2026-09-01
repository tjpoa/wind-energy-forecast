from __future__ import annotations

import json
from pathlib import Path

import pytest

from wind_forecast.manifest_validation import (
    ManifestValidationError,
    main,
    validate_dataset_manifest,
)
from wind_forecast.manifests import DatasetManifest, sha256_file


def _write_fixture_manifest(
    tmp_path: Path,
    *,
    complete_provenance: bool = False,
) -> tuple[Path, Path, dict[str, Path]]:
    root = tmp_path / "repository"
    files = {
        "data/raw/one.csv": root / "data/raw/one.csv",
        "data/raw/two.csv": root / "data/raw/two.csv",
    }
    for index, path in enumerate(files.values(), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"value\n{index}\n", encoding="utf-8")

    if complete_provenance:
        metadata = {
            "provider": "Example Provider",
            "source_identifier": "example-dataset-v1",
            "retrieval_timestamp": "2026-01-01T00:00:00Z",
            "license": "Example License",
            "attribution": "Example Provider",
            "status": "provenance_complete",
            "geographic_coverage": {"provider_confirmed": True},
        }
    else:
        metadata = {
            "status": "provenance_incomplete",
            "geographic_coverage": {"provider_confirmed": False},
        }
    manifest = DatasetManifest(
        dataset_version="v1",
        dataset_role="legacy_v1_source_contract",
        raw_file_paths=tuple(files),
        sha256_checksums={relative: sha256_file(path) for relative, path in files.items()},
        **metadata,
    )
    manifest_path = root / "data/manifests/v1_source_contract.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root, manifest_path, files


def test_validate_dataset_manifest_checks_the_complete_snapshot(tmp_path: Path) -> None:
    root, manifest_path, files = _write_fixture_manifest(tmp_path)

    result = validate_dataset_manifest(
        manifest_path,
        repository_root=root,
        required_paths=[files["data/raw/one.csv"]],
    )

    assert result.mode == "integrity"
    assert set(result.verified_paths) == set(files.values())


@pytest.mark.parametrize("failure", ["changed", "missing"])
def test_validate_dataset_manifest_rejects_changed_or_missing_files(
    tmp_path: Path, failure: str
) -> None:
    root, manifest_path, files = _write_fixture_manifest(tmp_path)
    target = files["data/raw/one.csv"]
    if failure == "changed":
        target.write_text("value\nchanged\n", encoding="utf-8")
    else:
        target.unlink()

    expected_message = "hash mismatch" if failure == "changed" else "missing file"
    with pytest.raises(ManifestValidationError, match=expected_message):
        validate_dataset_manifest(manifest_path, repository_root=root)


def test_validate_dataset_manifest_rejects_undeclared_required_path(tmp_path: Path) -> None:
    root, manifest_path, _ = _write_fixture_manifest(tmp_path)
    undeclared = root / "data/raw/not-in-manifest.csv"
    undeclared.write_text("value\n3\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="not declared"):
        validate_dataset_manifest(
            manifest_path,
            repository_root=root,
            required_paths=[undeclared],
        )


def test_validate_dataset_manifest_rejects_incomplete_checksum_map(tmp_path: Path) -> None:
    root, manifest_path, _ = _write_fixture_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sha256_checksums"].pop("data/raw/two.csv")
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="must match exactly"):
        validate_dataset_manifest(manifest_path, repository_root=root)


def test_validate_dataset_manifest_rejects_invalid_checksum_value(tmp_path: Path) -> None:
    root, manifest_path, _ = _write_fixture_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sha256_checksums"]["data/raw/two.csv"] = "not-a-sha256"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="invalid SHA-256"):
        validate_dataset_manifest(manifest_path, repository_root=root)


def test_release_validation_requires_complete_provenance(tmp_path: Path) -> None:
    root, manifest_path, _ = _write_fixture_manifest(tmp_path)
    with pytest.raises(ManifestValidationError, match="provenance"):
        validate_dataset_manifest(manifest_path, repository_root=root, mode="release")

    complete_root, complete_manifest, _ = _write_fixture_manifest(
        tmp_path / "complete", complete_provenance=True
    )
    result = validate_dataset_manifest(
        complete_manifest, repository_root=complete_root, mode="release"
    )
    assert result.mode == "release"


def test_manifest_cli_reports_success_and_validation_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, manifest_path, files = _write_fixture_manifest(tmp_path)
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--repository-root",
                str(root),
                "--require-path",
                str(files["data/raw/one.csv"]),
            ]
        )
        == 0
    )
    assert '"mode": "integrity"' in capsys.readouterr().out

    files["data/raw/two.csv"].write_text("value\nbroken\n", encoding="utf-8")
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--repository-root",
                str(root),
            ]
        )
        == 1
    )
    assert "Manifest validation failed" in capsys.readouterr().err
