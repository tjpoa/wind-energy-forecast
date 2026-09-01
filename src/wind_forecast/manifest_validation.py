"""Validate dataset manifests before a workflow consumes their files.

The validator deliberately separates local integrity checks from release
provenance checks.  Integrity checks are suitable for reproducible local
workflows; release checks additionally require the source metadata needed to
make a provenance claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any, Literal

from .manifests import DatasetManifest, manifest_from_json, sha256_file
from .paths import manifests_dir, project_root


ManifestValidationMode = Literal["metadata", "integrity", "release"]
V1_DATASET_VERSION = "v1"
V1_DATASET_ROLE = "legacy_v1_source_contract"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """Raised when a manifest or one of its declared files is invalid."""


@dataclass(frozen=True)
class ManifestValidationResult:
    """Evidence returned after a manifest has been validated successfully."""

    manifest: DatasetManifest
    manifest_path: Path
    repository_root: Path
    mode: ManifestValidationMode
    verified_paths: tuple[Path, ...]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready summary without duplicating file hashes."""
        return {
            "manifest_path": str(self.manifest_path),
            "repository_root": str(self.repository_root),
            "dataset_version": self.manifest.dataset_version,
            "dataset_role": self.manifest.dataset_role,
            "status": self.manifest.status,
            "mode": self.mode,
            "verified_paths": [str(path) for path in self.verified_paths],
        }


def default_v1_source_contract_path() -> Path:
    """Return the repository's immutable v1 source-contract path."""
    return manifests_dir() / "v1_source_contract.json"


def validate_dataset_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    mode: ManifestValidationMode = "integrity",
    required_paths: Sequence[str | Path] = (),
) -> ManifestValidationResult:
    """Validate a manifest, its complete snapshot, and requested input paths.

    ``metadata`` validates the manifest declaration without requiring local
    copies.  ``integrity`` and ``release`` additionally verify every declared
    file.  The ``required_paths`` argument prevents a caller from reading a
    file that is not declared by the manifest; it does not narrow the full
    snapshot check in the file-verifying modes.
    """
    if mode not in {"metadata", "integrity", "release"}:
        raise ManifestValidationError(
            f"Unsupported manifest validation mode: {mode!r}."
        )

    root = Path(repository_root or project_root()).resolve()
    if not root.is_dir():
        raise ManifestValidationError(f"Repository root is not a directory: {root}.")

    supplied_manifest_path = Path(manifest_path)
    resolved_manifest_path = (
        supplied_manifest_path
        if supplied_manifest_path.is_absolute()
        else root / supplied_manifest_path
    ).resolve()
    if not resolved_manifest_path.is_file():
        raise ManifestValidationError(
            f"Manifest file is missing: {resolved_manifest_path}."
        )

    try:
        manifest = manifest_from_json(
            resolved_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ManifestValidationError(
            f"Could not parse manifest {resolved_manifest_path}: {exc}"
        ) from exc

    _validate_manifest_shape(manifest)
    declared_paths = _resolve_declared_paths(manifest, root)
    _validate_required_paths(required_paths, root, declared_paths)
    if mode != "metadata":
        _validate_declared_files(manifest, declared_paths)

    if mode == "release":
        _validate_release_provenance(manifest)

    return ManifestValidationResult(
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        repository_root=root,
        mode=mode,
        verified_paths=tuple(declared_paths.values()),
    )


def validate_v1_source_contract(
    *,
    mode: ManifestValidationMode = "integrity",
    required_paths: Sequence[str | Path] = (),
    manifest_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> ManifestValidationResult:
    """Validate the repository's v1 source contract and its declared files."""
    result = validate_dataset_manifest(
        manifest_path or default_v1_source_contract_path(),
        repository_root=repository_root,
        mode=mode,
        required_paths=required_paths,
    )
    if result.manifest.dataset_version != V1_DATASET_VERSION:
        raise ManifestValidationError(
            "The v1 source contract must declare dataset_version='v1'."
        )
    if result.manifest.dataset_role != V1_DATASET_ROLE:
        raise ManifestValidationError(
            "The v1 source contract has an unexpected dataset_role."
        )
    return result


def _validate_manifest_shape(manifest: DatasetManifest) -> None:
    if not manifest.raw_file_paths:
        raise ManifestValidationError("Manifest does not declare any raw files.")
    if len(manifest.raw_file_paths) != len(set(manifest.raw_file_paths)):
        raise ManifestValidationError("Manifest declares duplicate raw file paths.")

    declared = set(manifest.raw_file_paths)
    checksummed = set(manifest.sha256_checksums)
    if declared != checksummed:
        missing = sorted(declared - checksummed)
        extra = sorted(checksummed - declared)
        raise ManifestValidationError(
            "Manifest raw_file_paths and sha256_checksums must match exactly "
            f"(missing checksums: {missing}; extra checksums: {extra})."
        )

    invalid = sorted(
        path
        for path, checksum in manifest.sha256_checksums.items()
        if not isinstance(checksum, str) or not _SHA256_PATTERN.fullmatch(checksum)
    )
    if invalid:
        raise ManifestValidationError(
            "Manifest contains invalid SHA-256 values for: "
            + ", ".join(invalid)
            + "."
        )


def _resolve_declared_paths(
    manifest: DatasetManifest, root: Path
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    resolved_to_declared: dict[Path, str] = {}
    for relative in manifest.raw_file_paths:
        candidate = (root / Path(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ManifestValidationError(
                f"Manifest path escapes repository root: {relative!r}."
            ) from exc
        previous = resolved_to_declared.get(candidate)
        if previous is not None:
            raise ManifestValidationError(
                f"Manifest paths resolve to the same file: {previous!r} and {relative!r}."
            )
        resolved[relative] = candidate
        resolved_to_declared[candidate] = relative
    return resolved


def _validate_required_paths(
    required_paths: Sequence[str | Path],
    root: Path,
    declared_paths: dict[str, Path],
) -> None:
    declared_by_resolved = {path: relative for relative, path in declared_paths.items()}
    for raw_path in required_paths:
        supplied = Path(raw_path)
        candidate = (root / supplied if not supplied.is_absolute() else supplied).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ManifestValidationError(
                f"Requested v1 input is outside repository root: {raw_path!r}."
            ) from exc
        if candidate not in declared_by_resolved:
            relative = candidate.relative_to(root).as_posix()
            raise ManifestValidationError(
                f"Requested v1 input is not declared by the manifest: {relative}."
            )


def _validate_declared_files(
    manifest: DatasetManifest, declared_paths: dict[str, Path]
) -> None:
    issues: list[str] = []
    for relative, path in declared_paths.items():
        if not path.is_file():
            issues.append(f"missing file {relative}")
            continue
        try:
            observed = sha256_file(path)
        except OSError as exc:
            issues.append(f"could not read {relative}: {exc}")
            continue
        expected = manifest.sha256_checksums[relative]
        if observed != expected:
            issues.append(
                f"hash mismatch for {relative} (expected {expected}, observed {observed})"
            )
    if issues:
        raise ManifestValidationError("; ".join(issues) + ".")


def _validate_release_provenance(manifest: DatasetManifest) -> None:
    missing = [
        field
        for field in (
            "provider",
            "source_identifier",
            "retrieval_timestamp",
            "license",
            "attribution",
        )
        if not isinstance(getattr(manifest, field), str)
        or not getattr(manifest, field).strip()
    ]
    if missing:
        raise ManifestValidationError(
            "Release validation requires populated provenance fields: "
            + ", ".join(missing)
            + "."
        )
    if manifest.status != "provenance_complete":
        raise ManifestValidationError(
            "Release validation requires status='provenance_complete'."
        )
    if manifest.geographic_coverage.get("provider_confirmed") is not True:
        raise ManifestValidationError(
            "Release validation requires geographic_coverage.provider_confirmed=true."
        )

    retrieval_timestamp = str(manifest.retrieval_timestamp).strip()
    try:
        parsed = datetime.fromisoformat(retrieval_timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestValidationError(
            "Release validation requires an ISO-8601 retrieval_timestamp."
        ) from exc
    if parsed.tzinfo is None:
        raise ManifestValidationError(
            "Release validation requires a timezone-aware retrieval_timestamp."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a dataset manifest and its declared files."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_v1_source_contract_path(),
        help="Manifest JSON path.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=project_root(),
        help="Repository root used to resolve manifest paths.",
    )
    parser.add_argument(
        "--mode",
        choices=("metadata", "integrity", "release"),
        default="integrity",
        help="Validation policy to apply.",
    )
    parser.add_argument(
        "--require-path",
        action="append",
        default=[],
        help="Input path that must be declared; may be supplied more than once.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest validator CLI."""
    args = _build_parser().parse_args(argv)
    try:
        result = validate_dataset_manifest(
            args.manifest,
            repository_root=args.repository_root,
            mode=args.mode,
            required_paths=args.require_path,
        )
    except ManifestValidationError as exc:
        print(f"Manifest validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ManifestValidationError",
    "ManifestValidationMode",
    "ManifestValidationResult",
    "default_v1_source_contract_path",
    "main",
    "validate_dataset_manifest",
    "validate_v1_source_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())
