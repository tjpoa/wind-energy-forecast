"""Deterministic release bundles and safe materialization helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping

from .manifests import sha256_file


BUNDLE_SCHEMA_VERSION = "wind_forecast.reproduction_bundle.v1"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
RELEASE_PATTERN = re.compile(r"^artifacts-v\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class BundleResult:
    archive_path: Path
    checksum_path: Path
    sha256: str
    manifest: dict[str, Any]


def build_reproduction_bundle(
    *,
    release: str,
    model_name: str,
    model_version: str,
    run_id: str,
    git_sha: str,
    files: Mapping[str, str | Path],
    output_dir: str | Path,
    redistribution: Mapping[str, Any],
) -> BundleResult:
    """Create a byte-stable ZIP and checksum from explicit source files."""
    validate_release(release)
    normalized: dict[str, Path] = {}
    for archive_name, source in files.items():
        name = _safe_member_name(archive_name)
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Bundle source is missing: {source_path}")
        normalized[name] = source_path
    if not normalized:
        raise ValueError("At least one bundle file is required.")

    file_entries = [
        {
            "path": name,
            "sha256": sha256_file(source),
            "size": source.stat().st_size,
        }
        for name, source in sorted(normalized.items())
    ]
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "release": release,
        "git_sha": git_sha,
        "registered_model_name": model_name,
        "model_version": str(model_version),
        "run_id": run_id,
        "redistribution": dict(redistribution),
        "files": file_entries,
        "dataset": _optional_json(normalized.get("manifests/dataset_manifest.json")),
        "model": _optional_json(normalized.get("manifests/model_manifest.json")),
        "metrics": _optional_json(normalized.get("baseline/metrics.json")),
        "environment": _optional_json(normalized.get("environment/environment.json")),
    }
    manifest_bytes = _canonical_json(manifest).encode("utf-8")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    version = release.removeprefix("artifacts-")
    archive_path = output / f"wind-energy-forecast-artifacts-{version}.zip"
    checksum_path = output / f"wind-energy-forecast-artifacts-{version}.sha256"
    if archive_path.exists() or checksum_path.exists():
        raise FileExistsError("Release bundle outputs already exist; use a new version.")

    temporary_archive = output / f".{archive_path.name}.tmp"
    temporary_checksum = output / f".{checksum_path.name}.tmp"
    try:
        with zipfile.ZipFile(
            temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            _writestr(archive, "manifest.json", manifest_bytes)
            for name, source in sorted(normalized.items()):
                _writestr(archive, name, source.read_bytes())
        digest = sha256_file(temporary_archive)
        temporary_checksum.write_text(
            f"{digest}  {archive_path.name}\n", encoding="ascii", newline="\n"
        )
        temporary_archive.replace(archive_path)
        temporary_checksum.replace(checksum_path)
    finally:
        temporary_archive.unlink(missing_ok=True)
        temporary_checksum.unlink(missing_ok=True)
    return BundleResult(archive_path, checksum_path, digest, manifest)


def verify_bundle(archive_path: str | Path, checksum_path: str | Path | None = None) -> dict[str, Any]:
    """Verify archive checksum, member safety, manifest, sizes, and file hashes."""
    archive_path = Path(archive_path)
    if checksum_path is not None:
        expected = Path(checksum_path).read_text(encoding="ascii").split()[0]
        actual = sha256_file(archive_path)
        if actual != expected:
            raise ValueError("Bundle SHA-256 does not match its checksum file.")
    with zipfile.ZipFile(archive_path) as archive:
        member_items = archive.infolist()
        names = [item.filename for item in member_items]
        if len(names) != len(set(names)):
            raise ValueError("Bundle contains duplicate member names.")
        members = {item.filename: item for item in member_items}
        for name in names:
            _safe_member_name(name)
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except KeyError as exc:
            raise ValueError("Bundle is missing manifest.json.") from exc
        if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError("Unsupported reproduction bundle schema.")
        declared_names = [entry.get("path") for entry in manifest.get("files", [])]
        if len(declared_names) != len(set(declared_names)):
            raise ValueError("Bundle manifest declares duplicate member names.")
        expected_names = {"manifest.json", *declared_names}
        actual_names = set(names)
        if expected_names != actual_names:
            missing = sorted(expected_names - actual_names)
            undeclared = sorted(actual_names - expected_names)
            raise ValueError(
                f"Bundle members differ from manifest; missing={missing}, undeclared={undeclared}."
            )
        for entry in manifest.get("files", []):
            name = _safe_member_name(entry["path"])
            if name not in members:
                raise ValueError(f"Bundle member is missing: {name}.")
            payload = archive.read(name)
            if len(payload) != int(entry["size"]):
                raise ValueError(f"Bundle member size mismatch: {name}.")
            if sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Bundle member checksum mismatch: {name}.")
    return manifest


def extract_bundle(
    archive_path: str | Path,
    destination: str | Path,
    *,
    refuse_overwrite: bool = True,
) -> Path:
    """Safely extract a verified bundle without archive path traversal."""
    archive_path = Path(archive_path)
    verify_bundle(archive_path)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Bundle destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        staging = Path(temporary) / "extracted"
        staging.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            items = list(archive.infolist())
            for item in items:
                name = _safe_member_name(item.filename)
                target = staging / Path(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        staging.replace(destination)
    return destination


def fetch_release_bundle(
    *,
    release: str,
    repository: str,
    destination: str | Path,
    opener: Callable[[str], Any] | None = None,
    expected_sha256: str | None = None,
) -> tuple[Path, Path]:
    """Download immutable release assets; verification remains a separate step."""
    validate_release(release)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must use the owner/name form.")
    version = release.removeprefix("artifacts-")
    archive_name = f"wind-energy-forecast-artifacts-{version}.zip"
    checksum_name = f"wind-energy-forecast-artifacts-{version}.sha256"
    base = f"https://github.com/{repository}/releases/download/{release}"
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Release destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_path = destination / archive_name
    checksum_path = destination / checksum_name
    open_url = opener or urllib.request.urlopen
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        staging = Path(temporary) / "download"
        staging.mkdir()
        staged_archive = staging / archive_name
        staged_checksum = staging / checksum_name
        for url, path in (
            (f"{base}/{archive_name}", staged_archive),
            (f"{base}/{checksum_name}", staged_checksum),
        ):
            with open_url(url) as response, path.open("xb") as output:
                shutil.copyfileobj(response, output)
        verify_bundle(staged_archive, staged_checksum)
        actual = sha256_file(staged_archive)
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError("Bundle SHA-256 does not match the tracked release catalog.")
        staging.replace(destination)
    return archive_path, checksum_path


def materialize_training_data(
    extracted_root: str | Path, destination: str | Path
) -> Path:
    """Copy the bundled v1 training table to its conventional local path."""
    source = Path(extracted_root) / "data" / "agg_data_ml.csv"
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError("Bundle does not contain data/agg_data_ml.csv.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copy2(source, temporary_path)
        if sha256_file(source) != sha256_file(temporary_path):
            raise ValueError("Materialized dataset checksum mismatch.")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def bundle_temp_extract(archive_path: str | Path):
    """Return a temporary directory context containing a verified extraction."""
    return _BundleTemporaryDirectory(archive_path)


class _BundleTemporaryDirectory:
    def __init__(self, archive_path: str | Path):
        self.archive_path = Path(archive_path)
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._temporary = tempfile.TemporaryDirectory()
        destination = Path(self._temporary.name) / "bundle"
        extract_bundle(self.archive_path, destination)
        return destination

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()


def _safe_member_name(name: str) -> str:
    raw = str(name).replace("\\", "/")
    path = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if not raw or path.is_absolute() or windows.is_absolute() or windows.drive or ".." in path.parts:
        raise ValueError(f"Unsafe bundle member path: {name!r}.")
    normalized = path.as_posix()
    if normalized == "." or normalized.endswith("/"):
        raise ValueError(f"Bundle members must be files: {name!r}.")
    return normalized


def validate_release(release: str) -> str:
    """Validate the immutable artifact tag convention used in paths and URLs."""
    if not RELEASE_PATTERN.fullmatch(release):
        raise ValueError("Release tags must match artifacts-v<major>.<minor>.<patch>.")
    return release


def _optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Bundle JSON evidence is invalid: {path}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Bundle JSON evidence must be a non-empty object: {path}")
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _writestr(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BundleResult",
    "build_reproduction_bundle",
    "extract_bundle",
    "fetch_release_bundle",
    "materialize_training_data",
    "verify_bundle",
    "validate_release",
]
