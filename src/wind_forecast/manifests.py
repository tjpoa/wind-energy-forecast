"""Dataset manifest helpers for versioned data contracts.

The helpers in this module are intentionally lightweight and side-effect free:
they do not read datasets, create directories, or perform network access during
import. They provide deterministic JSON serialization for future v2 manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


SCHEMA_VERSION = "wind_forecast.dataset_manifest.v1"

_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_SECRET_VALUE_MARKERS = (
    "api_key=",
    "apikey=",
    "authorization:",
    "bearer ",
    "password=",
    "secret=",
    "token=",
)


@dataclass(frozen=True)
class DatasetManifest:
    """Metadata describing a versioned dataset snapshot or planned contract."""

    dataset_version: str
    dataset_role: str
    schema_version: str = SCHEMA_VERSION
    provider: str | None = None
    source_identifier: str | None = None
    source_endpoint: str | None = None
    retrieval_timestamp: str | None = None
    coverage_start: str | None = None
    coverage_end: str | None = None
    temporal_granularity: str | None = None
    units: Mapping[str, str] = field(default_factory=dict)
    timezone: str | None = None
    geographic_coverage: Mapping[str, Any] = field(default_factory=dict)
    station_ids: tuple[str, ...] = field(default_factory=tuple)
    coordinates: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    raw_file_paths: tuple[str, ...] = field(default_factory=tuple)
    sha256_checksums: Mapping[str, str] = field(default_factory=dict)
    row_count: int | None = None
    column_count: int | None = None
    preprocessing_version: str | None = None
    known_warnings: tuple[str, ...] = field(default_factory=tuple)
    license: str | None = None
    attribution: str | None = None
    status: str | None = None
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported dataset manifest schema version: {self.schema_version!r}."
            )

        raw_file_paths = tuple(
            validate_repo_relative_path(path) for path in self.raw_file_paths
        )
        checksums = {
            validate_repo_relative_path(path): checksum
            for path, checksum in self.sha256_checksums.items()
        }

        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(
            self, "geographic_coverage", dict(self.geographic_coverage)
        )
        object.__setattr__(self, "station_ids", tuple(self.station_ids))
        object.__setattr__(
            self, "coordinates", tuple(dict(item) for item in self.coordinates)
        )
        object.__setattr__(self, "raw_file_paths", raw_file_paths)
        object.__setattr__(self, "sha256_checksums", checksums)
        object.__setattr__(self, "known_warnings", tuple(self.known_warnings))
        object.__setattr__(self, "extra_metadata", dict(self.extra_metadata))

        _validate_no_secret_markers(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready manifest dictionary."""
        return {
            "attribution": self.attribution,
            "column_count": self.column_count,
            "coordinates": [dict(item) for item in self.coordinates],
            "coverage_end": self.coverage_end,
            "coverage_start": self.coverage_start,
            "dataset_role": self.dataset_role,
            "dataset_version": self.dataset_version,
            "extra_metadata": dict(self.extra_metadata),
            "geographic_coverage": dict(self.geographic_coverage),
            "known_warnings": list(self.known_warnings),
            "license": self.license,
            "preprocessing_version": self.preprocessing_version,
            "provider": self.provider,
            "raw_file_paths": list(self.raw_file_paths),
            "retrieval_timestamp": self.retrieval_timestamp,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "sha256_checksums": dict(self.sha256_checksums),
            "source_endpoint": self.source_endpoint,
            "source_identifier": self.source_identifier,
            "station_ids": list(self.station_ids),
            "status": self.status,
            "temporal_granularity": self.temporal_granularity,
            "timezone": self.timezone,
            "units": dict(self.units),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetManifest":
        """Create a manifest from a JSON-decoded dictionary."""
        _validate_no_secret_markers(data)
        return cls(
            attribution=data.get("attribution"),
            column_count=data.get("column_count"),
            coordinates=tuple(data.get("coordinates") or ()),
            coverage_end=data.get("coverage_end"),
            coverage_start=data.get("coverage_start"),
            dataset_role=data["dataset_role"],
            dataset_version=data["dataset_version"],
            extra_metadata=data.get("extra_metadata") or {},
            geographic_coverage=data.get("geographic_coverage") or {},
            known_warnings=tuple(data.get("known_warnings") or ()),
            license=data.get("license"),
            preprocessing_version=data.get("preprocessing_version"),
            provider=data.get("provider"),
            raw_file_paths=tuple(data.get("raw_file_paths") or ()),
            retrieval_timestamp=data.get("retrieval_timestamp"),
            row_count=data.get("row_count"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            sha256_checksums=data.get("sha256_checksums") or {},
            source_endpoint=data.get("source_endpoint"),
            source_identifier=data.get("source_identifier"),
            station_ids=tuple(data.get("station_ids") or ()),
            status=data.get("status"),
            temporal_granularity=data.get("temporal_granularity"),
            timezone=data.get("timezone"),
            units=data.get("units") or {},
        )


def validate_repo_relative_path(path: str | Path) -> str:
    """Validate and normalize a repository-relative path for manifests."""
    raw = str(path)
    if not raw or raw.strip() != raw:
        raise ValueError("Manifest paths must be non-empty and trimmed.")

    windows_path = PureWindowsPath(raw)
    normalized = raw.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    if (
        windows_path.is_absolute()
        or windows_path.drive
        or posix_path.is_absolute()
        or normalized.startswith("/")
    ):
        raise ValueError(f"Manifest path must be repository-relative: {raw!r}.")
    if ".." in posix_path.parts:
        raise ValueError(f"Manifest path must not contain '..': {raw!r}.")

    return posix_path.as_posix()


def manifest_to_json(manifest: DatasetManifest) -> str:
    """Serialize a manifest as deterministic JSON with a trailing newline."""
    return json.dumps(
        manifest.to_dict(),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def manifest_from_json(text: str) -> DatasetManifest:
    """Deserialize a manifest from JSON text."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Dataset manifest JSON must decode to an object.")
    return DatasetManifest.from_dict(data)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum for an explicitly supplied file path."""
    digest = sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_no_secret_markers(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            if any(part in key_text for part in _SECRET_KEY_PARTS):
                raise ValueError(f"Manifest field must not contain secrets: {path}.{key}.")
            _validate_no_secret_markers(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_no_secret_markers(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _SECRET_VALUE_MARKERS):
            raise ValueError(f"Manifest value appears to contain a secret: {path}.")


__all__ = [
    "SCHEMA_VERSION",
    "DatasetManifest",
    "manifest_from_json",
    "manifest_to_json",
    "sha256_file",
    "validate_repo_relative_path",
]
