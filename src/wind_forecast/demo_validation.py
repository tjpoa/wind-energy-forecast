"""Validate the immutable synthetic dashboard bundle without writing to it."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .manifests import validate_repo_relative_path
from .monitoring_projection import MonitoringProjectionService
from .performance import PerformanceService


DEMO_BUNDLE_SCHEMA = "wind_forecast.demo_bundle.v1"
VALIDATION_SCHEMA = "wind_forecast.demo_validation.v1"


class DemoBundleValidationError(RuntimeError):
    """Raised when a demo bundle cannot be verified and read safely."""


@dataclass(frozen=True)
class DemoValidationResult:
    """Sanitized result emitted by the local and cloud validation command."""

    bundle_version: str
    file_count: int
    performance_observation_count: int
    monitoring_state: str
    monitoring_report_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATION_SCHEMA,
            "status": "succeeded",
            "bundle_version": self.bundle_version,
            "file_count": self.file_count,
            "performance_observation_count": self.performance_observation_count,
            "monitoring_state": self.monitoring_state,
            "monitoring_report_id": self.monitoring_report_id,
        }


def validate_demo_bundle(bundle_root: str | Path) -> DemoValidationResult:
    """Verify one complete bundle and exercise its read-only API projections."""
    root = Path(bundle_root)
    if not root.is_dir():
        raise DemoBundleValidationError("Demo bundle directory is unavailable.")

    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise DemoBundleValidationError("Demo bundle manifest is invalid.") from exc

    if not isinstance(manifest, Mapping):
        raise DemoBundleValidationError("Demo bundle manifest must be an object.")
    if manifest.get("schema_version") != DEMO_BUNDLE_SCHEMA:
        raise DemoBundleValidationError("Unsupported demo bundle schema.")
    if manifest.get("evidence_type") != "deterministic_synthetic":
        raise DemoBundleValidationError("Demo evidence type is not synthetic.")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or source.get("credentials_required") is not False:
        raise DemoBundleValidationError("Demo bundle must not require credentials.")
    if source.get("network_requests") is not False:
        raise DemoBundleValidationError("Demo bundle must not require network access.")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise DemoBundleValidationError("Demo bundle file manifest is invalid.")
    declared: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise DemoBundleValidationError("Demo bundle file entry is invalid.")
        try:
            relative = validate_repo_relative_path(str(entry["path"]))
            size = int(entry["size"])
            checksum = str(entry["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DemoBundleValidationError("Demo bundle file entry is invalid.") from exc
        if relative == "manifest.json" or relative in declared:
            raise DemoBundleValidationError("Demo bundle file entries are not unique.")
        if size < 0 or len(checksum) != 64:
            raise DemoBundleValidationError("Demo bundle file metadata is invalid.")
        declared[relative] = {"size": size, "sha256": checksum}

    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "manifest.json"
    }
    if set(declared) != set(actual):
        raise DemoBundleValidationError("Demo bundle contents differ from its manifest.")
    for relative, entry in declared.items():
        path = actual[relative]
        if path.stat().st_size != entry["size"] or _sha256_file(path) != entry["sha256"]:
            raise DemoBundleValidationError("Demo bundle checksum verification failed.")

    pipeline_path = root / "pipeline" / "run.json"
    try:
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise DemoBundleValidationError("Demo pipeline receipt is invalid.") from exc
    outputs = pipeline.get("outputs") if isinstance(pipeline, Mapping) else None
    if not isinstance(outputs, list) or any(
        not isinstance(item, str) or not (root / item).is_file() for item in outputs
    ):
        raise DemoBundleValidationError("Demo pipeline outputs are incomplete.")

    try:
        performance = PerformanceService.from_directory(root / "performance").get_performance()
        monitoring = MonitoringProjectionService(root / "monitoring").latest(
            now_utc=datetime.now(timezone.utc)
        )
    except Exception as exc:
        raise DemoBundleValidationError(
            "Demo performance or monitoring evidence is unavailable."
        ) from exc

    report = monitoring.get("report")
    report_id = report.get("report_id") if isinstance(report, Mapping) else None
    return DemoValidationResult(
        bundle_version=str(manifest.get("bundle_version", "")),
        file_count=len(declared),
        performance_observation_count=performance.observation_count,
        monitoring_state=str(monitoring.get("state", "")),
        monitoring_report_id=report_id if isinstance(report_id, str) else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_demo_bundle(args.bundle_root)
    except (DemoBundleValidationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": VALIDATION_SCHEMA,
                    "status": "failed",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEMO_BUNDLE_SCHEMA",
    "DemoBundleValidationError",
    "DemoValidationResult",
    "VALIDATION_SCHEMA",
    "main",
    "validate_demo_bundle",
]
