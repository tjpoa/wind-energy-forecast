"""Fail-closed validation for the tracked release catalogue.

The catalogue is a publication control, not a provenance claim by itself.  A
release can only be marked public when each source component has explicit
redistribution evidence.  Legacy catalogues remain readable only when they
are blocked; an old ``approved=true`` entry is never grandfathered in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from .manifest_validation import validate_v1_source_contract
from .paths import project_root
from .v1_contracts import V1ContractError, contract_sha256, load_processed_contract

CATALOG_SCHEMA_V1 = "wind_forecast.release_catalog.v1"
CATALOG_SCHEMA_V2 = "wind_forecast.release_catalog.v2"
CATALOG_SCHEMA_V3 = "wind_forecast.release_catalog.v3"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_PATTERN = re.compile(r"^artifacts-v\d+\.\d+\.\d+$")
_AUTHORIZATION_KINDS = {"public_license", "written_permission"}


class ReleaseCatalogError(ValueError):
    """Raised when the release catalogue cannot support a safe decision."""


def load_release_catalog(path: str | Path) -> dict[str, Any]:
    """Load and validate a release catalogue without approving any release."""
    catalog_path = Path(path)
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ReleaseCatalogError(
            f"Could not read release catalog {catalog_path}: {exc}"
        ) from exc
    validate_release_catalog(payload)
    return payload


def validate_release_catalog(catalog: Mapping[str, Any]) -> None:
    """Validate all catalogue entries, failing closed on unsupported fields."""
    if not isinstance(catalog, Mapping):
        raise ReleaseCatalogError("Release catalog must be a JSON object.")
    schema_version = catalog.get("schema_version")
    if not isinstance(schema_version, str) or schema_version not in {
        CATALOG_SCHEMA_V1,
        CATALOG_SCHEMA_V2,
        CATALOG_SCHEMA_V3,
    }:
        raise ReleaseCatalogError(
            "Unsupported release catalog schema version: "
            f"{schema_version!r}."
        )
    releases = catalog.get("releases")
    if not isinstance(releases, Mapping) or not releases:
        raise ReleaseCatalogError("Release catalog must declare releases.")
    for release, entry in releases.items():
        if not isinstance(release, str) or not _RELEASE_PATTERN.fullmatch(release):
            raise ReleaseCatalogError(f"Invalid release identifier: {release!r}.")
        _validate_release_entry(
            release,
            entry,
            schema_version=str(schema_version),
        )


def validate_release_entry(
    release: str,
    entry: Mapping[str, Any],
    *,
    schema_version: str = CATALOG_SCHEMA_V3,
) -> None:
    """Validate one release entry independently of the surrounding file."""
    _validate_release_entry(release, entry, schema_version=schema_version)


def require_release_approved(
    catalog: Mapping[str, Any], release: str
) -> Mapping[str, Any]:
    """Return an explicitly approved entry or raise before publication work."""
    validate_release_catalog(catalog)
    entry = catalog["releases"].get(release)
    if entry is None:
        raise ReleaseCatalogError(f"Release is not declared in the catalog: {release}.")
    if entry.get("redistribution", {}).get("approved") is not True:
        raise ReleaseCatalogError(
            f"Release {release} is not approved for redistribution; publication is blocked."
        )
    return entry


def _validate_release_entry(
    release: str,
    entry: Any,
    *,
    schema_version: str,
) -> None:
    if not isinstance(entry, Mapping):
        raise ReleaseCatalogError(f"Release entry is not an object: {release}.")
    bundle_sha256 = entry.get("bundle_sha256")
    if bundle_sha256 is not None and (
        not isinstance(bundle_sha256, str)
        or not _SHA256_PATTERN.fullmatch(bundle_sha256)
    ):
        raise ReleaseCatalogError(
            f"Release {release} has an invalid bundle_sha256 value."
        )
    redistribution = entry.get("redistribution")
    if not isinstance(redistribution, Mapping):
        raise ReleaseCatalogError(
            f"Release {release} must declare a redistribution object."
        )
    approved = redistribution.get("approved")
    if not isinstance(approved, bool):
        raise ReleaseCatalogError(
            f"Release {release} redistribution.approved must be boolean."
        )

    if schema_version == CATALOG_SCHEMA_V1:
        if approved:
            raise ReleaseCatalogError(
                f"Legacy catalog entry {release} cannot approve redistribution "
                "without explicit component evidence."
            )
        return

    if schema_version == CATALOG_SCHEMA_V2:
        if approved:
            raise ReleaseCatalogError(
                f"Legacy catalog entry {release} cannot approve redistribution."
            )
        return

    source_contract = _validate_contract_reference(release, entry, "source_contract")
    _validate_contract_reference(release, entry, "processed_contract")

    classification = redistribution.get("classification")
    status = redistribution.get("status")
    required_components = redistribution.get("required_components")
    evidence = redistribution.get("authorization_evidence")
    if classification not in {"internal", "public"}:
        raise ReleaseCatalogError(
            f"Release {release} must declare classification internal or public."
        )
    if not isinstance(status, str) or not status.strip():
        raise ReleaseCatalogError(f"Release {release} must declare a status.")
    components = _required_components(release, required_components)
    if not isinstance(evidence, list):
        raise ReleaseCatalogError(
            f"Release {release} authorization_evidence must be a list."
        )

    if not approved:
        if classification != "internal":
            raise ReleaseCatalogError(
                f"Blocked release {release} must be classified as internal."
            )
        if status != "blocked_provenance_incomplete":
            raise ReleaseCatalogError(
                f"Blocked release {release} must use status="
                "'blocked_provenance_incomplete'."
            )
        if evidence:
            raise ReleaseCatalogError(
                f"Blocked release {release} must not contain authorization evidence."
            )
        return

    if classification != "public" or status != "approved_for_redistribution":
        raise ReleaseCatalogError(
            f"Approved release {release} must be public and use status="
            "'approved_for_redistribution'."
        )
    # A public, authorized candidate may be built once before its deterministic
    # bundle digest is recorded.  Fetch/publication still requires the digest.
    if len(evidence) != len(components):
        raise ReleaseCatalogError(
            f"Approved release {release} must provide one authorization record "
            "per required component."
        )
    seen: set[str] = set()
    for item in evidence:
        _validate_authorization_record(
            release, item, components, seen, source_contract["sha256"]
        )
    if seen != set(components):
        missing = sorted(set(components) - seen)
        raise ReleaseCatalogError(
            f"Approved release {release} is missing authorization for: {missing}."
        )


def _required_components(release: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ReleaseCatalogError(
            f"Release {release} required_components must be a non-empty list."
        )
    components = tuple(value)
    if len(set(components)) != len(components):
        raise ReleaseCatalogError(
            f"Release {release} required_components must be unique."
        )
    return components


def _validate_authorization_record(
    release: str,
    item: Any,
    components: Sequence[str],
    seen: set[str],
    source_contract_sha256: str,
) -> None:
    if not isinstance(item, Mapping):
        raise ReleaseCatalogError(
            f"Release {release} authorization records must be objects."
        )
    component = item.get("component")
    if (
        not isinstance(component, str)
        or component not in components
        or component in seen
    ):
        raise ReleaseCatalogError(
            f"Release {release} has an invalid or duplicate authorization component: "
            f"{component!r}."
        )
    seen.add(str(component))
    if item.get("source_contract_sha256") != source_contract_sha256:
        raise ReleaseCatalogError(
            f"Release {release} authorization for {component} must cover the "
            "exact source contract hash."
        )
    for field in ("provider", "source_identifier", "license", "attribution"):
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ReleaseCatalogError(
                f"Release {release} authorization for {component} requires {field}."
            )
    kind = item.get("authorization_kind")
    if not isinstance(kind, str) or kind not in _AUTHORIZATION_KINDS:
        raise ReleaseCatalogError(
            f"Release {release} authorization for {component} has an unsupported kind."
        )
    reference = item.get("authorization_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ReleaseCatalogError(
            f"Release {release} authorization for {component} requires a reference."
        )
    if kind == "public_license" and not reference.startswith("https://"):
        raise ReleaseCatalogError(
            f"Public-license authorization for {component} must reference HTTPS."
        )
    scope = item.get("authorization_scope")
    if not isinstance(scope, list) or "redistribution" not in scope:
        raise ReleaseCatalogError(
            f"Release {release} authorization for {component} must include the "
            "redistribution scope."
        )
    if item.get("redistribution_permitted") is not True:
        raise ReleaseCatalogError(
            f"Release {release} authorization for {component} must explicitly "
            "set redistribution_permitted=true."
        )
    verified_at = item.get("verified_at_utc")
    if not isinstance(verified_at, str):
        raise ReleaseCatalogError(
            f"Release {release} authorization for {component} requires "
            "verified_at_utc."
        )
    try:
        parsed = datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseCatalogError(
            f"Release {release} authorization for {component} has an invalid "
            "verified_at_utc."
        ) from exc
    if parsed.tzinfo is None:
        raise ReleaseCatalogError(
            f"Release {release} authorization for {component} requires a "
            "timezone-aware verified_at_utc."
        )
    receipt_sha256 = item.get("receipt_sha256")
    if kind == "written_permission":
        if not isinstance(receipt_sha256, str) or not _SHA256_PATTERN.fullmatch(
            receipt_sha256
        ):
            raise ReleaseCatalogError(
                f"Written authorization for {component} requires a receipt_sha256."
            )
    elif receipt_sha256 is not None and (
        not isinstance(receipt_sha256, str)
        or not _SHA256_PATTERN.fullmatch(receipt_sha256)
    ):
        raise ReleaseCatalogError(
            f"Release {release} authorization for {component} has an invalid "
            "receipt_sha256."
        )


def _validate_contract_reference(
    release: str, entry: Mapping[str, Any], field: str
) -> Mapping[str, str]:
    reference = entry.get(field)
    if not isinstance(reference, Mapping):
        raise ReleaseCatalogError(f"Release {release} must declare {field}.")
    path = reference.get("path")
    digest = reference.get("sha256")
    if (
        not isinstance(path, str)
        or not path.strip()
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise ReleaseCatalogError(f"Release {release} has an unsafe {field}.path.")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ReleaseCatalogError(f"Release {release} has an invalid {field}.sha256.")
    return {"path": path, "sha256": digest}


def validate_release_contract_binding(
    entry: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    require_release_provenance: bool = True,
) -> None:
    """Verify catalog contract files and their source/processed linkage.

    Release builds use full source integrity/provenance. Fetches use metadata
    only because the raw snapshot is intentionally not distributed locally.
    """
    root = Path(repository_root or project_root()).resolve()
    source = _resolve_contract_file(entry, "source_contract", root)
    processed = _resolve_contract_file(entry, "processed_contract", root)
    try:
        source_result = validate_v1_source_contract(
            mode="release" if require_release_provenance else "metadata",
            manifest_path=source[0],
            repository_root=root,
        )
        processed_payload = load_processed_contract(
            processed[0], repository_root=root, verify_dataset=False
        )
    except (OSError, ValueError, V1ContractError) as exc:
        raise ReleaseCatalogError(f"Release contract validation failed: {exc}") from exc
    if source_result.manifest_path.relative_to(root).as_posix() != processed_payload[
        "source_contract_path"
    ]:
        raise ReleaseCatalogError("Processed contract is linked to another source contract.")
    if processed_payload["source_contract_sha256"] != source[1]:
        raise ReleaseCatalogError("Processed contract source hash differs from catalog.")


def _resolve_contract_file(
    entry: Mapping[str, Any], field: str, root: Path
) -> tuple[Path, str]:
    reference = entry.get(field)
    if not isinstance(reference, Mapping):
        raise ReleaseCatalogError(f"Release must declare {field}.")
    path = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise ReleaseCatalogError(f"Release {field} reference is incomplete.")
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReleaseCatalogError(f"Release {field} path escapes repository root.") from exc
    if not candidate.is_file():
        raise ReleaseCatalogError(f"Release {field} file is missing: {candidate}.")
    try:
        actual = contract_sha256(candidate)
    except OSError as exc:
        raise ReleaseCatalogError(f"Could not hash release {field} file: {exc}.") from exc
    if actual != digest:
        raise ReleaseCatalogError(f"Release {field} hash does not match catalog.")
    return candidate, digest


__all__ = [
    "CATALOG_SCHEMA_V1",
    "CATALOG_SCHEMA_V2",
    "CATALOG_SCHEMA_V3",
    "ReleaseCatalogError",
    "load_release_catalog",
    "require_release_approved",
    "validate_release_catalog",
    "validate_release_contract_binding",
    "validate_release_entry",
]
