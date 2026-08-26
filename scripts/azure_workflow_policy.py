"""Pure validation rules shared by the protected Azure workflows.

This module deliberately has no Azure, GitHub, Docker, or Terraform client
dependencies. The workflows provide JSON snapshots and environment values;
the functions below validate those snapshots without logging their contents.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import re
from typing import Any


class WorkflowPolicyError(ValueError):
    """Raised when a protected workflow input or evidence record is invalid."""


RELEASE_MANIFEST_SCHEMA = "wind_forecast.production_release_manifest.v2"
ROLLBACK_MANIFEST_SCHEMA = "wind_forecast.production_rollback_manifest.v1"
RECEIPT_SCHEMA = "wind_forecast.production_receipt.v1"

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[1-9][0-9]*$")
_IMAGE = re.compile(
    r"^(?P<registry>[^/\s]+)/(?P<repository>[^@\s]+)@sha256:(?P<digest>[0-9a-f]{64})$"
)


def require_release_enabled(value: object) -> None:
    """Fail closed unless the explicit release feature flag is true."""
    if str(value).strip().lower() != "true":
        raise WorkflowPolicyError(
            "PRODUCTION_RELEASE_ENABLED must be exactly true for this workflow."
        )


def missing_configuration(
    values: Mapping[str, object], required_names: Iterable[str]
) -> tuple[str, ...]:
    """Return required configuration names whose values are empty."""
    return tuple(
        name
        for name in required_names
        if values.get(name) is None or not str(values.get(name)).strip()
    )


def require_configuration(
    values: Mapping[str, object], required_names: Iterable[str]
) -> None:
    """Fail without exposing the value of any missing configuration."""
    missing = missing_configuration(values, required_names)
    if missing:
        raise WorkflowPolicyError(
            "Required protected workflow configuration is missing: "
            + ", ".join(missing)
        )


def require_run_id(value: object, *, name: str) -> str:
    """Validate a GitHub Actions numeric run identifier."""
    text = str(value).strip()
    if not _RUN_ID.fullmatch(text):
        raise WorkflowPolicyError(f"{name} must be a positive numeric run id.")
    return text


def require_commit_sha(value: object, *, name: str = "source_sha") -> str:
    """Validate a full Git commit SHA used by a release manifest."""
    text = str(value).strip().lower()
    if not _COMMIT_SHA.fullmatch(text):
        raise WorkflowPolicyError(f"{name} must be a full 40-character SHA.")
    return text


def parse_image_reference(value: object, *, name: str) -> tuple[str, str, str]:
    """Return registry, repository, and digest for one immutable image ref."""
    text = str(value).strip()
    match = _IMAGE.fullmatch(text)
    if match is None:
        raise WorkflowPolicyError(
            f"{name} must be registry/repository@sha256:<64 lowercase hex chars>."
        )
    return match.group("registry"), match.group("repository"), match.group("digest")


def validate_image_pair(
    api_image: object,
    frontend_image: object,
    *,
    expected_registry: str | None = None,
) -> dict[str, str]:
    """Validate the two expected repositories share one immutable registry."""
    api_registry, api_repository, api_digest = parse_image_reference(
        api_image, name="api_image"
    )
    frontend_registry, frontend_repository, frontend_digest = parse_image_reference(
        frontend_image, name="frontend_image"
    )
    if api_repository != "wind-forecast-api":
        raise WorkflowPolicyError("api_image must target wind-forecast-api.")
    if frontend_repository != "wind-forecast-frontend":
        raise WorkflowPolicyError(
            "frontend_image must target wind-forecast-frontend."
        )
    if api_registry != frontend_registry:
        raise WorkflowPolicyError("API and frontend images must use one registry.")
    if expected_registry is not None and api_registry != expected_registry:
        raise WorkflowPolicyError("Images do not use the configured registry.")
    return {
        "registry": api_registry,
        "api_image": f"{api_registry}/{api_repository}@sha256:{api_digest}",
        "frontend_image": (
            f"{frontend_registry}/{frontend_repository}@sha256:{frontend_digest}"
        ),
    }


def validate_release_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_source_sha: object | None = None,
    expected_release_run_id: object | None = None,
) -> dict[str, str]:
    """Validate and return the release identity fields used by later jobs."""
    _require_mapping(manifest, "release manifest")
    if manifest.get("schema_version") != RELEASE_MANIFEST_SCHEMA:
        raise WorkflowPolicyError("Unsupported release manifest schema version.")
    if manifest.get("manifest_type") != "release":
        raise WorkflowPolicyError("The artifact is not a release manifest.")

    source_sha = require_commit_sha(manifest.get("source_sha"), name="source_sha")
    if expected_source_sha is not None:
        expected = require_commit_sha(expected_source_sha, name="expected_source_sha")
        if source_sha != expected:
            raise WorkflowPolicyError("Release manifest source SHA does not match.")

    ci_run_id = require_run_id(manifest.get("ci_run_id"), name="ci_run_id")
    release_run_id = require_run_id(
        manifest.get("release_run_id"), name="release_run_id"
    )
    require_run_id(manifest.get("release_attempt"), name="release_attempt")
    if expected_release_run_id is not None:
        expected = require_run_id(
            expected_release_run_id, name="expected_release_run_id"
        )
        if release_run_id != expected:
            raise WorkflowPolicyError("Release manifest run id does not match.")

    images = validate_image_pair(manifest.get("api_image"), manifest.get("frontend_image"))
    for component in ("api", "frontend"):
        label = manifest.get(f"{component}_source_label")
        if label != source_sha:
            raise WorkflowPolicyError(
                f"{component}_source_label does not match source_sha."
            )
    return {
        "source_sha": source_sha,
        "ci_run_id": ci_run_id,
        "release_run_id": release_run_id,
        "api_image": images["api_image"],
        "frontend_image": images["frontend_image"],
    }


def validate_rollback_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_rollback_run_id: object | None = None,
) -> dict[str, str]:
    """Validate a rollback receipt derived from a registered release manifest."""
    _require_mapping(manifest, "rollback manifest")
    if manifest.get("schema_version") != ROLLBACK_MANIFEST_SCHEMA:
        raise WorkflowPolicyError("Unsupported rollback manifest schema version.")
    if manifest.get("manifest_type") != "rollback":
        raise WorkflowPolicyError("The artifact is not a rollback manifest.")

    rollback_run_id = require_run_id(
        manifest.get("rollback_run_id"), name="rollback_run_id"
    )
    if expected_rollback_run_id is not None:
        expected = require_run_id(
            expected_rollback_run_id, name="expected_rollback_run_id"
        )
        if rollback_run_id != expected:
            raise WorkflowPolicyError("Rollback manifest run id does not match.")
    request_sha = require_commit_sha(manifest.get("request_sha"), name="request_sha")
    source_sha = require_commit_sha(manifest.get("source_sha"), name="source_sha")
    ci_run_id = require_run_id(manifest.get("ci_run_id"), name="ci_run_id")
    release_run_id = require_run_id(
        manifest.get("release_run_id"), name="release_run_id"
    )
    images = validate_image_pair(manifest.get("api_image"), manifest.get("frontend_image"))
    return {
        "request_sha": request_sha,
        "source_sha": source_sha,
        "ci_run_id": ci_run_id,
        "release_run_id": release_run_id,
        "rollback_run_id": rollback_run_id,
        "api_image": images["api_image"],
        "frontend_image": images["frontend_image"],
    }


def forbidden_plan_changes(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return Terraform changes containing deletes or replacements."""
    _require_mapping(plan, "Terraform plan")
    changes = plan.get("resource_changes") or []
    if not isinstance(changes, Sequence) or isinstance(changes, (str, bytes)):
        raise WorkflowPolicyError("Terraform resource_changes must be an array.")

    forbidden: list[dict[str, Any]] = []
    for change in changes:
        if not isinstance(change, Mapping):
            raise WorkflowPolicyError("Terraform resource_changes entries must be objects.")
        change_payload = change.get("change")
        if not isinstance(change_payload, Mapping):
            raise WorkflowPolicyError("Terraform change entries must contain an object.")
        actions = change_payload.get("actions", [])
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise WorkflowPolicyError("Terraform change actions must be an array.")
        actions = [str(action) for action in actions]
        is_replacement = "create" in actions and "delete" in actions
        if "delete" in actions or "replace" in actions or is_replacement:
            forbidden.append(
                {"address": str(change.get("address", "<unknown>")), "actions": actions}
            )
    return forbidden


def validate_active_images(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, str]:
    """Require Azure's active container image refs to equal the manifest refs."""
    _require_mapping(expected, "expected images")
    _require_mapping(observed, "observed images")
    expected_pair = validate_image_pair(
        expected.get("api_image"), expected.get("frontend_image")
    )
    observed_pair = validate_image_pair(
        observed.get("api_image"), observed.get("frontend_image")
    )
    if expected_pair["api_image"] != observed_pair["api_image"]:
        raise WorkflowPolicyError("Active API image does not match the manifest.")
    if expected_pair["frontend_image"] != observed_pair["frontend_image"]:
        raise WorkflowPolicyError(
            "Active frontend image does not match the manifest."
        )
    return observed_pair


def build_receipt(
    *,
    operation: str,
    manifest: Mapping[str, Any],
    active_images: Mapping[str, Any],
    api_revision: object,
    frontend_revision: object,
    smoke_tests: Mapping[str, Any],
    terraform_post_plan_exit_code: int,
    workflow_run_id: object,
) -> dict[str, Any]:
    """Build a non-secret, evidence-oriented release or rollback receipt."""
    if operation == "release":
        identity = validate_release_manifest(manifest)
        manifest_type = RELEASE_MANIFEST_SCHEMA
        rollback_run_id = None
    elif operation == "rollback":
        identity = validate_rollback_manifest(manifest)
        manifest_type = ROLLBACK_MANIFEST_SCHEMA
        rollback_run_id = identity["rollback_run_id"]
    else:
        raise WorkflowPolicyError("operation must be release or rollback.")

    active = validate_active_images(
        {"api_image": identity["api_image"], "frontend_image": identity["frontend_image"]},
        active_images,
    )
    if not str(api_revision).strip() or not str(frontend_revision).strip():
        raise WorkflowPolicyError("Both active revision names are required.")
    if terraform_post_plan_exit_code != 0:
        raise WorkflowPolicyError("The post-deployment Terraform plan was not drift-free.")
    if not isinstance(smoke_tests, Mapping) or not smoke_tests:
        raise WorkflowPolicyError("At least one smoke-test result is required.")
    if any(value is not True for value in smoke_tests.values()):
        raise WorkflowPolicyError("Every smoke-test result must be true.")

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "operation": operation,
        "manifest_schema_version": manifest_type,
        "workflow_run_id": require_run_id(workflow_run_id, name="workflow_run_id"),
        "source_sha": identity["source_sha"],
        "ci_run_id": identity.get("ci_run_id"),
        "release_run_id": identity["release_run_id"],
        "rollback_run_id": rollback_run_id,
        "api_image": identity["api_image"],
        "frontend_image": identity["frontend_image"],
        "active_images": active,
        "revisions": {
            "api": str(api_revision),
            "frontend": str(frontend_revision),
        },
        "approval": {
            "environment": "production",
            "manual_required": True,
            "gate_passed": True,
            "mode": "maintainer_confirmation",
            "independent_human_review": "not_applicable_single_maintainer",
        },
        "smoke_tests": dict(smoke_tests),
        "terraform_post_deployment_plan_exit_code": terraform_post_plan_exit_code,
    }
    return receipt


def forbidden_tracked_paths(paths: Iterable[str]) -> list[str]:
    """Return tracked paths that must never enter the repository."""
    forbidden: list[str] = []
    for raw_path in paths:
        path = str(raw_path).replace("\\", "/")
        lower = path.lower()
        name = lower.rsplit("/", 1)[-1]
        if (
            name.endswith(".tfstate")
            or ".tfstate." in name
            or name.endswith(".tfplan")
            or (name.endswith(".tfvars") and not name.endswith(".tfvars.example"))
            or name.endswith((".pem", ".p12", ".pfx", ".jks", ".key", ".secret"))
            or any(
                marker in name
                for marker in (
                    "password",
                    "passwd",
                    "client_secret",
                    "client-secret",
                    "clientsecret",
                )
            )
        ):
            forbidden.append(path)
    return sorted(forbidden)


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise WorkflowPolicyError(f"{name} must be a JSON object.")


__all__ = [
    "RECEIPT_SCHEMA",
    "RELEASE_MANIFEST_SCHEMA",
    "ROLLBACK_MANIFEST_SCHEMA",
    "WorkflowPolicyError",
    "build_receipt",
    "forbidden_plan_changes",
    "forbidden_tracked_paths",
    "missing_configuration",
    "parse_image_reference",
    "require_commit_sha",
    "require_configuration",
    "require_release_enabled",
    "require_run_id",
    "validate_active_images",
    "validate_image_pair",
    "validate_release_manifest",
    "validate_rollback_manifest",
]
