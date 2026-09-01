"""Pure validation rules shared by the protected Azure workflows.

This module deliberately has no Azure, GitHub, Docker, or Terraform client
dependencies. The workflows provide JSON snapshots and environment values;
the functions below validate those snapshots without logging their contents.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any


class WorkflowPolicyError(ValueError):
    """Raised when a protected workflow input or evidence record is invalid."""


RELEASE_MANIFEST_SCHEMA = "wind_forecast.production_release_manifest.v2"
ROLLBACK_MANIFEST_SCHEMA = "wind_forecast.production_rollback_manifest.v1"
RECEIPT_SCHEMA = "wind_forecast.production_receipt.v1"
READINESS_COMPONENT_SCHEMA = "wind_forecast.azure_readiness_component.v1"
READINESS_SCHEMA = "wind_forecast.azure_external_readiness.v1"

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[1-9][0-9]*$")
_IMAGE = re.compile(
    r"^(?P<registry>[^/\s]+)/(?P<repository>[^@\s]+)@sha256:(?P<digest>[0-9a-f]{64})$"
)

READINESS_IDENTITIES = {
    "release-publish": "publisher",
    "release-plan": "planner",
    "release-deploy": "deployer",
    "foundation-plan": "planner",
    "foundation-apply": "deployer",
    "rollback-plan": "planner",
    "rollback-deploy": "deployer",
}

READINESS_ROLE_REQUIREMENTS = {
    "publisher": (
        ("Reader", "workload_resource_group", None),
        ("AcrPush", "container_registry", None),
    ),
    "planner": (
        ("Reader", "workload_resource_group", None),
        ("Reader", "state_storage_account", None),
        ("Storage Blob Data Contributor", "state_storage_account", None),
    ),
    "deployer": (
        ("Contributor", "workload_resource_group", None),
        ("RBAC Administrator", "workload_resource_group", "acr_roles_only"),
        ("Reader", "state_storage_account", None),
        ("Storage Blob Data Contributor", "state_storage_account", None),
    ),
}

READINESS_REQUIRED_UPSTREAM = {
    "foundation-plan": "bootstrap.tfstate",
    "foundation-apply": "bootstrap.tfstate",
    "release-plan": "foundation.tfstate",
    "release-deploy": "foundation.tfstate",
    "rollback-plan": "foundation.tfstate",
    "rollback-deploy": "foundation.tfstate",
}

_SENSITIVE_MARKERS = (
    "token",
    "secret",
    "password",
    "credential",
    "client_id",
    "tenant_id",
    "subscription_id",
    "authorization",
    "access_key",
    "sas",
)
_JWT = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
_SENSITIVE_VALUE = re.compile(
    r"(?:bearer\s+|(?:access[_-])?token\s*[:=]|(?:client[_-])?secret\s*[:=]|"
    r"password\s*[:=]|credential\s*[:=]|(?:shared[_-])?key\s*[:=]|sas\s*[:=])",
    re.IGNORECASE,
)
_READINESS_BINDING_FIELDS = {
    "repository",
    "workflow",
    "source_sha",
    "ref",
    "workflow_run_id",
    "run_attempt",
}


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


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return the canonical representation used for receipt identities."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_id(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 content identifier for a JSON object."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def assert_no_sensitive_evidence(value: Any, *, path: str = "evidence") -> None:
    """Reject evidence that could accidentally contain credentials or tokens."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            if any(marker in key_text for marker in _SENSITIVE_MARKERS):
                raise WorkflowPolicyError(
                    f"Sensitive field is not permitted in readiness evidence: {path}.{key}."
                )
            assert_no_sensitive_evidence(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_no_sensitive_evidence(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        stripped = value.strip()
        if _SENSITIVE_VALUE.search(stripped) or stripped.startswith(("ghp_", "github_pat_")):
            raise WorkflowPolicyError(f"Credential-like value is not permitted in {path}.")
        if stripped.startswith("eyJ") and _JWT.fullmatch(stripped):
            raise WorkflowPolicyError(f"JWT-like value is not permitted in {path}.")


def _expected_subject(identity: str, *, repository: str, branch: str, environment: str) -> str:
    if identity == "deployer":
        return f"repo:{repository}:environment:{environment}"
    return f"repo:{repository}:ref:refs/heads/{branch}"


def _check(check_name: str, passed: bool, reason: str | None = None, **evidence: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": bool(passed), "evidence": evidence}
    if reason is not None:
        result["reason_code"] = reason
    return result


def evaluate_readiness(
    *,
    mode: str,
    evidence: Mapping[str, Any],
    repository: str,
    branch: str = "master",
    environment: str = "production",
) -> dict[str, Any]:
    """Evaluate one sanitized live-evidence snapshot without side effects."""
    if mode not in READINESS_IDENTITIES:
        raise WorkflowPolicyError(f"Unsupported readiness mode: {mode}.")
    _require_mapping(evidence, "readiness evidence")
    assert_no_sensitive_evidence(evidence)
    identity = READINESS_IDENTITIES[mode]
    checks: dict[str, dict[str, Any]] = {}
    reason_codes: set[str] = set()

    configuration = evidence.get("configuration")
    if not isinstance(configuration, Mapping):
        checks["configuration"] = _check("configuration", False, "CONFIGURATION_EVIDENCE_MISSING")
    else:
        release_enabled = configuration.get("release_enabled") is True
        missing = configuration.get("missing", [])
        missing_ok = isinstance(missing, Sequence) and not isinstance(missing, (str, bytes)) and not missing
        passed = release_enabled and missing_ok
        reason = "RELEASE_DISABLED" if not release_enabled else None
        if not missing_ok:
            reason = "CONFIGURATION_MISSING"
        checks["configuration"] = _check(
            "configuration", passed, reason, release_enabled=release_enabled, missing=list(missing) if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes)) else [],
        )

    oidc = evidence.get("oidc")
    if not isinstance(oidc, Mapping):
        checks["oidc"] = _check("oidc", False, "OIDC_EVIDENCE_MISSING")
    else:
        claims = oidc.get("claims") if isinstance(oidc.get("claims"), Mapping) else {}
        expected = _expected_subject(identity, repository=repository, branch=branch, environment=environment)
        audience = claims.get("audience")
        if isinstance(audience, Sequence) and not isinstance(audience, (str, bytes)):
            audience_ok = "api://AzureADTokenExchange" in audience
        else:
            audience_ok = audience == "api://AzureADTokenExchange"
        claims_ok = (
            claims.get("issuer") == "https://token.actions.githubusercontent.com"
            and audience_ok
            and claims.get("subject") == expected
        )
        account_ok = oidc.get("account_match") is True
        login_ok = oidc.get("login_succeeded") is True
        passed = login_ok and account_ok and claims_ok
        reason = None
        if not login_ok:
            reason = "OIDC_LOGIN_FAILED"
        elif not account_ok:
            reason = "AZURE_ACCOUNT_MISMATCH"
        elif not claims_ok:
            reason = "OIDC_CLAIMS_MISMATCH"
        checks["oidc"] = _check(
            "oidc", passed, reason,
            identity=identity,
            expected_subject=expected,
            observed_subject=claims.get("subject"),
            issuer=claims.get("issuer"),
            audience=audience,
            login_succeeded=login_ok,
            account_match=account_ok,
        )

    if identity in {"planner", "deployer"}:
        state = evidence.get("state")
        state_ok = isinstance(state, Mapping)
        account = state.get("storage_account") if state_ok else None
        container = state.get("container") if state_ok else None
        blobs = state.get("blobs") if state_ok else None
        account_ok = isinstance(account, Mapping) and account.get("exists") is True and account.get("https_only") is True and account.get("shared_key_enabled") is False and account.get("min_tls_version") == "TLS_1_2" and account.get("versioning_enabled") is True
        container_ok = isinstance(container, Mapping) and container.get("probe_ok") is True and container.get("exists") is True and container.get("private") is True
        upstream_name = READINESS_REQUIRED_UPSTREAM[mode]
        upstream = blobs.get(upstream_name) if isinstance(blobs, Mapping) else None
        upstream_ok = isinstance(upstream, Mapping) and upstream.get("probe_ok") is True and upstream.get("exists") is True and upstream.get("lease_state") in {None, "available", "unlocked"}
        target_name = "foundation.tfstate" if mode.startswith("foundation") else "production.tfstate"
        target = blobs.get(target_name) if isinstance(blobs, Mapping) else None
        target_ok = (
            isinstance(target, Mapping)
            and target.get("probe_ok") is True
            and (
                target.get("exists") is False
                or target.get("lease_state") in {None, "available", "unlocked"}
            )
        )
        passed = account_ok and container_ok and upstream_ok and target_ok
        reason = None
        if not account_ok:
            reason = "STATE_STORAGE_NOT_READY"
        elif not container_ok:
            reason = "STATE_CONTAINER_NOT_READY"
        elif not upstream_ok:
            reason = "STATE_UPSTREAM_MISSING_OR_LOCKED"
        elif not target_ok:
            reason = "STATE_TARGET_LOCKED"
        checks["remote_state"] = _check(
            "remote_state", passed, reason,
            storage_account_ready=account_ok,
            container_ready=container_ok,
            upstream_key=upstream_name,
            upstream_ready=upstream_ok,
            target_key=target_name,
            target_ready=target_ok,
            target_key_may_be_absent=True,
        )

    permissions = evidence.get("permissions")
    assignments = permissions.get("assignments") if isinstance(permissions, Mapping) else None
    normalized: set[tuple[str, str, str | None]] = set()
    if isinstance(assignments, Sequence) and not isinstance(assignments, (str, bytes)):
        for item in assignments:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role", ""))
            scope = str(item.get("scope", ""))
            condition = item.get("condition")
            normalized.add((role, scope, str(condition) if condition is not None else None))
    required = READINESS_ROLE_REQUIREMENTS[identity]
    missing_roles = [
        {"role": role, "scope": scope, "condition": condition}
        for role, scope, condition in required
        if (role, scope, condition) not in normalized
    ]
    required_set = set(required)
    unexpected_roles = [
        {"role": role, "scope": scope, "condition": condition}
        for role, scope, condition in sorted(
            normalized - required_set,
            key=lambda item: tuple("" if value is None else str(value) for value in item),
        )
    ]
    permissions_ok = (
        isinstance(permissions, Mapping)
        and permissions.get("probe_ok") is True
        and not missing_roles
        and not unexpected_roles
    )
    permission_reason = None
    if missing_roles:
        permission_reason = "AZURE_PERMISSION_MISSING"
    elif unexpected_roles:
        permission_reason = "AZURE_PERMISSION_TOO_BROAD"
    checks["permissions"] = _check(
        "permissions", permissions_ok, permission_reason,
        required=list(required), missing=missing_roles, unexpected=unexpected_roles,
    )

    if identity == "deployer":
        github = evidence.get("github")
        environment_record = github.get("environment") if isinstance(github, Mapping) else None
        if not isinstance(environment_record, Mapping):
            checks["github_environment"] = _check("github_environment", False, "GITHUB_ENVIRONMENT_MISSING")
        else:
            reviewers_raw = environment_record.get("required_reviewer_count", 0)
            reviewers = reviewers_raw if isinstance(reviewers_raw, int) and not isinstance(reviewers_raw, bool) else 0
            branch_policy = environment_record.get("deployment_branch_policy")
            policy_ok = isinstance(branch_policy, Mapping) and (
                (branch_policy.get("protected_branches") is True)
                or (
                    branch_policy.get("custom_branch_policies") is True
                    and isinstance(branch_policy.get("custom_branches"), Sequence)
                    and not isinstance(branch_policy.get("custom_branches"), (str, bytes))
                    and branch in branch_policy.get("custom_branches", [])
                )
            )
            passed = environment_record.get("probe_ok") is True and environment_record.get("name") == environment and reviewers >= 1 and policy_ok
            reason = None
            if environment_record.get("probe_ok") is not True or environment_record.get("name") != environment:
                reason = "GITHUB_ENVIRONMENT_MISSING"
            elif reviewers < 1:
                reason = "GITHUB_ENVIRONMENT_UNPROTECTED"
            elif not policy_ok:
                reason = "GITHUB_BRANCH_POLICY_MISMATCH"
            checks["github_environment"] = _check(
                "github_environment", passed, reason,
                name=environment_record.get("name"),
                required_reviewer_count=reviewers,
                deployment_branch_policy=branch_policy,
            )

    for check in checks.values():
        reason = check.get("reason_code")
        if reason:
            reason_codes.add(str(reason))
    return {
        "identity": identity,
        "mode": mode,
        "decision": "GO" if not reason_codes and all(item.get("passed") is True for item in checks.values()) else "NO_GO",
        "reason_codes": sorted(reason_codes),
        "checks": checks,
    }


def build_readiness_component_receipt(
    *,
    mode: str,
    evidence: Mapping[str, Any],
    repository: str,
    workflow: str,
    source_sha: str,
    ref: str,
    workflow_run_id: str,
    run_attempt: str,
    observed_at_utc: str | None = None,
    branch: str = "master",
    environment: str = "production",
) -> dict[str, Any]:
    """Build one immutable, sanitized readiness component receipt."""
    _require_readiness_binding(
        {
            "repository": repository,
            "workflow": workflow,
            "source_sha": source_sha,
            "ref": ref,
            "workflow_run_id": workflow_run_id,
            "run_attempt": run_attempt,
        }
    )
    require_commit_sha(source_sha, name="source_sha")
    require_run_id(workflow_run_id, name="workflow_run_id")
    require_run_id(run_attempt, name="run_attempt")
    evaluation = evaluate_readiness(
        mode=mode, evidence=evidence, repository=repository, branch=branch, environment=environment
    )
    body = {
        "schema_version": READINESS_COMPONENT_SCHEMA,
        "mode": mode,
        "identity": evaluation["identity"],
        "decision": evaluation["decision"],
        "reason_codes": evaluation["reason_codes"],
        "binding": {
            "repository": repository,
            "workflow": workflow,
            "source_sha": source_sha.lower(),
            "ref": ref,
            "workflow_run_id": workflow_run_id,
            "run_attempt": run_attempt,
        },
        "observed_at_utc": observed_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "checks": evaluation["checks"],
    }
    assert_no_sensitive_evidence(body)
    return {"receipt_id": content_id(body), **body}


def validate_readiness_component_receipt(
    receipt: Mapping[str, Any], *, expected_mode: str | None = None, expected_binding: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Strictly validate one component receipt and its content address."""
    _require_mapping(receipt, "readiness component receipt")
    required = {"receipt_id", "schema_version", "mode", "identity", "decision", "reason_codes", "binding", "observed_at_utc", "checks"}
    if set(receipt) != required:
        raise WorkflowPolicyError("Readiness component receipt fields are invalid.")
    if receipt["schema_version"] != READINESS_COMPONENT_SCHEMA:
        raise WorkflowPolicyError("Unsupported readiness component schema version.")
    body = {key: receipt[key] for key in required if key != "receipt_id"}
    assert_no_sensitive_evidence(body)
    if receipt["receipt_id"] != content_id(body):
        raise WorkflowPolicyError("Readiness component receipt identity is corrupt.")
    mode = str(receipt["mode"])
    if mode not in READINESS_IDENTITIES or receipt["identity"] != READINESS_IDENTITIES[mode]:
        raise WorkflowPolicyError("Readiness component mode or identity is invalid.")
    if expected_mode is not None and mode != expected_mode:
        raise WorkflowPolicyError("Readiness component mode does not match.")
    binding = receipt["binding"]
    _require_readiness_binding(binding)
    if expected_binding is not None and any(binding.get(key) != value for key, value in expected_binding.items()):
        raise WorkflowPolicyError("Readiness component binding does not match.")
    if not isinstance(receipt["reason_codes"], list) or any(
        not isinstance(item, str) for item in receipt["reason_codes"]
    ):
        raise WorkflowPolicyError("Readiness component reason codes are invalid.")
    if not isinstance(receipt["checks"], Mapping):
        raise WorkflowPolicyError("Readiness component checks are invalid.")
    if not isinstance(receipt["decision"], str) or receipt["decision"] not in {"GO", "NO_GO"}:
        raise WorkflowPolicyError("Readiness component decision is invalid.")
    if receipt["decision"] == "GO" and receipt["reason_codes"]:
        raise WorkflowPolicyError("A GO readiness component must have no reason codes.")
    if receipt["decision"] == "NO_GO" and receipt["reason_codes"] == []:
        raise WorkflowPolicyError("A NO_GO readiness component must have reason codes.")
    return dict(receipt)


def build_readiness_receipt(
    *,
    components: Mapping[str, Mapping[str, Any]],
    required_modes: Sequence[str],
    repository: str,
    workflow: str,
    source_sha: str,
    ref: str,
    workflow_run_id: str,
    run_attempt: str,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Aggregate component evidence into one immutable workflow gate receipt."""
    required = list(dict.fromkeys(required_modes))
    if not required or any(mode not in READINESS_IDENTITIES for mode in required):
        raise WorkflowPolicyError("Readiness required modes are invalid.")
    _require_readiness_binding(
        {
            "repository": repository,
            "workflow": workflow,
            "source_sha": source_sha,
            "ref": ref,
            "workflow_run_id": workflow_run_id,
            "run_attempt": run_attempt,
        }
    )
    expected_binding = {
        "repository": repository,
        "workflow": workflow,
        "source_sha": source_sha.lower(),
        "ref": ref,
        "workflow_run_id": workflow_run_id,
        "run_attempt": run_attempt,
    }
    validated: dict[str, dict[str, Any]] = {}
    reasons: set[str] = set()
    for mode in required:
        component = components.get(mode)
        if component is None:
            reasons.add("EVIDENCE_MISSING")
            continue
        try:
            item = validate_readiness_component_receipt(component, expected_mode=mode, expected_binding=expected_binding)
        except WorkflowPolicyError:
            reasons.add("COMPONENT_INVALID")
            continue
        validated[mode] = item
        if item["decision"] != "GO":
            reasons.update(str(code) for code in item["reason_codes"])
    all_go = len(validated) == len(required) and all(validated[mode]["decision"] == "GO" for mode in required)
    body = {
        "schema_version": READINESS_SCHEMA,
        "decision": "GO" if all_go and not reasons else "NO_GO",
        "reason_codes": sorted(reasons),
        "allowed_modes": required if all_go and not reasons else [],
        "binding": expected_binding,
        "observed_at_utc": observed_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "required_modes": required,
        "components": {mode: validated[mode] for mode in sorted(validated)},
        "component_receipt_hashes": {mode: validated[mode]["receipt_id"] for mode in sorted(validated)},
    }
    assert_no_sensitive_evidence(body)
    return {"receipt_id": content_id(body), **body}


def validate_readiness_receipt(
    receipt: Mapping[str, Any], *, expected_mode: str | None = None, expected_binding: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate an aggregate receipt, optionally authorizing one workflow mode."""
    _require_mapping(receipt, "readiness receipt")
    required = {"receipt_id", "schema_version", "decision", "reason_codes", "allowed_modes", "binding", "observed_at_utc", "required_modes", "components", "component_receipt_hashes"}
    if set(receipt) != required:
        raise WorkflowPolicyError("Readiness receipt fields are invalid.")
    if receipt["schema_version"] != READINESS_SCHEMA:
        raise WorkflowPolicyError("Unsupported readiness schema version.")
    body = {key: receipt[key] for key in required if key != "receipt_id"}
    assert_no_sensitive_evidence(body)
    if receipt["receipt_id"] != content_id(body):
        raise WorkflowPolicyError("Readiness receipt identity is corrupt.")
    binding = receipt["binding"]
    _require_readiness_binding(binding)
    if expected_binding is not None:
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise WorkflowPolicyError("Readiness receipt binding does not match.")
    reason_codes = receipt["reason_codes"]
    allowed_modes = receipt["allowed_modes"]
    required_modes = receipt["required_modes"]
    components = receipt["components"]
    component_hashes = receipt["component_receipt_hashes"]
    if (
        not isinstance(reason_codes, list)
        or any(not isinstance(item, str) for item in reason_codes)
        or not isinstance(allowed_modes, list)
        or any(not isinstance(item, str) for item in allowed_modes)
        or any(item not in READINESS_IDENTITIES for item in allowed_modes)
        or not isinstance(required_modes, list)
        or not required_modes
        or any(not isinstance(item, str) for item in required_modes)
        or len(set(required_modes)) != len(required_modes)
        or any(item not in READINESS_IDENTITIES for item in required_modes)
        or not isinstance(components, Mapping)
        or not isinstance(component_hashes, Mapping)
        or set(components) != set(component_hashes)
        or not set(components).issubset(required_modes)
    ):
        raise WorkflowPolicyError("Readiness receipt component fields are invalid.")
    for mode, component in components.items():
        validate_readiness_component_receipt(component, expected_mode=mode, expected_binding=binding)
        if component_hashes[mode] != component["receipt_id"]:
            raise WorkflowPolicyError("Readiness component hash does not match.")
    if not isinstance(receipt["decision"], str) or receipt["decision"] not in {"GO", "NO_GO"}:
        raise WorkflowPolicyError("Readiness decision is invalid.")
    if receipt["decision"] == "GO" and receipt["reason_codes"]:
        raise WorkflowPolicyError("A GO readiness receipt must have no reason codes.")
    if receipt["decision"] == "GO" and allowed_modes != required_modes:
        raise WorkflowPolicyError("A GO readiness receipt must authorize every required mode.")
    if receipt["decision"] == "GO" and (
        set(components) != set(required_modes)
        or any(component["decision"] != "GO" for component in components.values())
    ):
        raise WorkflowPolicyError("A GO readiness receipt must include GO components for every required mode.")
    if receipt["decision"] == "NO_GO" and (receipt["reason_codes"] == [] or receipt["allowed_modes"] != []):
        raise WorkflowPolicyError("A NO_GO readiness receipt cannot authorize workflows.")
    if expected_mode is not None:
        if receipt["decision"] != "GO" or expected_mode not in receipt["allowed_modes"]:
            raise WorkflowPolicyError(f"Readiness receipt does not permit workflow {expected_mode!r}.")
    return dict(receipt)


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise WorkflowPolicyError(f"{name} must be a JSON object.")


def _require_readiness_binding(binding: Mapping[str, Any]) -> None:
    if not isinstance(binding, Mapping) or set(binding) != _READINESS_BINDING_FIELDS:
        raise WorkflowPolicyError("Readiness binding is invalid.")
    for name in ("repository", "workflow", "ref"):
        if not isinstance(binding[name], str) or not binding[name].strip():
            raise WorkflowPolicyError(f"Readiness binding {name} is required.")
    require_commit_sha(binding["source_sha"], name="binding.source_sha")
    require_run_id(binding["workflow_run_id"], name="binding.workflow_run_id")
    require_run_id(binding["run_attempt"], name="binding.run_attempt")


__all__ = [
    "READINESS_COMPONENT_SCHEMA",
    "READINESS_IDENTITIES",
    "READINESS_REQUIRED_UPSTREAM",
    "READINESS_ROLE_REQUIREMENTS",
    "READINESS_SCHEMA",
    "RECEIPT_SCHEMA",
    "RELEASE_MANIFEST_SCHEMA",
    "ROLLBACK_MANIFEST_SCHEMA",
    "WorkflowPolicyError",
    "build_receipt",
    "build_readiness_component_receipt",
    "build_readiness_receipt",
    "canonical_json",
    "content_id",
    "assert_no_sensitive_evidence",
    "evaluate_readiness",
    "forbidden_plan_changes",
    "forbidden_tracked_paths",
    "missing_configuration",
    "parse_image_reference",
    "require_commit_sha",
    "require_configuration",
    "require_release_enabled",
    "require_run_id",
    "validate_readiness_component_receipt",
    "validate_readiness_receipt",
    "validate_active_images",
    "validate_image_pair",
    "validate_release_manifest",
    "validate_rollback_manifest",
]
