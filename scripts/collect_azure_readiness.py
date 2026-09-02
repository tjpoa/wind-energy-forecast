"""Collect sanitized, read-only GitHub/Azure readiness evidence.

The collector is deliberately an adapter.  It may call the runner's GitHub
and Azure CLIs, but it only writes a normalized evidence object; tokens,
headers, resource IDs, and command output are never persisted.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    from azure_workflow_policy import READINESS_IDENTITIES, READINESS_REQUIRED_UPSTREAM
    from azure_workflow_policy import assert_no_sensitive_evidence, missing_configuration
    from validate_azure_workflow import MODE_CONFIGURATION
except ModuleNotFoundError as exc:
    if exc.name not in {"azure_workflow_policy", "validate_azure_workflow"}:
        raise
    from scripts.azure_workflow_policy import READINESS_IDENTITIES, READINESS_REQUIRED_UPSTREAM
    from scripts.azure_workflow_policy import assert_no_sensitive_evidence, missing_configuration
    from scripts.validate_azure_workflow import MODE_CONFIGURATION


OIDC_AUDIENCE = "api://AzureADTokenExchange"
OIDC_ISSUER = "https://token.actions.githubusercontent.com"
ROLE_ID_TO_NAME = {
    "acdd72a7-3385-48ef-bd42-f606fba81ae7": "Reader",
    "b24988ac-6180-42a0-ab88-20f7382dd24c": "Contributor",
    "8311e382-0749-4cb8-b61a-304f252e45ec": "AcrPush",
    "ba92f5b4-2d11-453d-a403-e96b0029c9fe": "Storage Blob Data Contributor",
    "f58310d9-a9f6-439a-9e8d-f62e7b41a168": "RBAC Administrator",
    "8e3af657-a8ff-443c-a75c-2fe8c4bcb635": "Owner",
}
ROLE_NAME_ALIASES = {
    "Role Based Access Control Administrator": "RBAC Administrator",
}
KNOWN_ROLE_NAMES = frozenset(ROLE_ID_TO_NAME.values())
EXPECTED_RBAC_ADMINISTRATOR_CONDITION_SHA256 = (
    "7fa6d91ed6cc2b7342a0da693abaa57f019f9f46ce757b5afa243b0c7f0cc13d"
)
_RBAC_ADMINISTRATOR_CONDITION_VERSION = "2.0"
_AZURE_SCOPE = re.compile(
    r"^/(?:subscriptions/[^/\s]+(?:/.*)?|providers/microsoft\.management/"
    r"managementgroups/[^/\s]+(?:/.*)?)$",
    re.IGNORECASE,
)
_SUBSCRIPTION_SCOPE = re.compile(r"^/subscriptions/[^/\s]+$", re.IGNORECASE)
_RESOURCE_GROUP_SCOPE = re.compile(
    r"^/subscriptions/[^/\s]+/resourcegroups/[^/\s]+$", re.IGNORECASE
)
_MANAGEMENT_GROUP_SCOPE = re.compile(
    r"^/providers/microsoft\.management/managementgroups/[^/\s]+$",
    re.IGNORECASE,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(READINESS_IDENTITIES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--workflow", default=os.environ.get("GITHUB_WORKFLOW", ""))
    parser.add_argument("--source-sha", default=os.environ.get("READINESS_SOURCE_SHA", os.environ.get("GITHUB_SHA", "")))
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--environment", default="production")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evidence = _empty_evidence(args.mode)
    try:
        evidence = _collect(args)
        assert_no_sensitive_evidence(evidence)
    except Exception:
        # A collector failure is itself a NO_GO input.  Do not expose provider
        # output in the receipt or turn an external error into a credential log.
        evidence["collector"] = {"probe_ok": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _empty_evidence(mode: str) -> dict[str, Any]:
    identity = READINESS_IDENTITIES[mode]
    return {
        "mode": mode,
        "identity": identity,
        "configuration": {"release_enabled": False, "missing": list(MODE_CONFIGURATION[mode])},
        "oidc": {"login_succeeded": False, "account_match": False, "claims": {}},
        "permissions": {"probe_ok": False, "assignments": []},
        "state": {"storage_account": {"exists": False}, "container": {"probe_ok": False, "exists": False}, "blobs": {}},
        "github": {"environment": {"probe_ok": False}},
        "observed_at_utc": _now(),
    }


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    env = os.environ
    mode = args.mode
    identity = READINESS_IDENTITIES[mode]
    configuration = {
        "release_enabled": str(env.get("PRODUCTION_RELEASE_ENABLED", "")).strip().lower() == "true",
        "missing": list(missing_configuration(env, MODE_CONFIGURATION[mode])),
    }
    evidence: dict[str, Any] = {
        "mode": mode,
        "identity": identity,
        "configuration": configuration,
        "oidc": _oidc_evidence(args),
        "permissions": {"probe_ok": False, "assignments": []},
        "state": {"storage_account": {"exists": False}, "container": {"probe_ok": False, "exists": False}, "blobs": {}},
        "github": {"environment": _github_environment_evidence(args)},
        "observed_at_utc": env.get("READINESS_OBSERVED_AT", _now()),
    }

    account = _az_json(["account", "show"])
    account_match = bool(account) and (
        str(account.get("tenantId", "")) == str(env.get("AZURE_TENANT_ID", ""))
        and str(account.get("id", "")) == str(env.get("AZURE_SUBSCRIPTION_ID", ""))
    )
    evidence["oidc"]["account_match"] = account_match
    if identity in {"planner", "deployer"}:
        evidence["state"] = _state_evidence(mode)

    evidence["permissions"] = _permission_evidence(mode, identity)
    return evidence


def _oidc_evidence(args: argparse.Namespace) -> dict[str, Any]:
    claims = _oidc_claims()
    login_outcome = os.environ.get("AZURE_LOGIN_OUTCOME", "").strip().lower()
    return {
        "login_succeeded": login_outcome == "success",
        "account_match": False,
        "claims": claims,
    }


def _oidc_claims() -> dict[str, Any]:
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not request_url or not request_token:
        return {}
    separator = "&" if "?" in request_url else "?"
    url = request_url + separator + "audience=" + quote(OIDC_AUDIENCE)
    request = Request(
        url,
        headers={
            "Authorization": f"bearer {request_token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, ValueError):
        return {}
    raw = payload.get("value") if isinstance(payload, dict) else None
    if not isinstance(raw, str):
        return {}
    try:
        token_payload = raw.split(".")[1]
        padding = "=" * (-len(token_payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(token_payload + padding).decode("utf-8"))
    except (IndexError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    audience = decoded.get("aud")
    return {
        "issuer": decoded.get("iss"),
        "audience": audience,
        "subject": decoded.get("sub"),
    }


def _github_environment_evidence(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository or os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("PREFLIGHT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token or "/" not in repository:
        return {"probe_ok": False}
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    owner, repo = repository.split("/", 1)
    url = f"{api_url}/repos/{quote(owner)}/{quote(repo)}/environments/{quote(args.environment, safe='')}"
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            environment = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, ValueError):
        return {"probe_ok": False}
    if not isinstance(environment, dict):
        return {"probe_ok": False}
    protection_rules = environment.get("protection_rules", [])
    if not isinstance(protection_rules, list):
        protection_rules = []
    reviewers = [
        reviewer
        for rule in protection_rules
        if isinstance(rule, dict) and rule.get("type") == "required_reviewers"
        for reviewer in (rule.get("reviewers") or [])
        if isinstance(reviewer, dict)
    ]
    branch_policy = environment.get("deployment_branch_policy")
    if not isinstance(branch_policy, dict):
        branch_policy = {}
    custom_branches: list[str] = []
    if branch_policy.get("custom_branch_policies") is True:
        policies_url = url + "/deployment-branch-policies"
        policies_request = Request(policies_url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        try:
            with urlopen(policies_request, timeout=10) as response:
                policy_payload = json.loads(response.read().decode("utf-8"))
            for item in policy_payload.get("branch_policies", []) if isinstance(policy_payload, dict) else []:
                if isinstance(item, dict) and item.get("name"):
                    custom_branches.append(str(item["name"]))
        except (HTTPError, URLError, OSError, ValueError):
            return {"probe_ok": False}
    return {
        "probe_ok": True,
        "name": environment.get("name"),
        "required_reviewer_count": len(reviewers),
        "required_reviewer_types": sorted({str(item.get("type", "")) for item in reviewers}),
        "deployment_branch_policy": {
            "protected_branches": branch_policy.get("protected_branches") is True,
            "custom_branch_policies": branch_policy.get("custom_branch_policies") is True,
            "custom_branches": sorted(custom_branches),
        },
    }


def _state_evidence(mode: str) -> dict[str, Any]:
    env = os.environ
    resource_group = env.get("TFSTATE_RESOURCE_GROUP_NAME", "")
    account_name = env.get("TFSTATE_STORAGE_ACCOUNT_NAME", "")
    container_name = env.get("TFSTATE_CONTAINER_NAME", "tfstate")
    if not resource_group or not account_name:
        return {"storage_account": {"exists": False}, "container": {"probe_ok": False, "exists": False}, "blobs": {}}
    account = _az_json(["storage", "account", "show", "--resource-group", resource_group, "--name", account_name])
    service = _az_json(["storage", "account", "blob-service-properties", "show", "--resource-group", resource_group, "--account-name", account_name])
    storage = {
        "exists": bool(account),
        "https_only": account.get("enableHttpsTrafficOnly") if account else None,
        "shared_key_enabled": account.get("allowSharedKeyAccess") if account else None,
        "min_tls_version": _normalize_tls(account.get("minimumTlsVersion")) if account else None,
        "versioning_enabled": service.get("isVersioningEnabled") if service else None,
    }
    container = _az_json([
        "storage", "container", "show", "--account-name", account_name,
        "--name", container_name, "--auth-mode", "login",
    ])
    public_access = None
    if container:
        properties = container.get("properties") if isinstance(container.get("properties"), dict) else {}
        public_access = container.get("publicAccess", properties.get("publicAccess"))
    container_evidence = {
        "probe_ok": container is not None,
        "exists": container is not None,
        "private": public_access in (None, "", "None", "none"),
    }
    keys = {READINESS_REQUIRED_UPSTREAM[mode]}
    keys.add("foundation.tfstate" if mode.startswith("foundation") else "production.tfstate")
    blobs: dict[str, Any] = {}
    for key in sorted(keys):
        exists_result = _az_json([
            "storage", "blob", "exists", "--account-name", account_name,
            "--container-name", container_name, "--name", key,
            "--auth-mode", "login",
        ])
        if exists_result is None:
            blobs[key] = {"probe_ok": False, "exists": False}
            continue
        exists = exists_result.get("exists") is True
        item: dict[str, Any] = {"probe_ok": True, "exists": exists}
        if exists:
            details = _az_json([
                "storage", "blob", "show", "--account-name", account_name,
                "--container-name", container_name, "--name", key,
                "--auth-mode", "login",
            ])
            properties = details.get("properties") if isinstance(details, dict) and isinstance(details.get("properties"), dict) else {}
            item["probe_ok"] = details is not None
            item["lease_state"] = properties.get("leaseState", properties.get("leaseStatus"))
        blobs[key] = item
    return {"storage_account": storage, "container": container_evidence, "blobs": blobs}


def _permission_evidence(mode: str, identity: str) -> dict[str, Any]:
    env = os.environ
    resource_group = env.get("AZURE_RESOURCE_GROUP", "wind-energy-forecast-demo")
    state_group = env.get("TFSTATE_RESOURCE_GROUP_NAME", "")
    state_account = env.get("TFSTATE_STORAGE_ACCOUNT_NAME", "")
    scopes: dict[str, str] = {}
    scope_probe_ok = True
    workload = _az_tsv(["group", "show", "--name", resource_group, "--query", "id"])
    if workload and _is_azure_scope(workload):
        scopes["workload_resource_group"] = workload.strip()
    else:
        scope_probe_ok = False
    if identity in {"planner", "deployer"} and state_group and state_account:
        state_id = _az_tsv(["storage", "account", "show", "--resource-group", state_group, "--name", state_account, "--query", "id"])
        if state_id and _is_azure_scope(state_id):
            scopes["state_storage_account"] = state_id.strip()
        else:
            scope_probe_ok = False
    elif identity in {"planner", "deployer"}:
        scope_probe_ok = False
    if identity == "publisher":
        acr_name = env.get("AZURE_ACR_NAME", "")
        if acr_name:
            acr_id = _az_tsv(["acr", "show", "--resource-group", resource_group, "--name", acr_name, "--query", "id"])
            if acr_id and _is_azure_scope(acr_id):
                scopes["container_registry"] = acr_id.strip()
            else:
                scope_probe_ok = False
        else:
            scope_probe_ok = False
    principal_id = _principal_id()
    if not principal_id or not scopes or not scope_probe_ok:
        return {"probe_ok": False, "assignments": []}
    assignments: set[tuple[str, str, str | None]] = set()
    probe_ok = True
    for symbolic_scope, scope_id in scopes.items():
        payload = _az_json([
            "role", "assignment", "list", "--assignee-object-id", principal_id,
            "--scope", scope_id, "--include-inherited", "--fill-principal-name", "false",
        ])
        if payload is None:
            probe_ok = False
            continue
        if not isinstance(payload, list):
            probe_ok = False
            continue
        for item in payload:
            if not isinstance(item, dict):
                probe_ok = False
                continue
            role = _resolve_role(item)
            if not role:
                probe_ok = False
                continue
            assignment_scope = item.get("scope")
            observed_scope = _symbolic_scope(
                assignment_scope, queried_scope=scope_id, direct_scopes=scopes
            )
            if observed_scope is None:
                probe_ok = False
                continue
            condition = _condition_label(
                item.get("condition"),
                role,
                item.get("conditionVersion"),
            )
            if _condition_is_malformed(
                item.get("condition"), role, item.get("conditionVersion")
            ):
                probe_ok = False
            assignments.add((role, observed_scope, condition))
    ordered = sorted(
        assignments,
        key=lambda item: (item[0], item[1], "" if item[2] is None else item[2]),
    )
    return {
        "probe_ok": probe_ok,
        "assignments": [
            {"role": role, "scope": scope, "condition": condition}
            for role, scope, condition in ordered
        ],
    }


def _resolve_role(item: dict[str, Any]) -> str | None:
    name = item.get("roleDefinitionName")
    role_definition_id = item.get("roleDefinitionId")
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            return None
        role = ROLE_NAME_ALIASES.get(name.strip(), name.strip())
        if role not in KNOWN_ROLE_NAMES:
            return None
    else:
        role = None
    if role_definition_id is not None:
        if not isinstance(role_definition_id, str) or not role_definition_id.strip():
            return None
        role_id = role_definition_id.rstrip("/").rsplit("/", 1)[-1].lower()
        id_role = ROLE_ID_TO_NAME.get(role_id)
        if id_role is None or (role is not None and id_role != role):
            return None
        role = id_role
    return role


def _condition_label(
    raw: Any, role: str, condition_version: Any = None
) -> str | None:
    if role != "RBAC Administrator":
        return None if raw is None else "invalid"
    if not isinstance(raw, str) or not raw.strip():
        return "invalid"
    if condition_version != _RBAC_ADMINISTRATOR_CONDITION_VERSION:
        return "invalid"
    canonical = re.sub(r"\s+", "", raw).lower()
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return (
        "acr_roles_only"
        if digest == EXPECTED_RBAC_ADMINISTRATOR_CONDITION_SHA256
        else "invalid"
    )


def _condition_is_malformed(raw: Any, role: str, condition_version: Any) -> bool:
    if role == "RBAC Administrator":
        return (
            not isinstance(raw, str)
            or not raw.strip()
            or not isinstance(condition_version, str)
            or not condition_version.strip()
        )
    if raw is None:
        return condition_version is not None
    if not isinstance(raw, str) or not raw.strip():
        return True
    return not isinstance(condition_version, str) or not condition_version.strip()


def _is_azure_scope(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().rstrip("/")
    return bool(normalized and _AZURE_SCOPE.fullmatch(normalized))


def _scope_is_at_or_below(scope: str, parent: str) -> bool:
    return scope == parent or scope.startswith(parent + "/")


def _symbolic_scope(
    assignment_scope: Any,
    *,
    queried_scope: str,
    direct_scopes: dict[str, str],
) -> str | None:
    if not _is_azure_scope(assignment_scope):
        return None
    assignment = assignment_scope.strip().rstrip("/").lower()
    queried = queried_scope.strip().rstrip("/").lower()
    for symbolic_scope, direct_scope in direct_scopes.items():
        direct = direct_scope.strip().rstrip("/").lower()
        if assignment == direct:
            return symbolic_scope if _scope_is_at_or_below(queried, direct) else None
    if _MANAGEMENT_GROUP_SCOPE.fullmatch(assignment):
        return "inherited_management_group"
    if not _scope_is_at_or_below(queried, assignment):
        return None
    if _SUBSCRIPTION_SCOPE.fullmatch(assignment):
        return "inherited_subscription"
    if _RESOURCE_GROUP_SCOPE.fullmatch(assignment):
        return "inherited_resource_group"
    return "inherited_parent_resource"


def _principal_id() -> str | None:
    payload = _az_json(["account", "get-access-token", "--resource", "https://management.azure.com/"])
    if not isinstance(payload, dict):
        return None
    token = payload.get("accessToken")
    if not isinstance(token, str) or token.count(".") != 2:
        return None
    try:
        encoded = token.split(".")[1]
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        claims = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    oid = claims.get("oid")
    return oid if isinstance(oid, str) and oid else None


def _az_json(arguments: list[str]) -> dict[str, Any] | list[Any] | None:
    try:
        completed = subprocess.run(
            ["az", *arguments, "--output", "json", "--only-show-errors"],
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _az_tsv(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["az", *arguments, "--output", "tsv", "--only-show-errors"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_tls(value: Any) -> Any:
    if value in {"TLS1_2", "TLS_1_2"}:
        return "TLS_1_2"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
