from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import collect_azure_readiness as collector
from scripts.azure_workflow_policy import build_readiness_component_receipt


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "azure_workflow"
WORKLOAD_SCOPE = "/subscriptions/subscription-id/resourceGroups/workload"
STATE_SCOPE = (
    "/subscriptions/subscription-id/resourceGroups/state/providers/"
    "Microsoft.Storage/storageAccounts/state"
)
ACR_SCOPE = (
    "/subscriptions/subscription-id/resourceGroups/workload/providers/"
    "Microsoft.ContainerRegistry/registries/acr"
)
SUBSCRIPTION_SCOPE = "/subscriptions/subscription-id"

_CONDITION = """
(
  (
    !(ActionMatches{'Microsoft.Authorization/roleAssignments/write'})
  )
  OR
  (
    @Request[Microsoft.Authorization/roleAssignments:RoleDefinitionId]
    ForAnyOfAnyValues:GuidEquals {
      7f951dda-4ed3-4680-a7ca-43fe172d538d,
      8311e382-0749-4cb8-b61a-304f252e45ec
    }
    AND
    @Request[Microsoft.Authorization/roleAssignments:PrincipalType]
    ForAnyOfAnyValues:StringEqualsIgnoreCase {'ServicePrincipal'}
  )
)
AND
(
  (
    !(ActionMatches{'Microsoft.Authorization/roleAssignments/delete'})
  )
  OR
  (
    @Resource[Microsoft.Authorization/roleAssignments:RoleDefinitionId]
    ForAnyOfAnyValues:GuidEquals {
      7f951dda-4ed3-4680-a7ca-43fe172d538d,
      8311e382-0749-4cb8-b61a-304f252e45ec
    }
    AND
    @Resource[Microsoft.Authorization/roleAssignments:PrincipalType]
    ForAnyOfAnyValues:StringEqualsIgnoreCase {'ServicePrincipal'}
  )
)
"""


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _configure_scopes(monkeypatch: pytest.MonkeyPatch, *, identity: str) -> None:
    monkeypatch.setenv("AZURE_RESOURCE_GROUP", "workload")
    monkeypatch.setenv("TFSTATE_RESOURCE_GROUP_NAME", "state")
    monkeypatch.setenv("TFSTATE_STORAGE_ACCOUNT_NAME", "state")
    if identity == "publisher":
        monkeypatch.setenv("AZURE_ACR_NAME", "acr")

    def fake_tsv(arguments: list[str]) -> str | None:
        if arguments[:2] == ["group", "show"]:
            return WORKLOAD_SCOPE
        if arguments[:2] == ["storage", "account"]:
            return STATE_SCOPE
        if arguments[:2] == ["acr", "show"]:
            return ACR_SCOPE
        raise AssertionError(arguments)

    monkeypatch.setattr(collector, "_az_tsv", fake_tsv)
    monkeypatch.setattr(collector, "_principal_id", lambda: "principal-id")


def _assignment(role: str, scope: str, **extra: Any) -> dict[str, Any]:
    return {"roleDefinitionName": role, "scope": scope, **extra}


def test_collector_preserves_inherited_owner_without_azure_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fixture("readiness-planner.json")
    # Use the real collector output, not a handcrafted sanitized permission set.
    _configure_scopes(monkeypatch, identity="planner")

    def fake_json(arguments: list[str]) -> Any:
        if "--scope" not in arguments:
            raise AssertionError(arguments)
        scope = arguments[arguments.index("--scope") + 1]
        return {
            WORKLOAD_SCOPE: [
                _assignment("Owner", SUBSCRIPTION_SCOPE),
                _assignment("Reader", WORKLOAD_SCOPE),
            ],
            STATE_SCOPE: [
                _assignment("Reader", STATE_SCOPE),
                _assignment("Storage Blob Data Contributor", STATE_SCOPE),
            ],
        }[scope]

    monkeypatch.setattr(collector, "_az_json", fake_json)
    permissions = collector._permission_evidence("release-plan", "planner")

    assert permissions["probe_ok"] is True
    assert {
        "condition": None,
        "role": "Owner",
        "scope": "inherited_subscription",
    } in permissions["assignments"]
    serialized = json.dumps(permissions, sort_keys=True)
    for value in ("subscription-id", "resourceGroups", "principal-id", "assignment-id"):
        assert value not in serialized

    evidence["permissions"] = permissions
    component = build_readiness_component_receipt(
        mode="release-plan",
        evidence=evidence,
        repository="tjpoa/wind-energy-forecast",
        workflow="Release production",
        source_sha="a" * 40,
        ref="refs/heads/master",
        workflow_run_id="910",
        run_attempt="1",
    )
    assert component["decision"] == "NO_GO"
    assert component["reason_codes"] == ["AZURE_PERMISSION_TOO_BROAD"]
    serialized_receipt = json.dumps(component, sort_keys=True)
    for value in ("subscription-id", "resourceGroups", "principal-id", "assignment-id"):
        assert value not in serialized_receipt


def test_collector_deduplicates_direct_assignment_seen_through_child_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scopes(monkeypatch, identity="publisher")

    def fake_json(arguments: list[str]) -> Any:
        scope = arguments[arguments.index("--scope") + 1]
        return {
            WORKLOAD_SCOPE: [_assignment("Reader", WORKLOAD_SCOPE)],
            ACR_SCOPE: [
                _assignment("Reader", WORKLOAD_SCOPE),
                _assignment("AcrPush", ACR_SCOPE),
            ],
        }[scope]

    monkeypatch.setattr(collector, "_az_json", fake_json)
    permissions = collector._permission_evidence("release-publish", "publisher")

    assert permissions == {
        "probe_ok": True,
        "assignments": [
            {"condition": None, "role": "AcrPush", "scope": "container_registry"},
            {"condition": None, "role": "Reader", "scope": "workload_resource_group"},
        ],
    }


def test_collector_uses_symbolic_names_for_every_inherited_scope_kind() -> None:
    direct_scopes = {
        "workload_resource_group": WORKLOAD_SCOPE,
        "state_storage_account": STATE_SCOPE,
        "container_registry": ACR_SCOPE,
    }
    assert (
        collector._symbolic_scope(
            SUBSCRIPTION_SCOPE,
            queried_scope=WORKLOAD_SCOPE,
            direct_scopes=direct_scopes,
        )
        == "inherited_subscription"
    )
    assert (
        collector._symbolic_scope(
            "/providers/Microsoft.Management/managementGroups/platform",
            queried_scope=WORKLOAD_SCOPE,
            direct_scopes=direct_scopes,
        )
        == "inherited_management_group"
    )
    assert (
        collector._symbolic_scope(
            "/subscriptions/subscription-id/resourceGroups/state",
            queried_scope=STATE_SCOPE,
            direct_scopes=direct_scopes,
        )
        == "inherited_resource_group"
    )
    assert (
        collector._symbolic_scope(
            ACR_SCOPE + "/replications",
            queried_scope=ACR_SCOPE + "/replications/replica",
            direct_scopes=direct_scopes,
        )
        == "inherited_parent_resource"
    )


@pytest.mark.parametrize(
    "payload",
    [
        [_assignment("Unknown Role", WORKLOAD_SCOPE)],
        {"not": "an assignment list"},
        [_assignment("Reader", "/invalid-scope")],
        [{"roleDefinitionName": "Reader"}],
        [
            _assignment(
                "RBAC Administrator",
                WORKLOAD_SCOPE,
                condition=None,
                conditionVersion="2.0",
            )
        ],
        [_assignment("Reader", WORKLOAD_SCOPE, condition="conditional")],
    ],
    ids=[
        "unknown-role",
        "non-list-payload",
        "invalid-scope",
        "missing-scope",
        "malformed-condition",
        "condition-without-version",
    ],
)
def test_collector_fails_closed_on_invalid_rbac_records(
    monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    _configure_scopes(monkeypatch, identity="planner")

    def fake_json(arguments: list[str]) -> Any:
        scope = arguments[arguments.index("--scope") + 1]
        return payload if scope == WORKLOAD_SCOPE else []

    monkeypatch.setattr(collector, "_az_json", fake_json)
    permissions = collector._permission_evidence("release-plan", "planner")

    assert permissions["probe_ok"] is False


def test_rbac_condition_accepts_only_exact_canonical_terraform_expression() -> None:
    assert (
        collector._condition_label(_CONDITION, "RBAC Administrator", "2.0")
        == "acr_roles_only"
    )
    variant = "\n".join(line.upper() for line in _CONDITION.splitlines())
    assert (
        collector._condition_label(variant, "RBAC Administrator", "2.0")
        == "acr_roles_only"
    )
    assert (
        collector._condition_label(_CONDITION.replace("\n", " "), "RBAC Administrator", "2.0")
        == "acr_roles_only"
    )
    assert (
        collector._condition_label(
            _CONDITION.replace("OR", "AND", 1), "RBAC Administrator", "2.0"
        )
        == "invalid"
    )
    assert collector._condition_label(_CONDITION, "RBAC Administrator", "1.0") == "invalid"
    assert (
        collector._condition_label(
            _CONDITION.replace("ServicePrincipal", "User"),
            "RBAC Administrator",
            "2.0",
        )
        == "invalid"
    )


def test_policy_classifies_unknown_role_as_probe_failure() -> None:
    evidence = _fixture("readiness-planner.json")
    evidence["permissions"]["assignments"].append(
        {"condition": None, "role": "Unknown Role", "scope": "workload_resource_group"}
    )
    component = build_readiness_component_receipt(
        mode="release-plan",
        evidence=evidence,
        repository="tjpoa/wind-energy-forecast",
        workflow="Release production",
        source_sha="a" * 40,
        ref="refs/heads/master",
        workflow_run_id="911",
        run_attempt="1",
    )
    assert component["decision"] == "NO_GO"
    assert "AZURE_PERMISSION_PROBE_FAILED" in component["reason_codes"]


def test_policy_classifies_invalid_rbac_condition_as_too_broad() -> None:
    evidence = _fixture("readiness-deployer.json")
    evidence["permissions"]["assignments"][0]["condition"] = "invalid"
    component = build_readiness_component_receipt(
        mode="release-deploy",
        evidence=evidence,
        repository="tjpoa/wind-energy-forecast",
        workflow="Release production",
        source_sha="a" * 40,
        ref="refs/heads/master",
        workflow_run_id="912",
        run_attempt="1",
    )
    assert component["decision"] == "NO_GO"
    assert "AZURE_PERMISSION_TOO_BROAD" in component["reason_codes"]


def test_azure_cli_uses_resolved_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(collector.shutil, "which", lambda _name: "C:/Azure/az.CMD")

    def fake_run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(collector.subprocess, "run", fake_run)

    assert collector._az_json(["account", "show"]) == {}
    assert calls[0][0] == "C:/Azure/az.CMD"
