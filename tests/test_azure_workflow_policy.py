from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.azure_workflow_policy import (
    WorkflowPolicyError,
    build_receipt,
    forbidden_plan_changes,
    forbidden_tracked_paths,
    require_configuration,
    require_release_enabled,
    validate_active_images,
    validate_image_pair,
    validate_release_manifest,
    validate_rollback_manifest,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "azure_workflow"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _fixture(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _repository_text(path: str) -> str:
    return (REPOSITORY_ROOT / path).read_text(encoding="utf-8")


def test_release_flag_and_configuration_fail_closed() -> None:
    require_release_enabled("true")
    with pytest.raises(WorkflowPolicyError):
        require_release_enabled("false")
    with pytest.raises(WorkflowPolicyError, match="AZURE_CLIENT_ID"):
        require_configuration({"AZURE_TENANT_ID": "tenant"}, ["AZURE_TENANT_ID", "AZURE_CLIENT_ID"])


def test_image_pair_requires_expected_immutable_repositories() -> None:
    pair = validate_image_pair(
        "example.azurecr.io/wind-forecast-api@sha256:" + "1" * 64,
        "example.azurecr.io/wind-forecast-frontend@sha256:" + "2" * 64,
        expected_registry="example.azurecr.io",
    )
    assert pair["registry"] == "example.azurecr.io"
    with pytest.raises(WorkflowPolicyError):
        validate_image_pair(
            "other.azurecr.io/wind-forecast-api@sha256:" + "1" * 64,
            "example.azurecr.io/wind-forecast-frontend@sha256:" + "2" * 64,
        )


def test_release_manifest_fixture_is_source_and_label_pinned() -> None:
    manifest = _fixture("release-manifest.json")
    identity = validate_release_manifest(
        manifest,
        expected_source_sha=manifest["source_sha"],
        expected_release_run_id="200",
    )
    assert identity["ci_run_id"] == "100"
    invalid = dict(manifest, frontend_source_label="b" * 40)
    with pytest.raises(WorkflowPolicyError, match="frontend_source_label"):
        validate_release_manifest(invalid)


def test_terraform_plan_rejects_delete_and_replacement() -> None:
    assert forbidden_plan_changes(_fixture("terraform-plan-noop.json")) == []
    forbidden = forbidden_plan_changes(_fixture("terraform-plan-delete.json"))
    assert forbidden == [{"address": "azurerm_container_app.api", "actions": ["delete", "create"]}]


def test_active_images_must_match_manifest() -> None:
    manifest = _fixture("release-manifest.json")
    observed = _fixture("active-images.json")
    identity = validate_release_manifest(manifest)
    assert validate_active_images(identity, observed)["api_image"] == manifest["api_image"]
    with pytest.raises(WorkflowPolicyError, match="frontend"):
        validate_active_images(identity, dict(observed, frontend_image="example.azurecr.io/wind-forecast-frontend@sha256:" + "3" * 64))


def test_receipt_requires_passed_smoke_and_drift_checks() -> None:
    manifest = _fixture("release-manifest.json")
    receipt = build_receipt(
        operation="release",
        manifest=manifest,
        active_images=_fixture("active-images.json"),
        api_revision="api--revision",
        frontend_revision="frontend--revision",
        smoke_tests={"dashboard": True, "validation_job": True},
        terraform_post_plan_exit_code=0,
        workflow_run_id="300",
    )
    assert receipt["release_run_id"] == "200"
    assert receipt["approval"]["gate_passed"] is True
    assert receipt["approval"]["mode"] == "maintainer_confirmation"
    assert (
        receipt["approval"]["independent_human_review"]
        == "not_applicable_single_maintainer"
    )
    with pytest.raises(WorkflowPolicyError):
        build_receipt(
            operation="release",
            manifest=manifest,
            active_images=_fixture("active-images.json"),
            api_revision="api--revision",
            frontend_revision="frontend--revision",
            smoke_tests={"dashboard": True},
            terraform_post_plan_exit_code=2,
            workflow_run_id="300",
        )


def test_rollback_manifest_keeps_registered_release_identity() -> None:
    release = _fixture("release-manifest.json")
    rollback = {
        "schema_version": "wind_forecast.production_rollback_manifest.v1",
        "manifest_type": "rollback",
        "request_sha": "b" * 40,
        "rollback_run_id": "400",
        "source_sha": release["source_sha"],
        "ci_run_id": release["ci_run_id"],
        "release_run_id": release["release_run_id"],
        "api_image": release["api_image"],
        "frontend_image": release["frontend_image"],
    }
    identity = validate_rollback_manifest(rollback, expected_rollback_run_id="400")
    assert identity["ci_run_id"] == "100"
    receipt = build_receipt(
        operation="rollback",
        manifest=rollback,
        active_images=_fixture("active-images.json"),
        api_revision="api--rollback",
        frontend_revision="frontend--rollback",
        smoke_tests={"dashboard": True},
        terraform_post_plan_exit_code=0,
        workflow_run_id="401",
    )
    assert receipt["release_run_id"] == "200"
    assert receipt["rollback_run_id"] == "400"


def test_forbidden_tracked_paths_are_detected_without_content_scanning() -> None:
    paths = [
        "infra/azure/terraform/production.tfstate",
        "infra/azure/terraform/production.tfplan",
        "infra/azure/terraform/production.tfvars",
        "tmp/client_secret.txt",
        "docs/allowed.tfvars.example",
    ]
    assert forbidden_tracked_paths(paths) == sorted(paths[:4])


def test_production_plan_uses_explicit_planner_oidc_before_backend_init() -> None:
    workflow = _repository_text(".github/workflows/release-production.yml")
    plan_start = workflow.index("  terraform-plan:")
    plan_end = workflow.index("\n  deploy-production:", plan_start)
    plan_job = workflow[plan_start:plan_end]

    login = "      - name: Sign in to Azure with production planner OIDC"
    backend_init = "      - name: Initialize the production backend with OIDC"
    assert plan_job.index(login) < plan_job.index(backend_init)
    assert "        uses: azure/login@v2" in plan_job
    assert "          client-id: ${{ secrets.AZURE_PLANNER_CLIENT_ID }}" in plan_job
    assert '      ARM_USE_OIDC: "true"' in plan_job
    assert '      ARM_USE_AZUREAD: "true"' in plan_job
    assert "      ARM_CLIENT_ID: ${{ secrets.AZURE_PLANNER_CLIENT_ID }}" in plan_job
    assert "      ARM_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}" in plan_job
    assert (
        "      ARM_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}"
        in plan_job
    )


def test_release_frontend_smoke_resolves_azure_upstream_locally() -> None:
    workflow = _repository_text(".github/workflows/release-production.yml")
    smoke_start = workflow.index(
        "      - name: Smoke test the exact frontend image before publication"
    )
    smoke_end = workflow.index(
        "      - name: Show frontend smoke logs on failure", smoke_start
    )
    smoke = workflow[smoke_start:smoke_end]

    assert "--add-host wind-forecast-api:host-gateway" in smoke


def test_bootstrap_limits_deployer_rbac_delegation_to_acr_data_roles() -> None:
    bootstrap = _repository_text("infra/azure/terraform/bootstrap/main.tf")
    publisher_start = bootstrap.index(
        'resource "azurerm_role_assignment" "publisher_workload_reader"'
    )
    assignment_start = bootstrap.index(
        'resource "azurerm_role_assignment" '
        '"deployer_workload_rbac_administrator"'
    )
    publisher = bootstrap[publisher_start:assignment_start]
    assignment = bootstrap[assignment_start:]

    assert "scope              = azurerm_resource_group.workload.id" in publisher
    assert "role_definition_id = local.reader_role_id" in publisher
    assert "azurerm_user_assigned_identity.publisher.principal_id" in publisher
    assert 'name = "Role Based Access Control Administrator"' in bootstrap
    assert "scope              = azurerm_resource_group.workload.id" in assignment
    assert "role_definition_id = local.rbac_administrator_role_id" in assignment
    assert "azurerm_user_assigned_identity.deployer.principal_id" in assignment
    assert 'condition_version  = "2.0"' in assignment
    assert "Microsoft.Authorization/roleAssignments/write" in assignment
    assert "Microsoft.Authorization/roleAssignments/delete" in assignment
    assert assignment.count("ForAnyOfAnyValues:GuidEquals") == 2
    assert assignment.count("${local.acr_pull_role_definition_guid}") == 2
    assert assignment.count("${local.acr_push_role_definition_guid}") == 2
    assert assignment.count("PrincipalType") == 2
    assert assignment.count(
        "ForAnyOfAnyValues:StringEqualsIgnoreCase {'ServicePrincipal'}"
    ) == 2
    assert "7f951dda-4ed3-4680-a7ca-43fe172d538d" in bootstrap
    assert "8311e382-0749-4cb8-b61a-304f252e45ec" in bootstrap
    assert "data.azurerm_client_config.current.subscription_id" in bootstrap
    assert "basename(data.azurerm_role_definition.reader.role_definition_id)" in bootstrap
    assert (
        "basename(data.azurerm_role_definition.rbac_administrator.role_definition_id)"
        in bootstrap
    )


def test_foundation_apply_fails_closed_on_any_post_apply_drift() -> None:
    workflow = _repository_text(".github/workflows/foundation-production.yml")
    apply_start = workflow.index("  apply-foundation:")
    apply_job = workflow[apply_start:]
    apply_plan = apply_job.index(
        "      - name: Apply the exact approved foundation plan"
    )
    drift_check = apply_job.index(
        "      - name: Verify Terraform foundation state is drift-free after apply"
    )

    assert apply_plan < drift_check
    assert "-detailed-exitcode" in apply_job[drift_check:]
    assert 'case "$plan_exit" in' in apply_job[drift_check:]
    assert "drift detected; the workflow failed closed" in apply_job[drift_check:]
    assert "{address, actions: .change.actions}" in apply_job[drift_check:]
    assert "exit 1" in apply_job[drift_check:]


def test_foundation_uses_subscription_qualified_acr_role_ids() -> None:
    foundation = _repository_text("infra/azure/terraform/foundation/main.tf")
    assert "role_definition_id = local.acr_pull_role_id" in foundation
    assert "role_definition_id = local.acr_push_role_id" in foundation
    assert "data.azurerm_client_config.current.subscription_id" in foundation
    assert "basename(data.azurerm_role_definition.acr_pull.role_definition_id)" in foundation
    assert "basename(data.azurerm_role_definition.acr_push.role_definition_id)" in foundation
