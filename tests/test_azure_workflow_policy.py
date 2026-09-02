from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.azure_workflow_policy import (
    READINESS_SCHEMA,
    WorkflowPolicyError,
    build_receipt,
    build_readiness_component_receipt,
    build_readiness_receipt,
    forbidden_plan_changes,
    forbidden_tracked_paths,
    require_configuration,
    require_release_enabled,
    validate_readiness_receipt,
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


def test_protected_production_jobs_expose_budget_configuration_to_preflight() -> None:
    workflow = _repository_text(".github/workflows/release-production.yml")

    for job_name, next_job in (
        ("  terraform-plan:", "\n  deploy-production:"),
        ("  deploy-production:", None),
    ):
        job_start = workflow.index(job_name)
        job_end = workflow.index(next_job, job_start) if next_job else len(workflow)
        job = workflow[job_start:job_end]
        assert (
            "      AZURE_BUDGET_ALERT_EMAIL: ${{ secrets.AZURE_BUDGET_ALERT_EMAIL }}"
            in job
        )
        assert (
            "      AZURE_BUDGET_START_DATE: ${{ vars.AZURE_BUDGET_START_DATE }}"
            in job
        )
        assert (
            "      AZURE_BUDGET_END_DATE: ${{ vars.AZURE_BUDGET_END_DATE }}"
            in job
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
    drift_gate = apply_job[drift_check:]
    assert "id: post_foundation_drift" in drift_gate
    assert "python scripts/verify_terraform_drift.py" in drift_gate
    assert "--directory infra/azure/terraform/foundation" in drift_gate
    assert "--plan-path post-foundation.tfplan" in drift_gate
    assert "--operation foundation-apply" in drift_gate


def test_release_and_rollback_use_shared_drift_gate_and_observed_receipt_code() -> None:
    workflows = (
        (
            ".github/workflows/release-production.yml",
            "      - name: Apply the exact reviewed-run plan",
            "      - name: Verify Terraform state is drift-free after deployment",
            "post_deployment_drift",
            "post-deployment.tfplan",
        ),
        (
            ".github/workflows/rollback-production.yml",
            "      - name: Apply the exact approved rollback plan",
            "      - name: Verify Terraform state is drift-free after rollback",
            "post_rollback_drift",
            "post-rollback.tfplan",
        ),
    )

    for path, apply_marker, drift_marker, step_id, plan_path in workflows:
        workflow = _repository_text(path)
        apply_index = workflow.index(apply_marker)
        drift_index = workflow.index(drift_marker)
        receipt_index = workflow.index("python scripts/create_azure_receipt.py receipt")
        drift_gate = workflow[drift_index:receipt_index]

        assert apply_index < drift_index < receipt_index
        assert f"id: {step_id}" in drift_gate
        assert "python scripts/verify_terraform_drift.py" in drift_gate
        assert "--directory infra/azure/terraform/production" in drift_gate
        assert f"--plan-path {plan_path}" in drift_gate
        assert "--detailed-exitcode" not in drift_gate
        expected_output = (
            '--terraform-post-plan-exit-code "${{ steps.'
            + step_id
            + '.outputs.plan_exit_code }}"'
        )
        assert expected_output in workflow
        assert "--terraform-post-plan-exit-code 0" not in workflow


def test_foundation_uses_subscription_qualified_acr_role_ids() -> None:
    foundation = _repository_text("infra/azure/terraform/foundation/main.tf")
    assert "role_definition_id = local.acr_pull_role_id" in foundation
    assert "role_definition_id = local.acr_push_role_id" in foundation
    assert "data.azurerm_client_config.current.subscription_id" in foundation
    assert "basename(data.azurerm_role_definition.acr_pull.role_definition_id)" in foundation
    assert "basename(data.azurerm_role_definition.acr_push.role_definition_id)" in foundation


def _readiness_component(name: str, mode: str, binding: dict[str, str]) -> dict:
    return build_readiness_component_receipt(
        mode=mode,
        evidence=_fixture(name),
        repository=binding["repository"],
        workflow=binding["workflow"],
        source_sha=binding["source_sha"],
        ref=binding["ref"],
        workflow_run_id=binding["workflow_run_id"],
        run_attempt=binding["run_attempt"],
        observed_at_utc="2026-09-01T12:00:00Z",
    )


def test_readiness_receipt_is_content_addressed_and_binds_all_components() -> None:
    binding = {
        "repository": "tjpoa/wind-energy-forecast",
        "workflow": "Release production",
        "source_sha": "a" * 40,
        "ref": "refs/heads/master",
        "workflow_run_id": "900",
        "run_attempt": "1",
    }
    components = {
        "release-publish": _readiness_component("readiness-publisher.json", "release-publish", binding),
        "release-plan": _readiness_component("readiness-planner.json", "release-plan", binding),
        "release-deploy": _readiness_component("readiness-deployer.json", "release-deploy", binding),
    }
    receipt = build_readiness_receipt(
        components=components,
        required_modes=tuple(components),
        observed_at_utc="2026-09-01T12:01:00Z",
        **binding,
    )
    assert receipt["schema_version"] == READINESS_SCHEMA
    assert receipt["decision"] == "GO"
    assert receipt["allowed_modes"] == list(components)
    assert validate_readiness_receipt(receipt, expected_mode="release-deploy", expected_binding=binding)["receipt_id"] == receipt["receipt_id"]

    tampered = dict(receipt)
    tampered["decision"] = "NO_GO"
    with pytest.raises(WorkflowPolicyError, match="identity"):
        validate_readiness_receipt(tampered)


def test_readiness_blocks_unprotected_environment_and_insecure_state() -> None:
    binding = {
        "repository": "tjpoa/wind-energy-forecast",
        "workflow": "Release production",
        "source_sha": "a" * 40,
        "ref": "refs/heads/master",
        "workflow_run_id": "901",
        "run_attempt": "1",
    }
    deployer = _fixture("readiness-deployer.json")
    deployer["github"]["environment"]["required_reviewer_count"] = 0
    component = build_readiness_component_receipt(
        mode="release-deploy",
        evidence=deployer,
        observed_at_utc="2026-09-01T12:00:00Z",
        **binding,
    )
    assert component["decision"] == "NO_GO"
    assert "GITHUB_ENVIRONMENT_UNPROTECTED" in component["reason_codes"]

    planner = _fixture("readiness-planner.json")
    planner["state"]["storage_account"]["shared_key_enabled"] = True
    component = build_readiness_component_receipt(
        mode="release-plan",
        evidence=planner,
        observed_at_utc="2026-09-01T12:00:00Z",
        **binding,
    )
    assert component["decision"] == "NO_GO"
    assert "STATE_STORAGE_NOT_READY" in component["reason_codes"]

    planner = _fixture("readiness-planner.json")
    planner["state"]["blobs"]["production.tfstate"] = {
        "exists": True,
        "lease_state": "leased",
        "probe_ok": True,
    }
    component = build_readiness_component_receipt(
        mode="release-plan",
        evidence=planner,
        observed_at_utc="2026-09-01T12:00:00Z",
        **binding,
    )
    assert component["decision"] == "NO_GO"
    assert "STATE_TARGET_LOCKED" in component["reason_codes"]


def test_readiness_accepts_custom_environment_policy_that_allows_master() -> None:
    evidence = _fixture("readiness-deployer.json")
    environment = evidence["github"]["environment"]
    environment["deployment_branch_policy"] = {
        "protected_branches": False,
        "custom_branch_policies": True,
        "custom_branches": ["release", "master"],
    }
    component = build_readiness_component_receipt(
        mode="release-deploy",
        evidence=evidence,
        repository="tjpoa/wind-energy-forecast",
        workflow="Release production",
        source_sha="a" * 40,
        ref="refs/heads/master",
        workflow_run_id="904",
        run_attempt="1",
    )
    assert component["decision"] == "GO"


def test_readiness_rejects_an_unexpected_broad_role() -> None:
    evidence = _fixture("readiness-planner.json")
    evidence["permissions"]["assignments"].append(
        {"condition": None, "role": "Owner", "scope": "workload_resource_group"}
    )
    component = build_readiness_component_receipt(
        mode="release-plan",
        evidence=evidence,
        repository="tjpoa/wind-energy-forecast",
        workflow="Release production",
        source_sha="a" * 40,
        ref="refs/heads/master",
        workflow_run_id="905",
        run_attempt="1",
    )
    assert component["decision"] == "NO_GO"
    assert "AZURE_PERMISSION_TOO_BROAD" in component["reason_codes"]


def test_readiness_rejects_changed_deployer_rbac_condition() -> None:
    evidence = _fixture("readiness-deployer.json")
    evidence["permissions"]["assignments"][0]["condition"] = "changed"
    component = build_readiness_component_receipt(
        mode="release-deploy",
        evidence=evidence,
        repository="tjpoa/wind-energy-forecast",
        workflow="Release production",
        source_sha="a" * 40,
        ref="refs/heads/master",
        workflow_run_id="906",
        run_attempt="1",
    )
    assert component["decision"] == "NO_GO"
    assert "AZURE_PERMISSION_MISSING" in component["reason_codes"]


def test_readiness_requires_the_expected_oidc_claims_and_foundation_upstream() -> None:
    publisher = _fixture("readiness-publisher.json")
    publisher["oidc"]["claims"]["subject"] = "repo:tjpoa/wind-energy-forecast:ref:refs/heads/feature"
    component = build_readiness_component_receipt(
        mode="release-publish",
        evidence=publisher,
        repository="tjpoa/wind-energy-forecast",
        workflow="Release production",
        source_sha="a" * 40,
        ref="refs/heads/master",
        workflow_run_id="907",
        run_attempt="1",
    )
    assert component["decision"] == "NO_GO"
    assert "OIDC_CLAIMS_MISMATCH" in component["reason_codes"]

    planner = _fixture("readiness-planner.json")
    planner["mode"] = "foundation-plan"
    planner["state"]["blobs"].pop("foundation.tfstate")
    planner["state"]["blobs"]["bootstrap.tfstate"] = {
        "exists": True,
        "lease_state": "available",
        "probe_ok": True,
    }
    planner["state"]["blobs"]["foundation.tfstate"] = {
        "exists": False,
        "probe_ok": True,
    }
    component = build_readiness_component_receipt(
        mode="foundation-plan",
        evidence=planner,
        repository="tjpoa/wind-energy-forecast",
        workflow="Apply Azure foundation",
        source_sha="a" * 40,
        ref="refs/heads/master",
        workflow_run_id="908",
        run_attempt="1",
    )
    assert component["decision"] == "GO"


def test_readiness_rejects_sensitive_evidence_and_missing_components() -> None:
    evidence = _fixture("readiness-publisher.json")
    evidence["probe_note"] = "password=not-permitted"
    with pytest.raises(WorkflowPolicyError, match="permitted"):
        build_readiness_component_receipt(
            mode="release-publish",
            evidence=evidence,
            repository="tjpoa/wind-energy-forecast",
            workflow="Release production",
            source_sha="a" * 40,
            ref="refs/heads/master",
            workflow_run_id="902",
            run_attempt="1",
        )

    receipt = build_readiness_receipt(
        components={},
        required_modes=("release-publish",),
        repository="tjpoa/wind-energy-forecast",
        workflow="Release production",
        source_sha="a" * 40,
        ref="refs/heads/master",
        workflow_run_id="903",
        run_attempt="1",
        observed_at_utc="2026-09-01T12:00:00Z",
    )
    assert receipt["decision"] == "NO_GO"
    assert receipt["allowed_modes"] == []
    assert receipt["reason_codes"] == ["EVIDENCE_MISSING"]


def test_bootstrap_grants_only_read_control_plane_state_visibility() -> None:
    bootstrap = _repository_text("infra/azure/terraform/bootstrap/main.tf")
    for identity in ("planner", "deployer"):
        marker = f'resource "azurerm_role_assignment" "{identity}_state_control_reader"'
        start = bootstrap.index(marker)
        block = bootstrap[start : bootstrap.index("\n}", start) + 2]
        assert "scope              = azurerm_storage_account.state.id" in block
        assert "role_definition_id = local.reader_role_id" in block
        assert f"azurerm_user_assigned_identity.{identity}.principal_id" in block


def test_terraform_workflows_gate_mutations_on_aggregate_readiness() -> None:
    release = _repository_text(".github/workflows/release-production.yml")
    assert "actions: read" in release
    assert release.index("readiness-gate:") < release.index("docker push")
    assert "needs: readiness-gate" in release
    assert "readiness-receipt" in release

    foundation = _repository_text(".github/workflows/foundation-production.yml")
    assert "actions: read" in foundation
    assert foundation.index("readiness-gate:") < foundation.index("Apply the exact approved foundation plan")
    assert "environment: production" in foundation[foundation.index("readiness-deployer:") : foundation.index("plan-foundation:")]

    rollback = _repository_text(".github/workflows/rollback-production.yml")
    assert rollback.index("readiness-gate:") < rollback.index("Create rollback manifest")
    assert "readiness-receipt" in rollback

    for workflow in (release, foundation, rollback):
        assert workflow.count("contents: read") == 1
        assert workflow.count("actions: read") == 1
        assert workflow.count("id-token: write") == 1
