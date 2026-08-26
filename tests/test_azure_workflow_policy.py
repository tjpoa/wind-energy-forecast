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


def _fixture(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


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
