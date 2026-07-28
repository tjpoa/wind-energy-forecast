from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import wind_forecast.retraining_deployment as deployment
from wind_forecast.manifests import sha256_file
from wind_forecast.retraining_deployment import (
    DeploymentBootstrapConfig,
    RetrainingDeploymentError,
    RetrainingDeploymentReconciliationError,
    bootstrap_v2_deployment,
    load_bootstrap_approval,
    load_verified_deployment_pointer,
)


SHA = "a" * 64
NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)


class Client:
    def __init__(self) -> None:
        self.model_exists = False
        self.aliases: dict[str, str] = {}
        self.tags: dict[str, str] = {}
        self.register_calls = 0

    def get_registered_model(self, name: str):
        if not self.model_exists:
            raise LookupError(name)
        return SimpleNamespace(name=name)

    def search_model_versions(self):
        if not self.model_exists:
            return []
        return [SimpleNamespace(name="wind-v2", version="1")]

    def get_model_version_by_alias(self, name: str, alias: str):
        if not self.model_exists or alias not in self.aliases:
            raise LookupError(alias)
        return SimpleNamespace(version=self.aliases[alias])

    def get_model_version(self, name: str, version: str):
        if not self.model_exists:
            raise LookupError(version)
        return SimpleNamespace(
            name=name,
            version=str(version),
            run_id="run-1",
            tags=dict(self.tags),
        )

    def set_model_version_tag(
        self,
        name: str,
        version: str,
        key: str,
        value: str,
    ) -> None:
        self.tags[key] = value

    def set_registered_model_alias(
        self,
        name: str,
        alias: str,
        version: str,
    ) -> None:
        self.aliases[alias] = str(version)

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        self.aliases.pop(alias, None)

    def get_run(self, run_id: str):
        return SimpleNamespace(info=SimpleNamespace(status="FINISHED"))


class Mlflow:
    def __init__(self, client: Client) -> None:
        self.client = client
        self.tracking_uri = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def MlflowClient(self) -> Client:
        return self.client

    def register_model(self, *, model_uri: str, name: str):
        self.client.register_calls += 1
        self.client.model_exists = True
        return SimpleNamespace(version="1", run_id="run-1")


def _config(tmp_path: Path, **changes) -> DeploymentBootstrapConfig:
    paths = {
        "model_bundle": tmp_path / "bundle",
        "calibration_dir": tmp_path / "calibration",
        "monitoring_store_root": tmp_path / "monitoring",
        "deployment_root": tmp_path / "deployment",
        "registry_lock_root": tmp_path / "registry-lock",
    }
    for path in paths.values():
        if path.name not in {"deployment", "registry-lock"}:
            path.mkdir(parents=True, exist_ok=True)
    (paths["calibration_dir"] / "calibration.json").write_text(
        "{}",
        encoding="utf-8",
    )
    ledger_path = paths["monitoring_store_root"] / "state" / "current.json"
    ledger_path.parent.mkdir()
    ledger_path.write_text("{}", encoding="utf-8")
    values = {
        **paths,
        "registered_model_name": "wind-v2",
        "tracking_uri": "http://127.0.0.1:5000",
        "expect_no_deployment_pointer": True,
        "expect_no_v2_registry_state": True,
        "dry_run": True,
        "now_utc": NOW,
    }
    values.update(changes)
    return DeploymentBootstrapConfig(**values)


def _patch_verified_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deployment,
        "_load_exact_bundle",
        lambda _root: {
            "bundle_sha256": "1" * 64,
            "model_manifest": {
                "model_sha256": "2" * 64,
                "feature_schema_sha256": "3" * 64,
            },
            "dataset_manifest": {
                "sha256": "4" * 64,
                "splits": {"validation": {"end": "2024-12-31"}},
            },
        },
    )
    monkeypatch.setattr(
        deployment,
        "_load_calibration",
        lambda _root: {
            "calibration_id": "calibration-1",
            "reference_id": "reference-1",
            "_reference_manifest": {"reference_id": "reference-1"},
        },
    )
    monkeypatch.setattr(
        deployment,
        "_load_ledger",
        lambda _root: {
            "model_snapshot_id": "5" * 64,
            "activation_date": "2026-01-15",
        },
    )
    monkeypatch.setattr(
        deployment,
        "_load_ledger_snapshot",
        lambda _root, _ledger: {
            "model_snapshot_id": "5" * 64,
            "model": {"model_sha256": "2" * 64},
            "feature_schema_sha256": "3" * 64,
            "dataset": {"dataset_sha256": "4" * 64},
        },
    )
    monkeypatch.setattr(
        deployment,
        "_verify_mlflow",
        lambda config, **_kwargs: {
            "run_id": "run-1",
            "model_uri": "models:/m-approved",
        },
    )


def _approval_for(
    tmp_path: Path,
    plan,
) -> tuple[Path, str]:
    payload = dict(plan.approval_template)
    payload.update(
        {
            "approved_by": "operator@example.test",
            "approved_at_utc": "2026-07-28T11:30:00Z",
            "note": "Approved one-time migration.",
        }
    )
    path = tmp_path / "approval.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, sha256_file(path)


def _approved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: Client,
    mlflow: Mlflow,
) -> DeploymentBootstrapConfig:
    _patch_verified_inputs(monkeypatch)
    dry = _config(tmp_path)
    plan = bootstrap_v2_deployment(
        dry,
        client=client,
        mlflow_module=mlflow,
    ).plan
    approval_path, approval_sha = _approval_for(tmp_path, plan)
    return replace(
        dry,
        dry_run=False,
        approval_path=approval_path,
        approval_sha256=approval_sha,
    )


def _input_bytes(config: DeploymentBootstrapConfig) -> dict[str, bytes]:
    roots = (
        config.model_bundle,
        config.calibration_dir,
        config.monitoring_store_root,
    )
    return {
        str(path.resolve()): path.read_bytes()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }


def test_dry_run_is_read_only_and_emits_checksum_pinned_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_verified_inputs(monkeypatch)
    client = Client()
    mlflow = Mlflow(client)
    config = _config(tmp_path)

    result = bootstrap_v2_deployment(
        config,
        client=client,
        mlflow_module=mlflow,
    )

    assert result.status == "planned"
    assert result.plan.approval_template["bootstrap_exception"] is True
    assert (
        result.plan.approval_template["expected_bundle_sha256"] == "1" * 64
    )
    assert not config.deployment_root.exists()
    assert not config.registry_lock_root.exists()
    assert client.register_calls == 0


def test_bootstrap_initializes_only_stable_and_champion_and_verifies_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_verified_inputs(monkeypatch)
    client = Client()
    mlflow = Mlflow(client)
    dry = _config(tmp_path)
    plan = bootstrap_v2_deployment(
        dry,
        client=client,
        mlflow_module=mlflow,
    ).plan
    approval_path, approval_sha = _approval_for(tmp_path, plan)
    config = replace(
        dry,
        dry_run=False,
        approval_path=approval_path,
        approval_sha256=approval_sha,
    )
    inputs_before = _input_bytes(config)

    result = bootstrap_v2_deployment(
        config,
        client=client,
        mlflow_module=mlflow,
    )
    loaded = load_verified_deployment_pointer(
        config.deployment_root,
        client=client,
    )

    assert result.status == "bootstrapped"
    assert client.aliases == {"stable": "1", "champion": "1"}
    assert loaded["pointer"]["generation"] == 1
    assert loaded["state"]["predecessor"] is None
    assert loaded["receipt"]["bootstrap_exception"] is True
    assert loaded["state"]["pins"] == result.plan.pins
    assert sha256_file(result.state_manifest_path) == (
        loaded["pointer"]["state_manifest_sha256"]
    )
    assert _input_bytes(config) == inputs_before
    assert "candidate" not in client.aliases


def test_existing_pointer_fails_before_registry_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_verified_inputs(monkeypatch)
    config = _config(tmp_path)
    pointer = config.deployment_root / "state" / "current.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("{}", encoding="utf-8")
    client = Client()

    with pytest.raises(RetrainingDeploymentError, match="already exists"):
        bootstrap_v2_deployment(
            config,
            client=client,
            mlflow_module=Mlflow(client),
        )
    assert client.register_calls == 0


def test_existing_registered_model_fails_before_version_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_verified_inputs(monkeypatch)
    client = Client()
    client.model_exists = True

    with pytest.raises(RetrainingDeploymentError, match="already exists"):
        bootstrap_v2_deployment(
            _config(tmp_path),
            client=client,
            mlflow_module=Mlflow(client),
        )
    assert client.register_calls == 0


def test_real_bootstrap_requires_approval_and_exact_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_verified_inputs(monkeypatch)
    client = Client()
    mlflow = Mlflow(client)
    dry = _config(tmp_path)

    with pytest.raises(RetrainingDeploymentError, match="approval-path"):
        bootstrap_v2_deployment(
            replace(dry, dry_run=False),
            client=client,
            mlflow_module=mlflow,
        )
    assert client.register_calls == 0

    plan = bootstrap_v2_deployment(
        dry,
        client=client,
        mlflow_module=mlflow,
    ).plan
    approval_path, _approval_sha = _approval_for(tmp_path, plan)
    with pytest.raises(RetrainingDeploymentError, match="checksum differs"):
        bootstrap_v2_deployment(
            replace(
                dry,
                dry_run=False,
                approval_path=approval_path,
                approval_sha256="f" * 64,
            ),
            client=client,
            mlflow_module=mlflow,
        )
    assert client.register_calls == 0


def test_approval_is_strict_checksum_pinned_and_utc(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "wind_forecast.bootstrap_approval.v1",
        "approved_by": "operator",
        "approved_at_utc": "2026-07-28T12:00:00Z",
        "note": "migration",
        "bootstrap_exception": True,
        "deployment_root": str((tmp_path / "deployment").resolve()),
        "registered_model_name": "wind-v2",
        "run_id": "run-1",
        "model_uri": "models:/m-approved",
        "expected_bundle_sha256": SHA,
        "expected_calibration_sha256": SHA,
        "expected_ledger_sha256": SHA,
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_bootstrap_approval(path) == payload

    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RetrainingDeploymentError, match="fields differ"):
        load_bootstrap_approval(path)


def test_pointer_loader_rejects_absolute_and_traversal_state_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deployment"
    pointer = root / "state" / "current.json"
    pointer.parent.mkdir(parents=True)
    base = {
        "schema_version": "wind_forecast.active_deployment_pointer.v1",
        "generation": 1,
        "deployment_id": "deployment",
        "deployment_state_id": "state",
        "state_manifest_path": "../outside.json",
        "state_manifest_sha256": SHA,
        "updated_at_utc": "2026-07-28T12:00:00Z",
    }
    pointer.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(RetrainingDeploymentError, match="unsafe"):
        load_verified_deployment_pointer(root, client=Client())

    base["state_manifest_path"] = str((tmp_path / "outside.json").resolve())
    pointer.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(RetrainingDeploymentError, match="unsafe"):
        load_verified_deployment_pointer(root, client=Client())


def test_alias_failure_compensates_and_seals_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_verified_inputs(monkeypatch)

    class FailingClient(Client):
        def set_registered_model_alias(
            self,
            name: str,
            alias: str,
            version: str,
        ) -> None:
            if alias == "champion":
                raise RuntimeError("alias write failed")
            super().set_registered_model_alias(name, alias, version)

    client = FailingClient()
    mlflow = Mlflow(client)
    dry = _config(tmp_path)
    plan = bootstrap_v2_deployment(
        dry,
        client=client,
        mlflow_module=mlflow,
    ).plan
    approval_path, approval_sha = _approval_for(tmp_path, plan)

    with pytest.raises(RetrainingDeploymentError, match="Manual reconciliation"):
        bootstrap_v2_deployment(
            replace(
                dry,
                dry_run=False,
                approval_path=approval_path,
                approval_sha256=approval_sha,
            ),
            client=client,
            mlflow_module=mlflow,
        )

    assert client.aliases == {}
    assert not (dry.deployment_root / "state" / "current.json").exists()
    assert list((dry.deployment_root / "reconciliation").glob("*.json"))


def test_champion_race_is_detected_before_overwrite_and_stable_is_compensated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RacingClient(Client):
        def get_model_version_by_alias(self, name: str, alias: str):
            if alias == "champion" and self.aliases.get("stable") == "1":
                self.aliases["champion"] = "99"
            return super().get_model_version_by_alias(name, alias)

    client = RacingClient()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)

    with pytest.raises(
        RetrainingDeploymentReconciliationError,
        match="compensation was incomplete",
    ):
        bootstrap_v2_deployment(
            config,
            client=client,
            mlflow_module=mlflow,
        )

    assert client.aliases == {"champion": "99"}
    assert not (config.deployment_root / "state" / "current.json").exists()


def test_registry_version_race_is_detected_before_alias_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VersionRacingClient(Client):
        search_reads = 0

        def search_model_versions(self):
            self.search_reads += 1
            versions = super().search_model_versions()
            if self.model_exists and self.search_reads >= 3:
                return [
                    *versions,
                    SimpleNamespace(name="wind-v2", version="2"),
                ]
            return versions

    client = VersionRacingClient()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)

    with pytest.raises(RetrainingDeploymentError, match="Manual reconciliation"):
        bootstrap_v2_deployment(
            config,
            client=client,
            mlflow_module=mlflow,
        )

    assert client.aliases == {}
    assert not (config.deployment_root / "state" / "current.json").exists()


def test_compensation_rechecks_alias_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StickyClient(Client):
        def set_registered_model_alias(
            self,
            name: str,
            alias: str,
            version: str,
        ) -> None:
            if alias == "champion":
                raise RuntimeError("champion failed")
            super().set_registered_model_alias(name, alias, version)

        def delete_registered_model_alias(self, name: str, alias: str) -> None:
            return None

    client = StickyClient()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)

    with pytest.raises(
        RetrainingDeploymentReconciliationError,
        match="compensation was incomplete",
    ):
        bootstrap_v2_deployment(
            config,
            client=client,
            mlflow_module=mlflow,
        )

    evidence = json.loads(
        next((config.deployment_root / "reconciliation").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["alias_compensation_errors"]


def test_pointer_cleanup_failure_is_post_publication_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)
    original_unlink = Path.unlink

    def fail_current_temp_unlink(path: Path, *args, **kwargs):
        if path.name.startswith(".current.") and path.suffix == ".tmp":
            raise OSError("simulated cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_current_temp_unlink)

    with pytest.raises(
        RetrainingDeploymentReconciliationError,
        match="pointer was published",
    ):
        bootstrap_v2_deployment(
            config,
            client=client,
            mlflow_module=mlflow,
        )

    assert (config.deployment_root / "state" / "current.json").is_file()
    assert client.aliases == {"stable": "1", "champion": "1"}
    evidence = json.loads(
        next((config.deployment_root / "reconciliation").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["pointer_published"] is True
    assert list((config.deployment_root / "state").glob(".current.*.tmp"))


def test_post_pointer_failure_preserves_aliases_and_records_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)

    def fail_postcheck(*_args, **_kwargs):
        raise RetrainingDeploymentError("postcheck failed")

    monkeypatch.setattr(
        deployment,
        "load_verified_deployment_pointer",
        fail_postcheck,
    )

    with pytest.raises(
        RetrainingDeploymentReconciliationError,
        match="pointer was published",
    ):
        bootstrap_v2_deployment(
            config,
            client=client,
            mlflow_module=mlflow,
        )

    assert (config.deployment_root / "state" / "current.json").is_file()
    assert client.aliases == {"stable": "1", "champion": "1"}
    evidence = json.loads(
        next((config.deployment_root / "reconciliation").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["pointer_published"] is True


@pytest.mark.parametrize(
    "failure_stage",
    [
        "tag",
        "tag_persistence",
        "receipt",
        "state",
        "stable",
        "champion",
        "pointer",
    ],
)
def test_pre_pointer_failures_compensate_and_leave_reconciliation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    class FailingClient(Client):
        def set_model_version_tag(
            self,
            name: str,
            version: str,
            key: str,
            value: str,
        ) -> None:
            if failure_stage == "tag":
                raise RuntimeError("tag failed")
            if failure_stage == "tag_persistence":
                return None
            super().set_model_version_tag(name, version, key, value)

        def set_registered_model_alias(
            self,
            name: str,
            alias: str,
            version: str,
        ) -> None:
            if failure_stage == alias:
                raise RuntimeError(f"{alias} failed")
            super().set_registered_model_alias(name, alias, version)

    client = FailingClient()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)
    if failure_stage == "receipt":
        monkeypatch.setattr(
            deployment,
            "_seal_bootstrap_receipt",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("receipt failed")
            ),
        )
    if failure_stage == "state":
        monkeypatch.setattr(
            deployment,
            "_seal_deployment_state",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("state failed")
            ),
        )
    if failure_stage == "pointer":
        monkeypatch.setattr(
            deployment,
            "_publish_pointer",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("pointer failed")
            ),
        )

    with pytest.raises(RetrainingDeploymentError, match="Manual reconciliation"):
        bootstrap_v2_deployment(
            config,
            client=client,
            mlflow_module=mlflow,
        )

    assert client.aliases == {}
    assert not (config.deployment_root / "state" / "current.json").exists()
    assert list((config.deployment_root / "reconciliation").glob("*.json"))


def test_pointer_create_if_absent_collision_preserves_racing_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)

    def racing_link(_source: Path, destination: Path) -> None:
        destination.write_text('{"racing": true}\n', encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(deployment.os, "link", racing_link)

    with pytest.raises(RetrainingDeploymentError, match="Manual reconciliation"):
        bootstrap_v2_deployment(
            config,
            client=client,
            mlflow_module=mlflow,
        )

    pointer = config.deployment_root / "state" / "current.json"
    assert json.loads(pointer.read_text(encoding="utf-8")) == {"racing": True}
    assert client.aliases == {}
    evidence = json.loads(
        next((config.deployment_root / "reconciliation").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["pointer_published"] is False


def test_receipt_loader_rejects_semantically_inconsistent_embedded_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)
    result = bootstrap_v2_deployment(
        config,
        client=client,
        mlflow_module=mlflow,
    )
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    receipt["approval"]["run_id"] = "different-run"
    receipt["approval_payload_sha256"] = deployment.sha256(
        deployment._canonical(receipt["approval"])
    ).hexdigest()
    body = {
        key: value
        for key, value in receipt.items()
        if key != "bootstrap_receipt_id"
    }
    receipt_id = deployment._identifier("bootstrap_receipt", body)
    receipt["bootstrap_receipt_id"] = receipt_id
    path = config.deployment_root / "receipts" / receipt_id / "receipt.json"
    path.parent.mkdir()
    path.write_bytes(deployment._json_bytes(receipt))

    with pytest.raises(RetrainingDeploymentError, match="approval pins differ"):
        deployment.load_bootstrap_receipt(path)


@pytest.mark.parametrize(
    "corruption",
    ["pointer_fields", "state_checksum", "state_identity", "receipt_bytes"],
)
def test_pointer_loader_rejects_corrupt_chain_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    client = Client()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)
    result = bootstrap_v2_deployment(
        config,
        client=client,
        mlflow_module=mlflow,
    )
    pointer = json.loads(result.pointer_path.read_text(encoding="utf-8"))
    if corruption == "pointer_fields":
        pointer["unexpected"] = True
        result.pointer_path.write_bytes(deployment._json_bytes(pointer))
    elif corruption == "state_checksum":
        pointer["state_manifest_sha256"] = "0" * 64
        result.pointer_path.write_bytes(deployment._json_bytes(pointer))
    elif corruption == "state_identity":
        state = json.loads(result.state_manifest_path.read_text(encoding="utf-8"))
        state["deployment_state_id"] = "f" * 64
        result.state_manifest_path.write_bytes(deployment._json_bytes(state))
        pointer["deployment_state_id"] = "f" * 64
        pointer["state_manifest_sha256"] = sha256_file(
            result.state_manifest_path
        )
        result.pointer_path.write_bytes(deployment._json_bytes(pointer))
    else:
        receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
        receipt["unexpected"] = True
        result.receipt_path.write_bytes(deployment._json_bytes(receipt))

    with pytest.raises(RetrainingDeploymentError):
        load_verified_deployment_pointer(
            config.deployment_root,
            client=client,
        )


def test_receipt_loader_rejects_missing_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)
    result = bootstrap_v2_deployment(
        config,
        client=client,
        mlflow_module=mlflow,
    )
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    receipt.pop("expected_aliases")
    result.receipt_path.write_bytes(deployment._json_bytes(receipt))

    with pytest.raises(RetrainingDeploymentError, match="fields differ"):
        deployment.load_bootstrap_receipt(result.receipt_path)


@pytest.mark.parametrize("kind", ["pointer", "state", "receipt"])
def test_pointer_loader_rejects_symlinked_chain_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    client = Client()
    mlflow = Mlflow(client)
    config = _approved_config(tmp_path, monkeypatch, client, mlflow)
    result = bootstrap_v2_deployment(
        config,
        client=client,
        mlflow_module=mlflow,
    )
    selected = {
        "pointer": result.pointer_path,
        "state": result.state_manifest_path,
        "receipt": result.receipt_path,
    }[kind]
    target = selected.with_name(f"{selected.name}.real")
    selected.rename(target)
    try:
        selected.symlink_to(target)
    except OSError:
        target.rename(selected)
        pytest.skip("Symlink creation is unavailable on this platform.")

    with pytest.raises(RetrainingDeploymentError, match="symlink"):
        load_verified_deployment_pointer(
            config.deployment_root,
            client=client,
        )


def test_public_json_loader_rejects_approval_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "approval-target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "approval-link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform.")

    with pytest.raises(RetrainingDeploymentError, match="non-symlink"):
        load_bootstrap_approval(link)


def test_missing_ledger_fails_before_mlflow_or_registry_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    (config.monitoring_store_root / "state" / "current.json").unlink()
    monkeypatch.setattr(
        deployment,
        "_load_exact_bundle",
        lambda _root: {"unused": True},
    )
    monkeypatch.setattr(
        deployment,
        "_load_calibration",
        lambda _root: {"unused": True},
    )

    class UntouchedClient(Client):
        def get_registered_model(self, name: str):
            raise AssertionError("Registry must not be read without a ledger.")

    client = UntouchedClient()
    with pytest.raises(RetrainingDeploymentError, match="ledger"):
        bootstrap_v2_deployment(
            config,
            client=client,
            mlflow_module=Mlflow(client),
        )
    assert client.register_calls == 0


@pytest.mark.parametrize("evidence", ["calibration", "snapshot"])
def test_incompatible_calibration_or_ledger_snapshot_fails_before_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: str,
) -> None:
    _patch_verified_inputs(monkeypatch)
    if evidence == "calibration":
        monkeypatch.setattr(
            deployment,
            "_load_calibration",
            lambda _root: {
                "calibration_id": "calibration-1",
                "reference_id": "reference-1",
                "_reference_manifest": {"reference_id": "different"},
            },
        )
    else:
        monkeypatch.setattr(
            deployment,
            "_load_ledger_snapshot",
            lambda _root, _ledger: {
                "model_snapshot_id": "5" * 64,
                "model": {"model_sha256": "9" * 64},
                "feature_schema_sha256": "3" * 64,
                "dataset": {"dataset_sha256": "4" * 64},
            },
        )
    client = Client()

    with pytest.raises(RetrainingDeploymentError):
        bootstrap_v2_deployment(
            _config(tmp_path),
            client=client,
            mlflow_module=Mlflow(client),
        )
    assert client.register_calls == 0


def test_exact_bundle_rejects_extra_files_before_validation(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in deployment._BUNDLE_FILES:
        (bundle / name).write_bytes(b"fixture")
    (bundle / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(RetrainingDeploymentError, match="file set differs"):
        deployment._load_exact_bundle(bundle)


def _mlflow_validation_fixture(tmp_path: Path):
    features = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    model = RandomForestRegressor(n_estimators=2, random_state=1).fit(
        features,
        np.asarray([2.0, 4.0, 6.0, 8.0]),
    )
    expected = model.predict(features)
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    pd.DataFrame(
        {
            "feature": features["feature"],
            "Expected_Prediction": expected,
        }
    ).to_csv(bundle_root / "reload_sample.csv", index=False)
    model_uri = "models:/m-approved"
    bundle = {
        "mlflow_receipt": {
            "experiment_id": "experiment-1",
            "model_uri": model_uri,
            "run_id": "run-1",
            "tracking_uri": "http://127.0.0.1:5000",
        },
        "mlflow_reload_validation": {
            "model_uri": model_uri,
            "row_count": len(expected),
            "rtol": 1e-12,
            "atol": 1e-9,
            "max_absolute_difference": 0.0,
        },
        "model_manifest": {
            "model_type": "random_forest",
            "parameters": model.get_params(deep=True),
        },
        "dataset_manifest": {
            "sha256": "4" * 64,
            "split_assignment_sha256": "6" * 64,
        },
        "summary": {"selected_model": "random_forest"},
        "feature_names": ["feature"],
    }
    params = {
        "workflow": "train_v2_reference",
        "dataset_version": "v2",
        "dataset_sha256": "4" * 64,
        "split_assignment_sha256": "6" * 64,
        "feature_count": "1",
        "scaler_required": "False",
        "logged_model_uri": model_uri,
        "selected_model": "random_forest",
        "seed": "1",
        "n_estimators": "2",
    }
    tags = {
        "forecast_contract": "historical_daily_hindcast",
        "reference_gate_passed": "True",
        "reference_status": "selected_not_promoted",
        "registry_used": "False",
        "automatic_promotion": "False",
    }

    def numeric(name: str) -> SimpleNamespace:
        return SimpleNamespace(name=name, type="double")

    signature = SimpleNamespace(
        inputs=SimpleNamespace(inputs=[numeric("feature")]),
        outputs=SimpleNamespace(inputs=[numeric("prediction")]),
    )
    client = SimpleNamespace(
        get_run=lambda _run_id: SimpleNamespace(
            info=SimpleNamespace(
                status="FINISHED",
                experiment_id="experiment-1",
            ),
            data=SimpleNamespace(params=params, tags=tags),
        )
    )
    mlflow = SimpleNamespace(
        models=SimpleNamespace(
            get_model_info=lambda _uri: SimpleNamespace(
                run_id="run-1",
                signature=signature,
            )
        ),
        pyfunc=SimpleNamespace(load_model=lambda _uri: model),
    )
    config = _config(tmp_path, model_bundle=bundle_root)
    return config, bundle, client, mlflow, model, params


def test_mlflow_validation_checks_complete_expected_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, bundle, client, mlflow, model, _params = _mlflow_validation_fixture(
        tmp_path
    )
    monkeypatch.setattr(deployment.joblib, "load", lambda _path: model)

    assert deployment._verify_mlflow(
        config,
        bundle=bundle,
        client=client,
        mlflow=mlflow,
    ) == {"run_id": "run-1", "model_uri": "models:/m-approved"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "RUNNING", "not FINISHED"),
        ("experiment_id", "other", "experiment identities differ"),
        ("selected_model", "extra_trees", "selected_model differs"),
    ],
)
def test_mlflow_validation_rejects_incompatible_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    config, bundle, client, mlflow, model, params = _mlflow_validation_fixture(
        tmp_path
    )
    original_run = client.get_run("run-1")
    if field == "selected_model":
        params[field] = value
    else:
        setattr(original_run.info, field, value)
        client.get_run = lambda _run_id: original_run
    monkeypatch.setattr(deployment.joblib, "load", lambda _path: model)

    with pytest.raises(RetrainingDeploymentError, match=message):
        deployment._verify_mlflow(
            config,
            bundle=bundle,
            client=client,
            mlflow=mlflow,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("model_uri", "model URIs differ"),
        ("signature", "no signature"),
        ("prediction", "predictions differ"),
        ("row_count", "predictions differ"),
        ("rtol", "predictions differ"),
        ("max_difference", "predictions differ"),
    ],
)
def test_mlflow_validation_rejects_uri_signature_and_reload_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    config, bundle, client, mlflow, model, _params = _mlflow_validation_fixture(
        tmp_path
    )
    if case == "model_uri":
        bundle["mlflow_reload_validation"]["model_uri"] = "models:/other"
    elif case == "signature":
        mlflow.models.get_model_info = lambda _uri: SimpleNamespace(
            run_id="run-1",
            signature=None,
        )
    elif case == "prediction":
        mlflow.pyfunc.load_model = lambda _uri: SimpleNamespace(
            predict=lambda frame: model.predict(frame) + 1.0
        )
    elif case == "row_count":
        bundle["mlflow_reload_validation"]["row_count"] = 99
    elif case == "rtol":
        bundle["mlflow_reload_validation"]["rtol"] = 1e-6
    else:
        bundle["mlflow_reload_validation"]["max_absolute_difference"] = 1.0
    monkeypatch.setattr(deployment.joblib, "load", lambda _path: model)

    with pytest.raises(RetrainingDeploymentError, match=message):
        deployment._verify_mlflow(
            config,
            bundle=bundle,
            client=client,
            mlflow=mlflow,
        )
