from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import wind_forecast.retraining_lifecycle as lifecycle
from wind_forecast.deployment_runtime import verify_active_model_era
from wind_forecast.manifests import sha256_file
from wind_forecast.retraining_deployment import (
    RetrainingDeploymentError,
    load_verified_deployment_pointer,
)
from wind_forecast.retraining_lifecycle import (
    ExpectedDeploymentState,
    LifecycleConfig,
    LifecyclePlan,
    RetrainingLifecycleError,
    RetrainingLifecycleReconciliationError,
)
from wind_forecast.retraining_policy import ActiveDeploymentPointer


SHA = "a" * 64
NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)


class _LifecycleClient:
    def __init__(self) -> None:
        self.aliases = {"candidate": "2", "champion": "1", "stable": "1"}
        self.tags = {"validation_status": "passed"}

    def get_model_version_by_alias(self, _name, alias):
        if alias not in self.aliases:
            raise KeyError("missing")
        return SimpleNamespace(version=self.aliases[alias])

    def get_model_version(self, _name, version):
        return SimpleNamespace(version=str(version), run_id="run-2", tags=self.tags)

    def set_registered_model_alias(self, _name, alias, version):
        self.aliases[alias] = str(version)

    def delete_registered_model_alias(self, _name, alias):
        self.aliases.pop(alias, None)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _promotion_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LifecycleConfig, _LifecycleClient, dict]:
    root = tmp_path / "deploy"
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    _write_json(candidate / "bundle_manifest.json", {"accepted": True})
    (candidate / "model.joblib").write_bytes(b"candidate-model")
    incumbent = tmp_path / "incumbent"
    incumbent.mkdir()
    _write_json(incumbent / "run_summary.json", {"accepted": True})
    (incumbent / "model.joblib").write_bytes(b"incumbent-model")

    candidate_reference = tmp_path / "candidate-reference"
    _write_json(candidate_reference / "manifest.json", {"reference": "candidate"})
    (candidate_reference / "reference.csv").write_text("x\n1\n", encoding="utf-8")
    candidate_calibration = tmp_path / "candidate-calibration"
    _write_json(candidate_calibration / "calibration.json", {"candidate": True})
    _write_json(candidate_calibration / "backtest_summary.json", {})
    incumbent_reference = tmp_path / "incumbent-reference"
    _write_json(incumbent_reference / "manifest.json", {"reference": "incumbent"})
    (incumbent_reference / "reference.csv").write_text("x\n1\n", encoding="utf-8")
    incumbent_calibration = tmp_path / "incumbent-calibration"
    _write_json(incumbent_calibration / "calibration.json", {"incumbent": True})
    _write_json(incumbent_calibration / "backtest_summary.json", {})
    registration = tmp_path / "registration" / "receipt.json"
    _write_json(registration, {"registered": True})

    old_state_id = "1" * 64
    old_state_path = root / "states" / old_state_id / "state.json"
    _write_json(old_state_path, {"deployment_state_id": old_state_id})
    pointer_path = root / "state" / "current.json"
    pointer = ActiveDeploymentPointer(
        generation=1,
        deployment_id="2" * 64,
        deployment_state_id=old_state_id,
        state_manifest_path=f"states/{old_state_id}/state.json",
        state_manifest_sha256=sha256_file(old_state_path),
        updated_at_utc="2026-07-28T10:00:00Z",
    )
    _write_json(pointer_path, pointer.to_dict())
    pointer_sha = sha256_file(pointer_path)

    candidate_model_sha = "3" * 64
    candidate_bundle_sha = sha256_file(candidate / "bundle_manifest.json")
    incumbent_bundle_sha = "4" * 64
    incumbent_calibration_sha = sha256_file(
        incumbent_calibration / "calibration.json"
    )
    backtest = {
        "backtest_id": "backtest-accepted",
        "outcome": "accepted",
        "cutoffs": {
            "candidate_fit_cutoff": "2026-04-30",
            "monitoring_evaluation_cutoff": "2026-05-31",
        },
        "final_training": {
            "candidate_model_sha256": candidate_model_sha,
            "dataset_sha256": "5" * 64,
        },
        "identities": {"feature_schema_sha256": "6" * 64},
    }
    sealed_backtest = {"backtest_id": "backtest-accepted", "backtest": backtest}
    registration_payload = {
        "registered_model_name": "wind-v2",
        "model_version": "2",
        "backtest_id": "backtest-accepted",
        "champion_after": "1",
        "stable_after": "1",
        "candidate_model_sha256": candidate_model_sha,
        "run_id": "run-2",
        "model_uri": "models:/wind-v2/2",
        "tags": {"validation_status": "passed"},
    }
    candidate_calibration_payload = {
        "calibration_id": "candidate-cal",
        "reference_id": "candidate-ref",
        "_reference_manifest": {
            "model_sha256": candidate_model_sha,
            "calibration_subject": {"backtest_id": "backtest-accepted"},
        },
        "_reference_path": str(candidate_reference / "reference.csv"),
    }
    incumbent_calibration_payload = {
        "calibration_id": "incumbent-cal",
        "reference_id": "incumbent-ref",
        "_reference_path": str(incumbent_reference / "reference.csv"),
    }
    old_state = {
        "schema_version": "wind_forecast.deployment_state.v1",
        "deployment_state_id": old_state_id,
        "deployment_id": "2" * 64,
        "generation": 1,
        "registry": {
            "tracking_uri": "file:mlruns",
            "registered_model_name": "wind-v2",
            "model_version": "1",
            "run_id": "run-1",
            "model_uri": "models:/wind-v2/1",
        },
        "expected_aliases": {
            "candidate": "2",
            "champion": "1",
            "stable": "1",
        },
        "pins": {
            "bundle_sha256": incumbent_bundle_sha,
            "calibration_sha256": incumbent_calibration_sha,
            "model_sha256": "7" * 64,
            "dataset_sha256": "8" * 64,
            "feature_schema_sha256": "6" * 64,
        },
        "calibration": {
            "calibration_id": "incumbent-cal",
            "reference_id": "incumbent-ref",
        },
        "monitoring": {},
        "cutoffs": {
            "fit_cutoff": "2025-12-31",
            "activation_cutoff": "2026-01-15",
        },
    }
    current = {
        "pointer": pointer.to_dict(),
        "state": old_state,
        "pointer_path": str(pointer_path),
        "state_manifest_path": str(old_state_path),
        "receipt_path": str(tmp_path / "bootstrap-receipt.json"),
    }
    config = LifecycleConfig(
        action="promote",
        deployment_root=root,
        registry_lock_root=tmp_path / "lock",
        registered_model_name="wind-v2",
        tracking_uri="file:mlruns",
        expected=ExpectedDeploymentState(
            generation=1,
            deployment_state_id=old_state_id,
            pointer_sha256=pointer_sha,
            candidate="2",
            champion="1",
            stable="1",
        ),
        dry_run=True,
        candidate_bundle=candidate,
        candidate_calibration=candidate_calibration,
        incumbent_bundle=incumbent,
        incumbent_calibration=incumbent_calibration,
        registration_receipt=registration,
        promotion_effective_date=date(2026, 6, 1),
        now_utc=NOW,
    )
    monkeypatch.setattr(
        lifecycle,
        "load_retraining_registration_receipt",
        lambda _path: registration_payload,
    )
    monkeypatch.setattr(
        lifecycle, "load_retraining_backtest", lambda _path: sealed_backtest
    )
    monkeypatch.setattr(
        lifecycle,
        "load_monitoring_calibration",
        lambda path: (
            candidate_calibration_payload
            if Path(path) == candidate_calibration
            else incumbent_calibration_payload
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "_load_runtime_bundle",
        lambda _path: {"bundle_sha256": incumbent_bundle_sha},
    )
    import wind_forecast.retraining_deployment as deployment

    monkeypatch.setattr(
        deployment,
        "load_exact_v2_bundle",
        lambda _path: {"bundle_sha256": incumbent_bundle_sha},
    )
    monkeypatch.setattr(lifecycle, "_verified_current", lambda *_args: current)
    return config, _LifecycleClient(), {
        "current": current,
        "candidate_bundle_sha": candidate_bundle_sha,
        "candidate_calibration": candidate_calibration_payload,
    }


def _config(tmp_path: Path, action: str = "stabilize", **changes) -> LifecycleConfig:
    values = {
        "action": action,
        "deployment_root": tmp_path / "deploy",
        "registry_lock_root": tmp_path / "lock",
        "registered_model_name": "wind-v2",
        "tracking_uri": "file:mlruns",
        "expected": ExpectedDeploymentState(
            generation=2,
            deployment_state_id="b" * 64,
            pointer_sha256=SHA,
            candidate=None,
            champion="2",
            stable="1",
        ),
    }
    values.update(changes)
    return LifecycleConfig(**values)


def _probationary_state() -> dict:
    return {
        "schema_version": lifecycle.STATE_SCHEMA,
        "deployment_state_id": "b" * 64,
        "deployment_id": "c" * 64,
        "generation": 2,
        "lifecycle_status": "probationary",
        "registry": {"model_version": "2"},
        "expected_aliases": {
            "candidate": None,
            "champion": "2",
            "stable": "1",
        },
        "cutoffs": {
            "fit_cutoff": "2026-01-01",
            "promotion_effective_date": "2026-02-01",
            "observation_cutoff": None,
        },
        "rollback_target": {
            "deployment_state_id": "d" * 64,
            "state_manifest_sha256": "e" * 64,
            "path": "states/old/state.json",
            "expected_aliases": {
                "candidate": None,
                "champion": "1",
                "stable": "1",
            },
            "promotion_receipt_id": "r" * 64,
        },
    }


def _stability_inputs(tmp_path: Path, count: int) -> LifecycleConfig:
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "automation": {"automatic_stability": False},
                "stability": {
                    "allowed_issuance_kinds": ["scheduled", "catch_up"],
                    "minimum_eligible_observations": 90,
                    "require_no_active_warning_or_critical": True,
                    "require_second_manual_approval": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monitoring_policy = tmp_path / "monitoring-policy.json"
    monitoring_policy.write_text(
        json.dumps(
            {
                "schema_version": "wind_forecast.monitoring_policy.v1",
                "windows_days": [30, 90],
            }
        ),
        encoding="utf-8",
    )
    return _config(
        tmp_path,
        monitoring_store_root=tmp_path / "monitoring",
        monitoring_report=report,
        policy_path=policy,
        monitoring_policy_path=monitoring_policy,
        observation_cutoff=date(2026, 6, 1),
    )


@pytest.mark.parametrize("count", [89, 91])
def test_stabilization_requires_exactly_90_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    config = _stability_inputs(tmp_path, count)
    days = {
        f"2026-02-{index + 1:02d}" if index < 28 else f"2026-03-{index - 27:02d}": f"p{index}"
        for index in range(min(count, 59))
    }
    # Use a synthetic chronological range without relying on calendar continuity.
    days = {
        (date(2026, 2, 1).fromordinal(date(2026, 2, 1).toordinal() + index)).isoformat(): f"p{index}"
        for index in range(count)
    }
    monitoring_policy_payload = json.loads(
        config.monitoring_policy_path.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        lifecycle,
        "load_monitoring_report",
        lambda _path: {
            "report_id": "report",
            "through_date": "2026-06-01",
            "active_alerts": {},
            "breaches": [],
            "quality": {"issues": []},
            "config": monitoring_policy_payload,
            "reference": {
                "policy_sha256": sha256(
                    config.monitoring_policy_path.read_bytes()
                ).hexdigest()
            },
            "model_era": {
                "deployment_id": "c" * 64,
                "model_version": "2",
                "model_era_id": "era",
            },
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "load_monitoring_report_state",
        lambda _root: {
            "latest_report_id": "report",
            "latest_through_date": "2026-06-01",
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "load_verified_monitoring_state",
        lambda _root: {
            "active_model_era_id": "era",
            "as_issued": days,
            "actuals": {key: "a" for key in days},
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "load_prediction_evidence",
        lambda _root, _prediction: {
            "prediction": {
                "model_era_id": "era",
                "issuance_kind": "scheduled",
                "target_date": next(
                    day for day, value in days.items() if value == _prediction
                ),
                "prediction": 1.0,
            },
            "actual_revisions": [
                {
                    "actual_revision_id": "a",
                    "target_date": next(
                        day for day, value in days.items() if value == _prediction
                    ),
                    "actual": 1.0,
                }
            ],
        },
    )
    with pytest.raises(RetrainingLifecycleError, match="exactly 90"):
        lifecycle._plan_stabilization(config, _probationary_state())


def test_stabilization_accepts_exactly_90_and_moves_only_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _stability_inputs(tmp_path, 90)
    days = {
        date.fromordinal(date(2026, 2, 1).toordinal() + index).isoformat(): f"p{index}"
        for index in range(90)
    }
    monitoring_policy_payload = json.loads(
        config.monitoring_policy_path.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        lifecycle,
        "load_monitoring_report",
        lambda _path: {
            "report_id": "report",
            "through_date": "2026-06-01",
            "active_alerts": {},
            "breaches": [],
            "quality": {"issues": []},
            "config": monitoring_policy_payload,
            "reference": {
                "policy_sha256": sha256(
                    config.monitoring_policy_path.read_bytes()
                ).hexdigest()
            },
            "model_era": {
                "deployment_id": "c" * 64,
                "model_version": "2",
                "model_era_id": "era",
            },
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "load_monitoring_report_state",
        lambda _root: {
            "latest_report_id": "report",
            "latest_through_date": "2026-06-01",
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "load_verified_monitoring_state",
        lambda _root: {
            "active_model_era_id": "era",
            "as_issued": days,
            "actuals": {key: "a" for key in days},
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "load_prediction_evidence",
        lambda _root, _prediction: {
            "prediction": {
                "model_era_id": "era",
                "issuance_kind": "catch_up",
                "target_date": next(
                    day for day, value in days.items() if value == _prediction
                ),
                "prediction": 1.0,
            },
            "actual_revisions": [
                {
                    "actual_revision_id": "a",
                    "target_date": next(
                        day for day, value in days.items() if value == _prediction
                    ),
                    "actual": 1.0,
                }
            ],
        },
    )
    aliases, evidence = lifecycle._plan_stabilization(
        config, _probationary_state()
    )
    assert aliases == {"candidate": None, "champion": "2", "stable": "2"}
    assert evidence["eligible_observation_count"] == 90


def test_stabilization_blocks_active_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _stability_inputs(tmp_path, 90)
    monkeypatch.setattr(
        lifecycle,
        "load_monitoring_report",
        lambda _path: {
            "active_alerts": {"rule": "alert"},
            "breaches": [],
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "load_monitoring_report_state",
        lambda _root: None,
    )
    with pytest.raises(RetrainingLifecycleError, match="warning or critical"):
        lifecycle._plan_stabilization(config, _probationary_state())


def test_rollback_is_limited_to_promotion_fixed_last_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "promotion.json"
    receipt_path.write_text("{}", encoding="utf-8")
    config = _config(
        tmp_path,
        action="rollback",
        promotion_receipt=receipt_path,
        expected_rollback_state_id="d" * 64,
    )
    monkeypatch.setattr(
        lifecycle,
        "load_transition_receipt",
        lambda _path: {
            "action": "promote",
            "transition_receipt_id": "r" * 64,
            "rollback_target_state_id": "d" * 64,
        },
    )
    aliases, evidence = lifecycle._plan_rollback(
        config, _probationary_state()
    )
    assert aliases == {"candidate": None, "champion": "1", "stable": "1"}
    assert evidence["rollback_target_state_id"] == "d" * 64

    wrong = _config(
        tmp_path,
        action="rollback",
        promotion_receipt=receipt_path,
        expected_rollback_state_id="f" * 64,
    )
    with pytest.raises(RetrainingLifecycleError, match="last stable"):
        lifecycle._plan_rollback(wrong, _probationary_state())


def test_pointer_publication_uses_replace_after_reverification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "deploy"
    pointer_path = root / "state" / "current.json"
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_text("original", encoding="utf-8")
    config = _config(
        tmp_path,
        expected=ExpectedDeploymentState(
            generation=2,
            deployment_state_id="b" * 64,
            pointer_sha256=sha256(b"original").hexdigest(),
            candidate=None,
            champion="2",
            stable="1",
        ),
    )
    state_path = root / "states" / "next" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("state", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "_require_original_pointer",
        lambda _config: calls.append("verify"),
    )
    real_replace = lifecycle.os.replace

    def replace(source, target):
        calls.append("replace")
        return real_replace(source, target)

    monkeypatch.setattr(lifecycle.os, "replace", replace)
    lifecycle._publish_pointer(
        config,
        {
            "generation": 3,
            "deployment_id": "c" * 64,
            "deployment_state_id": "d" * 64,
        },
        state_path,
    )
    assert calls == ["verify", "verify", "replace"]


def test_cli_contract_rejects_v1_model_name(tmp_path: Path) -> None:
    with pytest.raises(RetrainingLifecycleError, match="non-V1"):
        _config(
            tmp_path,
            registered_model_name="wind-energy-forecast-original",
        )


def test_promotion_effective_date_must_follow_monitoring_cutoff() -> None:
    backtest = {
        "cutoffs": {
            "candidate_fit_cutoff": "2026-04-30",
            "monitoring_evaluation_cutoff": "2026-05-31",
        }
    }
    with pytest.raises(
        RetrainingLifecycleError,
        match="monitoring evaluation cutoff",
    ):
        lifecycle._require_promotion_after_monitoring(
            date(2026, 5, 31),
            backtest,
        )
    lifecycle._require_promotion_after_monitoring(
        date(2026, 6, 1),
        backtest,
    )


def test_stale_approval_checksum_fails_before_transition(tmp_path: Path) -> None:
    approval = tmp_path / "approval.json"
    approval.write_text("{}", encoding="utf-8")
    config = _config(
        tmp_path,
        approval_path=approval,
        approval_sha256="f" * 64,
    )
    with pytest.raises(RetrainingLifecycleError, match="checksum differs"):
        lifecycle._load_approval(config, {})


def test_alias_compensation_is_compare_and_set_safe(tmp_path: Path) -> None:
    class Client:
        aliases = {"candidate": None, "champion": "2", "stable": "2"}

        def get_model_version_by_alias(self, _name, alias):
            value = self.aliases.get(alias)
            if value is None:
                raise KeyError("missing")
            return type("Version", (), {"version": value})()

        def set_registered_model_alias(self, _name, alias, version):
            self.aliases[alias] = version

        def delete_registered_model_alias(self, _name, alias):
            self.aliases.pop(alias, None)

    client = Client()
    config = _config(tmp_path)
    errors = lifecycle._compensate_aliases(
        config, client, [("stable", "1", "2")]
    )
    assert errors == ()
    assert client.aliases["stable"] == "1"

    client.aliases["stable"] = "99"
    errors = lifecycle._compensate_aliases(
        config, client, [("stable", "1", "2")]
    )
    assert errors
    assert client.aliases["stable"] == "99"


def _approved_promotion(
    config: LifecycleConfig,
    client: _LifecycleClient,
    tmp_path: Path,
) -> tuple[LifecycleConfig, LifecyclePlan]:
    plan = lifecycle.plan_lifecycle_transition(config, client=client)
    approval = dict(plan.approval_template)
    approval.update(
        {
            "approved_by": "operator@example.test",
            "approved_at_utc": "2026-07-28T11:00:00Z",
            "note": "Manual candidate promotion.",
        }
    )
    approval_path = tmp_path / "promotion-approval.json"
    _write_json(approval_path, approval)
    return (
        replace(
            config,
            dry_run=False,
            approval_path=approval_path,
            approval_sha256=sha256_file(approval_path),
        ),
        plan,
    )


def test_public_promotion_plan_accepts_verified_candidate_from_bootstrap_gen1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, client, evidence = _promotion_setup(tmp_path, monkeypatch)

    plan = lifecycle.plan_lifecycle_transition(config, client=client)

    assert plan.next_generation == 2
    assert plan.before_aliases == {
        "candidate": "2",
        "champion": "1",
        "stable": "1",
    }
    assert plan.after_aliases == {
        "candidate": None,
        "champion": "2",
        "stable": "1",
    }
    assert plan.evidence["promotion_effective_date"] == "2026-06-01"
    assert (
        plan.evidence["candidate_bundle_manifest_sha256"]
        == evidence["candidate_bundle_sha"]
    )


def test_public_promotion_execution_seals_state_moves_aliases_and_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, client, _evidence = _promotion_setup(tmp_path, monkeypatch)
    approved, _plan = _approved_promotion(config, client, tmp_path)

    result = lifecycle.execute_lifecycle_transition(approved, client=client)
    loaded = load_verified_deployment_pointer(
        approved.deployment_root, client=client
    )

    assert result.status == "probationary"
    assert client.aliases == {"champion": "2", "stable": "1"}
    assert loaded["pointer"]["generation"] == 2
    assert loaded["state"]["action"] == "promote"
    assert loaded["state"]["lifecycle_status"] == "probationary"
    assert loaded["receipt"]["automatic"] is False
    assert loaded["state"]["rollback_target"]["deployment_state_id"] == "1" * 64
    assert loaded["state"]["rollback_target"]["artifacts"]
    assert Path(result.receipt_path).is_file()
    assert Path(result.state_manifest_path).is_file()


def test_post_pointer_failure_preserves_aliases_and_writes_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, client, _evidence = _promotion_setup(tmp_path, monkeypatch)
    approved, _plan = _approved_promotion(config, client, tmp_path)
    monkeypatch.setattr(
        lifecycle,
        "load_verified_deployment_pointer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RetrainingDeploymentError("postcheck failed")
        ),
    )

    with pytest.raises(RetrainingLifecycleReconciliationError):
        lifecycle.execute_lifecycle_transition(approved, client=client)

    pointer = ActiveDeploymentPointer.from_dict(
        json.loads(
            (approved.deployment_root / "state" / "current.json").read_text(
                encoding="utf-8"
            )
        )
    )
    records = list((approved.deployment_root / "reconciliation").glob("*.json"))
    reconciliation = json.loads(records[-1].read_text(encoding="utf-8"))
    assert pointer.generation == 2
    assert client.aliases == {"champion": "2", "stable": "1"}
    assert reconciliation["pointer_published"] is True
    assert reconciliation["automatic_rollback_attempted"] is False


def test_external_pointer_divergence_after_aliases_skips_compensation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, client, _evidence = _promotion_setup(tmp_path, monkeypatch)
    approved, _plan = _approved_promotion(config, client, tmp_path)
    pointer_path = approved.deployment_root / "state" / "current.json"
    original_require_aliases = lifecycle._require_aliases

    def diverge_after_aliases(registry, name, expected):
        original_require_aliases(registry, name, expected)
        pointer_path.write_text("external-pointer", encoding="utf-8")

    monkeypatch.setattr(lifecycle, "_require_aliases", diverge_after_aliases)

    with pytest.raises(
        RetrainingLifecycleReconciliationError,
        match="pointer diverged after alias mutation",
    ):
        lifecycle.execute_lifecycle_transition(approved, client=client)

    reconciliation_path = next(
        (approved.deployment_root / "reconciliation").glob("*.json")
    )
    reconciliation = json.loads(
        reconciliation_path.read_text(encoding="utf-8")
    )
    assert pointer_path.read_text(encoding="utf-8") == "external-pointer"
    assert client.aliases == {"champion": "2", "stable": "1"}
    assert reconciliation["pointer_published"] is False
    assert reconciliation["compensation_errors"]
    assert reconciliation["automatic_rollback_attempted"] is False


def test_rollback_after_stabilization_uses_original_receipt_and_target_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, client, fixture = _promotion_setup(tmp_path, monkeypatch)
    approved, _plan = _approved_promotion(config, client, tmp_path)
    promoted = lifecycle.execute_lifecycle_transition(approved, client=client)
    promotion_state = lifecycle.load_deployment_state_v2(
        promoted.state_manifest_path
    )
    promotion_receipt = lifecycle.load_transition_receipt(promoted.receipt_path)
    stable_state = {
        **promotion_state,
        "generation": 3,
        "lifecycle_status": "stable",
        "action": "stabilize",
        "expected_aliases": {
            "candidate": None,
            "champion": "2",
            "stable": "2",
        },
    }
    stable_body = {
        key: value
        for key, value in stable_state.items()
        if key != "deployment_state_id"
    }
    stable_state_id = lifecycle._identifier("deployment_state", stable_body)
    stable_state["deployment_state_id"] = stable_state_id
    stable_path = (
        approved.deployment_root / "states" / stable_state_id / "state.json"
    )
    _write_json(stable_path, stable_state)
    client.aliases = {"champion": "2", "stable": "2"}
    rollback = LifecycleConfig(
        action="rollback",
        deployment_root=approved.deployment_root,
        registry_lock_root=approved.registry_lock_root,
        registered_model_name="wind-v2",
        tracking_uri="file:mlruns",
        expected=ExpectedDeploymentState(
            generation=3,
            deployment_state_id=stable_state_id,
            pointer_sha256="a" * 64,
            candidate=None,
            champion="2",
            stable="2",
        ),
        promotion_receipt=promoted.receipt_path,
        expected_rollback_state_id="1" * 64,
    )
    monkeypatch.setattr(
        lifecycle,
        "_verified_current",
        lambda *_args: {
            "state": stable_state,
            "state_manifest_path": str(stable_path),
        },
    )

    plan = lifecycle.plan_lifecycle_transition(rollback, client=client)
    artifacts = lifecycle._seal_transition_artifacts(rollback, stable_state)

    assert plan.after_aliases == {
        "candidate": None,
        "champion": "1",
        "stable": "1",
    }
    assert (
        plan.evidence["promotion_receipt_id"]
        == promotion_receipt["transition_receipt_id"]
    )
    assert artifacts == stable_state["rollback_target"]["artifacts"]
    assert artifacts == promotion_state["rollback_target"]["artifacts"]
    assert fixture["current"]["state"]["deployment_state_id"] == "1" * 64
    stable_pointer = ActiveDeploymentPointer(
        generation=3,
        deployment_id=stable_state["deployment_id"],
        deployment_state_id=stable_state_id,
        state_manifest_path=f"states/{stable_state_id}/state.json",
        state_manifest_sha256=sha256_file(stable_path),
        updated_at_utc="2026-07-28T12:30:00Z",
    )
    pointer_path = approved.deployment_root / "state" / "current.json"
    _write_json(pointer_path, stable_pointer.to_dict())
    rollback = replace(
        rollback,
        expected=replace(
            rollback.expected,
            pointer_sha256=sha256_file(pointer_path),
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "_load_target_state",
        lambda *_args: fixture["current"]["state"],
    )
    current_stable = {
        "state": stable_state,
        "state_manifest_path": str(stable_path),
    }
    monkeypatch.setattr(
        lifecycle, "_verified_current", lambda *_args: current_stable
    )
    rollback_plan = lifecycle.plan_lifecycle_transition(rollback, client=client)
    rollback_approval = dict(rollback_plan.approval_template)
    rollback_approval.update(
        {
            "approved_by": "operator@example.test",
            "approved_at_utc": "2026-07-28T12:45:00Z",
            "note": "Manual rollback after stabilization.",
        }
    )
    rollback_approval_path = tmp_path / "rollback-approval.json"
    _write_json(rollback_approval_path, rollback_approval)
    rollback = replace(
        rollback,
        approval_path=rollback_approval_path,
        approval_sha256=sha256_file(rollback_approval_path),
    )

    rolled_back = lifecycle.execute_lifecycle_transition(
        rollback, client=client
    )
    rollback_state = lifecycle.load_deployment_state_v2(
        rolled_back.state_manifest_path
    )
    assert rolled_back.status == "rolled_back"
    assert client.aliases == {"champion": "1", "stable": "1"}
    assert rollback_state["artifacts"] == promotion_state["rollback_target"][
        "artifacts"
    ]
    assert rollback_state["registry"]["model_version"] == "1"


def test_runtime_resolves_materialized_bundle_and_calibration_from_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, client, _evidence = _promotion_setup(tmp_path, monkeypatch)
    approved, _plan = _approved_promotion(config, client, tmp_path)
    result = lifecycle.execute_lifecycle_transition(approved, client=client)
    state = lifecycle.load_deployment_state_v2(result.state_manifest_path)

    import wind_forecast.monitoring as monitoring
    import wind_forecast.monitoring_reporting as reporting

    monkeypatch.setattr(
        monitoring,
        "validate_monitoring_model_bundle",
        lambda _path: {
            "bundle_sha256": state["pins"]["bundle_sha256"],
            "model_manifest": {
                "model_sha256": state["pins"]["model_sha256"],
                "feature_schema_sha256": state["pins"]["feature_schema_sha256"],
            },
            "dataset_manifest": {"sha256": state["pins"]["dataset_sha256"]},
        },
    )
    monkeypatch.setattr(
        reporting,
        "load_monitoring_calibration",
        lambda _path: {
            "calibration_id": state["calibration"]["calibration_id"],
            "reference_id": state["calibration"]["reference_id"],
        },
    )

    era = verify_active_model_era(
        approved.deployment_root,
        client=client,
    )

    assert era["deployment"]["generation"] == 2
    assert era["registry"]["model_version"] == "2"
    assert era["calibration"] == state["calibration"]


def test_loader_rejects_corrupt_materialized_rollback_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, client, _evidence = _promotion_setup(tmp_path, monkeypatch)
    approved, _plan = _approved_promotion(config, client, tmp_path)
    result = lifecycle.execute_lifecycle_transition(approved, client=client)
    state = lifecycle.load_deployment_state_v2(result.state_manifest_path)
    rollback_calibration = (
        approved.deployment_root
        / state["rollback_target"]["artifacts"]["calibration"]["path"]
        / "calibration.json"
    )
    rollback_calibration.write_text("corrupt", encoding="utf-8")

    with pytest.raises(
        RetrainingDeploymentError,
        match="runtime artifact checksum is invalid",
    ):
        load_verified_deployment_pointer(
            approved.deployment_root, client=client
        )


@pytest.mark.parametrize("corruption", ["traversal", "checksum"])
def test_v2_loader_rejects_predecessor_traversal_or_corrupt_checksum(
    tmp_path: Path, corruption: str
) -> None:
    root = tmp_path / "deployment"
    bundle_manifest = root / "artifacts" / "bundle" / "bundle_manifest.json"
    calibration_manifest = (
        root / "artifacts" / "calibration" / "calibration.json"
    )
    _write_json(bundle_manifest, {"bundle": True})
    _write_json(calibration_manifest, {"calibration": True})
    predecessor = root / "states" / ("1" * 64) / "state.json"
    _write_json(predecessor, {"deployment_state_id": "1" * 64})
    predecessor_path = (
        "../outside.json"
        if corruption == "traversal"
        else "states/" + "1" * 64 + "/state.json"
    )
    predecessor_sha = (
        sha256_file(predecessor) if corruption == "traversal" else "f" * 64
    )
    body = {
        "schema_version": lifecycle.STATE_SCHEMA,
        "generation": 2,
        "deployment_id": "2" * 64,
        "lifecycle_status": "probationary",
        "action": "promote",
        "registry": {
            "tracking_uri": "file:mlruns",
            "registered_model_name": "wind-v2",
            "model_version": "2",
            "run_id": "run-2",
            "model_uri": "models:/wind-v2/2",
        },
        "expected_aliases": {
            "candidate": None,
            "champion": "2",
            "stable": "1",
        },
        "pins": {},
        "artifacts": {
            "bundle": {
                "path": "artifacts/bundle",
                "manifest_sha256": sha256_file(bundle_manifest),
            },
            "calibration": {
                "path": "artifacts/calibration",
                "sha256": sha256_file(calibration_manifest),
            },
        },
        "calibration": {},
        "monitoring": {"new_model_era_required": True},
        "cutoffs": {},
        "predecessor": {
            "deployment_state_id": "1" * 64,
            "path": predecessor_path,
            "state_manifest_sha256": predecessor_sha,
        },
        "rollback_target": None,
        "authorizing_receipt": {
            "transition_receipt_id": "3" * 64,
            "path": "receipts/missing/receipt.json",
            "sha256": "4" * 64,
        },
    }
    state_id = lifecycle._identifier("deployment_state", body)
    state = {"deployment_state_id": state_id, **body}
    state_path = root / "states" / state_id / "state.json"
    _write_json(state_path, state)
    pointer = ActiveDeploymentPointer(
        generation=2,
        deployment_id=body["deployment_id"],
        deployment_state_id=state_id,
        state_manifest_path=f"states/{state_id}/state.json",
        state_manifest_sha256=sha256_file(state_path),
        updated_at_utc="2026-07-28T12:00:00Z",
    )
    _write_json(root / "state" / "current.json", pointer.to_dict())

    with pytest.raises(
        RetrainingDeploymentError,
        match="unsafe state path|outside deployment root|checksum is invalid",
    ):
        load_verified_deployment_pointer(root, client=_LifecycleClient())
