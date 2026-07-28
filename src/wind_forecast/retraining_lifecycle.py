"""Manual, optimistic and auditable lifecycle transitions for V2 deployments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Literal, Mapping
from uuid import uuid4

from wind_forecast.manifests import sha256_file
from wind_forecast.monitoring import (
    load_prediction_evidence,
    load_verified_monitoring_state,
    validate_monitoring_model_bundle,
)
from wind_forecast.monitoring_reporting import (
    load_monitoring_calibration,
    load_monitoring_report,
    load_monitoring_report_state,
)
from wind_forecast.retraining_backtesting import load_retraining_backtest
from wind_forecast.retraining_deployment import (
    POINTER_RELATIVE_PATH,
    load_exact_v2_bundle,
    load_verified_deployment_pointer,
)
from wind_forecast.retraining_policy import ActiveDeploymentPointer
from wind_forecast.retraining_registry import (
    acquire_registry_lock,
    load_retraining_registration_receipt,
    release_registry_lock,
)
from wind_forecast.tracking import DEFAULT_REGISTERED_MODEL_NAME, _load_mlflow


APPROVAL_SCHEMA = "wind_forecast.deployment_transition_approval.v1"
RECEIPT_SCHEMA = "wind_forecast.deployment_transition_receipt.v1"
STATE_SCHEMA = "wind_forecast.deployment_state.v2"
RECONCILIATION_SCHEMA = "wind_forecast.deployment_reconciliation.v1"
RUNTIME_BUNDLE_SCHEMA = "wind_forecast.runtime_bundle_manifest.v1"
ALIASES = ("candidate", "champion", "stable")


class RetrainingLifecycleError(RuntimeError):
    """Raised before publication or after successful pre-publication compensation."""


class RetrainingLifecycleReconciliationError(RuntimeError):
    """Raised after pointer publication; no automatic rollback is attempted."""


@dataclass(frozen=True)
class ExpectedDeploymentState:
    generation: int
    deployment_state_id: str
    pointer_sha256: str
    candidate: str | None
    champion: str | None
    stable: str | None

    def aliases(self) -> dict[str, str | None]:
        return {
            "candidate": self.candidate,
            "champion": self.champion,
            "stable": self.stable,
        }


@dataclass(frozen=True)
class LifecycleConfig:
    action: Literal["promote", "stabilize", "rollback"]
    deployment_root: Path
    registry_lock_root: Path
    registered_model_name: str
    tracking_uri: str
    expected: ExpectedDeploymentState
    approval_path: Path | None = None
    approval_sha256: str | None = None
    dry_run: bool = False
    candidate_bundle: Path | None = None
    candidate_calibration: Path | None = None
    incumbent_bundle: Path | None = None
    incumbent_calibration: Path | None = None
    registration_receipt: Path | None = None
    promotion_effective_date: date | str | None = None
    monitoring_store_root: Path | None = None
    monitoring_report: Path | None = None
    policy_path: Path | None = None
    monitoring_policy_path: Path | None = None
    observation_cutoff: date | str | None = None
    promotion_receipt: Path | None = None
    expected_rollback_state_id: str | None = None
    now_utc: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "deployment_root",
            "registry_lock_root",
            "candidate_bundle",
            "candidate_calibration",
            "incumbent_bundle",
            "incumbent_calibration",
            "registration_receipt",
            "monitoring_store_root",
            "monitoring_report",
            "policy_path",
            "monitoring_policy_path",
            "promotion_receipt",
            "approval_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        for name in ("promotion_effective_date", "observation_cutoff"):
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, date.fromisoformat(value))
        if self.action not in {"promote", "stabilize", "rollback"}:
            raise RetrainingLifecycleError("Unsupported lifecycle action.")
        if (
            not self.registered_model_name.strip()
            or self.registered_model_name == DEFAULT_REGISTERED_MODEL_NAME
        ):
            raise RetrainingLifecycleError(
                "An explicit non-V1 registered model name is required."
            )
        if not self.tracking_uri.strip():
            raise RetrainingLifecycleError("tracking_uri must be explicit.")
        if self.expected.generation < 1:
            raise RetrainingLifecycleError("Expected generation must be positive.")
        _sha(self.expected.pointer_sha256, "expected pointer")
        if self.now_utc is not None and self.now_utc.tzinfo is None:
            raise RetrainingLifecycleError("now_utc must be timezone-aware.")


@dataclass(frozen=True)
class LifecyclePlan:
    status: str
    action: str
    deployment_id: str
    next_generation: int
    before_aliases: Mapping[str, str | None]
    after_aliases: Mapping[str, str | None]
    approval_template: Mapping[str, Any]
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LifecycleResult:
    status: str
    plan: LifecyclePlan
    receipt_path: Path | None = None
    state_manifest_path: Path | None = None
    pointer_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["plan"] = self.plan.to_dict()
        for name in ("receipt_path", "state_manifest_path", "pointer_path"):
            item = getattr(self, name)
            value[name] = None if item is None else str(item)
        return value


def plan_lifecycle_transition(
    config: LifecycleConfig,
    *,
    client: Any | None = None,
    mlflow_module: Any | None = None,
) -> LifecyclePlan:
    """Verify every transition input without writing or acquiring a lock."""
    client = _client(config, client, mlflow_module)
    current = _verified_current(config, client)
    state = current["state"]
    before = config.expected.aliases()
    if config.action == "promote":
        after, evidence = _plan_promotion(config, state, client)
    elif config.action == "stabilize":
        after, evidence = _plan_stabilization(config, state)
    else:
        after, evidence = _plan_rollback(config, state)
    deployment_id = _identifier(
        "deployment_transition",
        {
            "action": config.action,
            "predecessor": state["deployment_state_id"],
            "generation": config.expected.generation + 1,
            "before_aliases": before,
            "after_aliases": after,
            "evidence": evidence,
        },
    )
    template = {
        "schema_version": APPROVAL_SCHEMA,
        "action": config.action,
        "approved_by": "<operator>",
        "approved_at_utc": "<YYYY-MM-DDTHH:MM:SSZ>",
        "note": f"<manual {config.action} reason>",
        "deployment_root": str(config.deployment_root.resolve()),
        "registered_model_name": config.registered_model_name,
        "expected_generation": config.expected.generation,
        "expected_deployment_state_id": config.expected.deployment_state_id,
        "expected_pointer_sha256": config.expected.pointer_sha256,
        "expected_aliases": before,
        "next_deployment_id": deployment_id,
        "evidence": evidence,
    }
    if config.approval_path is not None or config.approval_sha256 is not None:
        _load_approval(config, template)
    return LifecyclePlan(
        status="planned",
        action=config.action,
        deployment_id=deployment_id,
        next_generation=config.expected.generation + 1,
        before_aliases=before,
        after_aliases=after,
        approval_template=template,
        evidence=evidence,
    )


def execute_lifecycle_transition(
    config: LifecycleConfig,
    *,
    client: Any | None = None,
    mlflow_module: Any | None = None,
) -> LifecycleResult:
    """Execute one explicitly approved manual transition."""
    plan = plan_lifecycle_transition(
        config, client=client, mlflow_module=mlflow_module
    )
    if config.dry_run:
        return LifecycleResult(status="planned", plan=plan)
    approval, approval_sha = _load_approval(config, plan.approval_template)
    client = _client(config, client, mlflow_module)
    lock = acquire_registry_lock(
        config.registry_lock_root,
        config.registered_model_name,
        {
            "action": f"manual_{config.action}",
            "deployment_id": plan.deployment_id,
        },
    )
    pointer_published = False
    changed: list[tuple[str, str | None, str | None]] = []
    try:
        current = _verified_current(config, client)
        locked_plan = plan_lifecycle_transition(config, client=client)
        if locked_plan != plan:
            raise RetrainingLifecycleError(
                "Transition inputs changed after approval."
            )
        artifacts = _seal_transition_artifacts(config, current["state"])
        receipt, receipt_path = _seal_receipt(
            config, plan, approval, approval_sha, artifacts
        )
        state, state_path = _seal_state(
            config, plan, current, receipt, receipt_path, artifacts
        )
        for alias in ALIASES:
            old = plan.before_aliases[alias]
            new = plan.after_aliases[alias]
            if old == new:
                continue
            _require_alias(client, config.registered_model_name, alias, old)
            if new is None:
                client.delete_registered_model_alias(
                    config.registered_model_name, alias
                )
            else:
                client.set_registered_model_alias(
                    config.registered_model_name, alias, new
                )
            changed.append((alias, old, new))
            _require_alias(client, config.registered_model_name, alias, new)
        _require_aliases(client, config.registered_model_name, plan.after_aliases)
        _require_original_pointer(config)
        pointer_path = _publish_pointer(config, state, state_path)
        pointer_published = True
        load_verified_deployment_pointer(config.deployment_root, client=client)
        return LifecycleResult(
            status={
                "promote": "probationary",
                "stabilize": "stable",
                "rollback": "rolled_back",
            }[config.action],
            plan=plan,
            receipt_path=receipt_path,
            state_manifest_path=state_path,
            pointer_path=pointer_path,
        )
    except Exception as exc:
        if pointer_published:
            _write_reconciliation(config, plan, exc, True, ())
            raise RetrainingLifecycleReconciliationError(
                "The pointer was published; aliases and pointer were preserved "
                "for manual reconciliation. No automatic rollback was attempted."
            ) from exc
        try:
            _require_original_pointer(config)
        except Exception as pointer_exc:
            _write_reconciliation(
                config,
                plan,
                exc,
                False,
                (
                    "alias compensation skipped because the original pointer "
                    f"diverged: {type(pointer_exc).__name__}: "
                    f"{str(pointer_exc)[:300]}",
                ),
            )
            raise RetrainingLifecycleReconciliationError(
                "The deployment pointer diverged after alias mutation; aliases "
                "were preserved because they may belong to the external "
                "deployment. No automatic rollback was attempted."
            ) from exc
        errors = _compensate_aliases(config, client, changed)
        _write_reconciliation(config, plan, exc, False, errors)
        if errors:
            raise RetrainingLifecycleReconciliationError(
                "Pre-publication failure and alias compensation failure require "
                "manual reconciliation."
            ) from exc
        raise RetrainingLifecycleError(
            "Transition failed before pointer publication; changed aliases were "
            "safely restored and immutable evidence was retained."
        ) from exc
    finally:
        try:
            release_registry_lock(lock)
        except Exception as exc:
            _write_reconciliation(
                config, plan, exc, pointer_published, ()
            )
            if pointer_published:
                raise RetrainingLifecycleReconciliationError(
                    "Pointer publication succeeded but governance lock release "
                    "failed; no automatic rollback was attempted."
                ) from exc
            raise RetrainingLifecycleError(
                "Transition stopped before publication and governance lock "
                "release failed; inspect immutable reconciliation evidence."
            ) from exc


def load_transition_receipt(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = _read_json(target)
    required = {
        "schema_version",
        "transition_receipt_id",
        "action",
        "deployment_id",
        "generation",
        "registered_model_name",
        "before_aliases",
        "expected_aliases",
        "approval",
        "approval_sha256",
        "approval_payload_sha256",
        "evidence",
        "artifacts",
        "rollback_target_state_id",
        "executed_at_utc",
        "automatic",
    }
    if set(payload) != required or payload.get("schema_version") != RECEIPT_SCHEMA:
        raise RetrainingLifecycleError("Unsupported transition receipt schema.")
    identifier = payload.get("transition_receipt_id")
    body = {k: v for k, v in payload.items() if k != "transition_receipt_id"}
    if (
        identifier != _identifier("deployment_transition_receipt", body)
        or target.parent.name != identifier
    ):
        raise RetrainingLifecycleError("Transition receipt identity is corrupt.")
    if payload.get("action") not in {"promote", "stabilize", "rollback"}:
        raise RetrainingLifecycleError("Transition receipt action is invalid.")
    if payload.get("automatic") is not False:
        raise RetrainingLifecycleError("Automatic lifecycle evidence is forbidden.")
    approval = payload.get("approval")
    if (
        not isinstance(approval, Mapping)
        or approval.get("schema_version") != APPROVAL_SCHEMA
        or sha256(_canonical(approval)).hexdigest()
        != payload.get("approval_payload_sha256")
    ):
        raise RetrainingLifecycleError("Embedded approval evidence is corrupt.")
    _sha(str(payload.get("approval_sha256") or ""), "approval")
    return payload


def load_deployment_state_v2(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = _read_json(target)
    required = {
        "schema_version",
        "deployment_state_id",
        "generation",
        "deployment_id",
        "lifecycle_status",
        "action",
        "registry",
        "expected_aliases",
        "pins",
        "artifacts",
        "calibration",
        "monitoring",
        "cutoffs",
        "predecessor",
        "rollback_target",
        "authorizing_receipt",
    }
    if set(payload) != required or payload.get("schema_version") != STATE_SCHEMA:
        raise RetrainingLifecycleError("Deployment state fields differ from strict v2.")
    identifier = payload.get("deployment_state_id")
    body = {k: v for k, v in payload.items() if k != "deployment_state_id"}
    if (
        identifier != _identifier("deployment_state", body)
        or target.parent.name != identifier
    ):
        raise RetrainingLifecycleError("Deployment state identity is corrupt.")
    if (
        isinstance(payload["generation"], bool)
        or not isinstance(payload["generation"], int)
        or payload["generation"] < 2
    ):
        raise RetrainingLifecycleError("Deployment generation is invalid.")
    _sha(str(payload["deployment_id"]), "deployment id")
    if set(payload["expected_aliases"]) != set(ALIASES):
        raise RetrainingLifecycleError("Expected aliases are invalid.")
    if payload["expected_aliases"]["champion"] != payload["registry"]["model_version"]:
        raise RetrainingLifecycleError("Active Registry model and champion differ.")
    if payload["lifecycle_status"] not in {"probationary", "stable"}:
        raise RetrainingLifecycleError("Deployment lifecycle status is invalid.")
    return payload


def _plan_promotion(
    config: LifecycleConfig, state: Mapping[str, Any], client: Any
) -> tuple[dict[str, str | None], dict[str, Any]]:
    required = {
        "candidate_bundle": config.candidate_bundle,
        "candidate_calibration": config.candidate_calibration,
        "incumbent_bundle": config.incumbent_bundle,
        "incumbent_calibration": config.incumbent_calibration,
        "registration_receipt": config.registration_receipt,
        "promotion_effective_date": config.promotion_effective_date,
    }
    if any(value is None for value in required.values()):
        raise RetrainingLifecycleError(
            "Promotion requires candidate bundle, calibration, registration "
            "receipt and effective date."
        )
    if _status(state) == "probationary":
        raise RetrainingLifecycleError("Another model is already probationary.")
    if config.expected.candidate is None:
        raise RetrainingLifecycleError("Promotion requires an expected candidate.")
    receipt = load_retraining_registration_receipt(config.registration_receipt)
    bundle = load_retraining_backtest(config.candidate_bundle)
    calibration = load_monitoring_calibration(config.candidate_calibration)
    incumbent_bundle = _load_runtime_bundle(config.incumbent_bundle)
    incumbent_calibration = load_monitoring_calibration(
        config.incumbent_calibration
    )
    backtest = bundle["backtest"]
    if backtest["outcome"] != "accepted":
        raise RetrainingLifecycleError("Only an accepted candidate may be promoted.")
    if (
        receipt["registered_model_name"] != config.registered_model_name
        or receipt["model_version"] != config.expected.candidate
        or receipt["backtest_id"] != backtest["backtest_id"]
        or receipt["champion_after"] != config.expected.champion
        or receipt["stable_after"] != config.expected.stable
        or receipt["candidate_model_sha256"]
        != backtest["final_training"]["candidate_model_sha256"]
    ):
        raise RetrainingLifecycleError(
            "Candidate registration receipt, alias and accepted bundle differ."
        )
    incumbent_pins = state["pins"]
    if (
        incumbent_bundle["bundle_sha256"]
        != incumbent_pins["bundle_sha256"]
        or sha256_file(config.incumbent_calibration / "calibration.json")
        != incumbent_pins["calibration_sha256"]
        or incumbent_calibration["calibration_id"]
        != state["calibration"]["calibration_id"]
        or incumbent_calibration["reference_id"]
        != state["calibration"]["reference_id"]
    ):
        raise RetrainingLifecycleError(
            "Explicit incumbent bundle/calibration differ from active state."
        )
    candidate_version = client.get_model_version(
        config.registered_model_name, config.expected.candidate
    )
    observed_tags = dict(getattr(candidate_version, "tags", {}) or {})
    if (
        str(getattr(candidate_version, "run_id", "")) != receipt["run_id"]
        or any(
            str(observed_tags.get(key)) != str(value)
            for key, value in receipt["tags"].items()
        )
    ):
        raise RetrainingLifecycleError(
            "Candidate Registry version/tags and registration receipt differ."
        )
    reference = calibration["_reference_manifest"]
    if (
        reference.get("model_sha256") != receipt["candidate_model_sha256"]
        or (reference.get("calibration_subject") or {}).get("backtest_id")
        != backtest["backtest_id"]
    ):
        raise RetrainingLifecycleError(
            "Candidate calibration is not candidate-specific."
        )
    _require_promotion_after_monitoring(
        config.promotion_effective_date,
        backtest,
    )
    after = {
        "candidate": None,
        "champion": config.expected.candidate,
        "stable": config.expected.stable,
    }
    evidence = {
        "registration_receipt_path": str(config.registration_receipt.resolve()),
        "registration_receipt_sha256": sha256_file(config.registration_receipt),
        "candidate_bundle_path": str(config.candidate_bundle.resolve()),
        "candidate_bundle_manifest_sha256": sha256_file(
            config.candidate_bundle / "bundle_manifest.json"
        ),
        "candidate_calibration_path": str(config.candidate_calibration.resolve()),
        "candidate_calibration_sha256": sha256_file(
            config.candidate_calibration / "calibration.json"
        ),
        "incumbent_bundle_path": str(config.incumbent_bundle.resolve()),
        "incumbent_bundle_sha256": incumbent_bundle["bundle_sha256"],
        "incumbent_calibration_path": str(
            config.incumbent_calibration.resolve()
        ),
        "incumbent_calibration_sha256": sha256_file(
            config.incumbent_calibration / "calibration.json"
        ),
        "backtest_id": backtest["backtest_id"],
        "candidate_fit_cutoff": backtest["cutoffs"]["candidate_fit_cutoff"],
        "promotion_effective_date": config.promotion_effective_date.isoformat(),
        "candidate_run_id": receipt["run_id"],
        "candidate_model_uri": receipt["model_uri"],
        "candidate_model_sha256": receipt["candidate_model_sha256"],
        "calibration_id": calibration["calibration_id"],
        "reference_id": calibration["reference_id"],
    }
    return after, evidence


def _plan_stabilization(
    config: LifecycleConfig, state: Mapping[str, Any]
) -> tuple[dict[str, str | None], dict[str, Any]]:
    if _status(state) != "probationary":
        raise RetrainingLifecycleError(
            "Only the current probationary champion may be stabilized."
        )
    if any(
        value is None
        for value in (
            config.monitoring_store_root,
            config.monitoring_report,
            config.policy_path,
            config.monitoring_policy_path,
            config.observation_cutoff,
        )
    ):
        raise RetrainingLifecycleError(
            "Stabilization requires monitoring store, report, policy and cutoff."
        )
    report = load_monitoring_report(config.monitoring_report)
    policy = _read_json(config.policy_path)
    monitoring_policy = _read_json(config.monitoring_policy_path)
    report_state = load_monitoring_report_state(config.monitoring_store_root)
    stability = policy.get("stability") or {}
    automation = policy.get("automation") or {}
    if (
        automation.get("automatic_stability") is not False
        or stability.get("require_second_manual_approval") is not True
        or stability.get("require_no_active_warning_or_critical") is not True
        or int(stability.get("minimum_eligible_observations", 0)) != 90
        or set(stability.get("allowed_issuance_kinds") or [])
        != {"scheduled", "catch_up"}
    ):
        raise RetrainingLifecycleError(
            "Policy does not enforce the fixed manual stability contract."
        )
    if report.get("active_alerts") or any(
        item.get("severity") in {"warning", "critical"}
        for item in report.get("breaches", [])
    ):
        raise RetrainingLifecycleError(
            "Stabilization is blocked by warning or critical monitoring evidence."
        )
    if (
        report_state is None
        or report_state.get("latest_report_id") != report.get("report_id")
        or report_state.get("latest_through_date") != report.get("through_date")
        or (report.get("reference") or {}).get("policy_sha256")
        != sha256_file(config.monitoring_policy_path)
        or report.get("config") != monitoring_policy
    ):
        raise RetrainingLifecycleError(
            "Monitoring report is not the exact current policy-pinned report."
        )
    era = report.get("model_era") or {}
    if (
        era.get("deployment_id") != state["deployment_id"]
        or str(era.get("model_version")) != str(config.expected.champion)
        or report.get("through_date") != config.observation_cutoff.isoformat()
    ):
        raise RetrainingLifecycleError(
            "Monitoring report is not current for the probationary deployment."
        )
    ledger = load_verified_monitoring_state(config.monitoring_store_root)
    if ledger is None:
        raise RetrainingLifecycleError("Verified monitoring ledger is absent.")
    era_id = str(era.get("model_era_id") or "")
    if ledger.get("active_model_era_id") != era_id:
        raise RetrainingLifecycleError(
            "Monitoring ledger is not on the probationary model era."
        )
    start = date.fromisoformat(state["cutoffs"]["promotion_effective_date"])
    eligible: list[str] = []
    for day, prediction_id in sorted((ledger.get("as_issued") or {}).items()):
        target = date.fromisoformat(day)
        if target < start or target > config.observation_cutoff:
            continue
        evidence = load_prediction_evidence(
            config.monitoring_store_root, str(prediction_id)
        )
        prediction = evidence["prediction"]
        actual_id = (ledger.get("actuals") or {}).get(day)
        actual = next(
            (
                item
                for item in evidence.get("actual_revisions", [])
                if item.get("actual_revision_id") == actual_id
            ),
            None,
        )
        if (
            prediction.get("model_era_id") != era_id
            or prediction.get("issuance_kind") not in {"scheduled", "catch_up"}
            or prediction.get("target_date") != day
            or actual is None
            or actual.get("target_date") != day
            or not math.isfinite(float(prediction.get("prediction")))
            or not math.isfinite(float(actual.get("actual")))
        ):
            raise RetrainingLifecycleError(
                "Probation observation set contains ineligible evidence."
            )
        eligible.append(day)
    if len(eligible) != 90:
        raise RetrainingLifecycleError(
            f"Stabilization requires exactly 90 eligible observations; found {len(eligible)}."
        )
    if (report.get("quality") or {}).get("issues"):
        raise RetrainingLifecycleError(
            "Stabilization report contains quality exclusions."
        )
    after = {
        "candidate": None,
        "champion": config.expected.champion,
        "stable": config.expected.champion,
    }
    return after, {
        "monitoring_report_path": str(config.monitoring_report.resolve()),
        "monitoring_report_sha256": sha256_file(config.monitoring_report),
        "monitoring_report_id": report["report_id"],
        "policy_path": str(config.policy_path.resolve()),
        "policy_sha256": sha256_file(config.policy_path),
        "monitoring_policy_path": str(
            config.monitoring_policy_path.resolve()
        ),
        "monitoring_policy_sha256": sha256_file(
            config.monitoring_policy_path
        ),
        "observation_cutoff": config.observation_cutoff.isoformat(),
        "eligible_observation_dates": eligible,
        "eligible_observation_count": 90,
        "model_era_id": era_id,
    }


def _plan_rollback(
    config: LifecycleConfig, state: Mapping[str, Any]
) -> tuple[dict[str, str | None], dict[str, Any]]:
    if config.promotion_receipt is None or not config.expected_rollback_state_id:
        raise RetrainingLifecycleError(
            "Rollback requires the original promotion receipt and expected target."
        )
    receipt = load_transition_receipt(config.promotion_receipt)
    if receipt.get("action") != "promote":
        raise RetrainingLifecycleError("Rollback receipt is not a promotion receipt.")
    target = state.get("rollback_target")
    if (
        not isinstance(target, Mapping)
        or target.get("deployment_state_id")
        != config.expected_rollback_state_id
        or target.get("deployment_state_id")
        != receipt.get("rollback_target_state_id")
        or target.get("promotion_receipt_id")
        != receipt.get("transition_receipt_id")
    ):
        raise RetrainingLifecycleError(
            "Rollback target is not the last stable fixed by promotion."
        )
    aliases = target.get("expected_aliases") or {}
    after = {
        "candidate": None,
        "champion": aliases.get("champion"),
        "stable": aliases.get("stable"),
    }
    if after["champion"] is None or after["champion"] != after["stable"]:
        raise RetrainingLifecycleError("Rollback target is not a stable deployment.")
    return after, {
        "promotion_receipt_path": str(config.promotion_receipt.resolve()),
        "promotion_receipt_sha256": sha256_file(config.promotion_receipt),
        "promotion_receipt_id": receipt["transition_receipt_id"],
        "rollback_target_state_id": target["deployment_state_id"],
        "rollback_target_state_sha256": target["state_manifest_sha256"],
    }


def _seal_transition_artifacts(
    config: LifecycleConfig, current: Mapping[str, Any]
) -> dict[str, Any]:
    if config.action == "rollback":
        target = (current.get("rollback_target") or {}).get("artifacts")
        if not isinstance(target, Mapping) or not target:
            raise RetrainingLifecycleError(
                "Rollback target has no immutable runtime artifacts."
            )
        return dict(target)
    if config.action != "promote":
        return dict(current.get("artifacts") or {})
    bundle_manifest_sha = sha256_file(
        config.candidate_bundle / "bundle_manifest.json"
    )
    bundle_target = (
        config.deployment_root / "artifacts" / "bundles" / bundle_manifest_sha
    )
    _immutable_tree(config.candidate_bundle, bundle_target)
    calibration = load_monitoring_calibration(config.candidate_calibration)
    calibration_sha = sha256_file(config.candidate_calibration / "calibration.json")
    set_root = (
        config.deployment_root / "artifacts" / "calibration_sets" / calibration_sha
    )
    calibration_target = (
        set_root / "calibrations" / str(calibration["calibration_id"])
    )
    reference_source = Path(calibration["_reference_path"]).parent
    reference_target = set_root / "references" / str(calibration["reference_id"])
    _immutable_tree(config.candidate_calibration, calibration_target)
    _immutable_tree(reference_source, reference_target)
    incumbent_bundle = _load_runtime_bundle(config.incumbent_bundle)
    incumbent_bundle_target = (
        config.deployment_root
        / "artifacts"
        / "bundles"
        / incumbent_bundle["bundle_sha256"]
    )
    _immutable_tree(config.incumbent_bundle, incumbent_bundle_target)
    incumbent_calibration = load_monitoring_calibration(
        config.incumbent_calibration
    )
    incumbent_calibration_sha = sha256_file(
        config.incumbent_calibration / "calibration.json"
    )
    incumbent_set_root = (
        config.deployment_root
        / "artifacts"
        / "calibration_sets"
        / incumbent_calibration_sha
    )
    incumbent_calibration_target = (
        incumbent_set_root
        / "calibrations"
        / str(incumbent_calibration["calibration_id"])
    )
    incumbent_reference_source = Path(
        incumbent_calibration["_reference_path"]
    ).parent
    _immutable_tree(
        config.incumbent_calibration, incumbent_calibration_target
    )
    _immutable_tree(
        incumbent_reference_source,
        incumbent_set_root
        / "references"
        / str(incumbent_calibration["reference_id"]),
    )
    return {
        "bundle": {
            "path": _relative(config.deployment_root, bundle_target),
            "manifest_sha256": bundle_manifest_sha,
        },
        "calibration": {
            "path": _relative(config.deployment_root, calibration_target),
            "sha256": calibration_sha,
        },
        "rollback": {
            "bundle": {
                "path": _relative(
                    config.deployment_root, incumbent_bundle_target
                ),
                "manifest_sha256": incumbent_bundle["bundle_sha256"],
            },
            "calibration": {
                "path": _relative(
                    config.deployment_root, incumbent_calibration_target
                ),
                "sha256": incumbent_calibration_sha,
            },
        },
    }


def _seal_receipt(
    config: LifecycleConfig,
    plan: LifecyclePlan,
    approval: Mapping[str, Any],
    approval_sha: str,
    artifacts: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    rollback_id = (
        config.expected.deployment_state_id if config.action == "promote" else None
    )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "action": config.action,
        "deployment_id": plan.deployment_id,
        "generation": plan.next_generation,
        "registered_model_name": config.registered_model_name,
        "before_aliases": dict(plan.before_aliases),
        "expected_aliases": dict(plan.after_aliases),
        "approval": dict(approval),
        "approval_sha256": approval_sha,
        "approval_payload_sha256": sha256(_canonical(approval)).hexdigest(),
        "evidence": dict(plan.evidence),
        "artifacts": dict(artifacts),
        "rollback_target_state_id": rollback_id,
        "executed_at_utc": _now(config),
        "automatic": False,
    }
    identifier = _identifier("deployment_transition_receipt", body)
    payload = {"transition_receipt_id": identifier, **body}
    target = config.deployment_root / "receipts" / identifier / "receipt.json"
    _immutable_json(target, payload)
    load_transition_receipt(target)
    return payload, target


def _seal_state(
    config: LifecycleConfig,
    plan: LifecyclePlan,
    current: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_path: Path,
    artifacts: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    old = current["state"]
    active_version = str(plan.after_aliases["champion"])
    if config.action == "promote":
        registry = {
            "tracking_uri": config.tracking_uri,
            "registered_model_name": config.registered_model_name,
            "model_version": active_version,
            "run_id": plan.evidence["candidate_run_id"],
            "model_uri": plan.evidence["candidate_model_uri"],
        }
        calibration = {
            "calibration_id": plan.evidence["calibration_id"],
            "reference_id": plan.evidence["reference_id"],
        }
        pins = {
            "bundle_sha256": plan.evidence["candidate_bundle_manifest_sha256"],
            "calibration_sha256": plan.evidence["candidate_calibration_sha256"],
            "model_sha256": plan.evidence["candidate_model_sha256"],
            "dataset_sha256": load_retraining_backtest(
                config.candidate_bundle
            )["backtest"]["final_training"]["dataset_sha256"],
            "feature_schema_sha256": load_retraining_backtest(
                config.candidate_bundle
            )["backtest"]["identities"]["feature_schema_sha256"],
        }
        cutoffs = {
            "fit_cutoff": plan.evidence["candidate_fit_cutoff"],
            "promotion_effective_date": plan.evidence[
                "promotion_effective_date"
            ],
            "observation_cutoff": None,
        }
        lifecycle = "probationary"
        rollback = _stable_target(
            config,
            current,
            artifacts,
            promotion_receipt_id=receipt["transition_receipt_id"],
        )
    elif config.action == "rollback":
        target = _load_target_state(config, old["rollback_target"])
        registry = dict(target["registry"])
        calibration = dict(target["calibration"])
        pins = dict(target["pins"])
        artifacts = dict(
            (old.get("rollback_target") or {}).get("artifacts")
            or target.get("artifacts")
            or {}
        )
        cutoffs = {
            **dict(target["cutoffs"]),
            "observation_cutoff": config.observation_cutoff.isoformat()
            if config.observation_cutoff
            else None,
        }
        lifecycle = "stable"
        rollback = None
    else:
        registry = dict(old["registry"])
        calibration = dict(old["calibration"])
        pins = dict(old["pins"])
        cutoffs = {
            **dict(old["cutoffs"]),
            "observation_cutoff": config.observation_cutoff.isoformat(),
        }
        lifecycle = "stable"
        rollback = dict(old["rollback_target"])
    predecessor = {
        "deployment_state_id": old["deployment_state_id"],
        "path": _relative(
            config.deployment_root, Path(current["state_manifest_path"])
        ),
        "state_manifest_sha256": sha256_file(current["state_manifest_path"]),
    }
    body = {
        "schema_version": STATE_SCHEMA,
        "generation": plan.next_generation,
        "deployment_id": plan.deployment_id,
        "lifecycle_status": lifecycle,
        "action": config.action,
        "registry": registry,
        "expected_aliases": dict(plan.after_aliases),
        "pins": pins,
        "artifacts": dict(artifacts),
        "calibration": calibration,
        "monitoring": {"new_model_era_required": True},
        "cutoffs": cutoffs,
        "predecessor": predecessor,
        "rollback_target": rollback,
        "authorizing_receipt": {
            "transition_receipt_id": receipt["transition_receipt_id"],
            "path": _relative(config.deployment_root, receipt_path),
            "sha256": sha256_file(receipt_path),
        },
    }
    identifier = _identifier("deployment_state", body)
    payload = {"deployment_state_id": identifier, **body}
    target = config.deployment_root / "states" / identifier / "state.json"
    _immutable_json(target, payload)
    load_deployment_state_v2(target)
    return payload, target


def _stable_target(
    config: LifecycleConfig,
    current: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    *,
    promotion_receipt_id: str,
) -> dict[str, Any]:
    state = current["state"]
    if config.expected.stable != config.expected.champion:
        raise RetrainingLifecycleError(
            "Promotion must start from a stable incumbent."
        )
    return {
        "deployment_state_id": state["deployment_state_id"],
        "path": _relative(
            config.deployment_root, Path(current["state_manifest_path"])
        ),
        "state_manifest_sha256": sha256_file(current["state_manifest_path"]),
        "expected_aliases": dict(state["expected_aliases"]),
        "artifacts": dict((artifacts or {}).get("rollback") or {}),
        "promotion_receipt_id": promotion_receipt_id,
    }


def _load_target_state(
    config: LifecycleConfig, target: Mapping[str, Any]
) -> dict[str, Any]:
    path = config.deployment_root / str(target["path"])
    if sha256_file(path) != target["state_manifest_sha256"]:
        raise RetrainingLifecycleError("Rollback target state is corrupt.")
    if _read_json(path).get("schema_version") == STATE_SCHEMA:
        return load_deployment_state_v2(path)
    from wind_forecast.retraining_deployment import _load_deployment_state

    return _load_deployment_state(path)


def _publish_pointer(
    config: LifecycleConfig, state: Mapping[str, Any], state_path: Path
) -> Path:
    _require_original_pointer(config)
    pointer_path = config.deployment_root / POINTER_RELATIVE_PATH
    pointer = ActiveDeploymentPointer(
        generation=int(state["generation"]),
        deployment_id=str(state["deployment_id"]),
        deployment_state_id=str(state["deployment_state_id"]),
        state_manifest_path=_relative(config.deployment_root, state_path),
        state_manifest_sha256=sha256_file(state_path),
        updated_at_utc=_now(config),
    )
    temporary = pointer_path.parent / f".current.{uuid4().hex}.tmp"
    _write_bytes(temporary, _json_bytes(pointer.to_dict()), exclusive=True)
    try:
        _require_original_pointer(config)
        os.replace(temporary, pointer_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return pointer_path


def _verified_current(
    config: LifecycleConfig, client: Any
) -> dict[str, Any]:
    verified = load_verified_deployment_pointer(
        config.deployment_root, client=client
    )
    pointer_path = Path(verified["pointer_path"])
    pointer = verified["pointer"]
    if (
        pointer["generation"] != config.expected.generation
        or pointer["deployment_state_id"]
        != config.expected.deployment_state_id
        or sha256_file(pointer_path) != config.expected.pointer_sha256
    ):
        raise RetrainingLifecycleError(
            "Active deployment pointer differs from the expected state."
        )
    if verified["state"]["registry"]["registered_model_name"] != (
        config.registered_model_name
    ):
        raise RetrainingLifecycleError("Active deployment model name differs.")
    _require_aliases(client, config.registered_model_name, config.expected.aliases())
    return verified


def _require_original_pointer(config: LifecycleConfig) -> None:
    path = config.deployment_root / POINTER_RELATIVE_PATH
    if not path.is_file() or sha256_file(path) != config.expected.pointer_sha256:
        raise RetrainingLifecycleError(
            "Deployment pointer changed before atomic publication."
        )
    pointer = ActiveDeploymentPointer.from_dict(_read_json(path))
    if (
        pointer.generation != config.expected.generation
        or pointer.deployment_state_id != config.expected.deployment_state_id
    ):
        raise RetrainingLifecycleError(
            "Deployment pointer identity changed before publication."
        )


def _load_approval(
    config: LifecycleConfig, template: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    if config.approval_path is None or config.approval_sha256 is None:
        raise RetrainingLifecycleError(
            "Execution requires --approval-path and --approval-sha256."
        )
    if (
        not config.approval_path.is_file()
        or config.approval_path.is_symlink()
        or sha256_file(config.approval_path) != config.approval_sha256
    ):
        raise RetrainingLifecycleError("Approval file or checksum differs.")
    approval = _read_json(config.approval_path)
    expected_keys = set(template)
    if set(approval) != expected_keys or approval.get("schema_version") != APPROVAL_SCHEMA:
        raise RetrainingLifecycleError("Approval fields differ from strict v1.")
    for key, value in template.items():
        if key in {"approved_by", "approved_at_utc", "note"}:
            if not str(approval.get(key) or "").strip() or str(
                approval[key]
            ).startswith("<"):
                raise RetrainingLifecycleError(
                    f"Approval {key} must be completed by an operator."
                )
        elif approval.get(key) != value:
            raise RetrainingLifecycleError(
                f"Approval field {key} differs from the verified plan."
            )
    datetime.fromisoformat(
        str(approval["approved_at_utc"]).replace("Z", "+00:00")
    )
    return approval, config.approval_sha256


def _client(
    config: LifecycleConfig, client: Any | None, mlflow_module: Any | None
) -> Any:
    if client is not None:
        return client
    mlflow = mlflow_module or _load_mlflow()
    if hasattr(mlflow, "set_tracking_uri"):
        mlflow.set_tracking_uri(config.tracking_uri)
    return mlflow.MlflowClient()


def _status(state: Mapping[str, Any]) -> str:
    if state.get("schema_version") == STATE_SCHEMA:
        return str(state["lifecycle_status"])
    return "stable"


def _require_promotion_after_monitoring(
    promotion_effective_date: date,
    backtest: Mapping[str, Any],
) -> None:
    monitoring_cutoff = date.fromisoformat(
        str(backtest["cutoffs"]["monitoring_evaluation_cutoff"])
    )
    if promotion_effective_date <= monitoring_cutoff:
        raise RetrainingLifecycleError(
            "Promotion effective date must follow monitoring evaluation cutoff."
        )


def _load_runtime_bundle(path: Path) -> dict[str, Any]:
    if (path / "bundle_manifest.json").is_file():
        return validate_monitoring_model_bundle(path)
    return load_exact_v2_bundle(path)


def _alias(client: Any, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except Exception as exc:
        text = str(exc).lower()
        if isinstance(exc, KeyError) or any(
            token in text
            for token in ("not found", "does not exist", "resource", "missing")
        ):
            return None
        raise


def _require_alias(
    client: Any, name: str, alias: str, expected: str | None
) -> None:
    actual = _alias(client, name, alias)
    if actual != expected:
        raise RetrainingLifecycleError(
            f"Registry alias {alias} differs: expected {expected or 'none'}, "
            f"found {actual or 'none'}."
        )


def _require_aliases(
    client: Any, name: str, expected: Mapping[str, str | None]
) -> None:
    for alias in ALIASES:
        _require_alias(client, name, alias, expected[alias])


def _compensate_aliases(
    config: LifecycleConfig,
    client: Any,
    changed: list[tuple[str, str | None, str | None]],
) -> tuple[str, ...]:
    errors = []
    for alias, old, new in reversed(changed):
        try:
            if _alias(client, config.registered_model_name, alias) != new:
                raise RetrainingLifecycleError(
                    f"Alias {alias} changed outside lifecycle lock."
                )
            if old is None:
                client.delete_registered_model_alias(
                    config.registered_model_name, alias
                )
            else:
                client.set_registered_model_alias(
                    config.registered_model_name, alias, old
                )
            _require_alias(client, config.registered_model_name, alias, old)
        except Exception as exc:
            errors.append(f"{alias}: {type(exc).__name__}: {str(exc)[:300]}")
    return tuple(errors)


def _write_reconciliation(
    config: LifecycleConfig,
    plan: LifecyclePlan,
    exc: Exception,
    pointer_published: bool,
    compensation_errors: tuple[str, ...],
) -> Path:
    body = {
        "schema_version": RECONCILIATION_SCHEMA,
        "action": config.action,
        "deployment_id": plan.deployment_id,
        "expected_generation": config.expected.generation,
        "pointer_published": pointer_published,
        "automatic_rollback_attempted": False,
        "compensation_errors": list(compensation_errors),
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
        "recorded_at_utc": _now(config),
    }
    identifier = _identifier("deployment_reconciliation", body)
    payload = {"reconciliation_id": identifier, **body}
    path = config.deployment_root / "reconciliation" / f"{identifier}.json"
    _immutable_json(path, payload)
    return path


def _immutable_tree(source: Path, target: Path) -> None:
    source = Path(source)
    if not source.is_dir() or source.is_symlink():
        raise RetrainingLifecycleError("Immutable artifact source must be a directory.")
    files = [item for item in source.rglob("*") if item.is_file()]
    if any(item.is_symlink() for item in source.rglob("*")):
        raise RetrainingLifecycleError("Immutable artifact trees cannot contain links.")
    if target.exists():
        for item in files:
            destination = target / item.relative_to(source)
            if not destination.is_file() or sha256_file(destination) != sha256_file(item):
                raise RetrainingLifecycleError("Immutable artifact target differs.")
        return
    prepared = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    prepared.mkdir(parents=True)
    try:
        for item in files:
            destination = prepared / item.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, destination)
        prepared.rename(target)
    finally:
        if prepared.exists():
            shutil.rmtree(prepared)


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RetrainingLifecycleError(
                f"Immutable path contains different bytes: {path}."
            )
        return
    _write_bytes(path, data, exclusive=True)


def _write_bytes(path: Path, data: bytes, *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrainingLifecycleError(f"Invalid JSON artifact: {path}.") from exc
    if not isinstance(value, dict):
        raise RetrainingLifecycleError("JSON evidence must be an object.")
    return value


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RetrainingLifecycleError(
            "Lifecycle evidence must remain under deployment root."
        ) from exc


def _sha(value: str, label: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RetrainingLifecycleError(f"{label} must be a lowercase SHA-256.")
    return value


def _identifier(kind: str, body: Mapping[str, Any]) -> str:
    return sha256(kind.encode("utf-8") + b":" + _canonical(body)).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, default=str)
        + "\n"
    ).encode("utf-8")


def _now(config: LifecycleConfig) -> str:
    value = config.now_utc or datetime.now(timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "APPROVAL_SCHEMA",
    "ExpectedDeploymentState",
    "LifecycleConfig",
    "LifecyclePlan",
    "LifecycleResult",
    "RECEIPT_SCHEMA",
    "RetrainingLifecycleError",
    "RetrainingLifecycleReconciliationError",
    "STATE_SCHEMA",
    "execute_lifecycle_transition",
    "load_deployment_state_v2",
    "load_transition_receipt",
    "plan_lifecycle_transition",
]
