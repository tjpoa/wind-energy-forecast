"""Monthly recommendation-only coordinator for controlled retraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
from zoneinfo import ZoneInfo

from wind_forecast.manifests import sha256_file
from wind_forecast.monitoring import (
    MonitoringError,
    load_model_era,
    load_prediction_evidence,
    load_verified_monitoring_state,
)
from wind_forecast.monitoring_reporting import (
    MonitoringReportingError,
    load_monitoring_report,
    load_monitoring_report_state,
)
from wind_forecast.retraining_deployment import (
    RetrainingDeploymentError,
    load_verified_deployment_pointer,
)
from wind_forecast.retraining_evaluation import (
    MonthlyRetrainingEvaluationConfig,
    RetrainingEvaluationError,
    run_monthly_retraining_evaluation,
)
from wind_forecast.retraining_policy import RetrainingContractError, RetrainingPolicy


RECOMMENDATION_SCHEMA = "wind_forecast.monthly_governance_recommendation.v1"
RETRAINING_DECISIONS = (
    "evaluated",
    "not_applicable_probationary",
    "not_applicable_model_era_changed",
)
STABILITY_DECISIONS = (
    "not_applicable_stable",
    "insufficient_observations",
    "blocked_health",
    "ready_for_second_manual_approval",
)


class MonthlyGovernanceError(RuntimeError):
    """Raised when a monthly recommendation cannot be trusted or sealed."""


@dataclass(frozen=True)
class MonthlyGovernanceConfig:
    policy_path: Path
    monitoring_policy_path: Path
    monitoring_store_root: Path
    deployment_root: Path
    logical_at_utc: datetime | str
    output_root: Path = Path(
        "data/processed/v2/retraining/monthly_recommendations"
    )
    evaluation_output_root: Path = Path(
        "data/processed/v2/retraining/evaluations"
    )
    dry_run: bool = False
    now_utc: datetime | str | None = None

    def __post_init__(self) -> None:
        for name in (
            "policy_path",
            "monitoring_policy_path",
            "monitoring_store_root",
            "deployment_root",
            "output_root",
            "evaluation_output_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        logical = self.logical_at_utc
        if isinstance(logical, str):
            try:
                logical = datetime.fromisoformat(logical.replace("Z", "+00:00"))
            except ValueError as exc:
                raise MonthlyGovernanceError(
                    "logical_at_utc must be an ISO-8601 timestamp."
                ) from exc
        if (
            not isinstance(logical, datetime)
            or logical.tzinfo is None
            or logical.utcoffset() != timedelta(0)
        ):
            raise MonthlyGovernanceError(
                "logical_at_utc must be an explicit UTC timestamp."
            )
        object.__setattr__(
            self,
            "logical_at_utc",
            logical.astimezone(timezone.utc),
        )
        current = self.now_utc or datetime.now(timezone.utc)
        if isinstance(current, str):
            try:
                current = datetime.fromisoformat(
                    current.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise MonthlyGovernanceError(
                    "now_utc must be an ISO-8601 timestamp."
                ) from exc
        if (
            not isinstance(current, datetime)
            or current.tzinfo is None
            or current.utcoffset() != timedelta(0)
        ):
            raise MonthlyGovernanceError(
                "now_utc must be an explicit UTC timestamp."
            )
        object.__setattr__(
            self,
            "now_utc",
            current.astimezone(timezone.utc),
        )


@dataclass(frozen=True)
class MonthlyGovernanceResult:
    status: str
    recommendation_id: str
    recommendation_path: Path | None
    evaluation_period: str
    retraining_decision: str
    stability_decision: str

    def summary(self) -> dict[str, Any]:
        value = asdict(self)
        value["recommendation_path"] = (
            None
            if self.recommendation_path is None
            else str(self.recommendation_path)
        )
        return value


def canonical_monthly_logical_time(
    policy: RetrainingPolicy,
    evaluation_period: str,
) -> datetime:
    """Return the policy-fixed UTC logical timestamp for YYYY-MM."""
    try:
        month = date.fromisoformat(f"{evaluation_period}-01")
    except ValueError as exc:
        raise MonthlyGovernanceError(
            "evaluation_period must use YYYY-MM."
        ) from exc
    local = datetime.combine(
        month.replace(day=policy.evaluation_day_of_month),
        time(hour=policy.evaluation_hour_local),
        tzinfo=ZoneInfo(policy.evaluation_timezone),
    )
    return local.astimezone(timezone.utc)


def run_monthly_governance(
    config: MonthlyGovernanceConfig,
    *,
    client: Any | None = None,
    mlflow_module: Any | None = None,
) -> MonthlyGovernanceResult:
    """Verify evidence and seal recommendation-only monthly governance."""
    try:
        policy = RetrainingPolicy.load(config.policy_path)
    except RetrainingContractError as exc:
        raise MonthlyGovernanceError(str(exc)) from exc
    logical_local = config.logical_at_utc.astimezone(
        ZoneInfo(policy.evaluation_timezone)
    )
    evaluation_period = logical_local.strftime("%Y-%m")
    canonical = canonical_monthly_logical_time(policy, evaluation_period)
    if config.logical_at_utc != canonical:
        raise MonthlyGovernanceError(
            "logical_at_utc must equal the policy-fixed monthly schedule."
        )
    if config.now_utc < config.logical_at_utc:
        raise MonthlyGovernanceError(
            "The monthly schedule has not been reached in wall-clock time."
        )
    monitoring_cutoff = logical_local.date().replace(day=1) - timedelta(days=1)
    selected_report_path, selected_report = select_month_end_report(
        config.monitoring_store_root,
        monitoring_cutoff,
    )
    try:
        ledger = load_verified_monitoring_state(config.monitoring_store_root)
        report_state = load_monitoring_report_state(config.monitoring_store_root)
        verified = load_verified_deployment_pointer(
            config.deployment_root,
            client=client,
            mlflow_module=mlflow_module,
        )
    except (
        MonitoringError,
        MonitoringReportingError,
        RetrainingDeploymentError,
    ) as exc:
        raise MonthlyGovernanceError(str(exc)) from exc
    if ledger is None or report_state is None:
        raise MonthlyGovernanceError(
            "Verified monitoring ledger/report state is absent."
        )
    active_era_id = str(ledger.get("active_model_era_id") or "")
    if not active_era_id:
        raise MonthlyGovernanceError("Active monitoring model era is absent.")
    try:
        active_era = load_model_era(
            config.monitoring_store_root,
            active_era_id,
        )
    except MonitoringError as exc:
        raise MonthlyGovernanceError(str(exc)) from exc
    state = verified["state"]
    lifecycle_status = str(state.get("lifecycle_status") or "stable")
    if lifecycle_status not in {"stable", "probationary"}:
        raise MonthlyGovernanceError("Active deployment lifecycle is invalid.")
    if not _same_active_deployment(state, active_era):
        raise MonthlyGovernanceError(
            "Monitoring era and active deployment identities differ."
        )

    if lifecycle_status == "probationary":
        retraining = {
            "decision": "not_applicable_probationary",
            "outcome": None,
            "evaluation_id": None,
            "evaluation_path": None,
            "reason": "current_champion_is_probationary",
        }
    elif (selected_report.get("model_era") or {}).get(
        "model_era_id"
    ) != active_era_id:
        retraining = {
            "decision": "not_applicable_model_era_changed",
            "outcome": None,
            "evaluation_id": None,
            "evaluation_path": None,
            "reason": "month_end_report_is_not_from_active_model_era",
        }
    else:
        if not _report_matches_active_era(selected_report, state, active_era):
            raise MonthlyGovernanceError(
                "Month-end report model-era identity is inconsistent."
            )
        try:
            evaluation = run_monthly_retraining_evaluation(
                MonthlyRetrainingEvaluationConfig(
                    policy_path=config.policy_path,
                    monitoring_store_root=config.monitoring_store_root,
                    monitoring_report_path=selected_report_path,
                    incumbent_id=str(ledger.get("model_snapshot_id") or ""),
                    incumbent_fit_cutoff=str(state["cutoffs"]["fit_cutoff"]),
                    model_era_id=active_era_id,
                    output_root=config.evaluation_output_root,
                    evaluated_at_utc=config.logical_at_utc,
                    dry_run=config.dry_run,
                )
            )
        except RetrainingEvaluationError as exc:
            raise MonthlyGovernanceError(str(exc)) from exc
        retraining = {
            "decision": "evaluated",
            "outcome": evaluation.outcome,
            "evaluation_id": evaluation.evaluation_id,
            "evaluation_path": (
                None
                if evaluation.evaluation_path is None
                else str(evaluation.evaluation_path)
            ),
            "reason": evaluation.plan.reasons[0],
        }

    stability = _stability_recommendation(
        config,
        policy,
        state=state,
        verified=verified,
        ledger=ledger,
        report_state=report_state,
        active_era_id=active_era_id,
        lifecycle_status=lifecycle_status,
    )
    body = {
        "schema_version": RECOMMENDATION_SCHEMA,
        "evaluation_period": evaluation_period,
        "logical_at_utc": _utc_text(config.logical_at_utc),
        "schedule": {
            "day_of_month": policy.evaluation_day_of_month,
            "hour_local": policy.evaluation_hour_local,
            "timezone": policy.evaluation_timezone,
            "logical_at_local": logical_local.isoformat(),
            "monitoring_cutoff": monitoring_cutoff.isoformat(),
        },
        "policy": {
            "path": str(config.policy_path),
            "sha256": sha256_file(config.policy_path),
            "monitoring_policy_path": str(config.monitoring_policy_path),
            "monitoring_policy_sha256": sha256_file(
                config.monitoring_policy_path
            ),
        },
        "deployment": {
            "pointer_path": str(verified["pointer_path"]),
            "pointer_sha256": sha256_file(verified["pointer_path"]),
            "generation": verified["pointer"]["generation"],
            "deployment_id": state["deployment_id"],
            "deployment_state_id": state["deployment_state_id"],
            "lifecycle_status": lifecycle_status,
            "expected_aliases": dict(state["expected_aliases"]),
            "model_era_id": active_era_id,
        },
        "selected_month_end_report": {
            "report_id": selected_report["report_id"],
            "path": str(selected_report_path),
            "sha256": sha256_file(selected_report_path),
            "through_date": selected_report["through_date"],
            "created_at_utc": selected_report["created_at_utc"],
        },
        "retraining": retraining,
        "stability": stability,
        "safeguards": {
            "recommendation_only": True,
            "backtest": False,
            "training": False,
            "registry_write": False,
            "promotion": False,
            "stability_transition": False,
            "rollback": False,
            "deployment_write": False,
            "monitoring_state_write": False,
        },
    }
    recommendation_id = _identifier("monthly_governance_recommendation", body)
    record = {"recommendation_id": recommendation_id, **body}
    target = (
        config.output_root
        / evaluation_period
        / recommendation_id
        / "recommendation.json"
    )
    if not config.dry_run:
        _seal_period(config.output_root / evaluation_period, record)
        load_monthly_governance_recommendation(target)
    return MonthlyGovernanceResult(
        status="planned" if config.dry_run else "succeeded",
        recommendation_id=recommendation_id,
        recommendation_path=None if config.dry_run else target,
        evaluation_period=evaluation_period,
        retraining_decision=str(retraining["decision"]),
        stability_decision=str(stability["decision"]),
    )


def select_month_end_report(
    monitoring_store_root: str | Path,
    cutoff: date | str,
) -> tuple[Path, dict[str, Any]]:
    """Select the latest valid immutable report for one exact month end."""
    target = date.fromisoformat(str(cutoff))
    root = Path(monitoring_store_root).resolve() / "reporting" / "reports"
    candidates: list[tuple[datetime, str, Path, dict[str, Any]]] = []
    if root.is_dir():
        for path in sorted(root.glob("*/report.json")):
            try:
                report = load_monitoring_report(path)
            except MonitoringReportingError as exc:
                raise MonthlyGovernanceError(str(exc)) from exc
            if report.get("through_date") != target.isoformat():
                continue
            created = _parse_utc(
                report.get("created_at_utc"),
                "report created_at_utc",
            )
            candidates.append(
                (created, str(report["report_id"]), path.resolve(), report)
            )
    if not candidates:
        raise MonthlyGovernanceError(
            f"No verified monitoring report exists for {target.isoformat()}."
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    latest = candidates[-1]
    if (
        len(candidates) > 1
        and candidates[-2][0] == latest[0]
        and candidates[-2][1] != latest[1]
    ):
        raise MonthlyGovernanceError(
            "Month-end reports have an ambiguous latest created_at_utc."
        )
    return latest[2], latest[3]


def load_monthly_governance_recommendation(
    path: str | Path,
) -> dict[str, Any]:
    """Load one strict content-addressed monthly recommendation."""
    target = Path(path)
    payload = _read_json(target)
    required = {
        "recommendation_id",
        "schema_version",
        "evaluation_period",
        "logical_at_utc",
        "schedule",
        "policy",
        "deployment",
        "selected_month_end_report",
        "retraining",
        "stability",
        "safeguards",
    }
    if set(payload) != required or payload.get("schema_version") != (
        RECOMMENDATION_SCHEMA
    ):
        raise MonthlyGovernanceError(
            "Monthly governance recommendation fields are invalid."
        )
    identifier = payload.get("recommendation_id")
    body = {key: value for key, value in payload.items() if key != "recommendation_id"}
    if identifier != _identifier("monthly_governance_recommendation", body):
        raise MonthlyGovernanceError(
            "Monthly governance recommendation identity is corrupt."
        )
    if (
        target.name != "recommendation.json"
        or target.parent.name != identifier
        or target.parent.parent.name != payload.get("evaluation_period")
    ):
        raise MonthlyGovernanceError(
            "Monthly governance recommendation path differs from its identity."
        )
    if (payload.get("retraining") or {}).get("decision") not in (
        RETRAINING_DECISIONS
    ) or (payload.get("stability") or {}).get("decision") not in (
        STABILITY_DECISIONS
    ):
        raise MonthlyGovernanceError(
            "Monthly governance decisions are invalid."
        )
    safeguards = payload.get("safeguards") or {}
    if safeguards.get("recommendation_only") is not True or any(
        safeguards.get(key) is not False
        for key in (
            "backtest",
            "training",
            "registry_write",
            "promotion",
            "stability_transition",
            "rollback",
            "deployment_write",
            "monitoring_state_write",
        )
    ):
        raise MonthlyGovernanceError(
            "Monthly governance safeguards are invalid."
        )
    return payload


def _stability_recommendation(
    config: MonthlyGovernanceConfig,
    policy: RetrainingPolicy,
    *,
    state: Mapping[str, Any],
    verified: Mapping[str, Any],
    ledger: Mapping[str, Any],
    report_state: Mapping[str, Any],
    active_era_id: str,
    lifecycle_status: str,
) -> dict[str, Any]:
    if lifecycle_status == "stable":
        return {
            "decision": "not_applicable_stable",
            "reason": "current_champion_is_already_stable",
            "eligible_observation_count": 0,
            "fixed_observations": [],
            "observation_cutoff": None,
            "current_health_report": None,
            "second_manual_approval_required": True,
        }
    latest_id = str(report_state.get("latest_report_id") or "")
    latest_path = (
        config.monitoring_store_root
        / "reporting"
        / "reports"
        / latest_id
        / "report.json"
    )
    try:
        latest = load_monitoring_report(latest_path)
    except MonitoringReportingError as exc:
        raise MonthlyGovernanceError(str(exc)) from exc
    try:
        era = load_model_era(config.monitoring_store_root, active_era_id)
    except MonitoringError as exc:
        raise MonthlyGovernanceError(str(exc)) from exc
    monitoring_policy = _read_json(config.monitoring_policy_path)
    if (
        latest.get("through_date") != report_state.get("latest_through_date")
        or not _report_matches_active_era(latest, state, era)
        or (latest.get("reference") or {}).get("policy_sha256")
        != sha256_file(config.monitoring_policy_path)
        or latest.get("config") != monitoring_policy
        or (latest.get("active_alerts") or {})
        != (report_state.get("active_alerts") or {})
    ):
        raise MonthlyGovernanceError(
            "Latest monitoring report is not current for the probationary era."
        )
    start = date.fromisoformat(state["cutoffs"]["promotion_effective_date"])
    end = date.fromisoformat(str(latest["through_date"]))
    adopted = era.get("_adopted_state") or {}
    adopted_ids = {
        str(value)
        for value in (
            list((adopted.get("as_issued") or {}).values())
            + list((adopted.get("restated") or {}).values())
        )
    }
    eligible: list[dict[str, str]] = []
    for day, prediction_id in sorted((ledger.get("as_issued") or {}).items()):
        target = date.fromisoformat(day)
        if target < start or target > end:
            continue
        try:
            evidence = load_prediction_evidence(
                config.monitoring_store_root,
                str(prediction_id),
            )
        except MonitoringError as exc:
            raise MonthlyGovernanceError(str(exc)) from exc
        prediction = evidence["prediction"]
        if (
            prediction.get("model_era_id") != active_era_id
            and str(prediction_id) not in adopted_ids
        ):
            continue
        actual_id = (ledger.get("actuals") or {}).get(day)
        actual = next(
            (
                item
                for item in evidence.get("actual_revisions", [])
                if item.get("actual_revision_id") == actual_id
            ),
            None,
        )
        if prediction.get("target_date") != day:
            raise MonthlyGovernanceError(
                "Probation prediction target date differs from the ledger."
            )
        if prediction.get("issuance_kind") not in set(
            policy.stability_allowed_issuance_kinds
        ):
            continue
        if actual_id is None:
            continue
        if actual is None or actual.get("target_date") != day:
            raise MonthlyGovernanceError(
                "Probation actual revision differs from the ledger."
            )
        if not _finite(prediction.get("prediction")) or not _finite(
            actual.get("actual")
        ):
            continue
        eligible.append(
            {
                "target_date": day,
                "prediction_id": str(prediction_id),
                "actual_revision_id": str(actual_id),
            }
        )
    minimum = policy.stability_minimum_eligible_observations
    fixed = eligible[:minimum]
    health = {
        "report_id": latest["report_id"],
        "path": str(latest_path.resolve()),
        "sha256": sha256_file(latest_path),
        "through_date": latest["through_date"],
    }
    common = {
        "eligible_observation_count": len(eligible),
        "fixed_observations": fixed,
        "observation_cutoff": (
            fixed[-1]["target_date"] if len(fixed) == minimum else None
        ),
        "current_health_report": health,
        "expected_pointer_sha256": sha256_file(verified["pointer_path"]),
        "second_manual_approval_required": True,
    }
    if len(fixed) < minimum:
        return {
            "decision": "insufficient_observations",
            "reason": "first_90_eligible_probation_observations_not_available",
            **common,
        }
    unhealthy = bool((latest.get("quality") or {}).get("issues")) or bool(
        latest.get("active_alerts")
    ) or any(
        item.get("severity") in {"warning", "critical"}
        for item in latest.get("breaches", [])
    )
    if unhealthy:
        return {
            "decision": "blocked_health",
            "reason": "current_monitoring_health_blocks_stability",
            **common,
        }
    return {
        "decision": "ready_for_second_manual_approval",
        "reason": "first_90_fixed_and_current_monitoring_is_healthy",
        **common,
    }


def _same_active_deployment(
    state: Mapping[str, Any],
    era: Mapping[str, Any],
) -> bool:
    deployment = era.get("deployment") or {}
    registry = era.get("registry") or {}
    state_registry = state.get("registry") or {}
    return (
        deployment.get("deployment_id") == state.get("deployment_id")
        and deployment.get("deployment_state_id")
        == state.get("deployment_state_id")
        and deployment.get("generation") == state.get("generation")
        and registry.get("registered_model_name")
        == state_registry.get("registered_model_name")
        and str(registry.get("model_version"))
        == str(state_registry.get("model_version"))
        and era.get("cutoffs") == state.get("cutoffs")
    )


def _report_matches_active_era(
    report: Mapping[str, Any],
    state: Mapping[str, Any],
    era: Mapping[str, Any],
) -> bool:
    report_era = report.get("model_era") or {}
    deployment = era.get("deployment") or {}
    registry = era.get("registry") or {}
    return (
        _same_active_deployment(state, era)
        and report_era.get("model_era_id") == era.get("model_era_id")
        and report_era.get("deployment_id")
        == deployment.get("deployment_id")
        and report_era.get("deployment_state_id")
        == deployment.get("deployment_state_id")
        and report_era.get("deployment_generation")
        == deployment.get("generation")
        and report_era.get("registered_model_name")
        == registry.get("registered_model_name")
        and str(report_era.get("model_version"))
        == str(registry.get("model_version"))
        and report_era.get("cutoffs") == era.get("cutoffs")
        and report_era.get("pins") == era.get("pins")
    )


def _seal_period(period_root: Path, record: Mapping[str, Any]) -> None:
    recommendation_id = str(record["recommendation_id"])
    if period_root.exists():
        entries = list(period_root.iterdir())
        if (
            len(entries) != 1
            or not entries[0].is_dir()
            or entries[0].is_symlink()
            or entries[0].name != recommendation_id
        ):
            raise MonthlyGovernanceError(
                "A different recommendation is already sealed for this period."
            )
        existing = entries[0] / "recommendation.json"
        if not existing.is_file() or _read_json(existing) != record:
            raise MonthlyGovernanceError(
                "Sealed monthly recommendation differs from the retry."
            )
        return
    period_root.parent.mkdir(parents=True, exist_ok=True)
    prepared = period_root.parent / (
        f".{period_root.name}.{recommendation_id}.{uuid4().hex}.tmp"
    )
    target = prepared / recommendation_id / "recommendation.json"
    prepared.mkdir(exist_ok=False)
    try:
        target.parent.mkdir()
        _immutable_json(target, record)
        try:
            os.rename(prepared, period_root)
        except FileExistsError:
            _seal_period(period_root, record)
    finally:
        if prepared.exists():
            for child in sorted(prepared.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            prepared.rmdir()


def _identifier(kind: str, payload: Mapping[str, Any]) -> str:
    return sha256(kind.encode("utf-8") + b":" + _canonical(payload)).hexdigest()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise MonthlyGovernanceError(f"{name} must be UTC.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MonthlyGovernanceError(f"{name} must be UTC.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise MonthlyGovernanceError(f"{name} must be UTC.")
    return parsed


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonthlyGovernanceError(f"Invalid JSON artifact: {path}.") from exc
    if not isinstance(payload, dict):
        raise MonthlyGovernanceError(f"JSON artifact is not an object: {path}.")
    return payload


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise MonthlyGovernanceError(
                f"Immutable recommendation differs: {path}."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != data:
            raise MonthlyGovernanceError(
                f"Immutable recommendation differs: {path}."
            )


__all__ = [
    "MonthlyGovernanceConfig",
    "MonthlyGovernanceError",
    "MonthlyGovernanceResult",
    "RECOMMENDATION_SCHEMA",
    "STABILITY_DECISIONS",
    "canonical_monthly_logical_time",
    "load_monthly_governance_recommendation",
    "run_monthly_governance",
    "select_month_end_report",
]
