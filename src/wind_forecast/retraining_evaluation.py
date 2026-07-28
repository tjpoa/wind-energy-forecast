"""Manual, fail-closed monthly eligibility evaluation for controlled retraining."""

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
    load_alert_history,
    load_monitoring_report,
    load_monitoring_report_state,
)
from wind_forecast.retraining_policy import (
    ObservationEvidence,
    RetrainingContractError,
    RetrainingPolicy,
    select_eligible_observations,
)


EVALUATION_SCHEMA = "wind_forecast.monthly_retraining_evaluation.v1"
EVALUATION_SCHEMA_V2 = "wind_forecast.monthly_retraining_evaluation.v2"
EVALUATION_OUTCOMES = (
    "blocked_quality",
    "insufficient_observations",
    "no_trigger",
    "eligible_for_manual_backtest",
)


class RetrainingEvaluationError(RuntimeError):
    """Raised when monthly evaluation evidence cannot be trusted or sealed."""


@dataclass(frozen=True)
class MonthlyRetrainingEvaluationConfig:
    """Explicit inputs for one operator-pinned monthly evaluation."""

    policy_path: Path
    monitoring_store_root: Path
    monitoring_report_path: Path
    incumbent_id: str
    incumbent_fit_cutoff: date
    evaluated_at_utc: datetime | str
    model_era_id: str | None = None
    output_root: Path = Path("data/processed/v2/retraining/evaluations")
    dry_run: bool = False

    def __post_init__(self) -> None:
        for name in (
            "policy_path",
            "monitoring_store_root",
            "monitoring_report_path",
            "output_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if not isinstance(self.incumbent_id, str) or not self.incumbent_id.strip():
            raise RetrainingEvaluationError("incumbent_id must be a non-empty string.")
        if self.model_era_id is not None and (
            not isinstance(self.model_era_id, str)
            or not self.model_era_id.strip()
        ):
            raise RetrainingEvaluationError(
                "model_era_id must be a non-empty string when supplied."
            )
        cutoff = self.incumbent_fit_cutoff
        if not isinstance(cutoff, date):
            try:
                cutoff = date.fromisoformat(str(cutoff))
            except ValueError as exc:
                raise RetrainingEvaluationError(
                    "incumbent_fit_cutoff must be an ISO-8601 date."
                ) from exc
            object.__setattr__(self, "incumbent_fit_cutoff", cutoff)
        evaluated = self.evaluated_at_utc
        if isinstance(evaluated, str):
            try:
                evaluated = datetime.fromisoformat(evaluated.replace("Z", "+00:00"))
            except ValueError as exc:
                raise RetrainingEvaluationError(
                    "evaluated_at_utc must be an ISO-8601 timestamp."
                ) from exc
        if not isinstance(evaluated, datetime) or evaluated.tzinfo is None:
            raise RetrainingEvaluationError(
                "evaluated_at_utc must be an explicit timezone-aware timestamp."
            )
        if evaluated.utcoffset() != timedelta(0):
            raise RetrainingEvaluationError("evaluated_at_utc must be UTC.")
        object.__setattr__(self, "evaluated_at_utc", evaluated.astimezone(timezone.utc))


@dataclass(frozen=True)
class MonthlyRetrainingEvaluationPlan:
    """Read-only evaluation result, including the record that would be sealed."""

    status: str
    outcome: str
    evaluation_period: str
    scheduled_at_local: str
    record: Mapping[str, Any] | None
    reasons: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "record": None if self.record is None else dict(self.record),
        }


@dataclass(frozen=True)
class MonthlyRetrainingEvaluationResult:
    """Persisted, dry-run, or not-due monthly evaluation outcome."""

    status: str
    outcome: str
    evaluation_id: str | None
    evaluation_path: Path | None
    plan: MonthlyRetrainingEvaluationPlan

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "outcome": self.outcome,
            "evaluation_id": self.evaluation_id,
            "evaluation_path": (
                str(self.evaluation_path) if self.evaluation_path else None
            ),
            "plan": self.plan.summary(),
        }


def plan_monthly_retraining_evaluation(
    config: MonthlyRetrainingEvaluationConfig,
) -> MonthlyRetrainingEvaluationPlan:
    """Verify pinned evidence and compute a recommendation without writing."""
    try:
        policy = RetrainingPolicy.load(config.policy_path)
    except RetrainingContractError as exc:
        raise RetrainingEvaluationError(str(exc)) from exc
    report, report_state, alerts = _load_pinned_report(config)
    monitoring_cutoff = _parse_date(report.get("through_date"), "report through_date")
    next_month = (
        monitoring_cutoff.replace(day=1) + timedelta(days=32)
    ).replace(day=1)
    local_zone = ZoneInfo(policy.evaluation_timezone)
    evaluated_local = config.evaluated_at_utc.astimezone(local_zone)
    scheduled_local = datetime.combine(
        next_month.replace(day=policy.evaluation_day_of_month),
        time(hour=policy.evaluation_hour_local),
        tzinfo=local_zone,
    )
    evaluation_period = scheduled_local.strftime("%Y-%m")
    if evaluated_local < scheduled_local:
        return MonthlyRetrainingEvaluationPlan(
            status="not_due",
            outcome="not_due",
            evaluation_period=evaluation_period,
            scheduled_at_local=scheduled_local.isoformat(),
            record=None,
            reasons=("monthly_schedule_not_reached",),
        )

    if config.incumbent_fit_cutoff >= monitoring_cutoff:
        raise RetrainingEvaluationError(
            "incumbent_fit_cutoff must precede the pinned monitoring cutoff."
        )
    if monitoring_cutoff + timedelta(days=7) > evaluated_local.date():
        raise RetrainingEvaluationError(
            "Pinned monitoring cutoff has not crossed the D+7 lateness boundary."
        )
    ledger = _load_ledger(config.monitoring_store_root)
    if ledger.get("model_snapshot_id") != config.incumbent_id:
        raise RetrainingEvaluationError(
            "Explicit incumbent_id differs from the verified Phase 9 model snapshot."
        )
    if config.model_era_id is not None:
        if (
            ledger.get("active_model_era_id") != config.model_era_id
            or (report.get("model_era") or {}).get("model_era_id")
            != config.model_era_id
        ):
            raise RetrainingEvaluationError(
                "Explicit model_era_id differs from the active ledger/report era."
            )
    _cross_check_report_ledger(report, ledger)
    alert_details = _active_alert_details(report, alerts, policy)
    blockers = tuple(
        item
        for item in alert_details
        if item["category"] in policy.blocking_categories
    )
    triggers = tuple(
        item
        for item in alert_details
        if item["category"] in policy.trigger_categories
        and item["severity"] in policy.trigger_severities
        and item["consecutive"] >= policy.phase9_persistence_distinct_reports
        and item["required"] == policy.phase9_persistence_distinct_reports
    )
    eligibility = _build_eligibility(
        config,
        ledger,
        monitoring_cutoff=monitoring_cutoff,
    )
    eligible_count = len(eligibility["eligible_observation_ids"])
    if blockers:
        outcome = "blocked_quality"
        reasons = ("active_quality_alert",)
    elif eligible_count < policy.minimum_new_eligible_observations:
        outcome = "insufficient_observations"
        reasons = ("minimum_new_eligible_observations_not_met",)
    elif not triggers:
        outcome = "no_trigger"
        reasons = ("no_persistent_warning_or_critical_trigger",)
    else:
        outcome = "eligible_for_manual_backtest"
        reasons = ("data_and_phase9_trigger_gates_passed",)
    if outcome not in EVALUATION_OUTCOMES:
        raise RetrainingEvaluationError("Evaluation produced an unsupported outcome.")

    state_path = config.monitoring_store_root / "reporting" / "state" / "current.json"
    ledger_path = config.monitoring_store_root / "state" / "current.json"
    body = {
        "schema_version": (
            EVALUATION_SCHEMA_V2
            if config.model_era_id is not None
            else EVALUATION_SCHEMA
        ),
        "evaluation_period": evaluation_period,
        "evaluated_at_utc": _utc_text(config.evaluated_at_utc),
        "schedule": {
            "day_of_month": policy.evaluation_day_of_month,
            "hour_local": policy.evaluation_hour_local,
            "timezone": policy.evaluation_timezone,
            "scheduled_at_local": scheduled_local.isoformat(),
            "schedule_gate_passed": True,
        },
        "policy": {
            "path": str(config.policy_path.resolve()),
            "sha256": sha256_file(config.policy_path),
            "schema_version": policy.schema_version,
            "minimum_new_eligible_observations": (
                policy.minimum_new_eligible_observations
            ),
            "persistence_distinct_reports": (
                policy.phase9_persistence_distinct_reports
            ),
            "trigger_categories": list(policy.trigger_categories),
            "trigger_severities": list(policy.trigger_severities),
            "blocking_categories": list(policy.blocking_categories),
        },
        "incumbent": {
            "incumbent_id": config.incumbent_id,
            "identity_role": "transitional_incumbent_input",
            "champion_claim": False,
            **(
                {"model_era_id": config.model_era_id}
                if config.model_era_id is not None
                else {}
            ),
        },
        "cutoffs": {
            "incumbent_fit_cutoff": config.incumbent_fit_cutoff.isoformat(),
            "monitoring_evaluation_cutoff": monitoring_cutoff.isoformat(),
            "data_snapshot_cutoff": eligibility["data_snapshot_cutoff"],
        },
        "phase9_report": {
            "report_id": report["report_id"],
            "path": str(config.monitoring_report_path.resolve()),
            "sha256": sha256_file(config.monitoring_report_path),
            "through_date": report["through_date"],
            "report_ledger_generation": (report.get("lineage") or {}).get(
                "ledger_generation"
            ),
        },
        "phase9_report_state": {
            "generation": report_state.get("generation"),
            "latest_report_id": report_state.get("latest_report_id"),
            "latest_through_date": report_state.get("latest_through_date"),
            "path": str(state_path.resolve()),
            "sha256": sha256_file(state_path),
        },
        "phase9_ledger": {
            "generation": ledger.get("generation"),
            "model_snapshot_id": ledger.get("model_snapshot_id"),
            **(
                {"active_model_era_id": ledger.get("active_model_era_id")}
                if config.model_era_id is not None
                else {}
            ),
            "path": str(ledger_path.resolve()),
            "sha256": sha256_file(ledger_path),
            "feature_view": "as_issued",
            "target_revision_view": "current",
        },
        "eligibility": eligibility,
        "active_quality_blockers": list(blockers),
        "active_triggers": list(triggers),
        "outcome": outcome,
        "reasons": list(reasons),
        "recommendation": (
            "manual_temporal_backtest"
            if outcome == "eligible_for_manual_backtest"
            else "no_retraining_action"
        ),
        "safeguards": {
            "training": False,
            "registry_write": False,
            "deployment_write": False,
            "promotion": False,
            "stability": False,
            "rollback": False,
            "network_requests": False,
            "monitoring_state_write": False,
            "monitoring_persistence_increment": False,
            "restatements_used_for_features": False,
            "recommendation_only": True,
        },
    }
    record = _with_id(body)
    return MonthlyRetrainingEvaluationPlan(
        status="planned",
        outcome=outcome,
        evaluation_period=evaluation_period,
        scheduled_at_local=scheduled_local.isoformat(),
        record=record,
        reasons=reasons,
    )


def run_monthly_retraining_evaluation(
    config: MonthlyRetrainingEvaluationConfig,
) -> MonthlyRetrainingEvaluationResult:
    """Plan and, unless dry-run/not-due, seal one immutable monthly record."""
    plan = plan_monthly_retraining_evaluation(config)
    if plan.status == "not_due":
        return MonthlyRetrainingEvaluationResult(
            status="not_due",
            outcome=plan.outcome,
            evaluation_id=None,
            evaluation_path=None,
            plan=plan,
        )
    if plan.record is None:
        raise RetrainingEvaluationError("Planned evaluation record is absent.")
    evaluation_id = str(plan.record["evaluation_id"])
    path = (
        config.output_root
        / plan.evaluation_period
        / evaluation_id
        / "evaluation.json"
    )
    if config.dry_run:
        return MonthlyRetrainingEvaluationResult(
            status="planned",
            outcome=plan.outcome,
            evaluation_id=evaluation_id,
            evaluation_path=None,
            plan=plan,
        )
    _seal_period(config.output_root / plan.evaluation_period, plan.record)
    return MonthlyRetrainingEvaluationResult(
        status="succeeded",
        outcome=plan.outcome,
        evaluation_id=evaluation_id,
        evaluation_path=path,
        plan=plan,
    )


def load_monthly_retraining_evaluation(path: str | Path) -> dict[str, Any]:
    """Load one strict content-addressed monthly recommendation."""
    record_path = Path(path)
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrainingEvaluationError(
            f"Invalid monthly retraining evaluation: {record_path}."
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        EVALUATION_SCHEMA,
        EVALUATION_SCHEMA_V2,
    }:
        raise RetrainingEvaluationError("Unsupported monthly evaluation schema.")
    identifier = payload.get("evaluation_id")
    if not isinstance(identifier, str) or identifier != _record_id(
        {key: value for key, value in payload.items() if key != "evaluation_id"}
    ):
        raise RetrainingEvaluationError(
            "Corrupt monthly evaluation content-addressed identity."
        )
    if record_path.name != "evaluation.json" or record_path.parent.name != identifier:
        raise RetrainingEvaluationError(
            "Monthly evaluation path differs from its content identity."
        )
    required = {
        "evaluation_id",
        "schema_version",
        "evaluation_period",
        "evaluated_at_utc",
        "schedule",
        "policy",
        "incumbent",
        "cutoffs",
        "phase9_report",
        "phase9_report_state",
        "phase9_ledger",
        "eligibility",
        "active_quality_blockers",
        "active_triggers",
        "outcome",
        "reasons",
        "recommendation",
        "safeguards",
    }
    if set(payload) != required or payload.get("outcome") not in EVALUATION_OUTCOMES:
        raise RetrainingEvaluationError(
            "Monthly evaluation fields differ from the v1 contract."
        )
    incumbent = payload.get("incumbent")
    ledger = payload.get("phase9_ledger")
    if not isinstance(incumbent, dict) or not isinstance(ledger, dict):
        raise RetrainingEvaluationError("Monthly evaluation identities are invalid.")
    if payload["schema_version"] == EVALUATION_SCHEMA_V2:
        era_id = incumbent.get("model_era_id")
        if (
            not isinstance(era_id, str)
            or not era_id
            or ledger.get("active_model_era_id") != era_id
        ):
            raise RetrainingEvaluationError(
                "Monthly v2 evaluation model-era identity is invalid."
            )
    elif set(incumbent) != {
        "incumbent_id",
        "identity_role",
        "champion_claim",
    }:
        raise RetrainingEvaluationError(
            "Monthly v1 evaluation incumbent fields are invalid."
        )
    if record_path.parent.parent.name != payload.get("evaluation_period"):
        raise RetrainingEvaluationError(
            "Monthly evaluation period path differs from its record."
        )
    return payload


def _load_pinned_report(
    config: MonthlyRetrainingEvaluationConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    expected_root = (
        config.monitoring_store_root.resolve() / "reporting" / "reports"
    )
    path = config.monitoring_report_path.resolve()
    try:
        path.relative_to(expected_root)
    except ValueError as exc:
        raise RetrainingEvaluationError(
            "Pinned report must belong to the configured Phase 9 store."
        ) from exc
    try:
        report = load_monitoring_report(path)
        state = load_monitoring_report_state(config.monitoring_store_root)
        history = load_alert_history(config.monitoring_store_root)
    except (MonitoringReportingError, OSError) as exc:
        raise RetrainingEvaluationError(str(exc)) from exc
    if state is None:
        raise RetrainingEvaluationError("Verified Phase 9 report state is absent.")
    generation = state.get("generation")
    latest_report_id = state.get("latest_report_id")
    latest_cutoff = _parse_date(
        state.get("latest_through_date"), "report-state latest_through_date"
    )
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
        or not isinstance(latest_report_id, str)
        or not latest_report_id
        or latest_cutoff < _parse_date(report.get("through_date"), "report through_date")
    ):
        raise RetrainingEvaluationError(
            "Verified Phase 9 report state does not cover the pinned report."
        )
    if path.parent.name != report.get("report_id"):
        raise RetrainingEvaluationError(
            "Pinned report directory differs from its content identity."
        )
    events = {str(item["alert_event_id"]): item for item in history}
    return report, state, events


def _load_ledger(root: Path) -> dict[str, Any]:
    try:
        ledger = load_verified_monitoring_state(root)
    except MonitoringError as exc:
        raise RetrainingEvaluationError(str(exc)) from exc
    if ledger is None:
        raise RetrainingEvaluationError("Verified Phase 9 ledger is absent.")
    return ledger


def _cross_check_report_ledger(
    report: Mapping[str, Any], ledger: Mapping[str, Any]
) -> None:
    as_issued = ledger.get("as_issued")
    if not isinstance(as_issued, dict):
        raise RetrainingEvaluationError("Phase 9 as-issued ledger view is invalid.")
    lineage = report.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("primary_view") != "as_issued":
        raise RetrainingEvaluationError(
            "Pinned report does not declare the as-issued primary view."
        )
    for prediction_id in lineage.get("prediction_ids") or []:
        if prediction_id not in as_issued.values():
            raise RetrainingEvaluationError(
                "Pinned report lineage is absent from the verified as-issued ledger."
            )


def _active_alert_details(
    report: Mapping[str, Any],
    events: Mapping[str, Mapping[str, Any]],
    policy: RetrainingPolicy,
) -> tuple[dict[str, Any], ...]:
    active = report.get("active_alerts")
    persistence = report.get("persistence")
    breaches = report.get("breaches")
    if (
        not isinstance(active, dict)
        or not isinstance(persistence, dict)
        or not isinstance(breaches, list)
    ):
        raise RetrainingEvaluationError("Pinned report alert evidence is invalid.")
    breaches_by_rule: dict[str, Mapping[str, Any]] = {}
    for item in breaches:
        if not isinstance(item, dict) or not isinstance(item.get("rule_id"), str):
            raise RetrainingEvaluationError("Pinned report breach evidence is invalid.")
        rule_id = item["rule_id"]
        if rule_id in breaches_by_rule:
            raise RetrainingEvaluationError(
                "Pinned report contains duplicate breach rules."
            )
        breaches_by_rule[rule_id] = item
    details: list[dict[str, Any]] = []
    allowed_categories = set(policy.trigger_categories) | set(
        policy.blocking_categories
    )
    severity_order = {"warning": 1, "critical": 2}
    report_cutoff = _parse_date(report.get("through_date"), "report through_date")
    for rule_id, event_id in sorted(active.items()):
        if not isinstance(rule_id, str) or not isinstance(event_id, str):
            raise RetrainingEvaluationError("Pinned active-alert mapping is invalid.")
        breach = breaches_by_rule.get(rule_id)
        rule = persistence.get(rule_id)
        event = events.get(event_id)
        if breach is None or not isinstance(rule, dict) or event is None:
            raise RetrainingEvaluationError(
                "Pinned active alert lacks matching breach, state, or event evidence."
            )
        category = breach.get("category")
        severity = rule.get("severity")
        consecutive = rule.get("consecutive")
        required = rule.get("required")
        rule_date = _parse_date(rule.get("last_date"), "active rule last_date")
        event_date = _parse_date(
            event.get("through_date"), "active alert event through_date"
        )
        if (
            category not in allowed_categories
            or breach.get("severity") not in policy.trigger_severities
            or severity not in policy.trigger_severities
            or severity_order[str(breach.get("severity"))]
            > severity_order[str(severity)]
            or event.get("severity") != severity
            or not isinstance(consecutive, int)
            or isinstance(consecutive, bool)
            or not isinstance(required, int)
            or isinstance(required, bool)
            or not rule.get("active")
            or rule.get("last_event_id") != event_id
            or event.get("rule_id") != rule_id
            or event.get("event_type") not in {"opened", "escalated"}
            or rule_date != report_cutoff
            or event_date > rule_date
        ):
            raise RetrainingEvaluationError(
                "Pinned active-alert category/state/event evidence disagrees."
            )
        details.append(
            {
                "rule_id": rule_id,
                "category": category,
                "severity": severity,
                "breach_severity": breach["severity"],
                "alert_event_id": event_id,
                "event_type": event["event_type"],
                "event_through_date": event["through_date"],
                "consecutive": consecutive,
                "required": required,
            }
        )
    return tuple(details)


def _build_eligibility(
    config: MonthlyRetrainingEvaluationConfig,
    ledger: Mapping[str, Any],
    *,
    monitoring_cutoff: date,
) -> dict[str, Any]:
    as_issued = ledger.get("as_issued") or {}
    actuals = ledger.get("actuals") or {}
    restated = ledger.get("restated") or {}
    if not all(isinstance(item, dict) for item in (as_issued, actuals, restated)):
        raise RetrainingEvaluationError("Phase 9 observation views are invalid.")
    candidate_days = [
        day
        for day in sorted(as_issued)
        if config.incumbent_fit_cutoff < _parse_date(day, "as-issued date")
        <= monitoring_cutoff
    ]
    observations: list[ObservationEvidence] = []
    exclusions: dict[str, tuple[str, ...]] = {}
    expected_contract: str | None = None
    expected_transformation: str | None = None
    expected_schema: str | None = None
    restatements_ignored: list[dict[str, str]] = []
    adopted_prediction_ids: set[str] = set()
    if config.model_era_id is not None:
        try:
            era = load_model_era(
                config.monitoring_store_root,
                config.model_era_id,
            )
        except MonitoringError as exc:
            raise RetrainingEvaluationError(str(exc)) from exc
        adopted = era.get("_adopted_state") or {}
        adopted_prediction_ids = {
            str(value)
            for value in (
                list((adopted.get("as_issued") or {}).values())
                + list((adopted.get("restated") or {}).values())
            )
        }
    for day in candidate_days:
        prediction_id = as_issued[day]
        if not isinstance(prediction_id, str):
            raise RetrainingEvaluationError("As-issued prediction ID is invalid.")
        try:
            evidence = load_prediction_evidence(
                config.monitoring_store_root, prediction_id
            )
        except MonitoringError as exc:
            raise RetrainingEvaluationError(str(exc)) from exc
        prediction = evidence["prediction"]
        snapshot = evidence["model_input_snapshot"]
        model = evidence["model_snapshot"]
        prediction_era_id = prediction.get("model_era_id")
        if config.model_era_id is not None and (
            prediction_era_id != config.model_era_id
            and prediction_id not in adopted_prediction_ids
        ):
            exclusions[prediction_id] = ("different_model_era",)
            continue
        if (
            prediction.get("view") != "as_issued"
            or prediction.get("target_date") != day
            or snapshot.get("target_date") != day
            or prediction.get("model_snapshot_id") != config.incumbent_id
            or model.get("model_snapshot_id") != config.incumbent_id
        ):
            raise RetrainingEvaluationError(
                "As-issued observation disagrees with the explicit incumbent contract."
            )
        current_actual_id = actuals.get(day)
        if current_actual_id is None:
            exclusions[prediction_id] = ("missing_current_target_revision",)
            continue
        current_actual = next(
            (
                item
                for item in evidence.get("actual_revisions") or []
                if item.get("actual_revision_id") == current_actual_id
            ),
            None,
        )
        if current_actual is None:
            raise RetrainingEvaluationError(
                "Current target revision is absent from verified prediction evidence."
            )
        contract = current_actual.get("target_contract_id")
        transformation = (snapshot.get("transformation") or {}).get("version")
        feature_schema = snapshot.get("feature_schema_sha256")
        if not all(
            isinstance(value, str) and value
            for value in (contract, transformation, feature_schema)
        ):
            raise RetrainingEvaluationError(
                "Observation contract metadata is missing or invalid."
            )
        expected_contract = expected_contract or contract
        expected_transformation = expected_transformation or transformation
        expected_schema = expected_schema or feature_schema
        if (
            contract != expected_contract
            or transformation != expected_transformation
            or feature_schema != expected_schema
        ):
            raise RetrainingEvaluationError(
                "Eligible observations contain incompatible contracts."
            )
        feature_values = snapshot.get("feature_values")
        target_value = current_actual.get("actual")
        if (
            not isinstance(feature_values, list)
            or not feature_values
            or not all(_finite_number(value) for value in feature_values)
            or not _finite_number(target_value)
        ):
            raise RetrainingEvaluationError(
                "Eligible observation contains non-finite or invalid numeric evidence."
            )
        dependencies = snapshot.get("dependencies")
        if not isinstance(dependencies, dict):
            raise RetrainingEvaluationError("Observation lineage is invalid.")
        source_revision_ids = _source_revision_ids(
            dependencies, current_actual
        )
        lineage = {
            "dependencies": dependencies,
            "target_revision_id": current_actual_id,
        }
        observation_id = sha256(
            b"retraining_observation:" + _canonical(
                {
                    "prediction_id": prediction_id,
                    "feature_snapshot_id": snapshot["model_input_snapshot_id"],
                    "target_revision_id": current_actual_id,
                }
            )
        ).hexdigest()
        observations.append(
            ObservationEvidence(
                observation_id=observation_id,
                target_date=day,
                feature_snapshot_id=snapshot["model_input_snapshot_id"],
                target_revision_id=current_actual_id,
                feature_schema_sha256=feature_schema,
                lineage_sha256=sha256(_canonical(lineage)).hexdigest(),
                target_contract_id=contract,
                transformation_version=transformation,
                source_revision_ids=source_revision_ids,
                feature_values=tuple(feature_values),
                target_value=float(target_value),
            )
        )
        if day in restated:
            restated_id = restated[day]
            if not isinstance(restated_id, str):
                raise RetrainingEvaluationError("Restated prediction ID is invalid.")
            restatements_ignored.append(
                {
                    "target_date": day,
                    "as_issued_prediction_id": prediction_id,
                    "ignored_restatement_prediction_id": restated_id,
                }
            )
    if observations:
        assert expected_contract is not None
        assert expected_transformation is not None
        assert expected_schema is not None
        try:
            selection = select_eligible_observations(
                observations,
                expected_target_contract_id=expected_contract,
                expected_transformation_version=expected_transformation,
                expected_feature_schema_sha256=expected_schema,
            )
        except RetrainingContractError as exc:
            raise RetrainingEvaluationError(str(exc)) from exc
        eligible = selection.eligible
        selection_exclusions = dict(selection.exclusions)
    else:
        eligible = ()
        selection_exclusions = {}
    combined_exclusions = {
        **exclusions,
        **selection_exclusions,
    }
    return {
        "eligible_observation_count": len(eligible),
        "eligible_observation_ids": [
            item.observation_id for item in eligible
        ],
        "eligible_observations": [
            {
                "observation_id": item.observation_id,
                "target_date": item.target_date.isoformat(),
                "feature_snapshot_id": item.feature_snapshot_id,
                "target_revision_id": item.target_revision_id,
                "lineage_sha256": item.lineage_sha256,
            }
            for item in eligible
        ],
        "excluded": {
            key: list(value)
            for key, value in sorted(combined_exclusions.items())
        },
        "data_snapshot_cutoff": (
            eligible[-1].target_date.isoformat() if eligible else None
        ),
        "feature_view": "as_issued",
        "target_revision_view": "current",
        "restatements_ignored": restatements_ignored,
    }


def _source_revision_ids(
    dependencies: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> tuple[str, ...]:
    values: set[str] = set()
    for dependency in dependencies.values():
        if not isinstance(dependency, dict):
            raise RetrainingEvaluationError("Observation dependency is invalid.")
        revisions = dependency.get("source_revisions")
        if not isinstance(revisions, list):
            raise RetrainingEvaluationError(
                "Observation dependency source revisions are invalid."
            )
        for revision in revisions:
            value = revision.get("revision_id") if isinstance(revision, dict) else None
            if not isinstance(value, str) or not value:
                raise RetrainingEvaluationError(
                    "Observation source revision ID is invalid."
                )
            values.add(value)
    target_revision = actual.get("source_revision_id")
    if not isinstance(target_revision, str) or not target_revision:
        raise RetrainingEvaluationError("Target source revision ID is invalid.")
    values.add(target_revision)
    return tuple(sorted(values))


def _seal_period(period_root: Path, record: Mapping[str, Any]) -> None:
    evaluation_id = str(record["evaluation_id"])
    if period_root.exists():
        _validate_sealed_period(period_root, evaluation_id)
        return
    period_root.parent.mkdir(parents=True, exist_ok=True)
    prepared = period_root.parent / (
        f".{period_root.name}.{evaluation_id}.{uuid4().hex}.tmp"
    )
    target = prepared / evaluation_id / "evaluation.json"
    data = _json_bytes(record)
    prepared.mkdir(exist_ok=False)
    try:
        target.parent.mkdir()
        with target.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _publish_prepared_period(prepared, period_root)
        except OSError as exc:
            if not period_root.is_dir():
                raise RetrainingEvaluationError(
                    "Atomic evaluation-period publication failed."
                ) from exc
            _validate_sealed_period(period_root, evaluation_id)
    finally:
        _cleanup_prepared_period(prepared, evaluation_id)
    _validate_sealed_period(period_root, evaluation_id)


def _publish_prepared_period(prepared: Path, period_root: Path) -> None:
    """Atomically publish one fully prepared period directory."""
    prepared.rename(period_root)


def _validate_sealed_period(period_root: Path, evaluation_id: str) -> None:
    try:
        entries = list(period_root.iterdir())
    except OSError as exc:
        raise RetrainingEvaluationError(
            "Sealed evaluation period cannot be inspected."
        ) from exc
    if len(entries) != 1 or not entries[0].is_dir():
        raise RetrainingEvaluationError(
            "Sealed evaluation period must contain exactly one record."
        )
    sealed_id = entries[0].name
    loaded = load_monthly_retraining_evaluation(
        entries[0] / "evaluation.json"
    )
    if sealed_id != loaded["evaluation_id"]:
        raise RetrainingEvaluationError(
            "Sealed evaluation directory differs from its content identity."
        )
    if sealed_id != evaluation_id:
        raise RetrainingEvaluationError(
            "Conflicting evidence already seals this evaluation period."
        )


def _cleanup_prepared_period(prepared: Path, evaluation_id: str) -> None:
    """Remove only this call's unpublished, fully known temporary structure."""
    if not prepared.exists():
        return
    record_dir = prepared / evaluation_id
    record_path = record_dir / "evaluation.json"
    try:
        if record_path.is_file():
            record_path.unlink()
        if record_dir.is_dir():
            record_dir.rmdir()
        prepared.rmdir()
    except OSError as exc:
        if prepared.exists():
            raise RetrainingEvaluationError(
                "Temporary evaluation publication could not be cleaned."
            ) from exc


def _with_id(body: Mapping[str, Any]) -> dict[str, Any]:
    ready = json.loads(_canonical(body))
    return {"evaluation_id": _record_id(ready), **ready}


def _record_id(body: Mapping[str, Any]) -> str:
    return sha256(b"monthly_retraining_evaluation:" + _canonical(body)).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _parse_date(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise RetrainingEvaluationError(
            f"{name} must be an ISO-8601 calendar date."
        ) from exc


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "EVALUATION_OUTCOMES",
    "EVALUATION_SCHEMA",
    "EVALUATION_SCHEMA_V2",
    "MonthlyRetrainingEvaluationConfig",
    "MonthlyRetrainingEvaluationPlan",
    "MonthlyRetrainingEvaluationResult",
    "RetrainingEvaluationError",
    "load_monthly_retraining_evaluation",
    "plan_monthly_retraining_evaluation",
    "run_monthly_retraining_evaluation",
]
