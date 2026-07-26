"""Side-effect-free contracts for controlled v2 retraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import re
from numbers import Real
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


POLICY_SCHEMA = "wind_forecast.retraining_policy.v1"
OBSERVATION_SCHEMA = "wind_forecast.retraining_observation.v1"
POINTER_SCHEMA = "wind_forecast.active_deployment_pointer.v1"
DEPLOYMENT_STATE_SCHEMA = "wind_forecast.deployment_state.v1"

TRIGGER_CATEGORIES = (
    "feature_drift",
    "prediction_drift",
    "target_drift",
    "performance",
)
TRIGGER_SEVERITIES = ("warning", "critical")
BLOCKING_CATEGORIES = ("quality",)
ALLOWED_ISSUANCE_KINDS = ("scheduled", "catch_up")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RetrainingContractError(ValueError):
    """Raised when controlled-retraining evidence violates its contract."""


@dataclass(frozen=True)
class RetrainingPolicy:
    """Validated, versioned controlled-retraining policy."""

    evaluation_day_of_month: int
    evaluation_hour_local: int
    evaluation_timezone: str
    minimum_new_eligible_observations: int
    phase9_persistence_distinct_reports: int
    trigger_categories: tuple[str, ...]
    trigger_severities: tuple[str, ...]
    blocking_categories: tuple[str, ...]
    fold_observation_count: int
    minimum_complete_folds: int
    baseline_feature: str
    performance_limit_contract: str
    require_aggregate_mae_strictly_better: bool
    require_each_fold_mae_not_worse: bool
    require_no_performance_breach: bool
    stability_minimum_eligible_observations: int
    stability_allowed_issuance_kinds: tuple[str, ...]
    require_no_active_warning_or_critical: bool
    require_second_manual_approval: bool
    automatic_training: bool
    automatic_promotion: bool
    automatic_stability: bool
    schema_version: str = POLICY_SCHEMA

    @classmethod
    def load(cls, path: str | Path) -> "RetrainingPolicy":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrainingContractError(
                f"Retraining policy is missing or invalid: {path}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != POLICY_SCHEMA:
            raise RetrainingContractError("Unsupported retraining policy schema.")
        try:
            evaluation = _strict_mapping(payload["evaluation"], "evaluation")
            alerts = _strict_mapping(payload["phase9_alerts"], "phase9_alerts")
            backtest = _strict_mapping(payload["backtest"], "backtest")
            stability = _strict_mapping(payload["stability"], "stability")
            automation = _strict_mapping(payload["automation"], "automation")
            policy = cls(
                evaluation_day_of_month=_strict_int(
                    evaluation["day_of_month"], "day_of_month"
                ),
                evaluation_hour_local=_strict_int(
                    evaluation["hour_local"], "hour_local"
                ),
                evaluation_timezone=_strict_text(
                    evaluation["timezone"], "timezone"
                ),
                minimum_new_eligible_observations=_strict_int(
                    evaluation["minimum_new_eligible_observations"],
                    "minimum_new_eligible_observations",
                ),
                phase9_persistence_distinct_reports=_strict_int(
                    alerts["persistence_distinct_reports"],
                    "persistence_distinct_reports",
                ),
                trigger_categories=_strict_text_tuple(
                    alerts["trigger_categories"], "trigger_categories"
                ),
                trigger_severities=_strict_text_tuple(
                    alerts["trigger_severities"], "trigger_severities"
                ),
                blocking_categories=_strict_text_tuple(
                    alerts["blocking_categories"], "blocking_categories"
                ),
                fold_observation_count=_strict_int(
                    backtest["fold_observation_count"], "fold_observation_count"
                ),
                minimum_complete_folds=_strict_int(
                    backtest["minimum_complete_folds"], "minimum_complete_folds"
                ),
                baseline_feature=_strict_text(
                    backtest["baseline_feature"], "baseline_feature"
                ),
                performance_limit_contract=_strict_text(
                    backtest["performance_limit_contract"],
                    "performance_limit_contract",
                ),
                require_aggregate_mae_strictly_better=_strict_bool(
                    backtest["require_aggregate_mae_strictly_better"],
                    "require_aggregate_mae_strictly_better",
                ),
                require_each_fold_mae_not_worse=_strict_bool(
                    backtest["require_each_fold_mae_not_worse"],
                    "require_each_fold_mae_not_worse",
                ),
                require_no_performance_breach=_strict_bool(
                    backtest["require_no_performance_breach"],
                    "require_no_performance_breach",
                ),
                stability_minimum_eligible_observations=_strict_int(
                    stability["minimum_eligible_observations"],
                    "minimum_eligible_observations",
                ),
                stability_allowed_issuance_kinds=_strict_text_tuple(
                    stability["allowed_issuance_kinds"],
                    "allowed_issuance_kinds",
                ),
                require_no_active_warning_or_critical=_strict_bool(
                    stability["require_no_active_warning_or_critical"],
                    "require_no_active_warning_or_critical",
                ),
                require_second_manual_approval=_strict_bool(
                    stability["require_second_manual_approval"],
                    "require_second_manual_approval",
                ),
                automatic_training=_strict_bool(
                    automation["automatic_training"], "automatic_training"
                ),
                automatic_promotion=_strict_bool(
                    automation["automatic_promotion"], "automatic_promotion"
                ),
                automatic_stability=_strict_bool(
                    automation["automatic_stability"], "automatic_stability"
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RetrainingContractError):
                raise
            raise RetrainingContractError(
                "Retraining policy is missing a required field or value."
            ) from exc
        policy.validate()
        return policy

    def validate(self) -> None:
        if not 1 <= self.evaluation_day_of_month <= 28:
            raise RetrainingContractError(
                "Evaluation day must be between 1 and 28 for every month."
            )
        if not 0 <= self.evaluation_hour_local <= 23:
            raise RetrainingContractError(
                "Evaluation hour must be between 0 and 23."
            )
        try:
            ZoneInfo(self.evaluation_timezone)
        except ZoneInfoNotFoundError as exc:
            raise RetrainingContractError(
                "Evaluation timezone must be an installed IANA timezone."
            ) from exc
        if self.minimum_new_eligible_observations < 1:
            raise RetrainingContractError(
                "Minimum new eligible observations must be positive."
            )
        if self.phase9_persistence_distinct_reports != 3:
            raise RetrainingContractError(
                "The v1 policy must reuse Phase 9 persistence of three reports."
            )
        if self.trigger_categories != TRIGGER_CATEGORIES:
            raise RetrainingContractError(
                "Trigger categories must match the approved Phase 9 categories."
            )
        if self.trigger_severities != TRIGGER_SEVERITIES:
            raise RetrainingContractError(
                "Trigger severities must be warning and critical."
            )
        if self.blocking_categories != BLOCKING_CATEGORIES:
            raise RetrainingContractError(
                "Quality alerts must block rather than trigger retraining."
            )
        if self.fold_observation_count != 30 or self.minimum_complete_folds != 3:
            raise RetrainingContractError(
                "Backtesting requires at least three complete folds of 30 observations."
            )
        required = self.fold_observation_count * self.minimum_complete_folds
        if self.minimum_new_eligible_observations < required:
            raise RetrainingContractError(
                "The data minimum must cover all required complete folds."
            )
        if self.baseline_feature != "Wind_Production_Lag1":
            raise RetrainingContractError(
                "The approved baseline is Wind_Production_Lag1 persistence."
            )
        if self.performance_limit_contract != "incumbent_phase9_performance_30":
            raise RetrainingContractError(
                "Backtest folds must reuse the incumbent Phase 9 performance.30 limits."
            )
        if not all(
            (
                self.require_aggregate_mae_strictly_better,
                self.require_each_fold_mae_not_worse,
                self.require_no_performance_breach,
            )
        ):
            raise RetrainingContractError(
                "All conservative v1 backtest gates must remain enabled."
            )
        if self.stability_minimum_eligible_observations != 90:
            raise RetrainingContractError(
                "Stability requires exactly 90 eligible probation observations."
            )
        if self.stability_allowed_issuance_kinds != ALLOWED_ISSUANCE_KINDS:
            raise RetrainingContractError(
                "Only scheduled and catch_up observations count toward stability."
            )
        if not (
            self.require_no_active_warning_or_critical
            and self.require_second_manual_approval
        ):
            raise RetrainingContractError(
                "Stability requires healthy alerts and second manual approval."
            )
        if any(
            (
                self.automatic_training,
                self.automatic_promotion,
                self.automatic_stability,
            )
        ):
            raise RetrainingContractError(
                "Training, promotion, and stability must never be automatic."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation": {
                "day_of_month": self.evaluation_day_of_month,
                "hour_local": self.evaluation_hour_local,
                "minimum_new_eligible_observations": (
                    self.minimum_new_eligible_observations
                ),
                "timezone": self.evaluation_timezone,
            },
            "phase9_alerts": {
                "blocking_categories": list(self.blocking_categories),
                "persistence_distinct_reports": (
                    self.phase9_persistence_distinct_reports
                ),
                "trigger_categories": list(self.trigger_categories),
                "trigger_severities": list(self.trigger_severities),
            },
            "backtest": {
                "baseline_feature": self.baseline_feature,
                "fold_observation_count": self.fold_observation_count,
                "minimum_complete_folds": self.minimum_complete_folds,
                "performance_limit_contract": self.performance_limit_contract,
                "require_aggregate_mae_strictly_better": (
                    self.require_aggregate_mae_strictly_better
                ),
                "require_each_fold_mae_not_worse": (
                    self.require_each_fold_mae_not_worse
                ),
                "require_no_performance_breach": (
                    self.require_no_performance_breach
                ),
            },
            "stability": {
                "allowed_issuance_kinds": list(
                    self.stability_allowed_issuance_kinds
                ),
                "minimum_eligible_observations": (
                    self.stability_minimum_eligible_observations
                ),
                "require_no_active_warning_or_critical": (
                    self.require_no_active_warning_or_critical
                ),
                "require_second_manual_approval": (
                    self.require_second_manual_approval
                ),
            },
            "automation": {
                "automatic_promotion": self.automatic_promotion,
                "automatic_stability": self.automatic_stability,
                "automatic_training": self.automatic_training,
            },
        }


@dataclass(frozen=True)
class TemporalCutoffs:
    """Distinct lifecycle cutoffs; optional values belong to later transitions."""

    incumbent_fit_cutoff: date
    monitoring_evaluation_cutoff: date
    data_snapshot_cutoff: date
    candidate_fit_cutoff: date | None = None
    promotion_effective_date: date | None = None
    observation_cutoff: date | None = None

    def __post_init__(self) -> None:
        for field_name in asdict(self):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, date):
                object.__setattr__(
                    self, field_name, _parse_date(value, field_name)
                )
        self.validate()

    def validate(self) -> None:
        if self.incumbent_fit_cutoff >= self.data_snapshot_cutoff:
            raise RetrainingContractError(
                "data_snapshot_cutoff must follow incumbent_fit_cutoff."
            )
        if self.data_snapshot_cutoff > self.monitoring_evaluation_cutoff:
            raise RetrainingContractError(
                "data_snapshot_cutoff cannot follow monitoring_evaluation_cutoff."
            )
        if self.candidate_fit_cutoff is not None and not (
            self.incumbent_fit_cutoff
            < self.candidate_fit_cutoff
            <= self.data_snapshot_cutoff
        ):
            raise RetrainingContractError(
                "candidate_fit_cutoff must follow incumbent fit and not exceed the snapshot."
            )
        if self.promotion_effective_date is not None:
            if self.candidate_fit_cutoff is None:
                raise RetrainingContractError(
                    "promotion_effective_date requires candidate_fit_cutoff."
                )
            if self.promotion_effective_date <= self.monitoring_evaluation_cutoff:
                raise RetrainingContractError(
                    "promotion_effective_date must follow monitoring_evaluation_cutoff."
                )
        if self.observation_cutoff is not None:
            if self.promotion_effective_date is None:
                raise RetrainingContractError(
                    "observation_cutoff requires promotion_effective_date."
                )
            if self.observation_cutoff < self.promotion_effective_date:
                raise RetrainingContractError(
                    "observation_cutoff cannot precede promotion_effective_date."
                )

    def to_dict(self) -> dict[str, str | None]:
        return {
            name: value.isoformat() if value is not None else None
            for name, value in asdict(self).items()
        }


@dataclass(frozen=True)
class ObservationEvidence:
    """Pinned evidence for one possible retraining observation."""

    observation_id: str
    target_date: date
    feature_snapshot_id: str
    target_revision_id: str
    feature_schema_sha256: str
    lineage_sha256: str
    target_contract_id: str
    transformation_version: str
    source_revision_ids: tuple[str, ...]
    feature_values: tuple[float, ...]
    target_value: float
    quality_exclusions: tuple[str, ...] = ()
    schema_version: str = OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.target_date, date):
            object.__setattr__(
                self,
                "target_date",
                _parse_date(self.target_date, "target_date"),
            )
        source_revision_ids = _strict_text_sequence(
            self.source_revision_ids,
            "source_revision_ids",
            allow_empty=False,
        )
        object.__setattr__(
            self, "source_revision_ids", source_revision_ids
        )
        if not isinstance(self.feature_values, (list, tuple)):
            raise RetrainingContractError(
                "feature_values must be an array."
            )
        object.__setattr__(self, "feature_values", tuple(self.feature_values))
        quality_exclusions = _strict_text_sequence(
            self.quality_exclusions,
            "quality_exclusions",
            allow_empty=True,
        )
        object.__setattr__(
            self, "quality_exclusions", quality_exclusions
        )
        if self.schema_version != OBSERVATION_SCHEMA:
            raise RetrainingContractError("Unsupported observation evidence schema.")
        for name in (
            "observation_id",
            "feature_snapshot_id",
            "target_revision_id",
            "target_contract_id",
            "transformation_version",
        ):
            _strict_text(getattr(self, name), name)
        for name in ("feature_schema_sha256", "lineage_sha256"):
            value = _strict_text(getattr(self, name), name)
            if not SHA256_PATTERN.fullmatch(value):
                raise RetrainingContractError(f"{name} must be a SHA-256 digest.")


@dataclass(frozen=True)
class EligibilitySelection:
    """Eligible evidence plus reasons for every excluded observation."""

    eligible: tuple[ObservationEvidence, ...]
    exclusions: Mapping[str, tuple[str, ...]]
    target_contract_id: str
    transformation_version: str
    feature_schema_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.eligible, (list, tuple)):
            raise RetrainingContractError(
                "eligible must be an observation array."
            )
        object.__setattr__(self, "eligible", tuple(self.eligible))
        _strict_text(self.target_contract_id, "target_contract_id")
        _strict_text(self.transformation_version, "transformation_version")
        feature_schema = _strict_text(
            self.feature_schema_sha256, "feature_schema_sha256"
        )
        if not SHA256_PATTERN.fullmatch(feature_schema):
            raise RetrainingContractError(
                "Eligibility selection feature schema must be a SHA-256 digest."
            )
        if not isinstance(self.exclusions, Mapping):
            raise RetrainingContractError("exclusions must be an object.")
        normalized_exclusions: dict[str, tuple[str, ...]] = {}
        for observation_id, reasons in self.exclusions.items():
            identifier = _strict_text(observation_id, "excluded observation ID")
            normalized_exclusions[identifier] = _strict_text_sequence(
                reasons,
                "exclusion reasons",
                allow_empty=False,
            )
        object.__setattr__(self, "exclusions", normalized_exclusions)
        _reject_duplicate_evidence(self.eligible)
        for observation in self.eligible:
            if (
                observation.target_contract_id != self.target_contract_id
                or observation.transformation_version != self.transformation_version
                or observation.feature_schema_sha256 != self.feature_schema_sha256
                or observation.quality_exclusions
                or not observation.feature_values
                or not all(
                    _is_finite_number(value)
                    for value in observation.feature_values
                )
                or not _is_finite_number(observation.target_value)
            ):
                raise RetrainingContractError(
                    "Eligibility selection contains incompatible or excluded evidence."
                )
        overlap = {
            observation.observation_id for observation in self.eligible
        }.intersection(self.exclusions)
        if overlap:
            raise RetrainingContractError(
                "Eligible and excluded observation IDs must be disjoint."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_contract_id": self.target_contract_id,
            "transformation_version": self.transformation_version,
            "feature_schema_sha256": self.feature_schema_sha256,
            "eligible_observation_ids": [
                observation.observation_id for observation in self.eligible
            ],
            "excluded": {
                key: list(value) for key, value in sorted(self.exclusions.items())
            },
        }


def select_eligible_observations(
    observations: Sequence[ObservationEvidence],
    *,
    expected_target_contract_id: str,
    expected_transformation_version: str,
    expected_feature_schema_sha256: str,
) -> EligibilitySelection:
    """Select compatible finite observations without cleaning or coercion."""
    if not SHA256_PATTERN.fullmatch(expected_feature_schema_sha256):
        raise RetrainingContractError(
            "Expected feature schema must be a SHA-256 digest."
        )
    ordered = tuple(sorted(observations, key=lambda item: item.target_date))
    _reject_duplicate_evidence(ordered)
    eligible: list[ObservationEvidence] = []
    exclusions: dict[str, tuple[str, ...]] = {}
    for observation in ordered:
        issues = list(observation.quality_exclusions)
        if observation.target_contract_id != expected_target_contract_id:
            issues.append("target_contract_mismatch")
        if observation.transformation_version != expected_transformation_version:
            issues.append("transformation_version_mismatch")
        if observation.feature_schema_sha256 != expected_feature_schema_sha256:
            issues.append("feature_schema_mismatch")
        if not observation.feature_values:
            issues.append("empty_feature_vector")
        elif not all(_is_finite_number(value) for value in observation.feature_values):
            issues.append("non_finite_feature")
        if not _is_finite_number(observation.target_value):
            issues.append("non_finite_target")
        if issues:
            exclusions[observation.observation_id] = tuple(sorted(set(issues)))
        else:
            eligible.append(observation)
    return EligibilitySelection(
        eligible=tuple(eligible),
        exclusions=exclusions,
        target_contract_id=expected_target_contract_id,
        transformation_version=expected_transformation_version,
        feature_schema_sha256=expected_feature_schema_sha256,
    )


@dataclass(frozen=True)
class ObservationFold:
    """One non-overlapping fold defined by observation positions."""

    fold_index: int
    fold_train_cutoff: date
    fold_evaluation_start: date
    fold_evaluation_end: date
    observation_ids: tuple[str, ...]
    calendar_gap_dates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "fold_train_cutoff": self.fold_train_cutoff.isoformat(),
            "fold_evaluation_start": self.fold_evaluation_start.isoformat(),
            "fold_evaluation_end": self.fold_evaluation_end.isoformat(),
            "observation_ids": list(self.observation_ids),
            "calendar_gap_dates": list(self.calendar_gap_dates),
        }


@dataclass(frozen=True)
class ObservationFoldPlan:
    """Complete folds and the deliberately unused incomplete tail."""

    folds: tuple[ObservationFold, ...]
    trailing_observation_ids: tuple[str, ...]
    fold_observation_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_observation_count": self.fold_observation_count,
            "folds": [fold.to_dict() for fold in self.folds],
            "trailing_observation_ids": list(self.trailing_observation_ids),
        }


def build_observation_folds(
    selection: EligibilitySelection,
    *,
    incumbent_fit_cutoff: date | str,
    fold_observation_count: int = 30,
    minimum_complete_folds: int = 3,
) -> ObservationFoldPlan:
    """Build complete folds from observations, never calendar windows."""
    cutoff = (
        incumbent_fit_cutoff
        if isinstance(incumbent_fit_cutoff, date)
        else _parse_date(incumbent_fit_cutoff, "incumbent_fit_cutoff")
    )
    if fold_observation_count < 1 or minimum_complete_folds < 1:
        raise RetrainingContractError(
            "Fold size and minimum complete folds must be positive."
        )
    ordered = tuple(
        sorted(
            (item for item in selection.eligible if item.target_date > cutoff),
            key=lambda item: item.target_date,
        )
    )
    _reject_duplicate_evidence(ordered)
    complete_count = len(ordered) // fold_observation_count
    if complete_count < minimum_complete_folds:
        raise RetrainingContractError(
            "Insufficient eligible observations for the required complete folds."
        )
    folds: list[ObservationFold] = []
    train_cutoff = cutoff
    for index in range(complete_count):
        start = index * fold_observation_count
        block = ordered[start : start + fold_observation_count]
        folds.append(
            ObservationFold(
                fold_index=index + 1,
                fold_train_cutoff=train_cutoff,
                fold_evaluation_start=block[0].target_date,
                fold_evaluation_end=block[-1].target_date,
                observation_ids=tuple(item.observation_id for item in block),
                calendar_gap_dates=_calendar_gaps(
                    block[0].target_date,
                    block[-1].target_date,
                    {item.target_date for item in block},
                ),
            )
        )
        train_cutoff = block[-1].target_date
    tail_start = complete_count * fold_observation_count
    return ObservationFoldPlan(
        folds=tuple(folds),
        trailing_observation_ids=tuple(
            item.observation_id for item in ordered[tail_start:]
        ),
        fold_observation_count=fold_observation_count,
    )


@dataclass(frozen=True)
class ActiveDeploymentPointer:
    """Exact mutable pointer shape; referenced state remains immutable."""

    generation: int
    deployment_id: str
    deployment_state_id: str
    state_manifest_path: str
    state_manifest_sha256: str
    updated_at_utc: str
    schema_version: str = POINTER_SCHEMA

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveDeploymentPointer":
        if payload.get("schema_version") != POINTER_SCHEMA:
            raise RetrainingContractError("Unsupported active deployment pointer schema.")
        required = {
            "generation",
            "deployment_id",
            "deployment_state_id",
            "state_manifest_path",
            "state_manifest_sha256",
            "updated_at_utc",
        }
        if set(payload) != required | {"schema_version"}:
            raise RetrainingContractError(
                "Active deployment pointer fields differ from the v1 contract."
            )
        try:
            pointer = cls(
                generation=_strict_int(payload["generation"], "generation"),
                deployment_id=_strict_text(
                    payload["deployment_id"], "deployment_id"
                ),
                deployment_state_id=_strict_text(
                    payload["deployment_state_id"], "deployment_state_id"
                ),
                state_manifest_path=_strict_text(
                    payload["state_manifest_path"], "state_manifest_path"
                ),
                state_manifest_sha256=_strict_text(
                    payload["state_manifest_sha256"], "state_manifest_sha256"
                ),
                updated_at_utc=_strict_text(
                    payload["updated_at_utc"], "updated_at_utc"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise RetrainingContractError(
                "Active deployment pointer contains an invalid value."
            ) from exc
        pointer.validate()
        return pointer

    def validate(self) -> None:
        if self.generation < 1:
            raise RetrainingContractError(
                "Deployment pointer generation must be positive."
            )
        for name in (
            "deployment_id",
            "deployment_state_id",
            "state_manifest_path",
        ):
            if not str(getattr(self, name)).strip():
                raise RetrainingContractError(f"{name} must be non-empty.")
        if not SHA256_PATTERN.fullmatch(self.state_manifest_sha256):
            raise RetrainingContractError(
                "state_manifest_sha256 must be a SHA-256 digest."
            )
        _parse_utc(self.updated_at_utc, "updated_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "deployment_id": self.deployment_id,
            "deployment_state_id": self.deployment_state_id,
            "state_manifest_path": self.state_manifest_path,
            "state_manifest_sha256": self.state_manifest_sha256,
            "updated_at_utc": self.updated_at_utc,
        }


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RetrainingContractError(f"{name} must be a JSON boolean.")
    return value


def _strict_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RetrainingContractError(f"{name} must be a JSON integer.")
    return value


def _strict_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrainingContractError(f"{name} must be a non-empty JSON string.")
    return value


def _strict_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RetrainingContractError(
            f"{name} must be a non-empty JSON string array."
        )
    return tuple(_strict_text(item, name) for item in value)


def _strict_text_sequence(
    value: Any,
    name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise RetrainingContractError(f"{name} must be {qualifier} of strings.")
    return tuple(_strict_text(item, name) for item in value)


def _strict_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RetrainingContractError(f"{name} must be a JSON object.")
    return value


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _parse_date(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise RetrainingContractError(
            f"{name} must be an ISO-8601 calendar date."
        ) from exc


def _reject_duplicate_evidence(
    observations: Sequence[ObservationEvidence],
) -> None:
    identifiers = [item.observation_id for item in observations]
    dates = [item.target_date for item in observations]
    if len(identifiers) != len(set(identifiers)):
        raise RetrainingContractError("Observation IDs must be unique.")
    if len(dates) != len(set(dates)):
        raise RetrainingContractError("Eligible target dates must be unique.")


def _calendar_gaps(
    start: date, end: date, observed: set[date]
) -> tuple[str, ...]:
    values: list[str] = []
    current = start
    while current <= end:
        if current not in observed:
            values.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(values)


def _parse_utc(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RetrainingContractError(
            f"{name} must be an ISO-8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RetrainingContractError(f"{name} must be timezone-aware UTC.")
    return parsed


__all__ = [
    "ActiveDeploymentPointer",
    "DEPLOYMENT_STATE_SCHEMA",
    "EligibilitySelection",
    "OBSERVATION_SCHEMA",
    "ObservationEvidence",
    "ObservationFold",
    "ObservationFoldPlan",
    "POINTER_SCHEMA",
    "POLICY_SCHEMA",
    "RetrainingContractError",
    "RetrainingPolicy",
    "TemporalCutoffs",
    "build_observation_folds",
    "select_eligible_observations",
]
