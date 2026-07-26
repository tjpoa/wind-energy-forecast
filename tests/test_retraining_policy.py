from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import pytest

from wind_forecast.retraining_policy import (
    ActiveDeploymentPointer,
    EligibilitySelection,
    ObservationEvidence,
    RetrainingContractError,
    RetrainingPolicy,
    TemporalCutoffs,
    build_observation_folds,
    select_eligible_observations,
)


POLICY_PATH = Path("config/retraining_policy_v1.json")
SHA = "a" * 64


def _observation(
    index: int, *, day: date | None = None, **changes
) -> ObservationEvidence:
    values = {
        "observation_id": f"observation-{index:03d}",
        "target_date": day or date(2026, 1, 1) + timedelta(days=index),
        "feature_snapshot_id": f"feature-{index:03d}",
        "target_revision_id": f"target-{index:03d}",
        "feature_schema_sha256": SHA,
        "lineage_sha256": "b" * 64,
        "target_contract_id": "target-v1",
        "transformation_version": "features-v1",
        "source_revision_ids": (f"source-{index:03d}",),
        "feature_values": (float(index), float(index + 1)),
        "target_value": float(index + 100),
    }
    values.update(changes)
    return ObservationEvidence(**values)


def _selection(
    observations: list[ObservationEvidence],
) -> EligibilitySelection:
    return select_eligible_observations(
        observations,
        expected_target_contract_id="target-v1",
        expected_transformation_version="features-v1",
        expected_feature_schema_sha256=SHA,
    )


def test_repository_policy_round_trips_and_disables_automation() -> None:
    policy = RetrainingPolicy.load(POLICY_PATH)

    assert policy.to_dict() == json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert policy.minimum_new_eligible_observations == 90
    assert policy.phase9_persistence_distinct_reports == 3
    assert not policy.automatic_training
    assert not policy.automatic_promotion
    assert not policy.automatic_stability


def test_policy_rejects_weaker_phase9_persistence(tmp_path: Path) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["phase9_alerts"]["persistence_distinct_reports"] = 2
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrainingContractError, match="three reports"):
        RetrainingPolicy.load(path)


def test_policy_rejects_automatic_promotion(tmp_path: Path) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["automation"]["automatic_promotion"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrainingContractError, match="must never be automatic"):
        RetrainingPolicy.load(path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evaluation", "day_of_month", 8.9),
        ("evaluation", "hour_local", "13"),
        ("evaluation", "minimum_new_eligible_observations", True),
        ("backtest", "baseline_feature", None),
    ],
)
def test_policy_rejects_implicit_type_coercion(
    tmp_path: Path, section: str, field: str, value
) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload[section][field] = value
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrainingContractError):
        RetrainingPolicy.load(path)


def test_temporal_cutoffs_remain_distinct_and_ordered() -> None:
    cutoffs = TemporalCutoffs(
        incumbent_fit_cutoff="2025-12-31",
        monitoring_evaluation_cutoff="2026-04-30",
        data_snapshot_cutoff="2026-04-28",
        candidate_fit_cutoff="2026-04-28",
        promotion_effective_date="2026-05-10",
        observation_cutoff="2026-08-31",
    )

    assert cutoffs.to_dict()["monitoring_evaluation_cutoff"] == "2026-04-30"
    assert cutoffs.to_dict()["data_snapshot_cutoff"] == "2026-04-28"

    with pytest.raises(RetrainingContractError, match="cannot follow"):
        TemporalCutoffs(
            incumbent_fit_cutoff="2025-12-31",
            monitoring_evaluation_cutoff="2026-04-30",
            data_snapshot_cutoff="2026-05-01",
        )
    with pytest.raises(RetrainingContractError, match="evaluation_cutoff"):
        TemporalCutoffs(
            incumbent_fit_cutoff="2025-12-31",
            monitoring_evaluation_cutoff="2026-04-30",
            data_snapshot_cutoff="2026-04-28",
            candidate_fit_cutoff="2026-04-28",
            promotion_effective_date="2026-04-29",
        )


def test_eligibility_reports_quality_and_contract_exclusions() -> None:
    observations = [
        _observation(1),
        _observation(2, quality_exclusions=("source_late",)),
        _observation(3, transformation_version="features-v2"),
        _observation(4, feature_values=(1.0, float("nan"))),
        _observation(5, feature_values=(1.0, "2.0")),
    ]

    selection = select_eligible_observations(
        observations,
        expected_target_contract_id="target-v1",
        expected_transformation_version="features-v1",
        expected_feature_schema_sha256=SHA,
    )

    assert [item.observation_id for item in selection.eligible] == [
        "observation-001"
    ]
    assert selection.exclusions == {
        "observation-002": ("source_late",),
        "observation-003": ("transformation_version_mismatch",),
        "observation-004": ("non_finite_feature",),
        "observation-005": ("non_finite_feature",),
    }


def test_eligibility_rejects_duplicate_dates_instead_of_cleaning() -> None:
    shared_day = date(2026, 2, 1)
    observations = [
        _observation(1, day=shared_day),
        _observation(2, day=shared_day),
    ]

    with pytest.raises(RetrainingContractError, match="dates must be unique"):
        select_eligible_observations(
            observations,
            expected_target_contract_id="target-v1",
            expected_transformation_version="features-v1",
            expected_feature_schema_sha256=SHA,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_id", None),
        ("feature_snapshot_id", 123),
        ("target_revision_id", True),
        ("target_contract_id", None),
        ("transformation_version", 7),
    ],
)
def test_observation_evidence_rejects_non_string_identifiers(
    field: str, value
) -> None:
    with pytest.raises(RetrainingContractError, match="JSON string"):
        _observation(1, **{field: value})


@pytest.mark.parametrize(
    "source_revision_ids",
    ["revision-1", (None,), (True,)],
)
def test_observation_evidence_rejects_invalid_source_revision_arrays(
    source_revision_ids,
) -> None:
    with pytest.raises(RetrainingContractError):
        _observation(1, source_revision_ids=source_revision_ids)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"target_contract_id": None}, "JSON string"),
        ({"transformation_version": 4}, "JSON string"),
        ({"feature_schema_sha256": None}, "JSON string"),
        ({"exclusions": {None: ("source_late",)}}, "JSON string"),
        ({"exclusions": {"observation-1": "source_late"}}, "array"),
        ({"exclusions": {"observation-1": ()}}, "non-empty array"),
    ],
)
def test_eligibility_selection_rejects_invalid_contracts_and_exclusions(
    changes, match: str
) -> None:
    values = {
        "eligible": (),
        "exclusions": {},
        "target_contract_id": "target-v1",
        "transformation_version": "features-v1",
        "feature_schema_sha256": SHA,
    }
    values.update(changes)

    with pytest.raises(RetrainingContractError, match=match):
        EligibilitySelection(**values)


def test_folds_use_observation_count_and_record_calendar_gaps() -> None:
    current = date(2026, 1, 1)
    observations = []
    for index in range(95):
        if index in {10, 40, 70}:
            current += timedelta(days=1)
        observations.append(_observation(index, day=current))
        current += timedelta(days=1)

    plan = build_observation_folds(
        _selection(observations),
        incumbent_fit_cutoff="2025-12-31",
    )

    assert len(plan.folds) == 3
    assert all(len(fold.observation_ids) == 30 for fold in plan.folds)
    assert plan.folds[0].calendar_gap_dates == ("2026-01-11",)
    assert plan.folds[1].calendar_gap_dates == ("2026-02-11",)
    assert plan.folds[2].calendar_gap_dates == ("2026-03-14",)
    assert len(plan.trailing_observation_ids) == 5
    assert plan.folds[1].fold_train_cutoff == plan.folds[0].fold_evaluation_end


def test_folds_require_three_complete_observation_blocks() -> None:
    observations = [_observation(index) for index in range(89)]

    with pytest.raises(RetrainingContractError, match="Insufficient"):
        build_observation_folds(
            _selection(observations),
            incumbent_fit_cutoff="2025-12-31",
        )


def test_fold_selection_rejects_bypassed_quality_exclusions() -> None:
    observations = [_observation(index) for index in range(90)]
    observations[45] = _observation(
        45,
        quality_exclusions=("source_late",),
    )

    with pytest.raises(RetrainingContractError, match="incompatible or excluded"):
        EligibilitySelection(
            eligible=tuple(observations),
            exclusions={},
            target_contract_id="target-v1",
            transformation_version="features-v1",
            feature_schema_sha256=SHA,
        )


def test_active_pointer_has_exact_schema_and_utc_timestamp() -> None:
    payload = {
        "schema_version": "wind_forecast.active_deployment_pointer.v1",
        "generation": 1,
        "deployment_id": "deployment-1",
        "deployment_state_id": "state-1",
        "state_manifest_path": "outputs/deployments/state-1.json",
        "state_manifest_sha256": SHA,
        "updated_at_utc": "2026-07-26T12:00:00Z",
    }

    pointer = ActiveDeploymentPointer.from_dict(payload)

    assert pointer.to_dict() == payload
    with pytest.raises(RetrainingContractError, match="fields differ"):
        ActiveDeploymentPointer.from_dict({**payload, "unexpected": True})
    with pytest.raises(RetrainingContractError, match="timezone-aware UTC"):
        ActiveDeploymentPointer.from_dict(
            {**payload, "updated_at_utc": "2026-07-26T12:00:00"}
        )
    with pytest.raises(RetrainingContractError, match="invalid value"):
        ActiveDeploymentPointer.from_dict({**payload, "generation": "not-an-int"})
    with pytest.raises(RetrainingContractError, match="invalid value"):
        ActiveDeploymentPointer.from_dict({**payload, "generation": True})
    with pytest.raises(RetrainingContractError, match="invalid value"):
        ActiveDeploymentPointer.from_dict({**payload, "state_manifest_path": None})
