"""Fail-closed temporal backtesting for manually approved v2 retraining."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from importlib import metadata
from io import BytesIO
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from wind_forecast.manifests import sha256_file
from wind_forecast.monitoring import (
    MonitoringError,
    load_prediction_evidence,
    load_verified_monitoring_state,
    validate_monitoring_model_bundle,
)
from wind_forecast.monitoring_reporting import (
    MonitoringReportingError,
    load_monitoring_calibration,
    load_monitoring_report,
    load_monitoring_report_state,
)
from wind_forecast.monitoring_statistics import regression_metrics, threshold_severity
from wind_forecast.retraining_evaluation import (
    RetrainingEvaluationError,
    load_monthly_retraining_evaluation,
)
from wind_forecast.retraining_policy import (
    EligibilitySelection,
    ObservationEvidence,
    RetrainingContractError,
    RetrainingPolicy,
    build_observation_folds,
)
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN
from wind_forecast.tracking import git_state
from wind_forecast.v2_training import PERSISTENCE_COLUMN


BACKTEST_SCHEMA = "wind_forecast.retraining_backtest.v1"
BACKTEST_BUNDLE_SCHEMA = "wind_forecast.retraining_backtest_bundle.v1"
BACKTEST_OUTCOMES = ("accepted", "rejected")
MODEL_TYPES = {
    "extra_trees": ExtraTreesRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "random_forest": RandomForestRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}
REQUIRED_PERFORMANCE_METRICS = (
    "MAE",
    "RMSE",
    "MAPE_percent",
    "R2",
    "absolute_bias",
)


class RetrainingBacktestError(RuntimeError):
    """Raised when backtest evidence is incompatible, corrupt, or conflicting."""


@dataclass(frozen=True)
class RetrainingBacktestConfig:
    """Explicit inputs for one operator-pinned temporal backtest."""

    evaluation_path: Path
    monitoring_store_root: Path
    incumbent_bundle: Path
    incumbent_base_dataset: Path
    calibration_dir: Path
    policy_path: Path
    output_root: Path
    dry_run: bool = False

    def __post_init__(self) -> None:
        for name in (
            "evaluation_path",
            "monitoring_store_root",
            "incumbent_bundle",
            "incumbent_base_dataset",
            "calibration_dir",
            "policy_path",
            "output_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for input_path in (
            self.evaluation_path,
            self.monitoring_store_root,
            self.incumbent_bundle,
            self.incumbent_base_dataset,
            self.calibration_dir,
            self.policy_path,
        ):
            if _paths_overlap(self.output_root, input_path):
                raise RetrainingBacktestError(
                    "Backtest output root must not overlap input evidence."
                )


@dataclass(frozen=True)
class RetrainingBacktestPlan:
    """Verified in-memory backtest and the record that may be sealed."""

    outcome: str
    evaluation_period: str
    backtest_id: str
    record: Mapping[str, Any]
    final_model: Any | None
    predictions: pd.DataFrame
    training_frame: pd.DataFrame | None
    training_evidence: pd.DataFrame | None
    model_bytes: bytes | None

    def summary(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "evaluation_period": self.evaluation_period,
            "backtest_id": self.backtest_id,
            "record": dict(self.record),
        }


@dataclass(frozen=True)
class RetrainingBacktestResult:
    """Dry-run or immutable temporal-backtest result."""

    status: str
    outcome: str
    backtest_id: str
    backtest_dir: Path | None
    manifest_path: Path | None
    plan: RetrainingBacktestPlan

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "outcome": self.outcome,
            "backtest_id": self.backtest_id,
            "backtest_dir": str(self.backtest_dir) if self.backtest_dir else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "plan": self.plan.summary(),
        }


def evaluate_no_performance_breach(
    metrics: Mapping[str, Any],
    limits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the exact public Phase 9 threshold semantics to one 30-row fold."""
    values = {
        "MAE": metrics.get("MAE"),
        "RMSE": metrics.get("RMSE"),
        "MAPE_percent": metrics.get("MAPE_percent"),
        "R2": metrics.get("R2"),
        "absolute_bias": (
            abs(float(metrics["bias"])) if _finite(metrics.get("bias")) else None
        ),
    }
    if set(limits) != set(REQUIRED_PERFORMANCE_METRICS):
        raise RetrainingBacktestError(
            "Calibration performance.30 metrics differ from the v1 contract."
        )
    severities: dict[str, str] = {}
    for name, value in values.items():
        node = limits[name]
        if not isinstance(node, Mapping) or not {
            "warning",
            "critical",
            "direction",
        }.issubset(node):
            raise RetrainingBacktestError(
                f"Calibration limit for {name} is invalid."
            )
        severities[name] = threshold_severity(value, node)
    breached = sorted(
        name
        for name, severity in severities.items()
        if severity in {"warning", "critical"}
    )
    return {
        "contract": "incumbent_phase9_performance_30",
        "values": values,
        "severity": severities,
        "passed": not breached,
        "breached_metrics": breached,
    }


def plan_retraining_backtest(
    config: RetrainingBacktestConfig,
) -> RetrainingBacktestPlan:
    """Verify all pinned evidence and execute an in-memory temporal backtest."""
    try:
        evaluation = load_monthly_retraining_evaluation(config.evaluation_path)
        policy = RetrainingPolicy.load(config.policy_path)
        bundle = validate_monitoring_model_bundle(config.incumbent_bundle)
        calibration = load_monitoring_calibration(config.calibration_dir)
    except (
        MonitoringError,
        MonitoringReportingError,
        RetrainingContractError,
        RetrainingEvaluationError,
        OSError,
        ValueError,
    ) as exc:
        raise RetrainingBacktestError(str(exc)) from exc
    if evaluation["outcome"] != "eligible_for_manual_backtest":
        raise RetrainingBacktestError(
            "Monthly evaluation is not eligible_for_manual_backtest."
        )
    pinned_ledger = _verify_pinned_inputs(
        config, evaluation, policy, bundle, calibration
    )
    limits, r2_minimum = _validate_calibration_preflight(
        config, calibration
    )
    observations = _reconstruct_observations(
        config, evaluation, bundle, pinned_ledger
    )
    try:
        selection = EligibilitySelection(
            target_contract_id=observations[0].target_contract_id,
            transformation_version=observations[0].transformation_version,
            feature_schema_sha256=observations[0].feature_schema_sha256,
            eligible=tuple(observations),
            exclusions={},
        )
        fold_plan = build_observation_folds(
            selection,
            incumbent_fit_cutoff=evaluation["cutoffs"]["incumbent_fit_cutoff"],
            fold_observation_count=policy.fold_observation_count,
            minimum_complete_folds=policy.minimum_complete_folds,
        )
    except (IndexError, RetrainingContractError) as exc:
        raise RetrainingBacktestError(str(exc)) from exc

    base = _load_base_dataset(config.incumbent_base_dataset, bundle, evaluation)
    model_class, parameters = _incumbent_recipe(bundle)
    incumbent = joblib.load(config.incumbent_bundle / "model.joblib")
    feature_names = list(bundle["feature_names"])
    _validate_loaded_incumbent(
        incumbent, model_class, parameters, feature_names
    )
    by_id = {item.observation_id: item for item in observations}
    fold_records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    prior: pd.DataFrame | None = None
    all_fold_ids: list[str] = []
    for fold in fold_plan.folds:
        test_observations = [by_id[item] for item in fold.observation_ids]
        train = (
            base.copy()
            if prior is None
            else pd.concat([base, prior], ignore_index=True)
        )
        if pd.to_datetime(train[DATE_COLUMN]).max().date() > fold.fold_train_cutoff:
            raise RetrainingBacktestError(
                "Expanding-window training data exceeds fold_train_cutoff."
            )
        candidate = model_class(**parameters).fit(
            train[feature_names], train[TARGET_COLUMN]
        )
        x_test = pd.DataFrame(
            [item.feature_values for item in test_observations],
            columns=feature_names,
        )
        y_test = np.asarray(
            [item.target_value for item in test_observations], dtype=float
        )
        predictions = {
            "candidate": np.asarray(candidate.predict(x_test), dtype=float),
            "incumbent": np.asarray(incumbent.predict(x_test), dtype=float),
            "persistence": x_test[PERSISTENCE_COLUMN].to_numpy(dtype=float),
        }
        metrics = {
            name: regression_metrics(
                y_test,
                values,
                mape_epsilon=float(calibration["mape_epsilon"]),
                r2_minimum_samples=r2_minimum,
            )
            for name, values in predictions.items()
        }
        threshold_gate = evaluate_no_performance_breach(
            metrics["candidate"], limits
        )
        mae_gate = (
            metrics["candidate"]["MAE"] <= metrics["incumbent"]["MAE"]
            and metrics["candidate"]["MAE"] <= metrics["persistence"]["MAE"]
        )
        fold_records.append(
            {
                **fold.to_dict(),
                "metrics": metrics,
                "candidate_mae_not_worse_than_comparators": mae_gate,
                "no_performance_breach": threshold_gate,
                "passed": mae_gate and threshold_gate["passed"],
            }
        )
        for index, observation in enumerate(test_observations):
            all_fold_ids.append(observation.observation_id)
            for model_name, values in predictions.items():
                prediction_rows.append(
                    {
                        "observation_id": observation.observation_id,
                        DATE_COLUMN: observation.target_date.isoformat(),
                        "model": model_name,
                        "Actual_Wind_Production": observation.target_value,
                        "Predicted_Wind_Production": float(values[index]),
                    }
                )
        block_frame = _observation_frame(test_observations, feature_names)
        prior = (
            block_frame
            if prior is None
            else pd.concat([prior, block_frame], ignore_index=True)
        )

    predictions_frame = pd.DataFrame(prediction_rows)
    aggregate = {
        name: regression_metrics(
            predictions_frame.loc[
                predictions_frame["model"].eq(name), "Actual_Wind_Production"
            ],
            predictions_frame.loc[
                predictions_frame["model"].eq(name), "Predicted_Wind_Production"
            ],
            mape_epsilon=float(calibration["mape_epsilon"]),
            r2_minimum_samples=r2_minimum,
        )
        for name in ("candidate", "incumbent", "persistence")
    }
    aggregate_gate = (
        aggregate["candidate"]["MAE"] < aggregate["incumbent"]["MAE"]
        and aggregate["candidate"]["MAE"] < aggregate["persistence"]["MAE"]
    )
    accepted = aggregate_gate and all(item["passed"] for item in fold_records)
    outcome = "accepted" if accepted else "rejected"
    final_training = pd.concat(
        [base, _observation_frame(observations, feature_names)],
        ignore_index=True,
    )
    final_model = None
    training_evidence = None
    model_bytes = None
    final_training_dataset_sha256 = None
    candidate_model_sha256 = None
    if accepted:
        final_model = model_class(**parameters).fit(
            final_training[feature_names], final_training[TARGET_COLUMN]
        )
        training_evidence = final_training.copy()
        training_evidence["Expected_Prediction"] = np.asarray(
            final_model.predict(final_training[feature_names]), dtype=float
        )
        final_training_dataset_sha256 = sha256(
            _csv_bytes(final_training)
        ).hexdigest()
        buffer = BytesIO()
        joblib.dump(final_model, buffer)
        model_bytes = buffer.getvalue()
        candidate_model_sha256 = sha256(model_bytes).hexdigest()
    final_observation_ids = [item.observation_id for item in observations]
    final_training_identity = {
        "base_dataset_sha256": bundle["dataset_manifest"]["sha256"],
        "incumbent_fit_cutoff": evaluation["cutoffs"]["incumbent_fit_cutoff"],
        "data_snapshot_cutoff": evaluation["cutoffs"]["data_snapshot_cutoff"],
        "base_row_count": len(base),
        "candidate_observation_ids": final_observation_ids,
        "row_count": len(final_training),
    }
    body = {
        "schema_version": BACKTEST_SCHEMA,
        "evaluation_period": evaluation["evaluation_period"],
        "evaluation_id": evaluation["evaluation_id"],
        "outcome": outcome,
        "cutoffs": {
            **evaluation["cutoffs"],
            "candidate_fit_cutoff": (
                evaluation["cutoffs"]["data_snapshot_cutoff"] if accepted else None
            ),
        },
        "identities": {
            "policy_sha256": sha256_file(config.policy_path),
            "evaluation_sha256": sha256_file(config.evaluation_path),
            "incumbent_model_sha256": bundle["model_manifest"]["model_sha256"],
            "incumbent_dataset_sha256": bundle["dataset_manifest"]["sha256"],
            "calibration_id": calibration["calibration_id"],
            "reference_id": calibration["reference_id"],
            "feature_schema_sha256": bundle["model_manifest"][
                "feature_schema_sha256"
            ],
        },
        "recipe": {
            "model_type": model_class.__name__,
            "parameters": parameters,
            "source": "exact_incumbent_model_manifest_get_params",
        },
        "fold_plan": fold_plan.to_dict(),
        "evaluated_complete_fold_observation_ids": all_fold_ids,
        "candidate_snapshot_observation_ids": final_observation_ids,
        "final_training": (
            {
                **final_training_identity,
                "identity_sha256": sha256(
                    _canonical(final_training_identity)
                ).hexdigest(),
                "dataset_sha256": final_training_dataset_sha256,
                "candidate_model_sha256": candidate_model_sha256,
            }
            if accepted
            else None
        ),
        "folds": fold_records,
        "aggregate_metrics": aggregate,
        "gates": {
            "aggregate_mae_strictly_better_than_incumbent_and_persistence": (
                aggregate_gate
            ),
            "every_fold_mae_not_worse_than_comparators": all(
                item["candidate_mae_not_worse_than_comparators"]
                for item in fold_records
            ),
            "every_fold_has_no_phase9_performance_breach": all(
                item["no_performance_breach"]["passed"] for item in fold_records
            ),
        },
        "safeguards": {
            "fold_observation_count": 30,
            "minimum_complete_folds": 3,
            "calendar_gaps_preserved": True,
            "incomplete_tail_excluded": True,
            "identical_observation_ids_for_comparators": True,
            "expanding_window": True,
            "future_observations_used_for_training": False,
            "frozen_incumbent": True,
            "automatic_registry_write": False,
            "automatic_promotion": False,
            "network_requests": False,
            "rejected_model_persisted": False,
        },
        "git": git_state(),
        "environment": _environment(include_git=False),
    }
    record = _with_id(body)
    return RetrainingBacktestPlan(
        outcome=outcome,
        evaluation_period=evaluation["evaluation_period"],
        backtest_id=record["backtest_id"],
        record=record,
        final_model=final_model,
        predictions=predictions_frame,
        training_frame=final_training if accepted else None,
        training_evidence=training_evidence,
        model_bytes=model_bytes,
    )


def run_retraining_backtest(
    config: RetrainingBacktestConfig,
) -> RetrainingBacktestResult:
    """Plan and, unless dry-run, atomically seal one immutable period bundle."""
    plan = plan_retraining_backtest(config)
    target = config.output_root / plan.evaluation_period / plan.backtest_id
    if config.dry_run:
        return RetrainingBacktestResult(
            status="planned",
            outcome=plan.outcome,
            backtest_id=plan.backtest_id,
            backtest_dir=None,
            manifest_path=None,
            plan=plan,
        )
    _seal_backtest(config.output_root / plan.evaluation_period, plan)
    return RetrainingBacktestResult(
        status="succeeded",
        outcome=plan.outcome,
        backtest_id=plan.backtest_id,
        backtest_dir=target,
        manifest_path=target / "bundle_manifest.json",
        plan=plan,
    )


def load_retraining_backtest(path: str | Path) -> dict[str, Any]:
    """Load and verify a strict content-addressed v1 backtest bundle."""
    root = Path(path)
    if root.is_file():
        root = root.parent
    manifest = _read_json(root / "bundle_manifest.json")
    required = {
        "schema_version",
        "backtest_id",
        "backtest",
        "files",
        "git",
        "environment",
    }
    if set(manifest) != required or manifest.get("schema_version") != (
        BACKTEST_BUNDLE_SCHEMA
    ):
        raise RetrainingBacktestError("Backtest bundle manifest is not strict v1.")
    backtest = manifest.get("backtest")
    if not isinstance(backtest, Mapping):
        raise RetrainingBacktestError("Backtest record is absent.")
    expected_backtest_fields = {
        "backtest_id",
        "schema_version",
        "evaluation_period",
        "evaluation_id",
        "outcome",
        "cutoffs",
        "identities",
        "recipe",
        "fold_plan",
        "evaluated_complete_fold_observation_ids",
        "candidate_snapshot_observation_ids",
        "final_training",
        "folds",
        "aggregate_metrics",
        "gates",
        "safeguards",
        "git",
        "environment",
    }
    if (
        set(backtest) != expected_backtest_fields
        or backtest.get("schema_version") != BACKTEST_SCHEMA
        or backtest.get("outcome") not in BACKTEST_OUTCOMES
    ):
        raise RetrainingBacktestError("Backtest record fields differ from strict v1.")
    _verify_id(dict(backtest))
    identifier = str(backtest["backtest_id"])
    if (
        identifier != manifest.get("backtest_id")
        or root.name != identifier
        or root.parent.name != backtest["evaluation_period"]
    ):
        raise RetrainingBacktestError(
            "Backtest bundle path and content identity differ."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise RetrainingBacktestError("Backtest bundle checksums are invalid.")
    expected_names = {
        "backtest.json",
        "predictions.csv",
        "fold_metrics.json",
        "aggregate_metrics.json",
        "lineage.json",
        "safeguards.json",
        "environment.json",
    }
    if backtest.get("outcome") == "accepted":
        expected_names |= {
            "model.joblib",
            "model_manifest.json",
            "dataset_manifest.json",
            "reload_sample.csv",
            "training_evidence.csv",
        }
    if set(files) != expected_names:
        raise RetrainingBacktestError(
            "Backtest bundle file set differs from its outcome contract."
        )
    entries = list(root.iterdir())
    if {path.name for path in entries} != {
        *expected_names,
        "bundle_manifest.json",
    }:
        raise RetrainingBacktestError(
            "Backtest bundle contains unexpected or missing filesystem entries."
        )
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise RetrainingBacktestError(
            "Backtest bundle entries must be regular root-level files."
        )
    for name, digest in files.items():
        declared = Path(name)
        if declared.name != name or declared.is_absolute() or ".." in declared.parts:
            raise RetrainingBacktestError("Backtest manifest contains an unsafe path.")
        file_path = root / name
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not file_path.is_file()
            or sha256_file(file_path) != digest
        ):
            raise RetrainingBacktestError(f"Backtest bundle file is corrupt: {name}.")
    environment = _read_json(root / "environment.json")
    if (
        manifest["environment"] != environment
        or manifest["git"] != environment.get("git")
        or backtest["environment"] != {
            key: value for key, value in environment.items() if key != "git"
        }
        or backtest["git"] != environment.get("git")
    ):
        raise RetrainingBacktestError(
            "Backtest manifest and environment lineage differ."
        )
    stored = _read_json(root / "backtest.json")
    if stored != backtest:
        raise RetrainingBacktestError("Manifest and backtest record differ.")
    if backtest["outcome"] == "accepted":
        model_manifest = _read_json(root / "model_manifest.json")
        final_training = backtest.get("final_training")
        if not isinstance(final_training, Mapping):
            raise RetrainingBacktestError("Accepted final-training identity is absent.")
        if (
            model_manifest.get("model_sha256") != sha256_file(root / "model.joblib")
            or model_manifest.get("model_sha256")
            != final_training.get("candidate_model_sha256")
        ):
            raise RetrainingBacktestError("Accepted candidate model checksum is invalid.")
        sample = pd.read_csv(root / "reload_sample.csv")
        features = list(model_manifest.get("feature_names") or [])
        if sample.columns.tolist() != [*features, "Expected_Prediction"]:
            raise RetrainingBacktestError("Candidate reload sample schema is invalid.")
        model = joblib.load(root / "model.joblib")
        actual = np.asarray(model.predict(sample[features]), dtype=float)
        expected = sample["Expected_Prediction"].to_numpy(dtype=float)
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-9):
            raise RetrainingBacktestError(
                "Reloaded candidate predictions differ from sealed evidence."
            )
        training_evidence = pd.read_csv(root / "training_evidence.csv")
        if training_evidence.columns.tolist() != [
            DATE_COLUMN,
            TARGET_COLUMN,
            *features,
            "Expected_Prediction",
        ]:
            raise RetrainingBacktestError(
                "Complete candidate training-evidence schema is invalid."
            )
        exact_training = training_evidence.drop(columns=["Expected_Prediction"])
        if sha256(_csv_bytes(exact_training)).hexdigest() != final_training.get(
            "dataset_sha256"
        ):
            raise RetrainingBacktestError(
                "Complete final-training dataset checksum is invalid."
            )
    elif backtest.get("final_training") is not None:
        raise RetrainingBacktestError(
            "Rejected backtest must not claim final-training evidence."
        )
    return manifest


def _verify_pinned_inputs(
    config: RetrainingBacktestConfig,
    evaluation: Mapping[str, Any],
    policy: RetrainingPolicy,
    bundle: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> Mapping[str, Any]:
    if evaluation["policy"]["sha256"] != sha256_file(config.policy_path):
        raise RetrainingBacktestError("Evaluation and policy checksums differ.")
    if evaluation["policy"]["schema_version"] != policy.schema_version:
        raise RetrainingBacktestError("Evaluation and policy schemas differ.")
    expected_report_state = (
        config.monitoring_store_root / "reporting" / "state" / "current.json"
    ).resolve()
    expected_ledger = (
        config.monitoring_store_root / "state" / "current.json"
    ).resolve()
    if Path(evaluation["phase9_report_state"]["path"]).resolve() != expected_report_state:
        raise RetrainingBacktestError(
            "Evaluation report-state path is not the configured fixed pointer."
        )
    if Path(evaluation["phase9_ledger"]["path"]).resolve() != expected_ledger:
        raise RetrainingBacktestError(
            "Evaluation ledger path is not the configured fixed pointer."
        )
    report_path = Path(evaluation["phase9_report"]["path"]).resolve()
    expected_reports_root = (
        config.monitoring_store_root / "reporting" / "reports"
    ).resolve()
    try:
        report_path.relative_to(expected_reports_root)
    except ValueError as exc:
        raise RetrainingBacktestError(
            "Evaluation report does not belong to the configured store."
        ) from exc
    for section, path in (
        ("phase9_report", report_path),
        ("phase9_report_state", expected_report_state),
        ("phase9_ledger", expected_ledger),
    ):
        if not path.is_file() or sha256_file(path) != evaluation[section]["sha256"]:
            raise RetrainingBacktestError(
                f"Evaluation-pinned {section} is missing or corrupt."
            )
    try:
        report = load_monitoring_report(report_path)
        ledger = load_verified_monitoring_state(config.monitoring_store_root)
        report_state = load_monitoring_report_state(config.monitoring_store_root)
    except (MonitoringError, MonitoringReportingError) as exc:
        raise RetrainingBacktestError(str(exc)) from exc
    if ledger is None:
        raise RetrainingBacktestError("Verified monitoring ledger is absent.")
    if report_state is None:
        raise RetrainingBacktestError("Verified monitoring report state is absent.")
    exact_ledger = _read_json(expected_ledger)
    exact_report_state = _read_json(expected_report_state)
    if ledger != exact_ledger or report_state != exact_report_state:
        raise RetrainingBacktestError(
            "Verified monitoring state differs from the exact pinned records."
        )
    if report["report_id"] != evaluation["phase9_report"]["report_id"]:
        raise RetrainingBacktestError("Evaluation and Phase 9 report IDs differ.")
    if ledger.get("generation") != evaluation["phase9_ledger"]["generation"]:
        raise RetrainingBacktestError("Evaluation and Phase 9 ledger generations differ.")
    if ledger.get("model_snapshot_id") != evaluation["incumbent"]["incumbent_id"]:
        raise RetrainingBacktestError("Evaluation and model snapshot IDs differ.")
    if (
        report_state.get("generation")
        != evaluation["phase9_report_state"]["generation"]
        or report_state.get("latest_report_id")
        != evaluation["phase9_report_state"]["latest_report_id"]
        or report_state.get("latest_through_date")
        != evaluation["phase9_report_state"]["latest_through_date"]
    ):
        raise RetrainingBacktestError(
            "Evaluation and exact Phase 9 report-state semantics differ."
        )
    if calibration["policy_sha256"] != (
        (report.get("reference") or {}).get("policy_sha256")
    ):
        raise RetrainingBacktestError("Report and calibration policy identities differ.")
    reference = report.get("reference") or {}
    if (
        reference.get("calibration_id") != calibration["calibration_id"]
        or reference.get("reference_id") != calibration["reference_id"]
    ):
        raise RetrainingBacktestError(
            "Evaluation report and calibration identities differ."
        )
    if calibration["_reference_manifest"]["model_sha256"] != (
        bundle["model_manifest"]["model_sha256"]
    ):
        raise RetrainingBacktestError("Calibration and incumbent model identities differ.")
    return ledger


def _reconstruct_observations(
    config: RetrainingBacktestConfig,
    evaluation: Mapping[str, Any],
    bundle: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> list[ObservationEvidence]:
    expected = list(evaluation["eligibility"]["eligible_observations"])
    if [item["observation_id"] for item in expected] != (
        evaluation["eligibility"]["eligible_observation_ids"]
    ):
        raise RetrainingBacktestError("Evaluation eligible observation order is corrupt.")
    expected_by_date = {item["target_date"]: item for item in expected}
    if len(expected_by_date) != len(expected):
        raise RetrainingBacktestError("Evaluation contains duplicate target dates.")
    observations = []
    for day, pinned in expected_by_date.items():
        prediction_id = (ledger.get("as_issued") or {}).get(day)
        actual_id = (ledger.get("actuals") or {}).get(day)
        if not isinstance(prediction_id, str) or not isinstance(actual_id, str):
            raise RetrainingBacktestError("Evaluation evidence is absent from the ledger.")
        try:
            evidence = load_prediction_evidence(
                config.monitoring_store_root, prediction_id
            )
        except MonitoringError as exc:
            raise RetrainingBacktestError(str(exc)) from exc
        prediction = evidence["prediction"]
        snapshot = evidence["model_input_snapshot"]
        actual = next(
            (
                item
                for item in evidence["actual_revisions"]
                if item.get("actual_revision_id") == actual_id
            ),
            None,
        )
        if actual is None:
            raise RetrainingBacktestError("Current actual revision is not verified.")
        lineage = {
            "dependencies": snapshot["dependencies"],
            "target_revision_id": actual_id,
        }
        observation_id = sha256(
            b"retraining_observation:"
            + _canonical(
                {
                    "prediction_id": prediction_id,
                    "feature_snapshot_id": snapshot["model_input_snapshot_id"],
                    "target_revision_id": actual_id,
                }
            )
        ).hexdigest()
        if (
            observation_id != pinned["observation_id"]
            or snapshot["model_input_snapshot_id"] != pinned["feature_snapshot_id"]
            or actual_id != pinned["target_revision_id"]
            or sha256(_canonical(lineage)).hexdigest() != pinned["lineage_sha256"]
            or prediction.get("view") != "as_issued"
            or prediction.get("model_snapshot_id")
            != evaluation["incumbent"]["incumbent_id"]
            or snapshot.get("feature_names") != bundle["feature_names"]
            or evidence["model_snapshot"].get("model", {}).get("model_sha256")
            != bundle["model_manifest"]["model_sha256"]
            or evidence["model_snapshot"].get("dataset", {}).get(
                "dataset_sha256"
            )
            != bundle["dataset_manifest"]["sha256"]
            or evidence["model_snapshot"].get("feature_schema_sha256")
            != bundle["model_manifest"]["feature_schema_sha256"]
        ):
            raise RetrainingBacktestError(
                "Reconstructed observation differs from monthly evidence."
            )
        dependencies = snapshot["dependencies"]
        source_ids = []
        for dependency in dependencies.values():
            source_ids.extend(
                item["revision_id"] for item in dependency["source_revisions"]
            )
        source_ids.append(actual["source_revision_id"])
        observation = ObservationEvidence(
            observation_id=observation_id,
            target_date=date.fromisoformat(day),
            feature_snapshot_id=snapshot["model_input_snapshot_id"],
            target_revision_id=actual_id,
            feature_schema_sha256=snapshot["feature_schema_sha256"],
            lineage_sha256=pinned["lineage_sha256"],
            target_contract_id=actual["target_contract_id"],
            transformation_version=snapshot["transformation"]["version"],
            source_revision_ids=tuple(sorted(set(source_ids))),
            feature_values=tuple(snapshot["feature_values"]),
            target_value=float(actual["actual"]),
        )
        observations.append(observation)
    ordered = sorted(observations, key=lambda item: item.target_date)
    if [item.observation_id for item in ordered] != (
        evaluation["eligibility"]["eligible_observation_ids"]
    ):
        raise RetrainingBacktestError("Reconstructed observation order differs.")
    return ordered


def _load_base_dataset(
    path: Path,
    bundle: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> pd.DataFrame:
    if sha256_file(path) != bundle["dataset_manifest"]["sha256"]:
        raise RetrainingBacktestError("Incumbent base dataset checksum is invalid.")
    frame = pd.read_csv(path)
    expected = [DATE_COLUMN, TARGET_COLUMN, *bundle["feature_names"]]
    if frame.columns.tolist() != expected:
        raise RetrainingBacktestError("Incumbent base dataset schema/order differs.")
    dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    numeric = frame[[TARGET_COLUMN, *bundle["feature_names"]]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
        or numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise RetrainingBacktestError("Incumbent base dataset is not strict and finite.")
    cutoff = pd.Timestamp(evaluation["cutoffs"]["incumbent_fit_cutoff"])
    selected = frame.loc[dates.le(cutoff)].copy().reset_index(drop=True)
    expected_rows = int(
        bundle["dataset_manifest"]["splits"]["row_counts"]["refit_train_validation"]
    )
    if len(selected) != expected_rows or selected.empty:
        raise RetrainingBacktestError(
            "Historical rows through incumbent_fit_cutoff differ from fit evidence."
        )
    return selected


def _incumbent_recipe(
    bundle: Mapping[str, Any],
) -> tuple[type[ExtraTreesRegressor] | type[RandomForestRegressor], dict[str, Any]]:
    name = bundle["model_manifest"].get("model_type")
    model_class = MODEL_TYPES.get(str(name))
    parameters = bundle["model_manifest"].get("parameters")
    if model_class is None or not isinstance(parameters, dict):
        raise RetrainingBacktestError("Unsupported incumbent retraining recipe.")
    try:
        probe = model_class(**parameters)
    except (TypeError, ValueError) as exc:
        raise RetrainingBacktestError("Incumbent recipe parameters are invalid.") from exc
    if probe.get_params(deep=True) != parameters:
        raise RetrainingBacktestError(
            "Incumbent recipe is not the exact serialized get_params contract."
        )
    return model_class, dict(parameters)


def _performance_limits(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    limits = ((calibration.get("thresholds") or {}).get("performance") or {}).get(
        "30"
    )
    if not isinstance(limits, Mapping):
        raise RetrainingBacktestError("Calibration performance.30 limits are absent.")
    return limits


def _phase9_r2_minimum(calibration: Mapping[str, Any]) -> int:
    value = ((calibration.get("policy") or {}).get("r2_minimum_samples") or {}).get(
        "30"
    )
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 2 <= value <= 30
    ):
        raise RetrainingBacktestError(
            "Calibration Phase 9 R2 minimum for performance.30 is invalid."
        )
    return value


def _validate_calibration_preflight(
    config: RetrainingBacktestConfig,
    calibration: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int]:
    if config.calibration_dir.resolve().name != calibration.get("calibration_id"):
        raise RetrainingBacktestError(
            "Calibration directory and content identity differ."
        )
    monitoring_policy = calibration.get("policy")
    if (
        not isinstance(monitoring_policy, Mapping)
        or monitoring_policy.get("schema_version")
        != "wind_forecast.monitoring_policy.v1"
        or 30 not in monitoring_policy.get("windows_days", ())
    ):
        raise RetrainingBacktestError(
            "Calibration monitoring policy is missing or incompatible."
        )
    limits = _performance_limits(calibration)
    if set(limits) != set(REQUIRED_PERFORMANCE_METRICS):
        raise RetrainingBacktestError(
            "Calibration performance.30 leaf set differs from strict v1."
        )
    expected_directions = {
        "MAE": "upper",
        "RMSE": "upper",
        "MAPE_percent": "upper",
        "R2": "lower",
        "absolute_bias": "upper",
    }
    for name, direction in expected_directions.items():
        node = limits[name]
        if (
            not isinstance(node, Mapping)
            or set(node).difference(
                {
                    "warning",
                    "critical",
                    "direction",
                    "override",
                    "calibrated_warning",
                    "calibrated_critical",
                }
            )
            or not {"warning", "critical", "direction"}.issubset(node)
            or node["direction"] != direction
            or not _finite(node["warning"])
            or not _finite(node["critical"])
        ):
            raise RetrainingBacktestError(
                f"Calibration performance.30 {name} leaf is invalid."
            )
        warning = float(node["warning"])
        critical = float(node["critical"])
        if (
            direction == "upper"
            and warning > critical
            or direction == "lower"
            and warning < critical
        ):
            raise RetrainingBacktestError(
                f"Calibration performance.30 {name} severity order is invalid."
            )
    return limits, _phase9_r2_minimum(calibration)


def _validate_loaded_incumbent(
    incumbent: Any,
    model_class: type[ExtraTreesRegressor] | type[RandomForestRegressor],
    parameters: Mapping[str, Any],
    feature_names: Sequence[str],
) -> None:
    if type(incumbent) is not model_class:
        raise RetrainingBacktestError(
            "Loaded frozen incumbent class differs from its manifest."
        )
    if incumbent.get_params(deep=True) != dict(parameters):
        raise RetrainingBacktestError(
            "Loaded frozen incumbent parameters differ from its manifest."
        )
    if list(getattr(incumbent, "feature_names_in_", ())) != list(feature_names):
        raise RetrainingBacktestError(
            "Loaded frozen incumbent feature order differs from its manifest."
        )


def _observation_frame(
    observations: Sequence[ObservationEvidence],
    features: Sequence[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                DATE_COLUMN: item.target_date.isoformat(),
                TARGET_COLUMN: item.target_value,
                **dict(zip(features, item.feature_values, strict=True)),
            }
            for item in observations
        ],
        columns=[DATE_COLUMN, TARGET_COLUMN, *features],
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve()
    right = second.resolve()
    return left == right or left in right.parents or right in left.parents


def _seal_backtest(period_root: Path, plan: RetrainingBacktestPlan) -> None:
    if period_root.exists():
        _validate_sealed_period(period_root, plan.backtest_id)
        return
    prepared = period_root.parent / (
        f".{period_root.name}.{plan.backtest_id}.{uuid4().hex}.tmp"
    )
    target = prepared / plan.backtest_id
    prepared.mkdir(parents=True, exist_ok=False)
    try:
        target.mkdir()
        _write_json(target / "backtest.json", plan.record)
        plan.predictions.to_csv(
            target / "predictions.csv",
            index=False,
            lineterminator="\n",
            float_format="%.12g",
        )
        _write_json(target / "fold_metrics.json", {"folds": plan.record["folds"]})
        _write_json(
            target / "aggregate_metrics.json",
            {"aggregate_metrics": plan.record["aggregate_metrics"]},
        )
        _write_json(target / "lineage.json", plan.record["identities"])
        _write_json(target / "safeguards.json", plan.record["safeguards"])
        environment = {**plan.record["environment"], "git": plan.record["git"]}
        _write_json(target / "environment.json", environment)
        if plan.outcome == "accepted":
            assert (
                plan.final_model is not None
                and plan.training_frame is not None
                and plan.training_evidence is not None
                and plan.model_bytes is not None
            )
            (target / "model.joblib").write_bytes(plan.model_bytes)
            features = list(plan.final_model.feature_names_in_)
            sample = plan.training_frame[features].tail(5).copy()
            sample["Expected_Prediction"] = plan.final_model.predict(sample)
            sample.to_csv(
                target / "reload_sample.csv",
                index=False,
                lineterminator="\n",
                float_format="%.12g",
            )
            plan.training_evidence.to_csv(
                target / "training_evidence.csv",
                index=False,
                lineterminator="\n",
                float_format="%.12g",
            )
            _write_json(
                target / "model_manifest.json",
                {
                    "schema_version": "wind_forecast.retraining_candidate_model.v1",
                    "model_type": type(plan.final_model).__name__,
                    "model_sha256": sha256_file(target / "model.joblib"),
                    "feature_names": features,
                    "feature_schema_sha256": plan.record["identities"][
                        "feature_schema_sha256"
                    ],
                    "parameters": plan.final_model.get_params(deep=True),
                    "candidate_fit_cutoff": plan.record["cutoffs"][
                        "candidate_fit_cutoff"
                    ],
                    "backtest_id": plan.backtest_id,
                    "final_training_identity_sha256": plan.record[
                        "final_training"
                    ]["identity_sha256"],
                },
            )
            _write_json(
                target / "dataset_manifest.json",
                {
                    "schema_version": (
                        "wind_forecast.retraining_candidate_dataset.v1"
                    ),
                    "base_dataset_sha256": plan.record["identities"][
                        "incumbent_dataset_sha256"
                    ],
                    "feature_schema_sha256": plan.record["identities"][
                        "feature_schema_sha256"
                    ],
                    "row_count": len(plan.training_frame),
                    "candidate_fit_cutoff": plan.record["cutoffs"][
                        "candidate_fit_cutoff"
                    ],
                    "candidate_observation_ids": plan.record[
                        "candidate_snapshot_observation_ids"
                    ],
                    "evaluated_complete_fold_observation_ids": plan.record[
                        "evaluated_complete_fold_observation_ids"
                    ],
                    "final_training_identity_sha256": plan.record[
                        "final_training"
                    ]["identity_sha256"],
                    "final_training_dataset_sha256": plan.record[
                        "final_training"
                    ]["dataset_sha256"],
                },
            )
        names = sorted(
            path.name
            for path in target.iterdir()
            if path.name != "bundle_manifest.json"
        )
        manifest = {
            "schema_version": BACKTEST_BUNDLE_SCHEMA,
            "backtest_id": plan.backtest_id,
            "backtest": dict(plan.record),
            "files": {name: sha256_file(target / name) for name in names},
            "git": environment["git"],
            "environment": environment,
        }
        _write_json(target / "bundle_manifest.json", manifest)
        try:
            prepared.rename(period_root)
        except OSError as exc:
            if not period_root.is_dir():
                raise RetrainingBacktestError(
                    "Atomic backtest-period publication failed."
                ) from exc
            _validate_sealed_period(period_root, plan.backtest_id)
    finally:
        if prepared.exists():
            _remove_known_tree(prepared, plan.backtest_id)
    _validate_sealed_period(period_root, plan.backtest_id)


def _validate_sealed_period(period_root: Path, expected_id: str) -> None:
    entries = list(period_root.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        raise RetrainingBacktestError(
            "Sealed backtest period must contain exactly one bundle."
        )
    manifest = load_retraining_backtest(entries[0])
    if manifest["backtest_id"] != expected_id:
        raise RetrainingBacktestError(
            "Conflicting evidence already seals this backtest period."
        )


def _remove_known_tree(root: Path, identifier: str) -> None:
    target = root / identifier
    if target.is_dir():
        for path in target.iterdir():
            if path.is_file():
                path.unlink()
        target.rmdir()
    root.rmdir()


def _environment(*, include_git: bool = True) -> dict[str, Any]:
    packages = {}
    for package in ("joblib", "numpy", "pandas", "scikit-learn"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    result = {
        "schema_version": "wind_forecast.retraining_environment.v1",
        "python": platform.python_version(),
        "packages": packages,
    }
    if include_git:
        result["git"] = git_state()
    return result


def _with_id(body: Mapping[str, Any]) -> dict[str, Any]:
    ready = json.loads(_canonical(body))
    return {"backtest_id": _record_id(ready), **ready}


def _record_id(body: Mapping[str, Any]) -> str:
    return sha256(b"retraining_backtest:" + _canonical(body)).hexdigest()


def _verify_id(record: dict[str, Any]) -> None:
    identifier = record.get("backtest_id")
    if identifier != _record_id(
        {key: value for key, value in record.items() if key != "backtest_id"}
    ):
        raise RetrainingBacktestError("Backtest content-addressed identity is corrupt.")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    data = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrainingBacktestError(f"Invalid backtest JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise RetrainingBacktestError(f"Backtest JSON must be an object: {path}.")
    return value


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


__all__ = [
    "BACKTEST_BUNDLE_SCHEMA",
    "BACKTEST_OUTCOMES",
    "BACKTEST_SCHEMA",
    "RetrainingBacktestConfig",
    "RetrainingBacktestError",
    "RetrainingBacktestPlan",
    "RetrainingBacktestResult",
    "evaluate_no_performance_breach",
    "load_retraining_backtest",
    "plan_retraining_backtest",
    "run_retraining_backtest",
]
