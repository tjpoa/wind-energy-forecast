"""Sealed-test challenger backtesting for the scaled ANN v2 candidate."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .manifests import sha256_file
from .monitoring import validate_monitoring_model_bundle
from .monitoring_reporting import load_monitoring_calibration
from .monitoring_statistics import regression_metrics, threshold_severity
from .paths import project_root
from .schemas import DATE_COLUMN, TARGET_COLUMN
from .v2_ann import load_v2_ann_bundle
from .v2_training import PERSISTENCE_COLUMN


CHALLENGER_BACKTEST_SCHEMA = "wind_forecast.v2_ann_challenger_backtest.v1"
CHALLENGER_BUNDLE_SCHEMA = "wind_forecast.v2_ann_challenger_bundle.v1"
DEFAULT_OUTPUT_ROOT = project_root() / "outputs/backtests/v2_ann_challenger"
DEFAULT_EVALUATION_PERIOD = "sealed_test_2025_2026"
DEFAULT_FOLD_SIZE = 30
REQUIRED_METRICS = ("MAE", "RMSE", "MAPE_percent", "R2", "absolute_bias")
COMPARATORS = ("candidate", "incumbent", "persistence")


@dataclass(frozen=True)
class ChallengerBacktestConfig:
    """Explicit inputs for one sealed challenger backtest."""

    candidate_bundle: Path
    incumbent_bundle: Path
    dataset_path: Path
    incumbent_calibration: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT
    evaluation_period: str = DEFAULT_EVALUATION_PERIOD
    test_start: str = "2025-01-01"
    test_end: str = "2026-06-27"
    fold_size: int = DEFAULT_FOLD_SIZE
    dry_run: bool = False

    def __post_init__(self) -> None:
        for name in (
            "candidate_bundle",
            "incumbent_bundle",
            "dataset_path",
            "incumbent_calibration",
            "output_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if not self.evaluation_period or "/" in self.evaluation_period or "\\" in self.evaluation_period:
            raise ValueError("evaluation_period must be a safe non-empty name.")
        if self.fold_size < 1:
            raise ValueError("fold_size must be positive.")


@dataclass(frozen=True)
class ChallengerBacktestPlan:
    """Read-only result of challenger modelling and gate evaluation."""

    outcome: str
    backtest_id: str
    record: Mapping[str, Any]
    predictions: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "backtest_id": self.backtest_id,
            "record": dict(self.record),
        }


@dataclass(frozen=True)
class ChallengerBacktestResult:
    """Dry-run or sealed challenger backtest result."""

    status: str
    outcome: str
    backtest_id: str
    backtest_dir: Path | None
    manifest_path: Path | None
    plan: ChallengerBacktestPlan

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "outcome": self.outcome,
            "backtest_id": self.backtest_id,
            "backtest_dir": str(self.backtest_dir) if self.backtest_dir else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "plan": self.plan.summary(),
        }


def run_v2_ann_challenger_backtest(
    config: ChallengerBacktestConfig,
) -> ChallengerBacktestResult:
    """Evaluate one frozen ANN against the v2 incumbent and persistence."""
    plan = plan_v2_ann_challenger_backtest(config)
    if config.dry_run:
        return ChallengerBacktestResult(
            status="planned",
            outcome=plan.outcome,
            backtest_id=plan.backtest_id,
            backtest_dir=None,
            manifest_path=None,
            plan=plan,
        )
    target = config.output_root / config.evaluation_period / plan.backtest_id
    if target.exists():
        existing = _read_json(target / "backtest.json")
        if existing.get("backtest_id") != plan.backtest_id:
            raise ChallengerBacktestError("Existing challenger backtest identity differs.")
        return ChallengerBacktestResult(
            status="no_op",
            outcome=plan.outcome,
            backtest_id=plan.backtest_id,
            backtest_dir=target,
            manifest_path=target / "bundle_manifest.json",
            plan=plan,
        )
    _seal_challenger_bundle(target, plan, config)
    return ChallengerBacktestResult(
        status="created",
        outcome=plan.outcome,
        backtest_id=plan.backtest_id,
        backtest_dir=target,
        manifest_path=target / "bundle_manifest.json",
        plan=plan,
    )


def plan_v2_ann_challenger_backtest(
    config: ChallengerBacktestConfig,
) -> ChallengerBacktestPlan:
    """Run all validation and in-memory predictions without writing output."""
    try:
        candidate = load_v2_ann_bundle(config.candidate_bundle)
        incumbent = validate_monitoring_model_bundle(config.incumbent_bundle)
        calibration = load_monitoring_calibration(config.incumbent_calibration)
    except Exception as exc:
        if isinstance(exc, ChallengerBacktestError):
            raise
        raise ChallengerBacktestError(str(exc)) from exc
    if "root" not in incumbent or not (Path(incumbent["root"]) / "model.joblib").is_file():
        raise ChallengerBacktestError("The incumbent bundle has no sealed model.joblib.")
    frame = pd.read_csv(config.dataset_path)
    if sha256_file(config.dataset_path) != _candidate_dataset_sha(config.candidate_bundle):
        raise ChallengerBacktestError("Dataset checksum differs from the candidate bundle.")
    feature_names = list(candidate.feature_names)
    expected = [DATE_COLUMN, TARGET_COLUMN, *feature_names]
    if frame.columns.tolist() != expected:
        raise ChallengerBacktestError("Challenger dataset columns/order differ from the candidate schema.")
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if frame[DATE_COLUMN].isna().any() or frame[DATE_COLUMN].duplicated().any():
        raise ChallengerBacktestError("Challenger dates are invalid or duplicated.")
    test = frame.loc[frame[DATE_COLUMN].between(config.test_start, config.test_end)].copy()
    if test.empty or len(test) % config.fold_size:
        raise ChallengerBacktestError("The sealed challenger test period must contain complete folds.")
    if test[DATE_COLUMN].min() != pd.Timestamp(config.test_start) or test[DATE_COLUMN].max() != pd.Timestamp(config.test_end):
        raise ChallengerBacktestError("The challenger test period does not match its explicit bounds.")
    if (
        config.test_start == "2025-01-01"
        and config.test_end == "2026-06-27"
        and (len(test), config.fold_size) != (450, DEFAULT_FOLD_SIZE)
    ):
        raise ChallengerBacktestError("The governed challenger test must be 15 consecutive folds of 30 rows.")
    _validate_candidate_fit_cutoff(config.candidate_bundle, test[DATE_COLUMN].min())
    incumbent_model = joblib.load(Path(incumbent["root"]) / "model.joblib")
    incumbent_features = list(incumbent.get("feature_names") or ())
    if incumbent_features != feature_names:
        raise ChallengerBacktestError("Candidate and incumbent feature order differs.")
    if PERSISTENCE_COLUMN not in feature_names:
        raise ChallengerBacktestError("Challenger comparisons require Wind_Production_Lag1.")
    mape_epsilon = float(calibration.get("mape_epsilon", 0.0))
    r2_minimum = _r2_minimum(calibration)
    limits = _performance_limits(calibration)
    prediction_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for fold_index, start in enumerate(range(0, len(test), config.fold_size), start=1):
        part = test.iloc[start : start + config.fold_size].copy()
        actual = part[TARGET_COLUMN].to_numpy(dtype=float)
        candidate_prediction = candidate.predict(part[feature_names])
        incumbent_prediction = np.asarray(
            incumbent_model.predict(part[feature_names]), dtype=float
        ).reshape(-1)
        persistence_prediction = part[PERSISTENCE_COLUMN].to_numpy(dtype=float)
        for model_name, values in (
            ("candidate", candidate_prediction),
            ("incumbent", incumbent_prediction),
            ("persistence", persistence_prediction),
        ):
            if len(values) != len(part) or not np.isfinite(values).all():
                raise ChallengerBacktestError(
                    f"{model_name} produced non-finite challenger predictions."
                )
        predictions = {
            "candidate": candidate_prediction,
            "incumbent": incumbent_prediction,
            "persistence": persistence_prediction,
        }
        metrics = {
            name: _metric_values(actual, values, mape_epsilon, r2_minimum)
            for name, values in predictions.items()
        }
        gate = {
            "candidate_mae_not_worse_than_comparators": (
                metrics["candidate"]["MAE"] <= metrics["incumbent"]["MAE"]
                and metrics["candidate"]["MAE"] <= metrics["persistence"]["MAE"]
            ),
            "no_incumbent_calibration_breach": _no_breach(
                metrics["candidate"], limits
            ),
        }
        fold_records.append(
            {
                "fold": fold_index,
                "start": _date_text(part[DATE_COLUMN].iloc[0]),
                "end": _date_text(part[DATE_COLUMN].iloc[-1]),
                "row_count": len(part),
                "metrics": metrics,
                "gate": gate,
                "passed": all(gate.values()),
            }
        )
        for row_index, (_, row) in enumerate(part.iterrows()):
            observation_id = f"{fold_index:02d}-{row_index:02d}-{_date_text(row[DATE_COLUMN])}"
            for name, values in predictions.items():
                prediction_rows.append(
                    {
                        "observation_id": observation_id,
                        DATE_COLUMN: _date_text(row[DATE_COLUMN]),
                        "model": name,
                        "Actual_Wind_Production": float(actual[row_index]),
                        "Predicted_Wind_Production": float(values[row_index]),
                    }
                )
    predictions_frame = pd.DataFrame(prediction_rows)
    aggregate = {
        name: _metric_values(
            predictions_frame.loc[predictions_frame["model"].eq(name), "Actual_Wind_Production"].to_numpy(float),
            predictions_frame.loc[predictions_frame["model"].eq(name), "Predicted_Wind_Production"].to_numpy(float),
            mape_epsilon,
            r2_minimum,
        )
        for name in COMPARATORS
    }
    aggregate_gate = {
        "candidate_mae_strictly_better_than_incumbent": aggregate["candidate"]["MAE"] < aggregate["incumbent"]["MAE"],
        "candidate_mae_strictly_better_than_persistence": aggregate["candidate"]["MAE"] < aggregate["persistence"]["MAE"],
        "every_fold_passed": all(record["passed"] for record in fold_records),
        "candidate_within_incumbent_calibration": _no_breach(
            aggregate["candidate"], limits
        ),
    }
    accepted = all(aggregate_gate.values())
    body = {
        "schema_version": CHALLENGER_BACKTEST_SCHEMA,
        "evaluation_period": config.evaluation_period,
        "outcome": "accepted" if accepted else "rejected",
        "test_period": {"start": config.test_start, "end": config.test_end},
        "fold_size": config.fold_size,
        "fold_count": len(fold_records),
        "identities": {
            "candidate_bundle_sha256": sha256_file(Path(config.candidate_bundle) / "model_manifest.json"),
            "incumbent_bundle_sha256": sha256_file(Path(incumbent["root"]) / "model_manifest.json"),
            "dataset_sha256": sha256_file(config.dataset_path),
            "calibration_id": calibration["calibration_id"],
            "calibration_sha256": sha256_file(Path(config.incumbent_calibration) / "calibration.json"),
            "feature_schema_sha256": _hash_json(feature_names),
        },
        "folds": fold_records,
        "aggregate_metrics": aggregate,
        "gates": aggregate_gate,
        "safeguards": {
            "test_used_for_selection": False,
            "candidate_frozen": True,
            "incumbent_frozen": True,
            "same_observations_for_comparators": True,
            "calendar_order_preserved": True,
            "automatic_registry_write": False,
            "automatic_promotion": False,
            "network_requests": False,
        },
        "git": _git_state(),
        "environment": _environment_manifest(),
    }
    body["backtest_id"] = _record_id(body)
    return ChallengerBacktestPlan(
        outcome=body["outcome"],
        backtest_id=body["backtest_id"],
        record=body,
        predictions=predictions_frame,
    )


class ChallengerBacktestError(RuntimeError):
    """Raised when challenger evidence is invalid or fails closed."""


def _seal_challenger_bundle(
    target: Path,
    plan: ChallengerBacktestPlan,
    config: ChallengerBacktestConfig,
) -> None:
    target.mkdir(parents=True)
    candidate_root = Path(config.candidate_bundle)
    if plan.outcome == "accepted":
        for source in candidate_root.iterdir():
            if source.is_file():
                shutil.copy2(source, target / source.name)
    else:
        # A rejected result is audit evidence only.  Do not copy the model or
        # scalers into a directory that could be mistaken for a registrable
        # candidate bundle.
        evidence_names = {
            "metrics.json",
            "variant_comparison.json",
            "training_history.json",
            "dataset_manifest.json",
            "environment.json",
            "run_summary.json",
            "test_predictions.csv",
            "validation_predictions.csv",
        }
        for name in evidence_names:
            source = candidate_root / name
            if source.is_file():
                shutil.copy2(source, target / source.name)
    plan.predictions.to_csv(target / "predictions.csv", index=False, lineterminator="\n")
    _write_json(target / "backtest.json", dict(plan.record))
    _write_json(
        target / "fold_metrics.json",
        {"schema_version": CHALLENGER_BACKTEST_SCHEMA, "folds": plan.record["folds"]},
    )
    _write_json(
        target / "aggregate_metrics.json",
        {"schema_version": CHALLENGER_BACKTEST_SCHEMA, "metrics": plan.record["aggregate_metrics"], "gates": plan.record["gates"]},
    )
    _write_json(
        target / "lineage.json",
        {
            "schema_version": "wind_forecast.v2_ann_challenger_lineage.v1",
            "candidate_bundle": str(candidate_root.resolve()),
            "candidate_manifest_sha256": sha256_file(candidate_root / "model_manifest.json"),
            "dataset_sha256": sha256_file(config.dataset_path),
            "incumbent_bundle": str(Path(config.incumbent_bundle).resolve()),
            "calibration": str(Path(config.incumbent_calibration).resolve()),
        },
    )
    _write_json(target / "safeguards.json", dict(plan.record["safeguards"]))
    _write_json(target / "environment.json", dict(plan.record["environment"]))
    expected = {path.name for path in target.iterdir() if path.is_file()}
    manifest = {
        "schema_version": CHALLENGER_BUNDLE_SCHEMA,
        "backtest_id": plan.backtest_id,
        "outcome": plan.outcome,
        "files": {
            name: {"sha256": sha256_file(target / name)}
            for name in sorted(expected)
        },
    }
    _write_json(target / "bundle_manifest.json", manifest)


def load_v2_ann_challenger_bundle(path: str | Path) -> dict[str, Any]:
    """Load and verify one accepted, immutable challenger bundle."""
    root = Path(path)
    manifest = _read_json(root / "bundle_manifest.json")
    if manifest.get("schema_version") != CHALLENGER_BUNDLE_SCHEMA:
        raise ChallengerBacktestError("Unsupported ANN challenger bundle schema.")
    if manifest.get("outcome") != "accepted":
        raise ChallengerBacktestError("Only an accepted challenger is registrable.")
    backtest = _read_json(root / "backtest.json")
    if backtest.get("schema_version") != CHALLENGER_BACKTEST_SCHEMA:
        raise ChallengerBacktestError("Unsupported ANN challenger backtest schema.")
    backtest_id = str(backtest.get("backtest_id") or "")
    if not backtest_id:
        raise ChallengerBacktestError("Challenger backtest has no identity.")
    unsigned = dict(backtest)
    unsigned.pop("backtest_id", None)
    if _record_id(unsigned) != backtest_id or manifest.get("backtest_id") != backtest_id:
        raise ChallengerBacktestError("Challenger backtest identity is invalid.")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ChallengerBacktestError("Challenger bundle has no file manifest.")
    expected = {str(name) for name in files}
    actual = {item.name for item in root.iterdir() if item.is_file()}
    if actual != expected | {"bundle_manifest.json"}:
        raise ChallengerBacktestError("Challenger bundle file set differs from its manifest.")
    for name, entry in files.items():
        file_path = root / str(name)
        if not isinstance(entry, Mapping) or entry.get("sha256") != sha256_file(file_path):
            raise ChallengerBacktestError(f"Challenger bundle checksum is invalid: {name}.")
    candidate_manifest = _read_json(root / "model_manifest.json")
    if candidate_manifest.get("artifact_type") != "keras_scaled_v2":
        raise ChallengerBacktestError("Challenger bundle does not contain a scaled ANN candidate.")
    return {
        "root": root,
        "backtest": backtest,
        "bundle_manifest": manifest,
        "model_manifest": candidate_manifest,
        "dataset_manifest": _read_json(root / "dataset_manifest.json"),
        "lineage": _read_json(root / "lineage.json"),
    }


def _candidate_dataset_sha(path: Path) -> str:
    manifest = _read_json(path / "dataset_manifest.json")
    value = str(manifest.get("sha256") or "")
    if len(value) != 64:
        raise ChallengerBacktestError("Candidate dataset manifest has no valid checksum.")
    return value


def _validate_candidate_fit_cutoff(path: Path, test_start: pd.Timestamp) -> None:
    manifest = _read_json(path / "dataset_manifest.json")
    fit_end = str((manifest.get("splits") or {}).get("validation", {}).get("end") or "")
    if not fit_end or pd.Timestamp(fit_end) >= test_start:
        raise ChallengerBacktestError("Candidate fit period overlaps the sealed test.")


def _metric_values(actual: Any, predicted: Any, mape_epsilon: float, r2_minimum: int) -> dict[str, float | None]:
    result = regression_metrics(
        actual,
        predicted,
        mape_epsilon=mape_epsilon,
        r2_minimum_samples=r2_minimum,
    )
    return {
        "MAE": _float(result["MAE"]),
        "RMSE": _float(result["RMSE"]),
        "MAPE_percent": _float(result["MAPE_percent"]),
        "R2": None if result["R2"] is None else _float(result["R2"]),
        "bias": _float(result["bias"]),
        "absolute_bias": _float(abs(result["bias"])),
    }


def _no_breach(metrics: Mapping[str, Any], limits: Mapping[str, Any]) -> bool:
    values = {
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE_percent": metrics["MAPE_percent"],
        "R2": metrics["R2"],
        "absolute_bias": metrics["absolute_bias"],
    }
    return all(threshold_severity(values[name], limits[name]) == "ok" for name in REQUIRED_METRICS)


def _performance_limits(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    limits = ((calibration.get("thresholds") or {}).get("performance") or {}).get("30")
    if not isinstance(limits, Mapping) or set(limits) != set(REQUIRED_METRICS):
        raise ChallengerBacktestError("Incumbent calibration has incompatible performance.30 limits.")
    return limits


def _r2_minimum(calibration: Mapping[str, Any]) -> int:
    value = ((calibration.get("policy") or {}).get("r2_minimum_samples") or {}).get("30")
    if not isinstance(value, int) or isinstance(value, bool) or value < 2:
        raise ChallengerBacktestError("Incumbent calibration has no valid R2 minimum.")
    return value


def _hash_json(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _record_id(body: Mapping[str, Any]) -> str:
    return sha256(_canonical(body)).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_ready(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _git_state() -> dict[str, Any]:
    from .tracking import git_state

    return dict(git_state())


def _environment_manifest() -> dict[str, Any]:
    packages = {}
    for package in ("mlflow", "numpy", "pandas", "scikit-learn", "tensorflow"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {"schema_version": "wind_forecast.v2_ann_environment.v1", "packages": packages}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ChallengerBacktestError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(value), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _float(value: Any) -> float:
    return float(value)


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")
