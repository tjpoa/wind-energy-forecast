"""Build the tracked, deterministic synthetic dashboard demo bundle."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


BUNDLE_SCHEMA = "wind_forecast.demo_bundle.v1"
BUNDLE_VERSION = "demo-v1"
SEED = 2026
GENERATED_AT = "2026-08-24T00:00:00Z"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo") / "v1",
        help="Directory to create (default: demo/v1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the exact output directory when it already exists.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to replace existing demo bundle: {output_dir}. "
            "Pass --overwrite to rebuild it."
        )

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _build_bundle(temporary_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    print(f"Built {BUNDLE_VERSION} at {output_dir}")
    print(f"Files: {len(list(output_dir.rglob('*')))}")
    return 0


def _build_bundle(root: Path) -> None:
    performance = _build_performance(root / "performance")
    pipeline_path = root / "pipeline" / "run.json"
    _write_json(
        pipeline_path,
        {
            "schema_version": "wind_forecast.demo_pipeline_run.v1",
            "bundle_version": BUNDLE_VERSION,
            "run_id": "demo-pipeline-20260824",
            "status": "succeeded",
            "started_at_utc": "2026-08-24T00:00:00Z",
            "finished_at_utc": "2026-08-24T00:00:02Z",
            "through_date": performance["end_date"],
            "source_mode": "deterministic_synthetic",
            "credentials_required": False,
            "network_requests": False,
            "outputs": [
                "performance/predictions.csv",
                "monitoring/reporting/state/current.json",
                "deployment/evidence.json",
            ],
        },
    )

    model_sha256 = _sha256_text("demo-synthetic-model-v1")
    dataset_sha256 = _sha256_text("demo-synthetic-dataset-v1")
    policy_sha256 = _sha256_text("demo-synthetic-policy-v1")
    monitoring_ids = _build_monitoring(
        root / "monitoring",
        performance=performance,
        model_sha256=model_sha256,
        dataset_sha256=dataset_sha256,
        policy_sha256=policy_sha256,
        pipeline_sha256=_sha256_file(pipeline_path),
    )
    _write_json(
        root / "deployment" / "evidence.json",
        {
            "schema_version": "wind_forecast.demo_deployment_evidence.v1",
            "bundle_version": BUNDLE_VERSION,
            "deployment_id": "demo-deployment-v1",
            "deployment_state_id": "demo-deployment-state-v1",
            "generation": 1,
            "status": "verified",
            "serving_mode": "read_only_dashboard_projection",
            "registered_model_name": "demo-synthetic-wind-forecast",
            "model_version": "1",
            "model_sha256": model_sha256,
            "dataset_sha256": dataset_sha256,
            "transformation_version": "demo-synthetic-v1",
            "monitoring_report_id": monitoring_ids["report_id"],
            "source_pipeline_run_id": "demo-pipeline-20260824",
            "registry_required": False,
            "mlflow_required": False,
        },
    )
    _write_text(
        root / "README.md",
        (
            "# Deterministic synthetic demo bundle\n\n"
            "This is `demo-v1`, a clearly labelled synthetic evidence set for the\n"
            "local dashboard. It is not REN data, CDS/ERA5-Land data, a trained\n"
            "model release, or a production deployment. Values and identities are\n"
            "generated deterministically by `scripts/build_demo_bundle.py` with\n"
            "seed 2026. No credentials, network calls, MLflow state, or ignored\n"
            "files are required.\n\n"
            "The bundle contains a tiny historical-performance artifact, one\n"
            "verified retrospective monitoring report, deployment attribution, and\n"
            "a succeeded synthetic pipeline-run receipt.\n"
        ),
    )

    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    _write_json(
        root / "manifest.json",
        {
            "schema_version": BUNDLE_SCHEMA,
            "bundle_version": BUNDLE_VERSION,
            "generated_at_utc": GENERATED_AT,
            "generator": "scripts/build_demo_bundle.py",
            "seed": SEED,
            "evidence_type": "deterministic_synthetic",
            "source": {
                "provider": None,
                "license": "synthetic-original",
                "redistributable": True,
                "credentials_required": False,
                "network_requests": False,
            },
            "claims": {
                "historical_production": False,
                "production_model": False,
                "live_monitoring": False,
                "cloud_deployment": False,
            },
            "files": files,
        },
    )


def _build_performance(root: Path) -> dict[str, Any]:
    actual = np.array(
        [1020, 1080, 1115, 1060, 1140, 1195, 1160, 1210, 1255, 1185, 1230, 1280, 1265, 1310],
        dtype=float,
    )
    errors = np.array([18, -12, 26, -20, 9, 31, -18, 14, -24, 16, -8, 22, -15, 11], dtype=float)
    predicted = actual + errors
    dates = [date(2026, 8, 1) + timedelta(days=index) for index in range(len(actual))]
    metrics = {
        "R2": float(r2_score(actual, predicted)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE (%)": float(mean_absolute_percentage_error(actual, predicted) * 100),
    }
    predictions_path = root / "predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Date", "Actual_Wind_Production", "Predicted_Wind_Production"])
        for current_date, observed, forecast in zip(dates, actual, predicted, strict=True):
            writer.writerow([current_date.isoformat(), f"{observed:.2f}", f"{forecast:.2f}"])
    _write_json(root / "metrics.json", metrics)
    _write_json(
        root / "run_summary.json",
        {
            "schema_version": "wind_forecast.demo_performance_run.v1",
            "model_type": "DeterministicSyntheticBaseline",
            "seed": SEED,
            "test_fraction": 0.2,
            "dataset_version": BUNDLE_VERSION,
            "test_start_date": dates[0].isoformat(),
            "test_end_date": dates[-1].isoformat(),
            "test_row_count": len(dates),
            "metrics": metrics,
            "source_mode": "deterministic_synthetic",
            "network_requests": False,
            "mlflow_run_id": None,
        },
    )
    return {
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "row_count": len(dates),
        "metrics": metrics,
        "bias": float(np.mean(errors)),
    }


def _build_monitoring(
    root: Path,
    *,
    performance: Mapping[str, Any],
    model_sha256: str,
    dataset_sha256: str,
    policy_sha256: str,
    pipeline_sha256: str,
) -> dict[str, str]:
    reporting_root = root / "reporting"
    reference_csv = "Date,wind_speed_current,Reference_Prediction,Actual\n2026-07-31,8.2,1180.00,1190.00\n"
    reference_body = {
        "schema_version": "wind_forecast.monitoring_reference.v1",
        "dataset_sha256": dataset_sha256,
        "dataset_version": BUNDLE_VERSION,
        "model_sha256": model_sha256,
        "feature_schema_sha256": _sha256_text("wind_speed_current"),
        "transformation_version": "demo-synthetic-v1",
        "period": {"start": "2026-01-01", "end": "2026-07-31"},
        "row_count": 1,
        "feature_names": ["wind_speed_current"],
        "target": "Actual",
        "reference_prediction_column": "Reference_Prediction",
        "reference_csv_sha256": _sha256_text(reference_csv),
        "prediction_role": "in_sample_distribution_reference_only",
        "performance_claim": False,
        "calibration_subject": {"kind": "deterministic_synthetic_demo"},
    }
    reference = _record("monitoring_reference", "reference_id", reference_body)
    reference_dir = reporting_root / "references" / reference["reference_id"]
    _write_text(reference_dir / "reference.csv", reference_csv)
    _write_json(reference_dir / "manifest.json", {**reference, "reference_path": "reference.csv"})

    limits_upper = {"warning": 100.0, "critical": 160.0, "direction": "upper"}
    limits_bias = {"warning": 25.0, "critical": 50.0, "direction": "upper"}
    limits_lower = {"warning": 0.7, "critical": 0.4, "direction": "lower"}
    drift_limits = {
        "normalized_wasserstein": {"warning": 0.10, "critical": 0.20, "direction": "upper"},
        "ks_statistic": {"warning": 0.08, "critical": 0.16, "direction": "upper"},
    }
    policy = {
        "schema_version": "wind_forecast.monitoring_policy.v1",
        "reference_start": "2026-01-01",
        "reference_end": "2026-07-31",
        "windows_days": [30, 90],
        "warning_quantile": 0.95,
        "critical_quantile": 0.99,
        "minimum_samples": {"30": 7, "90": 30},
        "r2_minimum_samples": {"30": 7, "90": 30},
        "mape_epsilon_quantile": 0.05,
        "alert_persistence_distinct_dates": 2,
        "source_objective_days": 5,
        "source_late_days": 7,
        "hard_quality_tolerance": 0,
        "overrides": {},
    }
    backtest_summary = {
        "schema_version": "wind_forecast.monitoring_backtest_summary.v1",
        "windows": {
            "30": {"accepted_backtest_windows": 1, "minimum_samples": 7},
            "90": {"accepted_backtest_windows": 1, "minimum_samples": 30},
        },
        "performance": {"30": {"MAE": 1, "RMSE": 1, "R2": 1}},
    }
    backtest_bytes = _json_bytes(backtest_summary)
    thresholds = {
        "performance": {
            "30": {
                "MAE": limits_upper,
                "RMSE": limits_upper,
                "absolute_bias": limits_bias,
                "MAPE_percent": limits_upper,
                "R2": limits_lower,
            },
            "90": {},
        },
        "feature_drift": {
            "wind_speed_current": {"30": {"global": drift_limits}},
        },
    }
    calibration_body = {
        "schema_version": "wind_forecast.monitoring_calibration.v1",
        "reference_id": reference["reference_id"],
        "reference_manifest_sha256": _sha256_file(reference_dir / "manifest.json"),
        "policy": policy,
        "policy_sha256": policy_sha256,
        "backtest_stride_days": 7,
        "mape_epsilon": 1.0,
        "mape_epsilon_role": "reference_positive_target_quantile",
        "thresholds": thresholds,
        "backtest_summary": backtest_summary,
        "backtest_summary_sha256": hashlib.sha256(backtest_bytes).hexdigest(),
        "safeguards": {
            "ledger_prediction_write": False,
            "model_write": False,
            "training": False,
            "network_requests": False,
        },
    }
    calibration = _record("monitoring_calibration", "calibration_id", calibration_body)
    calibration_dir = reporting_root / "calibrations" / calibration["calibration_id"]
    _write_json(
        calibration_dir / "calibration.json",
        {**calibration, "reference_dir": f"references/{reference['reference_id']}"},
    )
    _write_text(calibration_dir / "backtest_summary.json", backtest_bytes.decode("utf-8"))

    metrics = performance["metrics"]
    bias = performance["bias"]
    report_body = {
        "schema_version": "wind_forecast.monitoring_report.v2",
        "run_id": "demo-report-20260824",
        "created_at_utc": "2026-08-24T00:00:02Z",
        "through_date": performance["end_date"],
        "model_era": {
            "model_era_id": "demo-model-era-v1",
            "association_kind": "active_deployment",
            "deployment_id": "demo-deployment-v1",
            "deployment_state_id": "demo-deployment-state-v1",
            "deployment_generation": 1,
            "registered_model_name": "demo-synthetic-wind-forecast",
            "model_version": "1",
            "cutoffs": {"monitoring_evaluation_cutoff": performance["end_date"]},
            "pins": {
                "model_sha256": model_sha256,
                "dataset_sha256": dataset_sha256,
                "transformation_version": "demo-synthetic-v1",
            },
        },
        "source_batch": {
            "run_id": "demo-pipeline-20260824",
            "status": "succeeded",
            "manifest_path": "pipeline/run.json",
            "manifest_sha256": pipeline_sha256,
        },
        "reference": {
            "reference_id": reference["reference_id"],
            "calibration_id": calibration["calibration_id"],
            "policy_sha256": policy_sha256,
        },
        "config": policy,
        "quality": {
            "status": "available",
            "issues": [],
            "freshness": {
                "common_validated_watermark": performance["end_date"],
                "unresolved_late_dates": [],
            },
        },
        "windows": {
            "30": {
                "status": "available",
                "sample_count": performance["row_count"],
                "minimum_samples": 7,
                "calendar_start": performance["start_date"],
                "calendar_end": performance["end_date"],
                "coverage_ratio": 1.0,
                "coverage_severity": "ok",
                "performance": {
                    "status": "available",
                    "metrics": {
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "bias": bias,
                        "MAPE_percent": metrics["MAPE (%)"],
                        "R2": metrics["R2"],
                        "R2_status": "available",
                    },
                    "severity": {
                        "MAE": "ok",
                        "RMSE": "ok",
                        "bias": "ok",
                        "MAPE_percent": "ok",
                        "R2": "ok",
                    },
                },
                "feature_drift": {
                    "wind_speed_current": {
                        "global": {
                            "normalized_wasserstein": 0.14,
                            "ks_statistic": 0.07,
                        }
                    }
                },
            },
            "90": {
                "status": "insufficient_data",
                "sample_count": performance["row_count"],
                "minimum_samples": 30,
                "calendar_start": None,
                "calendar_end": None,
                "coverage_ratio": None,
                "coverage_severity": None,
                "performance": {},
                "feature_drift": {},
            },
        },
        "breaches": [],
        "persistence": {},
        "alert_events": [],
        "active_alerts": {},
        "lineage": {"prediction_ids": []},
        "safeguards": {
            "predictions_unchanged": True,
            "models_unchanged": True,
            "as_issued_primary": True,
            "restatements_alerting": False,
            "training": False,
            "network_requests": False,
        },
    }
    report = _record("monitoring_report", "report_id", report_body)
    report_dir = reporting_root / "reports" / report["report_id"]
    _write_json(report_dir / "report.json", report)
    _write_text(
        report_dir / "report.md",
        "# Synthetic retrospective monitoring report\n\n"
        f"- Report: `{report['report_id']}`\n"
        f"- Source pipeline: `demo-pipeline-20260824`\n"
        f"- Through date: `{performance['end_date']}`\n"
        "- Evidence type: `deterministic_synthetic`\n"
        "- Network requests: `false`\n",
    )

    plan = {
        "status": "planned",
        "through_date": performance["end_date"],
        "source_run_id": "demo-pipeline-20260824",
        "source_status": "succeeded",
        "calibration_id": calibration["calibration_id"],
        "ledger_available": False,
        "quality_available": True,
        "model_era_id": "demo-model-era-v1",
        "deployment_id": "demo-deployment-v1",
        "model_version": "1",
    }
    run_dir = reporting_root / "runs" / "demo-report-20260824"
    _write_json(
        run_dir / "request.json",
        {
            "schema_version": "wind_forecast.monitoring_report_request.v2",
            "run_id": "demo-report-20260824",
            "requested_at_utc": "2026-08-24T00:00:02Z",
            "plan": plan,
        },
    )
    _write_json(
        run_dir / "result.json",
        {
            "status": "succeeded",
            "run_id": "demo-report-20260824",
            "report_id": report["report_id"],
            "active_alert_count": 0,
            "plan": plan,
        },
    )
    _write_json(
        reporting_root / "state" / "current.json",
        {
            "schema_version": "wind_forecast.monitoring_report_state.v2",
            "generation": 1,
            "updated_at_utc": "2026-08-24T00:00:02Z",
            "latest_report_id": report["report_id"],
            "latest_through_date": performance["end_date"],
            "model_era_id": "demo-model-era-v1",
            "deployment_id": "demo-deployment-v1",
            "active_alerts": {},
            "rules": {},
        },
    )
    return {"report_id": report["report_id"]}


def _record(kind: str, id_field: str, body: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(body, sort_keys=True))
    identifier = hashlib.sha256(
        kind.encode("utf-8") + b":" + _canonical_bytes(payload)
    ).hexdigest()
    return {id_field: identifier, **payload}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, _json_bytes(value))


def _write_text(path: Path, value: str) -> None:
    _write_bytes(path, value.encode("utf-8"))


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
