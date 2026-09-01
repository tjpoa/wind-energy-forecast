import argparse
import json
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from wind_forecast.artifacts import build_reproduction_bundle, validate_release
from wind_forecast.manifests import sha256_file
from wind_forecast.paths import processed_data_dir, project_root
from wind_forecast.release_catalog import (
    ReleaseCatalogError,
    load_release_catalog,
    require_release_approved,
)
from wind_forecast.tracking import (
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_TRACKING_URI,
    TrackingConfig,
    _load_mlflow,
    configure_tracking,
)


CATALOG_PATH = project_root() / "artifacts" / "catalog.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic candidate bundle.")
    parser.add_argument("--release", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--registered-model", default=DEFAULT_REGISTERED_MODEL_NAME)
    parser.add_argument("--dataset", type=Path, default=processed_data_dir() / "agg_data_ml.csv")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=project_root() / "dist")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_release(args.release)
    try:
        catalog = load_release_catalog(args.catalog)
        release_entry = require_release_approved(catalog, args.release)
    except ReleaseCatalogError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    config = TrackingConfig(
        tracking_uri=args.tracking_uri,
        registered_model_name=args.registered_model,
    )
    mlflow = _load_mlflow()
    configure_tracking(config)
    client = mlflow.MlflowClient()
    candidate = client.get_model_version_by_alias(args.registered_model, "candidate")
    if str(candidate.version) != str(args.model_version):
        raise SystemExit(
            f"ERROR: candidate is version {candidate.version}, not {args.model_version}."
        )
    candidate_tags = dict(getattr(candidate, "tags", {}))
    required_tags = {
        "validation_status",
        "dataset_version",
        "dataset_sha256",
        "feature_schema_sha256",
        "git_sha",
        "source_run_id",
        "target_contract",
        "model_sha256",
    }
    missing_tags = sorted(required_tags.difference(candidate_tags))
    if missing_tags:
        raise SystemExit(f"ERROR: candidate tags are incomplete: {missing_tags}")
    if candidate_tags["validation_status"] != "passed":
        raise SystemExit("ERROR: candidate has not passed validation.")
    run_id = str(candidate.run_id)
    if candidate_tags["source_run_id"] != run_id:
        raise SystemExit("ERROR: candidate source run tag does not match model version.")
    if candidate_tags["target_contract"] != "original":
        raise SystemExit("ERROR: only the original-target model contract can be bundled.")
    if not re.fullmatch(r"[0-9a-f]{40,64}", candidate_tags["git_sha"]):
        raise SystemExit("ERROR: candidate git_sha tag is invalid.")
    if sha256_file(args.dataset) != candidate_tags["dataset_sha256"]:
        raise SystemExit("ERROR: selected dataset does not match the candidate checksum.")
    source_run = client.get_run(run_id)
    run_params = dict(source_run.data.params)
    run_tag_checks = {
        "dataset_version": "dataset_version",
        "dataset_sha256": "dataset_sha256",
        "feature_schema_sha256": "feature_schema_sha256",
        "git_sha": "git_sha",
        "target_contract": "target_contract",
    }
    run_mismatches = [
        tag
        for tag, param in run_tag_checks.items()
        if candidate_tags[tag] != run_params.get(param)
    ]
    if run_mismatches:
        raise SystemExit(
            f"ERROR: candidate tags do not match source run lineage: {run_mismatches}"
        )

    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        dataset_manifest_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="manifests/dataset_manifest.json",
                dst_path=str(temporary_path / "preflight"),
            )
        )
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        manifest_checks = {
            "dataset_version": candidate_tags["dataset_version"],
            "sha256": candidate_tags["dataset_sha256"],
            "feature_schema_sha256": candidate_tags["feature_schema_sha256"],
            "target": "Wind_Production",
        }
        mismatches = [
            key for key, expected in manifest_checks.items()
            if dataset_manifest.get(key) != expected
        ]
        if mismatches:
            raise SystemExit(
                f"ERROR: candidate dataset manifest mismatches: {mismatches}"
            )
        model_root = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{args.registered_model}/{args.model_version}",
                dst_path=str(temporary_path / "downloaded-model"),
            )
        )
        files: dict[str, Path] = {"data/agg_data_ml.csv": args.dataset}
        for path in sorted(model_root.rglob("*")):
            if path.is_file():
                files[f"mlflow-model/{path.relative_to(model_root).as_posix()}"] = path
        run_artifacts = {
            "baseline/model.joblib": "baseline/model.joblib",
            "baseline/metrics.json": "baseline/metrics.json",
            "baseline/predictions.csv": "baseline/predictions.csv",
            "baseline/run_summary.json": "baseline/run_summary.json",
            "evaluation/actual_vs_predicted.png": "evaluation/actual_vs_predicted.png",
            "manifests/dataset_manifest.json": "manifests/dataset_manifest.json",
            "manifests/model_manifest.json": "manifests/model_manifest.json",
            "environment/environment.json": "environment/environment.json",
            "validation/validation_sample.csv": "validation/validation_sample.csv",
        }
        for bundle_name, artifact_path in run_artifacts.items():
            if artifact_path == "manifests/dataset_manifest.json":
                files[bundle_name] = dataset_manifest_path
            else:
                files[bundle_name] = Path(
                    mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path=artifact_path,
                        dst_path=str(temporary_path / "run-artifacts"),
                    )
                )
        model_manifest = json.loads(
            files["manifests/model_manifest.json"].read_text(encoding="utf-8")
        )
        model_checks = {
            "dataset_version": candidate_tags["dataset_version"],
            "dataset_sha256": candidate_tags["dataset_sha256"],
            "feature_schema_sha256": candidate_tags["feature_schema_sha256"],
            "target_contract": candidate_tags["target_contract"],
            "model_sha256": candidate_tags["model_sha256"],
        }
        model_mismatches = [
            key for key, expected in model_checks.items()
            if model_manifest.get(key) != expected
        ]
        if model_mismatches:
            raise SystemExit(
                f"ERROR: candidate model manifest mismatches: {model_mismatches}"
            )
        if sha256_file(files["baseline/model.joblib"]) != candidate_tags["model_sha256"]:
            raise SystemExit("ERROR: baseline model artifact checksum changed.")
        result = build_reproduction_bundle(
            release=args.release,
            model_name=args.registered_model,
            model_version=str(args.model_version),
            run_id=run_id,
            git_sha=candidate_tags["git_sha"],
            files=files,
            output_dir=args.output_dir,
            redistribution=release_entry["redistribution"],
        )
    print(f"Bundle: {result.archive_path}")
    print(f"SHA-256: {result.sha256}")
    print("Record this SHA-256 in artifacts/catalog.json before publication.")
    if not release_entry["redistribution"].get("approved", False):
        print("PUBLICATION BLOCKED: redistribution approval is unresolved.")


if __name__ == "__main__":
    main()
