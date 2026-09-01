import argparse
from collections.abc import Sequence
from pathlib import Path

from wind_forecast.artifacts import (
    extract_bundle,
    fetch_release_bundle,
    materialize_training_data,
    verify_bundle,
)
from wind_forecast.paths import processed_data_dir, project_root
from wind_forecast.release_catalog import (
    ReleaseCatalogError,
    load_release_catalog,
    require_release_approved,
    validate_release_contract_binding,
)


CATALOG_PATH = project_root() / "artifacts" / "catalog.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and verify a release bundle.")
    parser.add_argument("--release", required=True)
    parser.add_argument("--repository", default="tjpoa/wind-energy-forecast")
    parser.add_argument("--output-root", type=Path, default=project_root() / "artifacts" / "releases")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        catalog = load_release_catalog(args.catalog)
        release_entry = require_release_approved(catalog, args.release)
        validate_release_contract_binding(
            release_entry, repository_root=project_root(), require_release_provenance=False
        )
    except ReleaseCatalogError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    expected_sha256 = release_entry.get("bundle_sha256")
    if not expected_sha256:
        raise SystemExit(
            "ERROR: approved release has no bundle_sha256 in the tracked catalog."
        )
    release_root = args.output_root / args.release
    archive, checksum = fetch_release_bundle(
        release=args.release,
        repository=args.repository,
        destination=release_root,
        expected_sha256=expected_sha256,
    )
    manifest = verify_bundle(archive, checksum)
    redistribution = manifest.get("redistribution")
    if not isinstance(redistribution, dict):
        raise SystemExit("ERROR: bundle is missing redistribution contract linkage.")
    if redistribution.get("source_contract_sha256") != release_entry["source_contract"]["sha256"]:
        raise SystemExit("ERROR: bundle source contract hash is not catalog-bound.")
    if redistribution.get("processed_contract_sha256") != release_entry["processed_contract"]["sha256"]:
        raise SystemExit("ERROR: bundle processed contract hash is not catalog-bound.")
    extracted = extract_bundle(archive, release_root / "extracted")
    if args.materialize:
        materialize_training_data(
            extracted, processed_data_dir() / "agg_data_ml.csv"
        )
    print(f"Verified release {manifest['release']} at {release_root}")


if __name__ == "__main__":
    main()
