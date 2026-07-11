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
    import json

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    release_entry = catalog.get("releases", {}).get(args.release)
    expected_sha256 = None if release_entry is None else release_entry.get("bundle_sha256")
    if not expected_sha256:
        raise SystemExit(
            "ERROR: release is absent from the tracked catalog or has no approved bundle_sha256."
        )
    if release_entry["redistribution"].get("approved") is not True:
        raise SystemExit("ERROR: release redistribution is not approved in the catalog.")
    release_root = args.output_root / args.release
    archive, checksum = fetch_release_bundle(
        release=args.release,
        repository=args.repository,
        destination=release_root,
        expected_sha256=expected_sha256,
    )
    manifest = verify_bundle(archive, checksum)
    extracted = extract_bundle(archive, release_root / "extracted")
    if args.materialize:
        materialize_training_data(
            extracted, processed_data_dir() / "agg_data_ml.csv"
        )
    print(f"Verified release {manifest['release']} at {release_root}")


if __name__ == "__main__":
    main()
