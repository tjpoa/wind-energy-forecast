"""Build a temporary Phase 8 source fixture from the repository test builders."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types


def _builder():
    path = Path(__file__).parents[2] / "tests" / "test_incremental.py"
    spec = importlib.util.spec_from_file_location("fixture_incremental", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # The repository test builder imports pytest only for its fixture decorator;
    # keep this image-side utility independent of the full test dependency set.
    pytest_stub = types.ModuleType("pytest")
    pytest_stub.fixture = lambda *args, **kwargs: (lambda function: function)
    pytest_stub.mark = types.SimpleNamespace(
        parametrize=lambda *args, **kwargs: (lambda function: function)
    )
    sys.modules.setdefault("pytest", pytest_stub)
    spec.loader.exec_module(module)
    return module._build_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    config = _builder()(args.root)
    print(
        {
            "ren_root": str(config.ren_root),
            "era5_root": str(config.era5_root),
            "station_mapping": str(config.station_mapping),
            "v1_feature_table": str(config.v1_feature_table),
            "baseline_integrated_root": str(config.baseline_integrated_root),
            "baseline_feature_root": str(config.baseline_feature_root),
            "store_root": str(config.store_root),
            "bootstrap_start": str(config.bootstrap_start),
            "bootstrap_end": str(config.bootstrap_end),
        }
    )


if __name__ == "__main__":
    main()
