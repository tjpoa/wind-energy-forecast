"""Start the Compose demo stack and verify its browser-facing dashboard flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_NAME = "wind-energy-forecast-demo-smoke"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    args = parser.parse_args()

    environment = os.environ.copy()
    performance_dir = (ROOT / "demo" / "v1" / "performance").resolve()
    environment.update(
        {
            "WIND_FORECAST_PERFORMANCE_ARTIFACT_HOST_DIR": performance_dir.as_posix(),
            "WIND_FORECAST_CORS_ALLOW_ORIGINS": "http://localhost:5173",
            "VITE_API_BASE_URL": "http://localhost:8000",
        }
    )
    compose = ["docker", "compose", "--project-name", args.project_name]
    try:
        _run(compose + ["up", "--build", "--detach", "--wait"], environment)
        _verify_stack(timeout_seconds=args.timeout_seconds)
        print("PASS: demo Compose stack served health, monitoring, performance, and frontend evidence.")
        return 0
    finally:
        result = subprocess.run(
            compose + ["down", "--remove-orphans"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)


def _verify_stack(*, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    health = _wait_for_json("http://localhost:8000/health", deadline)
    if health.get("status") != "ok":
        raise RuntimeError(f"Unexpected health response: {health}")

    latest = _wait_for_json("http://localhost:8000/api/v1/monitoring/latest", deadline)
    report = latest.get("report")
    if (
        latest.get("state") != "available"
        or not isinstance(report, dict)
        or report.get("source_pipeline", {}).get("run_id") != "demo-pipeline-20260824"
        or report.get("model_era", {}).get("deployment_id") != "demo-deployment-v1"
        or report.get("windows", {}).get("30", {}).get("status") != "available"
    ):
        raise RuntimeError(f"Unexpected latest monitoring response: {latest}")

    history = _wait_for_json("http://localhost:8000/api/v1/monitoring/history", deadline)
    runs = history.get("runs", {}).get("items", [])
    if history.get("state") != "available" or not any(
        run.get("status") == "succeeded" for run in runs
    ):
        raise RuntimeError(f"Unexpected monitoring history response: {history}")

    performance = _wait_for_json("http://localhost:8000/api/v1/performance", deadline)
    if (
        performance.get("observation_count") != 14
        or performance.get("result", {}).get("dataset_version") != "demo-v1"
        or len(performance.get("observations", [])) != 14
    ):
        raise RuntimeError(f"Unexpected performance response: {performance}")

    frontend = _wait_for_text("http://localhost:5173", deadline)
    if '<div id="root"></div>' not in frontend or 'type="module"' not in frontend:
        raise RuntimeError("Frontend did not serve the dashboard shell.")


def _wait_for_json(url: str, deadline: float) -> dict[str, Any]:
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected an object response from {url}.")
            return payload
        except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def _wait_for_text(url: str, deadline: float) -> str:
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=3) as response:
                return response.read().decode("utf-8")
        except (OSError, URLError, UnicodeError) as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def _run(command: list[str], environment: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
