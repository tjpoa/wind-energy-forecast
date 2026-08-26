"""CI entry point for the tracked deployment-artifact safety gate."""

from __future__ import annotations

from validate_azure_workflow import main


if __name__ == "__main__":
    raise SystemExit(main(["tracked"]))
