"""Compatibility wrapper for the stable local batch CLI."""

from wind_forecast.batch_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
