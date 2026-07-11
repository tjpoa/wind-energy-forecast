from fastapi import FastAPI

import wind_forecast
from wind_forecast.api import create_app


def test_package_imports() -> None:
    assert wind_forecast.__name__ == "wind_forecast"


def test_api_app_can_be_created_without_local_artifacts() -> None:
    app = create_app()

    assert isinstance(app, FastAPI)
