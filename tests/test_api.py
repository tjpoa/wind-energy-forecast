from datetime import date
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from wind_forecast.api import (
    ArtifactNotReadyError,
    ModelArtifactInfo,
    ModelInfoResponse,
    PredictionItem,
    PredictionResponse,
    PredictionService,
    create_app,
    get_performance_service,
    get_prediction_service,
)
from wind_forecast.performance import (
    InvalidPerformanceIntervalError,
    NoPerformanceObservationsError,
    PerformanceArtifactMissingError,
    PerformanceInterval,
    PerformanceMetrics,
    PerformanceObservation,
    PerformanceReport,
    PerformanceResultInfo,
)
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN


def _client_with_service(service) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: service
    return TestClient(app)


def _client_with_performance_service(service) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_performance_service] = lambda: service
    return TestClient(app)


def _performance_report() -> PerformanceReport:
    metrics = PerformanceMetrics(r2=0.91, mae=12.3, rmse=18.4, mape_percent=5.6)
    return PerformanceReport(
        interval=PerformanceInterval(
            requested_start_date=None,
            requested_end_date=None,
            available_start_date="2026-01-01",
            available_end_date="2026-01-02",
            returned_start_date="2026-01-01",
            returned_end_date="2026-01-02",
        ),
        observation_count=2,
        metrics=metrics,
        observations=(
            PerformanceObservation(
                date="2026-01-01",
                actual=100.0,
                predicted=90.0,
                error=-10.0,
                absolute_error=10.0,
            ),
            PerformanceObservation(
                date="2026-01-02",
                actual=120.0,
                predicted=130.0,
                error=10.0,
                absolute_error=10.0,
            ),
        ),
        result=PerformanceResultInfo(
            model_type="baseline",
            seed=42,
            test_fraction=0.2,
            dataset_version="v1",
            evaluation_start_date="2026-01-01",
            evaluation_end_date="2026-01-02",
            artifact_metrics=metrics,
        ),
    )


def test_health_endpoint_returns_ok():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_reports_artifact_and_feature_reference_status(tmp_path: Path):
    models_path = tmp_path / "models"
    models_path.mkdir()
    historical_file = tmp_path / "agg_data_ml.csv"
    pd.DataFrame(columns=[DATE_COLUMN, TARGET_COLUMN, "Feature_A", "Feature_B"]).to_csv(
        historical_file,
        index=False,
    )
    service = PredictionService(
        models_path=models_path,
        historical_file=historical_file,
    )

    info = service.model_info()

    assert info.feature_reference_exists is True
    assert info.feature_count == 2
    assert {model.target_type for model in info.models} == {"original", "log"}
    assert all(model.model_exists is False for model in info.models)


def test_predict_endpoint_uses_injected_service_with_synthetic_payload():
    class FakePredictionService:
        def predict(self, request):
            assert request.target_type == "log"
            assert request.records[0].date.isoformat() == "2026-01-01"
            assert request.records[0].features["Average_Wind_Speed"] == 4.2
            return PredictionResponse(
                target_type="log",
                model_name="ANN_Tuned",
                feature_count=3,
                predictions=[
                    PredictionItem(date="2026-01-01", prediction=123.4),
                ],
            )

    client = _client_with_service(FakePredictionService())

    response = client.post(
        "/predict",
        json={
            "target_type": "log",
            "records": [
                {
                    "Date": "2026-01-01",
                    "features": {
                        "Average_Wind_Speed": 4.2,
                        "Average_Temperature": 12.5,
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "target_type": "log",
        "model_name": "ANN_Tuned",
        "feature_count": 3,
        "predictions": [{"date": "2026-01-01", "prediction": 123.4}],
    }


def test_predict_endpoint_rejects_empty_records():
    client = TestClient(create_app())

    response = client.post("/predict", json={"records": []})

    assert response.status_code == 422


def test_predict_endpoint_requires_feature_payload():
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={"records": [{"Date": "2026-01-01"}]},
    )

    assert response.status_code == 422


def test_predict_endpoint_maps_missing_artifacts_to_service_unavailable():
    class MissingArtifactService:
        def predict(self, request):
            raise ArtifactNotReadyError("missing model artifacts")

    client = _client_with_service(MissingArtifactService())

    response = client.post(
        "/predict",
        json={
            "records": [
                {
                    "Date": "2026-01-01",
                    "features": {"Average_Wind_Speed": 4.2},
                }
            ]
        },
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "missing model artifacts"}


def test_model_info_endpoint_uses_injected_service():
    class FakeModelInfoService:
        def model_info(self):
            return ModelInfoResponse(
                models=[
                    ModelArtifactInfo(
                        target_type="log",
                        model_name="ANN_Tuned",
                        model_path="models/best_model_log_target_ANN_Tuned.keras",
                        model_exists=True,
                        scaler_x_path="models/scaler_X_log_ann.joblib",
                        scaler_x_exists=True,
                        scaler_y_path="models/scaler_y_log_ann.joblib",
                        scaler_y_exists=True,
                    )
                ],
                feature_reference_path="data/processed/agg_data_ml.csv",
                feature_reference_exists=True,
                feature_count=56,
            )

    client = _client_with_service(FakeModelInfoService())

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["feature_count"] == 56
    assert response.json()["models"][0]["target_type"] == "log"


def test_performance_endpoint_returns_typed_json_from_injected_service():
    class FakePerformanceService:
        def get_performance(self, *, start_date, end_date):
            assert start_date is None
            assert end_date is None
            return _performance_report()

    response = _client_with_performance_service(FakePerformanceService()).get(
        "/api/v1/performance"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "interval": {
            "requested_start_date": None,
            "requested_end_date": None,
            "available_start_date": "2026-01-01",
            "available_end_date": "2026-01-02",
            "returned_start_date": "2026-01-01",
            "returned_end_date": "2026-01-02",
        },
        "observation_count": 2,
        "metrics": {"r2": 0.91, "mae": 12.3, "rmse": 18.4, "mape_percent": 5.6},
        "result": {
            "model_type": "baseline",
            "seed": 42,
            "test_fraction": 0.2,
            "dataset_version": "v1",
            "evaluation_start_date": "2026-01-01",
            "evaluation_end_date": "2026-01-02",
            "artifact_metrics": {
                "r2": 0.91,
                "mae": 12.3,
                "rmse": 18.4,
                "mape_percent": 5.6,
            },
        },
        "observations": [
            {
                "date": "2026-01-01",
                "actual": 100.0,
                "predicted": 90.0,
                "error": -10.0,
                "absolute_error": 10.0,
            },
            {
                "date": "2026-01-02",
                "actual": 120.0,
                "predicted": 130.0,
                "error": 10.0,
                "absolute_error": 10.0,
            },
        ],
    }


def test_performance_endpoint_passes_date_filters_to_injected_service():
    class FakePerformanceService:
        def get_performance(self, *, start_date, end_date):
            assert start_date == date(2026, 1, 1)
            assert end_date == date(2026, 1, 2)
            return _performance_report()

    response = _client_with_performance_service(FakePerformanceService()).get(
        "/api/v1/performance?start_date=2026-01-01&end_date=2026-01-02"
    )

    assert response.status_code == 200


def test_performance_endpoint_maps_domain_errors_to_http_responses():
    class ErrorPerformanceService:
        def __init__(self, error):
            self.error = error

        def get_performance(self, *, start_date, end_date):
            raise self.error

    cases = (
        (InvalidPerformanceIntervalError("inverted interval"), 400),
        (NoPerformanceObservationsError("no observations"), 404),
        (PerformanceArtifactMissingError("missing artifact"), 503),
    )
    for error, status_code in cases:
        response = _client_with_performance_service(ErrorPerformanceService(error)).get(
            "/api/v1/performance"
        )
        assert response.status_code == status_code


def test_performance_endpoint_rejects_invalid_date_format():
    response = TestClient(create_app()).get(
        "/api/v1/performance?start_date=01-01-2026"
    )

    assert response.status_code == 422
