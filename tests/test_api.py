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
    get_prediction_service,
)
from wind_forecast.schemas import DATE_COLUMN, TARGET_COLUMN


def _client_with_service(service) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: service
    return TestClient(app)


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
