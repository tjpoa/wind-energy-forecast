"""Minimal FastAPI application for saved wind-forecast model inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .inference import (
    BEST_MODEL_LOG_NAME_FROM_NOTEBOOK,
    BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK,
    historical_positive_cap,
    load_training_feature_columns,
    load_trained_model_and_scalers,
    make_predictions,
    model_and_scaler_paths,
    prepare_data_for_prediction,
)
from .paths import models_dir, processed_data_dir, project_root
from .schemas import DATE_COLUMN, TARGET_COLUMN


ModelTarget = Literal["original", "log"]

MODEL_NAMES: dict[ModelTarget, str] = {
    "original": BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK,
    "log": BEST_MODEL_LOG_NAME_FROM_NOTEBOOK,
}


class ArtifactNotReadyError(RuntimeError):
    """Raised when local model-serving artifacts are incomplete."""


class InvalidPredictionInputError(ValueError):
    """Raised when a syntactically valid request cannot be served."""


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str


class ModelArtifactInfo(BaseModel):
    """Saved model and scaler artifact metadata for one target type."""

    target_type: ModelTarget
    model_name: str
    model_path: str
    model_exists: bool
    scaler_x_path: str | None
    scaler_x_exists: bool | None
    scaler_y_path: str | None
    scaler_y_exists: bool | None


class ModelInfoResponse(BaseModel):
    """Model-serving readiness metadata."""

    models: list[ModelArtifactInfo]
    feature_reference_path: str
    feature_reference_exists: bool
    feature_count: int | None
    feature_error: str | None = None


class PredictionRecord(BaseModel):
    """One feature-ready row for prediction."""

    model_config = ConfigDict(populate_by_name=True)

    date: Date = Field(alias=DATE_COLUMN)
    wind_production: float | None = Field(default=None, alias=TARGET_COLUMN)
    features: dict[str, float] = Field(min_length=1)


class PredictionRequest(BaseModel):
    """Prediction request for the currently selected saved model."""

    target_type: ModelTarget = "log"
    records: list[PredictionRecord] = Field(min_length=1)


class PredictionItem(BaseModel):
    """Prediction for one requested row."""

    date: str
    prediction: float


class PredictionResponse(BaseModel):
    """Prediction endpoint response."""

    target_type: ModelTarget
    model_name: str
    feature_count: int
    predictions: list[PredictionItem]


@dataclass(frozen=True)
class PredictionService:
    """Serve predictions from the repository's saved model artifacts."""

    models_path: Path
    historical_file: Path

    def model_info(self) -> ModelInfoResponse:
        """Return lightweight model metadata without loading model objects."""
        feature_count = None
        feature_error = None
        if self.historical_file.exists():
            try:
                feature_count = len(load_training_feature_columns(self.historical_file))
            except Exception as exc:  # pragma: no cover - defensive status detail
                feature_error = str(exc)

        return ModelInfoResponse(
            models=[
                self._artifact_info(target_type)
                for target_type in ("original", "log")
            ],
            feature_reference_path=self._display_path(self.historical_file),
            feature_reference_exists=self.historical_file.exists(),
            feature_count=feature_count,
            feature_error=feature_error,
        )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Prepare feature rows, load artifacts, and return model predictions."""
        self._ensure_historical_features_ready()
        self._ensure_model_artifacts_ready(request.target_type)

        model_name = MODEL_NAMES[request.target_type]
        model, scaler_x, scaler_y = load_trained_model_and_scalers(
            model_name,
            request.target_type,
            self.models_path,
        )
        if model is None:
            raise ArtifactNotReadyError(
                f"Saved {request.target_type} model artifact could not be loaded."
            )

        frame = self._prediction_frame(request.records)
        x_data, feature_columns = prepare_data_for_prediction(
            frame,
            scaler_x,
            self.historical_file,
        )
        predictions = make_predictions(
            model,
            x_data,
            scaler_y,
            is_log_target=request.target_type == "log",
            positive_inf_cap=historical_positive_cap(self.historical_file),
        )

        return PredictionResponse(
            target_type=request.target_type,
            model_name=model_name,
            feature_count=len(feature_columns),
            predictions=[
                PredictionItem(
                    date=str(frame.iloc[index][DATE_COLUMN].date()),
                    prediction=float(value),
                )
                for index, value in enumerate(predictions)
            ],
        )

    def _artifact_info(self, target_type: ModelTarget) -> ModelArtifactInfo:
        model_path, scaler_x_path, scaler_y_path = model_and_scaler_paths(
            MODEL_NAMES[target_type],
            target_type,
            self.models_path,
        )
        return ModelArtifactInfo(
            target_type=target_type,
            model_name=MODEL_NAMES[target_type],
            model_path=self._display_path(model_path),
            model_exists=model_path.exists(),
            scaler_x_path=self._display_path(scaler_x_path) if scaler_x_path else None,
            scaler_x_exists=scaler_x_path.exists() if scaler_x_path else None,
            scaler_y_path=self._display_path(scaler_y_path) if scaler_y_path else None,
            scaler_y_exists=scaler_y_path.exists() if scaler_y_path else None,
        )

    def _ensure_historical_features_ready(self) -> None:
        if not self.historical_file.exists():
            raise ArtifactNotReadyError(
                f"Feature reference file is missing: {self._display_path(self.historical_file)}"
            )

    def _ensure_model_artifacts_ready(self, target_type: ModelTarget) -> None:
        artifact = self._artifact_info(target_type)
        missing = []
        if not artifact.model_exists:
            missing.append(artifact.model_path)
        if artifact.scaler_x_exists is False:
            missing.append(artifact.scaler_x_path)
        if artifact.scaler_y_exists is False:
            missing.append(artifact.scaler_y_path)

        if missing:
            missing_paths = ", ".join(path for path in missing if path)
            raise ArtifactNotReadyError(
                f"Required {target_type} model-serving artifacts are missing: {missing_paths}"
            )

    @staticmethod
    def _prediction_frame(records: list[PredictionRecord]) -> pd.DataFrame:
        reserved_feature_keys = {DATE_COLUMN, TARGET_COLUMN}
        rows = []
        for record in records:
            reserved_present = reserved_feature_keys.intersection(record.features)
            if reserved_present:
                keys = ", ".join(sorted(reserved_present))
                raise InvalidPredictionInputError(
                    f"features must not include reserved keys: {keys}"
                )

            row = {
                DATE_COLUMN: pd.to_datetime(record.date),
                TARGET_COLUMN: 0.0 if record.wind_production is None else record.wind_production,
            }
            row.update(record.features)
            rows.append(row)

        return pd.DataFrame(rows).sort_values(DATE_COLUMN).reset_index(drop=True)

    @staticmethod
    def _display_path(path: Path) -> str:
        resolved_path = path.resolve()
        try:
            return str(resolved_path.relative_to(project_root()))
        except ValueError:
            return str(resolved_path)


def default_historical_file() -> Path:
    """Return the historical feature table used for model feature ordering."""
    return processed_data_dir() / "agg_data_ml.csv"


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Return the cached default prediction service."""
    return PredictionService(
        models_path=models_dir(),
        historical_file=default_historical_file(),
    )


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    api = FastAPI(
        title="Wind Energy Forecast API",
        version="0.1.0",
    )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.get("/model-info", response_model=ModelInfoResponse)
    def model_info(
        service: PredictionService = Depends(get_prediction_service),
    ) -> ModelInfoResponse:
        return service.model_info()

    @api.post("/predict", response_model=PredictionResponse)
    def predict(
        request: PredictionRequest,
        service: PredictionService = Depends(get_prediction_service),
    ) -> PredictionResponse:
        try:
            return service.predict(request)
        except InvalidPredictionInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ArtifactNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return api


app = create_app()
