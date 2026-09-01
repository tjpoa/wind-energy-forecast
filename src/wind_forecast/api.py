"""Minimal FastAPI application for saved wind-forecast model inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import load_cors_config
from .inference import (
    V1_MODEL_NAME,
    historical_positive_cap,
    load_training_feature_columns,
    load_trained_model_and_scalers,
    make_predictions,
    model_and_scaler_paths,
    prepare_data_for_prediction,
)
from .paths import models_dir, processed_data_dir, project_root
from .monitoring_projection import (
    MonitoringProjectionError,
    MonitoringProjectionService,
    MonitoringRunNotFoundError,
)
from .operational_api import operational_router
from .performance import (
    InvalidPerformanceIntervalError,
    NoPerformanceObservationsError,
    PerformanceArtifactError,
    PerformanceMetrics,
    PerformanceReport,
    PerformanceService,
)
from .schemas import DATE_COLUMN, TARGET_COLUMN
from .v1_contracts import V1ContractError, load_processed_contract, load_serving_contract


ModelTarget = Literal["original", "log"]

MODEL_NAMES: dict[ModelTarget, str] = {
    "original": V1_MODEL_NAME,
    "log": V1_MODEL_NAME,
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


class PerformanceIntervalResponse(BaseModel):
    """Requested, available, and returned performance date bounds."""

    requested_start_date: str | None
    requested_end_date: str | None
    available_start_date: str
    available_end_date: str
    returned_start_date: str
    returned_end_date: str


class PerformanceMetricsResponse(BaseModel):
    """Regression metrics calculated for the returned observations."""

    r2: float | None
    mae: float
    rmse: float
    mape_percent: float


class PerformanceResultInfoResponse(BaseModel):
    """Basic non-sensitive provenance for the evaluated results."""

    model_type: str
    seed: int
    test_fraction: float
    dataset_version: str | None
    evaluation_start_date: str
    evaluation_end_date: str
    artifact_metrics: PerformanceMetricsResponse | None


class PerformanceObservationResponse(BaseModel):
    """Actual and predicted production for one historical date."""

    date: str
    actual: float
    predicted: float
    error: float
    absolute_error: float


class PerformanceResponse(BaseModel):
    """Historical performance returned by the performance endpoint."""

    interval: PerformanceIntervalResponse
    observation_count: int
    metrics: PerformanceMetricsResponse
    result: PerformanceResultInfoResponse | None
    observations: list[PerformanceObservationResponse]


class MonitoringRunFailureResponse(BaseModel):
    """Sanitized reporting-run failure metadata."""

    failed_at_utc: str | None
    error_type: str | None
    message: str


class MonitoringRunSummaryResponse(BaseModel):
    """One reporting attempt in the monitoring history."""

    run_id: str
    attempted_at_utc: str
    through_date: str | None
    source_pipeline_run_id: str | None
    source_pipeline_status: str | None
    status: Literal["succeeded", "failed", "in_progress"]
    report_id: str | None
    active_alert_count: int
    failure: MonitoringRunFailureResponse | None


class MonitoringAlertResponse(BaseModel):
    """One sanitized immutable alert transition."""

    alert_event_id: str
    rule_id: str
    through_date: str
    event_type: Literal["opened", "escalated", "resolved"]
    severity: Literal["not_available", "ok", "warning", "critical"]
    previous_alert_event_id: str | None


class MonitoringMetricResponse(BaseModel):
    """One moving metric and its sealed-test v2 thresholds."""

    metric: str
    label: str
    value: float | None
    status: str
    severity: Literal["not_available", "ok", "warning", "critical"]
    warning: float
    critical: float
    direction: Literal["upper", "lower"]


class MonitoringDriftResponse(BaseModel):
    """One ranked feature-drift result."""

    feature: str
    comparator: Literal["global", "seasonal"]
    detector: Literal["normalized_wasserstein", "ks_statistic"]
    value: float
    severity: Literal["not_available", "ok", "warning", "critical"]
    threshold: float
    threshold_ratio: float


class MonitoringWindowResponse(BaseModel):
    """One 30- or 90-day monitoring window."""

    window_days: int
    status: Literal["available", "insufficient_data", "not_available"]
    sample_count: int
    minimum_samples: int | None
    calendar_start: str | None
    calendar_end: str | None
    coverage_ratio: float | None
    coverage_severity: Literal[
        "not_available", "ok", "warning", "critical"
    ] | None
    performance: list[MonitoringMetricResponse]
    top_drift: list[MonitoringDriftResponse]


class MonitoringFreshnessResponse(BaseModel):
    """D+5/D+7 freshness derived from verified source evidence."""

    status: Literal["within_objective", "behind_objective", "late", "unknown"]
    watermark_date: str | None
    objective_at: str | None
    late_at: str | None
    timezone: Literal["Europe/Lisbon"]
    objective_days: int
    late_days: int


class MonitoringModelResponse(BaseModel):
    """Report-scoped model identity."""

    snapshot_id: str | None
    checksum: str
    model_type: str | None
    dataset_version: str | None
    dataset_checksum: str
    transformation_version: str
    status: Literal["selected_not_promoted", "champion"]


class MonitoringModelEraResponse(BaseModel):
    """Deployment attribution for one monitoring report."""

    model_era_id: str | None = None
    association_kind: Literal[
        "active_deployment", "bootstrap_adopted", "legacy_unassociated"
    ]
    deployment_id: str | None = None
    deployment_state_id: str | None = None
    deployment_generation: int | None = None
    registered_model_name: str | None = None
    model_version: str | None = None
    cutoffs: dict[str, str] | None = None
    pins: dict[str, str] | None = None


class MonitoringSourcePipelineResponse(BaseModel):
    """Source pipeline linked to one report."""

    run_id: str
    status: str


class MonitoringReportResponse(BaseModel):
    """Sanitized projection of one immutable Phase 9 report."""

    report_id: str
    reporting_run_id: str
    created_at_utc: str
    as_of_date: str
    source_pipeline: MonitoringSourcePipelineResponse
    freshness: MonitoringFreshnessResponse
    model: MonitoringModelResponse
    model_era: MonitoringModelEraResponse
    windows: dict[str, MonitoringWindowResponse]
    active_alerts: list[MonitoringAlertResponse]
    target_scale: Literal["sum_of_15_minute_MW_observations"]


class MonitoringLatestResponse(BaseModel):
    """Latest report and latest reporting attempt."""

    state: Literal["empty", "available"]
    mode: Literal["retrospective_historical_batch_not_real_time"]
    served_at_utc: str
    message: str | None
    latest_attempt: MonitoringRunSummaryResponse | None
    report: MonitoringReportResponse | None


class MonitoringRunPageResponse(BaseModel):
    """Paginated reporting attempts."""

    items: list[MonitoringRunSummaryResponse]
    total: int
    limit: int
    offset: int


class MonitoringAlertPageResponse(BaseModel):
    """Paginated immutable alert transitions."""

    items: list[MonitoringAlertResponse]
    total: int
    limit: int
    offset: int


class MonitoringHistoryResponse(BaseModel):
    """Monitoring run and alert history."""

    state: Literal["empty", "available"]
    mode: Literal["retrospective_historical_batch_not_real_time"]
    runs: MonitoringRunPageResponse
    alerts: MonitoringAlertPageResponse


class MonitoringRunResponse(BaseModel):
    """One reporting attempt and its optional report."""

    state: Literal["available"]
    mode: Literal["retrospective_historical_batch_not_real_time"]
    run: MonitoringRunSummaryResponse
    report: MonitoringReportResponse | None


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
        try:
            model_path, scaler_x_path, scaler_y_path = model_and_scaler_paths(
                MODEL_NAMES[target_type], target_type, self.models_path
            )
        except V1ContractError:
            # Keep the status endpoint useful when the serving contract itself
            # is invalid; prediction remains fail-closed in _ensure_*.
            model_path = self.models_path / (
                f"best_model_{target_type}_target_{MODEL_NAMES[target_type]}.keras"
            )
            scaler_x_path = self.models_path / f"scaler_X_{target_type}_ann.joblib"
            scaler_y_path = self.models_path / f"scaler_y_{target_type}_ann.joblib"
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
        if self.models_path.resolve() == models_dir().resolve():
            try:
                contract = load_processed_contract(verify_dataset=True)
                expected = (project_root() / contract["dataset_path"]).resolve()
                if self.historical_file.resolve() != expected:
                    raise V1ContractError(
                        "historical_file is not the dataset declared by v1_processed_contract."
                    )
            except V1ContractError as exc:
                raise ArtifactNotReadyError(
                    f"v1 processed contract validation failed: {exc}"
                ) from exc

    def _ensure_model_artifacts_ready(self, target_type: ModelTarget) -> None:
        if self.models_path.resolve() == models_dir().resolve():
            try:
                load_serving_contract(verify_files=True)
            except V1ContractError as exc:
                raise ArtifactNotReadyError(
                    f"v1 serving contract validation failed: {exc}"
                ) from exc
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


@lru_cache(maxsize=1)
def get_performance_service() -> PerformanceService:
    """Return the cached performance service configured for this process."""
    return PerformanceService.from_config()


@lru_cache(maxsize=1)
def get_monitoring_service() -> MonitoringProjectionService:
    """Return the cached read-only Phase 9 projection service."""
    return MonitoringProjectionService.from_config()


def _performance_metrics_response(
    metrics: PerformanceMetrics,
) -> PerformanceMetricsResponse:
    """Map domain metrics to the stable HTTP response contract."""
    return PerformanceMetricsResponse(
        r2=metrics.r2,
        mae=metrics.mae,
        rmse=metrics.rmse,
        mape_percent=metrics.mape_percent,
    )


def _performance_response(report: PerformanceReport) -> PerformanceResponse:
    """Map a domain performance report without exposing artifact internals."""
    result = report.result
    return PerformanceResponse(
        interval=PerformanceIntervalResponse(
            requested_start_date=report.interval.requested_start_date,
            requested_end_date=report.interval.requested_end_date,
            available_start_date=report.interval.available_start_date,
            available_end_date=report.interval.available_end_date,
            returned_start_date=report.interval.returned_start_date,
            returned_end_date=report.interval.returned_end_date,
        ),
        observation_count=report.observation_count,
        metrics=_performance_metrics_response(report.metrics),
        result=(
            None
            if result is None
            else PerformanceResultInfoResponse(
                model_type=result.model_type,
                seed=result.seed,
                test_fraction=result.test_fraction,
                dataset_version=result.dataset_version,
                evaluation_start_date=result.evaluation_start_date,
                evaluation_end_date=result.evaluation_end_date,
                artifact_metrics=(
                    None
                    if result.artifact_metrics is None
                    else _performance_metrics_response(result.artifact_metrics)
                ),
            )
        ),
        observations=[
            PerformanceObservationResponse(
                date=observation.date,
                actual=observation.actual,
                predicted=observation.predicted,
                error=observation.error,
                absolute_error=observation.absolute_error,
            )
            for observation in report.observations
        ],
    )


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    api = FastAPI(
        title="Wind Energy Forecast API",
        version="0.1.0",
    )
    cors_config = load_cors_config()
    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    api.include_router(operational_router)

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

    @api.get("/api/v1/performance", response_model=PerformanceResponse)
    def performance(
        start_date: Date | None = None,
        end_date: Date | None = None,
        service: PerformanceService = Depends(get_performance_service),
    ) -> PerformanceResponse:
        """Return read-only historical prediction performance."""
        try:
            report = service.get_performance(
                start_date=start_date,
                end_date=end_date,
            )
            return _performance_response(report)
        except InvalidPerformanceIntervalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except NoPerformanceObservationsError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PerformanceArtifactError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @api.get(
        "/api/v1/monitoring/latest",
        response_model=MonitoringLatestResponse,
    )
    def monitoring_latest(
        service: MonitoringProjectionService = Depends(get_monitoring_service),
    ) -> MonitoringLatestResponse:
        """Return the latest retrospective historical monitoring projection."""
        try:
            return MonitoringLatestResponse.model_validate(service.latest())
        except (MonitoringProjectionError, ValidationError) as exc:
            raise HTTPException(
                status_code=503,
                detail="Historical monitoring evidence is unavailable or corrupt.",
            ) from exc

    @api.get(
        "/api/v1/monitoring/history",
        response_model=MonitoringHistoryResponse,
    )
    def monitoring_history(
        run_limit: int = Query(default=20, ge=1, le=100),
        run_offset: int = Query(default=0, ge=0),
        alert_limit: int = Query(default=50, ge=1, le=200),
        alert_offset: int = Query(default=0, ge=0),
        service: MonitoringProjectionService = Depends(get_monitoring_service),
    ) -> MonitoringHistoryResponse:
        """Return paginated reporting attempts and immutable alert events."""
        try:
            return MonitoringHistoryResponse.model_validate(
                service.history(
                    run_limit=run_limit,
                    run_offset=run_offset,
                    alert_limit=alert_limit,
                    alert_offset=alert_offset,
                )
            )
        except (MonitoringProjectionError, ValidationError) as exc:
            raise HTTPException(
                status_code=503,
                detail="Historical monitoring evidence is unavailable or corrupt.",
            ) from exc

    @api.get(
        "/api/v1/monitoring/runs/{run_id}",
        response_model=MonitoringRunResponse,
    )
    def monitoring_run(
        run_id: str,
        service: MonitoringProjectionService = Depends(get_monitoring_service),
    ) -> MonitoringRunResponse:
        """Return one reporting attempt, with report detail when successful."""
        try:
            return MonitoringRunResponse.model_validate(service.run(run_id))
        except MonitoringRunNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Monitoring reporting run was not found.",
            ) from exc
        except (MonitoringProjectionError, ValidationError) as exc:
            raise HTTPException(
                status_code=503,
                detail="Historical monitoring evidence is unavailable or corrupt.",
            ) from exc

    return api


app = create_app()
