"""Runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
import ipaddress
import math
from pathlib import Path
from urllib.parse import urlparse

from .paths import project_root
from .tracking import DEFAULT_TRACKING_URI


PERFORMANCE_ARTIFACT_DIR_ENV = "WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR"
MONITORING_STORE_ROOT_ENV = "WIND_FORECAST_MONITORING_STORE_ROOT"
DEPLOYMENT_ROOT_ENV = "WIND_FORECAST_DEPLOYMENT_ROOT"
OPERATIONAL_MODEL_BUNDLE_ENV = "WIND_FORECAST_OPERATIONAL_MODEL_BUNDLE"
OPERATIONAL_CALIBRATION_DIR_ENV = "WIND_FORECAST_OPERATIONAL_CALIBRATION_DIR"
OPERATIONAL_QUERY_TIMEOUT_ENV = "WIND_FORECAST_OPERATIONAL_QUERY_TIMEOUT_SECONDS"
OPERATIONAL_OBSERVABILITY_ROOT_ENV = "WIND_FORECAST_OPERATIONAL_OBSERVABILITY_ROOT"
MLFLOW_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
CORS_ALLOWED_ORIGINS_ENV = "WIND_FORECAST_CORS_ALLOW_ORIGINS"
TRUSTED_LOCAL_CLIENTS_ENV = "WIND_FORECAST_TRUSTED_LOCAL_CLIENTS"
DEFAULT_CORS_ALLOWED_ORIGINS = ("http://localhost:5173",)
DEFAULT_OPERATIONAL_QUERY_TIMEOUT_SECONDS = 5.0
DEFAULT_OPERATIONAL_OBSERVABILITY_ROOT = "var/local_services/operational_observability"
OPERATIONAL_ENVIRONMENT_ID_ENV = "WIND_FORECAST_OPERATIONAL_ENVIRONMENT_ID"
OPERATIONAL_PROJECTION_MIGRATOR_DSN_ENV = (
    "WIND_FORECAST_OPERATIONAL_PROJECTION_MIGRATOR_DSN"
)
OPERATIONAL_PROJECTION_WRITER_DSN_ENV = (
    "WIND_FORECAST_OPERATIONAL_PROJECTION_WRITER_DSN"
)
OPERATIONAL_PROJECTION_READER_DSN_ENV = (
    "WIND_FORECAST_OPERATIONAL_PROJECTION_READER_DSN"
)
OPERATIONAL_PROJECTION_MODE_ENV = "WIND_FORECAST_OPERATIONAL_PROJECTION_MODE"
SUPPORTED_OPERATIONAL_ENVIRONMENT_ID = "local"
DOCUMENT_SYNTHESIS_BACKEND_ENV = "WIND_FORECAST_DOCUMENT_SYNTHESIS_BACKEND"
DOCUMENT_SYNTHESIS_MODEL_ENV = "WIND_FORECAST_DOCUMENT_SYNTHESIS_MODEL"


@dataclass(frozen=True)
class WeatherAPIConfig:
    """WeatherAPI settings loaded from environment variables."""

    api_key: str | None
    location: str
    days: int
    end_date: str | None


@dataclass(frozen=True)
class PerformanceArtifactsConfig:
    """Location of one explicitly selected performance-artifact set."""

    artifact_dir: Path | None


@dataclass(frozen=True)
class MonitoringStoreConfig:
    """Location of the immutable Phase 9 monitoring store."""

    store_root: Path


@dataclass(frozen=True)
class OperationalQueryConfig:
    """Read-only operational-query sources and bounded runtime settings."""

    deployment_root: Path
    monitoring_store_root: Path
    model_bundle: Path
    calibration_dir: Path
    timeout_seconds: float
    registry_uri: str | None
    projection_mode: str
    projection_environment_id: str | None
    projection_reader_dsn: str | None


@dataclass(frozen=True)
class OperationalObservabilityConfig:
    """Location of the separate local operational-observability store."""

    store_root: Path


@dataclass(frozen=True)
class OperationalProjectionDatabaseConfig:
    """One explicitly selected database role for the local projection."""

    environment_id: str
    role: str
    dsn: str


@dataclass(frozen=True)
class CORSConfig:
    """Explicit browser origins allowed to call the FastAPI application."""

    allowed_origins: tuple[str, ...]


@dataclass(frozen=True)
class DocumentSynthesisConfig:
    backend: str
    model: str | None
    api_key: str | None


def load_document_synthesis_config() -> DocumentSynthesisConfig:
    """Load optional backend-only synthesis settings without network I/O."""
    backend = os.getenv(DOCUMENT_SYNTHESIS_BACKEND_ENV, "disabled").strip()
    if backend not in {"disabled", "openai"}:
        raise ValueError(
            f"{DOCUMENT_SYNTHESIS_BACKEND_ENV} must be 'disabled' or 'openai'."
        )
    if backend == "disabled":
        return DocumentSynthesisConfig(backend=backend, model=None, api_key=None)
    model = os.getenv(DOCUMENT_SYNTHESIS_MODEL_ENV, "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not model:
        raise ValueError(
            f"{DOCUMENT_SYNTHESIS_MODEL_ENV} must be configured for OpenAI."
        )
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be configured for OpenAI synthesis.")
    return DocumentSynthesisConfig(backend=backend, model=model, api_key=api_key)


def load_weather_api_config(
    *,
    load_dotenv_file: bool = False,
    env_file: Path | None = None,
    require_api_key: bool = True,
) -> WeatherAPIConfig:
    """Load WeatherAPI configuration from environment variables.

    Set ``load_dotenv_file=True`` to explicitly load the local ``.env`` file.
    Importing this module never loads environment files or performs network I/O.
    """
    if load_dotenv_file:
        from dotenv import load_dotenv

        load_dotenv(env_file or (project_root() / ".env"))

    api_key = os.getenv("WEATHER_API_KEY") or None
    if require_api_key and not api_key:
        raise RuntimeError(
            "Set WEATHER_API_KEY in a local .env file before calling the API."
        )

    days_raw = os.getenv("WEATHER_API_DAYS", "44")
    try:
        days = int(days_raw)
    except ValueError as exc:
        raise ValueError("WEATHER_API_DAYS must be an integer.") from exc
    if days <= 0:
        raise ValueError("WEATHER_API_DAYS must be greater than zero.")

    return WeatherAPIConfig(
        api_key=api_key,
        location=os.getenv("WEATHER_API_LOCATION", "41.8345,-7.7889"),
        days=days,
        end_date=os.getenv("WEATHER_API_END_DATE") or None,
    )


def load_performance_artifacts_config() -> PerformanceArtifactsConfig:
    """Load the explicitly configured performance-artifact directory.

    Relative paths are resolved from the repository root. The directory is not
    required to exist at configuration-load time; the performance domain
    service owns artifact-readiness validation.
    """
    raw_path = os.getenv(PERFORMANCE_ARTIFACT_DIR_ENV)
    if raw_path is None or not raw_path.strip():
        return PerformanceArtifactsConfig(artifact_dir=None)

    configured_path = Path(raw_path.strip())
    if not configured_path.is_absolute():
        configured_path = project_root() / configured_path
    return PerformanceArtifactsConfig(artifact_dir=configured_path.resolve())


def load_monitoring_store_config() -> MonitoringStoreConfig:
    """Load the monitoring store, resolving relative paths from the project root."""
    raw_path = os.getenv(MONITORING_STORE_ROOT_ENV)
    configured_path = (
        Path(raw_path.strip())
        if raw_path is not None and raw_path.strip()
        else Path("data/processed/v2/monitoring")
    )
    if not configured_path.is_absolute():
        configured_path = project_root() / configured_path
    return MonitoringStoreConfig(store_root=configured_path.resolve())


def load_operational_query_config() -> OperationalQueryConfig:
    """Load the local-only operational-query configuration.

    Paths are resolved without requiring them to exist. A Registry URI is
    enabled only when it is an HTTP(S) endpoint whose numeric host is exactly
    the IPv4 or IPv6 loopback address.
    """
    deployment_root = _resolved_project_path(
        os.getenv(DEPLOYMENT_ROOT_ENV),
        default="data/processed/v2/deployment",
    )
    monitoring_root = load_monitoring_store_config().store_root
    model_bundle = _required_project_path(OPERATIONAL_MODEL_BUNDLE_ENV)
    calibration_dir = _required_project_path(OPERATIONAL_CALIBRATION_DIR_ENV)
    timeout_raw = os.getenv(
        OPERATIONAL_QUERY_TIMEOUT_ENV,
        str(DEFAULT_OPERATIONAL_QUERY_TIMEOUT_SECONDS),
    )
    try:
        timeout_seconds = float(timeout_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{OPERATIONAL_QUERY_TIMEOUT_ENV} must be a finite number greater "
            "than zero and no greater than 5."
        ) from exc
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > DEFAULT_OPERATIONAL_QUERY_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"{OPERATIONAL_QUERY_TIMEOUT_ENV} must be a finite number greater "
            "than zero and no greater than 5."
        )

    tracking_uri = os.getenv(MLFLOW_TRACKING_URI_ENV, DEFAULT_TRACKING_URI)
    registry_uri = (
        tracking_uri.strip()
        if _is_numeric_loopback_http_uri(tracking_uri.strip())
        else None
    )
    projection_mode = os.getenv(
        OPERATIONAL_PROJECTION_MODE_ENV,
        "disabled",
    )
    if projection_mode not in {"disabled", "required"}:
        raise ValueError(
            f"{OPERATIONAL_PROJECTION_MODE_ENV} must be exactly 'disabled' "
            "or 'required'."
        )

    projection_environment_id: str | None = None
    projection_reader_dsn: str | None = None
    if projection_mode == "required":
        projection_environment_id = os.getenv(
            OPERATIONAL_ENVIRONMENT_ID_ENV,
            SUPPORTED_OPERATIONAL_ENVIRONMENT_ID,
        ).strip()
        reader_dsn = os.getenv(OPERATIONAL_PROJECTION_READER_DSN_ENV)
        projection_reader_dsn = (
            reader_dsn.strip()
            if reader_dsn is not None and reader_dsn.strip()
            else None
        )
    return OperationalQueryConfig(
        deployment_root=deployment_root,
        monitoring_store_root=monitoring_root,
        model_bundle=model_bundle,
        calibration_dir=calibration_dir,
        timeout_seconds=timeout_seconds,
        registry_uri=registry_uri,
        projection_mode=projection_mode,
        projection_environment_id=projection_environment_id,
        projection_reader_dsn=projection_reader_dsn,
    )


def load_operational_observability_config() -> OperationalObservabilityConfig:
    """Load the separate local event-store path without touching the filesystem."""
    return OperationalObservabilityConfig(
        store_root=_resolved_project_path(
            os.getenv(OPERATIONAL_OBSERVABILITY_ROOT_ENV),
            default=DEFAULT_OPERATIONAL_OBSERVABILITY_ROOT,
        )
    )


def load_operational_projection_database_config(
    role: str,
) -> OperationalProjectionDatabaseConfig:
    """Load one projection DSN without importing a driver or opening a connection."""
    dsn_variables = {
        "migrator": OPERATIONAL_PROJECTION_MIGRATOR_DSN_ENV,
        "writer": OPERATIONAL_PROJECTION_WRITER_DSN_ENV,
        "reader": OPERATIONAL_PROJECTION_READER_DSN_ENV,
    }
    if role not in dsn_variables:
        raise ValueError("Operational projection database role is unsupported.")

    environment_id = os.getenv(
        OPERATIONAL_ENVIRONMENT_ID_ENV,
        SUPPORTED_OPERATIONAL_ENVIRONMENT_ID,
    ).strip()
    if environment_id != SUPPORTED_OPERATIONAL_ENVIRONMENT_ID:
        raise ValueError(
            f"{OPERATIONAL_ENVIRONMENT_ID_ENV} must be exactly "
            f"{SUPPORTED_OPERATIONAL_ENVIRONMENT_ID!r}."
        )

    dsn_variable = dsn_variables[role]
    dsn = os.getenv(dsn_variable)
    if dsn is None or not dsn.strip():
        raise ValueError(f"{dsn_variable} must be configured.")
    return OperationalProjectionDatabaseConfig(
        environment_id=environment_id,
        role=role,
        dsn=dsn.strip(),
    )


def load_cors_config() -> CORSConfig:
    """Load validated, comma-separated CORS origins from the environment.

    When the variable is absent, local Vite development is the only allowed
    browser origin. A variable that is present but invalid fails fast instead
    of silently broadening or changing the configured access boundary.
    """
    raw_origins = os.getenv(CORS_ALLOWED_ORIGINS_ENV)
    if raw_origins is None:
        return CORSConfig(allowed_origins=DEFAULT_CORS_ALLOWED_ORIGINS)

    allowed_origins = tuple(origin.strip() for origin in raw_origins.split(","))
    if not allowed_origins or any(not origin for origin in allowed_origins):
        raise ValueError(
            f"{CORS_ALLOWED_ORIGINS_ENV} must contain one or more comma-separated "
            "origins."
        )

    for origin in allowed_origins:
        _validate_cors_origin(origin)

    return CORSConfig(allowed_origins=allowed_origins)


def _validate_cors_origin(origin: str) -> None:
    """Reject values that cannot be used as an exact browser origin."""
    try:
        parsed = urlparse(origin)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(_cors_origin_error()) from exc

    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or "*" in parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in origin)
    ):
        raise ValueError(_cors_origin_error())


def _resolved_project_path(raw_path: str | None, *, default: str) -> Path:
    configured_path = (
        Path(raw_path.strip())
        if raw_path is not None and raw_path.strip()
        else Path(default)
    )
    if not configured_path.is_absolute():
        configured_path = project_root() / configured_path
    return configured_path.resolve()


def _required_project_path(variable: str) -> Path:
    raw_path = os.getenv(variable)
    if raw_path is None or not raw_path.strip():
        raise ValueError(f"{variable} must be configured.")
    configured_path = Path(raw_path.strip())
    if not configured_path.is_absolute():
        configured_path = project_root() / configured_path
    return configured_path.resolve()


def _is_numeric_loopback_http_uri(value: str) -> bool:
    if not value or any(character.isspace() for character in value):
        return False
    try:
        parsed = urlparse(value)
        _ = parsed.port
        address = ipaddress.ip_address(parsed.hostname or "")
    except (ValueError, TypeError):
        return False
    return (
        parsed.scheme in {"http", "https"}
        and address
        in {
            ipaddress.ip_address("127.0.0.1"),
            ipaddress.ip_address("::1"),
        }
        and parsed.username is None
        and parsed.password is None
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )


def _cors_origin_error() -> str:
    """Return the shared validation message for configured CORS origins."""
    return (
        f"{CORS_ALLOWED_ORIGINS_ENV} entries must be exact http(s) origins "
        "without wildcards, credentials, paths, query strings, or fragments."
    )
