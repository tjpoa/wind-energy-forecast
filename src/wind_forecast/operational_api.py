"""Local-only HTTP adapter for the read-only operational query layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import ipaddress
import json
import math
from typing import Any, Callable, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .config import load_operational_query_config
from .operational_query import OperationalQueryService
from .operational_query_models import (
    AnswerStatus,
    AuthorizationContext,
    CONTRACT_VERSION,
    EvidenceState,
    OperationalAnswer,
    OperationalFailure,
    Pagination,
    QueryKind,
    Selector,
)


MAX_OPERATIONAL_QUERY_BODY_BYTES = 64 * 1024
LOCAL_OPERATOR_PRINCIPAL = "local-api-operator"

STATUS_CODE_BY_ANSWER_STATUS = {
    AnswerStatus.ANSWERED: 200,
    AnswerStatus.EMPTY: 200,
    AnswerStatus.REFUSED: 400,
    AnswerStatus.UNAUTHORIZED: 403,
    AnswerStatus.NOT_FOUND: 404,
    AnswerStatus.UNAVAILABLE: 503,
    AnswerStatus.CORRUPT: 503,
    AnswerStatus.CONFLICT: 503,
    AnswerStatus.TIMEOUT: 504,
}


class OperationalHttpRequest(BaseModel):
    """Public request fields; transport metadata is server-controlled."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    contract_version: Literal[CONTRACT_VERSION]
    query_kind: QueryKind
    selector: Selector
    window_days: Literal[30, 90] | None = None
    pagination: Pagination | None = None

    @field_validator("query_kind", mode="before")
    @classmethod
    def parse_query_kind(cls, value: Any) -> QueryKind:
        return value if isinstance(value, QueryKind) else QueryKind(value)


def _allow_local_operator(
    context: AuthorizationContext,
    query_kind: QueryKind,
) -> bool:
    """Allow the closed query allowlist only for this process-local principal."""
    return (
        context.principal == LOCAL_OPERATOR_PRINCIPAL
        and context.trusted_local
        and query_kind in QueryKind
    )


@dataclass(frozen=True)
class _LocalOnlyMlflowRegistryClient:
    """Bounded GET-only Registry adapter with redirects disabled."""

    client: Any

    def get_model_version_by_alias(
        self,
        name: str,
        alias: str,
        *,
        timeout_seconds: float,
    ) -> Any:
        if (
            not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        from mlflow.entities.model_registry import ModelVersion
        from mlflow.protos.model_registry_pb2 import GetModelVersionByAlias
        from mlflow.store.model_registry.rest_store import RestStore
        from mlflow.utils.proto_json_utils import message_to_json, parse_dict
        from mlflow.utils.rest_utils import (
            http_request,
            verify_rest_response,
        )

        registry = self.client._get_registry_client()
        store = registry.store
        if not isinstance(store, RestStore):
            from .retraining_deployment import (
                RetrainingDeploymentUnavailableError,
            )

            raise RetrainingDeploymentUnavailableError(
                "The Registry backend does not support bounded REST reads."
            )
        request_body = message_to_json(
            GetModelVersionByAlias(name=name, alias=alias)
        )
        endpoint, method = store._get_endpoint_from_method(
            GetModelVersionByAlias
        )
        if method != "GET":
            from .retraining_deployment import (
                RetrainingDeploymentUnavailableError,
            )

            raise RetrainingDeploymentUnavailableError(
                "The Registry alias read is not a GET operation."
            )
        response = http_request(
            host_creds=store.get_host_creds(),
            endpoint=endpoint,
            method=method,
            params=json.loads(request_body),
            timeout=float(timeout_seconds),
            retry_timeout_seconds=float(timeout_seconds),
            max_retries=0,
            allow_redirects=False,
            proxies={"http": None, "https": None, "all": None},
        )
        verify_rest_response(response, endpoint)
        response_proto = GetModelVersionByAlias.Response()
        parse_dict(json.loads(response.text), response_proto)
        return ModelVersion.from_proto(response_proto.model_version)


def _registry_client(registry_uri: str | None) -> Any | None:
    """Create a bounded REST adapter without contacting the Registry."""
    if registry_uri is None:
        return None
    try:
        import mlflow

        client = mlflow.MlflowClient(
            tracking_uri=registry_uri,
            registry_uri=registry_uri,
        )
        return _LocalOnlyMlflowRegistryClient(client)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_operational_query_service() -> OperationalQueryService:
    """Return the configured, read-only operational query service."""
    config = load_operational_query_config()
    return OperationalQueryService(
        deployment_root=config.deployment_root,
        monitoring_store_root=config.monitoring_store_root,
        max_deadline_seconds=config.timeout_seconds,
        authorization_policy=_allow_local_operator,
        registry_client=_registry_client(config.registry_uri),
        registry_timeout_seconds=(
            config.timeout_seconds if config.registry_uri is not None else None
        ),
    )


def get_operational_query_service_factory() -> Callable[
    [], OperationalQueryService
]:
    """Return the lazy service factory without loading runtime configuration."""
    return get_operational_query_service


def _trusted_loopback(request: Request) -> bool:
    """Trust only an exact numeric loopback address from the ASGI socket."""
    if request.client is None:
        return False
    try:
        address = ipaddress.ip_address(request.client.host)
    except ValueError:
        return False
    return address in {
        ipaddress.ip_address("127.0.0.1"),
        ipaddress.ip_address("::1"),
    }


def _query_kind(value: Any) -> QueryKind | None:
    if not isinstance(value, dict):
        return None
    try:
        return QueryKind(value.get("query_kind"))
    except (TypeError, ValueError):
        return None


def _transport_refusal(
    *,
    correlation_id: str,
    served_at_utc: datetime,
    query_kind: QueryKind | None = None,
    code: str = "invalid_operational_query",
    message: str = "The operational query request body is invalid.",
) -> OperationalAnswer:
    return OperationalAnswer(
        query_kind=query_kind,
        status=AnswerStatus.REFUSED,
        summary=None,
        facts=(),
        evidence=(),
        limitations=(),
        failure=OperationalFailure(
            code=code,
            message=message,
            retryable=False,
            evidence_state=EvidenceState.UNSUPPORTED,
        ),
        served_at_utc=served_at_utc,
        correlation_id=correlation_id,
    )


def _service_unavailable(
    *,
    correlation_id: str,
    served_at_utc: datetime,
    query_kind: QueryKind,
) -> OperationalAnswer:
    return OperationalAnswer(
        query_kind=query_kind,
        status=AnswerStatus.UNAVAILABLE,
        summary=None,
        facts=(),
        evidence=(),
        limitations=(),
        failure=OperationalFailure(
            code="operational_query_service_unavailable",
            message="The operational query service is unavailable.",
            retryable=True,
            evidence_state=EvidenceState.UNAVAILABLE,
        ),
        served_at_utc=served_at_utc,
        correlation_id=correlation_id,
    )


async def _limited_json_body(request: Request) -> tuple[Any | None, str | None]:
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        return None, "invalid_content_type"

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            return None, "invalid_content_length"
        if declared_length < 0:
            return None, "invalid_content_length"
        if declared_length > MAX_OPERATIONAL_QUERY_BODY_BYTES:
            return None, "operational_query_body_too_large"

    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_OPERATIONAL_QUERY_BODY_BYTES:
                return None, "operational_query_body_too_large"
            body.extend(chunk)
        decoded = bytes(body).decode("utf-8")
        return json.loads(decoded), None
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        RuntimeError,
    ):
        return None, "invalid_json_body"


def _json_response(answer: OperationalAnswer) -> JSONResponse:
    return JSONResponse(
        status_code=STATUS_CODE_BY_ANSWER_STATUS[answer.status],
        content=jsonable_encoder(answer),
    )


operational_router = APIRouter()


@operational_router.post(
    "/api/v1/operational-query",
    response_model=OperationalAnswer,
    responses={
        400: {"model": OperationalAnswer, "description": "Request refused."},
        403: {"model": OperationalAnswer, "description": "Operator unauthorized."},
        404: {"model": OperationalAnswer, "description": "Evidence not found."},
        503: {
            "model": OperationalAnswer,
            "description": "Evidence unavailable, corrupt, or conflicting.",
        },
        504: {"model": OperationalAnswer, "description": "Query deadline expired."},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": OperationalHttpRequest.model_json_schema()
                }
            },
        }
    },
)
async def operational_query(
    request: Request,
    service_factory: Callable[
        [], OperationalQueryService
    ] = Depends(get_operational_query_service_factory),
) -> JSONResponse:
    """Serve one bounded, local-only operational query."""
    requested_at_utc = datetime.now(timezone.utc)
    correlation_id = uuid4().hex
    raw_body, body_error = await _limited_json_body(request)
    if body_error is not None:
        message = (
            "The operational query request body exceeds the 64 KiB limit."
            if body_error == "operational_query_body_too_large"
            else "The operational query request body is invalid."
        )
        return _json_response(
            _transport_refusal(
                correlation_id=correlation_id,
                served_at_utc=requested_at_utc,
                code=body_error,
                message=message,
            )
        )

    try:
        public_request = OperationalHttpRequest.model_validate(
            raw_body,
            strict=True,
        )
    except (ValidationError, TypeError, ValueError):
        return _json_response(
            _transport_refusal(
                correlation_id=correlation_id,
                served_at_utc=requested_at_utc,
                query_kind=_query_kind(raw_body),
            )
        )

    try:
        service = service_factory()
    except Exception:
        return _json_response(
            _service_unavailable(
                correlation_id=correlation_id,
                served_at_utc=requested_at_utc,
                query_kind=public_request.query_kind,
            )
        )

    query = public_request.model_dump(mode="python")
    query.update(
        requested_at_utc=requested_at_utc,
        correlation_id=correlation_id,
        deadline=requested_at_utc
        + timedelta(seconds=service.max_deadline_seconds),
    )
    authorization = AuthorizationContext(
        principal=LOCAL_OPERATOR_PRINCIPAL,
        trusted_local=_trusted_loopback(request),
    )
    try:
        answer = service.answer(query, authorization)
    except Exception:
        answer = _service_unavailable(
            correlation_id=correlation_id,
            served_at_utc=requested_at_utc,
            query_kind=public_request.query_kind,
        )
    return _json_response(answer)


__all__ = [
    "MAX_OPERATIONAL_QUERY_BODY_BYTES",
    "OperationalHttpRequest",
    "STATUS_CODE_BY_ANSWER_STATUS",
    "get_operational_query_service",
    "get_operational_query_service_factory",
    "operational_router",
]
