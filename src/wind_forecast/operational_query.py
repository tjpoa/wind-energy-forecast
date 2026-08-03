"""Deterministic read-only query layer over verified operational evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

from pydantic import ValidationError

from wind_forecast.deployment_runtime import (
    DeploymentRuntimeConflictError,
    DeploymentRuntimeError,
    DeploymentRuntimeNotInitializedError,
    DeploymentRuntimeUnavailableError,
    same_model_era,
    verify_active_model_era,
)
from wind_forecast.monitoring import MonitoringError, load_model_era
from wind_forecast.monitoring_reporting import (
    MonitoringReportingConflictError,
    MonitoringReportingError,
    MonitoringReportingUnavailableError,
    load_active_alerts,
    load_alert_history,
    load_monitoring_calibration,
    load_monitoring_report,
    load_monitoring_report_state,
    load_reporting_attempt,
    resolve_report_model_era,
)
from wind_forecast.monitoring_statistics import threshold_severity
from wind_forecast.operational_query_models import (
    AnswerStatus,
    AuthorizationContext,
    CONTRACT_VERSION,
    DateIntervalSelector,
    EvidenceCitation,
    EvidenceDomain,
    EvidenceState,
    ExactIdSelector,
    GroundedFact,
    LatestSelector,
    OPERATIONAL_MODE,
    OperationalAnswer,
    OperationalFailure,
    OperationalQuery,
    Pagination,
    QueryKind,
)
from wind_forecast.operational_projection_reader import (
    OperationalProjectionTimeoutError,
    OperationalProjectionUnavailableError,
    ProjectedAlerts,
    ProjectedEvidence,
    ProjectedReport,
    ProjectedRow,
)


TARGET_SCALE = "sum_of_15_minute_MW_observations"
MAX_PUBLIC_FACTS = 1_000
MAX_PUBLIC_COLLECTION_ITEMS = 1_000
MAX_PUBLIC_MAPPING_ITEMS = 256
MAX_PUBLIC_STRING_LENGTH = 2_048
AuthorizationPolicy = Callable[[AuthorizationContext, QueryKind], bool]


class _QueryConflict(RuntimeError):
    pass


class _QueryTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class _Fact:
    name: str
    value: Any
    unit_or_scale: str
    as_of: str
    citation_keys: tuple[str, ...]


@dataclass(frozen=True)
class _Citation:
    domain: EvidenceDomain
    source_kind: str
    schema_version: str
    record_id: str
    sha256: str
    effective_at: str
    observed_at_utc: datetime | None = None


@dataclass(frozen=True)
class _Result:
    status: AnswerStatus
    facts: tuple[_Fact, ...] = ()
    citations: tuple[tuple[str, _Citation], ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationalQueryService:
    """Answer the closed operational allowlist without mutating its sources."""

    deployment_root: Path
    monitoring_store_root: Path
    max_deadline_seconds: float
    authorization_policy: AuthorizationPolicy | None = None
    model_bundle: Path | None = None
    calibration_dir: Path | None = None
    registry_client: Any | None = None
    registry_timeout_seconds: float | None = None
    projection_reader: Any | None = None
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def __post_init__(self) -> None:
        object.__setattr__(self, "deployment_root", Path(self.deployment_root))
        object.__setattr__(
            self, "monitoring_store_root", Path(self.monitoring_store_root)
        )
        if self.model_bundle is not None:
            object.__setattr__(self, "model_bundle", Path(self.model_bundle))
        if self.calibration_dir is not None:
            object.__setattr__(self, "calibration_dir", Path(self.calibration_dir))
        if (
            not math.isfinite(self.max_deadline_seconds)
            or self.max_deadline_seconds <= 0
        ):
            raise ValueError("max_deadline_seconds must be finite and positive")
        if self.registry_client is not None and (
            self.registry_timeout_seconds is None
            or not math.isfinite(self.registry_timeout_seconds)
            or self.registry_timeout_seconds <= 0
        ):
            raise ValueError(
                "A Registry client requires a finite positive timeout declaration."
            )

    def answer(
        self,
        value: Any,
        authorization: AuthorizationContext | Mapping[str, Any] | None,
    ) -> OperationalAnswer:
        """Validate an untrusted envelope and always return a sanitized answer."""
        if isinstance(value, OperationalQuery):
            raw = value.model_dump()
        elif isinstance(value, Mapping):
            try:
                raw = dict(value)
            except Exception:
                raw = {}
        else:
            raw = {}
        correlation_id = _safe_correlation(raw.get("correlation_id"))
        raw_kind = raw.get("query_kind")
        try:
            query_kind = QueryKind(raw_kind)
        except (TypeError, ValueError):
            return self._failure_answer(
                query_kind=None,
                correlation_id=correlation_id,
                status=AnswerStatus.REFUSED,
                code="unsupported_query_kind",
                message="The requested operational question is not supported.",
                retryable=False,
                evidence_state=EvidenceState.UNSUPPORTED,
            )
        try:
            query = (
                value
                if isinstance(value, OperationalQuery)
                else OperationalQuery.model_validate(raw, strict=True)
            )
        except (ValidationError, TypeError, ValueError):
            return self._failure_answer(
                query_kind=query_kind,
                correlation_id=correlation_id,
                status=AnswerStatus.REFUSED,
                code="invalid_operational_query",
                message="The operational query envelope or selector is invalid.",
                retryable=False,
                evidence_state=EvidenceState.UNSUPPORTED,
            )
        if (
            query.deadline - query.requested_at_utc
        ).total_seconds() > self.max_deadline_seconds:
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.REFUSED,
                code="deadline_exceeds_service_limit",
                message="The requested deadline exceeds the service limit.",
                retryable=False,
                evidence_state=EvidenceState.UNSUPPORTED,
            )
        try:
            context = (
                authorization
                if isinstance(authorization, AuthorizationContext)
                else AuthorizationContext.model_validate(authorization, strict=True)
            )
        except (ValidationError, TypeError, ValueError):
            context = None
        policy = self.authorization_policy
        authorized = False
        if context is not None and policy is not None:
            try:
                authorized = context.trusted_local and bool(
                    policy(context, query.query_kind)
                )
            except Exception:
                authorized = False
        if not authorized:
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.UNAUTHORIZED,
                code="operator_not_authorized",
                message="The local operator is not authorized for this query.",
                retryable=False,
                evidence_state=EvidenceState.UNAUTHORIZED,
            )
        try:
            self._check_deadline(query)
            result = self._dispatch(query)
            self._check_deadline(query)
            return self._render(query, result)
        except _QueryTimeout:
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.TIMEOUT,
                code="operational_query_timeout",
                message="The operational query deadline expired.",
                retryable=True,
                evidence_state=EvidenceState.TIMEOUT,
            )
        except OperationalProjectionTimeoutError:
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.TIMEOUT,
                code="operational_query_timeout",
                message="The operational query deadline expired.",
                retryable=True,
                evidence_state=EvidenceState.TIMEOUT,
            )
        except OperationalProjectionUnavailableError:
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.UNAVAILABLE,
                code="required_projection_unavailable",
                message="The required operational projection is unavailable.",
                retryable=True,
                evidence_state=EvidenceState.UNAVAILABLE,
            )
        except DeploymentRuntimeNotInitializedError:
            if query.query_kind in {
                QueryKind.ACTIVE_DEPLOYMENT,
                QueryKind.ACTIVE_MODEL_METADATA,
                QueryKind.OPERATIONAL_SUMMARY,
            }:
                return self._absence_answer(query, AnswerStatus.EMPTY)
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.UNAVAILABLE,
                code="required_evidence_unavailable",
                message="Required verified operational evidence is unavailable.",
                retryable=True,
                evidence_state=EvidenceState.UNAVAILABLE,
            )
        except FileNotFoundError:
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.UNAVAILABLE,
                code="required_evidence_unavailable",
                message="Required verified operational evidence is unavailable.",
                retryable=True,
                evidence_state=EvidenceState.UNAVAILABLE,
            )
        except (
            DeploymentRuntimeConflictError,
            MonitoringReportingConflictError,
            _QueryConflict,
        ):
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.CONFLICT,
                code="operational_evidence_conflict",
                message="Verified operational evidence sources disagree.",
                retryable=False,
                evidence_state=EvidenceState.CONFLICT,
            )
        except DeploymentRuntimeUnavailableError:
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.UNAVAILABLE,
                code="required_dependency_unavailable",
                message="A required operational dependency is unavailable.",
                retryable=True,
                evidence_state=EvidenceState.UNAVAILABLE,
            )
        except (MonitoringReportingUnavailableError, OSError):
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.UNAVAILABLE,
                code="required_evidence_unavailable",
                message="Required verified operational evidence is unavailable.",
                retryable=True,
                evidence_state=EvidenceState.UNAVAILABLE,
            )
        except (DeploymentRuntimeError, MonitoringReportingError, MonitoringError):
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.CORRUPT,
                code="operational_evidence_corrupt",
                message="Stored operational evidence failed verification.",
                retryable=False,
                evidence_state=EvidenceState.CORRUPT,
            )
        except (KeyError, TypeError, ValueError):
            return self._failure_answer(
                query_kind=query.query_kind,
                correlation_id=query.correlation_id,
                status=AnswerStatus.CORRUPT,
                code="operational_evidence_invalid",
                message="Verified operational evidence has an invalid contract.",
                retryable=False,
                evidence_state=EvidenceState.CORRUPT,
            )

    def _dispatch(self, query: OperationalQuery) -> _Result:
        handlers = {
            QueryKind.OPERATIONAL_SUMMARY: self._operational_summary,
            QueryKind.ACTIVE_DEPLOYMENT: self._active_deployment,
            QueryKind.DATA_QUALITY: self._data_quality,
            QueryKind.MONITORING_PERFORMANCE: self._monitoring_performance,
            QueryKind.MONITORING_DRIFT: self._monitoring_drift,
            QueryKind.MONITORING_ALERTS: self._monitoring_alerts,
            QueryKind.ACTIVE_MODEL_METADATA: self._active_model_metadata,
            QueryKind.REPORTING_RUN: self._reporting_run,
        }
        return handlers[query.query_kind](query)

    def _verify_era(
        self, query: OperationalQuery, *, include_runtime_metadata: bool = False
    ) -> dict[str, Any]:
        self._check_deadline(query)
        if self.registry_client is None or self.registry_timeout_seconds is None:
            raise DeploymentRuntimeUnavailableError(
                "A timeout-bounded Registry client is required."
            )
        remaining = (query.deadline - self._now()).total_seconds()
        effective_timeout = min(self.registry_timeout_seconds, remaining)
        if effective_timeout <= 0:
            raise _QueryTimeout
        era = verify_active_model_era(
            self.deployment_root,
            self.model_bundle,
            calibration_dir=self.calibration_dir,
            client=self.registry_client,
            include_runtime_metadata=include_runtime_metadata,
            registry_timeout_seconds=effective_timeout,
        )
        self._check_deadline(query)
        return era

    def _operational_summary(self, query: OperationalQuery) -> _Result:
        era = self._verify_era(query)
        facts, citations = self._deployment_facts(era, query, metadata=False)
        state = load_monitoring_report_state(self.monitoring_store_root)
        self._check_deadline(query)
        limitations = [
            "Historical batch evidence is not real-time or a future forecast.",
            f"Target scale is {TARGET_SCALE}.",
        ]
        if state is None:
            return _Result(
                AnswerStatus.EMPTY,
                limitations=("No verified monitoring report is available.",),
            )
        else:
            report = self._load_report_id(str(state["latest_report_id"]), query)
            if (
                report.get("through_date") != state.get("latest_through_date")
                or report.get("report_id") != state.get("latest_report_id")
            ):
                raise _QueryConflict
            active_before = load_active_alerts(self.monitoring_store_root)
            self._check_deadline(query)
            if (
                active_before != (state.get("active_alerts") or {})
                or active_before != (report.get("active_alerts") or {})
            ):
                raise _QueryConflict
            state_key, state_citation = _report_state_citation(
                state, self._now()
            )
            report_key, report_citation = _report_citation(report)
            active_key, active_citation = _active_alert_citation(
                active_before, self._now()
            )
            citations.extend(
                [
                    (state_key, state_citation),
                    (report_key, report_citation),
                    (active_key, active_citation),
                ]
            )
            through = str(report["through_date"])
            quality = report.get("quality") or {}
            freshness = quality.get("freshness") or {}
            unresolved_late_dates = freshness.get("unresolved_late_dates") or []
            facts.extend(
                [
                    _Fact(
                        "monitoring.latest_report_id",
                        str(report["report_id"]),
                        "not_applicable",
                        through,
                        (state_key, report_key),
                    ),
                    _Fact(
                        "monitoring.freshness",
                        {
                            "watermark_date": freshness.get(
                                "common_validated_watermark"
                            ),
                            "watermark_age_days": freshness.get(
                                "watermark_age_days"
                            ),
                            "objective_missed": freshness.get(
                                "objective_missed"
                            ),
                            "unresolved_late_date_count": len(
                                unresolved_late_dates
                            ),
                        },
                        "days_and_state",
                        through,
                        (report_key,),
                    ),
                    _Fact(
                        "monitoring.active_alert_count",
                        len(active_before),
                        "count",
                        through,
                        (report_key, active_key),
                    ),
                ]
            )
            self._assert_latest_report_unchanged(query, report)
            self._assert_active_alerts_unchanged(query, active_before)
        verified_again = self._verify_era(query)
        if not same_model_era(era, verified_again):
            raise _QueryConflict
        return _Result(
            AnswerStatus.ANSWERED,
            tuple(facts),
            tuple(citations),
            tuple(limitations),
        )

    def _active_deployment(self, query: OperationalQuery) -> _Result:
        era = self._verify_era(query)
        facts, citations = self._deployment_facts(era, query, metadata=False)
        verified_again = self._verify_era(query)
        if not same_model_era(era, verified_again):
            raise _QueryConflict
        return _Result(
            AnswerStatus.ANSWERED,
            tuple(facts),
            tuple(citations),
            (
                "Registry aliases describe the binding observed at query time.",
                "Candidate aliases are outside runtime selection.",
            ),
        )

    def _active_model_metadata(self, query: OperationalQuery) -> _Result:
        era = self._verify_era(query, include_runtime_metadata=True)
        facts, citations = self._deployment_facts(era, query, metadata=True)
        verified_again = self._verify_era(query)
        if not same_model_era(era, verified_again):
            raise _QueryConflict
        return _Result(
            AnswerStatus.ANSWERED,
            tuple(facts),
            tuple(citations),
            (f"Target scale is {TARGET_SCALE}.",),
        )

    def _deployment_facts(
        self,
        era: Mapping[str, Any],
        query: OperationalQuery,
        *,
        metadata: bool,
    ) -> tuple[list[_Fact], list[tuple[str, _Citation]]]:
        observed = self._now()
        deployment = era["deployment"]
        registry = era["registry"]
        deployment_key = "deployment"
        registry_key = "registry"
        deployment_citation = _Citation(
            EvidenceDomain.DEPLOYMENT,
            "verify_active_model_era",
            str(era["schema_version"]),
            str(deployment["deployment_id"]),
            str(era["model_era_id"]),
            str(deployment["generation"]),
        )
        binding_digest = _digest(
            {
                "model_era_id": era["model_era_id"],
                "registry": {
                    "registered_model_name": registry["registered_model_name"],
                    "model_version": registry["model_version"],
                },
                "expected_aliases": era["expected_aliases"],
            }
        )
        registry_citation = _Citation(
            EvidenceDomain.REGISTRY,
            "verified_registry_alias_binding",
            str(era["schema_version"]),
            str(era["model_era_id"]),
            binding_digest,
            str(deployment["generation"]),
            observed,
        )
        citations = [
            (deployment_key, deployment_citation),
            (registry_key, registry_citation),
        ]
        as_of = str(deployment["generation"])
        facts = [
            _Fact(
                "deployment.deployment_id",
                str(deployment["deployment_id"]),
                "not_applicable",
                as_of,
                (deployment_key,),
            ),
            _Fact(
                "deployment.model_era_id",
                str(era["model_era_id"]),
                "not_applicable",
                as_of,
                (deployment_key,),
            ),
            _Fact(
                "deployment.generation",
                int(deployment["generation"]),
                "generation",
                as_of,
                (deployment_key,),
            ),
            _Fact(
                "deployment.registered_model_name",
                str(registry["registered_model_name"]),
                "not_applicable",
                as_of,
                (registry_key,),
            ),
            _Fact(
                "deployment.model_version",
                str(registry["model_version"]),
                "not_applicable",
                as_of,
                (registry_key,),
            ),
            _Fact(
                "deployment.expected_aliases",
                {
                    str(key): None if value is None else str(value)
                    for key, value in era["expected_aliases"].items()
                    if key in {"champion", "stable"}
                },
                "not_applicable",
                as_of,
                (registry_key,),
            ),
            _Fact(
                "deployment.cutoffs",
                {str(key): str(value) for key, value in era["cutoffs"].items()},
                "ISO_date",
                as_of,
                (deployment_key,),
            ),
            _Fact(
                "deployment.checksum_pins",
                {
                    "pointer_sha256": str(deployment["pointer_sha256"]),
                    "state_manifest_sha256": str(
                        deployment["state_manifest_sha256"]
                    ),
                    "authorizing_receipt_sha256": str(
                        deployment["authorizing_receipt_sha256"]
                    ),
                    **{
                        str(key): str(value)
                        for key, value in era["pins"].items()
                        if key
                        in {
                            "bundle_sha256",
                            "model_sha256",
                            "dataset_sha256",
                            "feature_schema_sha256",
                            "calibration_sha256",
                            "ledger_sha256",
                        }
                    },
                },
                "SHA-256",
                as_of,
                (deployment_key,),
            ),
        ]
        if metadata:
            runtime_metadata = era.get("_runtime_metadata")
            if not isinstance(runtime_metadata, Mapping):
                raise DeploymentRuntimeError(
                    "Verified runtime metadata is unavailable."
                )
            bundle_key = "model_bundle"
            calibration_key = "active_calibration"
            citations.extend(
                [
                    (
                        bundle_key,
                        _Citation(
                            EvidenceDomain.MODEL_BUNDLE,
                            "verify_active_model_era.model_bundle",
                            str(era["schema_version"]),
                            str(era["pins"]["bundle_sha256"]),
                            str(era["pins"]["bundle_sha256"]),
                            as_of,
                        ),
                    ),
                    (
                        calibration_key,
                        _Citation(
                            EvidenceDomain.CALIBRATION,
                            "verify_active_model_era.calibration",
                            str(era["schema_version"]),
                            str(era["calibration"]["calibration_id"]),
                            str(era["pins"]["calibration_sha256"]),
                            as_of,
                        ),
                    ),
                ]
            )
            facts.extend(
                [
                    _Fact(
                        "model.model_sha256",
                        str(era["pins"]["model_sha256"]),
                        "SHA-256",
                        as_of,
                        (bundle_key,),
                    ),
                    _Fact(
                        "model.bundle_sha256",
                        str(era["pins"]["bundle_sha256"]),
                        "SHA-256",
                        as_of,
                        (bundle_key,),
                    ),
                    _Fact(
                        "model.dataset_version",
                        str(runtime_metadata["dataset_version"]),
                        "not_applicable",
                        as_of,
                        (bundle_key,),
                    ),
                    _Fact(
                        "model.dataset_sha256",
                        str(era["pins"]["dataset_sha256"]),
                        "SHA-256",
                        as_of,
                        (bundle_key,),
                    ),
                    _Fact(
                        "model.feature_schema_sha256",
                        str(era["pins"]["feature_schema_sha256"]),
                        "SHA-256",
                        as_of,
                        (bundle_key,),
                    ),
                    _Fact(
                        "model.transformation_version",
                        str(runtime_metadata["transformation_version"]),
                        "not_applicable",
                        as_of,
                        (bundle_key,),
                    ),
                    _Fact(
                        "model.model_type",
                        str(runtime_metadata["model_type"]),
                        "not_applicable",
                        as_of,
                        (bundle_key,),
                    ),
                    _Fact(
                        "model.calibration",
                        {
                            str(key): str(value)
                            for key, value in era["calibration"].items()
                            if key in {"calibration_id", "reference_id"}
                        },
                        "not_applicable",
                        as_of,
                        (calibration_key,),
                    ),
                ]
            )
        return facts, citations

    def _data_quality(self, query: OperationalQuery) -> _Result:
        report, absent_status = self._select_report(query, allow_run=True)
        if report is None:
            return _Result(absent_status)
        key, citation = _report_citation(report)
        quality = report.get("quality")
        if not isinstance(quality, Mapping):
            raise MonitoringReportingError("Monitoring report quality is invalid.")
        through = str(report["through_date"])
        freshness = quality.get("freshness") or {}
        unresolved_late_dates = freshness.get("unresolved_late_dates") or []
        issue_codes = [
            str(item["code"])
            for item in quality.get("issues") or []
            if isinstance(item, Mapping) and item.get("code")
        ]
        facts = [
            _Fact(
                "data_quality.status",
                str(quality.get("batch_status") or quality.get("status") or "not_available"),
                "not_applicable",
                through,
                (key,),
            ),
            _Fact(
                "data_quality.verdict",
                str(quality.get("verdict") or "not_available"),
                "not_applicable",
                through,
                (key,),
            ),
            _Fact(
                "data_quality.source_run_id",
                str((report.get("source_batch") or {}).get("run_id") or ""),
                "not_applicable",
                through,
                (key,),
            ),
            _Fact(
                "data_quality.source_status",
                str((report.get("source_batch") or {}).get("status") or ""),
                "not_applicable",
                through,
                (key,),
            ),
            _Fact(
                "data_quality.freshness",
                {
                    "watermark_date": freshness.get(
                        "common_validated_watermark"
                    ),
                    "watermark_age_days": freshness.get("watermark_age_days"),
                    "objective_days": freshness.get("objective_days"),
                    "late_days": freshness.get("late_days"),
                    "objective_missed": freshness.get("objective_missed"),
                    "unresolved_late_date_count": len(unresolved_late_dates),
                },
                "days_and_state",
                through,
                (key,),
            ),
            _Fact(
                "data_quality.issue_codes",
                issue_codes,
                "not_applicable",
                through,
                (key,),
            ),
        ]
        coverage = quality.get("coverage")
        if isinstance(coverage, Mapping):
            facts.append(
                _Fact(
                    "data_quality.completeness",
                    {
                        name: coverage.get(name)
                        for name in (
                            "date_count",
                            "ren_complete_count",
                            "era5_complete_count",
                            "integration_ready_count",
                            "feature_ready_count",
                            "feature_ready_ratio",
                        )
                    },
                    "count_or_ratio",
                    through,
                    (key,),
                )
            )
        self._assert_latest_report_unchanged(query, report)
        return _Result(
            AnswerStatus.ANSWERED,
            tuple(facts),
            ((key, citation),),
            ("Quality evidence is report-scoped historical evidence.",),
        )

    def _monitoring_performance(self, query: OperationalQuery) -> _Result:
        report, absent_status = self._select_report(query)
        if report is None:
            return _Result(absent_status)
        window = str(query.window_days)
        payload = (report.get("windows") or {}).get(window)
        if not isinstance(payload, Mapping) or payload.get("status") != "available":
            return _Result(AnswerStatus.EMPTY)
        calibration = self._load_report_calibration(report, query)
        key, citation = _report_citation(report)
        cal_key, cal_citation = _calibration_citation(calibration, report)
        performance = payload.get("performance") or {}
        metrics = performance.get("metrics") or {}
        severities = performance.get("severity") or {}
        thresholds = calibration["thresholds"]["performance"][window]
        through = str(report["through_date"])
        coverage_limits = (
            calibration["thresholds"].get("coverage", {}).get(window) or {}
        )
        facts = [
            _Fact(
                "monitoring.performance.sample_count",
                int(payload["sample_count"]),
                "count",
                through,
                (key,),
            ),
            _Fact(
                "monitoring.performance.coverage",
                {
                    "ratio": payload.get("coverage_ratio"),
                    "severity": payload.get("coverage_severity"),
                    "minimum_samples": payload.get("minimum_samples"),
                    "calendar_start": payload.get("calendar_start"),
                    "calendar_end": payload.get("calendar_end"),
                    "warning": coverage_limits.get("warning"),
                    "critical": coverage_limits.get("critical"),
                    "direction": coverage_limits.get("direction"),
                },
                "ratio_and_count",
                through,
                (key,),
            ),
        ]
        for metric in ("MAE", "RMSE", "bias", "MAPE_percent", "R2"):
            limit_key = "absolute_bias" if metric == "bias" else metric
            limit = thresholds.get(limit_key)
            if limit is None:
                continue
            facts.append(
                _Fact(
                    f"monitoring.performance.{metric.lower()}",
                    {
                        "value": metrics.get(metric),
                        "status": (
                            metrics.get("R2_status")
                            if metric == "R2" and metrics.get(metric) is None
                            else "available"
                        ),
                        "severity": severities.get(metric, "not_available"),
                        "warning": limit.get("warning"),
                        "critical": limit.get("critical"),
                        "direction": limit.get("direction", "upper"),
                    },
                    (
                        TARGET_SCALE
                        if metric in {"MAE", "RMSE", "bias"}
                        else "percent"
                        if metric == "MAPE_percent"
                        else "not_applicable"
                    ),
                    through,
                    (key, cal_key),
                )
            )
        self._assert_latest_report_unchanged(query, report)
        return _Result(
            AnswerStatus.ANSWERED,
            tuple(facts),
            ((key, citation), (cal_key, cal_citation)),
            ("Metrics are historical and use the report's sealed calibration.",),
        )

    def _monitoring_drift(self, query: OperationalQuery) -> _Result:
        report, absent_status = self._select_report(query)
        if report is None:
            return _Result(absent_status)
        window = str(query.window_days)
        payload = (report.get("windows") or {}).get(window)
        if not isinstance(payload, Mapping) or payload.get("status") != "available":
            return _Result(AnswerStatus.EMPTY)
        calibration = self._load_report_calibration(report, query)
        key, citation = _report_citation(report)
        cal_key, cal_citation = _calibration_citation(calibration, report)
        through = str(report["through_date"])
        facts: list[_Fact] = []
        for feature in sorted((payload.get("feature_drift") or {})):
            comparisons = payload["feature_drift"][feature] or {}
            for comparator in sorted(comparisons):
                stats = comparisons[comparator] or {}
                for detector in ("ks_statistic", "normalized_wasserstein"):
                    value = stats.get(detector)
                    limits = (
                        calibration["thresholds"]["feature_drift"]
                        .get(feature, {})
                        .get(window, {})
                        .get(comparator, {})
                        .get(detector)
                    )
                    if not isinstance(value, (int, float)) or not limits:
                        continue
                    facts.append(
                        _Fact(
                            f"monitoring.drift.{len(facts) + 1}",
                            {
                                "feature": str(feature),
                                "comparator": str(comparator),
                                "detector": detector,
                                "value": float(value),
                                "severity": threshold_severity(float(value), limits),
                                "warning": float(limits["warning"]),
                                "critical": float(limits["critical"]),
                                "direction": str(limits.get("direction") or "upper"),
                            },
                            "calibrated_drift_statistic",
                            through,
                            (key, cal_key),
                        )
                    )
        self._assert_latest_report_unchanged(query, report)
        if not facts:
            return _Result(AnswerStatus.EMPTY)
        return _Result(
            AnswerStatus.ANSWERED,
            tuple(facts),
            ((key, citation), (cal_key, cal_citation)),
            ("Drift is observational and does not establish a root cause.",),
        )

    def _monitoring_alerts(self, query: OperationalQuery) -> _Result:
        active_before = load_active_alerts(self.monitoring_store_root)
        self._check_deadline(query)
        history = load_alert_history(self.monitoring_store_root)
        self._check_deadline(query)
        selector = query.selector
        projection = self._select_projected_alerts(
            query,
            active_before=active_before,
            history=history,
        )
        if projection is not None:
            by_id = {str(item["alert_event_id"]): item for item in history}
            try:
                selected = [by_id[item] for item in projection.selected_ids]
            except KeyError as exc:
                raise OperationalProjectionUnavailableError(
                    "Projected alert selection is stale."
                ) from exc
        elif isinstance(selector, LatestSelector):
            ids = set(active_before.values())
            selected = [item for item in history if item.get("alert_event_id") in ids]
        elif isinstance(selector, ExactIdSelector):
            selected = [
                item
                for item in history
                if item.get("alert_event_id") == selector.identifier
            ]
        elif isinstance(selector, DateIntervalSelector):
            selected = [
                item
                for item in history
                if selector.start_date
                <= date.fromisoformat(str(item["through_date"]))
                <= selector.end_date
            ]
        else:
            raise TypeError("Unsupported selector")
        if isinstance(selector, ExactIdSelector) and not selected:
            self._assert_active_alerts_unchanged(query, active_before)
            return _Result(AnswerStatus.NOT_FOUND)
        if not selected:
            self._assert_active_alerts_unchanged(query, active_before)
            return _Result(AnswerStatus.EMPTY)
        pagination = query.pagination
        if projection is None and not isinstance(selector, ExactIdSelector):
            pagination = pagination or Pagination()
        if projection is None and pagination is not None:
            selected = selected[pagination.offset : pagination.offset + pagination.limit]
        if not selected:
            self._assert_active_alerts_unchanged(query, active_before)
            return _Result(AnswerStatus.EMPTY)
        active_ids = set(active_before.values())
        observed = self._now()
        active_key, active_citation = _active_alert_citation(
            active_before, observed
        )
        citations: list[tuple[str, _Citation]] = []
        facts: list[_Fact] = []
        for item in selected:
            alert_id = str(item["alert_event_id"])
            key = f"alert:{alert_id}"
            citations.append(
                (
                    key,
                    _Citation(
                        EvidenceDomain.ALERT,
                        "load_alert_history",
                        str(item["schema_version"]),
                        alert_id,
                        alert_id,
                        str(item["through_date"]),
                    ),
                )
            )
            facts.append(
                _Fact(
                    f"monitoring.alert.{len(facts) + 1}",
                    {
                        "alert_event_id": alert_id,
                        "rule_id": str(item["rule_id"]),
                        "event_type": str(item["event_type"]),
                        "severity": str(item["severity"]),
                        "through_date": str(item["through_date"]),
                        "previous_alert_event_id": item.get("previous_alert_event_id"),
                        "active": alert_id in active_ids,
                    },
                    "not_applicable",
                    str(item["through_date"]),
                    (key, active_key),
                )
            )
        self._assert_active_alerts_unchanged(query, active_before)
        citations.append((active_key, active_citation))
        return _Result(
            AnswerStatus.ANSWERED,
            tuple(facts),
            tuple(citations),
            ("Alert transitions are immutable historical events.",),
        )

    def _reporting_run(self, query: OperationalQuery) -> _Result:
        selector = query.selector
        assert isinstance(selector, ExactIdSelector)
        kwargs = (
            {"reporting_run_id": selector.identifier}
            if selector.id_type == "reporting_run_id"
            else {"report_id": selector.identifier}
        )
        projected_attempt = self._select_projected_attempt(query, selector)
        attempt = load_reporting_attempt(self.monitoring_store_root, **kwargs)
        self._check_deadline(query)
        self._compare_projected_attempt(projected_attempt, attempt)
        if attempt is None:
            return _Result(AnswerStatus.NOT_FOUND)
        key, citation = _attempt_citation(attempt)
        as_of = str(attempt["attempted_at_utc"])
        facts = [
            _Fact(
                "reporting_run.identity",
                {
                    "reporting_run_id": str(attempt["run_id"]),
                    "report_id": attempt.get("report_id"),
                },
                "not_applicable",
                as_of,
                (key,),
            ),
            _Fact(
                "reporting_run.status",
                str(attempt["status"]),
                "not_applicable",
                as_of,
                (key,),
            ),
            _Fact(
                "reporting_run.source",
                {
                    "run_id": str(attempt["source_pipeline_run_id"]),
                    "status": str(attempt["source_pipeline_status"]),
                    "through_date": str(attempt["through_date"]),
                },
                "not_applicable",
                as_of,
                (key,),
            ),
            _Fact(
                "reporting_run.active_alert_count",
                int(attempt["active_alert_count"]),
                "count",
                as_of,
                (key,),
            ),
        ]
        citations: list[tuple[str, _Citation]] = [(key, citation)]
        if attempt.get("report_id"):
            report_id = str(attempt["report_id"])
            projected_report = self._select_projected_report_exact(
                query,
                report_id,
                detail="quality",
            )
            report = self._load_report_id(report_id, query)
            self._compare_projected_report(
                projected_report,
                report,
                query=query,
                detail="quality",
            )
            report_key, report_citation = _report_citation(report)
            citations.append((report_key, report_citation))
            facts.append(
                _Fact(
                    "reporting_run.verified_report",
                    {
                        "report_id": str(report["report_id"]),
                        "through_date": str(report["through_date"]),
                    },
                    "not_applicable",
                    str(report["through_date"]),
                    (report_key,),
                )
            )
        failure = attempt.get("failure")
        if isinstance(failure, Mapping):
            facts.append(
                _Fact(
                    "reporting_run.failure",
                    {
                        "failed_at_utc": failure.get("failed_at_utc"),
                        "error_type": failure.get("error_type"),
                        "message": failure.get("message"),
                    },
                    "not_applicable",
                    as_of,
                    (key,),
                )
            )
        return _Result(
            AnswerStatus.ANSWERED,
            tuple(facts),
            tuple(citations),
            (
                "Reporting failures expose only the existing sanitized operator state.",
            ),
        )

    def _select_report(
        self, query: OperationalQuery, *, allow_run: bool = False
    ) -> tuple[dict[str, Any] | None, AnswerStatus]:
        selector = query.selector
        detail = {
            QueryKind.DATA_QUALITY: "quality",
            QueryKind.MONITORING_PERFORMANCE: "performance",
            QueryKind.MONITORING_DRIFT: "drift",
        }[query.query_kind]
        if isinstance(selector, LatestSelector):
            state = load_monitoring_report_state(self.monitoring_store_root)
            self._check_deadline(query)
            projected = self._select_projected_report_latest(
                query,
                state=state,
                detail=detail,
            )
            if state is None:
                if projected is not None:
                    raise OperationalProjectionUnavailableError(
                        "Projected latest report is stale."
                    )
                state_after = load_monitoring_report_state(
                    self.monitoring_store_root
                )
                self._check_deadline(query)
                if state_after is not None:
                    raise _QueryConflict
                return None, AnswerStatus.EMPTY
            report_id = str(state["latest_report_id"])
            if projected is not None and (
                projected.report.values.get("report_id") != report_id
            ):
                raise OperationalProjectionUnavailableError(
                    "Projected latest report is stale."
                )
            report = self._load_report_id(report_id, query)
            if (
                report.get("report_id") != state.get("latest_report_id")
                or report.get("through_date") != state.get("latest_through_date")
            ):
                raise _QueryConflict
            self._compare_projected_report(
                projected,
                report,
                query=query,
                detail=detail,
            )
            return report, AnswerStatus.EMPTY
        if not isinstance(selector, ExactIdSelector):
            raise TypeError("Report query requires an exact or latest selector")
        if selector.id_type == "reporting_run_id":
            if not allow_run:
                raise TypeError("Reporting-run selector is not accepted")
            projected_attempt = self._select_projected_attempt(query, selector)
            attempt = load_reporting_attempt(
                self.monitoring_store_root,
                reporting_run_id=selector.identifier,
            )
            self._check_deadline(query)
            self._compare_projected_attempt(projected_attempt, attempt)
            if attempt is None:
                return None, AnswerStatus.NOT_FOUND
            if not attempt.get("report_id"):
                return None, AnswerStatus.EMPTY
            report_id = str(attempt["report_id"])
            projected = self._select_projected_report_exact(
                query,
                report_id,
                detail=detail,
            )
            report = self._load_report_id(report_id, query)
            self._compare_projected_report(
                projected,
                report,
                query=query,
                detail=detail,
            )
            return report, AnswerStatus.EMPTY
        projected = self._select_projected_report_exact(
            query,
            selector.identifier,
            detail=detail,
        )
        report = self._load_report_id(selector.identifier, query, exact=True)
        self._compare_projected_report(
            projected,
            report,
            query=query,
            detail=detail,
        )
        return (
            (None, AnswerStatus.NOT_FOUND)
            if report is None
            else (report, AnswerStatus.EMPTY)
        )

    def _select_projected_report_latest(
        self,
        query: OperationalQuery,
        *,
        state: Mapping[str, Any] | None,
        detail: str,
    ) -> ProjectedReport | None:
        if self.projection_reader is None:
            return None
        return self.projection_reader.select_report(
            selector="latest",
            report_id=None,
            report_state_sha256=(None if state is None else _digest(state)),
            report_state_schema_version=(
                None if state is None else str(state["schema_version"])
            ),
            report_state_effective_at=(
                None if state is None else str(state["latest_through_date"])
            ),
            detail=detail,
            window_days=query.window_days,
            timeout_seconds=self._remaining_seconds(query),
        )

    def _select_projected_report_exact(
        self,
        query: OperationalQuery,
        report_id: str,
        *,
        detail: str,
    ) -> ProjectedReport | None:
        if self.projection_reader is None:
            return None
        return self.projection_reader.select_report(
            selector="exact",
            report_id=report_id,
            report_state_sha256=None,
            report_state_schema_version=None,
            report_state_effective_at=None,
            detail=detail,
            window_days=query.window_days,
            timeout_seconds=self._remaining_seconds(query),
        )

    def _select_projected_attempt(
        self,
        query: OperationalQuery,
        selector: ExactIdSelector,
    ) -> ProjectedRow | None:
        if self.projection_reader is None:
            return None
        return self.projection_reader.select_attempt(
            id_type=selector.id_type,
            identifier=selector.identifier,
            timeout_seconds=self._remaining_seconds(query),
        )

    def _select_projected_alerts(
        self,
        query: OperationalQuery,
        *,
        active_before: Mapping[str, Any],
        history: list[dict[str, Any]],
    ) -> ProjectedAlerts | None:
        if self.projection_reader is None:
            return None
        selector = query.selector
        pagination = query.pagination or Pagination()
        kwargs: dict[str, Any] = {
            "timeout_seconds": self._remaining_seconds(query),
        }
        if isinstance(selector, LatestSelector):
            kwargs.update(
                selector="latest",
                limit=pagination.limit,
                offset=pagination.offset,
            )
        elif isinstance(selector, ExactIdSelector):
            kwargs.update(selector="exact", identifier=selector.identifier)
        elif isinstance(selector, DateIntervalSelector):
            kwargs.update(
                selector="date_interval",
                start_date=selector.start_date,
                end_date=selector.end_date,
                limit=pagination.limit,
                offset=pagination.offset,
            )
        else:
            raise TypeError("Unsupported selector")
        projection = self.projection_reader.select_alerts(**kwargs)
        expected_history = tuple(_normalized_alert(item) for item in history)
        if projection.history != expected_history:
            raise OperationalProjectionUnavailableError(
                "Projected alert history is stale."
            )
        expected_active = {
            str(key): str(value) for key, value in active_before.items()
        }
        if dict(projection.active) != expected_active:
            raise OperationalProjectionUnavailableError(
                "Projected active-alert state is stale."
            )
        if projection.active_evidence is not None:
            active_digest = _digest(dict(sorted(expected_active.items())))
            active_evidence = projection.active_evidence
            if (
                active_evidence.domain != "alert"
                or active_evidence.source_kind != "load_active_alerts"
                or active_evidence.schema_version
                != "wind_forecast.verified_active_alert_binding.v1"
                or active_evidence.record_id != active_digest
                or active_evidence.sha256 != active_digest
            ):
                raise OperationalProjectionUnavailableError(
                    "Projected active-alert evidence is stale."
                )
        return projection

    def _compare_projected_attempt(
        self,
        projected: ProjectedRow | None,
        attempt: Mapping[str, Any] | None,
    ) -> None:
        if self.projection_reader is None:
            return
        if projected is None or attempt is None:
            if projected is None and attempt is None:
                return
            raise OperationalProjectionUnavailableError(
                "Projected reporting attempt is stale."
            )
        if projected != _normalized_attempt(attempt):
            raise OperationalProjectionUnavailableError(
                "Projected reporting attempt is stale."
            )

    def _compare_projected_report(
        self,
        projected: ProjectedReport | None,
        report: Mapping[str, Any] | None,
        *,
        query: OperationalQuery,
        detail: str,
    ) -> None:
        if self.projection_reader is None:
            return
        if projected is None or report is None:
            if projected is None and report is None:
                return
            raise OperationalProjectionUnavailableError(
                "Projected monitoring report is stale."
            )
        resolved_era = resolve_report_model_era(
            self.monitoring_store_root,
            report,
        )
        model_era_id = (
            str(resolved_era["model_era_id"])
            if resolved_era.get("association_kind")
            in {"active_deployment", "bootstrap_adopted"}
            else None
        )
        if projected.report != _normalized_report(report, model_era_id):
            raise OperationalProjectionUnavailableError(
                "Projected monitoring report is stale."
            )
        if detail == "quality":
            expected_issues = _normalized_quality_issues(report)
            if projected.quality_issues != expected_issues:
                raise OperationalProjectionUnavailableError(
                    "Projected data-quality values are stale."
                )
            return

        calibration = self._load_report_calibration(report, query)
        expected_window = _normalized_window(
            report,
            int(query.window_days),
        )
        if projected.window != expected_window:
            raise OperationalProjectionUnavailableError(
                "Projected monitoring-window values are stale."
            )
        if projected.calibration != _normalized_calibration_evidence(calibration):
            raise OperationalProjectionUnavailableError(
                "Projected calibration lineage is stale."
            )
        if model_era_id is None:
            if projected.model_era is not None:
                raise OperationalProjectionUnavailableError(
                    "Projected model-era association is stale."
                )
        else:
            era = load_model_era(self.monitoring_store_root, model_era_id)
            self._check_deadline(query)
            if projected.model_era != _normalized_model_era(era):
                raise OperationalProjectionUnavailableError(
                    "Projected model-era values are stale."
                )
        if detail == "performance":
            expected_metrics = _normalized_performance_metrics(
                report,
                calibration,
                int(query.window_days),
            )
            if projected.performance_metrics != expected_metrics:
                raise OperationalProjectionUnavailableError(
                    "Projected performance values are stale."
                )
        else:
            expected_drift = _normalized_drift_measurements(
                report,
                calibration,
                int(query.window_days),
            )
            if projected.drift_measurements != expected_drift:
                raise OperationalProjectionUnavailableError(
                    "Projected drift values are stale."
                )

    def _remaining_seconds(self, query: OperationalQuery) -> float:
        self._check_deadline(query)
        remaining = (query.deadline - self._now()).total_seconds()
        if remaining <= 0:
            raise _QueryTimeout
        return min(self.max_deadline_seconds, remaining)

    def _load_report_id(
        self, report_id: str, query: OperationalQuery, *, exact: bool = False
    ) -> dict[str, Any] | None:
        path = (
            self.monitoring_store_root
            / "reporting"
            / "reports"
            / report_id
            / "report.json"
        )
        if exact and not path.is_file():
            return None
        report = load_monitoring_report(path)
        self._check_deadline(query)
        if report.get("report_id") != report_id:
            raise _QueryConflict
        return report

    def _load_report_calibration(
        self, report: Mapping[str, Any], query: OperationalQuery
    ) -> dict[str, Any]:
        reference = report.get("reference")
        if not isinstance(reference, Mapping):
            raise MonitoringReportingError("Report calibration reference is invalid.")
        calibration = load_monitoring_calibration(
            self.monitoring_store_root
            / "reporting"
            / "calibrations"
            / str(reference.get("calibration_id") or "")
        )
        self._check_deadline(query)
        model_era = resolve_report_model_era(self.monitoring_store_root, report)
        if (
            calibration.get("calibration_id") != reference.get("calibration_id")
            or calibration.get("reference_id") != reference.get("reference_id")
            or calibration.get("policy_sha256") != reference.get("policy_sha256")
            or model_era.get("association_kind") not in {
                "active_deployment",
                "bootstrap_adopted",
                "legacy_unassociated",
            }
        ):
            raise _QueryConflict
        return calibration

    def _assert_latest_report_unchanged(
        self, query: OperationalQuery, report: Mapping[str, Any]
    ) -> None:
        if not isinstance(query.selector, LatestSelector):
            return
        state = load_monitoring_report_state(self.monitoring_store_root)
        self._check_deadline(query)
        if (
            state is None
            or state.get("latest_report_id") != report.get("report_id")
            or state.get("latest_through_date") != report.get("through_date")
        ):
            raise _QueryConflict

    def _assert_active_alerts_unchanged(
        self, query: OperationalQuery, active_before: Mapping[str, Any]
    ) -> None:
        active_after = load_active_alerts(self.monitoring_store_root)
        self._check_deadline(query)
        if active_before != active_after:
            raise _QueryConflict

    def _check_deadline(self, query: OperationalQuery) -> None:
        if self._now() >= query.deadline:
            raise _QueryTimeout

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("clock must return a timezone-aware instant")
        return value.astimezone(timezone.utc)

    def _render(self, query: OperationalQuery, result: _Result) -> OperationalAnswer:
        if result.status != AnswerStatus.ANSWERED:
            return self._absence_answer(query, result.status, result.limitations)
        citation_map = dict(result.citations)
        ordered_keys = sorted(
            {
                key
                for fact in result.facts
                for key in fact.citation_keys
            },
            key=lambda key: (
                citation_map[key].domain.value,
                citation_map[key].record_id,
                key,
            ),
        )
        evidence_ids = {key: f"e{index}" for index, key in enumerate(ordered_keys, 1)}
        evidence = tuple(
            EvidenceCitation(
                evidence_id=evidence_ids[key],
                domain=citation_map[key].domain,
                source_kind=citation_map[key].source_kind,
                schema_version=citation_map[key].schema_version,
                record_id=citation_map[key].record_id,
                sha256=citation_map[key].sha256,
                effective_at=citation_map[key].effective_at,
                observed_at_utc=citation_map[key].observed_at_utc,
            )
            for key in ordered_keys
        )
        ordered_facts = sorted(
            result.facts,
            key=lambda item: (item.name, _canonical_text(item.value)),
        )
        if len(ordered_facts) > MAX_PUBLIC_FACTS:
            raise ValueError("Operational answer exceeds the public fact limit.")
        facts = tuple(
            GroundedFact(
                fact_id=f"f{index}",
                name=item.name,
                value=item.value,
                unit_or_scale=item.unit_or_scale,
                as_of=item.as_of,
                evidence_ids=tuple(evidence_ids[key] for key in item.citation_keys),
            )
            for index, item in enumerate(ordered_facts, 1)
        )
        _validate_public_value([item.model_dump() for item in facts])
        _validate_public_value([item.model_dump() for item in evidence])
        _validate_public_value(result.limitations)
        summary = " ".join(
            f"{fact.name}={_canonical_text(fact.value)} "
            f"{''.join(f'[{item}]' for item in fact.evidence_ids)}."
            for fact in facts
        )
        answer = OperationalAnswer(
            contract_version=CONTRACT_VERSION,
            query_kind=query.query_kind,
            status=AnswerStatus.ANSWERED,
            mode=OPERATIONAL_MODE,
            summary=summary,
            facts=facts,
            evidence=evidence,
            limitations=result.limitations,
            failure=None,
            served_at_utc=self._now(),
            correlation_id=query.correlation_id,
        )
        self._check_deadline(query)
        return OperationalAnswer.model_validate(
            answer.model_dump(), strict=True
        )

    def _absence_answer(
        self,
        query: OperationalQuery,
        status: AnswerStatus,
        limitations: tuple[str, ...] = (),
    ) -> OperationalAnswer:
        return OperationalAnswer(
            query_kind=query.query_kind,
            status=status,
            summary=None,
            facts=(),
            evidence=(),
            limitations=limitations,
            failure=None,
            served_at_utc=self._now(),
            correlation_id=query.correlation_id,
        )

    def _failure_answer(
        self,
        *,
        query_kind: QueryKind | None,
        correlation_id: str,
        status: AnswerStatus,
        code: str,
        message: str,
        retryable: bool,
        evidence_state: EvidenceState,
    ) -> OperationalAnswer:
        return OperationalAnswer(
            query_kind=query_kind,
            status=status,
            summary=None,
            facts=(),
            evidence=(),
            limitations=(),
            failure=OperationalFailure(
                code=code,
                message=message,
                retryable=retryable,
                evidence_state=evidence_state,
            ),
            served_at_utc=self._now(),
            correlation_id=correlation_id,
        )


def _normalized_report(
    report: Mapping[str, Any],
    model_era_id: str | None,
) -> ProjectedRow:
    quality = _required_mapping(report.get("quality"))
    freshness = quality.get("freshness") or {}
    coverage = quality.get("coverage") or {}
    source = _required_mapping(report.get("source_batch"))
    reference = _required_mapping(report.get("reference"))
    quality_status = str(
        quality.get("status") or quality.get("batch_status") or "not_available"
    )
    values = {
        "report_id": str(report["report_id"]),
        "reporting_run_id": str(report["run_id"]),
        "created_at_utc": _projection_utc(report["created_at_utc"]),
        "through_date": _projection_date(report["through_date"]),
        "source_run_id": str(source["run_id"]),
        "source_status": str(source["status"]),
        "calibration_id": str(reference["calibration_id"]),
        "reference_id": str(reference["reference_id"]),
        "policy_sha256": str(reference["policy_sha256"]),
        "quality_status": quality_status,
        "batch_status": str(quality.get("batch_status") or source.get("status")),
        "verdict": str(quality.get("verdict") or "not_available"),
        "watermark_date": _optional_projection_date(
            freshness.get("common_validated_watermark")
        ),
        "watermark_age_days": _optional_projection_int(
            freshness.get("watermark_age_days")
        ),
        "objective_days": _optional_projection_int(freshness.get("objective_days")),
        "late_days": _optional_projection_int(freshness.get("late_days")),
        "objective_missed": bool(freshness.get("objective_missed") or False),
        "unresolved_late_date_count": len(
            freshness.get("unresolved_late_dates") or []
        ),
        "date_count": int(coverage.get("date_count") or 0),
        "ren_complete_count": int(coverage.get("ren_complete_count") or 0),
        "era5_complete_count": int(coverage.get("era5_complete_count") or 0),
        "integration_ready_count": int(
            coverage.get("integration_ready_count") or 0
        ),
        "feature_ready_count": int(coverage.get("feature_ready_count") or 0),
        "model_era_id": model_era_id,
    }
    return ProjectedRow(values, _report_projection_evidence(report))


def _normalized_quality_issues(
    report: Mapping[str, Any],
) -> tuple[ProjectedRow, ...]:
    quality = _required_mapping(report.get("quality"))
    evidence = _report_projection_evidence(report)
    rows: list[ProjectedRow] = []
    for position, issue in enumerate(quality.get("issues") or []):
        if not isinstance(issue, Mapping) or issue.get("severity") not in {
            "warning",
            "critical",
        }:
            continue
        rows.append(
            ProjectedRow(
                {
                    "report_id": str(report["report_id"]),
                    "position": position,
                    "code": str(issue["code"]),
                    "severity": str(issue["severity"]),
                },
                evidence,
            )
        )
    return tuple(rows)


def _normalized_window(
    report: Mapping[str, Any],
    window_days: int,
) -> ProjectedRow:
    payload = (report.get("windows") or {}).get(str(window_days)) or {}
    configured = (report.get("config") or {}).get("minimum_samples") or {}
    available = payload.get("status") == "available"
    minimum = payload.get("minimum_samples", configured.get(str(window_days)))
    return ProjectedRow(
        {
            "report_id": str(report["report_id"]),
            "window_days": window_days,
            "status": "available" if available else "not_available",
            "sample_count": int(payload.get("sample_count") or 0),
            "coverage_ratio": _optional_projection_float(
                payload.get("coverage_ratio")
            ),
            "coverage_severity": (
                str(payload.get("coverage_severity") or "not_available")
                if available
                else "not_available"
            ),
            "minimum_samples": int(minimum or 0),
            "calendar_start": _optional_projection_date(
                payload.get("calendar_start")
            ),
            "calendar_end": _optional_projection_date(payload.get("calendar_end")),
        },
        _report_projection_evidence(report),
    )


def _normalized_performance_metrics(
    report: Mapping[str, Any],
    calibration: Mapping[str, Any],
    window_days: int,
) -> tuple[ProjectedRow, ...]:
    payload = (report.get("windows") or {}).get(str(window_days)) or {}
    performance = payload.get("performance") or {}
    metrics = performance.get("metrics") or {}
    severities = performance.get("severity") or {}
    thresholds = (
        calibration.get("thresholds", {})
        .get("performance", {})
        .get(str(window_days), {})
    )
    evidence = _report_projection_evidence(report)
    rows: list[ProjectedRow] = []
    for metric in ("MAE", "RMSE", "bias", "MAPE_percent", "R2"):
        limit_key = "absolute_bias" if metric == "bias" else metric
        limits = thresholds.get(limit_key)
        if not isinstance(limits, Mapping):
            continue
        value = _optional_projection_float(metrics.get(metric))
        if performance.get("status") != "available":
            value_status = (
                "insufficient_data"
                if performance.get("status") == "insufficient_data"
                else "not_available"
            )
        elif value is None and metric == "R2":
            candidate = str(metrics.get("R2_status") or "not_available")
            value_status = (
                candidate
                if candidate in {"insufficient_data", "constant_target"}
                else "not_available"
            )
        else:
            value_status = "available" if value is not None else "not_available"
        unit = (
            TARGET_SCALE
            if metric in {"MAE", "RMSE", "bias"}
            else "percent"
            if metric == "MAPE_percent"
            else "not_applicable"
        )
        rows.append(
            ProjectedRow(
                {
                    "report_id": str(report["report_id"]),
                    "window_days": window_days,
                    "metric_name": metric,
                    "value": value,
                    "value_status": value_status,
                    "severity": str(severities.get(metric) or "not_available"),
                    "warning_threshold": _optional_projection_float(
                        limits.get("warning")
                    ),
                    "critical_threshold": _optional_projection_float(
                        limits.get("critical")
                    ),
                    "direction": str(limits.get("direction") or "upper"),
                    "unit_or_scale": unit,
                },
                evidence,
            )
        )
    return tuple(sorted(rows, key=lambda item: str(item.values["metric_name"])))


def _normalized_drift_measurements(
    report: Mapping[str, Any],
    calibration: Mapping[str, Any],
    window_days: int,
) -> tuple[ProjectedRow, ...]:
    payload = (report.get("windows") or {}).get(str(window_days)) or {}
    drift = payload.get("feature_drift") or {}
    calibrated = calibration.get("thresholds", {}).get("feature_drift", {})
    evidence = _report_projection_evidence(report)
    rows: list[ProjectedRow] = []
    position = 0
    for feature in sorted(drift):
        for comparator in sorted((drift.get(feature) or {})):
            statistics = (drift[feature] or {}).get(comparator) or {}
            for detector in ("ks_statistic", "normalized_wasserstein"):
                value = _optional_projection_float(statistics.get(detector))
                limits = (
                    calibrated.get(feature, {})
                    .get(str(window_days), {})
                    .get(comparator, {})
                    .get(detector)
                )
                if value is None or not isinstance(limits, Mapping):
                    continue
                rows.append(
                    ProjectedRow(
                        {
                            "report_id": str(report["report_id"]),
                            "window_days": window_days,
                            "position": position,
                            "feature": str(feature),
                            "comparator": str(comparator),
                            "detector": detector,
                            "value": value,
                            "severity": threshold_severity(value, limits),
                            "warning_threshold": float(limits["warning"]),
                            "critical_threshold": float(limits["critical"]),
                            "direction": str(limits.get("direction") or "upper"),
                        },
                        evidence,
                    )
                )
                position += 1
    return tuple(rows)


def _normalized_model_era(era: Mapping[str, Any]) -> ProjectedRow:
    deployment = _required_mapping(era.get("deployment"))
    registry = _required_mapping(era.get("registry"))
    cutoffs = _required_mapping(era.get("cutoffs"))
    pins = _required_mapping(era.get("pins"))
    calibration = _required_mapping(era.get("calibration"))
    model_era_id = str(era["model_era_id"])
    return ProjectedRow(
        {
            "model_era_id": model_era_id,
            "association_kind": str(era["association_kind"]),
            "deployment_id": str(deployment["deployment_id"]),
            "deployment_generation": int(deployment["generation"]),
            "registered_model_name": str(registry["registered_model_name"]),
            "model_version": str(registry["model_version"]),
            "fit_cutoff": _projection_date(cutoffs["fit_cutoff"]),
            "activation_cutoff": _projection_date(cutoffs["activation_cutoff"]),
            "bundle_sha256": str(pins["bundle_sha256"]),
            "model_sha256": str(pins["model_sha256"]),
            "dataset_sha256": str(pins["dataset_sha256"]),
            "feature_schema_sha256": str(pins["feature_schema_sha256"]),
            "calibration_sha256": str(pins["calibration_sha256"]),
            "ledger_sha256": str(pins["ledger_sha256"]),
            "calibration_id": str(calibration["calibration_id"]),
            "reference_id": str(calibration["reference_id"]),
        },
        ProjectedEvidence(
            "prediction_model_era",
            "load_model_era",
            str(era["schema_version"]),
            model_era_id,
            model_era_id,
            str(cutoffs["activation_cutoff"]),
        ),
    )


def _normalized_attempt(attempt: Mapping[str, Any]) -> ProjectedRow:
    failure = attempt.get("failure")
    run_id = str(attempt["run_id"])
    return ProjectedRow(
        {
            "reporting_run_id": run_id,
            "attempted_at_utc": _projection_utc(attempt["attempted_at_utc"]),
            "through_date": _projection_date(attempt["through_date"]),
            "source_run_id": str(attempt["source_pipeline_run_id"]),
            "source_status": str(attempt["source_pipeline_status"]),
            "status": str(attempt["status"]),
            "report_id": attempt.get("report_id"),
            "active_alert_count": int(attempt.get("active_alert_count") or 0),
            "failure_at_utc": (
                _projection_utc(failure["failed_at_utc"])
                if isinstance(failure, Mapping)
                else None
            ),
            "failure_type": (
                str(failure["error_type"])
                if isinstance(failure, Mapping)
                else None
            ),
            "failure_message": (
                str(failure["message"])
                if isinstance(failure, Mapping)
                else None
            ),
        },
        ProjectedEvidence(
            "reporting_run",
            "load_reporting_attempts",
            "wind_forecast.monitoring_reporting_attempt_projection.v1",
            run_id,
            _digest(attempt),
            str(attempt["attempted_at_utc"]),
        ),
    )


def _normalized_alert(alert: Mapping[str, Any]) -> ProjectedRow:
    alert_id = str(alert["alert_event_id"])
    return ProjectedRow(
        {
            "alert_event_id": alert_id,
            "rule_id": str(alert["rule_id"]),
            "through_date": _projection_date(alert["through_date"]),
            "event_type": str(alert["event_type"]),
            "severity": str(alert["severity"]),
            "previous_alert_event_id": alert.get("previous_alert_event_id"),
        },
        ProjectedEvidence(
            "alert",
            "load_alert_history",
            str(alert["schema_version"]),
            alert_id,
            alert_id,
            str(alert["through_date"]),
        ),
    )


def _report_projection_evidence(report: Mapping[str, Any]) -> ProjectedEvidence:
    report_id = str(report["report_id"])
    return ProjectedEvidence(
        "monitoring_report",
        "load_monitoring_report",
        str(report["schema_version"]),
        report_id,
        report_id,
        str(report["through_date"]),
    )


def _normalized_calibration_evidence(
    calibration: Mapping[str, Any],
) -> ProjectedEvidence:
    calibration_id = str(calibration["calibration_id"])
    manifest = _required_mapping(calibration.get("_reference_manifest"))
    period = _required_mapping(manifest.get("period"))
    return ProjectedEvidence(
        "calibration_reference",
        "load_monitoring_calibration",
        str(calibration["schema_version"]),
        calibration_id,
        calibration_id,
        str(period["end"]),
    )


def _projection_date(value: Any) -> date:
    if isinstance(value, datetime):
        raise ValueError("Projection calendar value contains a time.")
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _optional_projection_date(value: Any) -> date | None:
    return None if value is None else _projection_date(value)


def _projection_utc(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Projection timestamp is invalid.")
    return value.astimezone(timezone.utc)


def _optional_projection_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_projection_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Projection number is invalid.")
    return number


def _required_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Verified operational value is invalid.")
    return value


def _report_citation(report: Mapping[str, Any]) -> tuple[str, _Citation]:
    report_id = str(report["report_id"])
    return (
        f"report:{report_id}",
        _Citation(
            EvidenceDomain.MONITORING_REPORT,
            "load_monitoring_report",
            str(report["schema_version"]),
            report_id,
            report_id,
            str(report["through_date"]),
        ),
    )


def _calibration_citation(
    calibration: Mapping[str, Any], report: Mapping[str, Any]
) -> tuple[str, _Citation]:
    calibration_id = str(calibration["calibration_id"])
    return (
        f"calibration:{calibration_id}",
        _Citation(
            EvidenceDomain.CALIBRATION,
            "load_monitoring_calibration",
            str(calibration["schema_version"]),
            calibration_id,
            calibration_id,
            str(report["through_date"]),
        ),
    )


def _attempt_citation(attempt: Mapping[str, Any]) -> tuple[str, _Citation]:
    run_id = str(attempt["run_id"])
    return (
        f"run:{run_id}",
        _Citation(
            EvidenceDomain.REPORTING_RUN,
            "load_reporting_attempt",
            "wind_forecast.monitoring_reporting_attempt_projection.v1",
            run_id,
            _digest(attempt),
            str(attempt["attempted_at_utc"]),
        ),
    )


def _active_alert_citation(
    active_alerts: Mapping[str, Any], observed_at_utc: datetime
) -> tuple[str, _Citation]:
    digest = _digest(dict(sorted(active_alerts.items())))
    observed = observed_at_utc.astimezone(timezone.utc)
    observed_text = observed.isoformat().replace("+00:00", "Z")
    return (
        f"active-alert-state:{digest}",
        _Citation(
            EvidenceDomain.ALERT,
            "load_active_alerts",
            "wind_forecast.verified_active_alert_binding.v1",
            digest,
            digest,
            observed_text,
            observed,
        ),
    )


def _report_state_citation(
    state: Mapping[str, Any], observed_at_utc: datetime
) -> tuple[str, _Citation]:
    digest = _digest(state)
    observed = observed_at_utc.astimezone(timezone.utc)
    return (
        f"report-state:{digest}",
        _Citation(
            EvidenceDomain.MONITORING_REPORT,
            "load_monitoring_report_state",
            str(state["schema_version"]),
            digest,
            digest,
            str(state["latest_through_date"]),
            observed,
        ),
    )


def _safe_correlation(value: Any) -> str:
    if (
        isinstance(value, str)
        and 0 < len(value) <= 128
        and all(char.isprintable() and char not in "/\\" for char in value)
    ):
        return value
    return "invalid-correlation-id"


def _validate_public_value(value: Any) -> None:
    if isinstance(value, Mapping):
        if len(value) > MAX_PUBLIC_MAPPING_ITEMS:
            raise ValueError("Operational answer mapping exceeds the public limit.")
        for key, item in value.items():
            _validate_public_value(key)
            _validate_public_value(item)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_PUBLIC_COLLECTION_ITEMS:
            raise ValueError("Operational answer collection exceeds the public limit.")
        for item in value:
            _validate_public_value(item)
        return
    if not isinstance(value, str):
        return
    lowered = value.lower()
    if (
        len(value) > MAX_PUBLIC_STRING_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or "\\" in value
        or value.startswith("/")
        or re.match(r"^[a-zA-Z]:[/\\]", value)
        or "://" in lowered
        or "models:/" in lowered
        or any(
            marker in lowered
            for marker in (
                "password=",
                "token=",
                "secret=",
                "connection_string",
                "tracking_uri",
            )
        )
    ):
        raise ValueError("Operational answer contains a non-public value.")


def _digest(value: Any) -> str:
    return sha256(_canonical_text(value).encode("ascii")).hexdigest()


def _canonical_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


__all__ = [
    "AuthorizationPolicy",
    "OperationalQueryService",
    "TARGET_SCALE",
]
