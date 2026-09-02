from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from wind_forecast.copilot_selector import DeterministicPortugueseSelector
from wind_forecast.operational_copilot import CopilotSelectionRefusal
from wind_forecast.operational_copilot_models import (
    CopilotRequest,
    allowed_operational_tools,
)


NOW = datetime(2026, 9, 2, tzinfo=timezone.utc)


def _select(question: str):
    return DeterministicPortugueseSelector().select(
        CopilotRequest(
            question=question,
            requested_at_utc=NOW,
            correlation_id="test-correlation",
            deadline=NOW + timedelta(seconds=5),
        ),
        tools=allowed_operational_tools(),
        timeout_seconds=1.0,
    )


@pytest.mark.parametrize(
    ("question", "kind", "window"),
    (
        ("Qual é o estado operacional verificado?", "operational_summary", None),
        ("Que deployment está ativo?", "active_deployment", None),
        ("Como estão a qualidade e a freshness dos dados?", "data_quality", None),
        ("Qual foi a performance/MAE dos últimos 30 dias?", "monitoring_performance", 30),
        ("Há drift nos últimos 90 dias?", "monitoring_drift", 90),
        ("Há alertas ativos?", "monitoring_alerts", None),
        ("Que modelo está ativo e que metadados o identificam?", "active_model_metadata", None),
    ),
)
def test_closed_portuguese_catalog(question: str, kind: str, window: int | None) -> None:
    selected = _select(question)

    assert selected.tool_name == "operational_query"
    assert selected.arguments["query_kind"] == kind
    assert selected.arguments["window_days"] == window


@pytest.mark.parametrize(
    ("question", "code"),
    (
        ("Dá-me previsões históricas", "forecast_replay_required"),
        ("Treina o modelo", "training_not_supported"),
        ("Promover o candidato", "write_operation_not_supported"),
        ("Qual foi a performance?", "window_required"),
        ("Conta uma anedota", "unsupported_question"),
    ),
)
def test_expected_refusals_have_stable_codes(question: str, code: str) -> None:
    with pytest.raises(CopilotSelectionRefusal) as raised:
        _select(question)

    assert raised.value.code == code


def test_refusal_exception_rejects_unbounded_or_unknown_codes() -> None:
    with pytest.raises(ValueError, match="unsupported Copilot refusal code"):
        CopilotSelectionRefusal("private provider error C:\\secret")
