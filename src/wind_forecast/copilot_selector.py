"""Bounded Portuguese selector for the local operational Copilot."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from .operational_copilot_models import (
    CopilotRequest,
    OperationalToolDefinition,
    OperationalToolSelection,
)
from .operational_copilot import CopilotSelectionRefusal


_SPACE = re.compile(r"\s+")
_WINDOW = re.compile(r"\b(30|90)\s*(?:dias|d)\b")
_REFUSALS = {
    "treino": "training_not_supported",
    "treinar": "training_not_supported",
    "treina": "training_not_supported",
    "retraining": "training_not_supported",
    "promover": "write_operation_not_supported",
    "promocao": "write_operation_not_supported",
    "deployar": "write_operation_not_supported",
    "rollback": "write_operation_not_supported",
    "escrever": "write_operation_not_supported",
    "causa": "undocumented_cause_not_supported",
    "porque": "undocumented_cause_not_supported",
}
_FUTURE_FORECAST_PHRASES = (
    "qual sera a previsao",
    "previsao amanha",
    "previsoes amanha",
    "previsao futura",
    "previsoes futuras",
    "faz uma previsao",
    "fazer uma previsao",
    "preve a proxima semana",
    "preve para a proxima semana",
    "preve o que vai acontecer",
)
_HISTORICAL_FORECAST_PHRASES = (
    "previsoes historicas",
    "previsao historica",
    "observacoes historicas",
    "historico de previs",
    "previsoes e valores observados",
    "previsao e o valor real",
    "qual foi a previsao",
    "consultar previsoes",
    "erro nesse periodo",
)
_DOCUMENTARY_TERMS = (
    "metodologia",
    "limitacoes",
    "decisoes",
    "execucao",
    "executar o projeto",
    "roadmap",
    "documentacao",
    "arquitetura",
)
_SYNONYMS = {
    "estado": ("operacional", "saude", "situacao"),
    "deployment": ("deploy", "implantacao", "versao ativa"),
    "qualidade": ("quality", "completude"),
    "freshness": ("atualidade", "frescura", "recencia"),
    "performance": ("desempenho", "mae", "erro medio"),
    "drift": ("desvio", "deriva"),
    "alertas": ("alertas ativos", "avisos"),
    "modelo": ("model", "metadados"),
}


def _normalize(question: str) -> str:
    folded = unicodedata.normalize("NFD", question.casefold())
    without_accents = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    return _SPACE.sub(" ", without_accents).strip()


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


QuestionRoute = Literal[
    "forbidden",
    "forecast_replay",
    "operational",
    "documentary",
    "ambiguous",
    "unsupported",
]


def classify_question(question: str) -> QuestionRoute:
    """Apply the Copilot route precedence without parsing dates or data."""
    text = _normalize(question)
    tokens = set(re.findall(r"[a-z0-9]+", text))
    for term, _code in _REFUSALS.items():
        if term in tokens or term in text:
            return "forbidden"
    if _has_any(text, _FUTURE_FORECAST_PHRASES):
        return "forbidden"
    if _has_any(text, _HISTORICAL_FORECAST_PHRASES):
        return "forecast_replay"

    operational = _has_any(
        text,
        (
            "estado operacional",
            "saude operacional",
            "situacao operacional",
            "deployment",
            "deploy",
            "implantacao",
            "versao ativa",
            "qualidade",
            "freshness",
            "atualidade",
            "frescura",
            "recencia",
            "performance",
            "desempenho",
            "mae",
            "erro medio",
            "drift",
            "desvio",
            "deriva",
            "alertas",
            "avisos",
            "modelo",
            "model",
            "metadados",
        ),
    )
    documentary = _has_any(text, _DOCUMENTARY_TERMS)
    if operational and documentary:
        return "ambiguous"
    if operational:
        return "operational"
    if documentary:
        return "documentary"
    return "unsupported"


def refusal_code(question: str) -> str:
    """Return the stable refusal code for a non-routable question."""
    text = _normalize(question)
    tokens = set(re.findall(r"[a-z0-9]+", text))
    for term, code in _REFUSALS.items():
        if term in tokens or term in text:
            return code
    if _has_any(text, _FUTURE_FORECAST_PHRASES):
        return "future_forecast_not_supported"
    if _has_any(text, _HISTORICAL_FORECAST_PHRASES):
        return "forecast_replay_required"
    if classify_question(question) == "ambiguous":
        return "ambiguous_question"
    return "unsupported_question"


class DeterministicPortugueseSelector:
    """Select exactly one latest operational query from bounded text."""

    def select(
        self,
        request: CopilotRequest,
        *,
        tools: tuple[OperationalToolDefinition, ...],
        timeout_seconds: float,
    ) -> object:
        del tools, timeout_seconds
        text = _normalize(request.question)
        tokens = set(re.findall(r"[a-z0-9]+", text))
        route = classify_question(request.question)
        if route == "forecast_replay":
            raise CopilotSelectionRefusal("forecast_replay_required")
        if route == "forbidden":
            raise CopilotSelectionRefusal(refusal_code(request.question))
        if route == "ambiguous":
            raise CopilotSelectionRefusal("ambiguous_question")
        if route != "operational":
            raise CopilotSelectionRefusal("unsupported_question")
        for term, code in _REFUSALS.items():
            if term in tokens or term in text:
                raise CopilotSelectionRefusal(code)

        matches: list[str] = []
        if _has_any(text, ("estado operacional", "saude operacional", "situacao operacional", *_SYNONYMS["estado"])):
            matches.append("operational_summary")
        if _has_any(text, ("deployment ativo", "deploy ativo", "implantacao ativa", "versao ativa", *_SYNONYMS["deployment"])):
            matches.append("active_deployment")
        if _has_any(text, ("qualidade", "freshness", "atualidade", "completude", *_SYNONYMS["qualidade"], *_SYNONYMS["freshness"])):
            matches.append("data_quality")
        if _has_any(text, ("performance", "desempenho", "mae", "erro medio", *_SYNONYMS["performance"])):
            matches.append("monitoring_performance")
        if _has_any(text, ("drift", "desvio", "deriva", *_SYNONYMS["drift"])):
            matches.append("monitoring_drift")
        if _has_any(text, ("alertas", "avisos", *_SYNONYMS["alertas"])):
            matches.append("monitoring_alerts")
        if _has_any(text, ("modelo ativo", "model ativo", "metadados do modelo", "metadados que identificam", *_SYNONYMS["modelo"])):
            matches.append("active_model_metadata")
        if len(set(matches)) != 1:
            raise CopilotSelectionRefusal(
                "unsupported_question" if not matches else "ambiguous_question"
            )
        query_kind = matches[0]
        window = _WINDOW.search(text)
        window_days = int(window.group(1)) if window else None
        if query_kind in {"monitoring_performance", "monitoring_drift"} and window_days is None:
            raise CopilotSelectionRefusal("window_required")
        arguments: dict[str, Any] = {
            "contract_version": "operational_read_only_copilot_v1",
            "query_kind": query_kind,
            "selector": {"kind": "latest"},
            "window_days": window_days,
            "pagination": None,
        }
        return OperationalToolSelection(tool_name="operational_query", arguments=arguments)


__all__ = [
    "DeterministicPortugueseSelector",
    "QuestionRoute",
    "classify_question",
    "refusal_code",
]
