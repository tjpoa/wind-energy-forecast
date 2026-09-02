"""Bounded Portuguese selector for the local operational Copilot."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .operational_copilot_models import (
    CopilotRequest,
    OperationalToolDefinition,
    OperationalToolSelection,
)


_SPACE = re.compile(r"\s+")
_WINDOW = re.compile(r"\b(30|90)\s*(?:dias|d)\b")
_REFUSALS = {
    "previsao": "future_forecast_not_supported",
    "previsoes": "future_forecast_not_supported",
    "treino": "training_not_supported",
    "treinar": "training_not_supported",
    "retraining": "training_not_supported",
    "promover": "write_operation_not_supported",
    "promocao": "write_operation_not_supported",
    "deployar": "write_operation_not_supported",
    "rollback": "write_operation_not_supported",
    "escrever": "write_operation_not_supported",
    "causa": "undocumented_cause_not_supported",
    "porque": "undocumented_cause_not_supported",
}
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
        if _has_any(text, ("previsoes historicas", "observacoes historicas", "historico de previs")):
            raise ValueError("forecast_replay_required")
        for term, code in _REFUSALS.items():
            if term in tokens or term in text:
                raise ValueError(code)

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
            raise ValueError("unsupported_question" if not matches else "ambiguous_question")
        query_kind = matches[0]
        window = _WINDOW.search(text)
        window_days = int(window.group(1)) if window else None
        if query_kind in {"monitoring_performance", "monitoring_drift"} and window_days is None:
            raise ValueError("window_required")
        arguments: dict[str, Any] = {
            "contract_version": "operational_read_only_copilot_v1",
            "query_kind": query_kind,
            "selector": {"kind": "latest"},
            "window_days": window_days,
            "pagination": None,
        }
        return OperationalToolSelection(tool_name="operational_query", arguments=arguments)


__all__ = ["DeterministicPortugueseSelector"]
