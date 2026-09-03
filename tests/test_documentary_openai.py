from __future__ import annotations

import json

import pytest

from wind_forecast.documentary_copilot import ManifestDocument, chunk_markdown
from wind_forecast.documentary_openai import OpenAIDocumentarySynthesizer


class Response:
    status_code = 200

    def __init__(self, body):
        self.content = json.dumps(body).encode()
        self._body = body

    def json(self):
        return self._body


class Transport:
    def __init__(self, body):
        self.body, self.calls = body, []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.body)


def _chunks():
    doc = ManifestDocument(
        id="readme",
        version=1,
        path="README.md",
        uri="docs://x/readme",
        title="README",
        sensitivity="public",
        sha256="0" * 64,
        updated_at="2026-09-03",
    )
    return chunk_markdown(doc, "# Método\nTexto verificável.")


def test_one_bounded_structured_call_with_backend_only_key() -> None:
    chunks = _chunks()
    output = json.dumps({"summary": "Resumo", "chunk_ids": [chunks[0].chunk_id]})
    transport = Transport(
        {
            "model": "configured",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": output}],
                }
            ],
        }
    )
    result = OpenAIDocumentarySynthesizer("configured", "secret", transport).synthesize(
        "método?", chunks
    )
    assert result.summary == "Resumo"
    assert len(transport.calls) == 1
    _, request = transport.calls[0]
    assert request["timeout"] == 5.0 and request["allow_redirects"] is False
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["json"]["store"] is False
    assert request["json"]["tools"] == []
    assert request["json"]["text"]["format"]["strict"] is True
    assert "secret" not in json.dumps(request["json"])


@pytest.mark.parametrize(
    "body",
    [
        {"model": "wrong", "status": "completed", "output": []},
        {"model": "configured", "status": "failed", "output": []},
        {"model": "configured", "status": "completed", "output": []},
    ],
)
def test_invalid_provider_responses_are_rejected(body) -> None:
    with pytest.raises(RuntimeError):
        OpenAIDocumentarySynthesizer(
            "configured", "secret", Transport(body)
        ).synthesize("?", _chunks())
