"""OpenAI Responses adapter for documentary synthesis only."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

from .documentary_copilot import DocumentChunk, SynthesisResult


OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
MAX_RESPONSE_BYTES = 64 * 1024


class ResponseLike(Protocol):
    status_code: int
    content: bytes

    def json(self) -> Any: ...


class ResponsesTransport(Protocol):
    def post(self, url: str, **kwargs: Any) -> ResponseLike: ...


def _default_transport() -> ResponsesTransport:
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    session.trust_env = False
    adapter = HTTPAdapter(max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@dataclass(frozen=True)
class OpenAIDocumentarySynthesizer:
    model: str
    api_key: str
    transport: ResponsesTransport | None = None

    def synthesize(
        self, question: str, chunks: tuple[DocumentChunk, ...]
    ) -> SynthesisResult:
        if not 1 <= len(chunks) <= 3:
            raise ValueError("one to three chunks are required")
        payload = {
            "model": self.model,
            "store": False,
            "stream": False,
            "tools": [],
            "tool_choice": "none",
            "max_output_tokens": 400,
            "instructions": "Responda apenas com base nos chunks. Cite pelo menos um chunk_id fornecido.",
            "input": json.dumps(
                {
                    "question": question,
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "heading": chunk.heading_path,
                            "text": chunk.body,
                        }
                        for chunk in chunks
                    ],
                },
                ensure_ascii=False,
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "documentary_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "summary": {"type": "string", "maxLength": 1200},
                            "chunk_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 3,
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["summary", "chunk_ids"],
                    },
                }
            },
        }
        transport = self.transport or _default_transport()
        response = transport.post(
            OPENAI_RESPONSES_ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=5.0,
            allow_redirects=False,
        )
        if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
            raise RuntimeError("invalid provider response")
        body = response.json()
        if body.get("model") != self.model or body.get("status") != "completed":
            raise RuntimeError("unexpected provider response")
        texts = [
            content.get("text")
            for item in body.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        ]
        if len(texts) != 1 or not isinstance(texts[0], str):
            raise RuntimeError("missing structured output")
        result = SynthesisResult.model_validate_json(texts[0], strict=True)
        if not set(result.chunk_ids).issubset({chunk.chunk_id for chunk in chunks}):
            raise ValueError("unknown provider citation")
        return result


__all__ = ["OpenAIDocumentarySynthesizer", "ResponsesTransport"]
