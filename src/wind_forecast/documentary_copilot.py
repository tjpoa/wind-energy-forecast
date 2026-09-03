"""Closed-corpus, deterministic documentary retrieval and optional synthesis."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Protocol
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .operational_copilot_models import (
    CopilotDocumentaryResponse,
    DocumentaryAnswer,
    DocumentaryEvidence,
    ProviderFailure,
)
from .paths import project_root


ALLOWED_DOCUMENTS = ("README.md", "OPERATIONS.md", "APP_COPILOT_ROADMAP.md")
MAX_CHUNKS = 3
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ManifestDocument(_StrictModel):
    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    path: str
    uri: str = Field(pattern=r"^docs://")
    title: str = Field(min_length=1)
    sensitivity: str = Field(pattern=r"^public$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    updated_at: str = Field(min_length=1)


class CorpusManifest(_StrictModel):
    schema_version: str = Field(pattern=r"^document_corpus_v1$")
    documents: tuple[ManifestDocument, ...]


class DocumentChunk(_StrictModel):
    chunk_id: str
    document_id: str
    document_title: str
    heading_path: tuple[str, ...]
    ordinal: int
    uri: str
    document_sha256: str
    document_updated_at: str
    body: str


class SynthesisResult(_StrictModel):
    summary: str = Field(min_length=1, max_length=1200)
    chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_CHUNKS)


class DocumentarySynthesizer(Protocol):
    def synthesize(
        self, question: str, chunks: tuple[DocumentChunk, ...]
    ) -> SynthesisResult: ...


def load_corpus(
    manifest_path: Path | None = None, *, root: Path | None = None
) -> tuple[CorpusManifest, tuple[DocumentChunk, ...]]:
    root = (root or project_root()).resolve()
    manifest_path = (manifest_path or root / "config/document_corpus_v1.json").resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("documents"), list):
            raw["documents"] = tuple(raw["documents"])
        manifest = CorpusManifest.model_validate(raw, strict=True)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("document corpus manifest is invalid") from exc
    if tuple(item.path for item in manifest.documents) != ALLOWED_DOCUMENTS:
        raise ValueError("document corpus allowlist is invalid")
    if len({item.id for item in manifest.documents}) != len(ALLOWED_DOCUMENTS):
        raise ValueError("document identifiers must be unique")
    chunks: list[DocumentChunk] = []
    for item in manifest.documents:
        relative = Path(item.path)
        if relative.is_absolute() or relative.parts != (item.path,):
            raise ValueError("document path is unsafe")
        path = (root / relative).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError("document is missing or outside the corpus")
        raw_content = path.read_bytes()
        try:
            content = raw_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("document is not UTF-8") from exc
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        digest = sha256(content.encode("utf-8")).hexdigest()
        if digest != item.sha256:
            raise ValueError("document hash mismatch")
        chunks.extend(chunk_markdown(item, content))
    return manifest, tuple(chunks)


def chunk_markdown(
    document: ManifestDocument, content: str
) -> tuple[DocumentChunk, ...]:
    headings: list[str] = []
    section_lines: list[str] = []
    sections: list[tuple[tuple[str, ...], str]] = []

    def flush() -> None:
        body = "\n".join(section_lines).strip()
        if body:
            sections.append((tuple(headings), body))

    for line in content.splitlines():
        match = _HEADING.match(line)
        if not match:
            section_lines.append(line)
            continue
        flush()
        section_lines = []
        level = len(match.group(1))
        headings = headings[: level - 1]
        headings.append(match.group(2).strip())
    flush()
    chunks = []
    for ordinal, (heading_path, body) in enumerate(sections):
        identity = json.dumps(
            [document.id, document.version, ordinal, heading_path, body],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.id}:{sha256(identity.encode('utf-8')).hexdigest()[:20]}",
                document_id=document.id,
                document_title=document.title,
                heading_path=heading_path,
                ordinal=ordinal,
                uri=f"{document.uri}#{ordinal}",
                document_sha256=document.sha256,
                document_updated_at=document.updated_at,
                body=body,
            )
        )
    return tuple(chunks)


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    folded = "".join(char for char in normalized if not unicodedata.combining(char))
    return tuple(_TOKEN.findall(folded))


def retrieve(
    question: str, chunks: tuple[DocumentChunk, ...]
) -> tuple[DocumentChunk, ...]:
    query = set(_tokens(question))
    if not query:
        return ()
    ranked: list[tuple[int, int, int, DocumentChunk]] = []
    document_order: dict[str, int] = {}
    for chunk in chunks:
        document_order.setdefault(chunk.document_id, len(document_order))
    for chunk in chunks:
        heading = set(_tokens(" ".join((chunk.document_title, *chunk.heading_path))))
        body = set(_tokens(chunk.body))
        score = 4 * len(query & heading) + len(query & body)
        if score:
            ranked.append(
                (-score, document_order[chunk.document_id], chunk.ordinal, chunk)
            )
    ranked.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in ranked[:MAX_CHUNKS])


_DOCUMENTARY_TERMS = frozenset(
    _tokens(
        "metodologia limitações limitacoes decisões decisoes execução execucao executar roadmap documentação documentacao arquitetura projeto"
    )
)


def is_documentary_question(question: str) -> bool:
    return bool(set(_tokens(question)) & _DOCUMENTARY_TERMS)


def _extractive_summary(chunk: DocumentChunk, limit: int = 480) -> str:
    compact = " ".join(chunk.body.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def answer_documentary(
    question: str,
    chunks: tuple[DocumentChunk, ...],
    synthesizer: DocumentarySynthesizer | None = None,
) -> CopilotDocumentaryResponse | None:
    selected = retrieve(question, chunks)
    if not selected:
        return None
    evidence = tuple(
        DocumentaryEvidence(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.document_title,
            heading=" > ".join(chunk.heading_path) or "Raiz",
            uri=chunk.uri,
            sha256=chunk.document_sha256,
            updated_at=chunk.document_updated_at,
        )
        for chunk in selected
    )
    summary = _extractive_summary(selected[0])
    provider_failure = None
    mode = "rag_local"
    if synthesizer is not None:
        try:
            result = synthesizer.synthesize(question, selected)
            allowed = {chunk.chunk_id for chunk in selected}
            if not set(result.chunk_ids).issubset(allowed):
                raise ValueError("unknown provider citation")
            summary = result.summary
            evidence = tuple(
                item for item in evidence if item.chunk_id in result.chunk_ids
            )
            mode = "rag_openai"
        except Exception:
            provider_failure = ProviderFailure(
                code="document_synthesis_failed",
                message="A síntese remota falhou; foi usada a resposta documental local.",
            )
    return CopilotDocumentaryResponse(
        route="documentary",
        mode=mode,
        answer=DocumentaryAnswer(summary=summary, evidence=evidence),
        limitations=("Resposta limitada ao corpus documental versionado.",),
        provider_failure=provider_failure,
    )


__all__ = [
    "ALLOWED_DOCUMENTS",
    "CorpusManifest",
    "DocumentChunk",
    "DocumentarySynthesizer",
    "ManifestDocument",
    "SynthesisResult",
    "answer_documentary",
    "chunk_markdown",
    "is_documentary_question",
    "load_corpus",
    "retrieve",
]
