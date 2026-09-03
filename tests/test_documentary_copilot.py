from __future__ import annotations

from hashlib import sha256
import json

import pytest

from wind_forecast.documentary_copilot import (
    ManifestDocument,
    SynthesisResult,
    answer_documentary,
    chunk_markdown,
    is_documentary_question,
    load_corpus,
    retrieve,
)


def _document(content: str = "texto") -> ManifestDocument:
    return ManifestDocument(
        id="readme",
        version=1,
        path="README.md",
        uri="docs://test/readme",
        title="README",
        sensitivity="public",
        sha256=sha256(content.encode()).hexdigest(),
        updated_at="2026-09-03",
    )


def test_chunking_and_ids_are_deterministic() -> None:
    content = "preâmbulo\n# Um\ncorpo\n## Dois\nmais\n# Vazio\n"
    first = chunk_markdown(_document(content), content)
    assert first == chunk_markdown(_document(content), content)
    assert [chunk.heading_path for chunk in first] == [(), ("Um",), ("Um", "Dois")]
    assert [chunk.ordinal for chunk in first] == [0, 1, 2]


def test_lexical_ranking_prefers_heading_then_stable_order() -> None:
    content = "# Metodologia\nlimitações execução\n# Outra\nmetodologia metodologia"
    chunks = chunk_markdown(_document(content), content)
    assert retrieve("metodologia", chunks)[0].heading_path == ("Metodologia",)


@pytest.mark.parametrize(
    "question",
    [
        "Qual é a metodologia?",
        "Quais são as limitações?",
        "Que decisões foram tomadas?",
        "Como executar o projeto localmente?",
        "Qual é o estado do roadmap?",
    ],
)
def test_five_documentary_families_are_routed(question) -> None:
    assert is_documentary_question(question)


def test_local_answer_has_typed_public_evidence_and_no_physical_path() -> None:
    content = "# Execução local\nExecute docker compose up para iniciar localmente."
    response = answer_documentary(
        "Como é a execução local?", chunk_markdown(_document(content), content)
    )
    assert response is not None
    assert response.mode == "rag_local"
    assert response.failure is None
    assert response.answer.evidence[0].uri.startswith("docs://")
    assert response.answer.evidence[0].updated_at == "2026-09-03"
    assert "README.md" not in response.model_dump_json()


def test_provider_success_and_unknown_citation_fallback() -> None:
    chunks = chunk_markdown(
        _document("# Roadmap\nEstado atual."), "# Roadmap\nEstado atual."
    )

    class Good:
        def synthesize(self, question, selected):
            return SynthesisResult(
                summary="Síntese.", chunk_ids=(selected[0].chunk_id,)
            )

    assert answer_documentary("roadmap", chunks, Good()).mode == "rag_openai"

    class Bad:
        def synthesize(self, question, selected):
            return SynthesisResult(summary="Inventado.", chunk_ids=("unknown",))

    fallback = answer_documentary("roadmap", chunks, Bad())
    assert fallback.mode == "rag_local"
    assert fallback.provider_failure.code == "document_synthesis_failed"


def test_manifest_is_closed_and_hash_verified(tmp_path) -> None:
    names = ("README.md", "OPERATIONS.md", "APP_COPILOT_ROADMAP.md")
    documents = []
    for index, name in enumerate(names):
        content = f"# Documento {index}\nconteúdo"
        (tmp_path / name).write_text(content, encoding="utf-8")
        documents.append(
            {
                "id": ("readme", "operations", "roadmap")[index],
                "version": 1,
                "path": name,
                "uri": f"docs://test/{index}",
                    "title": name,
                    "sensitivity": "public",
                    "sha256": sha256(content.encode()).hexdigest(),
                    "updated_at": "2026-09-03",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "document_corpus_v1", "documents": documents}),
        encoding="utf-8",
    )
    assert len(load_corpus(manifest, root=tmp_path)[1]) == 3
    missing = tmp_path / names[-1]
    missing.unlink()
    with pytest.raises(ValueError, match="missing"):
        load_corpus(manifest, root=tmp_path)
    missing.write_text("# Documento 2\nconteúdo", encoding="utf-8")
    documents[0]["sha256"] = "0" * 64
    manifest.write_text(
        json.dumps({"schema_version": "document_corpus_v1", "documents": documents}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_corpus(manifest, root=tmp_path)


@pytest.mark.parametrize("path", ["../README.md", "/README.md", "EXTRA.md"])
def test_manifest_rejects_paths_outside_allowlist(tmp_path, path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "document_corpus_v1",
                "documents": [
                    {
                        "id": "readme",
                        "version": 1,
                        "path": path,
                        "uri": "docs://x/y",
                        "title": "x",
                        "sensitivity": "public",
                        "sha256": "0" * 64,
                        "updated_at": "2026-09-03",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allowlist"):
        load_corpus(manifest, root=tmp_path)
