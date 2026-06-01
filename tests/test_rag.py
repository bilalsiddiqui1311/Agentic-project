from pathlib import Path

from app.rag.service import build_rag_service


def test_rag_service_retrieves_relevant_document_chunks(tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "rag.txt").write_text(
        "RAG retrieves relevant chunks, creates embeddings, and answers with sources.",
        encoding="utf-8",
    )
    (docs_path / "memory.txt").write_text(
        "Memory stores preferences and conversation state.",
        encoding="utf-8",
    )

    service = build_rag_service(docs_path)
    result = service.query("How does RAG use embeddings?", top_k=1)

    assert result.matches
    assert result.matches[0].chunk.source == "rag.txt"
    assert "Based on the retrieved documents" in result.answer


def test_rag_service_returns_helpful_empty_state(tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "tools.txt").write_text(
        "Tools are functions used by an agent.",
        encoding="utf-8",
    )

    service = build_rag_service(docs_path)
    result = service.query("bananas", top_k=3)

    assert result.matches == []
    assert "could not find relevant context" in result.answer


def test_rag_service_answers_profile_profession_questions(tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "profile.md").write_text(
        "# Muhammad Bilal Siddiqui\n\n**Senior DevOps Engineer**\n\n"
        "Bilal works on cloud infrastructure, CI/CD, and automation.",
        encoding="utf-8",
    )
    (docs_path / "rag.txt").write_text(
        "RAG retrieves relevant chunks before answering.",
        encoding="utf-8",
    )

    service = build_rag_service(docs_path)
    result = service.query("WWhat the profession of Bilal?", top_k=3)

    assert result.matches
    assert result.matches[0].chunk.source == "profile.md"
    assert result.answer == "Bilal's profession is Senior DevOps Engineer. Sources: profile.md."
