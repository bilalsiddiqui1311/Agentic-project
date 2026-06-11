from pathlib import Path

import json

from app.rag.answerer import GeneratedAnswer, OllamaAnswerer
from app.rag.models import Chunk, SearchMatch
from app.rag.service import RagService, build_answerer_from_env, build_rag_service


class FakeLLMAnswerer:
    mode = "fake_llm"
    model = "fake-model"

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def answer(
        self,
        query: str,
        matches: list[SearchMatch],
        prompt: str,
    ) -> GeneratedAnswer:
        self.last_prompt = prompt
        return GeneratedAnswer(
            text="Fake LLM answer with sources.",
            mode=self.mode,
            model=self.model,
        )


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
    assert result.answer_mode == "local_extractive"
    assert "RAG retrieves relevant chunks" in result.answer


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
    assert result.answer_mode == "local_extractive"
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


def test_rag_service_can_include_grounded_prompt(tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "rag.txt").write_text(
        "RAG answers should use retrieved context and show sources.",
        encoding="utf-8",
    )

    service = build_rag_service(docs_path)
    result = service.query("How should RAG answer?", top_k=1, include_prompt=True)

    assert result.prompt is not None
    assert "You are a grounded RAG answerer." in result.prompt
    assert "How should RAG answer?" in result.prompt
    assert "Source: rag.txt" in result.prompt


def test_rag_service_respects_numeric_project_identifiers(tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "project-03.txt").write_text(
        "Project 03 separates retrieval from answer generation.",
        encoding="utf-8",
    )
    (docs_path / "project-management.txt").write_text(
        "Project management is planning, organizing, and managing resources.",
        encoding="utf-8",
    )

    service = build_rag_service(docs_path)
    result = service.query("What is Project 03?", top_k=3)

    assert result.matches
    assert [match.chunk.source for match in result.matches] == ["project-03.txt"]


def test_rag_service_matches_project_numbers_with_leading_zeroes(tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "project-03.txt").write_text(
        "Project 03 separates retrieval from answer generation.",
        encoding="utf-8",
    )

    service = build_rag_service(docs_path)
    result = service.query("What is the project 3?", top_k=3)

    assert result.matches
    assert result.matches[0].chunk.source == "project-03.txt"
    assert "Project 03 separates retrieval from answer generation" in result.answer


def test_rag_service_can_use_injected_llm_answerer(tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "rag.txt").write_text(
        "RAG uses retrieval before answer generation.",
        encoding="utf-8",
    )
    answerer = FakeLLMAnswerer()

    service = RagService(docs_path, answerer=answerer)
    result = service.query("How does RAG answer?", top_k=1, include_prompt=True)

    assert result.answer == "Fake LLM answer with sources."
    assert result.answer_mode == "fake_llm"
    assert result.answer_model == "fake-model"
    assert answerer.last_prompt is not None
    assert "Source: rag.txt" in answerer.last_prompt
    assert result.prompt == answerer.last_prompt


def test_build_answerer_from_env_can_select_ollama(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ANSWER_MODE", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")

    answerer = build_answerer_from_env()

    assert answerer.mode == "ollama_llm"
    assert getattr(answerer, "model") == "qwen2.5:0.5b"
    assert getattr(answerer, "base_url") == "http://ollama:11434"


def test_ollama_answerer_posts_grounded_prompt(monkeypatch) -> None:
    captured_request = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"message": {"content": "Grounded Ollama answer. Sources: rag.txt."}}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured_request["url"] = request.full_url
        captured_request["timeout"] = timeout
        captured_request["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("app.rag.answerer.urlopen", fake_urlopen)
    answerer = OllamaAnswerer(
        model="qwen2.5:0.5b",
        base_url="http://ollama:11434/",
        timeout_seconds=12,
    )
    matches = [
        SearchMatch(
            chunk=Chunk(
                id="rag.txt:0",
                source="rag.txt",
                index=0,
                text="RAG uses retrieved context before answering.",
            ),
            score=1.0,
        )
    ]

    result = answerer.answer("How does RAG answer?", matches, "Grounded prompt")

    assert result.text == "Grounded Ollama answer. Sources: rag.txt."
    assert result.mode == "ollama_llm"
    assert result.model == "qwen2.5:0.5b"
    assert captured_request["url"] == "http://ollama:11434/api/chat"
    assert captured_request["timeout"] == 12
    assert captured_request["body"]["model"] == "qwen2.5:0.5b"
    assert captured_request["body"]["stream"] is False
    assert captured_request["body"]["messages"][-1]["content"] == "Grounded prompt"
