import os
from collections import Counter
from pathlib import Path

from app.rag.answerer import (
    Answerer,
    LocalGroundedAnswerer,
    OllamaAnswerer,
    OpenAIAnswerer,
)
from app.rag.chunker import chunk_documents
from app.rag.document_loader import load_documents
from app.rag.embeddings import HashingEmbeddingModel
from app.rag.models import RagAnswer
from app.rag.prompting import build_grounded_prompt
from app.rag.vector_store import InMemoryVectorStore


class RagService:
    def __init__(self, documents_path: Path, answerer: Answerer | None = None) -> None:
        self.documents_path = documents_path
        self.embedding_model = HashingEmbeddingModel()
        self.documents = load_documents(documents_path)
        self.chunks = chunk_documents(self.documents)
        self.vector_store = InMemoryVectorStore(self.embedding_model)
        self.vector_store.add_chunks(self.chunks)
        self.answerer = answerer or build_answerer_from_env()

    @property
    def chunk_count(self) -> int:
        return self.vector_store.chunk_count

    def query(self, query: str, top_k: int = 3, include_prompt: bool = False) -> RagAnswer:
        matches = self.vector_store.search(query, top_k=top_k)
        grounded_prompt = build_grounded_prompt(query, matches)
        generated_answer = self.answerer.answer(query, matches, grounded_prompt)
        prompt = grounded_prompt if include_prompt else None
        return RagAnswer(
            query=query,
            answer=generated_answer.text,
            matches=matches,
            answer_mode=generated_answer.mode,
            answer_model=generated_answer.model,
            prompt=prompt,
        )

    def document_summary(self) -> list[dict[str, int | str]]:
        chunk_counts = Counter(chunk.source for chunk in self.chunks)
        return [
            {
                "source": document.source,
                "chunks": chunk_counts[document.source],
                "characters": len(document.text),
            }
            for document in self.documents
        ]

    def runtime_summary(self) -> dict[str, bool | str | None]:
        return {
            "answer_mode": self.answerer.mode,
            "answer_model": getattr(self.answerer, "model", None),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "ollama_configured": self.answerer.mode == "ollama_llm",
            "ollama_base_url": getattr(self.answerer, "base_url", None),
            "ollama_timeout_seconds": getattr(self.answerer, "timeout_seconds", None),
        }


def build_answerer_from_env() -> Answerer:
    requested_mode = os.getenv("RAG_ANSWER_MODE", "local").strip().lower()
    api_key_available = bool(os.getenv("OPENAI_API_KEY"))

    if requested_mode in {"ollama", "ollama_llm", "local_llm"}:
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
        return OllamaAnswerer(
            model=model,
            base_url=base_url,
            timeout_seconds=timeout,
        )

    if requested_mode in {"openai", "openai_llm", "llm"} and api_key_available:
        model = os.getenv("OPENAI_MODEL", "gpt-5.5")
        timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
        return OpenAIAnswerer(model=model, timeout_seconds=timeout)

    if requested_mode == "auto" and api_key_available:
        model = os.getenv("OPENAI_MODEL", "gpt-5.5")
        timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
        return OpenAIAnswerer(model=model, timeout_seconds=timeout)

    return LocalGroundedAnswerer()


def build_rag_service(documents_path: Path) -> RagService:
    return RagService(documents_path=documents_path)
