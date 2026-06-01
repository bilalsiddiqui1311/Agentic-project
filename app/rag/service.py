from collections import Counter
from pathlib import Path
import re

from app.rag.chunker import chunk_documents
from app.rag.document_loader import load_documents
from app.rag.embeddings import HashingEmbeddingModel, tokenize
from app.rag.models import RagAnswer, SearchMatch
from app.rag.vector_store import InMemoryVectorStore


class RagService:
    def __init__(self, documents_path: Path) -> None:
        self.documents_path = documents_path
        self.embedding_model = HashingEmbeddingModel()
        self.documents = load_documents(documents_path)
        self.chunks = chunk_documents(self.documents)
        self.vector_store = InMemoryVectorStore(self.embedding_model)
        self.vector_store.add_chunks(self.chunks)

    @property
    def chunk_count(self) -> int:
        return self.vector_store.chunk_count

    def query(self, query: str, top_k: int = 3) -> RagAnswer:
        matches = self.vector_store.search(query, top_k=top_k)
        answer = self._compose_answer(query, matches)
        return RagAnswer(query=query, answer=answer, matches=matches)

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

    def _compose_answer(self, query: str, matches: list[SearchMatch]) -> str:
        if not matches:
            return (
                "I could not find relevant context in the indexed documents. "
                "Try asking about agents, RAG, tool calling, memory, or Project 02."
            )

        direct_answer = self._answer_profile_question(query, matches)
        if direct_answer:
            return direct_answer

        context = " ".join(self._clean_context(match.chunk.text) for match in matches[:2])
        sources = ", ".join(dict.fromkeys(match.chunk.source for match in matches[:2]))
        return f"Based on the retrieved documents: {context} Sources: {sources}."

    def _answer_profile_question(self, query: str, matches: list[SearchMatch]) -> str | None:
        query_tokens = set(tokenize(query))
        asks_for_profession = bool(
            query_tokens.intersection({"profession", "role", "title", "job", "occupation"})
        )

        if not asks_for_profession:
            return None

        for match in matches:
            title = self._extract_markdown_bold_title(match.chunk.text)
            if title:
                return f"Bilal's profession is {title}. Sources: {match.chunk.source}."

        return None

    def _extract_markdown_bold_title(self, text: str) -> str | None:
        match = re.search(r"\*\*([^*\n]+)\*\*", text)
        if not match:
            return None

        return match.group(1).strip()

    def _clean_context(self, text: str) -> str:
        return text.replace("#", "").strip()


def build_rag_service(documents_path: Path) -> RagService:
    return RagService(documents_path=documents_path)
