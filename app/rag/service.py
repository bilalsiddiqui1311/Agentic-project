from collections import Counter
from pathlib import Path

from app.rag.answerer import LocalGroundedAnswerer
from app.rag.chunker import chunk_documents
from app.rag.document_loader import load_documents
from app.rag.embeddings import HashingEmbeddingModel
from app.rag.models import RagAnswer
from app.rag.prompting import build_grounded_prompt
from app.rag.vector_store import InMemoryVectorStore


class RagService:
    def __init__(self, documents_path: Path) -> None:
        self.documents_path = documents_path
        self.embedding_model = HashingEmbeddingModel()
        self.documents = load_documents(documents_path)
        self.chunks = chunk_documents(self.documents)
        self.vector_store = InMemoryVectorStore(self.embedding_model)
        self.vector_store.add_chunks(self.chunks)
        self.answerer = LocalGroundedAnswerer()

    @property
    def chunk_count(self) -> int:
        return self.vector_store.chunk_count

    def query(self, query: str, top_k: int = 3, include_prompt: bool = False) -> RagAnswer:
        matches = self.vector_store.search(query, top_k=top_k)
        generated_answer = self.answerer.answer(query, matches)
        prompt = build_grounded_prompt(query, matches) if include_prompt else None
        return RagAnswer(
            query=query,
            answer=generated_answer.text,
            matches=matches,
            answer_mode=generated_answer.mode,
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

def build_rag_service(documents_path: Path) -> RagService:
    return RagService(documents_path=documents_path)
