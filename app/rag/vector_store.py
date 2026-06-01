from dataclasses import dataclass

from app.rag.embeddings import HashingEmbeddingModel, cosine_similarity, tokenize
from app.rag.models import Chunk, SearchMatch


@dataclass(frozen=True)
class StoredChunk:
    chunk: Chunk
    vector: list[float]


class InMemoryVectorStore:
    def __init__(self, embedding_model: HashingEmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self._items: list[StoredChunk] = []

    @property
    def chunk_count(self) -> int:
        return len(self._items)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._items.append(
                StoredChunk(
                    chunk=chunk,
                    vector=self.embedding_model.embed(chunk.text),
                )
            )

    def search(self, query: str, top_k: int = 3) -> list[SearchMatch]:
        query_vector = self.embedding_model.embed(query)
        query_tokens = set(tokenize(query))
        matches = [
            SearchMatch(
                chunk=item.chunk,
                score=self._hybrid_score(query_vector, query_tokens, item),
            )
            for item in self._items
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        positive_matches = [match for match in matches if match.score > 0]

        if not positive_matches:
            return []

        best_score = positive_matches[0].score
        return [
            match
            for match in positive_matches[:top_k]
            if match.score >= best_score * 0.5
        ]

    def _hybrid_score(
        self,
        query_vector: list[float],
        query_tokens: set[str],
        item: StoredChunk,
    ) -> float:
        if not query_tokens:
            return 0.0

        chunk_tokens = set(tokenize(item.chunk.text))
        overlap = query_tokens.intersection(chunk_tokens)
        lexical_score = len(overlap) / len(query_tokens)

        if lexical_score == 0:
            return 0.0

        semantic_score = cosine_similarity(query_vector, item.vector)
        return lexical_score + semantic_score
