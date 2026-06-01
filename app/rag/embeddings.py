import hashlib
import math
import re
from collections import Counter


class HashingEmbeddingModel:
    """Small local embedding model for learning RAG without API keys."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vector = [0.0] * self.dimensions

        if not tokens:
            return vector

        counts = Counter(tokens)
        for token, count in counts.items():
            index = self._index_for_token(token)
            vector[index] += float(count)

        return _normalize(vector)

    def _index_for_token(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, byteorder="big") % self.dimensions


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*|\d+", text.lower())
        if token not in _STOP_WORDS
    ]


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "with",
}
