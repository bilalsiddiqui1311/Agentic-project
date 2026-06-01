from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    index: int
    text: str


@dataclass(frozen=True)
class SearchMatch:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    matches: list[SearchMatch]
    answer_mode: str
    answer_model: str | None
    prompt: str | None
