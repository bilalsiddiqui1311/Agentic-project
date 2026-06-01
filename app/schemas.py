from typing import Literal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        examples=["What is RAG?", "Calculate 42 * 7", "How does an agent use tools?"],
    )


class AgentStep(BaseModel):
    step: int
    kind: Literal["thought", "action", "observation", "final"]
    content: str


class AgentResponse(BaseModel):
    input: str
    final_answer: str
    used_tool: str | None
    steps: list[AgentStep]


class RagQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        examples=["What is the RAG pipeline?", "How does Project 02 use documents?"],
    )
    top_k: int = Field(default=3, ge=1, le=5)


class RagSource(BaseModel):
    source: str
    chunk_id: str
    score: float
    preview: str


class RagQueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[RagSource]


class RagDocumentSummary(BaseModel):
    source: str
    chunks: int
    characters: int


class RagIndexResponse(BaseModel):
    documents: list[RagDocumentSummary]
    total_chunks: int
