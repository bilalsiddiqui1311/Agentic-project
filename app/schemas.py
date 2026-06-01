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
