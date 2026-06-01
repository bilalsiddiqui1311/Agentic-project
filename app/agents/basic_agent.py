import re

from app.schemas import AgentResponse, AgentStep
from app.tools import CalculatorTool, KnowledgeSearchTool
from app.tools.base import Tool, ToolOutput


class BasicAgent:
    """A tiny controller that demonstrates the agent loop without needing an LLM key."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def describe_tools(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]

    def run(self, message: str) -> AgentResponse:
        steps: list[AgentStep] = []
        steps.append(
            self._step(
                steps,
                "thought",
                "Classify the request, choose a useful tool, then answer from the observation.",
            )
        )

        tool = self._select_tool(message)
        steps.append(self._step(steps, "action", f"Call tool: {tool.name}"))

        try:
            observation = tool.run(message)
        except ValueError as exc:
            observation = ToolOutput(name=tool.name, content=str(exc))

        steps.append(self._step(steps, "observation", observation.content))

        final_answer = self._compose_answer(observation)
        steps.append(self._step(steps, "final", final_answer))

        return AgentResponse(
            input=message,
            final_answer=final_answer,
            used_tool=tool.name,
            steps=steps,
        )

    def _select_tool(self, message: str) -> Tool:
        lowered = message.lower()
        has_number = bool(re.search(r"\d", lowered))
        looks_like_math = has_number and bool(
            re.search(r"[\+\-\*/%()]|calculate|math|multiply|divide|sum|plus|minus", lowered)
        )

        if looks_like_math:
            return self.tools["calculator"]

        return self.tools["knowledge_search"]

    def _compose_answer(self, observation: ToolOutput) -> str:
        if observation.name == "calculator":
            return f"The calculation result is: {observation.content}"

        return observation.content

    def _step(self, existing_steps: list[AgentStep], kind: str, content: str) -> AgentStep:
        return AgentStep(step=len(existing_steps) + 1, kind=kind, content=content)


def build_basic_agent() -> BasicAgent:
    return BasicAgent(tools=[CalculatorTool(), KnowledgeSearchTool()])
