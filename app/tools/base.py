from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolOutput:
    name: str
    content: str


class Tool(Protocol):
    name: str
    description: str

    def run(self, query: str) -> ToolOutput:
        """Run the tool and return a readable observation."""
