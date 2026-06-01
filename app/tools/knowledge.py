from dataclasses import dataclass

from app.tools.base import ToolOutput


@dataclass(frozen=True)
class KnowledgeItem:
    title: str
    keywords: tuple[str, ...]
    content: str


KNOWLEDGE_BASE = [
    KnowledgeItem(
        title="Agent",
        keywords=("agent", "agentic", "goal", "loop", "tool"),
        content=(
            "An AI agent is a program that takes a user goal, decides what action is "
            "needed, optionally calls tools, observes the result, and returns an answer "
            "or completes a task."
        ),
    ),
    KnowledgeItem(
        title="RAG",
        keywords=("rag", "retrieval", "embedding", "vector", "documents"),
        content=(
            "RAG means Retrieval-Augmented Generation. The app retrieves relevant "
            "chunks from documents or a database, gives those chunks to the model, "
            "and asks the model to answer using that context."
        ),
    ),
    KnowledgeItem(
        title="Tool Calling",
        keywords=("tool", "function", "api", "calculator", "database"),
        content=(
            "Tool calling lets an agent use external capabilities such as calculators, "
            "search, APIs, databases, email, or files instead of relying only on text generation."
        ),
    ),
    KnowledgeItem(
        title="Memory",
        keywords=("memory", "state", "history", "preference", "context"),
        content=(
            "Memory stores useful information across steps or conversations. Short-term "
            "memory keeps the current task state; long-term memory stores durable facts "
            "such as user preferences or prior decisions."
        ),
    ),
]


class KnowledgeSearchTool:
    name = "knowledge_search"
    description = "Searches a tiny built-in knowledge base about agents and RAG."

    def run(self, query: str) -> ToolOutput:
        query_words = set(query.lower().split())
        ranked_items = sorted(
            KNOWLEDGE_BASE,
            key=lambda item: self._score(item, query_words, query.lower()),
            reverse=True,
        )
        best_items = [item for item in ranked_items if self._score(item, query_words, query.lower()) > 0]

        if not best_items:
            return ToolOutput(
                name=self.name,
                content=(
                    "I did not find an exact match, but this starter agent currently knows "
                    "about agents, RAG, tool calling, and memory."
                ),
            )

        summary = " ".join(item.content for item in best_items[:2])
        titles = ", ".join(item.title for item in best_items[:2])
        return ToolOutput(name=self.name, content=f"Matched: {titles}. {summary}")

    def _score(self, item: KnowledgeItem, query_words: set[str], query: str) -> int:
        keyword_hits = sum(2 for keyword in item.keywords if keyword in query)
        title_hit = 3 if item.title.lower() in query else 0
        content_words = set(item.content.lower().replace(".", "").replace(",", "").split())
        overlap = len(query_words.intersection(content_words))
        return keyword_hits + title_hit + overlap
