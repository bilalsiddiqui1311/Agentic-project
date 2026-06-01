from app.rag.service import RagService
from app.tools.base import ToolOutput


class RagSearchTool:
    name = "rag_search"
    description = "Retrieves relevant chunks from local documents and answers with sources."

    def __init__(self, rag_service: RagService) -> None:
        self.rag_service = rag_service

    def run(self, query: str) -> ToolOutput:
        result = self.rag_service.query(query, top_k=3)
        return ToolOutput(name=self.name, content=result.answer)
