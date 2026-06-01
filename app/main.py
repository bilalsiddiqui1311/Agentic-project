from pathlib import Path

from fastapi import FastAPI

from app.agents.basic_agent import build_basic_agent
from app.rag.service import build_rag_service
from app.schemas import (
    AgentRequest,
    AgentResponse,
    RagDocumentSummary,
    RagIndexResponse,
    RagQueryRequest,
    RagQueryResponse,
    RagSource,
)

DOCUMENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "documents"

app = FastAPI(
    title="Agentic Learning Lab",
    version="0.2.0",
    description="Project 02: a Dockerized agent with local RAG document retrieval.",
)

rag_service = build_rag_service(DOCUMENTS_PATH)
agent = build_basic_agent(rag_service=rag_service)


@app.get("/")
def read_root() -> dict[str, object]:
    return {
        "message": "Agentic Learning Lab is running.",
        "project": "02-rag-agent",
        "try": {
            "docs": "/docs",
            "run_agent": "POST /agent/run",
            "run_rag": "POST /rag/query",
            "rag_documents": "GET /rag/documents",
            "tools": "GET /tools",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> list[dict[str, str]]:
    return agent.describe_tools()


@app.post("/agent/run", response_model=AgentResponse)
def run_agent(request: AgentRequest) -> AgentResponse:
    return agent.run(request.message)


@app.get("/rag/documents", response_model=RagIndexResponse)
def list_rag_documents() -> RagIndexResponse:
    return RagIndexResponse(
        documents=[
            RagDocumentSummary(**document)
            for document in rag_service.document_summary()
        ],
        total_chunks=rag_service.chunk_count,
    )


@app.post("/rag/query", response_model=RagQueryResponse)
def query_rag(request: RagQueryRequest) -> RagQueryResponse:
    result = rag_service.query(request.query, top_k=request.top_k)
    return RagQueryResponse(
        query=result.query,
        answer=result.answer,
        sources=[
            RagSource(
                source=match.chunk.source,
                chunk_id=match.chunk.id,
                score=round(match.score, 4),
                preview=match.chunk.text[:220],
            )
            for match in result.matches
        ],
    )
