from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_run_agent_endpoint() -> None:
    response = client.post("/agent/run", json={"message": "How does an agent use tools?"})

    assert response.status_code == 200
    body = response.json()
    assert body["used_tool"] == "rag_search"
    assert "Sources:" in body["final_answer"]


def test_rag_documents_endpoint() -> None:
    response = client.get("/rag/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks"] >= 3
    assert {document["source"] for document in body["documents"]} >= {
        "agent-basics.md",
        "rag-basics.md",
        "project-02-rag-agent.md",
    }


def test_rag_config_endpoint() -> None:
    response = client.get("/rag/config")

    assert response.status_code == 200
    body = response.json()
    assert body["answer_mode"]
    assert "openai_configured" in body
    assert "ollama_configured" in body
    assert "ollama_base_url" in body


def test_rag_query_endpoint() -> None:
    response = client.post(
        "/rag/query",
        json={"query": "What is the RAG pipeline?", "top_k": 2, "include_prompt": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert "RAG pipeline" in body["answer"]
    assert body["answer_mode"] == "local_extractive"
    assert body["answer_model"] is None
    assert "You are a grounded RAG answerer." in body["prompt"]
    assert body["sources"]
    assert any(source["source"] == "rag-basics.md" for source in body["sources"])


def test_rag_query_can_answer_production_rag_limitations() -> None:
    response = client.post(
        "/rag/query",
        json={
            "query": "How do chunking, reranking, and hybrid search address production RAG limitations?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"]
    assert any(
        source["source"] == "production-rag-limitations.md"
        for source in body["sources"]
    )
    assert "Chunking" in body["answer"]
    assert "Hybrid search" in body["answer"]
    assert "Reranking" in body["answer"]
    assert "Project 02" not in body["answer"]
    assert "Sources:" in body["answer"]
