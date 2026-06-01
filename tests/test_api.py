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


def test_rag_query_endpoint() -> None:
    response = client.post(
        "/rag/query",
        json={"query": "What is the RAG pipeline?", "top_k": 2, "include_prompt": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert "RAG pipeline" in body["answer"]
    assert body["answer_mode"] == "local_extractive"
    assert "You are a grounded RAG answerer." in body["prompt"]
    assert body["sources"]
    assert any(source["source"] == "rag-basics.md" for source in body["sources"])
