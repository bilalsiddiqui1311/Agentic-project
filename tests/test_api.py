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
    assert body["used_tool"] == "knowledge_search"
    assert "Tool calling" in body["final_answer"]
