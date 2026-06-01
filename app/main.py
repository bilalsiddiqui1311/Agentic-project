from fastapi import FastAPI

from app.agents.basic_agent import build_basic_agent
from app.schemas import AgentRequest, AgentResponse

app = FastAPI(
    title="Agentic Learning Lab",
    version="0.1.0",
    description="Project 01: a tiny Dockerized agent that can choose and run tools.",
)

agent = build_basic_agent()


@app.get("/")
def read_root() -> dict[str, object]:
    return {
        "message": "Agentic Learning Lab is running.",
        "project": "01-basic-agent",
        "try": {
            "docs": "/docs",
            "run_agent": "POST /agent/run",
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
