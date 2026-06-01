# Agentic Project

A Dockerized learning lab for building agentic AI systems from first principles.

This repo starts with the smallest useful version of an agent. It does not need
an API key yet. The goal is to learn the shape of an agent before adding LLMs,
RAG, memory, and multi-agent workflows.

## Project 01: Basic Tool-Using Agent

An agent is a small decision loop:

```text
User goal
  -> choose a tool
  -> run the tool
  -> observe the result
  -> produce a final answer
```

This starter app includes:

- A FastAPI server
- A `BasicAgent` controller
- A `calculator` tool for arithmetic
- A `knowledge_search` tool for basic agent/RAG concepts
- Tests
- Docker and Docker Compose

## Run With Docker Compose

```bash
docker compose up --build
```

Open:

- API root: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Try the agent:

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "What is RAG?"}'
```

Calculator example:

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "Calculate 42 * 7"}'
```

## Run Tests

With Docker:

```bash
docker compose run --rm agent-api pytest
```

Or locally:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## How To Create An Agent

At the basic level, you need five pieces:

1. **Goal input**: what the user wants.
2. **Tools**: functions the agent can use, such as search, calculator, database, email, or APIs.
3. **Controller**: the logic that decides which tool to use.
4. **Observation**: the result returned by the tool.
5. **Final answer**: the response given back to the user.

In this repo:

- The controller lives in `app/agents/basic_agent.py`
- Tools live in `app/tools/`
- The API lives in `app/main.py`

The current controller is rule-based so it is easy to understand. Later, we can
replace the rule-based controller with an LLM and keep the same general shape.

## Where RAG Fits

RAG means Retrieval-Augmented Generation.

```text
Documents
  -> split into chunks
  -> create embeddings
  -> store in vector database
  -> retrieve relevant chunks for a question
  -> ask the LLM to answer using those chunks
```

RAG is usually the next step after a basic tool-using agent. The retrieval system
becomes one of the agent's tools.

## Learning Path

1. **Basic agent**: learn the agent loop and tool calling.
2. **RAG agent**: add document upload, embeddings, and vector search.
3. **Memory agent**: remember user preferences and previous task state.
4. **Workflow agent**: combine deterministic business rules with LLM decisions.
5. **Multi-agent system**: split work across specialist agents only when the task is complex enough.
