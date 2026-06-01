# Agentic Project

A Dockerized learning lab for building agentic AI systems from first principles.

This repo starts with the smallest useful version of an agent and grows one
project at a time. It does not need an API key yet. The goal is to learn the
shape of an agent before adding hosted LLMs, production vector databases,
memory, and multi-agent workflows.

## Project Structure

The runnable app is kept at the repo root so Docker Compose stays simple. Each
learning milestone also has a short project note:

- `projects/project-01-basic-agent/`
- `projects/project-02-rag-agent/`
- `projects/project-03-rag-answering/`
- `projects/project-04-llm-rag-answering/`

Each milestone is also preserved as its own branch:

- `codex/project-01-basic-agent`
- `codex/project-02-rag-agent`
- `codex/project-03-rag-answering`
- `codex/project-04-llm-rag-answering`

## Project 01: Basic Tool-Using Agent

An agent is a small decision loop:

```text
User goal
  -> choose a tool
  -> run the tool
  -> observe the result
  -> produce a final answer
```

Project 01 includes:

- A FastAPI server
- A `BasicAgent` controller
- A `calculator` tool for arithmetic
- A `knowledge_search` tool for basic agent/RAG concepts
- Tests
- Docker and Docker Compose

## Project 02: Local RAG Agent

Project 02 adds a local RAG pipeline:

```text
Documents
  -> chunks
  -> local embedding vectors
  -> in-memory vector search
  -> sourced answer
```

This project includes:

- Sample documents in `data/documents/`
- A document loader
- A chunker
- A lightweight local embedding model
- An in-memory vector store
- A `rag_search` agent tool
- Direct RAG endpoints

The current embedding model is intentionally simple so the app works in Docker
without paid keys or external services. It teaches the RAG architecture first.
Later we can swap it for OpenAI embeddings, Chroma, pgvector, Pinecone, Weaviate,
or another production retrieval stack.

## Project 03: Grounded RAG Answering

Project 03 adds an answer-generation layer after retrieval:

```text
question
  -> retrieve relevant chunks
  -> build grounded prompt
  -> generate concise answer from evidence
  -> return answer mode + sources
```

This project includes:

- A local extractive answerer
- A grounded prompt builder
- `answer_mode` in RAG responses
- Optional `include_prompt` debugging
- Tests for answer generation and prompt construction

This is still no-key by default. The learning goal is to understand where an
LLM fits in the RAG pipeline before plugging in a hosted model.

## Project 04: RAG With Real LLM Answering

Project 04 adds an optional OpenAI answerer using the Responses API:

```text
question
  -> retrieve relevant chunks
  -> build grounded prompt
  -> call LLM when enabled
  -> return answer mode, model, and sources
```

By default, the app still uses `local_extractive` mode so Docker works without
secrets. To enable the OpenAI answerer, configure:

```bash
RAG_ANSWER_MODE=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5.5
```

Then restart:

```bash
docker compose up -d --build
```

Check the active mode:

```bash
curl http://localhost:8000/rag/config
```

## Run With Docker Compose

```bash
docker compose up --build
```

Open:

- API root: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- RAG config: http://localhost:8000/rag/config
- RAG documents: http://localhost:8000/rag/documents

Try the agent:

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "What is RAG?"}'
```

RAG query example:

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the RAG pipeline?", "top_k": 2, "include_prompt": true}'
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

In Project 02, that tool is `rag_search`. When the agent receives a normal
knowledge question, it retrieves document chunks and returns an answer with
sources.

In Project 03, RAG adds an answerer after retrieval. Retrieval finds evidence;
the answerer turns that evidence into a shorter response. The optional prompt
shows what we would send to a real LLM in the next production-style upgrade.

In Project 04, the same prompt can be sent to OpenAI when `RAG_ANSWER_MODE` is
set to `openai` and `OPENAI_API_KEY` is configured. The response includes
`answer_mode` and `answer_model` so you can see whether the answer came from the
local answerer or the LLM.

## Swagger Tests To Try

Use http://localhost:8000/docs and try:

- `GET /tools`: confirm `calculator`, `knowledge_search`, and `rag_search`.
- `GET /rag/config`: confirm `answer_mode` and `answer_model`.
- `GET /rag/documents`: see which files were indexed.
- `POST /rag/query` with `{"query": "What is the RAG pipeline?", "top_k": 2, "include_prompt": true}`.
- `POST /agent/run` with `{"message": "How does Project 02 use documents?"}`.
- `POST /agent/run` with `{"message": "Calculate (10 + 5) * 3"}`.

## Learning Path

1. **Basic agent**: learn the agent loop and tool calling.
2. **RAG agent**: add document upload, embeddings, and vector search.
3. **Grounded answerer**: turn retrieved chunks into cleaner answers.
4. **LLM answerer**: send grounded prompts to a real model.
5. **Memory agent**: remember user preferences and previous task state.
6. **Workflow agent**: combine deterministic business rules with LLM decisions.
7. **Multi-agent system**: split work across specialist agents only when the task is complex enough.
