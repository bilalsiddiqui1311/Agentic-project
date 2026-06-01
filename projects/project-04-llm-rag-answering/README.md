# Project 04: RAG With Real LLM Answering

Goal: connect the grounded RAG prompt to a real LLM provider.

```text
question -> retrieve chunks -> build prompt -> call OpenAI -> answer with sources
```

What this project teaches:

- The LLM is a separate answer-generation step after retrieval.
- The same RAG prompt can be used by a local answerer or an API-backed model.
- Production apps should make the active answer mode visible.
- API keys must stay in environment variables, never source code.

Modes:

- `local_extractive`: default, no API key required.
- `openai_llm`: enabled when `RAG_ANSWER_MODE=openai` and `OPENAI_API_KEY` is set.

Environment:

```bash
RAG_ANSWER_MODE=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5.5
OPENAI_TIMEOUT_SECONDS=30
```

Swagger checks:

- `GET /rag/config`
- `POST /rag/query` with `include_prompt: true`

Main files:

- `app/rag/answerer.py`
- `app/rag/service.py`
- `app/rag/prompting.py`
- `.env.example`
- `docker-compose.yml`

Branch:

- `codex/project-04-llm-rag-answering`
