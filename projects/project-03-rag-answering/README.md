# Project 03: Grounded RAG Answering

Goal: separate retrieval from answer generation.

```text
question -> retrieve chunks -> build prompt -> generate answer -> show sources
```

What this project teaches:

- Retrieval and answer generation are different responsibilities.
- A grounded prompt tells an LLM to answer only from retrieved context.
- The answer mode should be visible so you know how the answer was produced.
- Good RAG APIs expose sources and enough diagnostics to debug bad answers.

Current implementation:

- `LocalGroundedAnswerer` produces deterministic extractive answers.
- `build_grounded_prompt` shows the prompt shape for a future LLM provider.
- `include_prompt` lets Swagger show the prompt for learning/debugging.

Main files:

- `app/rag/answerer.py`
- `app/rag/prompting.py`
- `app/rag/service.py`
- `app/schemas.py`
- `tests/test_rag.py`

Branch:

- `codex/project-03-rag-answering`
