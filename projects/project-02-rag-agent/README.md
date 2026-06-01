# Project 02: Local RAG Agent

Goal: turn document retrieval into an agent tool.

```text
documents -> chunks -> embeddings -> vector search -> sourced answer
```

What this project teaches:

- RAG retrieves context before answering.
- Documents need to be loaded and split into chunks.
- Queries and chunks can be compared through vectors.
- Retrieval quality affects answer quality.
- Sources help users inspect where the answer came from.

Current limitations:

- The embedding model is a local learning implementation.
- The answer builder is simple and not a full LLM reasoning step.
- Ranking is useful for learning but not production-grade.

Main files:

- `app/rag/service.py`
- `app/rag/vector_store.py`
- `app/rag/embeddings.py`
- `app/tools/rag_search.py`
- `data/documents/`
- `tests/test_rag.py`

Branch:

- `codex/project-02-rag-agent`
