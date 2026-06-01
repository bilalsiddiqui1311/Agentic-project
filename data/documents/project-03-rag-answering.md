# Project 03 RAG Answering

Project 03 separates retrieval from answer generation. Retrieval finds the most
relevant chunks. The answerer reads those chunks and produces a concise response
grounded in the retrieved evidence.

The service can also build a grounded prompt. That prompt instructs a future LLM
to answer only from the provided context, say when the context is insufficient,
keep the response concise, and include sources.

The current answer mode is local extractive answering. This keeps the project
usable in Docker without an API key while teaching the place where an LLM will
fit in a production RAG pipeline.
