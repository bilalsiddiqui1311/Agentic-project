# Project 04 LLM RAG Answering

Project 04 connects the RAG pipeline to an optional real LLM answerer. The app
still retrieves chunks locally, then builds a grounded prompt from the question
and retrieved evidence.

When OpenAI mode is enabled, the grounded prompt is sent to the OpenAI Responses
API. The model is instructed to answer only from the indexed context, say when
the context is insufficient, keep the answer concise, and include source
filenames.

The app keeps local extractive answering as the default mode so the project can
run in Docker without secrets. The `/rag/config` endpoint shows whether the
active answer mode is local or OpenAI-backed.
