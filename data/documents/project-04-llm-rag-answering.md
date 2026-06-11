# Project 04 LLM RAG Answering

Project 04 connects the RAG pipeline to optional real LLM answerers. The app
still retrieves chunks locally, then builds a grounded prompt from the question
and retrieved evidence.

When Ollama mode is enabled, the grounded prompt is sent to a local Ollama model
such as Qwen. This avoids hosted API cost, but uses the machine's CPU, RAM, and
GPU resources. In Docker on macOS, the app reaches host Ollama through
`http://host.docker.internal:11434`.

When OpenAI mode is enabled, the grounded prompt is sent to the OpenAI Responses
API. In both LLM modes, the model is instructed to answer only from the indexed
context, say when the context is insufficient, keep the answer concise, and
include source filenames.

The app keeps local extractive answering as the default mode so the project can
run in Docker without secrets. The `/rag/config` endpoint shows whether the
active answer mode is local, Ollama-backed, or OpenAI-backed.
