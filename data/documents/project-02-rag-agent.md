# Project 02 RAG Agent

Project 02 turns document search into an agent tool. The agent still follows
the same loop from Project 01, but it can now retrieve context from files before
answering knowledge questions.

This starter implementation uses local hashing embeddings so it can run in
Docker without an external API key. The same structure can later be upgraded to
OpenAI embeddings, Chroma, pgvector, Pinecone, Weaviate, or another vector
database.

The learning goal is to understand the parts of RAG: documents, chunks,
embeddings, vector similarity, retrieval, context, answers, and sources.
