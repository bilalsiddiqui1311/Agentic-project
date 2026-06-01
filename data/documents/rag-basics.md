# RAG Basics

RAG means Retrieval-Augmented Generation. A RAG system retrieves relevant
information before generating an answer.

The basic RAG pipeline is: load documents, split them into chunks, convert each
chunk into an embedding vector, store those vectors in a vector database or
vector index, retrieve the most relevant chunks for a user question, and answer
using the retrieved context.

RAG helps reduce hallucination because the model receives fresh or private
context at answer time. A good RAG response should show sources so the user can
inspect where the answer came from.
