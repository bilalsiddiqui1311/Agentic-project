# Production RAG Limitations

Production RAG systems often fail because retrieval quality is weaker than the
final answer needs. Common limitations include missing relevant context,
retrieving noisy chunks, splitting documents at poor boundaries, ranking chunks
by shallow similarity, failing on exact keywords or acronyms, and generating
answers from incomplete evidence.

Chunking addresses context quality. Good chunks should preserve meaning, keep
related facts together, and avoid splitting important sections across unrelated
fragments. Chunk size and overlap affect recall and precision. Small chunks can
improve precision but may lose context. Large chunks preserve context but can
include irrelevant material.

Hybrid search combines semantic vector search with keyword or lexical search.
It helps when questions include exact terms, names, acronyms, product codes, or
numbers. Vector search is useful for meaning; lexical search is useful for exact
matches. A hybrid strategy usually improves recall compared with either method
alone.

Reranking improves precision after retrieval. The first retriever can collect a
larger set of candidate chunks, then a reranker scores which chunks best answer
the question. This helps remove noisy or weakly related chunks before the LLM
generates the final answer.

A strong production RAG pipeline usually retrieves enough candidates, reranks
them, sends only the best evidence to the LLM, instructs the LLM to answer only
from context, and returns sources so the user can inspect the evidence.
