from app.rag.models import SearchMatch


def build_grounded_prompt(query: str, matches: list[SearchMatch]) -> str:
    context_blocks = "\n\n".join(
        f"Source: {match.chunk.source}\nChunk: {match.chunk.text}"
        for match in matches
    )

    return (
        "You are a grounded RAG answerer.\n"
        "Answer the user question using only the provided context.\n"
        "If the context is insufficient, say you do not know from the indexed documents.\n"
        "Keep the answer concise and include sources.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_blocks}"
    )
