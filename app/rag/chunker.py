import hashlib
import re

from app.rag.models import Chunk, Document


def chunk_documents(
    documents: list[Document],
    max_words: int = 120,
    overlap_words: int = 20,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        words = _tokenize_for_chunks(document.text)
        if not words:
            continue

        step = max_words - overlap_words
        if step <= 0:
            raise ValueError("max_words must be larger than overlap_words.")

        for index, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + max_words]
            if not chunk_words:
                continue

            text = " ".join(chunk_words)
            chunks.append(
                Chunk(
                    id=_chunk_id(document.source, index),
                    source=document.source,
                    index=index,
                    text=text,
                )
            )

            if start + max_words >= len(words):
                break

    return chunks


def _tokenize_for_chunks(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text)
    return normalized.split()


def _chunk_id(source: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}:{index}".encode("utf-8")).hexdigest()[:10]
    return f"{source}:{index}:{digest}"
