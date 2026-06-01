from pathlib import Path

from app.rag.models import Document

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(root: Path) -> list[Document]:
    if not root.exists():
        return []

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            Document(
                source=str(path.relative_to(root)),
                text=text,
            )
        )

    return documents
