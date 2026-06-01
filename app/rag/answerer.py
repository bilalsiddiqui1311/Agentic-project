import re
from dataclasses import dataclass

from app.rag.embeddings import tokenize
from app.rag.models import SearchMatch


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    mode: str


class LocalGroundedAnswerer:
    """Deterministic answerer that mimics the final LLM step without needing keys."""

    mode = "local_extractive"

    def answer(self, query: str, matches: list[SearchMatch]) -> GeneratedAnswer:
        if not matches:
            return GeneratedAnswer(
                text=(
                    "I could not find relevant context in the indexed documents. "
                    "Try asking about agents, RAG, tool calling, memory, or Project 02."
                ),
                mode=self.mode,
            )

        direct_answer = self._answer_profile_question(query, matches)
        if direct_answer:
            return GeneratedAnswer(text=direct_answer, mode=self.mode)

        selected_sentences = self._select_sentences(query, matches)
        if not selected_sentences:
            sources = self._format_sources(matches)
            return GeneratedAnswer(
                text=(
                    "I found related context, but not enough detail to answer directly. "
                    f"Sources: {sources}."
                ),
                mode=self.mode,
            )

        answer = " ".join(selected_sentences)
        sources = self._format_sources(matches)
        return GeneratedAnswer(
            text=f"{answer} Sources: {sources}.",
            mode=self.mode,
        )

    def _answer_profile_question(self, query: str, matches: list[SearchMatch]) -> str | None:
        query_tokens = set(tokenize(query))
        asks_for_profession = bool(
            query_tokens.intersection({"profession", "role", "title", "job", "occupation"})
        )

        if not asks_for_profession:
            return None

        for match in matches:
            title = self._extract_markdown_bold_title(match.chunk.text)
            if title:
                return f"Bilal's profession is {title}. Sources: {match.chunk.source}."

        return None

    def _select_sentences(self, query: str, matches: list[SearchMatch]) -> list[str]:
        query_tokens = set(tokenize(query))
        scored_sentences: list[tuple[float, str]] = []

        for match in matches:
            for sentence in self._split_context(match.chunk.text):
                sentence_tokens = set(tokenize(sentence))
                if not sentence_tokens:
                    continue

                overlap = query_tokens.intersection(sentence_tokens)
                score = len(overlap) / max(len(query_tokens), 1)

                if score == 0:
                    continue

                scored_sentences.append((score + match.score, sentence))

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        return self._dedupe([sentence for _, sentence in scored_sentences[:3]])

    def _split_context(self, text: str) -> list[str]:
        cleaned = self._clean_markdown(text)
        pieces = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        return [
            piece.strip()
            for piece in pieces
            if len(piece.strip()) >= 30
        ]

    def _clean_markdown(self, text: str) -> str:
        cleaned = text.replace("#", " ")
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"[*`|_-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _extract_markdown_bold_title(self, text: str) -> str | None:
        match = re.search(r"\*\*([^*\n]+)\*\*", text)
        if not match:
            return None

        return match.group(1).strip()

    def _format_sources(self, matches: list[SearchMatch]) -> str:
        return ", ".join(dict.fromkeys(match.chunk.source for match in matches))

    def _dedupe(self, sentences: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for sentence in sentences:
            normalized = sentence.lower()
            if normalized in seen:
                continue

            seen.add(normalized)
            result.append(sentence)

        return result
