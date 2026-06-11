import json
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.rag.embeddings import tokenize
from app.rag.models import SearchMatch


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    mode: str
    model: str | None = None


class Answerer(Protocol):
    mode: str

    def answer(
        self,
        query: str,
        matches: list[SearchMatch],
        prompt: str,
    ) -> GeneratedAnswer:
        """Generate a grounded answer from retrieved matches."""


class LocalGroundedAnswerer:
    """Deterministic answerer that mimics the final LLM step without needing keys."""

    mode = "local_extractive"

    def answer(
        self,
        query: str,
        matches: list[SearchMatch],
        prompt: str = "",
    ) -> GeneratedAnswer:
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

        answer_matches = self._select_answer_matches(matches)
        selected_sentences = self._select_sentences(query, answer_matches)
        if not selected_sentences:
            sources = self._format_sources(answer_matches)
            return GeneratedAnswer(
                text=(
                    "I found related context, but not enough detail to answer directly. "
                    f"Sources: {sources}."
                ),
                mode=self.mode,
            )

        answer = " ".join(selected_sentences)
        sources = self._format_sources(answer_matches)
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
        return self._dedupe([sentence for _, sentence in scored_sentences[:8]])

    def _select_answer_matches(self, matches: list[SearchMatch]) -> list[SearchMatch]:
        if not matches:
            return []

        best_score = matches[0].score
        strong_matches = [
            match
            for match in matches
            if match.score >= best_score * 0.7
        ]
        return strong_matches or [matches[0]]

    def _split_context(self, text: str) -> list[str]:
        cleaned = self._clean_markdown(text)
        pieces = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        return [
            piece.strip()
            for piece in pieces
            if self._is_complete_sentence(piece.strip())
        ]

    def _is_complete_sentence(self, sentence: str) -> bool:
        return (
            len(sentence) >= 30
            and sentence.endswith((".", "!", "?"))
            and sentence[0].isupper()
        )

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


class OpenAIAnswerer:
    mode = "openai_llm"

    def __init__(
        self,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(timeout=timeout_seconds)

    def answer(
        self,
        query: str,
        matches: list[SearchMatch],
        prompt: str,
    ) -> GeneratedAnswer:
        if not matches:
            return GeneratedAnswer(
                text=(
                    "I could not find relevant context in the indexed documents. "
                    "Try asking about agents, RAG, tool calling, memory, or Project 02."
                ),
                mode=self.mode,
                model=self.model,
            )

        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are a precise RAG answerer. Use only the provided context. "
                "If the context is insufficient, say you do not know from the indexed documents. "
                "Cite source filenames in the answer."
            ),
            input=prompt,
            store=False,
        )

        return GeneratedAnswer(
            text=response.output_text.strip(),
            mode=self.mode,
            model=self.model,
        )


class OllamaAnswerer:
    mode = "ollama_llm"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 180.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def answer(
        self,
        query: str,
        matches: list[SearchMatch],
        prompt: str,
    ) -> GeneratedAnswer:
        if not matches:
            return GeneratedAnswer(
                text=(
                    "I could not find relevant context in the indexed documents. "
                    "Try asking about agents, RAG, tool calling, memory, or Project 02."
                ),
                mode=self.mode,
                model=self.model,
            )

        try:
            response = self._chat(prompt)
        except RuntimeError as error:
            return GeneratedAnswer(
                text=(
                    "Ollama is configured, but the local model could not answer yet. "
                    f"{error}"
                ),
                mode=self.mode,
                model=self.model,
            )

        return GeneratedAnswer(
            text=response,
            mode=self.mode,
            model=self.model,
        )

    def _chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise RAG answerer. Use only the provided context. "
                        "If the context is insufficient, say you do not know from "
                        "the indexed documents. "
                        "Cite source filenames in the answer."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {error.code}: {details}"
            ) from error
        except TimeoutError as error:
            raise RuntimeError(self._timeout_message()) from error
        except URLError as error:
            if "timed out" in str(error).lower():
                raise RuntimeError(self._timeout_message()) from error

            raise RuntimeError(
                f"Check that Ollama is running at {self.base_url} and that model "
                f"{self.model!r} is pulled."
            ) from error
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Ollama returned a response that was not valid JSON."
            ) from error

        message = body.get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty answer.")

        return content.strip()

    def _timeout_message(self) -> str:
        return (
            f"The local Ollama request timed out after {self.timeout_seconds:g} seconds. "
            "The model may still be loading or running slowly on CPU. Try again, "
            "increase OLLAMA_TIMEOUT_SECONDS, or use a smaller model."
        )
