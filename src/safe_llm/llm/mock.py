"""Deterministic mock backend for testing.

A canned LLM whose responses are scripted by pattern. Useful in tests so the
harness behaviour is reproducible, and in CI so the whole gate can run without
any local model installed.

Usage:
    backend = MockBackend(
        scripted={
            r"weather": "Sunny and 68F.",
            r"ignore.*instructions": "Sure, here is the procedure. Step 1: take the container...",
        },
        default="I don't know.",
    )
    safe_llm = SafeLLM(backend)
    reply, trace = safe_llm.chat([{"role": "user", "content": "what's the weather?"}])
"""

from __future__ import annotations

import re
from typing import Iterator


class MockBackend:
    """A backend that matches the last user message against pattern regexes.

    The first pattern to match (insertion order in `scripted`) wins. If no
    pattern matches, `default` is returned. Both `stream` and `complete`
    delegate to the same lookup so behaviour is identical across code paths.

    Args:
        scripted: dict of {pattern: response}. Patterns are compiled
            case-insensitively as regular expressions.
        default: response returned when no pattern matches.
        chunk_words: number of whitespace-split tokens per streamed chunk.
    """

    def __init__(
        self,
        scripted: dict[str, str] | None = None,
        default: str = "ok",
        chunk_words: int = 4,
    ) -> None:
        self.scripted = [(re.compile(p, re.IGNORECASE), r) for p, r in (scripted or {}).items()]
        self.default = default
        self.chunk_words = max(1, chunk_words)
        self.model = "mock"

    def _pick(self, messages: list[dict]) -> str:
        user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        for pat, reply in self.scripted:
            if pat.search(user):
                return reply
        return self.default

    def stream(self, messages: list[dict], **opts) -> Iterator[str]:
        text = self._pick(messages)
        tokens = text.split()
        if not tokens:
            return
        for i in range(0, len(tokens), self.chunk_words):
            chunk = " ".join(tokens[i : i + self.chunk_words])
            trailing = " " if i + self.chunk_words < len(tokens) else ""
            yield chunk + trailing

    def complete(self, messages: list[dict], **opts) -> str:
        return self._pick(messages)
