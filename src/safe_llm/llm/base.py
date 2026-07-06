from __future__ import annotations

from typing import Iterator, Protocol

class LLMBackend(Protocol):
    model: str
    def stream(self, messages: list[dict], **opts) -> Iterator[str]:
        ...
    def complete(self, messages: list[dict], **opts) -> str:
        ...
