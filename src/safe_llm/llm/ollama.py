"""Ollama HTTP backend adapter.

Ollama exposes a streaming chat endpoint at POST /api/chat. Each line of the
response body is a JSON object with a `message.content` token chunk and a
`done` flag on the last line.

Reference: https://github.com/ollama/ollama/blob/main/docs/api.md

"""

from __future__ import annotations
from collections.abc import Iterator
from typing import List
from ollama import Client

class OllamaBackend:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self._client = Client(host = host)

    def has_model(self) -> None:
        return any([self.model in model for model in self.list_models()])

    def list_models(self) -> List[str]:
        return [m.model for m in self._client.list().models]

    def stream(self, messages, **opts) -> Iterator[str]:
        chat_stream = self._client.chat(model = self.model, messages = messages, stream=True, **opts)
        for chunk in chat_stream:
            yield chunk['message']['content']

    def complete(self, messages, **opts):
        return "".join(self.stream(messages, **opts))
