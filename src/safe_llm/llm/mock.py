"""Deterministic mock backend for testing.

A canned LLM whose responses are scripted by pattern, not by neural model.
Use this in tests so the harness behaviour is reproducible and so the gate
can be exercised without any local model installed.

TODO:
- Implement `MockBackend` that takes a dict of {pattern_regex: response} and
  matches the last user message against the patterns, falling back to a
  default reply.
- `stream(messages)` should split the chosen response into ~4-token chunks
  (mirror `safe_llm.mock_stream.stream`) so the during-gen filter has
  something realistic to buffer.
- `complete(messages)` should return the full response string.
"""

from __future__ import annotations

# TODO:
# import re
# from typing import Iterator
# from .base import LLMBackend
#
# class MockBackend:
#     def __init__(self, scripted: dict[str, str] | None = None, default: str = "ok"):
#         self.scripted = {re.compile(p, re.IGNORECASE): r for p, r in (scripted or {}).items()}
#         self.default = default
#         self.model = "mock"
#
#     def _pick(self, messages):
#         user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
#         for pat, reply in self.scripted.items():
#             if pat.search(user):
#                 return reply
#         return self.default
#
#     def stream(self, messages, chunk_tokens: int = 4, **opts) -> Iterator[str]:
#         text = self._pick(messages)
#         tokens = text.split()
#         for i in range(0, len(tokens), chunk_tokens):
#             yield " ".join(tokens[i : i + chunk_tokens]) + " "
#
#     def complete(self, messages, **opts) -> str:
#         return self._pick(messages)
