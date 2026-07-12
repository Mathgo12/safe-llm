"""Unit tests for the MockBackend."""

from __future__ import annotations

import unittest

from safe_llm.llm import MockBackend


class TestMockBackend(unittest.TestCase):
    def test_default_when_no_match(self) -> None:
        b = MockBackend(scripted={r"weather": "sunny"}, default="idk")
        reply = b.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(reply, "idk")

    def test_first_matching_pattern_wins(self) -> None:
        b = MockBackend(
            scripted={r"weather": "sunny", r"weather.*today": "hot"},
            default="idk",
        )
        reply = b.complete([{"role": "user", "content": "what's the weather today?"}])
        self.assertEqual(reply, "sunny")

    def test_case_insensitive(self) -> None:
        b = MockBackend(scripted={r"WEATHER": "sunny"}, default="idk")
        reply = b.complete([{"role": "user", "content": "any weather info?"}])
        self.assertEqual(reply, "sunny")

    def test_uses_last_user_message(self) -> None:
        b = MockBackend(scripted={r"weather": "sunny"}, default="idk")
        messages = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "sunny"},
            {"role": "user", "content": "tell me a joke"},
        ]
        self.assertEqual(b.complete(messages), "idk")

    def test_stream_yields_chunks(self) -> None:
        b = MockBackend(scripted={r"hi": "one two three four five six"}, default="")
        chunks = list(b.stream([{"role": "user", "content": "hi"}]))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), "one two three four five six")

    def test_stream_empty_response(self) -> None:
        b = MockBackend(scripted={}, default="")
        chunks = list(b.stream([{"role": "user", "content": "anything"}]))
        self.assertEqual(chunks, [])

    def test_model_attribute(self) -> None:
        self.assertEqual(MockBackend().model, "mock")


if __name__ == "__main__":
    unittest.main()
