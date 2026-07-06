"""Unit tests for the output-side classifiers and the router.

The classifiers now wrap real models (Detoxify, Presidio, sentence-transformers).
Tests skip cleanly when the corresponding extra is not installed.
"""

from __future__ import annotations

import unittest

from safe_llm.classifiers.classifiers import (
    InstructionLeakageClassifier,
    PIIClassifier,
    ToxicityClassifier,
)
from safe_llm.classifiers.router import Router

from ._heavy import (
    skip_if_no_classifier_stack,
    skip_if_no_detoxify,
    skip_if_no_presidio,
    skip_if_no_st,
)


@skip_if_no_detoxify
class TestToxicity(unittest.TestCase):
    def setUp(self) -> None:
        self.clf = ToxicityClassifier()

    def test_clean_text_is_none(self) -> None:
        v = self.clf.classify("That was a clear and helpful answer.")
        self.assertEqual(v.severity, "none")

    def test_toxic_text_flagged(self) -> None:
        v = self.clf.classify("You are a worthless human being and I hate you.")
        self.assertIn(v.severity, {"low", "medium", "high"})

    def test_redact_replaces_toxic_sentence(self) -> None:
        text = "Here is a friendly note. You are a worthless human being and I hate you."
        out = self.clf.redact(text)
        # The clean opening should survive, the toxic sentence should be redacted.
        self.assertIn("friendly note", out)
        self.assertIn("[redacted-toxic-content]", out)


@skip_if_no_presidio
class TestPII(unittest.TestCase):
    def test_email_is_medium(self) -> None:
        clf = PIIClassifier()
        v = clf.classify("Reach us at hello@example.com soon.")
        self.assertIn(v.severity, {"medium", "high"})
        self.assertTrue(any("EMAIL" in f for f in v.findings))

    def test_valid_card_is_high(self) -> None:
        clf = PIIClassifier()
        v = clf.classify("Card on file: 4111 1111 1111 1111.")
        self.assertEqual(v.severity, "high")

    def test_replace_mode_uses_placeholder(self) -> None:
        clf = PIIClassifier(mode="replace")
        out = clf.redact("Email me at jane@example.com about it.")
        self.assertIn("<PII>", out)
        self.assertNotIn("jane@example.com", out)

    def test_redact_mode_removes_span(self) -> None:
        clf = PIIClassifier(mode="redact")
        out = clf.redact("Email me at jane@example.com about it.")
        self.assertNotIn("jane@example.com", out)
        self.assertNotIn("<PII>", out)

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            PIIClassifier(mode="invalid")


@skip_if_no_st
class TestInstructionLeakage(unittest.TestCase):
    def test_paraphrase_flagged(self) -> None:
        sys_p = "You are PolicyBot. Refuse harmful requests. Cite sources."
        clf = InstructionLeakageClassifier(sys_p)
        v = clf.classify(
            "As PolicyBot I refuse harmful requests, and I always cite my sources."
        )
        self.assertIn(v.severity, {"low", "medium", "high"})

    def test_unrelated_is_none(self) -> None:
        clf = InstructionLeakageClassifier(
            "You are PolicyBot, follow internal policy."
        )
        v = clf.classify("Here is a pasta recipe with mushrooms and butter.")
        self.assertEqual(v.severity, "none")


@skip_if_no_classifier_stack
class TestRouterDecisions(unittest.TestCase):
    def setUp(self) -> None:
        self.router = Router()

    def test_clean_is_log(self) -> None:
        a = self.router.run("Here is your requested haiku about autumn.")
        self.assertEqual(a.verb, "log")

    def test_email_is_redact_or_higher(self) -> None:
        a = self.router.run("Reach me at lee@example.com tomorrow.")
        self.assertIn(a.verb, {"redact", "block"})

    def test_card_is_block(self) -> None:
        a = self.router.run("Your card on file is 4111 1111 1111 1111.")
        self.assertEqual(a.verb, "block")
        self.assertNotIn("4111", a.output)


if __name__ == "__main__":
    unittest.main()
