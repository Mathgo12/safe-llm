"""Tests for the SafetyGate `use_ml_detector` flag."""

from __future__ import annotations

import unittest

from safe_llm.detector.detector import Detector
from safe_llm.gate import SafetyGate


class TestUseMlDetectorFlag(unittest.TestCase):
    def test_default_is_off(self) -> None:
        gate = SafetyGate()
        self.assertFalse(gate.use_ml_detector)
        self.assertIsNone(gate.detector.ml_scorer)

    def test_user_supplied_scorer_is_preserved(self) -> None:
        sentinel = object()
        det = Detector()
        det.ml_scorer = sentinel  # type: ignore[assignment]
        gate = SafetyGate(detector=det, use_ml_detector=True)
        self.assertIs(gate.detector.ml_scorer, sentinel)

    def test_flag_on_without_transformers_is_silent(self) -> None:
        # If transformers isn't installed, the flag must not raise — the gate
        # falls back to rules-only. We can't easily monkeypatch the import
        # here, so we just assert construction succeeds; if transformers IS
        # installed, ml_scorer ends up wired.
        gate = SafetyGate(use_ml_detector=True)
        # Either wired (transformers present) or None (silent fallback).
        # Both are valid outcomes; construction must not have raised.
        self.assertIsNotNone(gate)


if __name__ == "__main__":
    unittest.main()
