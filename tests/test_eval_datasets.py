"""Unit tests for the eval-corpus loaders.

The hand-crafted loader runs offline. JailbreakBench and XSTest hit the
HuggingFace Hub on first invocation and are gated behind
SAFE_LLM_HEAVY_TESTS=1 to keep the default suite fast and offline.
"""

from __future__ import annotations

import unittest

from safe_llm.eval.datasets import (
    EvalItem,
    KNOWN_SOURCES,
    load_combined,
    load_handcrafted,
    load_jailbreakbench,
    load_xstest,
)

from ._heavy import skip_if_no_hf_datasets, skip_if_no_network


class TestKnownSources(unittest.TestCase):
    def test_known_sources_contains_three(self) -> None:
        self.assertEqual(set(KNOWN_SOURCES), {"hand-crafted", "jailbreakbench", "xstest"})


class TestHandcraftedLoader(unittest.TestCase):
    def test_loads_bundled_taxonomy(self) -> None:
        items = load_handcrafted()
        self.assertGreater(len(items), 0)
        self.assertTrue(all(isinstance(item, EvalItem) for item in items))

    def test_all_marked_unsafe(self) -> None:
        items = load_handcrafted()
        self.assertTrue(all(item.label == "unsafe" for item in items))

    def test_source_tagged(self) -> None:
        items = load_handcrafted()
        self.assertTrue(all(item.source == "hand-crafted" for item in items))

    def test_ids_are_unique(self) -> None:
        items = load_handcrafted()
        ids = [item.id for item in items]
        self.assertEqual(len(ids), len(set(ids)))


class TestLoadCombined(unittest.TestCase):
    def test_unknown_source_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_combined(["nonexistent"])

    def test_dedupes_repeated_sources(self) -> None:
        items = load_combined(["hand-crafted", "hand-crafted"])
        expected = load_handcrafted()
        self.assertEqual(len(items), len(expected))

    def test_single_source_matches_direct_load(self) -> None:
        items = load_combined(["hand-crafted"])
        expected = load_handcrafted()
        self.assertEqual(len(items), len(expected))


@skip_if_no_hf_datasets
class TestJailbreakBenchLoader(unittest.TestCase):
    def test_loads_and_labels_unsafe(self) -> None:
        items = load_jailbreakbench()
        self.assertGreater(len(items), 0)
        self.assertTrue(all(item.label == "unsafe" for item in items))
        self.assertTrue(all(item.source == "jailbreakbench" for item in items))
        self.assertTrue(all(item.prompt.strip() for item in items))


@skip_if_no_network
class TestXSTestLoader(unittest.TestCase):
    def test_loads_safe_only(self) -> None:
        items = load_xstest()
        self.assertGreater(len(items), 0)
        self.assertTrue(all(item.label == "safe" for item in items))
        self.assertTrue(all(item.source == "xstest" for item in items))
        self.assertTrue(all(item.prompt.strip() for item in items))


if __name__ == "__main__":
    unittest.main()
