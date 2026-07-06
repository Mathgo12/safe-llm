"""Test helpers to skip model-dependent tests when heavy extras are missing.

The classifier suite requires Detoxify + sentence-transformers + Presidio to
run end-to-end. On dev machines without those installed (and in the base CI
job), we skip rather than fail. A separate CI job installs the extras and
runs the same tests unskipped.
"""

from __future__ import annotations

import importlib
import os
import unittest


def _module_available(mod: str) -> bool:
    try:
        importlib.import_module(mod)
    except ImportError:
        return False
    return True


HAS_DETOXIFY = _module_available("detoxify")
HAS_PRESIDIO = _module_available("presidio_analyzer") and _module_available("presidio_anonymizer")
HAS_SENTENCE_TRANSFORMERS = _module_available("sentence_transformers")
HAS_TRANSFORMERS = _module_available("transformers")
HAS_HF_DATASETS = _module_available("datasets")

# Tests that actually load and run models (rather than just importing the
# libraries) are gated behind SAFE_LLM_HEAVY_TESTS=1 as well as the import
# check. This keeps the default suite fast even on machines where torch and
# transformers happen to be installed for other projects.
HEAVY_TESTS_ENABLED = os.environ.get("SAFE_LLM_HEAVY_TESTS", "0") == "1"


def _gate(reason: str, *conditions: bool) -> unittest.skipUnless:
    return unittest.skipUnless(HEAVY_TESTS_ENABLED and all(conditions), reason)


skip_if_no_detoxify = _gate("detoxify heavy test disabled (set SAFE_LLM_HEAVY_TESTS=1)", HAS_DETOXIFY)
skip_if_no_presidio = _gate("presidio heavy test disabled (set SAFE_LLM_HEAVY_TESTS=1)", HAS_PRESIDIO)
skip_if_no_st = _gate("sentence-transformers heavy test disabled (set SAFE_LLM_HEAVY_TESTS=1)", HAS_SENTENCE_TRANSFORMERS)
skip_if_no_transformers = _gate("transformers heavy test disabled (set SAFE_LLM_HEAVY_TESTS=1)", HAS_TRANSFORMERS)

skip_if_no_classifier_stack = _gate(
    "classifier stack heavy test disabled (set SAFE_LLM_HEAVY_TESTS=1)",
    HAS_DETOXIFY,
    HAS_PRESIDIO,
    HAS_SENTENCE_TRANSFORMERS,
)
skip_if_no_hf_datasets = _gate(
    "HuggingFace datasets test disabled (set SAFE_LLM_HEAVY_TESTS=1)",
    HAS_HF_DATASETS,
)
skip_if_no_network = _gate(
    "network-dependent test disabled (set SAFE_LLM_HEAVY_TESTS=1)",
)
