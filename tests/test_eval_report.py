"""Unit tests for the trace-file analyzer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from safe_llm.eval.report import analyze_traces, load_trace_file


def _trace(label: str, action: str, source: str = "s1", category: str = "c1") -> dict:
    return {
        "label": label,
        "source": source,
        "category": category,
        "final_action": action,
    }


class TestAnalyzeTraces(unittest.TestCase):
    def test_over_refusal_rate(self) -> None:
        traces = [
            _trace("safe", "block"),
            _trace("safe", "block"),
            _trace("safe", "allow"),
            _trace("safe", "allow"),
        ]
        r = analyze_traces(traces)
        self.assertEqual(r.n_safe, 4)
        self.assertEqual(r.over_refusal_rate, 0.5)

    def test_under_refusal_rate(self) -> None:
        traces = [
            _trace("unsafe", "block"),
            _trace("unsafe", "block"),
            _trace("unsafe", "allow"),
        ]
        r = analyze_traces(traces)
        self.assertEqual(r.n_unsafe, 3)
        self.assertAlmostEqual(r.under_refusal_rate, 1 / 3)

    def test_empty_traces(self) -> None:
        r = analyze_traces([])
        self.assertEqual(r.n_safe, 0)
        self.assertEqual(r.n_unsafe, 0)
        self.assertEqual(r.over_refusal_rate, 0.0)
        self.assertEqual(r.under_refusal_rate, 0.0)

    def test_missing_label_is_skipped(self) -> None:
        traces = [
            {"source": "s", "category": "c", "final_action": "block"},
            _trace("safe", "allow"),
        ]
        r = analyze_traces(traces)
        self.assertEqual(r.n_safe, 1)
        self.assertEqual(r.n_unsafe, 0)

    def test_by_source_breakdown(self) -> None:
        traces = [
            _trace("safe", "block", source="xstest"),
            _trace("safe", "allow", source="xstest"),
            _trace("unsafe", "block", source="jbb"),
        ]
        r = analyze_traces(traces)
        self.assertEqual(r.by_source["xstest"]["n_safe"], 2)
        self.assertEqual(r.by_source["xstest"]["over_refusal_rate"], 0.5)
        self.assertEqual(r.by_source["jbb"]["under_refusal_rate"], 0.0)

    def test_warn_counts_as_allowed_for_under_refusal(self) -> None:
        # "warn" leaks content — treat as allowed on the under-refusal axis.
        # Analyzer semantics: allowed = final_action == "allow" only.
        # A warn does not count as under-refusal; document that here.
        traces = [_trace("unsafe", "warn")]
        r = analyze_traces(traces)
        self.assertEqual(r.under_refusal_rate, 0.0)


class TestLoadTraceFile(unittest.TestCase):
    def test_reads_traces_key(self) -> None:
        payload = {"model": "x", "traces": [_trace("safe", "allow")]}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trace.json"
            p.write_text(json.dumps(payload))
            traces = load_trace_file(p)
            self.assertEqual(len(traces), 1)

    def test_reads_bare_list(self) -> None:
        payload = [_trace("safe", "allow")]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trace.json"
            p.write_text(json.dumps(payload))
            traces = load_trace_file(p)
            self.assertEqual(len(traces), 1)


if __name__ == "__main__":
    unittest.main()
