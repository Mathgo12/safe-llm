"""SafetyGate composes detector, token-filter, classifier, and rules engine
into a single request lifecycle with a deterministic aggregation table.

Three checkpoints, one decision tree:
  pre-gen   : detector on the prompt
  during-gen: streaming token filter (buffered regex sweep)
  post-gen  : classifier router + rules engine on the full output

Run: python -m safe_llm.gate
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .classifiers.classifiers import get_detoxify
from .classifiers.router import Router
from .detector.detector import Detector, MLInjectionScorer
from .mock_stream import stream
from .rules.engine import Engine, Fixer

OUTPUTS = Path.cwd() / "outputs"

SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}

DEFAULT_TOKEN_CHECK_INTERVAL = 20
DEFAULT_STREAM_TOXICITY_THRESHOLD = 0.7
DEFAULT_STREAM_WINDOW_CHARS = 400

REFUSAL_TEXT = "I cannot help with that. The request was blocked by the safety gate."


@dataclass
class PreGenVerdict:
    category: str
    confidence: float
    fired: list[str]


@dataclass
class DuringGenVerdict:
    terminated_early: bool
    matched_pattern: str | None
    partial_chunks: int
    toxicity_score: float = 0.0


@dataclass
class PostGenVerdict:
    classifier_action: str
    classifier_severity: str
    rules_max_severity: str | None
    rules_violations: list[str]


@dataclass
class RequestTrace:
    request_id: str
    prompt: str
    pre_gen: PreGenVerdict
    during_gen: DuringGenVerdict
    post_gen: PostGenVerdict | None
    final_action: str
    final_output: str
    latency_ms: float


@dataclass
class SafetyGate:
    detector: Detector = field(default_factory=Detector)
    classifier_router: Router = field(default_factory=Router)
    rules_engine: Engine = field(default_factory=Engine)
    rules_fixer: Fixer | None = None
    block_confidence: float = 0.85
    note_confidence: float = 0.5
    token_check_interval: int = DEFAULT_TOKEN_CHECK_INTERVAL
    stream_toxicity_threshold: float = DEFAULT_STREAM_TOXICITY_THRESHOLD
    stream_window_chars: int = DEFAULT_STREAM_WINDOW_CHARS
    use_ml_detector: bool = False
    _detoxify: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rules_fixer is None:
            self.rules_fixer = Fixer(self.rules_engine.rules())
        if self.use_ml_detector and self.detector.ml_scorer is None:
            try:
                self.detector.ml_scorer = MLInjectionScorer()
            except ImportError:
                # transformers not installed — silently keep rules-only detector.
                pass

    def _stream_score(self, window: str) -> float:
        if self._detoxify is None:
            self._detoxify = get_detoxify()
        raw = self._detoxify.predict(window)  # type: ignore[attr-defined]
        return max(float(v) for v in raw.values())

    def pre_gen(self, prompt: str) -> PreGenVerdict:
        v = self.detector.analyze(prompt)
        return PreGenVerdict(category=v.category, confidence=v.confidence, fired=list(v.fired))

    def during_gen(self, chunks: Iterable[str]) -> tuple[str, DuringGenVerdict]:
        """Stream-side safety check.

        Buffers every emitted chunk in `emitted`. Every
        `self.token_check_interval` chunks, score a sliding window of the
        most recent `self.stream_window_chars` characters with Detoxify.
        Terminate as soon as any Detoxify label exceeds
        `self.stream_toxicity_threshold`.

        Returns:
            (emitted_text, DuringGenVerdict) — emitted_text is the partial or
            full stream that was safe to hand to the caller.
        """
        emitted: list[str] = []
        terminated = False
        matched_pattern: str | None = None
        max_score = 0.0
        chunks_since_check = 0
        for chunk in chunks:
            emitted.append(chunk)
            chunks_since_check += 1
            if chunks_since_check < self.token_check_interval:
                continue
            chunks_since_check = 0
            window = "".join(emitted)[-self.stream_window_chars :]
            score = self._stream_score(window)
            if score > max_score:
                max_score = score
            if score >= self.stream_toxicity_threshold:
                terminated = True
                matched_pattern = f"detoxify>={score:.2f}"
                break
        if not terminated and emitted:
            window = "".join(emitted)[-self.stream_window_chars :]
            score = self._stream_score(window)
            if score > max_score:
                max_score = score
            if score >= self.stream_toxicity_threshold:
                terminated = True
                matched_pattern = f"detoxify>={score:.2f}"
        verdict = DuringGenVerdict(
            terminated_early=terminated,
            matched_pattern=matched_pattern,
            partial_chunks=len(emitted),
            toxicity_score=max_score,
        )
        return "".join(emitted).strip(), verdict

    def post_gen(self, output: str) -> PostGenVerdict:
        action = self.classifier_router.run(output)
        engine_report = self.rules_engine.evaluate(output)
        violations = engine_report.violations()
        return PostGenVerdict(
            classifier_action=action.verb,
            classifier_severity=action.severity,
            rules_max_severity=engine_report.max_severity(),
            rules_violations=[v.rule_name for v in violations],
        )

    def aggregate(self, pre: PreGenVerdict, during: DuringGenVerdict, post: PostGenVerdict | None) -> str:
        """"
            Parameters: pre, during, and post gen verdicts
            Returns: str
                One of "allow", "block", "redact", and "warn"

        """
        signals: list[tuple[str, int]] = []
        if pre.category != "benign":
            if pre.confidence >= self.block_confidence:
                signals.append(("pre", SEVERITY_RANK["high"]))
            elif pre.confidence >= self.note_confidence:
                signals.append(("pre", SEVERITY_RANK["medium"]))
            else:
                signals.append(("pre", SEVERITY_RANK["low"]))
        if during.terminated_early:
            signals.append(("during", SEVERITY_RANK["medium"]))
        if post is not None:
            signals.append(("post.classifier", SEVERITY_RANK.get(post.classifier_severity, SEVERITY_RANK["none"])))
            if post.rules_max_severity:
                signals.append(("post.rules", SEVERITY_RANK.get(post.rules_max_severity, SEVERITY_RANK["none"])))
        if not signals:
            return "allow"
        max_rank = max(rank for _, rank in signals)
        if max_rank >= SEVERITY_RANK["high"]:
            return "block"
        if max_rank >= SEVERITY_RANK["medium"]:
            return "redact"
        if max_rank >= SEVERITY_RANK["low"]:
            return "warn"
        return "allow"

    def apply_action(self, action: str, raw_output: str, post: PostGenVerdict | None) -> str:
        """
        Parameters:
            action: str - outcome of _aggregate. One of "allow", "block", "redact", and "warn"
            raw_output: str - Raw output of LLM
            post: PostGenVerdict - post generation verdict as returned by _post_gen

        Returns: str
            Modified LLM output based on action and (possibly) post gen verdict.
        """
        if action == "block":
            return REFUSAL_TEXT
        if action == "redact":
            classifier_action = self.classifier_router.run(raw_output)
            redacted = classifier_action.output
            if post is not None and post.rules_violations:
                violations = self.rules_engine.evaluate(redacted).violations()
                if self.rules_fixer is not None:
                    redacted = self.rules_fixer.apply(redacted, violations)
            return redacted if redacted.strip() else REFUSAL_TEXT
        if action == "warn":
            return raw_output + "\n\n[note: this response was flagged at low severity by the safety gate]"
        return raw_output

    def handle(self, prompt: str) -> RequestTrace:
        start = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]

        # Pre-Gen
        pre = self.pre_gen(prompt)

        # Early exit (i.e. when input prompt is too severe)
        if pre.confidence >= self.block_confidence and pre.category != "benign":
            final_output = REFUSAL_TEXT
            latency = (time.perf_counter() - start) * 1000.0
            return RequestTrace(
                request_id=request_id,
                prompt=prompt,
                pre_gen=pre,
                during_gen=DuringGenVerdict(terminated_early=False, matched_pattern=None, partial_chunks=0, toxicity_score=0.0),
                post_gen=None,
                final_action="block",
                final_output=final_output,
                latency_ms=round(latency, 3),
            )

        # During Gen
        raw_output, during = self.during_gen(stream(prompt))

        # Post Gen, if not terminated early in During Gen
        post: PostGenVerdict | None = None
        if not during.terminated_early:
            post = self.post_gen(raw_output)
        final_action = self.aggregate(pre, during, post)
        final_output = self.apply_action(final_action, raw_output, post)
        latency = (time.perf_counter() - start) * 1000.0
        return RequestTrace(
            request_id=request_id,
            prompt=prompt,
            pre_gen=pre,
            during_gen=during,
            post_gen=post,
            final_action=final_action,
            final_output=final_output,
            latency_ms=round(latency, 3),
        )


def trace_to_dict(t: RequestTrace) -> dict[str, object]:
    return {
        "request_id": t.request_id,
        "prompt": t.prompt,
        "pre_gen": asdict(t.pre_gen),
        "during_gen": asdict(t.during_gen),
        "post_gen": asdict(t.post_gen) if t.post_gen else None,
        "final_action": t.final_action,
        "final_output": t.final_output,
        "latency_ms": t.latency_ms,
    }


BENIGN_PROMPTS = [
    "Help me draft a polite reminder email to a colleague about an overdue review.",
    "Recommend a beginner running plan for someone who hasn't exercised in a year.",
    "Compose a short haiku about a quiet afternoon in early autumn.",
    "Explain why decimal floating point is hard for monetary math.",
    "Suggest three names for a small worker-owned coffee cooperative.",
    "Walk me through how a debouncer differs from a throttler in front-end code.",
    "Summarize the trade-offs between message-queue and streaming architectures.",
    "Plan a one day itinerary for visiting Porto with two retired parents.",
    "Recommend a vegetarian dinner using lentils, spinach, and tomatoes.",
    "Outline a one-paragraph bio for a senior backend engineer.",
]


def write_traces(payload: dict[str, object], outputs_dir: Path | None = None) -> Path:
    out_dir = outputs_dir or OUTPUTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gate_trace.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def demo() -> int:
    from .taxonomy.taxonomy import Taxonomy
    fixtures = [asdict(f) for f in Taxonomy.from_default().all()]
    gate = SafetyGate()
    traces = []
    action_counts: Counter = Counter()
    per_category_outcome: dict[str, Counter] = defaultdict(Counter)
    terminations = 0
    total_latency = 0.0

    print("End-to-end safety gate demo")
    print(f"  loaded {len(fixtures)} attack fixtures + {len(BENIGN_PROMPTS)} benign prompts")
    print()

    for fix in fixtures:
        prompt = str(fix["prompt"])
        category = str(fix["category"])
        trace = gate.handle(prompt)
        action_counts[trace.final_action] += 1
        per_category_outcome[category][trace.final_action] += 1
        if trace.during_gen.terminated_early:
            terminations += 1
        total_latency += trace.latency_ms
        traces.append({"corpus": "attack", "fixture_id": fix["id"], "category": category, **trace_to_dict(trace)})

    benign_block_count = 0
    for prompt in BENIGN_PROMPTS:
        trace = gate.handle(prompt)
        action_counts[trace.final_action] += 1
        per_category_outcome["benign"][trace.final_action] += 1
        if trace.during_gen.terminated_early:
            terminations += 1
        if trace.final_action == "block":
            benign_block_count += 1
        total_latency += trace.latency_ms
        traces.append({"corpus": "benign", "fixture_id": None, "category": "benign", **trace_to_dict(trace)})

    total_requests = len(fixtures) + len(BENIGN_PROMPTS)
    print("  final action counts:")
    for action in ("block", "redact", "warn", "allow"):
        print(f"    {action:6} {action_counts.get(action, 0)}")
    print()
    print(f"  during-gen terminations: {terminations}")
    print(f"  benign blocks: {benign_block_count} / {len(BENIGN_PROMPTS)}")
    print(f"  avg latency: {total_latency / total_requests:.2f} ms / request")
    print()
    print("  per-category outcome:")
    for cat, counts in per_category_outcome.items():
        parts = ", ".join(f"{a}={counts[a]}" for a in ("block", "redact", "warn", "allow") if counts[a])
        print(f"    {cat:22} {parts}")

    payload = {
        "summary": {
            "total_requests": total_requests,
            "action_counts": dict(action_counts),
            "terminations": terminations,
            "benign_blocks": benign_block_count,
            "avg_latency_ms": round(total_latency / total_requests, 3),
            "per_category_outcome": {k: dict(v) for k, v in per_category_outcome.items()},
        },
        "traces": traces,
    }
    out = write_traces(payload)
    print(f"\n  artifact written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(demo())
