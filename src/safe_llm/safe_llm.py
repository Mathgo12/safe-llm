"""SafeLLM — couples an LLMBackend to the SafetyGate.

This is the user-facing wrapper. It runs the gate's pre-gen check on the
prompt, streams from the LLM backend through the during-gen filter, and runs
the post-gen check on the full output before returning to the caller.

See the package README for the design write-up.
"""

from __future__ import annotations
from typing import Iterator
from pathlib import Path

from .gate import SafetyGate, PreGenVerdict, DuringGenVerdict, PostGenVerdict, RequestTrace, write_traces

from .classifiers.router import Router
from .detector.detector import Detector
from .mock_stream import stream
from .rules.engine import Engine, Fixer

from .llm.base import LLMBackend

import re
import time
import uuid

OUTPUTS = Path.cwd() / "outputs"

SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


TERMINATION_PATTERNS = [
    re.compile(r"\bsure[, ]\s*here\s+is\b", re.IGNORECASE),
    re.compile(r"\bstep\s*1[:.\s]\s*take\b", re.IGNORECASE),
    re.compile(r"\bhere\s+is\s+the\s+procedure\b", re.IGNORECASE),
    re.compile(r"\bcombine\s+it\s+with\s+the\s+catalyst\b", re.IGNORECASE),
]

REFUSAL_TEXT = "I cannot help with that. The request was blocked by the safety gate."

class SafeLLM:
    def __init__(self, backend: LLMBackend, gate: SafetyGate | None = None, during_gen_lookback = 5):
        self.backend = backend
        self.gate = SafetyGate() if gate is None else gate
        self.lookback = during_gen_lookback
        self.last_trace = None

    def upload_constitution(self, rules_yaml: Path):
        self.gate.engine = Engine(path=rules_yaml)

    def chat(self, messages: list[dict], **opts) -> tuple[str, RequestTrace]:
        # pre-gen
        start = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]

        latest_user_prompt = messages[-1]["content"]

        pregen_verdict: PreGenVerdict = self.gate.pre_gen(latest_user_prompt)

        # Early exit
        if pregen_verdict.confidence >= self.gate.block_confidence and pregen_verdict.category != "benign":
            final_output = REFUSAL_TEXT
            latency = (time.perf_counter() - start) * 1000.0
            self.last_trace = RequestTrace(
                request_id=request_id,
                prompt=latest_user_prompt,
                pre_gen=pregen_verdict,
                during_gen=DuringGenVerdict(terminated_early=False, matched_pattern=None, partial_chunks=0),
                post_gen=None,
                final_action="block",
                final_output=final_output,
                latency_ms=round(latency, 3),
            )
            return final_output, self.last_trace

        # during-gen
        stream_iter = self.backend.stream(messages)
        raw_output, during = self.gate.during_gen(stream_iter, buffer_cap = self.lookback)

        # post-gen, if not terminated early in During Gen
        post: PostGenVerdict | None = None
        if not during.terminated_early:
            post = self.gate.post_gen(raw_output)
        final_action = self.gate.aggregate(pregen_verdict, during, post)
        final_output = self.gate.apply_action(final_action, raw_output, post)
        latency = (time.perf_counter() - start) * 1000.0
        self.last_trace = RequestTrace(
            request_id=request_id,
            prompt=latest_user_prompt,
            pre_gen=pregen_verdict,
            during_gen=during,
            post_gen=post,
            final_action=final_action,
            final_output=final_output,
            latency_ms=round(latency, 3),
        )
        return final_output, self.last_trace

    def stream_chat(self, messages: list[dict], **opts) -> Iterator[str]:
        """
        Same as chat, but yields chunks to the caller as they pass the
        during-gen filter; post-gen runs on flush. The full RequestTrace
        is stashed on `self.last_trace` once the generator is exhausted.

        Streaming caveat: post-gen actions (warn/redact/block) cannot
        un-yield text already on the wire. When they fire, the generator
        emits a trailing safety notice and the trace records the real
        final_action / final_output the caller can inspect.
        """
        start = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]
        latest_user_prompt = messages[-1]["content"]

        # pre-gen
        pregen_verdict = self.gate.pre_gen(latest_user_prompt)
        if pregen_verdict.confidence >= self.gate.block_confidence and pregen_verdict.category != "benign":
            latency = (time.perf_counter() - start) * 1000.0
            self.last_trace = RequestTrace(
                request_id=request_id,
                prompt=latest_user_prompt,
                pre_gen=pregen_verdict,
                during_gen=DuringGenVerdict(terminated_early=False, matched_pattern=None, partial_chunks=0),
                post_gen=None,
                final_action="block",
                final_output=REFUSAL_TEXT,
                latency_ms=round(latency, 3),
            )
            yield REFUSAL_TEXT
            return

        # during-gen: same sliding-window filter as gate.during_gen, but yield as we go
        buffer: list[str] = []
        raw: list[str] = []
        terminated = False
        matched_pattern: str | None = None
        emitted = 0

        for chunk in self.backend.stream(messages, **opts):
            raw.append(chunk)
            buffer.append(chunk)
            if len(buffer) > self.lookback:
                evicted = buffer.pop(0)
                emitted += 1
                yield evicted
            window = "".join(buffer)
            for pat in TERMINATION_PATTERNS:
                if pat.search(window):
                    terminated = True
                    matched_pattern = pat.pattern
                    break
            if terminated:
                break

        if not terminated:
            for c in buffer:
                emitted += 1
                yield c

        raw_output = "".join(raw).strip()
        during = DuringGenVerdict(
            terminated_early=terminated,
            matched_pattern=matched_pattern,
            partial_chunks=emitted,
        )

        # post-gen + aggregate
        post: PostGenVerdict | None = None
        if not terminated:
            post = self.gate.post_gen(raw_output)
        final_action = self.gate.aggregate(pregen_verdict, during, post)

        # we've already streamed; only an out-of-band notice is possible now
        if final_action == "warn":
            note = "\n\n[note: this response was flagged at low severity by the safety gate]"
            yield note
            final_output = raw_output + note
        elif final_action in {"block", "redact"}:
            notice = f"\n\n[safety gate: post-stream {final_action} — see trace for details]"
            yield notice
            final_output = self.gate.apply_action(final_action, raw_output, post)
        else:
            final_output = raw_output

        latency = (time.perf_counter() - start) * 1000.0
        self.last_trace = RequestTrace(
            request_id=request_id,
            prompt=latest_user_prompt,
            pre_gen=pregen_verdict,
            during_gen=during,
            post_gen=post,
            final_action=final_action,
            final_output=final_output,
            latency_ms=round(latency, 3),
        )
