"""SafeLLM — couples an LLMBackend to the SafetyGate.

Runs the gate's pre-gen check on the prompt, streams from the LLM backend
through the during-gen filter, and runs the post-gen check on the full
output before returning to the caller.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Iterator

from .gate import (
    DuringGenVerdict,
    PostGenVerdict,
    RequestTrace,
    SafetyGate,
)
from .llm.base import LLMBackend
from .rules.engine import Engine

REFUSAL_TEXT = "I cannot help with that. The request was blocked by the safety gate."


class SafeLLM:
    def __init__(self, backend: LLMBackend, gate: SafetyGate | None = None) -> None:
        self.backend = backend
        self.gate = SafetyGate() if gate is None else gate
        self.last_trace: RequestTrace | None = None

    def upload_constitution(self, rules_yaml: Path) -> None:
        self.gate.rules_engine = Engine(path=rules_yaml)

    def _block_at_pre_gen(self) -> bool:
        assert self.last_trace is not None
        pre = self.last_trace.pre_gen
        return pre.confidence >= self.gate.block_confidence and pre.category != "benign"

    def chat(self, messages: list[dict], **opts) -> tuple[str, RequestTrace]:
        start = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]
        latest_user_prompt = messages[-1]["content"]

        pre = self.gate.pre_gen(latest_user_prompt)

        if pre.confidence >= self.gate.block_confidence and pre.category != "benign":
            latency = (time.perf_counter() - start) * 1000.0
            self.last_trace = RequestTrace(
                request_id=request_id,
                prompt=latest_user_prompt,
                pre_gen=pre,
                during_gen=DuringGenVerdict(
                    terminated_early=False, matched_pattern=None, partial_chunks=0
                ),
                post_gen=None,
                final_action="block",
                final_output=REFUSAL_TEXT,
                latency_ms=round(latency, 3),
            )
            return REFUSAL_TEXT, self.last_trace

        stream_iter = self.backend.stream(messages, **opts)
        raw_output, during = self.gate.during_gen(stream_iter)

        post: PostGenVerdict | None = None
        if not during.terminated_early:
            post = self.gate.post_gen(raw_output)
        final_action = self.gate.aggregate(pre, during, post)
        final_output = self.gate.apply_action(final_action, raw_output, post)
        latency = (time.perf_counter() - start) * 1000.0
        self.last_trace = RequestTrace(
            request_id=request_id,
            prompt=latest_user_prompt,
            pre_gen=pre,
            during_gen=during,
            post_gen=post,
            final_action=final_action,
            final_output=final_output,
            latency_ms=round(latency, 3),
        )
        return final_output, self.last_trace

    def stream_chat(self, messages: list[dict], **opts) -> Iterator[str]:
        """Stream the model reply through the gate.

        Yields chunks to the caller as they pass a Detoxify sliding-window
        check mirroring `SafetyGate.during_gen`. On post-gen block/redact,
        the caller has already seen partial content — the generator emits
        a trailing safety notice and `self.last_trace` reflects the true
        final action.
        """
        start = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]
        latest_user_prompt = messages[-1]["content"]

        pre = self.gate.pre_gen(latest_user_prompt)
        if pre.confidence >= self.gate.block_confidence and pre.category != "benign":
            latency = (time.perf_counter() - start) * 1000.0
            self.last_trace = RequestTrace(
                request_id=request_id,
                prompt=latest_user_prompt,
                pre_gen=pre,
                during_gen=DuringGenVerdict(
                    terminated_early=False, matched_pattern=None, partial_chunks=0
                ),
                post_gen=None,
                final_action="block",
                final_output=REFUSAL_TEXT,
                latency_ms=round(latency, 3),
            )
            yield REFUSAL_TEXT
            return

        emitted: list[str] = []
        max_score = 0.0
        matched_pattern: str | None = None
        terminated = False
        chunks_since_check = 0
        for chunk in self.backend.stream(messages, **opts):
            emitted.append(chunk)
            yield chunk
            chunks_since_check += 1
            if chunks_since_check < self.gate.token_check_interval:
                continue
            chunks_since_check = 0
            window = "".join(emitted)[-self.gate.stream_window_chars :]
            score = self.gate._stream_score(window)
            if score > max_score:
                max_score = score
            if score >= self.gate.stream_toxicity_threshold:
                terminated = True
                matched_pattern = f"detoxify>={score:.2f}"
                break

        raw_output = "".join(emitted).strip()
        during = DuringGenVerdict(
            terminated_early=terminated,
            matched_pattern=matched_pattern,
            partial_chunks=len(emitted),
            toxicity_score=max_score,
        )

        post: PostGenVerdict | None = None
        if not terminated:
            post = self.gate.post_gen(raw_output)
        final_action = self.gate.aggregate(pre, during, post)

        if terminated:
            notice = f"\n\n[safety gate: stream terminated ({matched_pattern})]"
            yield notice
            final_output = raw_output + notice
        elif final_action == "warn":
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
            pre_gen=pre,
            during_gen=during,
            post_gen=post,
            final_action=final_action,
            final_output=final_output,
            latency_ms=round(latency, 3),
        )
