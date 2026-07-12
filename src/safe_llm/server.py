"""OpenAI-compatible HTTP proxy in front of SafeLLM.

Any client that speaks the OpenAI Chat Completions API can point its
`base_url` at this server and get harnessed responses transparently. The
safety trace is included as an extension field (`x_safety_trace`) on both
streaming and non-streaming responses.

Endpoints:
    GET  /health
    POST /v1/chat/completions

Config via environment:
    SAFE_LLM_BACKEND        ollama | mock       (default: ollama)
    SAFE_LLM_MODEL          model name          (default: llama3.2:3b)
    SAFE_LLM_OLLAMA_HOST    ollama HTTP host    (default: http://localhost:11434)

The ML injection detector is off in server mode; use `safe-llm eval
--use-ml-detector` if you need to benchmark it.

To run:
    uvicorn safe_llm.server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from typing import Any, Iterator

from .gate import trace_to_dict
from .llm import MockBackend, OllamaBackend
from .llm.base import LLMBackend
from .safe_llm import SafeLLM

_MOCK_SCRIPT = {
    r"weather": "It is sunny and 68F.",
    r"hello|hi\b": "Hello. How can I help?",
}


def _build_backend() -> LLMBackend:
    backend_name = os.environ.get("SAFE_LLM_BACKEND", "ollama").lower()
    model = os.environ.get("SAFE_LLM_MODEL", "llama3.2:3b")
    if backend_name == "mock":
        return MockBackend(scripted=_MOCK_SCRIPT, default="I am a mock backend.")
    if backend_name == "ollama":
        host = os.environ.get("SAFE_LLM_OLLAMA_HOST", "http://localhost:11434")
        return OllamaBackend(model=model, host=host)
    raise ValueError(f"unknown SAFE_LLM_BACKEND={backend_name!r}; use 'ollama' or 'mock'")


def _make_app() -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the server. "
            'Install with: pip install "safe-llm[server]"'
        ) from exc

    app = FastAPI(title="safe-llm", version="0.1.0")
    backend = _build_backend()
    safe = SafeLLM(backend)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": type(backend).__name__, "model": backend.model}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages: required non-empty list")

        model_hint = str(body.get("model") or backend.model)
        stream = bool(body.get("stream", False))
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        if stream:
            def sse() -> Iterator[bytes]:
                yield _sse_delta(completion_id, created, model_hint, role="assistant")
                for chunk in safe.stream_chat(messages):
                    yield _sse_delta(completion_id, created, model_hint, content=chunk)
                trace_dict = trace_to_dict(safe.last_trace) if safe.last_trace else None
                yield _sse_delta(
                    completion_id, created, model_hint,
                    finish_reason=_finish_reason(safe),
                    x_safety_trace=trace_dict,
                )
                yield b"data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        reply, trace = safe.chat(messages)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_hint,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": _finish_reason(safe),
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "x_safety_trace": trace_to_dict(trace),
        })

    return app


def _finish_reason(safe: SafeLLM) -> str:
    """Map a SafeLLM final action to an OpenAI-style finish_reason."""
    trace = safe.last_trace
    if trace is None:
        return "stop"
    if trace.final_action == "block":
        return "content_filter"
    if trace.during_gen.terminated_early:
        return "content_filter"
    return "stop"


def _sse_delta(
    completion_id: str,
    created: int,
    model: str,
    *,
    role: str | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
    x_safety_trace: dict | None = None,
) -> bytes:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    if x_safety_trace is not None:
        payload["x_safety_trace"] = x_safety_trace
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


try:
    app = _make_app()
except ImportError:
    # Allows `import safe_llm.server` to succeed on installs without the
    # `server` extra — the server just isn't usable.
    app = None  # type: ignore[assignment]
