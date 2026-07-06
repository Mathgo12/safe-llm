"""OpenAI-compatible HTTP proxy in front of SafeLLM.

Any client that speaks the OpenAI Chat Completions API can point its
base_url at this server and get harnessed responses transparently.

TODO:
- Use fastapi + uvicorn (in the `server` optional dependency).
- Endpoint: POST /v1/chat/completions
    body: {"model": "...", "messages": [...], "stream": bool, ...}
    behavior:
      - Non-streaming: return {"choices": [{"message": {"role":"assistant",
        "content": reply}}], "x_safety_trace": trace_dict}
      - Streaming: return text/event-stream with `data: {...}\\n\\n` deltas,
        same shape as OpenAI; the during-gen filter wraps the upstream stream.
- Construct one `SafeLLM` at startup, reuse across requests.
- Optionally tee `RequestTrace` to a JSONL sink on disk for audit.

To run (once implemented):
    uvicorn safe_llm.server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

# TODO:
# from fastapi import FastAPI
# from . import SafeLLM
# from .llm.ollama import OllamaBackend
#
# app = FastAPI(title="safe-llm")
# _llm = SafeLLM(backend=OllamaBackend("llama3.2"))
#
# @app.post("/v1/chat/completions")
# async def chat_completions(req: dict):
#     reply, trace = _llm.chat(req["messages"])
#     return {
#         "choices": [{"message": {"role": "assistant", "content": reply}}],
#         "x_safety_trace": trace,
#     }
