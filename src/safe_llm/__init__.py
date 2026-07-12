"""safe-llm — a layered safety harness around any local LLM."""

from .gate import RequestTrace, SafetyGate
from .safe_llm import SafeLLM

__version__ = "0.1.0"
__all__ = ["SafeLLM", "SafetyGate", "RequestTrace"]
