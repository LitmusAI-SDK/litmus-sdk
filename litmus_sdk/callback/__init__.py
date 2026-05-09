"""
Callback module — LiteLLM integration + universal OpenAI SDK patching.

Two-layer instrumentation:
  Layer 1 (LitmusCallback): litellm.callbacks — rich metadata for litellm users.
  Layer 2 (openai_patch):   Patches openai.Completions.create directly — captures
                             calls from CrewAI, LangChain, raw openai, and any
                             other framework that bypasses litellm.
"""

from .core import LitmusCallback
from . import openai_patch

__all__ = ["LitmusCallback", "openai_patch"]
