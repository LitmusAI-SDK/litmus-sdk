"""
OpenAI SDK patching — universal instrumentation layer.

Problem being solved
────────────────────
litellm.callbacks only fires for code that calls litellm.completion().
Modern frameworks (CrewAI 1.9+, LangChain's OpenAI backend, Agno, raw
openai usage, etc.) call openai.chat.completions.create() directly,
bypassing litellm entirely.  Those traces are invisible to LitmusSDK.

Solution
────────
Monkey-patch openai.resources.chat.completions.Completions.create (and
its async twin) at the class level so that *every* OpenAI API call made
by *any* framework in the process is automatically captured.

Double-counting guard
─────────────────────
When the user calls litellm.completion(), litellm internally calls the
OpenAI SDK too — so without a guard we'd capture the same trace twice
(once from the OpenAI patch, once from LitmusCallback/litellm).
We detect this by inspecting the call stack for litellm frames.  This
is O(n) per call but acceptable for a regression-testing tool that runs
offline — not on a production hot path.

Usage
─────
Applied automatically by LitmusSDK.init().  Removed by LitmusSDK.close().
Multiple SDK instances share the same underlying patch; the active SDK
instance is determined at call time via the module-level _active_sdk ref.
"""

from __future__ import annotations

import inspect
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from litmus_sdk.core import LitmusSDK

# Module-level ref to the currently active SDK — set by LitmusSDK.init(),
# cleared by LitmusSDK.close().  Weak enough to GC if SDK is collected.
_active_sdk: Optional["LitmusSDK"] = None

# Originals stored here so we can unpatch cleanly.
_orig_sync_create = None
_orig_async_create = None
_patched = False


def _is_litellm_call() -> bool:
    """Return True if the current call stack contains a litellm frame.

    Used to skip tracing when litellm is already handling the request via
    its own callback mechanism (LitmusCallback).
    """
    for frame_info in inspect.stack():
        module: str = frame_info.frame.f_globals.get("__name__", "")
        if module.startswith("litellm"):
            return True
    return False


def _build_trace(
    sdk: "LitmusSDK",
    messages: list,
    model: str,
    temperature: float,
    response,
    latency_ms: float,
) -> None:
    """Build an LLMTrace from an OpenAI ChatCompletion response and persist it."""
    from litmus_sdk.testing.models import LLMTrace, PromptComponents
    from litmus_sdk.callback.utils import extract_system_user

    # Extract prompt parts
    system_prompt, user_input = extract_system_user(messages)
    final_prompt = "\n\n".join(
        m.get("content", "") for m in messages if m.get("content")
    )

    pc = PromptComponents(
        version=sdk.version,
        system_prompt=system_prompt,
        user_input=user_input,
        final_prompt=final_prompt,
        project_id=sdk.project_id,
    )

    # Extract response text
    response_text = ""
    try:
        response_text = response.choices[0].message.content or ""
    except Exception:
        pass

    # Token usage
    usage = getattr(response, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
    tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

    trace = LLMTrace(
        trace_id=str(uuid.uuid4()),
        run_id=sdk._current_run_id,
        version=sdk.version,
        timestamp=datetime.now(),
        prompt_components=pc,
        model=model,
        temperature=temperature,
        response=response_text,
        project_id=sdk.project_id,
        tokens_input=int(tokens_in),
        tokens_output=int(tokens_out),
        latency_ms=round(latency_ms, 2),
        cost=0.0,
    )

    sdk.client.store_trace(trace)


def apply_patch() -> None:
    """
    Monkey-patch openai.chat.completions.Completions.create (sync + async).

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _orig_sync_create, _orig_async_create, _patched

    if _patched:
        return

    try:
        import openai.resources.chat.completions as _oai_mod
        Completions = _oai_mod.Completions
        AsyncCompletions = _oai_mod.AsyncCompletions
    except ImportError:
        # openai not installed — nothing to patch
        return

    _orig_sync_create = Completions.create
    _orig_async_create = AsyncCompletions.create

    def _sync_create(self_client, *args, **kwargs):
        sdk = _active_sdk
        if sdk is None or sdk.version is None or _is_litellm_call():
            return _orig_sync_create(self_client, *args, **kwargs)

        # Check for injected run_id in metadata (set by GoldenTestRunner patch)
        meta = kwargs.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("litmus_run_id"):
            sdk._current_run_id = meta["litmus_run_id"]

        messages = kwargs.get("messages") or (args[0] if args else [])
        model = kwargs.get("model", "unknown")
        temperature = kwargs.get("temperature", 0.0) or 0.0

        t0 = time.perf_counter()
        try:
            response = _orig_sync_create(self_client, *args, **kwargs)
        except Exception:
            raise
        latency_ms = (time.perf_counter() - t0) * 1000

        try:
            _build_trace(sdk, messages, model, temperature, response, latency_ms)
        except Exception as exc:
            print(f"[Litmus] OpenAI patch capture error (suppressed): {exc}")

        return response

    async def _async_create(self_client, *args, **kwargs):
        sdk = _active_sdk
        if sdk is None or sdk.version is None or _is_litellm_call():
            return await _orig_async_create(self_client, *args, **kwargs)

        meta = kwargs.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("litmus_run_id"):
            sdk._current_run_id = meta["litmus_run_id"]

        messages = kwargs.get("messages") or (args[0] if args else [])
        model = kwargs.get("model", "unknown")
        temperature = kwargs.get("temperature", 0.0) or 0.0

        t0 = time.perf_counter()
        try:
            response = await _orig_async_create(self_client, *args, **kwargs)
        except Exception:
            raise
        latency_ms = (time.perf_counter() - t0) * 1000

        try:
            _build_trace(sdk, messages, model, temperature, response, latency_ms)
        except Exception as exc:
            print(f"[Litmus] OpenAI async patch capture error (suppressed): {exc}")

        return response

    Completions.create = _sync_create
    AsyncCompletions.create = _async_create
    _patched = True
    print("[Litmus] OpenAI SDK patched — will capture calls from all frameworks.")


def remove_patch() -> None:
    """Restore the original openai SDK methods."""
    global _orig_sync_create, _orig_async_create, _patched

    if not _patched:
        return

    try:
        import openai.resources.chat.completions as _oai_mod
        if _orig_sync_create is not None:
            _oai_mod.Completions.create = _orig_sync_create
        if _orig_async_create is not None:
            _oai_mod.AsyncCompletions.create = _orig_async_create
    except ImportError:
        pass

    _orig_sync_create = None
    _orig_async_create = None
    _patched = False
