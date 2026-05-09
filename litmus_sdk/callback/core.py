"""
LiteLLM native callback handler for Litmus SDK.

LitmusCallback subclasses CustomLogger — LiteLLM's officially supported
extension point for observability.  Registering it as:

    litellm.callbacks = [LitmusCallback(sdk)]

causes LiteLLM to call our hooks after every completion() or acompletion()
call made by the user's code (CrewAI, LangChain, Agno, or raw litellm).

Hooks used
──────────
  log_pre_api_call        (sync)  — injects a trace_id into litellm_params
                                    so we can correlate the pre/post events.
  log_success_event       (sync)  — stores trace for sync completion() calls.
  async_log_success_event (async) — stores trace for async acompletion() calls.
  log_failure_event / async_log_failure_event — records failed calls.

Why async hooks?
  Embedding computation hits the OpenAI API (~50-200 ms).  Running it in the
  async hook keeps it off the hot path so agent latency is unaffected.

Hooks NOT used (proxy-only, explained in TDD §6):
  async_pre_call_hook, async_moderation_hook, async_post_call_success_hook
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from litellm.integrations.custom_logger import CustomLogger

from litmus_sdk.testing.models import LLMTrace, PromptComponents

from .utils import extract_system_user, extract_response_text

if TYPE_CHECKING:
    from litmus_sdk.core import LitmusSDK


class LitmusCallback(CustomLogger):
    """
    LiteLLM CustomLogger subclass — the heart of Litmus's passive
    instrumentation.  One instance is registered per LitmusSDK instance.
    """

    def __init__(self, sdk: "LitmusSDK") -> None:
        super().__init__()
        self.sdk = sdk

    # ------------------------------------------------------------------
    # Pre-call: inject a trace_id we can retrieve post-call
    # ------------------------------------------------------------------

    def log_pre_api_call(
        self,
        model: str,
        messages: list,
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Called synchronously BEFORE the LLM API request is dispatched.

        We attach a Litmus-specific trace_id into the litellm_params →
        metadata dict so that log_success_event can find it later.
        This is the only officially-supported way to pass state from
        pre- to post-hook without monkey-patching.
        """
        # Ensure the metadata subdict exists
        litellm_params: Dict[str, Any] = kwargs.get("litellm_params", {})
        metadata: Dict[str, Any] = litellm_params.get("metadata") or {}
        if "litmus_trace_id" not in metadata:
            metadata["litmus_trace_id"] = str(uuid.uuid4())
        # Bake the current run_id into metadata NOW (synchronous pre-call hook),
        # so the async post-call callback can retrieve it even after the runner
        # has already cleared sdk._current_run_id in its finally block.
        if "litmus_run_id" not in metadata and self.sdk._current_run_id:
            metadata["litmus_run_id"] = self.sdk._current_run_id
        litellm_params["metadata"] = metadata
        kwargs["litellm_params"] = litellm_params

    # ------------------------------------------------------------------
    # Post-call (sync path: litellm.completion)
    # ------------------------------------------------------------------

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Called after a successful synchronous litellm.completion() call.
        """
        self._capture(kwargs, response_obj, start_time, end_time)

    def log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Called after a failed litellm.completion() call.

        We log a minimal trace so failures are visible in the dashboard
        and don't silently drop data points from stability calculations.
        """
        self._capture_failure(kwargs, start_time, end_time)

    # ------------------------------------------------------------------
    # Post-call (async path: litellm.acompletion)
    # ------------------------------------------------------------------

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Called after a successful async acompletion() call.

        Async variant keeps embedding computation off the hot path.
        """
        self._capture(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self._capture_failure(kwargs, start_time, end_time)

    # ------------------------------------------------------------------
    # Shared capture logic
    # ------------------------------------------------------------------

    def _capture(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Extract prompt + response, build an LLMTrace, and persist it.

        Design notes:
          • Embeddings are NOT computed here — that would add 50-200 ms to
            every LLM call.  They are computed lazily by DriftDetector when
            a version comparison is requested.
          • cost is read from kwargs["response_cost"] which LiteLLM populates
            automatically — no need for our own pricing table.
          • run_id comes from sdk._current_run_id, set by GoldenTestRunner.
        """
        if self.sdk.version is None:
            return  # SDK not initialised yet

        try:
            # Handle both datetime objects and numeric timestamps
            if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                # Numeric timestamps in seconds
                latency_ms = (end_time - start_time) * 1000
            else:
                # datetime objects
                latency_ms = (end_time - start_time).total_seconds() * 1000

            # --- Extract trace_id injected in log_pre_api_call ---
            metadata = kwargs.get("litellm_params", {}).get("metadata", {})
            trace_id = metadata.get("litmus_trace_id", str(uuid.uuid4()))

            # --- Prompt decomposition ---
            messages = kwargs.get("messages", [])
            system_prompt, user_input = extract_system_user(messages)
            final_prompt = "\n\n".join(
                m.get("content", "") for m in messages if m.get("content")
            )

            pc = PromptComponents(
                version=self.sdk.version,
                system_prompt=system_prompt,
                user_input=user_input,
                final_prompt=final_prompt,
                project_id=self.sdk.project_id,
            )

            # --- Response extraction ---
            response_text = extract_response_text(response_obj)

            # --- Usage & cost ---
            usage = getattr(response_obj, "usage", None)
            tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
            tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
            cost = kwargs.get("response_cost", 0.0) or 0.0

            # --- run_id: prefer metadata value (baked in by runner patch) ---
            run_id = (
                metadata.get("litmus_run_id")
                or self.sdk._current_run_id
            )

            # --- Build and store trace ---
            trace = LLMTrace(
                trace_id=trace_id,
                run_id=run_id,
                version=self.sdk.version,
                timestamp=datetime.now(),
                prompt_components=pc,
                model=kwargs.get("model", "unknown"),
                temperature=kwargs.get("temperature", 0.0) or 0.0,
                response=response_text,
                project_id=self.sdk.project_id,
                tokens_input=int(tokens_in),
                tokens_output=int(tokens_out),
                latency_ms=round(latency_ms, 2),
                cost=round(cost, 8),
            )

            self.sdk.storage.store_trace(trace)

        except Exception as exc:
            # Never let a Litmus error surface to the user's application
            print(f"[Litmus] Callback error (suppressed): {exc}")

    def _capture_failure(
        self,
        kwargs: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Store a minimal trace for a failed LLM call."""
        if self.sdk.version is None:
            return

        try:
            latency_ms = (end_time - start_time).total_seconds() * 1000
            messages = kwargs.get("messages", [])
            system_prompt, user_input = extract_system_user(messages)

            pc = PromptComponents(
                version=self.sdk.version,
                system_prompt=system_prompt,
                user_input=user_input,
                final_prompt="",
                project_id=self.sdk.project_id,
            )

            trace = LLMTrace(
                trace_id=str(uuid.uuid4()),
                run_id=self.sdk._current_run_id,
                version=self.sdk.version,
                timestamp=datetime.now(),
                prompt_components=pc,
                model=kwargs.get("model", "unknown"),
                temperature=kwargs.get("temperature", 0.0) or 0.0,
                response="[FAILED]",
                project_id=self.sdk.project_id,
                latency_ms=round(latency_ms, 2),
            )

            self.sdk.storage.store_trace(trace)

        except Exception as exc:
            print(f"[Litmus] Failure-capture error (suppressed): {exc}")
