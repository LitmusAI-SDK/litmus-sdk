"""
Unit tests for LitmusCallback.

Mocks litellm.completion so no real API calls are made.
Verifies that traces are captured and stored correctly.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestLitmusCallback(unittest.TestCase):
    """Test LitmusCallback captures and stores LLM traces."""

    def setUp(self):
        """Create an in-memory SDK + callback instance."""
        from litmus_sdk.core import LitmusSDK

        self.sdk = LitmusSDK(db_path=":memory:", chroma_path=None)
        self.sdk.init("test-v1")
        # Grab the callback we just registered
        import litellm
        from litmus_sdk.callback import LitmusCallback
        self.cb = next(
            c for c in litellm.callbacks if isinstance(c, LitmusCallback)
        )

    def tearDown(self):
        self.sdk.close()

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------

    def _mock_kwargs(self, system="Be helpful.", user="What is Python?"):
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "litellm_params": {"metadata": {}},
            "optional_params": {"temperature": 0.0},
            "response_cost": 0.001,
        }

    def _mock_response(self, text="Python is a programming language."):
        choice = SimpleNamespace(
            message=SimpleNamespace(content=text),
            finish_reason="stop",
        )
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=20, total_tokens=32)
        return SimpleNamespace(
            model="gpt-4o-mini",
            choices=[choice],
            usage=usage,
            _response_ms=150,
        )

    # -----------------------------------------------------------------------
    # tests
    # -----------------------------------------------------------------------

    def test_log_success_creates_trace(self):
        """Sync log_success_event stores exactly one trace."""
        kwargs = self._mock_kwargs()
        response = self._mock_response()

        self.cb.log_success_event(kwargs, response, start_time=0, end_time=0.15)

        traces = self.sdk.get_traces("test-v1")
        self.assertEqual(len(traces), 1)

    def test_trace_fields_populated(self):
        """Stored trace contains correct model and response text."""
        kwargs = self._mock_kwargs()
        response = self._mock_response("Python is a scripting language.")

        self.cb.log_success_event(kwargs, response, start_time=0, end_time=0.10)

        trace = self.sdk.get_traces("test-v1")[0]
        self.assertEqual(trace.model, "gpt-4o-mini")
        self.assertEqual(trace.response, "Python is a scripting language.")
        self.assertEqual(trace.version, "test-v1")

    def test_prompt_components_extracted(self):
        """System prompt and user input are split into PromptComponents."""
        kwargs = self._mock_kwargs(system="You are expert.", user="Explain SQL.")
        response = self._mock_response("SQL is a query language.")

        self.cb.log_success_event(kwargs, response, start_time=0, end_time=0.08)

        trace = self.sdk.get_traces("test-v1")[0]
        self.assertEqual(trace.prompt_components.system_prompt, "You are expert.")
        self.assertEqual(trace.prompt_components.user_input, "Explain SQL.")

    def test_cost_captured(self):
        """Cost is read from kwargs['response_cost']."""
        kwargs = self._mock_kwargs()
        kwargs["response_cost"] = 0.005
        response = self._mock_response()

        self.cb.log_success_event(kwargs, response, start_time=0, end_time=0.1)

        trace = self.sdk.get_traces("test-v1")[0]
        self.assertAlmostEqual(trace.cost, 0.005, places=4)

    def test_token_counts_captured(self):
        """Token counts (prompt + completion) are stored."""
        kwargs = self._mock_kwargs()
        response = self._mock_response()

        self.cb.log_success_event(kwargs, response, start_time=0, end_time=0.1)

        trace = self.sdk.get_traces("test-v1")[0]
        self.assertEqual(trace.tokens_input, 12)
        self.assertEqual(trace.tokens_output, 20)

    def test_async_log_success_creates_trace(self):
        """Async path also stores a trace."""
        kwargs = self._mock_kwargs()
        response = self._mock_response("Async response.")

        asyncio.get_event_loop().run_until_complete(
            self.cb.async_log_success_event(kwargs, response, start_time=0, end_time=0.05)
        )

        traces = self.sdk.get_traces("test-v1")
        self.assertGreaterEqual(len(traces), 1)

    def test_multiple_runs_stored(self):
        """Multiple calls produce multiple traces."""
        for i in range(3):
            self.cb.log_success_event(
                self._mock_kwargs(user=f"Question {i}"),
                self._mock_response(f"Answer {i}"),
                start_time=0,
                end_time=0.1,
            )

        traces = self.sdk.get_traces("test-v1")
        self.assertEqual(len(traces), 3)


if __name__ == "__main__":
    unittest.main()
