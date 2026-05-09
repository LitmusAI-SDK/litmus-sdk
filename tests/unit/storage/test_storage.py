"""
Unit tests for LitmusStorage.

Uses an in-memory SQLite database throughout.
"""
from __future__ import annotations

import unittest
import uuid
from datetime import datetime

import numpy as np

from litmus_sdk.testing.models import LLMTrace, PromptComponents, TestCase, TestResult
from litmus_sdk.storage import LitmusStorage



def _make_trace(version="v1", response="Hello world") -> LLMTrace:
    trace_id = f"trace-{uuid.uuid4()}"
    return LLMTrace(
        trace_id=trace_id,
        run_id="test_run_001",
        version=version,
        timestamp=datetime.utcnow(),
        prompt_components=PromptComponents(
            version=version,
            system_prompt="You are a helpful assistant.",
            user_input="Say hello.",
            context=None,
            final_prompt="You are a helpful assistant.\nSay hello.",
        ),
        model="gpt-4o-mini",
        temperature=0.0,
        response=response,
        prompt_embedding=np.array([0.1, 0.2, 0.3]),
        response_embedding=np.array([0.4, 0.5, 0.6]),
        tokens_input=10,
        tokens_output=5,
        latency_ms=120.0,
        cost=0.001,
        agent_name=None,
        step_number=0,
    )


class TestLitmusStorage(unittest.TestCase):

    def setUp(self):
        self.storage = LitmusStorage(":memory:", chroma_path=None)

    def tearDown(self):
        self.storage.close()

    # -----------------------------------------------------------------------
    # schema
    # -----------------------------------------------------------------------

    def test_tables_created(self):
        """All four tables exist after init."""
        cursor = self.storage.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        self.assertIn("traces", tables)
        self.assertIn("test_cases", tables)
        self.assertIn("test_runs", tables)
        self.assertIn("version_snapshots", tables)

    # -----------------------------------------------------------------------
    # trace CRUD
    # -----------------------------------------------------------------------

    def test_store_and_retrieve_trace(self):
        """Round-trip: store a trace, retrieve by version."""
        trace = _make_trace()
        self.storage.store_trace(trace)

        traces = self.storage.get_traces_by_version("v1")
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].response, "Hello world")

    def test_trace_embedding_round_trip(self):
        """Numpy embeddings survive ChromaDB upsert / fetch."""
        trace = _make_trace()
        trace.prompt_embedding = np.array([1.0, 2.0, 3.0])
        trace.response_embedding = np.array([4.0, 5.0, 6.0])

        self.storage.store_trace(trace)
        loaded = self.storage.get_traces_by_version("v1")[0]

        np.testing.assert_array_almost_equal(
            loaded.prompt_embedding, np.array([1.0, 2.0, 3.0])
        )
        np.testing.assert_array_almost_equal(
            loaded.response_embedding, np.array([4.0, 5.0, 6.0])
        )

    def test_trace_no_embedding_returns_none(self):
        """Traces stored without embeddings deserialise to None."""
        trace = _make_trace()
        trace.prompt_embedding = None
        trace.response_embedding = None

        self.storage.store_trace(trace)
        loaded = self.storage.get_traces_by_version("v1")[0]

        self.assertIsNone(loaded.prompt_embedding)
        self.assertIsNone(loaded.response_embedding)

    def test_get_traces_by_version_filters_correctly(self):
        """Only traces for the requested version are returned."""
        self.storage.store_trace(_make_trace(version="v1"))
        self.storage.store_trace(_make_trace(version="v2"))

        v1_traces = self.storage.get_traces_by_version("v1")
        v2_traces = self.storage.get_traces_by_version("v2")

        self.assertEqual(len(v1_traces), 1)
        self.assertEqual(len(v2_traces), 1)

    def test_get_versions(self):
        """get_versions returns all distinct version strings."""
        for v in ("v1", "v2", "v3", "v1"):    # v1 twice — should deduplicate
            t = _make_trace(version=v)
            t.trace_id = f"trace-{v}-{id(t)}"
            self.storage.store_trace(t)

        versions = self.storage.get_versions()
        self.assertIn("v1", versions)
        self.assertIn("v2", versions)
        self.assertIn("v3", versions)
        self.assertEqual(len(versions), 3)

    # -----------------------------------------------------------------------
    # update embeddings
    # -----------------------------------------------------------------------

    def test_update_embeddings(self):
        """update_embeddings persists new embedding values."""
        trace = _make_trace()
        trace.prompt_embedding = None
        trace.response_embedding = None
        self.storage.store_trace(trace)

        # Now backfill
        loaded = self.storage.get_traces_by_version("v1")[0]
        loaded.prompt_embedding  = np.array([0.9, 0.8, 0.7])
        loaded.response_embedding = np.array([0.6, 0.5, 0.4])
        self.storage.update_embeddings(loaded)

        updated = self.storage.get_traces_by_version("v1")[0]
        np.testing.assert_array_almost_equal(
            updated.prompt_embedding, np.array([0.9, 0.8, 0.7])
        )

    # -----------------------------------------------------------------------
    # test_cases CRUD
    # -----------------------------------------------------------------------

    def test_store_and_retrieve_test_case(self):
        """Stored TestCase can be retrieved by test_id."""
        tc = TestCase(
            test_id="tc-001",
            name="simple_select",
            inputs={"query": "Show all users"},
            expected_pattern=r"(?i)SELECT",
            expected_behavior="Returns a SELECT query.",
            run_function=None,
        )
        self.storage.store_test_case(tc)
        cases = self.storage.get_test_cases()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, "simple_select")

    # -----------------------------------------------------------------------
    # test_results CRUD
    # -----------------------------------------------------------------------

    def test_store_test_result(self):
        """TestResult is persisted and retrievable by version."""
        # First, store a test case so the FK constraint is satisfied
        tc = TestCase(
            test_id="tc-001",
            name="simple_select",
            inputs={"query": "Show all users"},
            expected_pattern=r"(?i)SELECT",
            expected_behavior="Returns a SELECT query.",
            run_function=None,
        )
        self.storage.store_test_case(tc)
        
        result = TestResult(
            test_name="simple_select",
            run_number=1,
            output="SELECT * FROM users",
            passed=True,
            latency_ms=850.0,
            timestamp=datetime.utcnow(),
            error=None,
        )
        self.storage.store_test_result(
            run_id="test_run_001",
            version="v1",
            test_id="tc-001",
            result=result,
        )
        results = self.storage.get_test_results_by_version("v1")
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["passed"])


if __name__ == "__main__":
    unittest.main()
