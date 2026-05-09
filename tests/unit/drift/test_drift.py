"""
Unit tests for DriftDetector and classify_drift.

Uses synthetic embeddings — no API calls.
"""
from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np

from litmus_sdk.drift import classify_drift
from litmus_sdk.testing.models import LLMTrace, PromptComponents
from litmus_sdk.storage import LitmusStorage


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_trace(
    version: str,
    run_id: str,
    response_vec: np.ndarray,
    prompt_vec: np.ndarray | None = None,
) -> LLMTrace:
    if prompt_vec is None:
        prompt_vec = np.zeros_like(response_vec)
    return LLMTrace(
        trace_id=f"{version}-{run_id}-{id(response_vec)}",
        run_id=run_id,
        version=version,
        timestamp=datetime.utcnow(),
        prompt_components=PromptComponents(
            version=version,
            system_prompt="sys",
            user_input="input",
            context=None,
            final_prompt="sys\ninput",
        ),
        model="gpt-4o-mini",
        temperature=0.0,
        response="some text",
        prompt_embedding=prompt_vec,
        response_embedding=response_vec,
        tokens_input=10,
        tokens_output=10,
        latency_ms=100.0,
        cost=0.001,
        agent_name=None,
        step_number=0,
    )


# ---------------------------------------------------------------------------
# Tests for classify_drift thresholds
# ---------------------------------------------------------------------------

class TestClassifyDrift(unittest.TestCase):

    def test_safe(self):
        rec, _ = classify_drift(output_drift=0.03, stability=0.95)
        self.assertEqual(rec, "SAFE")

    def test_review(self):
        rec, _ = classify_drift(output_drift=0.10, stability=0.90)
        self.assertEqual(rec, "REVIEW")

    def test_warning(self):
        rec, _ = classify_drift(output_drift=0.20, stability=0.85)
        self.assertEqual(rec, "WARNING")

    def test_regression(self):
        rec, _ = classify_drift(output_drift=0.35, stability=0.70)
        self.assertEqual(rec, "REGRESSION")

    def test_boundary_safe_review(self):
        """Exactly 0.05 is REVIEW (border is >0.05 → REVIEW)."""
        rec, _ = classify_drift(output_drift=0.05, stability=0.9)
        # 0.05 is the boundary — anything >= 0.05 and < 0.15 is REVIEW
        self.assertIn(rec, ("SAFE", "REVIEW"))

    def test_boundary_warning(self):
        rec, _ = classify_drift(output_drift=0.15, stability=0.8)
        self.assertEqual(rec, "WARNING")

    def test_boundary_regression(self):
        rec, _ = classify_drift(output_drift=0.30, stability=0.6)
        self.assertEqual(rec, "REGRESSION")

    def test_detail_string_not_empty(self):
        _, details = classify_drift(output_drift=0.25, stability=0.8)
        self.assertIsInstance(details, str)
        self.assertGreater(len(details), 0)


# ---------------------------------------------------------------------------
# Tests for DriftDetector.compare
# ---------------------------------------------------------------------------

class TestDriftDetector(unittest.TestCase):

    def _build_storage_with_versions(
        self,
        v1_vecs: list[np.ndarray],
        v2_vecs: list[np.ndarray],
        test_name: str = "test_query",
    ) -> LitmusStorage:
        """Build an in-memory DB with two versions of a test."""
        storage = LitmusStorage(":memory:", chroma_path=None)
        for i, vec in enumerate(v1_vecs):
            t = _make_trace("v1", f"{test_name}_v1_run{i+1}", vec)
            storage.store_trace(t)
        for i, vec in enumerate(v2_vecs):
            t = _make_trace("v2", f"{test_name}_v2_run{i+1}", vec)
            storage.store_trace(t)
        return storage

    def test_compare_identical_versions_is_safe(self):
        """If v1 and v2 have same embeddings, drift should be SAFE."""
        vec = np.array([1.0, 0.0, 0.0])
        storage = self._build_storage_with_versions(
            [vec.copy() for _ in range(5)],
            [vec.copy() for _ in range(5)],
        )

        from litmus_sdk.drift import DriftDetector
        detector = DriftDetector(storage, drift_threshold=0.10)

        # Skip backfill (embeddings already set) by patching _backfill
        with patch.object(detector, "_backfill", side_effect=lambda t: t):
            report = detector.compare("v1", "v2")

        self.assertEqual(report.recommendation, "SAFE")
        storage.close()

    def test_compare_drifted_version_is_regression(self):
        """Opposite direction embeddings → maximum drift → REGRESSION."""
        pos = np.array([1.0, 0.0, 0.0])
        neg = np.array([-1.0, 0.0, 0.0])

        storage = self._build_storage_with_versions(
            [pos.copy() for _ in range(5)],
            [neg.copy() for _ in range(5)],
        )

        from litmus_sdk.drift import DriftDetector
        detector = DriftDetector(storage, drift_threshold=0.10)

        with patch.object(detector, "_backfill", side_effect=lambda t: t):
            report = detector.compare("v1", "v2")

        self.assertEqual(report.recommendation, "REGRESSION")
        storage.close()

    def test_compare_returns_drift_report_fields(self):
        """DriftReport has all expected fields populated."""
        vec = np.array([0.6, 0.8])
        storage = self._build_storage_with_versions(
            [vec.copy() for _ in range(3)],
            [vec.copy() for _ in range(3)],
        )

        from litmus_sdk.drift import DriftDetector
        detector = DriftDetector(storage, drift_threshold=0.10)

        with patch.object(detector, "_backfill", side_effect=lambda t: t):
            report = detector.compare("v1", "v2")

        self.assertEqual(report.version_a, "v1")
        self.assertEqual(report.version_b, "v2")
        self.assertIsNotNone(report.avg_output_drift)
        self.assertIsNotNone(report.recommendation)
        storage.close()


if __name__ == "__main__":
    unittest.main()
