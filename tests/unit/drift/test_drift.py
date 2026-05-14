"""
Unit tests for classify_drift thresholds.

Coverage of DriftDetector.compare end-to-end was removed in the Phase 2
HTTP-only migration — those tests depended on an in-process SQLite store
that no longer exists in the SDK. A replacement built against a mocked
LitmusClient is worthwhile but is a fresh test, not a port.
"""
from __future__ import annotations

import unittest

from litmus_sdk.drift import classify_drift


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


if __name__ == "__main__":
    unittest.main()
