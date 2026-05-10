"""
Drift detection — compares two versions using embedding-space distance.

Algorithm (from TDD §4):
─────────────────────────
For each golden test case T shared between version_a and version_b:
  1. Retrieve all traces tagged with run_id matching that test case.
  2. Backfill embeddings if not already computed (lazy computation).
  3. Compute the centroid (mean) of ALL prompt  embeddings → avg_prompt_a/b
     Compute the centroid (mean) of ALL response embeddings → avg_output_a/b
  4. input_drift  = cosine_distance(avg_prompt_a,  avg_prompt_b)
     output_drift = cosine_distance(avg_output_a,  avg_output_b)
  5. stability_a/b = within-version output consistency (1 = perfectly stable)

Global drift = mean of per-test output_drift values.
Flagged test  = any test whose output_drift > threshold.
Recommendation = SAFE / REVIEW / WARNING / REGRESSION (see classify_drift).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from litmus_sdk.embeddings.utils import (
    batch_compute_embeddings,
    compute_stability,
    cosine_distance,
)
from litmus_sdk.testing.models import DriftReport, LLMTrace, TestDriftResult

from .config import DriftConfig, classify_drift
from .utils import (
    centroid_prompt_embedding,
    centroid_response_embedding,
    group_by_test,
    valid_embeddings,
)

if TYPE_CHECKING:
    from litmus_sdk.core import LitmusSDK


class DriftDetector:
    """
    Computes embedding-based drift between two SDK versions.

    Typical usage (via LitmusSDK.compare):
        drift = litmus.compare("v1.0.0", "v1.1.0")
        drift.display()

    Direct usage:
        detector = DriftDetector(sdk, drift_threshold=0.08)
        report = detector.compare("v1.0.0", "v1.1.0")
    """

    def __init__(
        self,
        sdk: "LitmusSDK",
        drift_threshold: float = 0.10,
    ) -> None:
        self.sdk = sdk
        self.config = DriftConfig(threshold_review=drift_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        version_a: str,
        version_b: str,
        judge: bool = False,
        judge_config: Optional[object] = None,
    ) -> DriftReport:
        """
        Compare version_a (baseline) with version_b (candidate).

        Algorithm (per-test-case centroid approach):
          1. Load & backfill traces for both versions.
          2. Group traces by test name (from run_id prefix).
          3. For each shared test:
               - avg_prompt_a  = centroid of all prompt  embeddings in version_a
               - avg_prompt_b  = centroid of all prompt  embeddings in version_b
               - avg_output_a  = centroid of all response embeddings in version_a
               - avg_output_b  = centroid of all response embeddings in version_b
               - input_drift   = cosine_distance(avg_prompt_a,  avg_prompt_b)
               - output_drift  = cosine_distance(avg_output_a,  avg_output_b)
               - stability_a/b = within-version output consistency
          4. Aggregate: global averages of per-test drifts / stabilities.
          5. Classify → SAFE / REVIEW / WARNING / REGRESSION.
          6. Optionally run LLM-as-a-judge if judge=True.
        """
        traces_a = self.sdk.get_traces(version_a)
        traces_b = self.sdk.get_traces(version_b)

        if not traces_a:
            raise ValueError(
                f"No traces found for version '{version_a}'. "
                "Run your agent with LitmusSDK.init() set to that version first."
            )
        if not traces_b:
            raise ValueError(
                f"No traces found for version '{version_b}'. "
                "Run your agent with LitmusSDK.init() set to that version first."
            )

        traces_a = self._backfill(traces_a)
        traces_b = self._backfill(traces_b)

        groups_a = group_by_test(traces_a)
        groups_b = group_by_test(traces_b)
        common_tests = sorted(set(groups_a) & set(groups_b))

        if not common_tests:
            recommendation, details = classify_drift(0.0, 1.0, self.config)
            return DriftReport(
                version_a=version_a,
                version_b=version_b,
                avg_input_drift=0.0,
                avg_output_drift=0.0,
                stability_a=1.0,
                stability_b=1.0,
                per_test={},
                flagged_tests=[],
                recommendation="UNKNOWN",
                details="No shared test cases found between the two versions.",
            )

        per_test: dict[str, TestDriftResult] = {}

        for test_name in common_tests:
            t_a = groups_a[test_name]
            t_b = groups_b[test_name]

            # --- Input (prompt) drift ---
            avg_prompt_a = centroid_prompt_embedding(t_a)
            avg_prompt_b = centroid_prompt_embedding(t_b)
            if avg_prompt_a is not None and avg_prompt_b is not None:
                input_drift = cosine_distance(avg_prompt_a, avg_prompt_b)
            else:
                input_drift = 0.0

            # --- Output (response) drift ---
            avg_output_a = centroid_response_embedding(t_a)
            avg_output_b = centroid_response_embedding(t_b)
            if avg_output_a is not None and avg_output_b is not None:
                output_drift = cosine_distance(avg_output_a, avg_output_b)
            else:
                output_drift = 0.0

            # --- Within-version stability ---
            resp_embs_a = valid_embeddings(t_a)
            resp_embs_b = valid_embeddings(t_b)
            stab_a = compute_stability(resp_embs_a) if resp_embs_a else 1.0
            stab_b = compute_stability(resp_embs_b) if resp_embs_b else 1.0

            per_test[test_name] = TestDriftResult(
                test_name=test_name,
                input_drift=round(input_drift, 6),
                output_drift=round(output_drift, 6),
                stability_a=round(stab_a, 6),
                stability_b=round(stab_b, 6),
                num_runs_a=len(t_a),
                num_runs_b=len(t_b),
                flagged=(output_drift > self.config.threshold_review),
            )

        # --- Aggregate global metrics ---
        avg_input_drift = float(np.mean([r.input_drift for r in per_test.values()]))
        avg_output_drift = float(np.mean([r.output_drift for r in per_test.values()]))
        global_stab_a = float(np.mean([r.stability_a for r in per_test.values()]))
        global_stab_b = float(np.mean([r.stability_b for r in per_test.values()]))
        flagged_tests = [name for name, r in per_test.items() if r.flagged]

        recommendation, details = classify_drift(avg_output_drift, global_stab_b, self.config)

        report = DriftReport(
            version_a=version_a,
            version_b=version_b,
            avg_input_drift=round(avg_input_drift, 6),
            avg_output_drift=round(avg_output_drift, 6),
            stability_a=round(global_stab_a, 6),
            stability_b=round(global_stab_b, 6),
            per_test=per_test,
            flagged_tests=flagged_tests,
            recommendation=recommendation,
            details=details,
        )

        if judge:
            from litmus_sdk.judge import JudgeConfig, run_judge

            cfg = judge_config if judge_config is not None else JudgeConfig()
            judge_report = run_judge(self.sdk, per_test, version_a, version_b, cfg)
            report.judge_report = judge_report

            # Build judge-specific recommendation string
            verdict = judge_report.summary_verdict
            if verdict == "REGRESSION":
                report.judge_recommendation = (
                    f"Judge flagged REGRESSION ({judge_report.regression_count} tests worse)"
                )
            elif verdict == "IMPROVEMENT":
                report.judge_recommendation = (
                    f"Judge flagged IMPROVEMENT ({judge_report.improvement_count} tests better)"
                )
            elif verdict == "INCONCLUSIVE":
                report.judge_recommendation = "Judge verdict INCONCLUSIVE — not enough signal"
            else:
                report.judge_recommendation = "Judge found no significant behavioral change (NEUTRAL)"

            # final_recommendation combines drift + judge signals
            if verdict == "REGRESSION":
                report.final_recommendation = "REGRESSION"
            elif verdict == "IMPROVEMENT" and recommendation in ("SAFE", "REVIEW"):
                report.final_recommendation = "IMPROVEMENT"
            else:
                report.final_recommendation = recommendation

        return report

    # ------------------------------------------------------------------
    # Embedding backfill
    # ------------------------------------------------------------------

    def _backfill(self, traces: List[LLMTrace]) -> List[LLMTrace]:
        """
        Compute and persist embeddings for any traces that don't have them.

        Batches all text needing embedding into one API call, then updates
        the database.  Skips traces that already have embeddings.
        """
        need_response = [
            t for t in traces if t.response_embedding is None and t.response
        ]
        need_prompt = [
            t
            for t in traces
            if t.prompt_embedding is None and t.prompt_components.final_prompt
        ]

        # --- Response embeddings ---
        if need_response:
            texts = [t.response for t in need_response]
            vecs = batch_compute_embeddings(texts)
            for trace, vec in zip(need_response, vecs):
                if vec is not None:
                    trace.response_embedding = vec

        # --- Prompt embeddings ---
        if need_prompt:
            texts = [t.prompt_components.final_prompt for t in need_prompt]
            vecs = batch_compute_embeddings(texts)
            for trace, vec in zip(need_prompt, vecs):
                if vec is not None:
                    trace.prompt_embedding = vec

        # Persist back to DB
        for trace in traces:
            if trace.response_embedding is not None or trace.prompt_embedding is not None:
                self.sdk.storage.update_embeddings(trace)

        return traces
