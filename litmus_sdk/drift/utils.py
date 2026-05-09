"""
Drift detection utilities — vector math, grouping, and filtering.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from litmus_sdk.testing.models import LLMTrace


def group_by_test(traces: List[LLMTrace]) -> Dict[str, List[LLMTrace]]:
    """
    Group traces by the test name extracted from run_id.

    run_id format set by GoldenTestRunner: "{test_name}_{version}_run{N}"
    Traces without a run_id (passive monitoring) are skipped.
    """
    groups: Dict[str, List[LLMTrace]] = {}
    for trace in traces:
        if not trace.run_id:
            continue
        # test_name is everything before the last two underscore segments
        # e.g. "simple_select_v1.0.0_run1" → "simple_select"
        parts = trace.run_id.rsplit("_", 2)
        test_name = parts[0] if len(parts) >= 3 else trace.run_id
        groups.setdefault(test_name, []).append(trace)
    return groups


def valid_embeddings(traces: List[LLMTrace]) -> List[np.ndarray]:
    """Extract valid response embeddings from traces."""
    return [t.response_embedding for t in traces if t.response_embedding is not None]


def first_prompt_embedding(traces: List[LLMTrace]) -> np.ndarray | None:
    """Get the first prompt embedding found in traces.

    .. deprecated::
        Use ``centroid_prompt_embedding`` for accurate per-test drift.
    """
    for t in traces:
        if t.prompt_embedding is not None:
            return t.prompt_embedding
    return None


def centroid_prompt_embedding(traces: List[LLMTrace]) -> np.ndarray | None:
    """
    Compute the mean (centroid) of all prompt embeddings across traces.

    Averages every trace's prompt_embedding, giving one representative vector
    per (test, version) group that is used for input-drift comparison.
    """
    embs = [t.prompt_embedding for t in traces if t.prompt_embedding is not None]
    if not embs:
        return None
    return np.mean(np.stack(embs), axis=0)


def centroid_response_embedding(traces: List[LLMTrace]) -> np.ndarray | None:
    """
    Compute the mean (centroid) of all response embeddings across traces.

    Averages every trace's response_embedding, giving one representative vector
    per (test, version) group that is used for output-drift comparison.
    """
    embs = [t.response_embedding for t in traces if t.response_embedding is not None]
    if not embs:
        return None
    return np.mean(np.stack(embs), axis=0)
