"""
Drift detection configuration and classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DriftConfig:
    """Configuration for drift detection thresholds."""

    # Output drift thresholds (cosine distance)
    threshold_safe: float = 0.05  # output_drift < this → SAFE
    threshold_review: float = 0.15  # output_drift < this → REVIEW
    threshold_warning: float = 0.30  # output_drift < this → WARNING

    # Stability thresholds (cosine similarity to centroid)
    stability_min_safe: float = 0.90  # stability must be ≥ this for SAFE
    stability_min_review: float = 0.80  # stability must be ≥ this for REVIEW


def classify_drift(
    output_drift: float, stability: float, config: DriftConfig = None
) -> Tuple[str, str]:
    """
    Classify a version comparison into an actionable category.

    Args:
        output_drift: Cosine distance between avg output embeddings [0, 2].
        stability: Mean cosine similarity of outputs to their centroid [0, 1].
        config: DriftConfig instance; uses defaults if None.

    Returns:
        (recommendation, details_text) tuple.
    """
    if config is None:
        config = DriftConfig()

    if (
        output_drift < config.threshold_safe
        and stability >= config.stability_min_safe
    ):
        return (
            "SAFE",
            "Outputs are semantically equivalent and highly consistent.",
        )
    elif (
        output_drift < config.threshold_review
        and stability >= config.stability_min_review
    ):
        return (
            "REVIEW",
            "Outputs shifted slightly. Change may be intentional — inspect flagged tests.",
        )
    elif output_drift < config.threshold_warning:
        return (
            "WARNING",
            "Significant semantic drift detected. Verify outputs are still correct.",
        )
    else:
        return (
            "REGRESSION",
            "Large semantic shift or unstable outputs. This version likely has regressions.",
        )
