from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JudgeConfig:
    """Configuration for LLM-as-a-judge evaluation."""
    model: str = ""                  # falls back to LITMUS_JUDGE_MODEL env or gpt-4o-mini
    mode: str = "flagged_only"       # "flagged_only" | "all"
    drift_threshold: float = 0.15   # output_drift above which a test is judged in flagged_only mode
