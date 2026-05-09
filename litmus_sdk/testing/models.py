"""
Data models for Litmus SDK.

These dataclasses represent the core domain objects:
  PromptComponents  — decomposed prompt (static vs dynamic parts)
  LLMTrace          — a single captured LLM call
  TestCase          — a golden test definition
  TestResult        — one execution of a test case
  TestReport        — aggregate report over N runs of M test cases
  DriftReport       — version-over-version embedding drift comparison
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Prompt decomposition
# ---------------------------------------------------------------------------

@dataclass
class PromptComponents:
    """
    Decompose a prompt into its trackable parts.

    Static parts (system_prompt, task_template) change between versions and
    are the "constants" in the drift equation.
    Dynamic parts (user_input, context) change per run and are "variables".

    Attributes:
        version:       Version string this prompt belongs to (e.g. "v1.0.0").
        system_prompt: The system / backstory message sent to the LLM.
        user_input:    The user-facing query or instruction.
        context:       Any additional dynamic context (RAG chunks, tool results,
                       conversation history). Stored as a free-form dict.
        final_prompt:  The fully-assembled prompt string sent to the LLM
                       (concatenation of all messages).
        project_id:    Project identifier that scopes this prompt.
    """
    version: str
    system_prompt: str
    user_input: str
    context: Dict[str, Any] = field(default_factory=dict)
    final_prompt: str = ""
    project_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "system_prompt": self.system_prompt,
            "user_input": self.user_input,
            "context": str(self.context),
            "final_prompt": self.final_prompt,
            "project_id": self.project_id,
        }


# ---------------------------------------------------------------------------
# Single LLM call trace
# ---------------------------------------------------------------------------

@dataclass
class LLMTrace:
    """
    A single captured LLM call — one row in the traces table.

    Attributes:
        trace_id:           UUID string, unique per call.
        run_id:             Groups multiple traces within one test run.
                            Set by GoldenTestRunner; None during passive monitoring.
        version:            Version string at capture time.
        project_id:         Project identifier that scopes this trace.
        timestamp:          Wall-clock time when the call completed.
        prompt_components:  Decomposed prompt.
        model:              Model identifier (e.g. "gpt-4o", "claude-3-sonnet").
        temperature:        Temperature setting used.
        response:           The LLM's text response.
        prompt_embedding:   Embedding vector of final_prompt  (shape: [1536]).
                            None until embeddings are computed.
        response_embedding: Embedding vector of response text (shape: [1536]).
                            None until embeddings are computed.
        tokens_input:       Prompt token count (from LLM usage metadata).
        tokens_output:      Completion token count.
        latency_ms:         End-to-end latency in milliseconds.
        cost:               Estimated USD cost of the call.
        agent_name:         Optional — name of the agent/chain that made the call
                            (useful for multi-step pipelines).
        step_number:        Which step in a multi-step agent pipeline (1-indexed).
    """
    trace_id: str
    run_id: Optional[str]
    version: str
    timestamp: datetime
    prompt_components: PromptComponents
    model: str
    temperature: float
    response: str
    project_id: Optional[str] = None
    prompt_embedding: Optional[np.ndarray] = None
    response_embedding: Optional[np.ndarray] = None
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    agent_name: Optional[str] = None
    step_number: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "version": self.version,
            "project_id": self.project_id,
            "timestamp": self.timestamp.isoformat(),
            "system_prompt": self.prompt_components.system_prompt,
            "user_input": self.prompt_components.user_input,
            "context": str(self.prompt_components.context),
            "final_prompt": self.prompt_components.final_prompt,
            "model": self.model,
            "temperature": self.temperature,
            "response": self.response,
            "prompt_embedding": (
                self.prompt_embedding.tolist()
                if self.prompt_embedding is not None
                else None
            ),
            "response_embedding": (
                self.response_embedding.tolist()
                if self.response_embedding is not None
                else None
            ),
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "agent_name": self.agent_name,
            "step_number": self.step_number,
        }


# ---------------------------------------------------------------------------
# Test runner data models
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """
    A single golden test definition.

    Attributes:
        test_id:            UUID string.
        name:               Human-readable identifier used in reports.
        inputs:             Keyword arguments passed to run_function.
        expected_pattern:   Regex pattern that the output must match to pass.
        expected_behavior:  Human-readable description of expected behaviour
                            (used in reports and LLM-as-judge prompts).
        run_function:       Callable that exercises the agent under test.
                            Should accept **inputs as kwargs and return a string.
    """
    test_id: str
    name: str
    inputs: Dict[str, Any]
    expected_pattern: str
    expected_behavior: str
    run_function: Optional[Any] = None  # Callable, typed as Any to avoid import


@dataclass
class TestCaseRecord:
    """
    Storage record for a test case definition.

    Attributes:
        test_id:            UUID of the test case.
        name:               Human-readable name.
        inputs:             Input parameters as dict.
        expected_pattern:   Regex pattern for pass condition.
        expected_behavior:  Human-readable description.
        project_id:         Project this test case belongs to.
    """
    test_id: str
    name: str
    inputs: Dict[str, Any]
    expected_pattern: str
    expected_behavior: str
    project_id: Optional[str] = None


@dataclass
class TestRunRecord:
    """
    Storage record for a single test run execution.

    Attributes:
        version:      Version under test.
        test_id:      UUID of the test case.
        run_number:   Which run (1-indexed).
        output:       Actual output produced.
        passed:       Whether it passed.
        latency_ms:   Execution time in milliseconds.
    """
    version: str
    test_id: str
    run_number: int
    output: str
    passed: bool
    latency_ms: float


@dataclass
class TestResult:
    """
    Result of a single execution of a TestCase.

    Attributes:
        test_name:   Name of the parent TestCase.
        run_number:  Which iteration this is (1-indexed, up to num_runs).
        output:      The actual string output produced.
        passed:      Whether the output matched expected_pattern.
        latency_ms:  Wall-clock duration of the agent call.
        timestamp:   When this run completed.
        error:       Exception message if the run raised, else None.
    """
    test_name: str
    run_number: int
    output: str
    passed: bool
    latency_ms: float
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class TestCaseSummary:
    """
    Summary statistics for a single test case across multiple runs.

    Attributes:
        test_id:          UUID of the test case.
        test_name:        Human-readable name.
        num_runs:         Total number of runs executed.
        pass_count:       Number of runs that passed.
        fail_count:       Number of runs that failed.
        pass_rate:        Pass rate as fraction [0, 1].
        avg_latency_ms:   Average latency across runs.
        stability_score:  Embedding-based stability [0, 1].
        outputs:          List of all outputs from the runs.
    """
    test_id: str
    test_name: str
    num_runs: int
    pass_count: int
    fail_count: int
    pass_rate: float
    avg_latency_ms: float
    stability_score: float
    outputs: List[str] = field(default_factory=list)


@dataclass
class TestReport:
    """
    Complete report for one execution of the full golden test suite.

    Attributes:
        version:     Version string under test.
        total_tests: Number of distinct test cases.
        total_runs:  Total executions (total_tests × num_runs).
        results:     Flat list of every TestResult.
        summary:     Per-test-case aggregated statistics dict.
                     Keys: test_name → {pass_rate, pass_count, fail_count,
                                        avg_latency_ms, variance, outputs}
    """
    version: str
    total_tests: int
    total_runs: int
    results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def display(self) -> None:
        """Print a formatted summary to stdout."""
        print(f"\n{'=' * 60}")
        print(f"Litmus Test Report — Version {self.version}")
        print(f"{'=' * 60}")
        print(f"Test cases : {self.total_tests}")
        print(f"Total runs : {self.total_runs}")

        all_pass = all(s["pass_rate"] == 100.0 for s in self.summary.values())
        overall = "ALL PASSING" if all_pass else "FAILURES DETECTED"
        print(f"Overall    : {overall}\n")

        for test_name, stats in self.summary.items():
            icon = "✓" if stats["pass_rate"] == 100.0 else "✗"
            print(f"  {icon} {test_name}")
            print(f"      Pass rate  : {stats['pass_rate']:.1f}%  ({stats['pass_count']}/{stats['pass_count'] + stats['fail_count']})")
            print(f"      Avg latency: {stats['avg_latency_ms']:.0f} ms")
            if stats.get("error_count", 0):
                print(f"      Errors     : {stats['error_count']}")
            print()

        print("=" * 60)


# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------

@dataclass
class TestDriftResult:
    """Per-test-case drift breakdown between two versions."""
    test_name: str
    input_drift: float    # cosine_distance(avg_prompt_a, avg_prompt_b)
    output_drift: float   # cosine_distance(avg_response_a, avg_response_b)
    stability_a: float    # within-version output consistency for version_a
    stability_b: float    # within-version output consistency for version_b
    num_runs_a: int
    num_runs_b: int
    flagged: bool         # output_drift > threshold


@dataclass
class DriftReport:
    """
    Version-over-version embedding drift comparison with per-test breakdown.

    Attributes:
        version_a:       Baseline version (e.g. "v1.0.0").
        version_b:       New version under comparison (e.g. "v1.1.0").
        avg_input_drift: Mean per-test cosine distance between prompt centroids.
        avg_output_drift:Mean per-test cosine distance between response centroids.
        stability_a:     Mean per-test within-version consistency for version_a.
        stability_b:     Mean per-test within-version consistency for version_b.
        per_test:        Per-test-case drift results keyed by test name.
        flagged_tests:   Names of test cases where output_drift exceeded threshold.
        recommendation:  "SAFE" | "REVIEW" | "WARNING" | "REGRESSION"
        details:         Free-text explanation of the recommendation.
    """
    version_a: str
    version_b: str
    avg_input_drift: float
    avg_output_drift: float
    stability_a: float
    stability_b: float
    per_test: Dict[str, "TestDriftResult"]
    flagged_tests: List[str]
    recommendation: str
    details: str = ""

    def display(self) -> None:
        """Print a formatted drift summary to stdout."""
        icon_map = {
            "SAFE": "✅",
            "REVIEW": "🔍",
            "WARNING": "⚠️",
            "REGRESSION": "🚨",
        }
        icon = icon_map.get(self.recommendation, "?")
        print(f"\n{'=' * 70}")
        print(f"Drift Report: {self.version_a} → {self.version_b}")
        print(f"{'=' * 70}")
        print(f"  Avg input drift  : {self.avg_input_drift:.4f}")
        print(f"  Avg output drift : {self.avg_output_drift:.4f}")
        print(f"  Stability [{self.version_a}] : {self.stability_a:.4f}")
        print(f"  Stability [{self.version_b}] : {self.stability_b:.4f}")
        print(f"  Recommendation   : {icon}  {self.recommendation}")
        if self.details:
            print(f"  Details          : {self.details}")

        if self.per_test:
            print(f"\n  {'Test':<28} {'InDrift':>8} {'OutDrift':>9} {'Stab-A':>7} {'Stab-B':>7} {'Runs':>6}  Flag")
            print(f"  {'-'*28} {'-'*8} {'-'*9} {'-'*7} {'-'*7} {'-'*6}  ----")
            for name, r in sorted(self.per_test.items()):
                flag = "🚩" if r.flagged else "  "
                runs = f"{r.num_runs_a}/{r.num_runs_b}"
                print(
                    f"  {name:<28} {r.input_drift:>8.4f} {r.output_drift:>9.4f}"
                    f" {r.stability_a:>7.4f} {r.stability_b:>7.4f} {runs:>6}  {flag}"
                )

        print(f"\n  Flagged tests : {self.flagged_tests or 'none'}")
        print("=" * 70)