"""
GoldenTestRunner: executes golden test cases N times per version,
persists results, computes stability, and returns a TestReport.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

from litmus_sdk.config_file import LitmusProjectConfig, discover_config, load_config, resolve_entrypoint
from litmus_sdk.embeddings.utils import compute_stability
from litmus_sdk.testing.models import TestCaseRecord
from litmus_sdk.testing.models import (
    TestCase,
    TestCaseSummary,
    TestReport,
    TestResult,
)

if TYPE_CHECKING:
    from litmus_sdk.core import LitmusSDK


class GoldenTestRunner:
    """
    Runs a suite of golden tests multiple times and measures consistency.

    Usage
    -----
    runner = GoldenTestRunner(litmus_sdk_instance)
    runner.add_test_case(
        name="my_test",
        inputs={"query": "hello"},
        run_function=my_function,
        expected_pattern=r"hello",
    )
    report = runner.run_tests(num_runs=5)
    report.display()
    """

    def __init__(self, sdk: "LitmusSDK") -> None:
        self._sdk = sdk
        self._test_cases: List[TestCase] = []

    # ------------------------------------------------------------------
    # Test registration
    # ------------------------------------------------------------------

    def add_test_case(
        self,
        name: str,
        inputs: dict,
        run_function,
        expected_pattern: str = "",
        expected_behavior: str = "",
        test_id: Optional[str] = None,
    ) -> None:
        """Register a golden test case."""
        # Use a deterministic test_id based on project + name so repeated runs
        # update the same DB row instead of creating duplicates.
        if test_id is None:
            stable_key = f"{self._sdk.project_id or ''}:{name}"
            test_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, stable_key))

        tc = TestCase(
            test_id=test_id,
            name=name,
            inputs=inputs,
            run_function=run_function,
            expected_pattern=expected_pattern,
            expected_behavior=expected_behavior,
        )
        self._test_cases.append(tc)

        # Persist test case definition so the storage layer knows about it
        storage = self._sdk.storage
        if storage:
            storage.store_test_case(
                TestCaseRecord(
                    test_id=tc.test_id,
                    name=tc.name,
                    inputs=tc.inputs,
                    expected_pattern=tc.expected_pattern,
                    expected_behavior=tc.expected_behavior,
                    project_id=self._sdk.project_id,
                )
            )

    def load_from_config(
        self,
        config: Optional[LitmusProjectConfig] = None,
        *,
        config_path: Optional[str] = None,
        run_function: Optional[Callable] = None,
        clear_existing: bool = True,
    ) -> int:
        """
        Load test cases from a litmus.toml config.

        If run_function is not supplied, the function is resolved from
        `agent.entrypoint` using `module:function` syntax.

        Returns:
            Number of test cases loaded.
        """
        effective_path = Path(config_path).resolve() if config_path else None
        effective_config = config

        if effective_config is None:
            if effective_path is None and self._sdk.config_path:
                effective_path = Path(self._sdk.config_path)
            if effective_path is None:
                effective_path = discover_config()
            if effective_path is None:
                raise FileNotFoundError("No litmus.toml found. Provide config_path or create one.")
            effective_config = load_config(effective_path)

        if clear_existing:
            self._test_cases.clear()

        active_run_function = run_function
        if active_run_function is None:
            entrypoint = effective_config.agent.entrypoint
            if not entrypoint:
                raise ValueError("No run function provided and no [agent].entrypoint found in litmus.toml.")
            base_dir = effective_path.parent if effective_path else None
            active_run_function = resolve_entrypoint(entrypoint, base_dir=base_dir)

        for test in effective_config.golden_tests:
            self.add_test_case(
                name=test.name,
                inputs=test.inputs,
                run_function=active_run_function,
                expected_pattern=test.expected_pattern,
                expected_behavior=test.expected_behavior,
                test_id=test.test_id,
            )

        return len(effective_config.golden_tests)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_tests(self, num_runs: int = 5) -> TestReport:
        """
        Execute every registered test case `num_runs` times.

        Returns a TestReport with per-test and aggregate statistics.
        """
        version = self._sdk.version
        all_results: List[TestResult] = []
        summaries: List[TestCaseSummary] = []

        for tc in self._test_cases:
            results = self._run_single_test(tc, num_runs, version)
            all_results.extend(results)
            summary = self._summarise(tc, results)
            summaries.append(summary)

        total_runs = len(all_results)
        total_passed = sum(1 for r in all_results if r.passed)
        overall_pass_rate = total_passed / total_runs if total_runs else 0.0
        overall_stability = (
            sum(s.stability_score for s in summaries) / len(summaries)
            if summaries
            else 1.0
        )

        # Build summary dict for each test
        summary_dict = {}
        for s in summaries:
            summary_dict[s.test_name] = {
                "pass_rate": s.pass_rate * 100.0,  # Convert to percentage
                "pass_count": s.pass_count,
                "fail_count": s.fail_count,
                "avg_latency_ms": s.avg_latency_ms,
                "stability_score": s.stability_score,
                "outputs": s.outputs,
            }

        report = TestReport(
            version=version,
            total_tests=len(self._test_cases),
            total_runs=total_runs,
            results=all_results,
            summary=summary_dict,
        )
        return report

    def run_and_report(self, *, version: Optional[str] = None, num_runs: int = 5) -> TestReport:
        if version:
            self._sdk.init(version=version)
        elif not self._sdk.version:
            self._sdk.init()
        return self.run_tests(num_runs=num_runs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_single_test(
        self,
        tc: TestCase,
        num_runs: int,
        version: str,
    ) -> List[TestResult]:
        results = []
        storage = self._sdk.storage

        for i in range(1, num_runs + 1):
            # Set run_id BEFORE calling run_function.
            # Format: "{test_name}_{version}_run{i}" — parsed by group_by_test().
            # The OpenAI SDK patch (openai_patch.py) reads this synchronously
            # *during* the API call, so it's always populated at the right time.
            run_id = f"{tc.name}_{version}_run{i}"
            self._sdk._current_run_id = run_id

            start = time.perf_counter()
            output = ""
            error: Optional[str] = None

            try:
                output = tc.run_function(**tc.inputs)
                if not isinstance(output, str):
                    output = str(output)
            except Exception as exc:
                error = str(exc)
                output = ""
            finally:
                self._sdk._current_run_id = None

            latency_ms = (time.perf_counter() - start) * 1000

            passed = self._check_pass(output, tc.expected_pattern, error)

            result = TestResult(
                test_name=tc.name,
                run_number=i,
                output=output,
                passed=passed,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                error=error,
            )
            results.append(result)

            # Persist run record — run_id matches what the callback tagged on traces
            if storage:
                storage.store_test_result(
                    run_id=run_id,
                    version=version,
                    test_id=tc.test_id,
                    result=result,
                    project_id=self._sdk.project_id,
                )

        return results

    @staticmethod
    def _check_pass(output: str, pattern: str, error: Optional[str]) -> bool:
        if error:
            return False
        if not pattern:
            # No pattern specified → pass if we got any output
            return bool(output.strip())
        return bool(re.search(pattern, output, re.IGNORECASE | re.DOTALL))

    def _summarise(self, tc: TestCase, results: List[TestResult]) -> TestCaseSummary:
        pass_count = sum(1 for r in results if r.passed)
        fail_count = len(results) - pass_count
        pass_rate = pass_count / len(results) if results else 0.0
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0.0
        outputs = [r.output for r in results]

        # Compute embedding-based stability if available
        stab = self._embedding_stability(outputs)

        return TestCaseSummary(
            test_id=tc.test_id,
            test_name=tc.name,
            num_runs=len(results),
            pass_count=pass_count,
            fail_count=fail_count,
            pass_rate=pass_rate,
            avg_latency_ms=avg_latency,
            stability_score=stab,
            outputs=outputs,
        )

    def _embedding_stability(self, outputs: List[str]) -> float:
        """
        If the SDK has an embedding provider, compute embedding stability.
        Falls back to exact-match stability (fraction of identical outputs).
        """
        provider = getattr(self._sdk, "_embedding_provider", None)
        if provider is None or not outputs:
            # Fallback: fraction of outputs that are identical to the most common
            if not outputs:
                return 1.0
            most_common = max(set(outputs), key=outputs.count)
            return outputs.count(most_common) / len(outputs)

        try:
            embeddings = provider.embed_batch(outputs)
            return compute_stability(embeddings)
        except Exception:
            return 1.0  # don't crash the runner on embedding failure
