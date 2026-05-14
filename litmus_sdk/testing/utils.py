"""
Testing utilities — helpers for test execution and reporting.

Deprecated: this module is kept for backward compatibility and is not used
by the current GoldenTestRunner implementation.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from litmus_sdk.testing.models import TestCase, TestReport, TestResult

if TYPE_CHECKING:
    from litmus_sdk.core import LitmusSDK


def run_single_test(
    sdk: "LitmusSDK",
    tc: TestCase,
    num_runs: int,
    stop_on_error: bool,
) -> List[TestResult]:
    """Execute a single test case num_runs times."""
    results: List[TestResult] = []
    pass_count = 0

    for i in range(1, num_runs + 1):
        run_id = f"{tc.name}_{sdk.version}_run{i}"
        sdk._current_run_id = run_id

        start = time.perf_counter()
        output = ""
        error: Optional[str] = None
        passed = False

        try:
            raw = tc.run_function(**tc.inputs)
            output = str(raw) if raw is not None else ""
            passed = bool(
                re.search(tc.expected_pattern, output, re.IGNORECASE | re.DOTALL)
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            output = f"[ERROR] {error}"
            if stop_on_error:
                raise

        latency_ms = (time.perf_counter() - start) * 1000

        result = TestResult(
            test_name=tc.name,
            run_number=i,
            output=output,
            passed=passed,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now(),
            error=error,
        )
        results.append(result)

        # Persist to DB
        sdk.client.store_test_result(run_id, sdk.version, tc.test_id, result)

        if passed:
            pass_count += 1

        icon = "✓" if passed else "✗"
        err_suffix = f"  [{error}]" if error else ""
        print(
            f"  {icon} {tc.name} run {i}/{num_runs} "
            f"({latency_ms:.0f} ms){err_suffix}"
        )

    pct = (pass_count / num_runs) * 100
    print(f"    → Pass rate: {pct:.0f}%\n")

    return results


def build_report(results: List[TestResult], version: str) -> TestReport:
    """Build a test report from flat results."""
    num_runs_per_test = max(r.run_number for r in results) if results else 0

    per_test: Dict[str, Dict[str, Any]] = {}

    for result in results:
        name = result.test_name
        if name not in per_test:
            per_test[name] = {
                "pass_count": 0,
                "fail_count": 0,
                "error_count": 0,
                "latencies": [],
                "outputs": [],
            }
        s = per_test[name]
        if result.passed:
            s["pass_count"] += 1
        else:
            s["fail_count"] += 1
        if result.error:
            s["error_count"] += 1
        s["latencies"].append(result.latency_ms)
        s["outputs"].append(result.output)

    summary: Dict[str, Any] = {}
    for name, s in per_test.items():
        total = s["pass_count"] + s["fail_count"]
        latencies = s["latencies"]
        summary[name] = {
            "pass_rate": (s["pass_count"] / total * 100) if total else 0.0,
            "pass_count": s["pass_count"],
            "fail_count": s["fail_count"],
            "error_count": s["error_count"],
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "min_latency_ms": min(latencies) if latencies else 0.0,
            "max_latency_ms": max(latencies) if latencies else 0.0,
            "outputs": s["outputs"],
        }

    return TestReport(
        version=version,
        total_tests=len(per_test),
        total_runs=len(results),
        results=results,
        summary=summary,
    )
