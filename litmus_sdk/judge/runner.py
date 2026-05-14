"""
LLM-as-a-judge runner.

For each test case selected by JudgeConfig.mode, samples representative
outputs from both versions and asks an LLM to compare them against the
test case's expected_behavior.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

import litellm

from litmus_sdk.judge.models import JudgeConfig
from litmus_sdk.testing.models import JudgeReport, JudgeVerdict, TestDriftResult

if TYPE_CHECKING:
    from litmus_sdk.core import LitmusSDK


_JUDGE_PROMPT = """\
You are an impartial evaluator comparing two versions of an AI agent.

Test case: {test_name}
Expected behavior: {expected_behavior}

Version A output sample:
{output_a}

Version B output sample:
{output_b}

Evaluate whether Version B is an IMPROVEMENT, NEUTRAL change, REGRESSION, or INCONCLUSIVE
compared to Version A with respect to the expected behavior.

Respond with valid JSON only:
{{
  "verdict": "IMPROVEMENT" | "NEUTRAL" | "REGRESSION" | "INCONCLUSIVE",
  "confidence": <float 0.0-1.0>,
  "score": <float 0.0-1.0, where 1.0=improvement, 0.5=neutral, 0.0=regression>,
  "reasoning": "<one or two sentences>"
}}"""


def _resolve_model(config: JudgeConfig) -> str:
    if config.model:
        return config.model
    return os.getenv("LITMUS_JUDGE_MODEL", "gpt-4o-mini")


def _sample_output(sdk: "LitmusSDK", test_name: str, version: str) -> str:
    output = sdk.client.get_latest_test_output(
        test_name, version, project_id=sdk.project_id
    )
    if output:
        return output[:2000]
    # Fall back to trace response
    traces = sdk.client.get_traces_by_version(version, project_id=sdk.project_id)
    for t in reversed(traces):
        if t.response:
            return t.response[:2000]
    return ""


def _expected_behavior(sdk: "LitmusSDK", test_name: str) -> str:
    return sdk.client.get_expected_behavior(test_name, project_id=sdk.project_id)


def _call_judge(model: str, prompt: str) -> tuple[str, int, int, float, float]:
    """Call LLM and return (raw_text, tokens_in, tokens_out, cost, latency_ms)."""
    start = time.perf_counter()
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    text = resp.choices[0].message.content or ""
    tokens_in = resp.usage.prompt_tokens if resp.usage else 0
    tokens_out = resp.usage.completion_tokens if resp.usage else 0
    cost = getattr(resp, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
    return text, tokens_in, tokens_out, cost, latency_ms


def _parse_verdict(raw: str) -> dict:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {}


def run_judge(
    sdk: "LitmusSDK",
    per_test: Dict[str, TestDriftResult],
    version_a: str,
    version_b: str,
    config: JudgeConfig,
) -> JudgeReport:
    """Run LLM-as-a-judge for selected tests and return a JudgeReport."""
    model = _resolve_model(config)

    # Decide which tests to judge
    if config.mode == "all":
        tests_to_judge = list(per_test.keys())
    else:
        tests_to_judge = [n for n, r in per_test.items() if r.flagged]

    verdicts: Dict[str, JudgeVerdict] = {}
    total_cost = 0.0
    total_latency = 0.0
    improvement = neutral = regression = inconclusive = 0

    for test_name in tests_to_judge:
        output_a = _sample_output(sdk, test_name, version_a)
        output_b = _sample_output(sdk, test_name, version_b)
        expected = _expected_behavior(sdk, test_name)
        prompt = _JUDGE_PROMPT.format(
            test_name=test_name,
            expected_behavior=expected or "Not specified",
            output_a=output_a or "(no output)",
            output_b=output_b or "(no output)",
        )

        error: Optional[str] = None
        parsed: dict = {}
        tokens_in = tokens_out = 0
        cost = latency_ms = 0.0
        try:
            raw, tokens_in, tokens_out, cost, latency_ms = _call_judge(model, prompt)
            parsed = _parse_verdict(raw)
            if not parsed.get("verdict"):
                raise ValueError(f"Could not parse verdict from: {raw[:200]}")
        except Exception as exc:
            error = str(exc)
            parsed = {"verdict": "INCONCLUSIVE", "confidence": 0.0, "score": 0.5, "reasoning": error}

        verdict_str = parsed.get("verdict", "INCONCLUSIVE")
        if verdict_str == "IMPROVEMENT":
            improvement += 1
        elif verdict_str == "NEUTRAL":
            neutral += 1
        elif verdict_str == "REGRESSION":
            regression += 1
        else:
            inconclusive += 1

        total_cost += cost
        total_latency += latency_ms

        jv = JudgeVerdict(
            test_name=test_name,
            verdict=verdict_str,
            confidence=float(parsed.get("confidence", 0.0)),
            score=float(parsed.get("score", 0.5)),
            reasoning=str(parsed.get("reasoning", "")),
            output_a_sample=output_a[:500],
            output_b_sample=output_b[:500],
            expected_behavior=expected,
            judge_model=model,
            judge_tokens_input=tokens_in,
            judge_tokens_output=tokens_out,
            judge_cost=cost,
            judge_latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc),
            error=error,
        )
        verdicts[test_name] = jv
        per_test[test_name].judge_verdict = jv

        # Persist to DB
        sdk.client.store_judge_verdict(version_a, version_b, jv, project_id=sdk.project_id)

    # Derive a summary verdict
    if not verdicts:
        summary = "NEUTRAL"
    elif regression > improvement:
        summary = "REGRESSION"
    elif improvement > regression:
        summary = "IMPROVEMENT"
    elif inconclusive == len(verdicts):
        summary = "INCONCLUSIVE"
    else:
        summary = "NEUTRAL"

    return JudgeReport(
        version_a=version_a,
        version_b=version_b,
        verdicts=verdicts,
        summary_verdict=summary,
        improvement_count=improvement,
        neutral_count=neutral,
        regression_count=regression,
        inconclusive_count=inconclusive,
        total_judge_cost=total_cost,
        total_judge_latency_ms=total_latency,
    )
