"""
HTTP client for the Litmus backend.

The SDK owns no SQLite, no ChromaDB, no schema, no row parsers. Every method
on LitmusClient makes a single call to the Litmus backend, which is the
single writer for all persistent state.
"""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from litmus_sdk.testing.models import LLMTrace, PromptComponents, TestCase, TestResult


def _litmus_url() -> str:
    return os.getenv("LITMUS_URL", "http://localhost:8000").rstrip("/")


class LitmusClient:
    """
    Thin HTTP client targeting the Litmus backend. Every method round-trips
    over the wire — the SDK holds no persistent state of its own.
    """

    def __init__(
        self,
        litmus_url: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        """
        Args:
            litmus_url: Base URL of the backend. Falls back to LITMUS_URL env
                        then http://localhost:8000.
            timeout:    Per-request HTTP timeout. 60s default — embeddings +
                        drift payloads can be large, and arm64-on-Rosetta
                        deployments are slow.
        """
        self.litmus_url = (litmus_url or _litmus_url()).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.litmus_url}{path}"
        try:
            r = self._session.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Litmus backend unreachable at {url}: {exc}. "
                "Is the backend running? Try `./start.sh` in litmus-setup."
            ) from exc
        if not r.ok:
            raise RuntimeError(
                f"Litmus backend rejected POST {path}: {r.status_code} {r.text}. "
                "Is the backend running and up-to-date with this SDK version?"
            )
        try:
            return r.json()
        except ValueError:
            return {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.litmus_url}{path}"
        try:
            r = self._session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Litmus backend unreachable at {url}: {exc}. "
                "Is the backend running? Try `./start.sh` in litmus-setup."
            ) from exc
        if not r.ok:
            raise RuntimeError(
                f"Litmus backend rejected GET {path}: {r.status_code} {r.text}."
            )
        return r.json()

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def store_trace(self, trace: LLMTrace) -> bool:
        """
        POST a single trace. Embeddings are computed eagerly on this side so
        the backend only ever does I/O — keeps the backend request thread free
        of LLM-API calls.
        """
        from litmus_sdk.embeddings.utils import batch_compute_embeddings

        needs_prompt = trace.prompt_embedding is None and bool(trace.prompt_components.final_prompt)
        needs_response = trace.response_embedding is None and bool(trace.response)
        if needs_prompt or needs_response:
            texts: list[str] = []
            if needs_prompt:
                texts.append(trace.prompt_components.final_prompt)
            if needs_response:
                texts.append(trace.response)
            try:
                vecs = batch_compute_embeddings(texts)
                idx = 0
                if needs_prompt:
                    if vecs[idx] is not None:
                        trace.prompt_embedding = vecs[idx]
                    idx += 1
                if needs_response:
                    if vecs[idx] is not None:
                        trace.response_embedding = vecs[idx]
            except Exception as exc:
                print(f"[Litmus] Embedding computation failed for {trace.trace_id[:8]}: {exc}")

        self._post(f"/api/projects/{trace.project_id}/traces", trace.to_dict())
        return True

    def update_embeddings(self, trace: LLMTrace) -> bool:
        """Upsert prompt+response embeddings on an existing trace."""
        try:
            payload = {
                "version": trace.version,
                "run_id": trace.run_id,
                "final_prompt": trace.prompt_components.final_prompt,
                "response": trace.response,
                "prompt_embedding": (
                    trace.prompt_embedding.tolist()
                    if trace.prompt_embedding is not None
                    else None
                ),
                "response_embedding": (
                    trace.response_embedding.tolist()
                    if trace.response_embedding is not None
                    else None
                ),
            }
            self._post(
                f"/api/projects/{trace.project_id}/traces/{trace.trace_id}/embeddings",
                payload,
            )
            return True
        except Exception as exc:
            print(f"[Litmus] Error updating embeddings for {trace.trace_id[:8]}: {exc}")
            return False

    def get_traces_by_version(self, version: str, project_id: Optional[str] = None) -> List[LLMTrace]:
        """
        Fetch all traces for (project, version) — including embedding vectors.
        Returns rehydrated LLMTrace objects so existing drift code is unchanged.
        """
        if not project_id:
            raise ValueError("project_id is required when fetching traces from the backend")
        rows = self._get(f"/api/projects/{project_id}/traces/by-version/{version}")
        return [self._row_to_trace(r) for r in rows]

    def get_all_versions(self, project_id: Optional[str] = None) -> List[str]:
        if not project_id:
            raise ValueError("project_id is required when listing versions")
        data = self._get(f"/api/projects/{project_id}/versions-list")
        return list(data.get("versions", []))

    def get_trace_count(self, version: Optional[str] = None, project_id: Optional[str] = None) -> int:
        # Not exposed as its own endpoint — fall back to len() over the list.
        if not project_id:
            raise ValueError("project_id is required")
        if version:
            return len(self.get_traces_by_version(version, project_id=project_id))
        return sum(
            len(self.get_traces_by_version(v, project_id=project_id))
            for v in self.get_all_versions(project_id=project_id)
        )

    # ------------------------------------------------------------------
    # Row deserialization
    # ------------------------------------------------------------------

    def _row_to_trace(self, row: Dict[str, Any]) -> LLMTrace:
        """
        Rebuild an LLMTrace from the dict shape returned by the backend's
        ``/api/projects/{pid}/traces/by-version/{v}`` endpoint, which is just
        ``LLMTrace.to_dict()`` round-tripped through JSON.
        """
        pc = PromptComponents(
            version=row.get("version", ""),
            system_prompt=row.get("system_prompt") or "",
            user_input=row.get("user_input") or "",
            context=self._parse_context(row.get("context")),
            final_prompt=row.get("final_prompt") or "",
            project_id=row.get("project_id"),
        )
        prompt_emb = row.get("prompt_embedding")
        resp_emb = row.get("response_embedding")
        return LLMTrace(
            trace_id=row["trace_id"],
            run_id=row.get("run_id"),
            version=row["version"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            prompt_components=pc,
            model=row.get("model") or "unknown",
            temperature=row.get("temperature") or 0.0,
            response=row.get("response") or "",
            project_id=row.get("project_id"),
            prompt_embedding=np.array(prompt_emb, dtype=np.float32) if prompt_emb else None,
            response_embedding=np.array(resp_emb, dtype=np.float32) if resp_emb else None,
            tokens_input=row.get("tokens_input") or 0,
            tokens_output=row.get("tokens_output") or 0,
            latency_ms=row.get("latency_ms") or 0.0,
            cost=row.get("cost") or 0.0,
            agent_name=row.get("agent_name"),
            step_number=row.get("step_number") or 1,
        )

    @staticmethod
    def _parse_context(raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {}

        text = raw.strip()
        if not text:
            return {}

        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {"value": value}
        except Exception:
            pass

        try:
            value = ast.literal_eval(text)
            return value if isinstance(value, dict) else {"value": value}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------

    def store_test_case(self, tc: TestCase) -> bool:
        """
        POST a test case to the backend. ``tc.test_id`` is the SDK-generated
        UUID; we still send ``name``, ``inputs``, etc. — backend ignores its
        own auto-generated id when test_id is supplied... actually, the
        existing /test-cases endpoint always generates its own UUID. To keep
        behaviour exactly compatible with the old in-process path (which
        used INSERT OR REPLACE on test_id), we go through a small wrapper.
        """
        project_id = getattr(tc, "project_id", None)
        if not project_id:
            raise ValueError("TestCase must have a project_id to be persisted")
        payload = {
            "test_id": tc.test_id,
            "name": tc.name,
            "inputs": tc.inputs,
            "expected_pattern": tc.expected_pattern or "",
            "expected_behavior": tc.expected_behavior or "",
        }
        try:
            self._post(f"/api/projects/{project_id}/test-cases", payload)
            return True
        except Exception as exc:
            print(f"[Litmus] Error storing test case {tc.name}: {exc}")
            return False

    def get_test_cases(self, project_id: Optional[str] = None) -> List[TestCase]:
        if not project_id:
            raise ValueError("project_id is required")
        rows = self._get(f"/api/projects/{project_id}/test-cases")
        out: List[TestCase] = []
        for r in rows:
            inputs = r.get("inputs", {})
            if isinstance(inputs, str):
                try:
                    inputs = json.loads(inputs)
                except Exception:
                    inputs = {}
            out.append(TestCase(
                test_id=r["test_id"],
                name=r["name"],
                inputs=inputs,
                expected_pattern=r.get("expected_pattern") or "",
                expected_behavior=r.get("expected_behavior") or "",
            ))
        return out

    # ------------------------------------------------------------------
    # Test runs
    # ------------------------------------------------------------------

    def store_test_result(
        self,
        run_id: str,
        version: str,
        test_id: str,
        result: TestResult,
        project_id: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> bool:
        """
        POST a single test result. The backend persists the row to SQLite and
        upserts any provided embeddings into ChromaDB on its side.
        """
        from litmus_sdk.embeddings.utils import batch_compute_embeddings

        payload = {
            "run_id": run_id,
            "version": version,
            "test_id": test_id,
            "run_number": result.run_number,
            "output": result.output or "",
            "passed": bool(result.passed),
            "latency_ms": float(result.latency_ms or 0.0),
            "error": result.error,
            "timestamp": result.timestamp.isoformat(),
        }

        try:
            self._post(f"/api/projects/{project_id}/test-runs", payload)
        except Exception as exc:
            print(f"[Litmus] Error storing test result: {exc}")
            return False

        # Compute prompt/response embeddings locally and push them to the
        # backend so PCA / drift work without re-running the embedding model
        # server-side.
        texts: list[str] = []
        targets: list[str] = []
        if prompt_text:
            texts.append(prompt_text)
            targets.append("prompt")
        if result.output:
            texts.append(result.output)
            targets.append("response")

        if texts:
            try:
                vecs = batch_compute_embeddings(texts)
                prompt_vec = response_vec = None
                for target, vec in zip(targets, vecs):
                    if vec is None:
                        continue
                    if target == "prompt":
                        prompt_vec = vec
                    else:
                        response_vec = vec

                # The test-runs row uses run_id as its trace key (matches
                # _test_run_row_to_trace on the backend), so upsert the
                # embeddings under run_id, not test_id.
                if prompt_vec is not None or response_vec is not None:
                    emb_payload = {
                        "version": version,
                        "run_id": run_id,
                        "final_prompt": prompt_text or "",
                        "response": result.output or "",
                        "prompt_embedding": prompt_vec.tolist() if prompt_vec is not None else None,
                        "response_embedding": response_vec.tolist() if response_vec is not None else None,
                    }
                    self._post(
                        f"/api/projects/{project_id}/traces/{run_id}/embeddings",
                        emb_payload,
                    )
            except Exception as exc:
                print(f"[Litmus] Embedding upsert failed for test run {run_id}: {exc}")

        return True

    def get_test_results_by_version(
        self, version: str, project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not project_id:
            raise ValueError("project_id is required")
        # Re-uses the existing test-results endpoint shape.
        rows = self._get("/api/test-results", params={"version": version, "project_id": project_id, "limit": 1000})
        return list(rows) if isinstance(rows, list) else []

    # ------------------------------------------------------------------
    # Judge verdicts
    # ------------------------------------------------------------------

    def store_judge_verdict(
        self,
        version_a: str,
        version_b: str,
        verdict: Any,  # JudgeVerdict
        project_id: Optional[str] = None,
    ) -> None:
        if not project_id:
            print("[Litmus] Warning: store_judge_verdict called without project_id; skipping.")
            return
        payload = {
            "version_a": version_a,
            "version_b": version_b,
            "test_name": verdict.test_name,
            "verdict": verdict.verdict,
            "confidence": float(verdict.confidence or 0.0),
            "score": float(verdict.score or 0.5),
            "reasoning": verdict.reasoning or "",
            "output_a_sample": verdict.output_a_sample or "",
            "output_b_sample": verdict.output_b_sample or "",
            "expected_behavior": verdict.expected_behavior or "",
            "judge_model": verdict.judge_model or "",
            "judge_tokens_input": int(verdict.judge_tokens_input or 0),
            "judge_tokens_output": int(verdict.judge_tokens_output or 0),
            "judge_cost": float(verdict.judge_cost or 0.0),
            "judge_latency_ms": float(verdict.judge_latency_ms or 0.0),
            "timestamp": verdict.timestamp.isoformat(),
            "error": verdict.error,
        }
        try:
            self._post(f"/api/projects/{project_id}/judge-verdicts", payload)
        except Exception as exc:
            print(f"[Litmus] Error storing judge verdict: {exc}")

    def get_judge_verdicts(
        self,
        version_a: str,
        version_b: str,
        project_id: Optional[str] = None,
    ) -> List[Any]:
        from litmus_sdk.testing.models import JudgeVerdict

        params = {"version_a": version_a, "version_b": version_b}
        if project_id:
            params["project_id"] = project_id
        rows = self._get("/api/judge-verdicts", params=params)

        verdicts: List[Any] = []
        for r in (rows or []):
            try:
                ts_raw = r.get("timestamp")
                ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now()
            except Exception:
                ts = datetime.now()
            verdicts.append(JudgeVerdict(
                test_name=r.get("test_name", ""),
                verdict=r.get("verdict", "INCONCLUSIVE"),
                confidence=r.get("confidence") or 0.0,
                score=r.get("score") or 0.5,
                reasoning=r.get("reasoning") or "",
                output_a_sample=r.get("output_a_sample") or "",
                output_b_sample=r.get("output_b_sample") or "",
                expected_behavior=r.get("expected_behavior") or "",
                judge_model=r.get("judge_model") or "",
                judge_tokens_input=r.get("judge_tokens_input") or 0,
                judge_tokens_output=r.get("judge_tokens_output") or 0,
                judge_cost=r.get("judge_cost") or 0.0,
                judge_latency_ms=r.get("judge_latency_ms") or 0.0,
                timestamp=ts,
                error=r.get("error"),
            ))
        return verdicts

    # ------------------------------------------------------------------
    # Helpers used by the judge runner — backed by narrow GET endpoints
    # ------------------------------------------------------------------

    def get_latest_test_output(self, test_name: str, version: str, project_id: str) -> str:
        data = self._get(
            f"/api/projects/{project_id}/test-cases/by-name/{test_name}/latest-output",
            params={"version": version},
        )
        return str(data.get("output") or "")

    def get_expected_behavior(self, test_name: str, project_id: str) -> str:
        data = self._get(
            f"/api/projects/{project_id}/test-cases/by-name/{test_name}/expected-behavior"
        )
        return str(data.get("expected_behavior") or "")

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    # Convenience aliases for backward compatibility
    def get_traces(self, version: str, project_id: Optional[str] = None) -> List[LLMTrace]:
        return self.get_traces_by_version(version, project_id=project_id)

    def get_versions(self, project_id: Optional[str] = None) -> List[str]:
        return self.get_all_versions(project_id=project_id)
