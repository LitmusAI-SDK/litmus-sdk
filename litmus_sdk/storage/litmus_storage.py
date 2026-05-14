"""
Hybrid storage layer for Litmus SDK.

Architecture:
    SQLite   — all relational data: trace metadata, test cases, test runs,
                version snapshots.  WAL mode, zero external infrastructure.
    ChromaDB — vector store for embeddings.  Two persistent collections:
                  • litmus_prompt_embeddings
                  • litmus_response_embeddings
               Documents are keyed on trace_id so SQLite and Chroma stay
               in sync.  ChromaDB stores on disk by default (chroma_path).
"""

from __future__ import annotations

import ast
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import numpy as np
import requests

from litmus_sdk.testing.models import LLMTrace, PromptComponents, TestCase, TestResult

from .schema import init_schema
from .utils import init_chroma, upsert_embedding, fetch_embeddings


def _api_url() -> str:
    return os.getenv("LITMUS_API_URL", "http://localhost:8000").rstrip("/")


def _post_or_raise(path: str, payload: Dict[str, Any]) -> None:
    """
    POST `payload` to the Litmus backend. Raises RuntimeError with an
    actionable message on any failure — the SDK must never silently fall
    back to local SQLite, which is what caused the host/container WAL
    divergence the previous architecture suffered from.
    """
    url = f"{_api_url()}{path}"
    try:
        r = requests.post(url, json=payload, timeout=10)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Litmus backend unreachable at {url}: {exc}. "
            "Is the backend running? Try `./start.sh` in litmus-setup."
        ) from exc
    if not r.ok:
        raise RuntimeError(
            f"Litmus backend rejected write to {path}: "
            f"{r.status_code} {r.text}. "
            "Is the backend running and up-to-date with this SDK version?"
        )


class LitmusStorage:
    """
    Hybrid persistence layer: SQLite for metadata, ChromaDB for embeddings.

    Args:
        db_path:    Path to the SQLite file (use ":memory:" for tests).
        chroma_path: Directory where ChromaDB persists its data.  Defaults
                     to a ``chroma/`` folder next to the SQLite file.  Pass
                     ``None`` to use an in-memory ChromaDB client (tests).

    Thread safety: ``check_same_thread=False`` + WAL mode for SQLite.
    ChromaDB's Python client is thread-safe by default.
    """

    # ChromaDB collection names
    _PROMPT_COLLECTION = "litmus_prompt_embeddings"
    _RESPONSE_COLLECTION = "litmus_response_embeddings"

    def __init__(
        self,
        db_path: str = "./litmus.db",
        chroma_path: Optional[str] = None,
    ) -> None:
        self.db_path = db_path

        _db_dir = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(_db_dir, exist_ok=True)

        # --- SQLite ---
        # isolation_level=None → autocommit; each INSERT commits immediately
        # and conn.commit() becomes a no-op, avoiding Python 3.12 WAL errors.
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        init_schema(self.conn)

        # --- ChromaDB ---
        self._chroma_client = init_chroma(db_path, chroma_path)
        self._prompt_col = self._chroma_client.get_or_create_collection(
            name=self._PROMPT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._response_col = self._chroma_client.get_or_create_collection(
            name=self._RESPONSE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def store_trace(self, trace: LLMTrace) -> bool:
        """
        Persist a single LLMTrace via the backend HTTP API.

        SQLite write + ChromaDB upsert both happen inside the backend — the
        SDK never opens the shared DB itself. Embeddings are computed
        locally (best effort) and forwarded in the POST body so the backend
        can persist them into its in-container Chroma without re-running
        the embedding model.
        """
        from litmus_sdk.embeddings.utils import batch_compute_embeddings

        # Best-effort eager embedding computation so PCA / drift work
        # immediately. Failures are non-fatal — the backend will simply
        # store None for missing vectors.
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

        payload = trace.to_dict()  # already serializes embeddings to lists
        _post_or_raise(
            f"/api/projects/{trace.project_id}/traces",
            payload,
        )
        return True

    def update_embeddings(self, trace: LLMTrace) -> bool:
        """
        Upsert newly-computed embeddings into ChromaDB.
        Called by DriftDetector after backfill computation.
        """
        try:
            meta = {"version": trace.version, "run_id": trace.run_id or "", "project_id": trace.project_id or ""}
            upsert_embedding(
                self._prompt_col,
                trace.trace_id,
                trace.prompt_embedding,
                trace.prompt_components.final_prompt,
                meta,
            )
            upsert_embedding(
                self._response_col,
                trace.trace_id,
                trace.response_embedding,
                trace.response,
                meta,
            )
            return True
        except Exception as exc:
            print(f"[Litmus] Error updating embeddings for {trace.trace_id[:8]}: {exc}")
            return False

    def get_traces_by_version(self, version: str, project_id: Optional[str] = None) -> List[LLMTrace]:
        """
        Return all traces for *version*, optionally scoped to *project_id*, ordered by timestamp.

        Results are the UNION of:
          • rows from the ``traces`` table (LiteLLM-callback traces), AND
          • rows from ``test_runs`` whose run_id is NOT already represented in
            the traces result (avoids duplicating runs that triggered both a
            LiteLLM callback AND a test_run record).
        """
        if project_id:
            rows = self.conn.execute(
                "SELECT * FROM traces WHERE version = ? AND project_id = ? ORDER BY timestamp ASC",
                (version, project_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM traces WHERE version = ? ORDER BY timestamp ASC",
                (version,),
            ).fetchall()
        traces = [self._row_to_trace(r) for r in rows]

        # --- UNION: include test_run rows not already covered by traces ---
        covered_run_ids = {t.run_id for t in traces if t.run_id}
        if project_id:
            test_rows = self.conn.execute(
                """
                SELECT tr.run_id, tr.version, tr.project_id, tr.output, tr.timestamp,
                       tr.latency_ms, tr.test_id, tr.run_number,
                       tc.name, tc.inputs
                FROM test_runs tr
                LEFT JOIN test_cases tc ON tc.test_id = tr.test_id
                WHERE tr.project_id = ? AND tr.version = ?
                  AND tr.output IS NOT NULL AND tr.output != ''
                ORDER BY tr.timestamp ASC
                """,
                (project_id, version),
            ).fetchall()
        else:
            test_rows = self.conn.execute(
                """
                SELECT tr.run_id, tr.version, tr.project_id, tr.output, tr.timestamp,
                       tr.latency_ms, tr.test_id, tr.run_number,
                       tc.name, tc.inputs
                FROM test_runs tr
                LEFT JOIN test_cases tc ON tc.test_id = tr.test_id
                WHERE tr.version = ?
                  AND tr.output IS NOT NULL AND tr.output != ''
                ORDER BY tr.timestamp ASC
                """,
                (version,),
            ).fetchall()

        for r in test_rows:
            if r["run_id"] not in covered_run_ids:
                traces.append(self._test_run_row_to_trace(r))
                covered_run_ids.add(r["run_id"])

        # Attach embeddings from ChromaDB (trace_id is the key for both real
        # traces AND synthetic test_run traces, since _test_run_row_to_trace
        # sets trace_id = run_id which is the ChromaDB key used by store_test_result)
        if not traces:
            return traces

        ids = [t.trace_id for t in traces]
        id_to_trace = {t.trace_id: t for t in traces}

        try:
            p_results = fetch_embeddings(self._prompt_col, ids, include_embeddings=True)
            for tid, emb in zip(p_results["ids"], p_results["embeddings"]):
                if emb is not None and tid in id_to_trace:
                    id_to_trace[tid].prompt_embedding = np.array(emb, dtype=np.float32)

            r_results = fetch_embeddings(self._response_col, ids, include_embeddings=True)
            for tid, emb in zip(r_results["ids"], r_results["embeddings"]):
                if emb is not None and tid in id_to_trace:
                    id_to_trace[tid].response_embedding = np.array(emb, dtype=np.float32)
        except Exception as exc:
            print(f"[Litmus] ChromaDB fetch failed for version {version}: {exc}")

        return traces

    def get_all_versions(self, project_id: Optional[str] = None) -> List[str]:
        """
        Return distinct version strings, optionally scoped to *project_id*,
        ordered by the earliest timestamp across both traces AND test_runs.
        """
        if project_id:
            rows = self.conn.execute(
                """
                SELECT version, MIN(ts) AS first_seen
                FROM (
                    SELECT version, timestamp AS ts FROM traces    WHERE project_id = ?
                    UNION ALL
                    SELECT version, timestamp AS ts FROM test_runs WHERE project_id = ?
                )
                GROUP BY version
                ORDER BY first_seen ASC
                """,
                (project_id, project_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT version, MIN(ts) AS first_seen
                FROM (
                    SELECT version, timestamp AS ts FROM traces
                    UNION ALL
                    SELECT version, timestamp AS ts FROM test_runs
                )
                GROUP BY version
                ORDER BY first_seen ASC
                """
            ).fetchall()
        return [r["version"] for r in rows]

    def get_trace_count(self, version: Optional[str] = None, project_id: Optional[str] = None) -> int:
        if version and project_id:
            return self.conn.execute(
                "SELECT COUNT(*) FROM traces WHERE version = ? AND project_id = ?", (version, project_id)
            ).fetchone()[0]
        if version:
            return self.conn.execute(
                "SELECT COUNT(*) FROM traces WHERE version = ?", (version,)
            ).fetchone()[0]
        if project_id:
            return self.conn.execute(
                "SELECT COUNT(*) FROM traces WHERE project_id = ?", (project_id,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]

    def _row_to_trace(self, row: sqlite3.Row) -> LLMTrace:
        """Convert a SQLite metadata row into an LLMTrace (embeddings loaded separately)."""
        pc = PromptComponents(
            version=row["version"],
            system_prompt=row["system_prompt"] or "",
            user_input=row["user_input"] or "",
            context=self._parse_context(row["context"]),
            final_prompt=row["final_prompt"] or "",
            project_id=row["project_id"] if "project_id" in row.keys() else None,
        )
        return LLMTrace(
            trace_id=row["trace_id"],
            run_id=row["run_id"],
            version=row["version"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            prompt_components=pc,
            model=row["model"] or "unknown",
            temperature=row["temperature"] or 0.0,
            response=row["response"] or "",
            project_id=row["project_id"] if "project_id" in row.keys() else None,
            prompt_embedding=None,   # populated by get_traces_by_version
            response_embedding=None, # populated by get_traces_by_version
            tokens_input=row["tokens_input"] or 0,
            tokens_output=row["tokens_output"] or 0,
            latency_ms=row["latency_ms"] or 0.0,
            cost=row["cost"] or 0.0,
            agent_name=row["agent_name"],
            step_number=row["step_number"] or 1,
        )

    def _test_run_row_to_trace(self, row: sqlite3.Row) -> LLMTrace:
        """
        Convert a test_runs (JOIN test_cases) row into a synthetic LLMTrace.

        trace_id is set to run_id so that ChromaDB embedding lookups work —
        store_test_result() already upserts embeddings under the run_id key.
        """
        row_keys = row.keys()
        inputs_raw = row["inputs"] if "inputs" in row_keys else "{}"
        try:
            inputs: Dict[str, Any] = json.loads(inputs_raw or "{}")
        except Exception:
            inputs = {}

        prompt_text = (
            inputs.get("question")
            or inputs.get("query")
            or inputs.get("prompt")
            or inputs.get("text")
            or inputs.get("input")
            or "\n".join(f"{k}: {v}" for k, v in inputs.items() if k != "system_prompt")
            or ""
        )

        pc = PromptComponents(
            version=row["version"],
            system_prompt="",
            user_input=str(prompt_text),
            context={},
            final_prompt=str(prompt_text),
            project_id=row["project_id"] if "project_id" in row_keys else None,
        )
        return LLMTrace(
            trace_id=row["run_id"],   # run_id is the ChromaDB embedding key
            run_id=row["run_id"],
            version=row["version"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            prompt_components=pc,
            model="test_runner",
            temperature=0.0,
            response=row["output"] or "",
            project_id=row["project_id"] if "project_id" in row_keys else None,
            prompt_embedding=None,   # populated by get_traces_by_version
            response_embedding=None, # populated by get_traces_by_version
            tokens_input=0,
            tokens_output=0,
            latency_ms=row["latency_ms"] or 0.0,
            cost=0.0,
            agent_name=row["name"] if "name" in row_keys else None,
            step_number=row["run_number"] if "run_number" in row_keys else 1,
        )

    @staticmethod
    def _parse_context(raw: Any) -> Dict[str, Any]:
        """Parse context stored as JSON or Python-literal string into dict."""
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
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO test_cases
                    (test_id, name, project_id, inputs, expected_pattern, expected_behavior, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tc.test_id,
                    tc.name,
                    getattr(tc, 'project_id', None),
                    json.dumps(tc.inputs),
                    tc.expected_pattern,
                    tc.expected_behavior,
                    datetime.now().isoformat(),
                ),
            )
            self.conn.commit()
            return True
        except Exception as exc:
            print(f"[Litmus] Error storing test case {tc.name}: {exc}")
            return False

    def get_test_cases(self) -> List[TestCase]:
        rows = self.conn.execute("SELECT * FROM test_cases").fetchall()
        return [
            TestCase(
                test_id=r["test_id"],
                name=r["name"],
                inputs=json.loads(r["inputs"]),
                expected_pattern=r["expected_pattern"] or "",
                expected_behavior=r["expected_behavior"] or "",
            )
            for r in rows
        ]

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
        from litmus_sdk.embeddings.utils import batch_compute_embeddings

        # Persist the row via the backend HTTP API. The backend owns the
        # shared SQLite file; the SDK must never write to it directly,
        # because host/container access through a Docker bind mount can't
        # safely coordinate SQLite WAL state on macOS.
        _post_or_raise(
            f"/api/projects/{project_id}/test-runs",
            {
                "run_id": run_id,
                "version": version,
                "test_id": test_id,
                "run_number": result.run_number,
                "output": result.output or "",
                "passed": bool(result.passed),
                "latency_ms": float(result.latency_ms or 0.0),
                "error": result.error,
                "timestamp": result.timestamp.isoformat(),
            },
        )

        # Embeddings stay local to the SDK's ChromaDB. Single-writer per
        # process — no WAL bug since Chroma uses its own files in a
        # directory owned by this process.
        meta = {"version": version, "run_id": run_id, "project_id": project_id or ""}
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
                for target, vec, text in zip(targets, vecs, texts):
                    col = self._prompt_col if target == "prompt" else self._response_col
                    upsert_embedding(col, run_id, vec, text, meta)
            except Exception as exc:
                print(f"[Litmus] Embedding computation failed for test run {run_id}: {exc}")

        return True

    def get_test_results_by_version(self, version: str, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if project_id:
            rows = self.conn.execute(
                "SELECT * FROM test_runs WHERE version = ? AND project_id = ? ORDER BY timestamp ASC",
                (version, project_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM test_runs WHERE version = ? ORDER BY timestamp ASC",
                (version,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Version snapshots
    # ------------------------------------------------------------------

    def store_version_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        # Drop embedding keys if caller passes them — they live in ChromaDB now
        snapshot = {
            k: v
            for k, v in snapshot.items()
            if k not in ("avg_prompt_embedding", "avg_response_embedding")
        }
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO version_snapshots
                    (version, created_at, pass_rate, avg_latency_ms,
                     avg_cost, stability_score, notes)
                VALUES (
                    :version, :created_at, :pass_rate, :avg_latency_ms,
                    :avg_cost, :stability_score, :notes
                )
                """,
                snapshot,
            )
            self.conn.commit()
            return True
        except Exception as exc:
            print(f"[Litmus] Error storing version snapshot: {exc}")
            return False

    def get_version_snapshot(self, version: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM version_snapshots WHERE version = ?", (version,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Judge verdicts
    # ------------------------------------------------------------------

    def store_judge_verdict(
        self,
        version_a: str,
        version_b: str,
        verdict: Any,  # JudgeVerdict — typed as Any to avoid circular import
        project_id: Optional[str] = None,
    ) -> None:
        try:
            self.conn.execute(
                """
                INSERT INTO judge_verdicts
                    (project_id, version_a, version_b, test_name, verdict, confidence, score,
                     reasoning, output_a_sample, output_b_sample, expected_behavior,
                     judge_model, judge_tokens_input, judge_tokens_output, judge_cost,
                     judge_latency_ms, timestamp, error)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    project_id or "",
                    version_a,
                    version_b,
                    verdict.test_name,
                    verdict.verdict,
                    verdict.confidence,
                    verdict.score,
                    verdict.reasoning,
                    verdict.output_a_sample,
                    verdict.output_b_sample,
                    verdict.expected_behavior,
                    verdict.judge_model,
                    verdict.judge_tokens_input,
                    verdict.judge_tokens_output,
                    verdict.judge_cost,
                    verdict.judge_latency_ms,
                    verdict.timestamp.isoformat(),
                    verdict.error,
                ),
            )
            self.conn.commit()
        except Exception as exc:
            print(f"[Litmus] Error storing judge verdict: {exc}")

    def get_judge_verdicts(
        self,
        version_a: str,
        version_b: str,
        project_id: Optional[str] = None,
    ) -> List[Any]:
        from litmus_sdk.testing.models import JudgeVerdict

        rows = self.conn.execute(
            """
            SELECT * FROM judge_verdicts
            WHERE version_a = ? AND version_b = ?
              AND (? IS NULL OR project_id = ?)
            ORDER BY timestamp DESC
            """,
            (version_a, version_b, project_id, project_id),
        ).fetchall()

        verdicts = []
        for r in rows:
            verdicts.append(JudgeVerdict(
                test_name=r["test_name"],
                verdict=r["verdict"],
                confidence=r["confidence"] or 0.0,
                score=r["score"] or 0.5,
                reasoning=r["reasoning"] or "",
                output_a_sample=r["output_a_sample"] or "",
                output_b_sample=r["output_b_sample"] or "",
                expected_behavior=r["expected_behavior"] or "",
                judge_model=r["judge_model"] or "",
                judge_tokens_input=r["judge_tokens_input"] or 0,
                judge_tokens_output=r["judge_tokens_output"] or 0,
                judge_cost=r["judge_cost"] or 0.0,
                judge_latency_ms=r["judge_latency_ms"] or 0.0,
                timestamp=datetime.fromisoformat(r["timestamp"]),
                error=r["error"],
            ))
        return verdicts

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close SQLite connection. ChromaDB PersistentClient flushes automatically."""
        try:
            self.conn.close()
        except Exception:
            pass

    # Convenience aliases for backward compatibility
    def get_traces(self, version: str, project_id: Optional[str] = None) -> List[LLMTrace]:
        """Alias for get_traces_by_version."""
        return self.get_traces_by_version(version, project_id=project_id)

    def get_versions(self, project_id: Optional[str] = None) -> List[str]:
        """Alias for get_all_versions."""
        return self.get_all_versions(project_id=project_id)
