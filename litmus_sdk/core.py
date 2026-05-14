"""
LitmusSDK — the main entry point for the Litmus SDK.

Usage (two lines to instrument any agent):

    from litmus_sdk import LitmusSDK

    litmus = LitmusSDK()
    litmus.init(version="v1.0.0")

    # All subsequent LLM calls are captured — regardless of framework.
    # Works with: litellm, raw openai, CrewAI, LangChain, Agno, and any
    # library that ultimately calls the OpenAI SDK.

Advanced:
    litmus = LitmusSDK(project_id="my-project", litmus_url="http://localhost:8000")
    litmus.init(version="v1.1.0")

    drift = litmus.compare("v1.0.0", "v1.1.0")
    drift.display()
"""
from __future__ import annotations

import os
import uuid as _uuid_mod
from pathlib import Path
from typing import List, Optional

import litellm

from litmus_sdk.callback import LitmusCallback
from litmus_sdk.callback import openai_patch
from litmus_sdk.config_file import discover_config, load_config
from litmus_sdk.testing.models import DriftReport, LLMTrace
from litmus_sdk.client import LitmusClient


class LitmusSDK:
    """
    Central SDK object.

    Responsibilities:
      1. Register the LitmusCallback with LiteLLM on init().
      2. Expose convenience methods for querying traces.
      3. Delegate version comparison to DriftDetector.
      4. Carry the mutable _current_run_id context that GoldenTestRunner sets.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        litmus_url: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Create a LitmusSDK instance.

        The SDK is now a pure HTTP client. It does not open SQLite or ChromaDB — persistence lives entirely in the Litmus backend.

        Args:
            project_id: Unique identifier for this project. All traces, test runs, and drift comparisons are scoped to this project. If ``None``, uses ``LITMUS_PROJECT_ID`` env, then ``litmus.toml``'s ``[project] id``, then a fresh UUID.
            litmus_url: Base URL of the Litmus backend. Falls back to ``LITMUS_URL`` env, then ``settings.litmus_url`` in ``litmus.toml``, then ``http://localhost:8000``.
            config_path: Optional path to a ``litmus.toml`` file. If omitted,
                        SDK searches upward from the current working directory.
        """
        discovered_config_path = Path(config_path).resolve() if config_path else discover_config(os.getcwd())
        self.config_path: Optional[str] = str(discovered_config_path) if discovered_config_path else None
        self.local_config = None

        if discovered_config_path:
            try:
                self.local_config = load_config(discovered_config_path)
            except Exception as exc:
                print(f"[Litmus] Warning: failed to parse config file '{discovered_config_path}': {exc}")

        config_project_id = None
        config_litmus_url = None
        if self.local_config:
            config_project_id = self.local_config.project.id
            config_litmus_url = (self.local_config.settings or {}).get("litmus_url")

        resolved_project_id = project_id or os.getenv("LITMUS_PROJECT_ID") or config_project_id or str(_uuid_mod.uuid4())
        resolved_litmus_url = litmus_url or os.getenv("LITMUS_URL") or config_litmus_url or "http://localhost:8000"

        self.litmus_url = resolved_litmus_url
        self.client = LitmusClient(litmus_url=resolved_litmus_url)
        self.project_id: str = resolved_project_id
        self.version: Optional[str] = None
        self._current_run_id: Optional[str] = None
        self._callback: Optional[LitmusCallback] = None

    # ------------------------------------------------------------------
    # Initialisation / version auto-resolution
    # ------------------------------------------------------------------

    _ENV_CANDIDATES = [
        # Explicit Litmus overrides
        "LITMUS_RUN_LABEL",
        "LITMUS_VERSION",
        # GitHub Actions — populated automatically on push/tag events
        "GITHUB_REF_NAME",   # e.g. "v1.2.3" for tags, "main" for branches
        "GITHUB_REF",        # e.g. "refs/tags/v1.2.3"
        "GITHUB_SHA",        # commit SHA
    ]

    def _resolve_version(self, version: Optional[str]) -> tuple[str, str]:
        """
        Return (active_label, source) where source describes how the label
        was resolved (for the startup log message).
        """
        if version is not None:
            v = str(version).strip()
            if not v:
                raise ValueError("Init label cannot be empty.")
            return v, "argument"

        # Try environment variables in priority order
        for var in self._ENV_CANDIDATES:
            val = os.getenv(var, "").strip()
            if val:
                return val, f"env:{var}"

        # Last resort: stable UUID that can be renamed later in the UI
        fallback = f"run-{_uuid_mod.uuid4().hex[:8]}"
        return fallback, "uuid-fallback"

    def init(
        self,
        version: Optional[str] = None,
        *,
        label: Optional[str] = None,
        tag: Optional[str] = None,
        ref: Optional[str] = None,
    ) -> "LitmusSDK":
        """
        Set the active run label and register the LiteLLM callback.

        Calling init() again with a different version is safe — it simply
        updates self.version and replaces the existing callback registration.
        All subsequent LLM calls are tagged with the new version string.

        **Version resolution order** when no explicit label is supplied:
          1. ``version`` / ``label`` / ``tag`` / ``ref`` keyword arguments
          2. ``LITMUS_RUN_LABEL`` environment variable
          3. ``LITMUS_VERSION`` environment variable
          4. ``GITHUB_REF_NAME`` (GitHub Actions tag / branch name)
          5. ``GITHUB_REF`` (full git ref, e.g. ``refs/tags/v1.2.3``)
          6. ``GITHUB_SHA`` (commit SHA)
          7. Auto-generated short UUID (``run-<8hex>``) — can be renamed in the UI

        Args:
            version: Default run label field (backward compatible). Can be any
                     non-empty string such as a semantic version, release tag,
                     branch name, commit SHA, or experiment ID.
            label:   Alias for ``version``.
            tag:     Alias for ``version`` (useful for GitHub releases/tags).
            ref:     Alias for ``version`` (useful for git refs).

        Returns:
            self — allows chaining: ``litmus.init().get_traces()``
        """
        # Collect explicitly provided aliases
        alias_candidates = [
            ("version", version),
            ("label", label),
            ("tag", tag),
            ("ref", ref),
        ]
        provided = [(name, value) for name, value in alias_candidates if value is not None]

        if len(provided) > 1:
            # Validate that all supplied aliases agree
            first_name, first_value = provided[0]
            for name, value in provided[1:]:
                if str(value).strip() != str(first_value).strip():
                    raise ValueError(
                        f"Conflicting init labels: {first_name}={first_value!r} and {name}={value!r}. "
                        "Pass only one label (or matching duplicates)."
                    )

        # Use the first explicitly provided value (or None to trigger auto-resolve)
        explicit = provided[0][1] if provided else None
        active_label, source = self._resolve_version(explicit)

        self.version = active_label
        self._callback = LitmusCallback(self)

        # --- Layer 1: litellm.callbacks ---
        # Captures calls made through litellm.completion() / acompletion().
        # Provides richer metadata (response_cost, etc.) for litellm users.
        existing = [
            cb for cb in (litellm.callbacks or [])
            if not isinstance(cb, LitmusCallback)
        ]
        litellm.callbacks = existing + [self._callback]

        # --- Layer 2: OpenAI SDK patch ---
        # Captures calls from ANY framework that calls the OpenAI SDK directly
        # (CrewAI 1.9+, LangChain, raw openai, etc.).
        # The patch skips captures when litellm is already in the call stack
        # to avoid double-counting Layer 1 + Layer 2 traces.
        openai_patch._active_sdk = self
        openai_patch.apply_patch()

        print(
            f"[Litmus] Initialized — project: {self.project_id} "
            f"| version: {active_label} (via {source}) "
            f"| api: {self.litmus_url}"
        )
        return self

    def init_from_env(
        self,
        *,
        default: Optional[str] = None,
        env_var: str = "LITMUS_RUN_LABEL",
    ) -> "LitmusSDK":
        """
        Initialise the SDK from environment variables.

        Deprecated shorthand — ``init()`` now auto-resolves from the same
        environment variables when no explicit label is supplied.  This method
        is kept for backward compatibility and simply calls ``init()`` after
        optionally injecting ``default`` as ``LITMUS_RUN_LABEL`` when not
        already set.
        """
        # Honour a caller-supplied default only if the env var isn't already set
        if default is not None and not os.getenv(env_var, "").strip():
            os.environ[env_var] = str(default).strip()
        return self.init()

    # ------------------------------------------------------------------
    # Trace access
    # ------------------------------------------------------------------

    def get_traces(self, version: Optional[str] = None) -> List[LLMTrace]:
        """
        Retrieve all traces for *version* (defaults to the current version),
        scoped to this SDK's project_id.

        Args:
            version: Version string.  Pass None to use the currently active version.

        Returns:
            List of LLMTrace objects ordered by timestamp ascending.
        """
        v = version or self.version
        if v is None:
            raise RuntimeError("Call litmus.init(version=...) before get_traces().")
        return self.client.get_traces_by_version(v, project_id=self.project_id)

    def get_versions(self) -> List[str]:
        """Return all version strings present in the database for this project."""
        return self.client.get_all_versions(project_id=self.project_id)

    # ------------------------------------------------------------------
    # Drift comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        version_a: str,
        version_b: str,
        drift_threshold: float = 0.10,
        judge: bool = False,
        judge_config=None,
    ) -> DriftReport:
        """
        Compare two versions using embedding-space drift.

        Embeddings are computed lazily (only the first time a version is
        compared) and persisted back to SQLite so re-comparisons are free.

        Args:
            version_a:       Baseline version string.
            version_b:       New version string.
            drift_threshold: Cosine distance above which individual test cases
                             are flagged.  Default 0.10.
            judge:           When True, run an LLM-as-a-judge evaluation on
                             flagged (or all) tests after drift computation.
            judge_config:    Optional JudgeConfig instance.  Defaults to
                             JudgeConfig() with gpt-4o-mini and flagged_only mode.

        Returns:
            DriftReport with prompt_drift, output_drift, stability scores,
            flagged tests, recommendation, and (if judge=True) judge_report.
        """
        # Lazy import to avoid circular dependency at module load time
        from litmus_sdk.drift import DriftDetector

        detector = DriftDetector(self, drift_threshold=drift_threshold)
        return detector.compare(version_a, version_b, judge=judge, judge_config=judge_config)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "LitmusSDK":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close the database connection and deregister all hooks.
        Call this (or use as a context manager) when your program exits.
        """
        # Remove litellm callback
        if litellm.callbacks:
            litellm.callbacks = [
                cb for cb in litellm.callbacks
                if not isinstance(cb, LitmusCallback)
            ]
        # Remove OpenAI SDK patch and deregister this SDK as the active one
        openai_patch._active_sdk = None
        openai_patch.remove_patch()
        self.client.close()
