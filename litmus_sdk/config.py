"""
Global SDK configuration.

Change embedding model by editing EmbeddingConfig in LitmusConfig.
Change storage backend by editing StorageConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from litmus_sdk.embeddings.config import EmbeddingConfig


# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------

@dataclass
class StorageConfig:
    """
    Controls where traces and test runs are persisted.

    Supported backends:
        - "sqlite"  → local SQLite file (default, zero-config)
    """

    backend: str = "sqlite"
    db_path: str = "litmus.db"


# ---------------------------------------------------------------------------
# Top-level SDK configuration
# ---------------------------------------------------------------------------

@dataclass
class LitmusConfig:
    """Master configuration object passed to LitmusSDK."""

    version: str = "0.0.1"
    project_id: Optional[str] = None  # auto-generated UUID if None
    run_id: Optional[str] = None      # auto-generated if None
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Drift thresholds (cosine distance)
    drift_safe_threshold: float = 0.05
    drift_review_threshold: float = 0.20
    stability_min: float = 0.80     # below this → REGRESSION regardless of drift
