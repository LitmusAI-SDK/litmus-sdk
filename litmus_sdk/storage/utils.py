"""
Storage utilities — ChromaDB initialization, embedding operations, and helpers.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def init_chroma(db_path: str, chroma_path: Optional[str]):
    """Return a ChromaDB client. In-memory when chroma_path is None."""
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for embedding storage. "
            "Install it with: pip install chromadb"
        ) from exc

    if chroma_path is None:
        if db_path == ":memory:":
            # Pure in-memory for unit tests
            return chromadb.Client()
        # Default: sibling chroma/ dir next to the SQLite file
        chroma_path = os.path.join(
            os.path.dirname(os.path.abspath(db_path)), "chroma"
        )

    return chromadb.PersistentClient(path=chroma_path)


def upsert_embedding(
    collection,
    trace_id: str,
    embedding: Optional[np.ndarray],
    document: Optional[str],
    metadata: dict,
) -> None:
    """Upsert a single embedding into a ChromaDB collection if available."""
    if embedding is None:
        return
    try:
        collection.upsert(
            ids=[trace_id],
            embeddings=[embedding.tolist()],
            documents=[document or ""],
            metadatas=[metadata],
        )
    except Exception as exc:
        print(f"[Litmus] ChromaDB upsert failed for {trace_id[:8]}: {exc}")


def fetch_embeddings(collection, ids: list, include_embeddings: bool = True):
    """Fetch embeddings from ChromaDB collection."""
    try:
        return collection.get(ids=ids, include=["embeddings"] if include_embeddings else [])
    except Exception as exc:
        print(f"[Litmus] ChromaDB fetch failed: {exc}")
        return {"ids": [], "embeddings": []}
