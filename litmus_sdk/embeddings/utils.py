"""
Vector mathematics and embedding utilities.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np


# In-process cache: md5(text) → np.ndarray
_CACHE: dict[str, np.ndarray] = {}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two embedding vectors.

    Returns a value in [-1, 1]; for typical text embeddings the range is
    approximately [0, 1] where 1 = identical.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return similarity


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance = 1 - cosine_similarity.

    Range [0, 2]; for text embeddings typically [0, 1].
    0 = identical, 1 = orthogonal (no semantic overlap).
    """
    return 1.0 - cosine_similarity(a, b)


def compute_stability(embeddings: List[np.ndarray]) -> float:
    """
    Measure output consistency across N runs.

    Algorithm:
        1. Compute the centroid (mean) of all embeddings.
        2. Compute cosine similarity of each embedding to the centroid.
        3. Return the mean similarity.

    Returns 1.0 if there is only one embedding (trivially stable).
    Returns 0.0 if the list is empty.

    Args:
        embeddings: List of float32 arrays, all the same shape.

    Returns:
        Stability score in [0, 1]; higher is better.
    """
    if not embeddings:
        return 0.0
    if len(embeddings) == 1:
        return 1.0

    matrix = np.stack(embeddings)        # shape: (N, D)
    centroid = np.mean(matrix, axis=0)   # shape: (D,)

    sims = [cosine_similarity(e, centroid) for e in embeddings]
    stability = float(np.mean(sims))
    return stability


# ---------------------------------------------------------------------------
# Embedding computation (provider-agnostic via EmbeddingConfig)
# ---------------------------------------------------------------------------

def _get_provider():
    """Get the configured embedding provider."""
    from litmus_sdk.config import LitmusConfig
    from litmus_sdk.embeddings.openai_embeddings import OpenAIEmbeddingProvider
    
    config = LitmusConfig()
    active_provider = config.embedding.active_provider
    
    if active_provider == "openai":
        return OpenAIEmbeddingProvider(config.embedding.openai)
    elif active_provider == "cohere":
        raise NotImplementedError("Cohere provider not yet implemented")
    else:
        raise ValueError(f"Unknown embedding provider: {active_provider}")


def compute_embedding(text: str) -> Optional[np.ndarray]:
    """
    Compute an embedding for *text* using the configured provider.

    Results are cached in memory keyed on MD5(text).

    Args:
        text: Input string to embed.

    Returns:
        Float32 numpy array or None if the embedding call fails.
    """
    if not text or not text.strip():
        return None

    # Cache lookup
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    if key in _CACHE:
        return _CACHE[key]

    try:
        provider = _get_provider()
        embedding = provider.embed(text)
        vec = np.array(embedding, dtype=np.float32) if embedding else None
    except Exception as exc:
        print(f"[Litmus] Embedding failed: {exc}")
        return None

    if vec is not None:
        _CACHE[key] = vec

    return vec


def batch_compute_embeddings(texts: List[str]) -> List[Optional[np.ndarray]]:
    """
    Compute embeddings for a list of texts.

    Uses the configured provider's batch API, with in-memory caching.

    Args:
        texts: List of input strings.

    Returns:
        List of embeddings in the same order. Individual items may be None
        if the embedding for that text failed.
    """
    if not texts:
        return []

    # Separate already-cached from uncached
    keys = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
    uncached_indices = [i for i, k in enumerate(keys) if k not in _CACHE]
    uncached_texts = [texts[i] for i in uncached_indices]

    results: List[Optional[np.ndarray]] = [None] * len(texts)

    if uncached_texts:
        try:
            provider = _get_provider()
            embeddings = provider.embed_batch(uncached_texts)
            for j, embedding in enumerate(embeddings):
                if embedding:
                    vec = np.array(embedding, dtype=np.float32)
                    idx = uncached_indices[j]
                    _CACHE[keys[idx]] = vec
        except Exception as exc:
            print(f"[Litmus] Batch embedding failed: {exc}")
            # Fallback to individual calls
            for i in uncached_indices:
                _CACHE[keys[i]] = compute_embedding(texts[i])

    for i, k in enumerate(keys):
        results[i] = _CACHE.get(k)

    return results