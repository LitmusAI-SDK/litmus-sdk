"""
Abstract base class for embedding providers.

Implement this to add a new provider (e.g. Cohere, sentence-transformers,
Voyage AI).  Then register it in factory.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Compute dense vector embeddings for text."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for a single text string."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Return embedding vectors for a list of texts.

        Providers that support batching should implement this efficiently.
        The default implementation just calls embed() in a loop — override
        it if the provider has a native batch endpoint.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the output vector."""
