"""
OpenAI embedding provider.

Uses the openai Python SDK to call the Embeddings API.
Supports batching to stay within rate limits.
"""

from __future__ import annotations

import os
from typing import List, Optional

from litmus_sdk.embeddings.base import EmbeddingProvider
from litmus_sdk.embeddings.config import OpenAIEmbeddingConfig


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Wraps OpenAI's text-embedding-* models."""

    def __init__(self, config: OpenAIEmbeddingConfig) -> None:
        self._config = config
        self._client = self._build_client(config.api_key)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(api_key: Optional[str]):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingProvider. "
                "Install it with: pip install openai"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment "
                "variable or pass api_key to OpenAIEmbeddingConfig."
            )
        return OpenAI(api_key=key)

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts, splitting into batches of config.batch_size
        to stay within API limits.
        """
        if not texts:
            return []

        results: List[List[float]] = []
        batch_size = self._config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self._config.model,
                input=batch,
                dimensions=self._config.dimensions,
            )
            # Response items are ordered to match input
            batch_embeddings = [item.embedding for item in response.data]
            results.extend(batch_embeddings)

        return results
