"""
Embedding configuration.

Define configs for each embedding model provider separately.
Use EmbeddingConfig to specify which one is active in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAIEmbeddingConfig:
    """OpenAI embedding model configuration."""
    model: str = "text-embedding-3-small"
    dimensions: int = 512
    api_key: Optional[str] = None
    batch_size: int = 32
    enabled: bool = True


@dataclass
class CohereEmbeddingConfig:
    """Cohere embedding model configuration."""
    model: str = "embed-english-v3.0"
    dimensions: int = 1024
    api_key: Optional[str] = None
    batch_size: int = 32
    enabled: bool = True

@dataclass
class EmbeddingConfig:
    """
    Main embedding configuration.

    Specifies which embedding model provider is currently active in production
    and its parameters.

    Usage example
    """

    # Which provider is active: "openai", "cohere", etc.
    active_provider: str = "openai"

    # Provider-specific configs (all available options)
    openai: OpenAIEmbeddingConfig = field(default_factory=OpenAIEmbeddingConfig)
    cohere: CohereEmbeddingConfig = field(default_factory=CohereEmbeddingConfig)

    def get_active_config(self):
        """Get the currently active provider's configuration."""
        if self.active_provider == "openai":
            return self.openai
        elif self.active_provider == "cohere":
            return self.cohere
        else:
            raise ValueError(f"Unknown provider: {self.active_provider}")
