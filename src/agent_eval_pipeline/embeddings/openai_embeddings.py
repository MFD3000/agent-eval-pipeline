"""
Embeddings Module - Single Responsibility: Generate text embeddings.

This module is extracted from pgvector_store.py following code-elevation principles.
It has ONE job: convert text to vector embeddings.

SOLID PRINCIPLE: Single Responsibility
- This module ONLY handles embedding generation
- No database logic, no document handling
- Easy to swap for different embedding providers
"""

import os
from typing import Protocol

import numpy as np
from openai import OpenAI


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers - enables easy swapping."""

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        ...


class OpenAIEmbeddings:
    """
    OpenAI-based embedding provider.

    Uses text-embedding-3-small by default (1536 dimensions).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self.model = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions for the model."""
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dims.get(self.model, 1536)

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        response = self._client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []

        response = self._client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [
            np.array(item.embedding, dtype=np.float32)
            for item in response.data
        ]


class MockEmbeddings:
    """
    Mock embedding provider for testing without API calls.

    Generates deterministic pseudo-embeddings from text hashes.
    NOT for production use - only for testing/development.
    """

    def __init__(self, dimensions: int = 1536):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic pseudo-embedding from text hash."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Repeat hash to fill dimensions
        repeated = h * (self._dimensions // 32 + 1)
        return np.frombuffer(repeated, dtype=np.float32)[:self._dimensions]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


def get_embedding_provider(use_mock: bool = False) -> EmbeddingProvider:
    """
    Factory function to get the appropriate embedding provider.

    Args:
        use_mock: If True, return MockEmbeddings (for testing)
    """
    if use_mock:
        return MockEmbeddings()
    return OpenAIEmbeddings()
