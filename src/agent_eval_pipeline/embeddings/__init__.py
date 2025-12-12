"""
Embeddings module - text embedding generation.

This is the GOLD STANDARD for code elevation in this project.
It demonstrates the ideal pattern:
1. Protocol (EmbeddingProvider) defines the interface
2. Production implementation (OpenAIEmbeddings)
3. Test double (MockEmbeddings) for fast testing
4. Factory function (get_embedding_provider)
"""

from agent_eval_pipeline.embeddings.openai_embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    MockEmbeddings,
    get_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "MockEmbeddings",
    "get_embedding_provider",
]
