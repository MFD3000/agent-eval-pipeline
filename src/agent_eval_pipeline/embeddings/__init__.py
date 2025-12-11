"""
Embeddings module - text embedding generation.

This is the GOLD STANDARD for code elevation in this project.
It demonstrates the ideal pattern:
1. Protocol (EmbeddingProvider) defines the interface
2. Production implementation (OpenAIEmbeddings)
3. Test double (MockEmbeddings) for fast testing
4. Factory function (get_embedding_provider)

INTERVIEW TALKING POINT:
------------------------
"The embeddings module is our reference architecture. Protocol defines
the contract, OpenAIEmbeddings is production, MockEmbeddings is for tests.
The factory function chooses the implementation. When testing the entire
agent, I inject MockEmbeddings - no API calls, deterministic results,
sub-millisecond execution."
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
