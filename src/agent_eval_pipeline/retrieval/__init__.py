"""
Retrieval module - vector similarity search for RAG.

This module provides:
- Document: The document model
- VectorStoreConfig: Configuration for stores
- PgVectorStore: PostgreSQL production store
- InMemoryVectorStore: Testing/development store
- get_vector_store(): Factory function

ARCHITECTURE:
-------------
Following the gold standard pattern from embeddings module:
1. Protocol defines the contract (in core.protocols)
2. Multiple implementations (PgVectorStore, InMemoryVectorStore)
3. Factory function for instantiation
4. Test doubles for fast unit tests

INTERVIEW TALKING POINT:
------------------------
"The retrieval layer is separated into focused modules: document.py for the
data model, store.py for implementations, seeds/ for knowledge base content.
Each module has a single responsibility. The store implementations follow
the Protocol pattern, so I can swap PgVectorStore for InMemoryVectorStore
in tests without changing any business logic."
"""

# Document model
from agent_eval_pipeline.retrieval.document import Document

# Store implementations and factory
from agent_eval_pipeline.retrieval.store import (
    VectorStoreConfig,
    PgVectorStore,
    InMemoryVectorStore,
    get_vector_store,
)

# Seed data
from agent_eval_pipeline.retrieval.seeds import (
    get_medical_documents,
    seed_vector_store,
)

__all__ = [
    # Document
    "Document",
    # Config
    "VectorStoreConfig",
    # Implementations
    "PgVectorStore",
    "InMemoryVectorStore",
    # Factory
    "get_vector_store",
    # Seeds
    "get_medical_documents",
    "seed_vector_store",
]
