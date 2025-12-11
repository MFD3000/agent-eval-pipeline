"""
pgvector-based Retrieval Module

BACKWARD COMPATIBILITY NOTICE:
------------------------------
This module has been ELEVATED following code-elevation principles.
The code has been split into focused modules:
- retrieval/document.py - Document model
- retrieval/store.py - VectorStore implementations
- retrieval/seeds/ - Medical knowledge base

This file re-exports for backward compatibility. New code should import
from the specific modules or from agent_eval_pipeline.retrieval directly.

INTERVIEW TALKING POINT:
------------------------
"We use pgvector for RAG because it lets us combine vector similarity
with traditional SQL queries - filtering by document type, freshness,
or member context. Plus, the ops team already knows Postgres, so
there's no new infrastructure to learn."
"""

# Re-export everything from the elevated modules for backward compatibility
from agent_eval_pipeline.retrieval.document import Document
from agent_eval_pipeline.retrieval.store import (
    VectorStoreConfig,
    PgVectorStore,
    InMemoryVectorStore,
    get_vector_store,
)
from agent_eval_pipeline.retrieval.seeds import (
    get_medical_documents,
    seed_vector_store,
)

# Backward compatibility alias
SimulatedVectorStore = InMemoryVectorStore

__all__ = [
    "Document",
    "VectorStoreConfig",
    "PgVectorStore",
    "InMemoryVectorStore",
    "SimulatedVectorStore",  # Alias for backward compatibility
    "get_vector_store",
    "get_medical_documents",
    "seed_vector_store",
]
