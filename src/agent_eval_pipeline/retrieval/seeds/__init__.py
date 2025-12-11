"""
Seed data for the retrieval system.

This package contains externalized knowledge base content.
Separating data from infrastructure enables:
- Content updates without code changes
- Different datasets for different environments
- Easy testing with controlled data
"""

from agent_eval_pipeline.retrieval.seeds.medical_knowledge import (
    get_medical_documents,
    seed_vector_store,
)

__all__ = ["get_medical_documents", "seed_vector_store"]
