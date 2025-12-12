"""
Retrieval node - fetches relevant documents from vector store.

This node is TESTABLE IN ISOLATION because:
1. VectorStore is injected, not global
2. No side effects beyond state updates
3. Deterministic given same inputs
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent_eval_pipeline.core import VectorStore
    from agent_eval_pipeline.agent.state import AgentState


def create_retrieve_node(
    store: VectorStore,
) -> Callable[[AgentState], dict]:
    """
    Factory that creates a retrieval node with injected store.

    Why a factory? So tests can inject InMemoryVectorStore.

    Args:
        store: VectorStore implementation (PgVectorStore, InMemoryVectorStore, etc.)

    Returns:
        A node function compatible with LangGraph
    """

    def retrieve_context(state: AgentState) -> dict:
        """
        Retrieve relevant medical documents based on lab markers.

        Reads from state:
        - labs: List of lab values with marker names

        Writes to state:
        - retrieved_docs: List of relevant documents
        - retrieval_latency_ms: Time taken for retrieval
        """
        start = time.time()

        # Extract markers from labs
        markers = [lab["marker"] for lab in state["labs"]]

        # Search vector store
        docs = store.search_by_markers(markers, limit=5)

        # Convert to dict format for state
        retrieved = [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "markers": doc.markers,
                "score": doc.score,
            }
            for doc in docs
        ]

        latency = (time.time() - start) * 1000

        return {
            "retrieved_docs": retrieved,
            "retrieval_latency_ms": latency,
        }

    return retrieve_context
