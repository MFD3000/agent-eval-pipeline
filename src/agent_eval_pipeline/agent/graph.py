"""
Graph construction with dependency injection.

The graph is just WIRING - all logic lives in nodes.
This separation means:
- Nodes can be tested in isolation
- Graph structure can change without touching node logic
- Dependencies are explicit and injectable

STREAMING SUPPORT:
------------------
Pass streaming=True to enable token streaming for better perceived latency.
Use on_token callback to receive tokens as they arrive for real-time UI updates.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent_eval_pipeline.agent.state import AgentState
from agent_eval_pipeline.agent.nodes import (
    create_retrieve_node,
    create_analyze_node,
    apply_safety,
)

if TYPE_CHECKING:
    from agent_eval_pipeline.core import VectorStore


def build_agent_graph(
    store: VectorStore,
    model: ChatOpenAI | None = None,
    streaming: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> StateGraph:
    """
    Build the LangGraph agent workflow with injected dependencies.

    Graph structure:
    START -> retrieve_context -> analyze_labs -> apply_safety -> END

    Args:
        store: VectorStore implementation for retrieval
        model: Optional ChatOpenAI for analysis (creates default if None)
        streaming: If True, stream tokens for better perceived latency
        on_token: Callback for each token during streaming (for real-time UI)

    Returns:
        Compiled StateGraph ready for invocation

    Example:
        # Production
        store = get_vector_store(use_postgres=True)
        graph = build_agent_graph(store)

        # With streaming
        def print_token(t): print(t, end="", flush=True)
        graph = build_agent_graph(store, streaming=True, on_token=print_token)

        # Testing
        mock_store = InMemoryVectorStore(MockEmbeddings())
        mock_model = MockChatOpenAI()
        graph = build_agent_graph(mock_store, mock_model)
    """
    # Create graph with state schema
    workflow = StateGraph(AgentState)

    # Create nodes with injected dependencies
    retrieve_node = create_retrieve_node(store)
    analyze_node = create_analyze_node(model, streaming=streaming, on_token=on_token)

    # Add nodes to graph
    workflow.add_node("retrieve_context", retrieve_node)
    workflow.add_node("analyze_labs", analyze_node)
    workflow.add_node("apply_safety", apply_safety)  # Pure, no deps needed

    # Define edges (linear flow for now)
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "analyze_labs")
    workflow.add_edge("analyze_labs", "apply_safety")
    workflow.add_edge("apply_safety", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# GRAPH CACHING
# ---------------------------------------------------------------------------
# For production use, we cache the compiled graph to avoid rebuilding.
# The cache is keyed by store type to allow different configurations.

_graph_cache: dict[str, StateGraph] = {}


def get_agent_graph(
    store: VectorStore | None = None,
    model: ChatOpenAI | None = None,
    force_rebuild: bool = False,
) -> StateGraph:
    """
    Get or create a cached agent graph.

    For most use cases, this is the preferred way to get a graph.
    It handles store initialization and caching.

    CACHING BEHAVIOR:
    - If store is explicitly provided, caching is BYPASSED to respect DI
    - If store is None (default), caches by store type for performance
    - Use force_rebuild=True to rebuild even with default store

    Args:
        store: Optional VectorStore. If None, creates default.
        model: Optional ChatOpenAI. If None, creates default.
        force_rebuild: If True, rebuilds graph even if cached.

    Returns:
        Compiled StateGraph
    """
    # If a store is explicitly provided, bypass cache to respect dependency injection
    # This ensures tests with different fixtures get fresh graphs
    if store is not None:
        return build_agent_graph(store, model)

    # For default store case, use caching for performance
    cache_key = "default"

    # Return cached if available
    if cache_key in _graph_cache and not force_rebuild:
        return _graph_cache[cache_key]

    # Create default store
    import os
    from agent_eval_pipeline.retrieval import get_vector_store, seed_vector_store

    use_postgres = os.environ.get("USE_POSTGRES", "false").lower() == "true"
    store = get_vector_store(use_postgres=use_postgres)
    store.connect()

    # Seed on first use
    seed_vector_store(store)

    # Build and cache
    graph = build_agent_graph(store, model)
    _graph_cache[cache_key] = graph

    return graph


def clear_graph_cache() -> None:
    """Clear the graph cache. Useful for testing."""
    _graph_cache.clear()
