"""
Agent module - provides both legacy and LangGraph-based agents.

ELEVATED ARCHITECTURE:
----------------------
The agent module follows hexagonal architecture with:
- State definition in state.py
- Individual nodes in nodes/ package
- Graph construction in graph.py
- Public API in langgraph_runner.py

The unified interface (run_agent) can use either implementation,
selected via environment variable or explicit parameter.

INTERVIEW TALKING POINT:
------------------------
"The agent module provides a unified interface that abstracts whether
we're using the LangGraph implementation or the legacy OpenAI agent.
The LangGraph version has injectable dependencies for testing - I can
swap in a mock vector store and mock LLM to test the entire flow
in under 100ms without any API calls."
"""

import os
from dataclasses import dataclass
from typing import Protocol

from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase

# Re-export from elevated modules for convenience
from agent_eval_pipeline.agent.state import AgentState, create_initial_state
from agent_eval_pipeline.agent.nodes import (
    create_retrieve_node,
    create_analyze_node,
    apply_safety,
)
from agent_eval_pipeline.agent.graph import (
    build_agent_graph,
    get_agent_graph,
    clear_graph_cache,
)
from agent_eval_pipeline.agent.langgraph_runner import (
    LangGraphAgentResult,
    LangGraphAgentError,
    run_langgraph_agent,
    run_langgraph_agent_raw,
)


# ---------------------------------------------------------------------------
# UNIFIED RESULT TYPES
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Unified result from any agent implementation."""

    output: LabInsightsSummary
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    # LangGraph-specific (optional)
    retrieved_docs: list[dict] | None = None
    retrieval_latency_ms: float | None = None


@dataclass
class AgentError:
    """Unified error from any agent implementation."""

    error_type: str
    error_message: str


# ---------------------------------------------------------------------------
# AGENT INTERFACE
# ---------------------------------------------------------------------------


class Agent(Protocol):
    """Protocol for agent implementations."""

    def run(self, case: GoldenCase) -> AgentResult | AgentError:
        """Run the agent on a golden case."""
        ...


# ---------------------------------------------------------------------------
# FACTORY FUNCTION
# ---------------------------------------------------------------------------


def run_agent(
    case: GoldenCase,
    use_langgraph: bool | None = None,
) -> AgentResult | AgentError:
    """
    Run the agent on a golden case.

    This is the main entry point for evaluation and general use.
    It provides a unified interface regardless of implementation.

    Args:
        case: The golden case to run
        use_langgraph: Force LangGraph (True) or legacy (False).
                       If None, uses AGENT_TYPE env var (default: langgraph)

    Returns:
        AgentResult on success, AgentError on failure
    """
    if use_langgraph is None:
        use_langgraph = (
            os.environ.get("AGENT_TYPE", "langgraph").lower() == "langgraph"
        )

    if use_langgraph:
        return _run_langgraph_agent(case)
    else:
        return _run_legacy_agent(case)


def _run_langgraph_agent(case: GoldenCase) -> AgentResult | AgentError:
    """Run the LangGraph-based agent."""
    result = run_langgraph_agent(case)

    if isinstance(result, LangGraphAgentError):
        return AgentError(
            error_type=result.error_type,
            error_message=result.error_message,
        )

    return AgentResult(
        output=result.output,
        latency_ms=result.total_latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.input_tokens + result.output_tokens,
        model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
        retrieved_docs=result.retrieved_docs,
        retrieval_latency_ms=result.retrieval_latency_ms,
    )


def _run_legacy_agent(case: GoldenCase) -> AgentResult | AgentError:
    """Run the legacy OpenAI-based agent."""
    from agent_eval_pipeline.agent.lab_insights_agent import (
        run_agent as legacy_run_agent,
        AgentResult as LegacyResult,
        AgentError as LegacyError,
    )

    result = legacy_run_agent(case)

    if isinstance(result, LegacyError):
        return AgentError(
            error_type=result.error_type,
            error_message=result.error_message,
        )

    return AgentResult(
        output=result.output,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        model=result.model,
    )


__all__ = [
    # Unified interface
    "AgentResult",
    "AgentError",
    "Agent",
    "run_agent",
    # State
    "AgentState",
    "create_initial_state",
    # Nodes
    "create_retrieve_node",
    "create_analyze_node",
    "apply_safety",
    # Graph
    "build_agent_graph",
    "get_agent_graph",
    "clear_graph_cache",
    # LangGraph runner
    "LangGraphAgentResult",
    "LangGraphAgentError",
    "run_langgraph_agent",
    "run_langgraph_agent_raw",
]
