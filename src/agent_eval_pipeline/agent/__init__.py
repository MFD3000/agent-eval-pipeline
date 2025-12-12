"""
Agent module - provides LangGraph and DSPy ReAct agent implementations.

ELEVATED ARCHITECTURE:
----------------------
The agent module follows hexagonal architecture with:
- State definition in state.py
- Individual nodes in nodes/ package
- Graph construction in graph.py
- Public API in langgraph_runner.py

AVAILABLE AGENTS:
-----------------
1. LangGraph Agent (default): RAG-enabled state machine with retrieval
2. DSPy ReAct Agent: Tool-using agent with explicit reasoning traces

The unified interface (run_agent) selects implementation via AGENT_TYPE env var.

INTERVIEW TALKING POINT:
------------------------
"The agent module provides a unified interface that abstracts the implementation.
LangGraph gives us a state machine with RAG retrieval. DSPy ReAct gives us
tool-using capabilities with explicit reasoning. Both produce the same
LabInsightsSummary output, enabling fair comparison in evals."
"""

import os
import time
from dataclasses import dataclass
from typing import Protocol, Literal

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
    agent_type: str  # "langgraph" or "dspy_react"
    # LangGraph-specific (optional)
    retrieved_docs: list[dict] | None = None
    retrieval_latency_ms: float | None = None
    # DSPy ReAct-specific (optional)
    tools_used: list[str] | None = None
    reasoning_steps: int | None = None


@dataclass
class AgentError:
    """Unified error from any agent implementation."""

    error_type: str
    error_message: str


# ---------------------------------------------------------------------------
# AGENT INTERFACE
# ---------------------------------------------------------------------------


AgentType = Literal["langgraph", "dspy_react"]


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
    agent_type: AgentType | None = None,
) -> AgentResult | AgentError:
    """
    Run the agent on a golden case.

    This is the main entry point for evaluation and general use.
    It provides a unified interface regardless of implementation.

    Args:
        case: The golden case to run
        agent_type: "langgraph" or "dspy_react".
                    If None, uses AGENT_TYPE env var (default: langgraph)

    Returns:
        AgentResult on success, AgentError on failure
    """
    if agent_type is None:
        agent_type = os.environ.get("AGENT_TYPE", "langgraph").lower()

    # Import tracer and attributes for observability
    from agent_eval_pipeline.observability.tracer import get_tracer
    from agent_eval_pipeline.observability.attributes import (
        AGENT_TYPE,
        EVAL_CASE_ID,
        GEN_AI_REQUEST_MODEL,
    )

    tracer = get_tracer()

    # Create parent span with agent type clearly visible
    span_name = f"agent.run.{agent_type}"
    with tracer.start_span(span_name, attributes={
        AGENT_TYPE: agent_type,
        EVAL_CASE_ID: case.id,
        "agent.case_description": case.description,
    }) as span:
        if agent_type == "langgraph":
            result = _run_langgraph_agent(case)
        elif agent_type == "dspy_react":
            result = _run_dspy_react_agent(case)
        else:
            return AgentError(
                error_type="InvalidAgentType",
                error_message=f"Unknown agent type: {agent_type}. Use 'langgraph' or 'dspy_react'",
            )

        # Add result attributes to span
        if isinstance(result, AgentResult):
            span.set_attribute(GEN_AI_REQUEST_MODEL, result.model)
            span.set_attribute("agent.latency_ms", result.latency_ms)
            span.set_attribute("agent.total_tokens", result.total_tokens)
            span.set_attribute("agent.success", True)
            if result.retrieved_docs:
                span.set_attribute("agent.retrieved_doc_count", len(result.retrieved_docs))
            if result.tools_used:
                span.set_attribute("agent.tools_used", ",".join(result.tools_used))
            if result.reasoning_steps:
                span.set_attribute("agent.reasoning_steps", result.reasoning_steps)
            span.set_status("ok")
        else:
            span.set_attribute("agent.success", False)
            span.set_attribute("agent.error_type", result.error_type)
            span.set_attribute("agent.error_message", result.error_message)
            span.set_status("error", result.error_message)

        return result


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
        agent_type="langgraph",
        retrieved_docs=result.retrieved_docs,
        retrieval_latency_ms=result.retrieval_latency_ms,
    )


def _run_dspy_react_agent(case: GoldenCase) -> AgentResult | AgentError:
    """Run the DSPy ReAct agent with tool use."""
    from agent_eval_pipeline.agent.dspy_react_agent import run_react_agent_for_eval

    start = time.time()
    try:
        result = run_react_agent_for_eval(case)
        latency_ms = (time.time() - start) * 1000

        return AgentResult(
            output=result.output,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
            model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
            agent_type="dspy_react",
            tools_used=result.tools_used,
            reasoning_steps=result.reasoning_steps,
        )
    except Exception as e:
        return AgentError(
            error_type=type(e).__name__,
            error_message=str(e),
        )


__all__ = [
    # Unified interface
    "AgentResult",
    "AgentError",
    "Agent",
    "AgentType",
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
