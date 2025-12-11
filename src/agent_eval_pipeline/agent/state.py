"""
Agent state definition - the data flowing through the LangGraph.

Separated because state schema changes for different reasons than
node logic or graph structure. Single Responsibility Principle.

INTERVIEW TALKING POINT:
------------------------
"The agent state is a TypedDict that flows through the LangGraph nodes.
By separating it into its own module, I can modify the state shape
independently of the node implementations. Each node reads from and
writes to specific state keys."
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary


class AgentState(TypedDict):
    """
    State that flows through the LangGraph.

    Each node can read from and write to this state.
    The graph orchestrates the flow between nodes.

    Input fields are set at invocation time.
    Intermediate fields are populated by nodes.
    Output fields contain the final result.
    Metrics fields enable eval tracking.
    """

    # -------------------------------------------------------------------------
    # INPUT (set at invocation)
    # -------------------------------------------------------------------------
    query: str
    labs: list[dict]
    history: list[dict]
    symptoms: list[str]

    # -------------------------------------------------------------------------
    # LLM CONTEXT (managed by LangGraph)
    # -------------------------------------------------------------------------
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # -------------------------------------------------------------------------
    # INTERMEDIATE (populated by nodes)
    # -------------------------------------------------------------------------
    retrieved_docs: list[dict]
    raw_analysis: dict | None

    # -------------------------------------------------------------------------
    # OUTPUT (final result)
    # -------------------------------------------------------------------------
    final_output: LabInsightsSummary | None

    # -------------------------------------------------------------------------
    # METRICS (for eval tracking)
    # -------------------------------------------------------------------------
    retrieval_latency_ms: float
    analysis_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int


def create_initial_state(
    query: str,
    labs: list[dict],
    history: list[dict] | None = None,
    symptoms: list[str] | None = None,
) -> AgentState:
    """
    Create an initial agent state for graph invocation.

    This is a convenience function that sets up all required fields
    with sensible defaults.
    """
    return AgentState(
        query=query,
        labs=labs,
        history=history or [],
        symptoms=symptoms or [],
        messages=[],
        retrieved_docs=[],
        raw_analysis=None,
        final_output=None,
        retrieval_latency_ms=0,
        analysis_latency_ms=0,
        total_latency_ms=0,
        input_tokens=0,
        output_tokens=0,
    )
