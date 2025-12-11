"""
LangGraph agent runner - the public API for running the agent.

This module provides the entry points for running the LangGraph agent
on golden cases or raw inputs. It handles:
- State initialization
- Graph invocation
- Result conversion

INTERVIEW TALKING POINT:
------------------------
"The runner is a thin layer that converts between external interfaces
(GoldenCase, raw dicts) and the internal AgentState. It orchestrates
graph invocation but doesn't contain business logic. The actual
intelligence is in the nodes."
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
from agent_eval_pipeline.agent.state import AgentState
from agent_eval_pipeline.agent.graph import get_agent_graph

if TYPE_CHECKING:
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase
    from agent_eval_pipeline.core import VectorStore


# ---------------------------------------------------------------------------
# RESULT TYPES
# ---------------------------------------------------------------------------


@dataclass
class LangGraphAgentResult:
    """Result from running the LangGraph agent."""

    output: LabInsightsSummary
    retrieved_docs: list[dict]
    retrieval_latency_ms: float
    analysis_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass
class LangGraphAgentError:
    """Error from running the LangGraph agent."""

    error_type: str
    error_message: str


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def run_langgraph_agent(
    case: GoldenCase,
    store: VectorStore | None = None,
) -> LangGraphAgentResult | LangGraphAgentError:
    """
    Run the LangGraph agent on a golden case.

    This is the main entry point for evaluation.

    Args:
        case: GoldenCase to run
        store: Optional VectorStore (uses default if not provided)

    Returns:
        LangGraphAgentResult on success, LangGraphAgentError on failure
    """
    start = time.time()

    try:
        # Convert GoldenCase to state
        initial_state: AgentState = {
            "query": case.query,
            "labs": [
                {
                    "marker": lab.marker,
                    "value": lab.value,
                    "unit": lab.unit,
                    "ref_low": lab.ref_low,
                    "ref_high": lab.ref_high,
                    "date": lab.date,
                }
                for lab in case.labs
            ],
            "history": [
                {
                    "marker": h.marker,
                    "value": h.value,
                    "unit": h.unit,
                    "date": h.date,
                }
                for h in case.history
            ],
            "symptoms": case.symptoms,
            "messages": [],
            "retrieved_docs": [],
            "raw_analysis": None,
            "final_output": None,
            "retrieval_latency_ms": 0,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # Get or create graph
        graph = get_agent_graph(store=store)

        # Run the graph
        final_state = graph.invoke(initial_state)

        total_latency = (time.time() - start) * 1000

        if final_state["final_output"] is None:
            return LangGraphAgentError(
                error_type="NoOutput",
                error_message="Agent did not produce output",
            )

        return LangGraphAgentResult(
            output=final_state["final_output"],
            retrieved_docs=final_state["retrieved_docs"],
            retrieval_latency_ms=final_state["retrieval_latency_ms"],
            analysis_latency_ms=final_state["analysis_latency_ms"],
            total_latency_ms=total_latency,
            input_tokens=final_state["input_tokens"],
            output_tokens=final_state["output_tokens"],
        )

    except Exception as e:
        return LangGraphAgentError(
            error_type=type(e).__name__,
            error_message=str(e),
        )


def run_langgraph_agent_raw(
    query: str,
    labs: list[dict],
    history: list[dict] | None = None,
    symptoms: list[str] | None = None,
    store: VectorStore | None = None,
) -> LangGraphAgentResult | LangGraphAgentError:
    """
    Run the agent with raw inputs (for testing/demo).

    This is a convenience function that doesn't require a GoldenCase.

    Args:
        query: User's query about their labs
        labs: List of lab value dicts with marker, value, unit, etc.
        history: Optional historical lab values
        symptoms: Optional list of symptoms
        store: Optional VectorStore

    Returns:
        LangGraphAgentResult on success, LangGraphAgentError on failure
    """
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase, LabValue

    # Create temporary LabValue objects
    lab_values = [
        LabValue(
            date=lab.get("date", "2025-01-01"),
            marker=lab["marker"],
            value=lab["value"],
            unit=lab["unit"],
            ref_low=lab.get("ref_low"),
            ref_high=lab.get("ref_high"),
        )
        for lab in labs
    ]

    history_values = [
        LabValue(
            date=h["date"],
            marker=h["marker"],
            value=h["value"],
            unit=h["unit"],
        )
        for h in (history or [])
    ]

    # Create temporary GoldenCase
    temp_case = GoldenCase(
        id="manual-test",
        description="Manual test case",
        member_id="test",
        query=query,
        labs=lab_values,
        history=history_values,
        symptoms=symptoms or [],
    )

    return run_langgraph_agent(temp_case, store=store)
