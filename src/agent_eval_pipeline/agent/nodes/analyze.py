"""
Analysis node - generates lab insights using LLM.

Dependencies are injectable for testing:
- LLM client can be mocked
- Prompt construction is a pure function

INTERVIEW TALKING POINT:
------------------------
"The analysis node separates prompt construction from LLM invocation.
The prompt builder is a pure function I can test without API calls.
The node factory accepts an optional model, so I can inject a mock
LLM for integration testing."
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Callable

from langchain_openai import ChatOpenAI

from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary

if TYPE_CHECKING:
    from agent_eval_pipeline.agent.state import AgentState


def build_analysis_prompt(state: AgentState) -> str:
    """
    Build the analysis prompt from state.

    This is a PURE FUNCTION - same inputs always produce same output.
    Can be tested independently without LLM calls.

    Args:
        state: Current agent state

    Returns:
        Formatted prompt string for the LLM
    """
    # Build context from retrieved docs
    context_text = "\n\n".join(
        [
            f"**{doc['title']}**\n{doc['content']}"
            for doc in state["retrieved_docs"]
        ]
    )

    # Build lab values text
    labs_text = "\n".join(
        [
            f"- {lab['marker']}: {lab['value']} {lab['unit']} "
            f"(ref: {lab.get('ref_low', '?')}-{lab.get('ref_high', '?')})"
            for lab in state["labs"]
        ]
    )

    # Build history text
    history_text = (
        "\n".join(
            [
                f"- {h['marker']}: {h['value']} {h['unit']} [{h['date']}]"
                for h in state["history"]
            ]
        )
        if state["history"]
        else "No historical data"
    )

    # Build symptoms text
    symptoms_text = (
        ", ".join(state["symptoms"]) if state["symptoms"] else "None reported"
    )

    return f"""Analyze these lab results and provide insights.

MEMBER QUERY: {state["query"]}

CURRENT LAB VALUES:
{labs_text}

HISTORICAL VALUES:
{history_text}

SYMPTOMS: {symptoms_text}

REFERENCE INFORMATION:
{context_text}

Provide a structured analysis with:
1. Summary (2-3 sentences overview)
2. Key insights for each relevant marker (status, trend, clinical relevance)
3. Topics to discuss with doctor
4. Lifestyle considerations
5. Safety notes (always include non-diagnostic disclaimer)

IMPORTANT:
- Do NOT diagnose conditions
- DO recommend discussing with healthcare provider
- Focus on education and empowerment"""


def create_analyze_node(
    model: ChatOpenAI | None = None,
) -> Callable[[AgentState], dict]:
    """
    Factory that creates an analysis node with injectable LLM.

    Args:
        model: Optional ChatOpenAI instance. If None, creates default.

    Returns:
        A node function compatible with LangGraph
    """

    def analyze_labs(state: AgentState) -> dict:
        """
        Generate lab analysis using the LLM with retrieved context.

        Reads from state:
        - query, labs, history, symptoms: Input data
        - retrieved_docs: Context from retrieval node

        Writes to state:
        - raw_analysis: Parsed output from LLM
        - analysis_latency_ms: Time taken
        - input_tokens, output_tokens: Token usage (estimated)
        """
        start = time.time()

        # Get or create LLM
        llm = model or ChatOpenAI(
            model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
            temperature=0.3,
        )

        # Use structured output
        structured_llm = llm.with_structured_output(LabInsightsSummary)

        # Build prompt
        prompt = build_analysis_prompt(state)

        # Invoke
        response = structured_llm.invoke(prompt)

        latency = (time.time() - start) * 1000

        # Estimate tokens (rough - would use tiktoken in production)
        input_tokens = int(len(prompt.split()) * 1.3)
        output_tokens = 500  # rough estimate for structured output

        return {
            "raw_analysis": response.model_dump() if response else None,
            "analysis_latency_ms": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    return analyze_labs
