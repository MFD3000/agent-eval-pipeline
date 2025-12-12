"""
DeepEval Adapters - Convert between project schemas and DeepEval test cases.

This module bridges our domain model (GoldenCase, LabInsightsSummary) with
DeepEval's generic LLMTestCase structure.

ADAPTER PATTERN:
----------------
Adapters allow us to use DeepEval's metrics without coupling our domain
models to their framework. If DeepEval changes its API, we only update
this file.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden

if TYPE_CHECKING:
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase, LabValue
    from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
    from agent_eval_pipeline.agent import AgentResult


# ---------------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------------


def format_labs_for_input(labs: list[LabValue]) -> str:
    """Format lab values as readable text for LLM input."""
    lines = []
    for lab in labs:
        ref_range = ""
        if lab.ref_low is not None and lab.ref_high is not None:
            ref_range = f" (ref: {lab.ref_low}-{lab.ref_high})"
        lines.append(f"- {lab.marker}: {lab.value} {lab.unit}{ref_range}")
    return "\n".join(lines)


def format_history_for_input(history: list[LabValue]) -> str:
    """Format historical lab values."""
    if not history:
        return "No historical data available"

    lines = []
    for lab in history:
        lines.append(f"- {lab.date}: {lab.marker} = {lab.value} {lab.unit}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CORE ADAPTERS
# ---------------------------------------------------------------------------


def golden_case_to_llm_test_case(
    case: GoldenCase,
    agent_output: LabInsightsSummary,
    retrieval_context: list[str] | None = None,
) -> LLMTestCase:
    """
    Convert a GoldenCase + agent output to DeepEval LLMTestCase.

    This is the primary adapter for running DeepEval metrics on our
    golden cases after the agent has produced output.

    Args:
        case: The golden case with input and evaluation criteria
        agent_output: The agent's structured output
        retrieval_context: Documents retrieved by RAG (if any)

    Returns:
        LLMTestCase ready for DeepEval metrics

    Example:
        >>> case = get_case_by_id("thyroid-001")
        >>> result = run_agent(case)
        >>> test_case = golden_case_to_llm_test_case(case, result.output)
        >>> assert_test(test_case, [safety_compliance])
    """
    # Build comprehensive input text
    labs_text = format_labs_for_input(case.labs)
    history_text = format_history_for_input(case.history)
    symptoms_text = ", ".join(case.symptoms) if case.symptoms else "None reported"

    input_text = f"""User Query: {case.query}

Lab Results:
{labs_text}

Historical Values:
{history_text}

Reported Symptoms: {symptoms_text}
"""

    # Expected output is the semantic points that should be covered
    expected_output = "The response should cover:\n" + "\n".join([
        f"- {point}" for point in case.expected_semantic_points
    ])

    # Build actual output from agent response
    actual_output = agent_output.summary

    # Add key insights if available
    if agent_output.key_insights:
        insights_text = "\n".join([
            f"- {insight.marker}: {insight.status} - {insight.clinical_relevance}"
            for insight in agent_output.key_insights
        ])
        actual_output += f"\n\nKey Insights:\n{insights_text}"

    # Add doctor topics if available
    if agent_output.recommended_topics_for_doctor:
        actual_output += f"\n\nDiscuss with Doctor: {', '.join(agent_output.recommended_topics_for_doctor)}"

    return LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context or [],
        context=[labs_text],  # Original context for hallucination checking
    )


def agent_result_to_test_case(
    case: GoldenCase,
    result: AgentResult,
) -> LLMTestCase | None:
    """
    Convert an AgentResult to LLMTestCase.

    Handles both success and error cases gracefully.

    Args:
        case: The golden case
        result: The agent result (may be AgentResult or AgentError)

    Returns:
        LLMTestCase if successful, None if agent failed
    """
    from agent_eval_pipeline.agent import AgentError

    if isinstance(result, AgentError):
        return None

    # Extract retrieved docs if available (retrieved_docs is list[dict])
    retrieved_docs = []
    if hasattr(result, 'retrieved_docs') and result.retrieved_docs:
        retrieved_docs = [doc["content"] for doc in result.retrieved_docs]

    return golden_case_to_llm_test_case(
        case=case,
        agent_output=result.output,
        retrieval_context=retrieved_docs,
    )


def golden_cases_to_dataset(
    cases: list[GoldenCase],
) -> EvaluationDataset:
    """
    Convert golden cases to DeepEval EvaluationDataset.

    This creates a dataset of "Goldens" (expected inputs/outputs) that
    can be used with DeepEval's dataset-based evaluation.

    Args:
        cases: List of golden cases

    Returns:
        EvaluationDataset with Golden entries

    Example:
        >>> cases = get_all_golden_cases()
        >>> dataset = golden_cases_to_dataset(cases)
        >>> for golden in dataset.goldens:
        ...     test_case = LLMTestCase(
        ...         input=golden.input,
        ...         actual_output=my_agent(golden.input)
        ...     )
    """
    goldens = []

    for case in cases:
        # Format input
        labs_text = format_labs_for_input(case.labs)
        symptoms_text = ", ".join(case.symptoms) if case.symptoms else "None"

        input_text = f"{case.query}\n\nLabs:\n{labs_text}\nSymptoms: {symptoms_text}"

        # Expected output is semantic points
        expected_output = "\n".join(case.expected_semantic_points)

        goldens.append(
            Golden(
                input=input_text,
                expected_output=expected_output,
            )
        )

    return EvaluationDataset(goldens=goldens)


# ---------------------------------------------------------------------------
# BATCH CONVERSION
# ---------------------------------------------------------------------------


def convert_eval_results_to_test_cases(
    cases: list[GoldenCase],
    results: list[AgentResult],
) -> list[LLMTestCase]:
    """
    Batch convert agent results to test cases.

    Filters out failed agent runs automatically.

    Args:
        cases: Golden cases (parallel with results)
        results: Agent results for each case

    Returns:
        List of LLMTestCase (may be shorter than input if some failed)
    """
    test_cases = []

    for case, result in zip(cases, results):
        test_case = agent_result_to_test_case(case, result)
        if test_case is not None:
            test_cases.append(test_case)

    return test_cases
