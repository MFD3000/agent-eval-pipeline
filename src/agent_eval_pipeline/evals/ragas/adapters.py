"""
RAGAS Adapters - Convert between project schemas and RAGAS samples.

RAGAS uses SingleTurnSample for individual evaluations and Dataset for
batch evaluation. This module bridges our domain models to RAGAS format.

RAGAS SCHEMA:
-------------
SingleTurnSample:
  - user_input: The question/query
  - response: The LLM's response
  - retrieved_contexts: List of retrieved document texts
  - reference: Ground truth answer (optional for some metrics)

INTERVIEW TALKING POINT:
------------------------
"RAGAS needs retrieved_contexts to evaluate faithfulness - it checks if
claims in the response are grounded in the retrieved docs. Our adapter
extracts the document content from our RAG pipeline and formats it for
RAGAS. If we don't have retrieval context, some metrics won't work."
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragas import SingleTurnSample
from datasets import Dataset

if TYPE_CHECKING:
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase, LabValue
    from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
    from agent_eval_pipeline.agent import AgentResult


# ---------------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------------


def format_labs_text(labs: list[LabValue]) -> str:
    """Format lab values as text."""
    lines = []
    for lab in labs:
        ref = ""
        if lab.ref_low is not None and lab.ref_high is not None:
            ref = f" (reference: {lab.ref_low}-{lab.ref_high})"
        lines.append(f"{lab.marker}: {lab.value} {lab.unit}{ref}")
    return "\n".join(lines)


def format_user_input(case: GoldenCase) -> str:
    """Format the complete user input for RAGAS."""
    labs_text = format_labs_text(case.labs)
    symptoms = ", ".join(case.symptoms) if case.symptoms else "None"

    return f"""{case.query}

Lab Results:
{labs_text}

Symptoms: {symptoms}"""


def format_response(output: LabInsightsSummary) -> str:
    """Format agent output as response text."""
    parts = [output.summary]

    if output.key_insights:
        insights = "\n".join([
            f"- {i.marker}: {i.status} - {i.clinical_relevance}"
            for i in output.key_insights
        ])
        parts.append(f"\nKey Insights:\n{insights}")

    if output.recommended_topics_for_doctor:
        topics = ", ".join(output.recommended_topics_for_doctor)
        parts.append(f"\nDiscuss with your doctor: {topics}")

    if output.lifestyle_considerations:
        lifestyle = ", ".join(output.lifestyle_considerations)
        parts.append(f"\nLifestyle considerations: {lifestyle}")

    return "\n".join(parts)


def format_reference(case: GoldenCase) -> str:
    """Format expected semantic points as reference."""
    return "\n".join([f"- {point}" for point in case.expected_semantic_points])


# ---------------------------------------------------------------------------
# CORE ADAPTERS
# ---------------------------------------------------------------------------


def golden_case_to_ragas_sample(
    case: GoldenCase,
    agent_output: LabInsightsSummary,
    retrieved_contexts: list[str],
) -> SingleTurnSample:
    """
    Convert a GoldenCase + agent output to RAGAS SingleTurnSample.

    Args:
        case: The golden case with query and expected points
        agent_output: The agent's structured output
        retrieved_contexts: List of retrieved document texts

    Returns:
        SingleTurnSample ready for RAGAS evaluation

    Example:
        >>> case = get_case_by_id("thyroid-001")
        >>> result = run_agent(case)
        >>> sample = golden_case_to_ragas_sample(
        ...     case, result.output, ["doc1 content", "doc2 content"]
        ... )
        >>> score = faithfulness.single_turn_ascore(sample)
    """
    return SingleTurnSample(
        user_input=format_user_input(case),
        response=format_response(agent_output),
        retrieved_contexts=retrieved_contexts,
        reference=format_reference(case),
    )


def agent_result_to_ragas_sample(
    case: GoldenCase,
    result: AgentResult,
) -> SingleTurnSample | None:
    """
    Convert an AgentResult to RAGAS SingleTurnSample.

    Handles extraction of retrieved documents from the agent result.

    Args:
        case: The golden case
        result: The agent result (may be error)

    Returns:
        SingleTurnSample if successful, None if agent failed
    """
    from agent_eval_pipeline.agent import AgentError

    if isinstance(result, AgentError):
        return None

    # Extract retrieved document contents (retrieved_docs is list[dict])
    retrieved_contexts = []
    if hasattr(result, 'retrieved_docs') and result.retrieved_docs:
        retrieved_contexts = [doc["content"] for doc in result.retrieved_docs]

    # If no retrieval context, add the lab data as context
    # (RAGAS needs some context for faithfulness)
    if not retrieved_contexts:
        retrieved_contexts = [format_labs_text(case.labs)]

    return golden_case_to_ragas_sample(
        case=case,
        agent_output=result.output,
        retrieved_contexts=retrieved_contexts,
    )


# ---------------------------------------------------------------------------
# DATASET CREATION
# ---------------------------------------------------------------------------


def create_ragas_dataset(
    cases: list[GoldenCase],
    outputs: list[LabInsightsSummary],
    contexts: list[list[str]],
) -> Dataset:
    """
    Create a HuggingFace Dataset for RAGAS batch evaluation.

    RAGAS uses datasets library for efficient batch processing.

    Args:
        cases: Golden cases (parallel arrays)
        outputs: Agent outputs for each case
        contexts: Retrieved contexts for each case

    Returns:
        HuggingFace Dataset ready for RAGAS evaluate()

    Example:
        >>> cases = get_all_golden_cases()
        >>> outputs = [run_agent(c).output for c in cases]
        >>> contexts = [[doc.content for doc in result.retrieved_docs] for result in results]
        >>> dataset = create_ragas_dataset(cases, outputs, contexts)
        >>> results = evaluate(dataset=dataset, metrics=metrics)
    """
    rows = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    for case, output, ctx in zip(cases, outputs, contexts):
        rows["user_input"].append(format_user_input(case))
        rows["response"].append(format_response(output))
        rows["retrieved_contexts"].append(ctx if ctx else [format_labs_text(case.labs)])
        rows["reference"].append(format_reference(case))

    return Dataset.from_dict(rows)


def create_ragas_dataset_from_results(
    cases: list[GoldenCase],
    results: list[AgentResult],
) -> tuple[Dataset, list[str]]:
    """
    Create RAGAS dataset from agent results, handling failures.

    Args:
        cases: All golden cases
        results: Agent results (may include errors)

    Returns:
        Tuple of (Dataset, list of failed case IDs)
    """
    from agent_eval_pipeline.agent import AgentError

    valid_cases = []
    valid_outputs = []
    valid_contexts = []
    failed_ids = []

    for case, result in zip(cases, results):
        if isinstance(result, AgentError):
            failed_ids.append(case.id)
            continue

        valid_cases.append(case)
        valid_outputs.append(result.output)

        # Extract contexts (retrieved_docs is list[dict])
        contexts = []
        if hasattr(result, 'retrieved_docs') and result.retrieved_docs:
            contexts = [doc["content"] for doc in result.retrieved_docs]
        valid_contexts.append(contexts)

    dataset = create_ragas_dataset(valid_cases, valid_outputs, valid_contexts)

    return dataset, failed_ids
