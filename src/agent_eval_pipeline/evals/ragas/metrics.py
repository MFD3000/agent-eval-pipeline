"""
RAGAS Metrics Configuration

RAGAS provides specialized metrics for RAG evaluation:
- Faithfulness: Are claims in the response supported by retrieved context?
- Context Precision: Are the retrieved documents relevant to the query?
- Context Recall: Does the context contain all info needed for the answer?
- Answer Relevancy: Does the response actually answer the question?

METRIC CATEGORIES:
------------------
1. Generation metrics (need response + context):
   - Faithfulness
   - Answer Relevancy

2. Retrieval metrics (need context + reference):
   - Context Precision
   - Context Recall

3. End-to-end metrics (need all fields):
   - Factual Correctness

INTERVIEW TALKING POINT:
------------------------
"RAGAS faithfulness works by extracting claims from the response, then
checking if each claim is supported by the retrieved context. It's more
sophisticated than simple similarity - it actually reasons about whether
the information could be derived from the source documents. This catches
subtle hallucinations that simpler metrics miss."
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    LLMContextPrecisionWithoutReference,
    AnswerRelevancy,
    FactualCorrectness,
)
from ragas.llms import LangchainLLMWrapper

if TYPE_CHECKING:
    from ragas.metrics.base import Metric


# ---------------------------------------------------------------------------
# METRIC NAMES
# ---------------------------------------------------------------------------

RAGAS_METRIC_NAMES = [
    "faithfulness",
    "context_recall",
    "context_precision",
    "answer_relevancy",
    "factual_correctness",
]


# ---------------------------------------------------------------------------
# LLM CONFIGURATION
# ---------------------------------------------------------------------------


def get_evaluator_llm(model: str | None = None) -> LangchainLLMWrapper:
    """
    Get configured LLM wrapper for RAGAS evaluation.

    RAGAS uses LangChain LLMs under the hood.

    Args:
        model: Model name. Defaults to JUDGE_MODEL env var or gpt-4o-mini.

    Returns:
        LangchainLLMWrapper for RAGAS metrics
    """
    from langchain_openai import ChatOpenAI

    model = model or os.environ.get("JUDGE_MODEL", "gpt-4o-mini")

    llm = ChatOpenAI(
        model=model,
        temperature=0,  # Deterministic for evaluation
    )

    return LangchainLLMWrapper(llm)


# ---------------------------------------------------------------------------
# INDIVIDUAL METRICS
# ---------------------------------------------------------------------------


def get_faithfulness_metric(llm: LangchainLLMWrapper | None = None) -> Faithfulness:
    """
    Get Faithfulness metric.

    Faithfulness measures whether the response is grounded in the retrieved
    context. It extracts claims from the response and verifies each against
    the context.

    Higher score = response sticks to what's in the context
    Lower score = response contains hallucinations or unsupported claims

    Returns:
        Configured Faithfulness metric
    """
    llm = llm or get_evaluator_llm()
    return Faithfulness(llm=llm)


def get_context_precision_metric(
    llm: LangchainLLMWrapper | None = None,
    with_reference: bool = True,
) -> LLMContextPrecisionWithReference | LLMContextPrecisionWithoutReference:
    """
    Get Context Precision metric.

    Context Precision measures whether retrieved documents are relevant
    to the query. High precision = retrieved docs are all useful.

    Args:
        llm: Evaluator LLM
        with_reference: Use reference-based (more accurate) or reference-free

    Returns:
        Configured Context Precision metric
    """
    llm = llm or get_evaluator_llm()

    if with_reference:
        return LLMContextPrecisionWithReference(llm=llm)
    else:
        return LLMContextPrecisionWithoutReference(llm=llm)


def get_context_recall_metric(
    llm: LangchainLLMWrapper | None = None,
) -> LLMContextRecall:
    """
    Get Context Recall metric.

    Context Recall measures whether the retrieved context contains all
    the information needed to answer the question correctly.

    High recall = context has everything needed
    Low recall = missing important information

    Returns:
        Configured Context Recall metric
    """
    llm = llm or get_evaluator_llm()
    return LLMContextRecall(llm=llm)


def get_answer_relevancy_metric(
    llm: LangchainLLMWrapper | None = None,
) -> AnswerRelevancy:
    """
    Get Answer Relevancy metric.

    Answer Relevancy measures how well the response addresses the
    original question. It doesn't check correctness, just relevance.

    Returns:
        Configured Answer Relevancy metric
    """
    llm = llm or get_evaluator_llm()
    return AnswerRelevancy(llm=llm)


def get_factual_correctness_metric(
    llm: LangchainLLMWrapper | None = None,
) -> FactualCorrectness:
    """
    Get Factual Correctness metric.

    Factual Correctness compares the response against a reference answer
    to check if the facts are correct.

    Returns:
        Configured Factual Correctness metric
    """
    llm = llm or get_evaluator_llm()
    return FactualCorrectness(llm=llm)


# ---------------------------------------------------------------------------
# METRIC COLLECTIONS
# ---------------------------------------------------------------------------


def get_context_metrics(
    llm: LangchainLLMWrapper | None = None,
) -> list[Metric]:
    """
    Get all context/retrieval-related metrics.

    These metrics evaluate the quality of the retrieval step.

    Returns:
        List of [ContextPrecision, ContextRecall]
    """
    llm = llm or get_evaluator_llm()

    return [
        get_context_precision_metric(llm),
        get_context_recall_metric(llm),
    ]


def get_generation_metrics(
    llm: LangchainLLMWrapper | None = None,
) -> list[Metric]:
    """
    Get all generation-related metrics.

    These metrics evaluate the quality of the generated response.

    Returns:
        List of [Faithfulness, AnswerRelevancy]
    """
    llm = llm or get_evaluator_llm()

    return [
        get_faithfulness_metric(llm),
        get_answer_relevancy_metric(llm),
    ]


def get_ragas_metrics(
    llm: LangchainLLMWrapper | None = None,
    include_factual: bool = True,
) -> list[Metric]:
    """
    Get all RAGAS metrics.

    Args:
        llm: Evaluator LLM (created if not provided)
        include_factual: Include FactualCorrectness (requires reference)

    Returns:
        List of all configured RAGAS metrics
    """
    llm = llm or get_evaluator_llm()

    metrics = [
        get_faithfulness_metric(llm),
        get_context_precision_metric(llm),
        get_context_recall_metric(llm),
        get_answer_relevancy_metric(llm),
    ]

    if include_factual:
        metrics.append(get_factual_correctness_metric(llm))

    return metrics


# ---------------------------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------------------------

# Default thresholds for pass/fail determination
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.8,  # High - we don't want hallucinations
    "context_precision": 0.6,
    "context_recall": 0.7,
    "answer_relevancy": 0.7,
    "factual_correctness": 0.7,
}


def check_thresholds(
    scores: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """
    Check if scores meet thresholds.

    Args:
        scores: Metric scores from RAGAS
        thresholds: Custom thresholds (uses defaults if not provided)

    Returns:
        Tuple of (all_passed, list of failed metric names)
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    failed = []
    for metric_name, threshold in thresholds.items():
        if metric_name in scores:
            if scores[metric_name] < threshold:
                failed.append(metric_name)

    return len(failed) == 0, failed
