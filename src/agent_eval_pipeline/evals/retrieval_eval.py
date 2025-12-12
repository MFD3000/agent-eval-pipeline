"""
Phase 5: Retrieval Quality Eval

This evaluates whether the agent retrieves the RIGHT documents/context
before generating its response.

WHY RETRIEVAL QUALITY MATTERS:
------------------------------
In RAG systems, garbage in = garbage out.
If you retrieve the wrong documents, even a perfect LLM will produce
wrong or hallucinated answers.

Retrieval eval catches:
- Changed embedding models breaking similarity
- Modified chunking strategies missing relevant content
- Filter/metadata changes excluding correct documents
- Query rewriting regressions

METRICS EXPLAINED:
------------------
RECALL: What fraction of expected docs did we retrieve?
  - High recall = we're not missing important context
  - Formula: |retrieved ∩ expected| / |expected|

PRECISION: What fraction of retrieved docs were expected?
  - High precision = we're not polluting context with noise
  - Formula: |retrieved ∩ expected| / |retrieved|

F1 SCORE: Harmonic mean of recall and precision
  - Balances both concerns
  - Formula: 2 * (precision * recall) / (precision + recall)

PRACTICAL NOTE:
---------------
In production, you'd hook this into your actual retrieval system.
Here we simulate retrieval to demonstrate the evaluation pattern.
The important thing is the STRUCTURE of how you evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING

from agent_eval_pipeline.golden_sets import GoldenCase, get_all_golden_cases

if TYPE_CHECKING:
    from agent_eval_pipeline.harness.context import AgentRunContext


# ---------------------------------------------------------------------------
# SIMULATED RETRIEVAL SYSTEM
# ---------------------------------------------------------------------------
# In production, this would call your actual vector store / search system.
# We simulate it here to demonstrate the evaluation pattern.

# Simulated document corpus
DOCUMENT_CORPUS = {
    "doc_thyroid_101": {
        "title": "Understanding Thyroid Function Tests",
        "content": "TSH, Free T4, and Free T3 are key thyroid markers...",
        "markers": ["TSH", "Free T4", "Free T3"],
    },
    "doc_tsh_interpretation": {
        "title": "Interpreting TSH Results",
        "content": "TSH above 4.0 mIU/L may indicate hypothyroidism...",
        "markers": ["TSH"],
    },
    "doc_hyperthyroid_patterns": {
        "title": "Hyperthyroidism: Lab Patterns",
        "content": "Low TSH with elevated Free T4 suggests hyperthyroidism...",
        "markers": ["TSH", "Free T4"],
    },
    "doc_subclinical_thyroid": {
        "title": "Subclinical Thyroid Disorders",
        "content": "Borderline TSH values require careful interpretation...",
        "markers": ["TSH"],
    },
    "doc_vitamin_d": {
        "title": "Vitamin D and Health",
        "content": "Vitamin D deficiency is common and can cause fatigue...",
        "markers": ["Vitamin D"],
    },
    # Noise documents (should not be retrieved for thyroid queries)
    "doc_cholesterol_basics": {
        "title": "Understanding Cholesterol",
        "content": "LDL, HDL, and triglycerides are key lipid markers...",
        "markers": ["LDL", "HDL", "Triglycerides"],
    },
    "doc_blood_sugar": {
        "title": "Blood Glucose Testing",
        "content": "HbA1c and fasting glucose monitor diabetes...",
        "markers": ["HbA1c", "Glucose"],
    },
}


def simulate_retrieval(
    markers: list[str],
    noise_probability: float = 0.1,
    rng: random.Random | None = None,
) -> list[str]:
    """
    Simulate document retrieval based on markers.

    In production, this would be:
    - Embedding the query
    - Vector similarity search
    - Metadata filtering
    - Reranking

    Args:
        markers: Lab markers to retrieve documents for
        noise_probability: Chance of including irrelevant docs (simulates imperfect retrieval)
        rng: Random number generator for reproducibility. If None, uses global random.

    Returns:
        List of document IDs
    """
    # Use provided RNG or fall back to global (for backwards compatibility)
    rand = rng.random if rng else random.random

    retrieved = []

    for doc_id, doc in DOCUMENT_CORPUS.items():
        # Check if any marker matches
        doc_markers = set(doc["markers"])
        query_markers = set(markers)

        if doc_markers & query_markers:
            retrieved.append(doc_id)
        elif rand() < noise_probability:
            # Simulate occasional noise
            retrieved.append(doc_id)

    return retrieved


# ---------------------------------------------------------------------------
# RETRIEVAL EVALUATION
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics for a single case."""
    recall: float
    precision: float
    f1_score: float
    retrieved_docs: list[str]
    expected_docs: list[str]
    missing_docs: list[str]
    extra_docs: list[str]


@dataclass
class RetrievalEvalResult:
    """Result of retrieval eval for a single case."""
    case_id: str
    passed: bool
    metrics: RetrievalMetrics


@dataclass
class RetrievalEvalReport:
    """Aggregate retrieval eval results."""
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_recall: float
    avg_precision: float
    avg_f1: float
    threshold: float
    results: list[RetrievalEvalResult]

    @property
    def all_passed(self) -> bool:
        return self.failed_cases == 0


def calculate_retrieval_metrics(
    retrieved: list[str],
    expected: list[str],
) -> RetrievalMetrics:
    """
    Calculate retrieval quality metrics.

    This is the core of retrieval evaluation.
    You'll use these same formulas in production.
    """
    retrieved_set = set(retrieved)
    expected_set = set(expected)

    # Handle edge cases
    if not expected_set:
        # No expected docs - can't calculate meaningful metrics
        return RetrievalMetrics(
            recall=1.0,  # Vacuously true
            precision=1.0 if not retrieved_set else 0.0,
            f1_score=1.0 if not retrieved_set else 0.0,
            retrieved_docs=retrieved,
            expected_docs=expected,
            missing_docs=[],
            extra_docs=list(retrieved_set),
        )

    # Calculate overlap
    overlap = retrieved_set & expected_set
    missing = expected_set - retrieved_set
    extra = retrieved_set - expected_set

    # Recall: what fraction of expected did we get?
    recall = len(overlap) / len(expected_set)

    # Precision: what fraction of retrieved was expected?
    precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0

    # F1: harmonic mean
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return RetrievalMetrics(
        recall=recall,
        precision=precision,
        f1_score=f1,
        retrieved_docs=retrieved,
        expected_docs=expected,
        missing_docs=list(missing),
        extra_docs=list(extra),
    )


# Default seed for reproducible CI runs
DEFAULT_RETRIEVAL_SEED = 42


def run_retrieval_eval(
    cases: list[GoldenCase] | None = None,
    contexts: list[AgentRunContext] | None = None,
    use_actual_retrieval: bool = False,
    threshold: float = 0.8,
    verbose: bool = False,
    seed: int | None = DEFAULT_RETRIEVAL_SEED,
) -> RetrievalEvalReport:
    """
    Run retrieval eval on golden cases.

    Args:
        cases: Cases to evaluate. Defaults to all golden cases.
        contexts: Pre-computed agent results (from harness). If provided,
                  enables use_actual_retrieval mode.
        use_actual_retrieval: If True and contexts provided, validate against
                              actual retrieved_docs from agent runs instead of
                              simulated retrieval.
        threshold: Minimum F1 score to pass. Default 0.8.
        verbose: Print progress.
        seed: Random seed for reproducible results. Default 42.
              Set to None for non-deterministic behavior.

    Returns:
        RetrievalEvalReport with metrics for each case.

    CONTEXT SHARING:
    ----------------
    When use_actual_retrieval=True with contexts, this evaluator validates
    against REAL documents retrieved during agent execution, not simulated
    retrieval. This catches actual RAG pipeline issues.
    """
    # If using actual retrieval from contexts
    if use_actual_retrieval and contexts is not None:
        return _run_with_actual_retrieval(contexts, threshold, verbose)

    # Legacy path: simulated retrieval
    cases = cases or get_all_golden_cases()
    results: list[RetrievalEvalResult] = []

    # Create seeded RNG for reproducibility
    rng = random.Random(seed) if seed is not None else None

    for case in cases:
        if verbose:
            print(f"Running retrieval eval: {case.id}...")

        # Skip cases without retrieval expectations
        if not case.expected_doc_ids:
            if verbose:
                print(f"  Skipping {case.id} - no expected docs defined")
            continue

        # Get markers from the case
        markers = [lab.marker for lab in case.labs]

        # Run retrieval (simulated) with seeded RNG
        retrieved = simulate_retrieval(markers, rng=rng)

        # Calculate metrics
        metrics = calculate_retrieval_metrics(retrieved, case.expected_doc_ids)

        # Determine pass/fail
        passed = metrics.f1_score >= threshold

        results.append(RetrievalEvalResult(
            case_id=case.id,
            passed=passed,
            metrics=metrics,
        ))

    # Calculate aggregates
    if results:
        avg_recall = sum(r.metrics.recall for r in results) / len(results)
        avg_precision = sum(r.metrics.precision for r in results) / len(results)
        avg_f1 = sum(r.metrics.f1_score for r in results) / len(results)
        passed = sum(1 for r in results if r.passed)
    else:
        avg_recall = avg_precision = avg_f1 = 0.0
        passed = 0

    return RetrievalEvalReport(
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=len(results) - passed,
        avg_recall=avg_recall,
        avg_precision=avg_precision,
        avg_f1=avg_f1,
        threshold=threshold,
        results=results,
    )


def _run_with_actual_retrieval(
    contexts: list[AgentRunContext],
    threshold: float,
    verbose: bool,
) -> RetrievalEvalReport:
    """
    Run retrieval eval using actual retrieved documents from agent runs.

    This validates against REAL documents retrieved during agent execution,
    catching actual RAG pipeline issues instead of using simulated retrieval.
    """
    results: list[RetrievalEvalResult] = []

    for ctx in contexts:
        if verbose:
            print(f"Running retrieval eval: {ctx.case_id}...")

        # Skip cases without retrieval expectations
        if not ctx.case.expected_doc_ids:
            if verbose:
                print(f"  Skipping {ctx.case_id} - no expected docs defined")
            continue

        # Skip failed agent runs
        if not ctx.success:
            if verbose:
                print(f"  Skipping {ctx.case_id} - agent failed")
            continue

        # Get actual retrieved document IDs from the agent run
        # The agent stores retrieved docs in result.retrieved_docs
        actual_doc_ids = []
        for doc in ctx.retrieved_docs:
            # Handle both dict format and potential other formats
            if isinstance(doc, dict):
                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                if doc_id:
                    actual_doc_ids.append(doc_id)

        # Calculate metrics against expected docs
        metrics = calculate_retrieval_metrics(actual_doc_ids, ctx.case.expected_doc_ids)

        # Determine pass/fail
        passed = metrics.f1_score >= threshold

        results.append(RetrievalEvalResult(
            case_id=ctx.case_id,
            passed=passed,
            metrics=metrics,
        ))

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] F1: {metrics.f1_score:.2f} "
                  f"(retrieved: {len(actual_doc_ids)}, expected: {len(ctx.case.expected_doc_ids)})")

    # Calculate aggregates
    if results:
        avg_recall = sum(r.metrics.recall for r in results) / len(results)
        avg_precision = sum(r.metrics.precision for r in results) / len(results)
        avg_f1 = sum(r.metrics.f1_score for r in results) / len(results)
        passed_count = sum(1 for r in results if r.passed)
    else:
        avg_recall = avg_precision = avg_f1 = 0.0
        passed_count = 0

    return RetrievalEvalReport(
        total_cases=len(results),
        passed_cases=passed_count,
        failed_cases=len(results) - passed_count,
        avg_recall=avg_recall,
        avg_precision=avg_precision,
        avg_f1=avg_f1,
        threshold=threshold,
        results=results,
    )


def run_retrieval_eval_cli():
    """CLI entry point for retrieval eval."""
    import argparse

    parser = argparse.ArgumentParser(description="Run retrieval quality eval")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RETRIEVAL_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_RETRIEVAL_SEED})",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Disable seeding for non-deterministic behavior",
    )
    args = parser.parse_args()

    seed = None if args.no_seed else args.seed

    print("=" * 60)
    print("RETRIEVAL QUALITY EVAL")
    print("=" * 60)
    if seed is not None:
        print(f"Using seed: {seed}")

    report = run_retrieval_eval(verbose=True, threshold=0.8, seed=seed)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        m = result.metrics
        print(f"  [{status}] {result.case_id}")
        print(f"        Recall: {m.recall:.2f} | Precision: {m.precision:.2f} | F1: {m.f1_score:.2f}")
        if m.missing_docs:
            print(f"        Missing: {m.missing_docs}")
        if m.extra_docs:
            print(f"        Extra: {m.extra_docs}")

    print("\n" + "-" * 60)
    print(f"Averages: Recall={report.avg_recall:.2f} | "
          f"Precision={report.avg_precision:.2f} | "
          f"F1={report.avg_f1:.2f}")
    print(f"Threshold: {report.threshold} | "
          f"Passed: {report.passed_cases}/{report.total_cases}")

    if report.all_passed:
        print("\n>>> RETRIEVAL EVAL GATE: PASSED <<<")
        return 0
    else:
        print("\n>>> RETRIEVAL EVAL GATE: FAILED <<<")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = run_retrieval_eval_cli()
    sys.exit(exit_code)
