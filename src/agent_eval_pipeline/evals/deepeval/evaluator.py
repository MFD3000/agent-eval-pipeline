"""
DeepEval Evaluator - Main evaluation runner using DeepEval framework.

This module provides the entry point for running DeepEval-based evaluations
on golden cases. It orchestrates:
1. Running the agent on cases (or using pre-computed AgentRunContext)
2. Converting results to LLMTestCase format
3. Running DeepEval metrics
4. Aggregating results
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from agent_eval_pipeline.evals.deepeval.adapters import (
    golden_case_to_llm_test_case,
    agent_result_to_test_case,
)
from agent_eval_pipeline.evals.deepeval.metrics import (
    get_healthcare_metrics,
    get_rag_metrics,
)

if TYPE_CHECKING:
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase
    from agent_eval_pipeline.harness.context import AgentRunContext


# ---------------------------------------------------------------------------
# RESULT SCHEMAS
# ---------------------------------------------------------------------------


@dataclass
class DeepEvalMetricResult:
    """Result from a single metric evaluation."""

    name: str
    score: float
    passed: bool
    threshold: float
    reason: str | None = None


@dataclass
class DeepEvalResult:
    """Result from evaluating a single case."""

    case_id: str
    passed: bool
    metrics: list[DeepEvalMetricResult]
    overall_score: float

    @property
    def failed_metrics(self) -> list[str]:
        """Names of metrics that failed."""
        return [m.name for m in self.metrics if not m.passed]


@dataclass
class DeepEvalReport:
    """Aggregated report from evaluating multiple cases."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    results: list[DeepEvalResult]
    metric_averages: dict[str, float] = field(default_factory=dict)
    critical_failures: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Percentage of cases that passed all metrics."""
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases

    @property
    def all_passed(self) -> bool:
        """True if all cases passed."""
        return self.failed_cases == 0


# ---------------------------------------------------------------------------
# EVALUATION FUNCTIONS
# ---------------------------------------------------------------------------


def evaluate_single_case(
    case: GoldenCase,
    test_case: LLMTestCase,
    include_rag_metrics: bool = True,
    verbose: bool = False,
) -> DeepEvalResult:
    """
    Evaluate a single test case with all metrics.

    Args:
        case: Original golden case (for case_id)
        test_case: DeepEval LLMTestCase to evaluate
        include_rag_metrics: Include RAG-specific metrics
        verbose: Print progress

    Returns:
        DeepEvalResult with scores from all metrics
    """
    # Build metrics list
    metrics = get_healthcare_metrics()

    if include_rag_metrics and test_case.retrieval_context:
        metrics.extend(get_rag_metrics())

    # Evaluate each metric
    metric_results = []
    all_passed = True

    for metric in metrics:
        try:
            metric.measure(test_case)

            result = DeepEvalMetricResult(
                name=metric.name,
                score=metric.score,
                passed=metric.is_successful(),
                threshold=metric.threshold,
                reason=getattr(metric, 'reason', None),
            )
            metric_results.append(result)

            if not result.passed:
                all_passed = False

            if verbose:
                status = "PASS" if result.passed else "FAIL"
                print(f"  {metric.name}: {result.score:.2f} [{status}]")

        except Exception as e:
            if verbose:
                print(f"  {metric.name}: ERROR - {e}")
            metric_results.append(
                DeepEvalMetricResult(
                    name=metric.name,
                    score=0.0,
                    passed=False,
                    threshold=0.0,
                    reason=f"Error: {e}",
                )
            )
            all_passed = False

    # Calculate overall score
    valid_scores = [m.score for m in metric_results if m.score > 0]
    overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return DeepEvalResult(
        case_id=case.id,
        passed=all_passed,
        metrics=metric_results,
        overall_score=overall_score,
    )


def run_deepeval_evaluation(
    cases: list[GoldenCase] | None = None,
    contexts: list[AgentRunContext] | None = None,
    include_rag_metrics: bool = True,
    verbose: bool = False,
) -> DeepEvalReport:
    """
    Run DeepEval evaluation on golden cases.

    This is the main entry point for DeepEval-based evaluation.

    Args:
        cases: Cases to evaluate. Defaults to all golden cases.
        contexts: Pre-computed AgentRunContext list (avoids re-running agents).
                  If provided, cases parameter is ignored.
        include_rag_metrics: Include RAG metrics (requires retrieval context)
        verbose: Print progress

    Returns:
        DeepEvalReport with aggregated results

    Example:
        >>> report = run_deepeval_evaluation(verbose=True)
        >>> print(f"Pass rate: {report.pass_rate:.1%}")
        >>> for result in report.results:
        ...     if not result.passed:
        ...         print(f"{result.case_id}: Failed {result.failed_metrics}")
    """
    from agent_eval_pipeline.golden_sets import get_all_golden_cases
    from agent_eval_pipeline.agent import run_agent, AgentError

    results: list[DeepEvalResult] = []
    critical_failures: list[str] = []

    # If contexts provided, use them; otherwise run agents (backwards compat)
    if contexts is not None:
        cases_to_eval = [(ctx.case, ctx.result) for ctx in contexts]
        if verbose:
            print(f"\nRunning DeepEval evaluation on {len(contexts)} cases (using cached contexts)...")
            print("=" * 50)
    else:
        cases = cases or get_all_golden_cases()
        if verbose:
            print(f"\nRunning DeepEval evaluation on {len(cases)} cases...")
            print("=" * 50)
        # Build case/result pairs by running agents
        cases_to_eval = []
        for case in cases:
            agent_result = run_agent(case)
            cases_to_eval.append((case, agent_result))

    for case, agent_result in cases_to_eval:
        if verbose:
            print(f"\n{case.id}: {case.description}")

        if isinstance(agent_result, AgentError):
            if verbose:
                print(f"  Agent error: {agent_result.error_message}")
            results.append(
                DeepEvalResult(
                    case_id=case.id,
                    passed=False,
                    metrics=[],
                    overall_score=0.0,
                )
            )
            critical_failures.append(f"{case.id}: Agent failed")
            continue

        # Convert to test case
        test_case = agent_result_to_test_case(case, agent_result)

        if test_case is None:
            results.append(
                DeepEvalResult(
                    case_id=case.id,
                    passed=False,
                    metrics=[],
                    overall_score=0.0,
                )
            )
            continue

        # Evaluate
        result = evaluate_single_case(
            case=case,
            test_case=test_case,
            include_rag_metrics=include_rag_metrics,
            verbose=verbose,
        )
        results.append(result)

        # Track safety failures as critical
        for metric in result.metrics:
            if metric.name == "Safety Compliance" and not metric.passed:
                critical_failures.append(f"{case.id}: Safety compliance failed")

    # Calculate aggregates
    passed_cases = sum(1 for r in results if r.passed)
    failed_cases = len(results) - passed_cases

    # Calculate metric averages
    metric_scores: dict[str, list[float]] = {}
    for result in results:
        for metric in result.metrics:
            if metric.name not in metric_scores:
                metric_scores[metric.name] = []
            if metric.score > 0:
                metric_scores[metric.name].append(metric.score)

    metric_averages = {
        name: sum(scores) / len(scores)
        for name, scores in metric_scores.items()
        if scores
    }

    report = DeepEvalReport(
        total_cases=len(results),
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        results=results,
        metric_averages=metric_averages,
        critical_failures=critical_failures,
    )

    if verbose:
        print("\n" + "=" * 50)
        print("DEEPEVAL SUMMARY")
        print("=" * 50)
        print(f"Total: {report.total_cases}")
        print(f"Passed: {report.passed_cases}")
        print(f"Failed: {report.failed_cases}")
        print(f"Pass Rate: {report.pass_rate:.1%}")
        print("\nMetric Averages:")
        for name, avg in report.metric_averages.items():
            print(f"  {name}: {avg:.2f}")
        if report.critical_failures:
            print("\nCritical Failures:")
            for failure in report.critical_failures:
                print(f"  - {failure}")

    return report


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------


def run_deepeval_cli():
    """CLI entry point for DeepEval evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run DeepEval evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-rag", action="store_true", help="Skip RAG metrics")
    args = parser.parse_args()

    report = run_deepeval_evaluation(
        include_rag_metrics=not args.no_rag,
        verbose=args.verbose,
    )

    # Exit with error code if any failures
    exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    run_deepeval_cli()
