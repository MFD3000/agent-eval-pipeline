"""
RAGAS Evaluator - Main evaluation runner using RAGAS framework.

RAGAS is designed for batch evaluation using the evaluate() function.
This module orchestrates:
1. Running the agent on golden cases (or using pre-computed AgentRunContext)
2. Converting results to RAGAS Dataset format
3. Running RAGAS metrics in batch
4. Aggregating and reporting results
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ragas import evaluate

from agent_eval_pipeline.evals.ragas.adapters import (
    create_ragas_dataset_from_results,
)
from agent_eval_pipeline.evals.ragas.metrics import (
    get_ragas_metrics,
    get_evaluator_llm,
    check_thresholds,
    DEFAULT_THRESHOLDS,
)

if TYPE_CHECKING:
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase
    from agent_eval_pipeline.harness.context import AgentRunContext


# ---------------------------------------------------------------------------
# RESULT SCHEMAS
# ---------------------------------------------------------------------------


@dataclass
class RagasResult:
    """Result from evaluating a single case with RAGAS."""

    case_id: str
    scores: dict[str, float]
    passed: bool
    failed_metrics: list[str] = field(default_factory=list)


@dataclass
class RagasReport:
    """Aggregated report from RAGAS evaluation."""

    total_cases: int
    evaluated_cases: int  # May differ if some cases failed
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    metric_averages: dict[str, float]
    results: list[RagasResult]
    failed_case_ids: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Percentage of evaluated cases that passed."""
        if self.evaluated_cases == 0:
            return 0.0
        return self.passed_cases / self.evaluated_cases

    @property
    def all_passed(self) -> bool:
        """True if all evaluated cases passed."""
        return self.failed_cases == 0 and self.skipped_cases == 0


# ---------------------------------------------------------------------------
# EVALUATION FUNCTIONS
# ---------------------------------------------------------------------------


def run_ragas_evaluation(
    cases: list[GoldenCase] | None = None,
    contexts: list[AgentRunContext] | None = None,
    model: str | None = None,
    thresholds: dict[str, float] | None = None,
    verbose: bool = False,
) -> RagasReport:
    """
    Run RAGAS evaluation on golden cases.

    This is the main entry point for RAGAS-based evaluation.

    Args:
        cases: Cases to evaluate. Defaults to all golden cases.
        contexts: Pre-computed AgentRunContext list (avoids re-running agents).
                  If provided, cases parameter is ignored.
        model: LLM model for evaluation. Defaults to JUDGE_MODEL env var.
        thresholds: Pass/fail thresholds per metric. Uses defaults if not provided.
        verbose: Print progress and results.

    Returns:
        RagasReport with scores and aggregates.

    Example:
        >>> report = run_ragas_evaluation(verbose=True)
        >>> print(f"Faithfulness avg: {report.metric_averages['faithfulness']:.2f}")
        >>> for result in report.results:
        ...     if not result.passed:
        ...         print(f"{result.case_id} failed: {result.failed_metrics}")
    """
    from agent_eval_pipeline.golden_sets import get_all_golden_cases
    from agent_eval_pipeline.agent import run_agent, AgentError

    thresholds = thresholds or DEFAULT_THRESHOLDS

    # If contexts provided, extract cases and results from them
    if contexts is not None:
        cases = [ctx.case for ctx in contexts]
        agent_results = [ctx.result for ctx in contexts]
        case_ids = [ctx.case_id for ctx in contexts]

        if verbose:
            print(f"\nRunning RAGAS evaluation on {len(cases)} cases (using cached contexts)...")
            print("=" * 50)
    else:
        # Backwards compatibility: run agents if no contexts provided
        cases = cases or get_all_golden_cases()

        if verbose:
            print(f"\nRunning RAGAS evaluation on {len(cases)} cases...")
            print("=" * 50)
            print("Running agent on cases...")

        agent_results = []
        case_ids = []

        for case in cases:
            if verbose:
                print(f"  {case.id}...", end=" ")
            result = run_agent(case)
            agent_results.append(result)
            case_ids.append(case.id)
            if verbose:
                status = "FAIL" if isinstance(result, AgentError) else "OK"
                print(status)

    # Create dataset
    if verbose:
        print("\nCreating RAGAS dataset...")

    dataset, failed_ids = create_ragas_dataset_from_results(cases, agent_results)

    if verbose:
        print(f"  Valid samples: {len(dataset)}")
        print(f"  Failed/skipped: {len(failed_ids)}")

    # If no valid samples, return empty report
    if len(dataset) == 0:
        return RagasReport(
            total_cases=len(cases),
            evaluated_cases=0,
            passed_cases=0,
            failed_cases=0,
            skipped_cases=len(cases),
            metric_averages={},
            results=[],
            failed_case_ids=failed_ids,
        )

    # Configure evaluator
    if verbose:
        print("\nConfiguring RAGAS metrics...")

    evaluator_llm = get_evaluator_llm(model)
    metrics = get_ragas_metrics(llm=evaluator_llm, include_factual=True)

    if verbose:
        print(f"  Metrics: {[m.__class__.__name__ for m in metrics]}")

    # Run evaluation
    if verbose:
        print("\nRunning RAGAS evaluation (this may take a minute)...")

    try:
        ragas_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
        )
    except Exception as e:
        if verbose:
            print(f"RAGAS evaluation error: {e}")
        return RagasReport(
            total_cases=len(cases),
            evaluated_cases=0,
            passed_cases=0,
            failed_cases=0,
            skipped_cases=len(cases),
            metric_averages={},
            results=[],
            failed_case_ids=list(case_ids),
        )

    # Process results
    if verbose:
        print("\nProcessing results...")

    # Get valid case IDs (those not in failed_ids)
    valid_case_ids = [cid for cid in case_ids if cid not in failed_ids]

    # Convert RAGAS result to our format
    results: list[RagasResult] = []
    passed_count = 0

    # RAGAS returns a dict with metric names as keys and lists of scores
    ragas_scores = ragas_result.to_pandas().to_dict('records')

    for i, scores_row in enumerate(ragas_scores):
        case_id = valid_case_ids[i] if i < len(valid_case_ids) else f"case_{i}"

        # Extract scores (filter out non-metric columns)
        scores = {
            k: v for k, v in scores_row.items()
            if k not in ['user_input', 'response', 'retrieved_contexts', 'reference']
            and isinstance(v, (int, float))
        }

        # Check thresholds
        passed, failed_metrics = check_thresholds(scores, thresholds)

        if passed:
            passed_count += 1

        results.append(RagasResult(
            case_id=case_id,
            scores=scores,
            passed=passed,
            failed_metrics=failed_metrics,
        ))

    # Calculate averages
    metric_averages = {}
    for metric_name in scores.keys():
        metric_scores = [r.scores.get(metric_name, 0) for r in results]
        if metric_scores:
            metric_averages[metric_name] = sum(metric_scores) / len(metric_scores)

    report = RagasReport(
        total_cases=len(cases),
        evaluated_cases=len(results),
        passed_cases=passed_count,
        failed_cases=len(results) - passed_count,
        skipped_cases=len(failed_ids),
        metric_averages=metric_averages,
        results=results,
        failed_case_ids=failed_ids,
    )

    if verbose:
        print("\n" + "=" * 50)
        print("RAGAS EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Cases: {report.total_cases}")
        print(f"Evaluated: {report.evaluated_cases}")
        print(f"Passed: {report.passed_cases}")
        print(f"Failed: {report.failed_cases}")
        print(f"Skipped: {report.skipped_cases}")
        print(f"Pass Rate: {report.pass_rate:.1%}")
        print("\nMetric Averages:")
        for name, avg in report.metric_averages.items():
            threshold = thresholds.get(name, 0)
            status = "PASS" if avg >= threshold else "FAIL"
            print(f"  {name}: {avg:.3f} (threshold: {threshold}) [{status}]")

        if report.failed_case_ids:
            print(f"\nSkipped Cases: {report.failed_case_ids}")

    return report


# ---------------------------------------------------------------------------
# ASYNC EVALUATION
# ---------------------------------------------------------------------------


async def run_ragas_evaluation_async(
    cases: list[GoldenCase] | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> RagasReport:
    """
    Async version of RAGAS evaluation.

    RAGAS metrics support async evaluation which can be faster.
    """
    # For now, wrap the sync version
    # TODO: Implement true async with single_turn_ascore
    return run_ragas_evaluation(cases, model, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------


def run_ragas_cli():
    """CLI entry point for RAGAS evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--model", "-m", type=str, help="Evaluator model")
    args = parser.parse_args()

    report = run_ragas_evaluation(
        model=args.model,
        verbose=args.verbose,
    )

    # Exit with error code if any failures
    exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    run_ragas_cli()
