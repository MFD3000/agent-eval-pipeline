"""
Performance evaluator - core evaluation logic with dependency injection.

The evaluator:
1. Runs the agent on golden cases
2. Collects performance metrics
3. Compares against baseline
4. Detects regressions

DEPENDENCY INJECTION:
---------------------
The run_perf_eval function accepts a BaselineStore parameter.
This allows testing regression detection without file I/O.


"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from agent_eval_pipeline.agent import run_agent, AgentResult, AgentError
from agent_eval_pipeline.evals.perf.baseline import (
    BaselineStore,
    PerformanceBaseline,
    get_baseline_store,
)
from agent_eval_pipeline.evals.perf.pricing import estimate_cost
from agent_eval_pipeline.evals.perf.metrics import (
    CasePerformance,
    PerformanceMetrics,
    RegressionCheck,
    PerfEvalResult,
)

if TYPE_CHECKING:
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_LATENCY_THRESHOLD = 0.15  # 15% slower triggers regression
DEFAULT_TOKEN_THRESHOLD = 0.20  # 20% more tokens triggers regression


# ---------------------------------------------------------------------------
# CORE EVALUATION
# ---------------------------------------------------------------------------


def run_perf_eval(
    cases: list[GoldenCase] | None = None,
    latency_regression_threshold: float = DEFAULT_LATENCY_THRESHOLD,
    token_regression_threshold: float = DEFAULT_TOKEN_THRESHOLD,
    update_baseline: bool = False,
    verbose: bool = False,
    baseline_store: BaselineStore | None = None,
) -> PerfEvalResult:
    """
    Run performance evaluation on golden cases.

    Args:
        cases: Cases to evaluate. Defaults to all golden cases.
        latency_regression_threshold: Max allowed latency increase (0.15 = 15%)
        token_regression_threshold: Max allowed token increase (0.20 = 20%)
        update_baseline: If True, update baseline with current results
        verbose: Print progress
        baseline_store: Injectable baseline store (uses file store if None)

    Returns:
        PerfEvalResult with metrics and regression checks
    """
    from agent_eval_pipeline.golden_sets.thyroid_cases import get_all_golden_cases

    # Use injected store or create default
    store = baseline_store or get_baseline_store()

    cases = cases or get_all_golden_cases()
    case_results: list[CasePerformance] = []

    for case in cases:
        if verbose:
            print(f"Running perf eval: {case.id}...")

        result = run_agent(case)

        if isinstance(result, AgentError):
            case_results.append(
                CasePerformance(
                    case_id=case.id,
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    model="",
                    success=False,
                    error=result.error_message,
                )
            )
        else:
            case_results.append(
                CasePerformance(
                    case_id=case.id,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    total_tokens=result.total_tokens,
                    model=result.model,
                    success=True,
                )
            )

    # Filter to successful cases for metrics
    successful = [r for r in case_results if r.success]

    if not successful:
        return PerfEvalResult(
            passed=False,
            metrics=PerformanceMetrics(
                p50_latency_ms=0,
                p95_latency_ms=0,
                avg_latency_ms=0,
                avg_input_tokens=0,
                avg_output_tokens=0,
                avg_total_tokens=0,
                total_cost_estimate=0,
            ),
            baseline=None,
            regression_checks=[],
            case_results=case_results,
            failure_reason="No successful cases to evaluate",
        )

    # Calculate metrics
    metrics = _calculate_metrics(successful)

    # Load baseline and check for regressions
    baseline = store.load()
    regression_checks, is_regression = _check_regressions(
        metrics=metrics,
        baseline=baseline,
        latency_threshold=latency_regression_threshold,
        token_threshold=token_regression_threshold,
        primary_model=successful[0].model,
    )

    # Update baseline if requested
    if update_baseline:
        new_baseline = PerformanceBaseline(
            p50_latency_ms=metrics.p50_latency_ms,
            p95_latency_ms=metrics.p95_latency_ms,
            avg_input_tokens=metrics.avg_input_tokens,
            avg_output_tokens=metrics.avg_output_tokens,
            avg_total_tokens=metrics.avg_total_tokens,
            expected_model=successful[0].model,
            run_count=(baseline.run_count + 1) if baseline else 1,
        )
        store.save(new_baseline)
        if verbose:
            print("Baseline updated")

    failure_reason = None
    if is_regression:
        failures = [c for c in regression_checks if c.is_regression]
        failure_reason = f"Regressions detected: {[c.metric for c in failures]}"

    return PerfEvalResult(
        passed=not is_regression,
        metrics=metrics,
        baseline=baseline,
        regression_checks=regression_checks,
        case_results=case_results,
        failure_reason=failure_reason,
    )


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def _calculate_metrics(successful: list[CasePerformance]) -> PerformanceMetrics:
    """Calculate aggregate metrics from successful case results."""
    latencies = sorted(r.latency_ms for r in successful)

    p50_idx = len(latencies) // 2
    p95_idx = int(len(latencies) * 0.95)

    p50_latency = latencies[p50_idx]
    p95_latency = latencies[min(p95_idx, len(latencies) - 1)]

    avg_input = statistics.mean(r.input_tokens for r in successful)
    avg_output = statistics.mean(r.output_tokens for r in successful)
    avg_total = statistics.mean(r.total_tokens for r in successful)

    models_used = {r.model for r in successful}

    total_cost = sum(
        estimate_cost(r.input_tokens, r.output_tokens, r.model) for r in successful
    )

    return PerformanceMetrics(
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        avg_latency_ms=statistics.mean(latencies),
        avg_input_tokens=avg_input,
        avg_output_tokens=avg_output,
        avg_total_tokens=avg_total,
        total_cost_estimate=total_cost,
        models_used=models_used,
    )


def _check_regressions(
    metrics: PerformanceMetrics,
    baseline: PerformanceBaseline | None,
    latency_threshold: float,
    token_threshold: float,
    primary_model: str,
) -> tuple[list[RegressionCheck], bool]:
    """Check for regressions against baseline."""
    regression_checks: list[RegressionCheck] = []
    is_regression = False

    if baseline is None:
        return regression_checks, is_regression

    # Check p95 latency
    if baseline.p95_latency_ms > 0:
        latency_change = (
            metrics.p95_latency_ms - baseline.p95_latency_ms
        ) / baseline.p95_latency_ms
        latency_check = RegressionCheck(
            metric="p95_latency_ms",
            baseline_value=baseline.p95_latency_ms,
            current_value=metrics.p95_latency_ms,
            change_percent=latency_change * 100,
            threshold_percent=latency_threshold * 100,
            is_regression=latency_change > latency_threshold,
        )
        regression_checks.append(latency_check)
        if latency_check.is_regression:
            is_regression = True

    # Check total tokens
    if baseline.avg_total_tokens > 0:
        token_change = (
            metrics.avg_total_tokens - baseline.avg_total_tokens
        ) / baseline.avg_total_tokens
        token_check = RegressionCheck(
            metric="avg_total_tokens",
            baseline_value=baseline.avg_total_tokens,
            current_value=metrics.avg_total_tokens,
            change_percent=token_change * 100,
            threshold_percent=token_threshold * 100,
            is_regression=token_change > token_threshold,
        )
        regression_checks.append(token_check)
        if token_check.is_regression:
            is_regression = True

    # Check model consistency
    if baseline.expected_model and primary_model != baseline.expected_model:
        regression_checks.append(
            RegressionCheck(
                metric="model",
                baseline_value=0,  # N/A
                current_value=0,
                change_percent=0,
                threshold_percent=0,
                is_regression=True,
            )
        )
        is_regression = True

    return regression_checks, is_regression
