"""
Performance Regression Evaluation

BACKWARD COMPATIBILITY NOTICE:
------------------------------
This module has been ELEVATED following code-elevation principles.
The code has been split into focused modules:
- evals/perf/baseline.py - BaselineStore protocol + implementations
- evals/perf/pricing.py - Cost estimation
- evals/perf/metrics.py - Data models
- evals/perf/evaluator.py - Core evaluation logic with DI

This file re-exports for backward compatibility. New code should import
from the specific modules or from agent_eval_pipeline.evals.perf directly.

WHY PERFORMANCE EVAL:
---------------------
A small prompt change can double token usage (and cost), triple latency,
or break KV-cache efficiency. These regressions are invisible in semantic
evals but devastate production.
"""

# Re-export everything from elevated modules for backward compatibility

# Baseline
from agent_eval_pipeline.evals.perf.baseline import (
    BaselineStore,
    PerformanceBaseline,
    FileBaselineStore,
    InMemoryBaselineStore,
    get_baseline_store,
)

# Pricing
from agent_eval_pipeline.evals.perf.pricing import (
    MODEL_PRICING,
    estimate_cost,
)

# Metrics
from agent_eval_pipeline.evals.perf.metrics import (
    CasePerformance,
    PerformanceMetrics,
    RegressionCheck,
    PerfEvalResult,
)

# Evaluator
from agent_eval_pipeline.evals.perf.evaluator import (
    run_perf_eval,
)


# Legacy aliases
BASELINE_FILE = FileBaselineStore().path
load_baseline = FileBaselineStore().load
save_baseline = FileBaselineStore().save


def run_perf_eval_cli(update_baseline: bool = False):
    """CLI entry point for performance eval."""
    print("=" * 60)
    print("PERFORMANCE REGRESSION EVAL")
    print("=" * 60)

    result = run_perf_eval(verbose=True, update_baseline=update_baseline)

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    m = result.metrics
    print(f"  Latency (p50): {m.p50_latency_ms:.0f}ms")
    print(f"  Latency (p95): {m.p95_latency_ms:.0f}ms")
    print(f"  Latency (avg): {m.avg_latency_ms:.0f}ms")
    print(f"  Tokens (avg input): {m.avg_input_tokens:.0f}")
    print(f"  Tokens (avg output): {m.avg_output_tokens:.0f}")
    print(f"  Tokens (avg total): {m.avg_total_tokens:.0f}")
    print(f"  Cost estimate: ${m.total_cost_estimate:.4f}")
    print(f"  Models used: {m.models_used}")

    if result.baseline:
        print("\n" + "-" * 60)
        print("BASELINE COMPARISON")
        print("-" * 60)
        b = result.baseline
        print(f"  Baseline p95 latency: {b.p95_latency_ms:.0f}ms")
        print(f"  Baseline avg tokens: {b.avg_total_tokens:.0f}")
        print(f"  Baseline model: {b.expected_model}")

    if result.regression_checks:
        print("\n" + "-" * 60)
        print("REGRESSION CHECKS")
        print("-" * 60)
        for check in result.regression_checks:
            status = "FAIL" if check.is_regression else "PASS"
            print(
                f"  [{status}] {check.metric}: "
                f"{check.change_percent:+.1f}% (threshold: {check.threshold_percent}%)"
            )

    print("\n" + "-" * 60)
    success_rate = sum(1 for r in result.case_results if r.success) / len(
        result.case_results
    )
    print(f"Success rate: {success_rate:.1%}")

    if result.passed:
        print("\n>>> PERFORMANCE EVAL GATE: PASSED <<<")
        return 0
    else:
        print(f"\n>>> PERFORMANCE EVAL GATE: FAILED <<<")
        print(f"Reason: {result.failure_reason}")
        return 1


__all__ = [
    # Baseline
    "BaselineStore",
    "PerformanceBaseline",
    "FileBaselineStore",
    "InMemoryBaselineStore",
    "get_baseline_store",
    "BASELINE_FILE",
    "load_baseline",
    "save_baseline",
    # Pricing
    "MODEL_PRICING",
    "estimate_cost",
    # Metrics
    "CasePerformance",
    "PerformanceMetrics",
    "RegressionCheck",
    "PerfEvalResult",
    # Evaluator
    "run_perf_eval",
    # CLI
    "run_perf_eval_cli",
]


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    # Check for --update-baseline flag
    update = "--update-baseline" in sys.argv

    exit_code = run_perf_eval_cli(update_baseline=update)
    sys.exit(exit_code)
