"""
Unified Evaluation Runner - Compare results across all evaluation frameworks.

This module provides a single entry point to run evaluations using:
- Custom LLM-as-Judge (our original implementation)
- DeepEval (pytest-native, G-Eval metrics)
- RAGAS (RAG-specialized metrics)
- DSPy Judge (optimizable judge)

EVALUATION-DRIVEN DEVELOPMENT:
------------------------------
Function Health practices "evaluation-driven development" where every change
to prompts, tools, or retrieval must pass automated eval gates. This unified
runner supports that by:
1. Running fast checks first (schema, retrieval)
2. Running LLM-based evals only if fast checks pass
3. Comparing results across frameworks
4. Gating on the most critical metrics (safety, faithfulness)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eval_pipeline.golden_sets import GoldenCase


# ---------------------------------------------------------------------------
# FRAMEWORK ENUM
# ---------------------------------------------------------------------------


class EvalFramework(Enum):
    """Available evaluation frameworks."""

    SCHEMA = "schema"          # Fast schema validation
    RETRIEVAL = "retrieval"    # Retrieval quality (P/R/F1)
    CUSTOM = "custom"          # Our LLM-as-judge
    DEEPEVAL = "deepeval"      # DeepEval G-Eval metrics
    RAGAS = "ragas"            # RAGAS RAG metrics
    DSPY = "dspy"              # DSPy-based judge


# Framework categories for ordering
FAST_FRAMEWORKS = [EvalFramework.SCHEMA, EvalFramework.RETRIEVAL]
LLM_FRAMEWORKS = [EvalFramework.CUSTOM, EvalFramework.DEEPEVAL, EvalFramework.RAGAS, EvalFramework.DSPY]


# ---------------------------------------------------------------------------
# RESULT SCHEMAS
# ---------------------------------------------------------------------------


@dataclass
class FrameworkResult:
    """Result from a single framework evaluation."""

    framework: EvalFramework
    passed: bool
    scores: dict[str, float]
    details: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        return "PASS" if self.passed else "FAIL"


@dataclass
class UnifiedEvalReport:
    """Combined report from all frameworks."""

    total_frameworks: int
    passed_frameworks: int
    failed_frameworks: int
    results: dict[EvalFramework, FrameworkResult]
    all_passed: bool = False

    @property
    def pass_rate(self) -> float:
        if self.total_frameworks == 0:
            return 0.0
        return self.passed_frameworks / self.total_frameworks

    def get_metric_comparison(self, metric_name: str) -> dict[str, float]:
        """Compare a metric across frameworks that have it."""
        comparison = {}
        for framework, result in self.results.items():
            if metric_name in result.scores:
                comparison[framework.value] = result.scores[metric_name]
        return comparison


# ---------------------------------------------------------------------------
# INDIVIDUAL FRAMEWORK RUNNERS
# ---------------------------------------------------------------------------


def run_schema_eval(cases: list[GoldenCase], verbose: bool = False) -> FrameworkResult:
    """Run schema validation."""
    try:
        from agent_eval_pipeline.evals.schema_eval import run_schema_eval as _run_schema_eval

        # Run the actual schema eval which validates outputs against expectations
        report = _run_schema_eval(cases=cases, verbose=verbose)

        return FrameworkResult(
            framework=EvalFramework.SCHEMA,
            passed=report.all_passed,
            scores={"pass_rate": report.pass_rate},
            details={
                "passed": report.passed_cases,
                "failed": report.failed_cases,
                "total": report.total_cases,
            },
        )
    except Exception as e:
        return FrameworkResult(
            framework=EvalFramework.SCHEMA,
            passed=False,
            scores={},
            error=str(e),
        )


def run_retrieval_eval(cases: list[GoldenCase], verbose: bool = False) -> FrameworkResult:
    """Run retrieval quality evaluation."""
    try:
        from agent_eval_pipeline.evals.retrieval_eval import run_retrieval_eval as _run_retrieval

        report = _run_retrieval(verbose=verbose)

        return FrameworkResult(
            framework=EvalFramework.RETRIEVAL,
            passed=report.avg_f1 >= 0.5,  # Threshold
            scores={
                "precision": report.avg_precision,
                "recall": report.avg_recall,
                "f1": report.avg_f1,
            },
            details={"report": report},
        )
    except Exception as e:
        return FrameworkResult(
            framework=EvalFramework.RETRIEVAL,
            passed=False,
            scores={},
            error=str(e),
        )


def run_custom_judge_eval(cases: list[GoldenCase], verbose: bool = False) -> FrameworkResult:
    """Run our custom LLM-as-judge evaluation."""
    try:
        from agent_eval_pipeline.evals.judge import run_judge_eval

        report = run_judge_eval(cases=cases, verbose=verbose)

        return FrameworkResult(
            framework=EvalFramework.CUSTOM,
            passed=report.failed_cases == 0,
            scores=report.dimension_averages,
            details={
                "avg_score": report.avg_score,
                "passed_cases": report.passed_cases,
                "failed_cases": report.failed_cases,
            },
        )
    except Exception as e:
        return FrameworkResult(
            framework=EvalFramework.CUSTOM,
            passed=False,
            scores={},
            error=str(e),
        )


def run_deepeval_eval(cases: list[GoldenCase], verbose: bool = False) -> FrameworkResult:
    """Run DeepEval evaluation."""
    try:
        from agent_eval_pipeline.evals.deepeval import run_deepeval_evaluation

        report = run_deepeval_evaluation(cases=cases, verbose=verbose)

        return FrameworkResult(
            framework=EvalFramework.DEEPEVAL,
            passed=report.all_passed,
            scores=report.metric_averages,
            details={
                "pass_rate": report.pass_rate,
                "passed_cases": report.passed_cases,
                "failed_cases": report.failed_cases,
            },
        )
    except Exception as e:
        return FrameworkResult(
            framework=EvalFramework.DEEPEVAL,
            passed=False,
            scores={},
            error=str(e),
        )


def run_ragas_eval(cases: list[GoldenCase], verbose: bool = False) -> FrameworkResult:
    """Run RAGAS evaluation."""
    try:
        from agent_eval_pipeline.evals.ragas import run_ragas_evaluation

        report = run_ragas_evaluation(cases=cases, verbose=verbose)

        return FrameworkResult(
            framework=EvalFramework.RAGAS,
            passed=report.all_passed,
            scores=report.metric_averages,
            details={
                "pass_rate": report.pass_rate,
                "evaluated_cases": report.evaluated_cases,
                "skipped_cases": report.skipped_cases,
            },
        )
    except Exception as e:
        return FrameworkResult(
            framework=EvalFramework.RAGAS,
            passed=False,
            scores={},
            error=str(e),
        )


def run_dspy_judge_eval(cases: list[GoldenCase], verbose: bool = False) -> FrameworkResult:
    """Run DSPy-based judge evaluation."""
    try:
        from agent_eval_pipeline.evals.judge.dspy_judge import run_dspy_judge

        # Run on first case as sample
        from agent_eval_pipeline.agent import run_agent

        scores = {}
        passed_count = 0

        for case in cases[:3]:  # Limit to first 3 for speed
            result = run_agent(case)
            if hasattr(result, 'output'):
                judge_result = run_dspy_judge(case, result.output)
                if judge_result:
                    scores[case.id] = judge_result.weighted_score
                    if judge_result.weighted_score >= 4.0:
                        passed_count += 1

        avg_score = sum(scores.values()) / len(scores) if scores else 0

        return FrameworkResult(
            framework=EvalFramework.DSPY,
            passed=avg_score >= 4.0,
            scores={"weighted_score": avg_score, **scores},
            details={"passed_count": passed_count},
        )
    except Exception as e:
        return FrameworkResult(
            framework=EvalFramework.DSPY,
            passed=False,
            scores={},
            error=str(e),
        )


# Framework runner mapping
FRAMEWORK_RUNNERS = {
    EvalFramework.SCHEMA: run_schema_eval,
    EvalFramework.RETRIEVAL: run_retrieval_eval,
    EvalFramework.CUSTOM: run_custom_judge_eval,
    EvalFramework.DEEPEVAL: run_deepeval_eval,
    EvalFramework.RAGAS: run_ragas_eval,
    EvalFramework.DSPY: run_dspy_judge_eval,
}


# ---------------------------------------------------------------------------
# UNIFIED RUNNER
# ---------------------------------------------------------------------------


def run_unified_eval(
    cases: list[GoldenCase] | None = None,
    frameworks: list[EvalFramework] | None = None,
    skip_on_failure: bool = True,
    verbose: bool = False,
) -> UnifiedEvalReport:
    """
    Run evaluation across multiple frameworks.

    Args:
        cases: Golden cases to evaluate. Defaults to all.
        frameworks: Which frameworks to run. Defaults to all.
        skip_on_failure: Skip LLM evals if fast evals fail.
        verbose: Print progress and results.

    Returns:
        UnifiedEvalReport with results from all frameworks.

    Example:
        >>> report = run_unified_eval(verbose=True)
        >>> print(f"Overall: {report.passed_frameworks}/{report.total_frameworks} passed")
        >>> for fw, result in report.results.items():
        ...     print(f"  {fw.value}: {result.status}")
    """
    from agent_eval_pipeline.golden_sets import get_all_golden_cases

    cases = cases or get_all_golden_cases()
    frameworks = frameworks or list(EvalFramework)

    results: dict[EvalFramework, FrameworkResult] = {}

    if verbose:
        print("\n" + "=" * 60)
        print("UNIFIED EVALUATION PIPELINE")
        print("=" * 60)
        print(f"Cases: {len(cases)}")
        print(f"Frameworks: {[f.value for f in frameworks]}")

    # Run fast frameworks first
    fast_passed = True
    for framework in frameworks:
        if framework in FAST_FRAMEWORKS:
            if verbose:
                print(f"\n[{framework.value.upper()}] Running...")

            runner = FRAMEWORK_RUNNERS[framework]
            result = runner(cases, verbose=verbose)
            results[framework] = result

            if verbose:
                print(f"[{framework.value.upper()}] {result.status}")
                for name, score in result.scores.items():
                    print(f"  {name}: {score:.3f}")

            if not result.passed:
                fast_passed = False

    # Check if we should skip LLM evals
    if skip_on_failure and not fast_passed:
        if verbose:
            print("\n⚠️  Fast evals failed - skipping LLM evals")

        for framework in frameworks:
            if framework in LLM_FRAMEWORKS and framework not in results:
                results[framework] = FrameworkResult(
                    framework=framework,
                    passed=False,
                    scores={},
                    error="Skipped due to fast eval failure",
                )
    else:
        # Run LLM frameworks
        for framework in frameworks:
            if framework in LLM_FRAMEWORKS:
                if verbose:
                    print(f"\n[{framework.value.upper()}] Running...")

                runner = FRAMEWORK_RUNNERS[framework]
                result = runner(cases, verbose=verbose)
                results[framework] = result

                if verbose:
                    print(f"[{framework.value.upper()}] {result.status}")
                    if result.error:
                        print(f"  Error: {result.error}")
                    else:
                        for name, score in list(result.scores.items())[:5]:
                            print(f"  {name}: {score:.3f}")

    # Calculate aggregates
    passed_count = sum(1 for r in results.values() if r.passed)
    failed_count = len(results) - passed_count

    report = UnifiedEvalReport(
        total_frameworks=len(results),
        passed_frameworks=passed_count,
        failed_frameworks=failed_count,
        results=results,
        all_passed=failed_count == 0,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("UNIFIED EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Frameworks Run: {report.total_frameworks}")
        print(f"Passed: {report.passed_frameworks}")
        print(f"Failed: {report.failed_frameworks}")
        print(f"Overall: {'PASS' if report.all_passed else 'FAIL'}")

        print("\nFramework Results:")
        for framework, result in report.results.items():
            status_emoji = "✓" if result.passed else "✗" if not result.error else "⚠"
            print(f"  {status_emoji} {framework.value}: {result.status}")

    return report


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------


def run_unified_cli():
    """CLI entry point for unified evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run unified evaluation pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--frameworks",
        "-f",
        nargs="+",
        choices=[f.value for f in EvalFramework],
        help="Frameworks to run",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip LLM evals on fast eval failure",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Parse frameworks
    frameworks = None
    if args.frameworks:
        frameworks = [EvalFramework(f) for f in args.frameworks]

    report = run_unified_eval(
        frameworks=frameworks,
        skip_on_failure=not args.no_skip,
        verbose=args.verbose,
    )

    if args.json:
        import json
        output = {
            "passed": report.all_passed,
            "frameworks": {
                fw.value: {
                    "passed": r.passed,
                    "scores": r.scores,
                    "error": r.error,
                }
                for fw, r in report.results.items()
            },
        }
        print(json.dumps(output, indent=2))

    exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    run_unified_cli()
