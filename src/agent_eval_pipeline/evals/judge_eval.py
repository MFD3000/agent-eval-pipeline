"""
LLM-as-Judge Evaluation

BACKWARD COMPATIBILITY NOTICE:
------------------------------
This module has been ELEVATED following code-elevation principles.
The code has been split into focused modules:
- evals/judge/prompts.py - Externalized prompts
- evals/judge/schemas.py - Data models
- evals/judge/evaluator.py - Core evaluation logic with DI

This file re-exports for backward compatibility. New code should import
from the specific modules or from agent_eval_pipeline.evals.judge directly.

WHY LLM-AS-JUDGE:
-----------------
Rule-based checks can't evaluate semantic quality like clarity,
clinical accuracy, or appropriate tone. LLM-as-judge enables
nuanced evaluation against rubrics.
"""

# Re-export everything from elevated modules for backward compatibility

# Schemas
from agent_eval_pipeline.evals.judge.schemas import (
    DimensionScore,
    JudgeOutput,
    JudgeEvalResult,
    JudgeEvalReport,
    WEIGHTS,
)

# Prompts
from agent_eval_pipeline.evals.judge.prompts import (
    JUDGE_SYSTEM_PROMPT,
    format_judge_user_prompt,
)

# Evaluator
from agent_eval_pipeline.evals.judge.evaluator import (
    run_judge,
    calculate_weighted_score,
    run_judge_eval,
)


def run_judge_eval_cli():
    """CLI entry point for judge eval."""
    print("=" * 60)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 60)

    report = run_judge_eval(verbose=True, threshold=4.2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"\n  [{status}] {result.case_id} (score: {result.weighted_score:.2f}/5)")

        if result.scores:
            for dim, score in result.scores.items():
                weight = WEIGHTS.get(dim, 0) * 100
                print(f"        {dim}: {score:.1f}/5 (weight: {weight:.0f}%)")

        if result.critical_issues:
            print(f"        CRITICAL: {result.critical_issues}")

        print(f"        Assessment: {result.overall_assessment[:100]}...")

    print("\n" + "-" * 60)
    print("DIMENSION AVERAGES:")
    for dim, avg in report.dimension_averages.items():
        print(f"  {dim}: {avg:.2f}/5")

    print(f"\nOverall Average: {report.avg_score:.2f}/5")
    print(f"Threshold: {report.threshold}/5")
    print(f"Passed: {report.passed_cases}/{report.total_cases}")

    if report.critical_failures:
        print(f"\nCRITICAL FAILURES: {report.critical_failures}")

    if report.all_passed:
        print("\n>>> LLM-AS-JUDGE GATE: PASSED <<<")
        return 0
    else:
        print("\n>>> LLM-AS-JUDGE GATE: FAILED <<<")
        return 1


__all__ = [
    # Schemas
    "DimensionScore",
    "JudgeOutput",
    "JudgeEvalResult",
    "JudgeEvalReport",
    "WEIGHTS",
    # Prompts
    "JUDGE_SYSTEM_PROMPT",
    "format_judge_user_prompt",
    # Evaluator
    "run_judge",
    "calculate_weighted_score",
    "run_judge_eval",
    # CLI
    "run_judge_eval_cli",
]


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    exit_code = run_judge_eval_cli()
    sys.exit(exit_code)
