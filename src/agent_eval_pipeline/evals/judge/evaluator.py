"""
Judge evaluator - core evaluation logic with dependency injection.

The evaluator orchestrates:
1. Running the agent on cases
2. Calling the judge model
3. Aggregating results

DEPENDENCY INJECTION:
---------------------
The run_judge function accepts an optional OpenAI client parameter.
This allows testing without API calls by injecting a mock client.

INTERVIEW TALKING POINT:
------------------------
"The judge evaluator follows the factory pattern - run_judge accepts
an optional client parameter, so I can inject a mock for testing.
This lets me verify the scoring logic works correctly without
making any API calls. The calculate_weighted_score function is
pure, so testing it is trivial."
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

from openai import OpenAI

from agent_eval_pipeline.agent import run_agent, AgentResult, AgentError
from agent_eval_pipeline.evals.judge.schemas import (
    DimensionScore,
    JudgeOutput,
    JudgeEvalResult,
    JudgeEvalReport,
    WEIGHTS,
)
from agent_eval_pipeline.evals.judge.prompts import (
    JUDGE_SYSTEM_PROMPT,
    format_judge_user_prompt,
)

if TYPE_CHECKING:
    from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase
    from agent_eval_pipeline.harness.context import AgentRunContext


# ---------------------------------------------------------------------------
# JUDGE CLIENT PROTOCOL
# ---------------------------------------------------------------------------


class JudgeClient(Protocol):
    """Protocol for judge client - enables DI for testing."""

    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> JudgeOutput | None:
        """Run evaluation and return structured output."""
        ...


class OpenAIJudgeClient:
    """Production judge client using OpenAI."""

    def __init__(self, client: OpenAI | None = None, model: str | None = None):
        self._client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model or os.environ.get("JUDGE_MODEL", "gpt-4o")

    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> JudgeOutput | None:
        """Run evaluation using OpenAI's structured output."""
        try:
            response = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=JudgeOutput,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"Judge error: {e}")
            return None


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------


def run_judge(
    case: GoldenCase,
    output: LabInsightsSummary,
    client: JudgeClient | None = None,
) -> JudgeOutput | None:
    """
    Run the judge model on a single case.

    Args:
        case: The golden case being evaluated
        output: The agent's output to evaluate
        client: Optional judge client (uses OpenAI if not provided)

    Returns:
        JudgeOutput on success, None on failure
    """
    # Use injected client or create default
    judge_client = client or OpenAIJudgeClient()

    user_prompt = format_judge_user_prompt(case, output)

    return judge_client.evaluate(JUDGE_SYSTEM_PROMPT, user_prompt)


def calculate_weighted_score(judge_output: JudgeOutput) -> float:
    """
    Calculate weighted score from judge output.

    This is a PURE FUNCTION - no side effects, no external calls.
    Easy to unit test.

    Args:
        judge_output: Structured output from the judge

    Returns:
        Weighted score between 1 and 5
    """
    scores = {
        "clinical_correctness": judge_output.clinical_correctness.score,
        "safety_compliance": judge_output.safety_compliance.score,
        "completeness": judge_output.completeness.score,
        "clarity": judge_output.clarity.score,
    }

    weighted = sum(scores[dim] * WEIGHTS[dim] for dim in WEIGHTS)
    return weighted


# ---------------------------------------------------------------------------
# BATCH EVALUATION
# ---------------------------------------------------------------------------


def run_judge_eval(
    cases: list[GoldenCase] | None = None,
    contexts: list[AgentRunContext] | None = None,
    threshold: float = 4.2,
    verbose: bool = False,
    client: JudgeClient | None = None,
) -> JudgeEvalReport:
    """
    Run LLM-as-judge evaluation on golden cases.

    Args:
        cases: Cases to evaluate. Defaults to all golden cases.
        contexts: Pre-computed agent results (from harness). If provided,
                  skips running agents and uses cached results.
        threshold: Minimum weighted score to pass. Default 4.2/5.
        verbose: Print progress.
        client: Optional judge client for DI (testing).

    Returns:
        JudgeEvalReport with scores for each case.

    CONTEXT SHARING:
    ----------------
    When called from the harness with contexts, this evaluator uses
    pre-computed agent results instead of calling run_agent() internally.
    """
    # If contexts provided, use them
    if contexts is not None:
        return _run_with_contexts(contexts, threshold, verbose, client)

    # Legacy path: run agents internally
    from agent_eval_pipeline.golden_sets.thyroid_cases import get_all_golden_cases

    cases = cases or get_all_golden_cases()
    results: list[JudgeEvalResult] = []
    critical_failures: list[str] = []

    for case in cases:
        if verbose:
            print(f"Running judge eval: {case.id}...")

        # First, run the agent
        agent_result = run_agent(case)

        if isinstance(agent_result, AgentError):
            if verbose:
                print(f"  Agent error: {agent_result.error_message}")
            results.append(
                JudgeEvalResult(
                    case_id=case.id,
                    passed=False,
                    weighted_score=0.0,
                    scores={},
                    reasoning={"error": agent_result.error_message},
                    overall_assessment="Agent failed to produce output",
                    critical_issues=["Agent error"],
                )
            )
            continue

        # Run the judge
        judge_output = run_judge(case, agent_result.output, client=client)

        if judge_output is None:
            results.append(
                JudgeEvalResult(
                    case_id=case.id,
                    passed=False,
                    weighted_score=0.0,
                    scores={},
                    reasoning={"error": "Judge failed"},
                    overall_assessment="Judge evaluation failed",
                    critical_issues=["Judge error"],
                )
            )
            continue

        # Calculate weighted score
        weighted_score = calculate_weighted_score(judge_output)

        # Check for critical issues
        has_critical = len(judge_output.critical_issues) > 0
        if has_critical:
            critical_failures.append(f"{case.id}: {judge_output.critical_issues}")

        # Determine pass/fail
        passed = weighted_score >= threshold and not has_critical

        results.append(
            JudgeEvalResult(
                case_id=case.id,
                passed=passed,
                weighted_score=weighted_score,
                scores={
                    "clinical_correctness": judge_output.clinical_correctness.score,
                    "safety_compliance": judge_output.safety_compliance.score,
                    "completeness": judge_output.completeness.score,
                    "clarity": judge_output.clarity.score,
                },
                reasoning={
                    "clinical_correctness": judge_output.clinical_correctness.reasoning,
                    "safety_compliance": judge_output.safety_compliance.reasoning,
                    "completeness": judge_output.completeness.reasoning,
                    "clarity": judge_output.clarity.reasoning,
                },
                overall_assessment=judge_output.overall_assessment,
                critical_issues=judge_output.critical_issues,
            )
        )

    return _aggregate_results(results, critical_failures, threshold)


def _run_with_contexts(
    contexts: list[AgentRunContext],
    threshold: float,
    verbose: bool,
    client: JudgeClient | None,
) -> JudgeEvalReport:
    """
    Run judge eval using pre-computed agent contexts.

    This is the context-aware path used by the harness.
    """
    results: list[JudgeEvalResult] = []
    critical_failures: list[str] = []

    for ctx in contexts:
        if verbose:
            print(f"Running judge eval: {ctx.case_id}...")

        # Check for agent errors
        if not ctx.success:
            if verbose:
                print(f"  Agent error: {ctx.error_message}")
            results.append(
                JudgeEvalResult(
                    case_id=ctx.case_id,
                    passed=False,
                    weighted_score=0.0,
                    scores={},
                    reasoning={"error": ctx.error_message},
                    overall_assessment="Agent failed to produce output",
                    critical_issues=["Agent error"],
                )
            )
            continue

        # Run the judge on pre-computed output
        judge_output = run_judge(ctx.case, ctx.output, client=client)

        if judge_output is None:
            results.append(
                JudgeEvalResult(
                    case_id=ctx.case_id,
                    passed=False,
                    weighted_score=0.0,
                    scores={},
                    reasoning={"error": "Judge failed"},
                    overall_assessment="Judge evaluation failed",
                    critical_issues=["Judge error"],
                )
            )
            continue

        # Calculate weighted score
        weighted_score = calculate_weighted_score(judge_output)

        # Check for critical issues
        has_critical = len(judge_output.critical_issues) > 0
        if has_critical:
            critical_failures.append(f"{ctx.case_id}: {judge_output.critical_issues}")

        # Determine pass/fail
        passed = weighted_score >= threshold and not has_critical

        results.append(
            JudgeEvalResult(
                case_id=ctx.case_id,
                passed=passed,
                weighted_score=weighted_score,
                scores={
                    "clinical_correctness": judge_output.clinical_correctness.score,
                    "safety_compliance": judge_output.safety_compliance.score,
                    "completeness": judge_output.completeness.score,
                    "clarity": judge_output.clarity.score,
                },
                reasoning={
                    "clinical_correctness": judge_output.clinical_correctness.reasoning,
                    "safety_compliance": judge_output.safety_compliance.reasoning,
                    "completeness": judge_output.completeness.reasoning,
                    "clarity": judge_output.clarity.reasoning,
                },
                overall_assessment=judge_output.overall_assessment,
                critical_issues=judge_output.critical_issues,
            )
        )

    return _aggregate_results(results, critical_failures, threshold)


def _aggregate_results(
    results: list[JudgeEvalResult],
    critical_failures: list[str],
    threshold: float,
) -> JudgeEvalReport:
    """Aggregate individual results into a report."""
    if results:
        valid_results = [r for r in results if r.weighted_score > 0]
        avg_score = (
            sum(r.weighted_score for r in valid_results) / len(valid_results)
            if valid_results
            else 0
        )

        dimension_avgs = {}
        for dim in WEIGHTS:
            dim_scores = [
                r.scores.get(dim, 0) for r in valid_results if dim in r.scores
            ]
            dimension_avgs[dim] = sum(dim_scores) / len(dim_scores) if dim_scores else 0

        passed = sum(1 for r in results if r.passed)
    else:
        avg_score = 0
        dimension_avgs = {dim: 0 for dim in WEIGHTS}
        passed = 0

    return JudgeEvalReport(
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=len(results) - passed,
        avg_score=avg_score,
        threshold=threshold,
        dimension_averages=dimension_avgs,
        results=results,
        critical_failures=critical_failures,
    )
