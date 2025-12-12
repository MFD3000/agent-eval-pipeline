"""
AgentRunContext - Shared context for evaluation runs.

This module enables running agents ONCE and sharing results across all evaluators,
rather than each evaluator calling run_agent() independently.

BENEFITS:
---------
1. Cost reduction: 4-5x fewer LLM API calls per evaluation run
2. Consistency: All evaluators score the SAME output (no LLM non-determinism)
3. Real retrieval validation: Can validate against actual retrieved_docs

USAGE:
------
# Harness builds contexts once
contexts = [
    AgentRunContext(case=case, result=run_agent(case))
    for case in cases
]

# Evaluators receive pre-computed results
schema_report = run_schema_eval(contexts=contexts)
judge_report = run_judge_eval(contexts=contexts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase
    from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
    from agent_eval_pipeline.agent import AgentResult, AgentError


@dataclass
class AgentRunContext:
    """
    Cached agent execution results for a single case.

    This is the unit of sharing between evaluators. It wraps an AgentResult
    (or AgentError) along with the original case for evaluation.

    Properties provide convenient access to common fields while handling
    the success/error distinction.
    """

    case: GoldenCase
    result: AgentResult | AgentError
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def case_id(self) -> str:
        """Shortcut to case.id."""
        return self.case.id

    @property
    def success(self) -> bool:
        """True if agent ran successfully (not an error)."""
        from agent_eval_pipeline.agent import AgentError
        return not isinstance(self.result, AgentError)

    @property
    def output(self) -> LabInsightsSummary | None:
        """The agent's structured output, or None if error."""
        if self.success:
            return self.result.output
        return None

    @property
    def retrieved_docs(self) -> list[dict]:
        """Retrieved documents from RAG, or empty list if error/none."""
        if self.success and hasattr(self.result, 'retrieved_docs') and self.result.retrieved_docs:
            return self.result.retrieved_docs
        return []

    @property
    def latency_ms(self) -> float | None:
        """Agent execution latency, or None if error."""
        if self.success:
            return self.result.latency_ms
        return None

    @property
    def total_tokens(self) -> int | None:
        """Total tokens used, or None if error."""
        if self.success:
            return self.result.total_tokens
        return None

    @property
    def input_tokens(self) -> int | None:
        """Input tokens used, or None if error."""
        if self.success:
            return self.result.input_tokens
        return None

    @property
    def output_tokens(self) -> int | None:
        """Output tokens used, or None if error."""
        if self.success:
            return self.result.output_tokens
        return None

    @property
    def model(self) -> str | None:
        """Model used, or None if error."""
        if self.success:
            return self.result.model
        return None

    @property
    def error_message(self) -> str | None:
        """Error message if failed, None if success."""
        from agent_eval_pipeline.agent import AgentError
        if isinstance(self.result, AgentError):
            return f"{self.result.error_type}: {self.result.error_message}"
        return None


@dataclass
class EvalRunContext:
    """
    Context for an entire evaluation run.

    Holds all AgentRunContexts for a batch of cases, keyed by case_id
    for efficient lookup. Also tracks run metadata.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_contexts: dict[str, AgentRunContext] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_context(self, context: AgentRunContext) -> None:
        """Add an agent context to the run."""
        self.agent_contexts[context.case_id] = context

    def get_context(self, case_id: str) -> AgentRunContext | None:
        """Get context for a specific case."""
        return self.agent_contexts.get(case_id)

    @property
    def contexts(self) -> list[AgentRunContext]:
        """All contexts as a list (for iteration)."""
        return list(self.agent_contexts.values())

    @property
    def cases(self) -> list[GoldenCase]:
        """All cases from contexts."""
        return [ctx.case for ctx in self.agent_contexts.values()]

    @property
    def successful_contexts(self) -> list[AgentRunContext]:
        """Only contexts where agent succeeded."""
        return [ctx for ctx in self.agent_contexts.values() if ctx.success]

    @property
    def failed_contexts(self) -> list[AgentRunContext]:
        """Only contexts where agent failed."""
        return [ctx for ctx in self.agent_contexts.values() if not ctx.success]

    @property
    def success_rate(self) -> float:
        """Fraction of cases where agent succeeded."""
        if not self.agent_contexts:
            return 0.0
        return len(self.successful_contexts) / len(self.agent_contexts)


def build_agent_contexts(
    cases: list[GoldenCase],
    verbose: bool = False,
) -> list[AgentRunContext]:
    """
    Run agents on all cases and build AgentRunContext list.

    This is the main entry point for building contexts. It runs agents
    once and returns contexts that can be shared across evaluators.

    Args:
        cases: Golden cases to run
        verbose: Print progress

    Returns:
        List of AgentRunContext, one per case
    """
    from agent_eval_pipeline.agent import run_agent

    contexts = []
    for i, case in enumerate(cases):
        if verbose:
            print(f"  Running agent on case {i+1}/{len(cases)}: {case.id}...")

        result = run_agent(case)
        ctx = AgentRunContext(case=case, result=result)
        contexts.append(ctx)

        if verbose:
            status = "OK" if ctx.success else f"ERROR: {ctx.error_message}"
            print(f"    {status}")

    return contexts


def build_eval_run_context(
    cases: list[GoldenCase],
    verbose: bool = False,
) -> EvalRunContext:
    """
    Build a complete EvalRunContext with all agent results.

    Convenience function that creates the run context and populates it.

    Args:
        cases: Golden cases to run
        verbose: Print progress

    Returns:
        EvalRunContext with all agent results
    """
    contexts = build_agent_contexts(cases, verbose=verbose)

    eval_context = EvalRunContext()
    for ctx in contexts:
        eval_context.add_context(ctx)

    return eval_context
