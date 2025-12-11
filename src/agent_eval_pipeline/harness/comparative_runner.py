"""
Comparative Evaluation Runner

Runs the same golden cases through multiple agents and compares results.
This enables data-driven decisions about which agent performs better.

COMPARISON METRICS:
-------------------
- Schema Validation Pass Rate
- LLM-as-Judge Scores (per dimension)
- Latency (p50, p95)
- Token Usage (cost proxy)
- Agent-specific metrics (tools used, retrieval docs, etc.)

INTERVIEW TALKING POINT:
------------------------
"We run comparative evals to make data-driven agent selection decisions.
Both agents produce the same LabInsightsSummary schema, so we can fairly
compare them on quality metrics from LLM-as-judge, latency, and cost.
LangGraph has RAG retrieval, DSPy ReAct has tool use - different approaches
to the same problem."
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Literal

from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase, get_all_golden_cases
from agent_eval_pipeline.agent import run_agent, AgentResult, AgentError, AgentType
from agent_eval_pipeline.evals.schema_eval import validate_case_output


@dataclass
class AgentCaseResult:
    """Result of running one agent on one case."""
    case_id: str
    agent_type: str
    success: bool
    error: str | None = None

    # Metrics (only if success)
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # Schema validation
    schema_passed: bool | None = None
    schema_errors: list[str] | None = None

    # Agent-specific
    tools_used: list[str] | None = None
    reasoning_steps: int | None = None
    retrieved_docs: list[dict] | None = None


@dataclass
class AgentSummary:
    """Summary statistics for one agent across all cases."""
    agent_type: str
    total_cases: int
    successful_cases: int
    failed_cases: int
    schema_pass_rate: float

    # Latency stats
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float

    # Token stats
    avg_tokens: float
    total_tokens: int

    # Agent-specific
    avg_tools_used: float | None = None  # DSPy ReAct
    avg_docs_retrieved: float | None = None  # LangGraph


@dataclass
class ComparativeReport:
    """Full comparative evaluation report."""
    timestamp: str
    cases_evaluated: int
    agents: list[str]
    summaries: dict[str, AgentSummary]
    case_results: list[AgentCaseResult]
    winner: str | None = None
    recommendation: str = ""

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "cases_evaluated": self.cases_evaluated,
            "agents": self.agents,
            "summaries": {k: asdict(v) for k, v in self.summaries.items()},
            "winner": self.winner,
            "recommendation": self.recommendation,
        }


def run_agent_on_case(
    case: GoldenCase,
    agent_type: AgentType,
) -> AgentCaseResult:
    """Run a single agent on a single case."""
    result = run_agent(case, agent_type=agent_type)

    if isinstance(result, AgentError):
        return AgentCaseResult(
            case_id=case.id,
            agent_type=agent_type,
            success=False,
            error=f"{result.error_type}: {result.error_message}",
        )

    # Validate schema
    schema_result = validate_case_output(case, result.output)
    schema_errors = []
    if schema_result.error:
        schema_errors.append(schema_result.error)
    if schema_result.invalid_values:
        schema_errors.extend([f"{k}: {v}" for k, v in schema_result.invalid_values.items()])

    return AgentCaseResult(
        case_id=case.id,
        agent_type=agent_type,
        success=True,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        schema_passed=schema_result.passed,
        schema_errors=schema_errors if schema_errors else None,
        tools_used=result.tools_used,
        reasoning_steps=result.reasoning_steps,
        retrieved_docs=result.retrieved_docs,
    )


def calculate_summary(
    agent_type: str,
    results: list[AgentCaseResult],
) -> AgentSummary:
    """Calculate summary statistics for an agent."""
    successful = [r for r in results if r.success]

    latencies = [r.latency_ms for r in successful if r.latency_ms]
    latencies.sort()

    tokens = [r.total_tokens for r in successful if r.total_tokens]

    schema_passed = sum(1 for r in successful if r.schema_passed)

    # Agent-specific metrics
    avg_tools = None
    avg_docs = None

    if agent_type == "dspy_react":
        tools_counts = [len(r.tools_used) for r in successful if r.tools_used is not None]
        if tools_counts:
            avg_tools = sum(tools_counts) / len(tools_counts)

    if agent_type == "langgraph":
        doc_counts = [len(r.retrieved_docs) for r in successful if r.retrieved_docs is not None]
        if doc_counts:
            avg_docs = sum(doc_counts) / len(doc_counts)

    return AgentSummary(
        agent_type=agent_type,
        total_cases=len(results),
        successful_cases=len(successful),
        failed_cases=len(results) - len(successful),
        schema_pass_rate=schema_passed / len(successful) if successful else 0.0,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p50_latency_ms=latencies[len(latencies) // 2] if latencies else 0.0,
        p95_latency_ms=latencies[int(len(latencies) * 0.95)] if latencies else 0.0,
        avg_tokens=sum(tokens) / len(tokens) if tokens else 0.0,
        total_tokens=sum(tokens),
        avg_tools_used=avg_tools,
        avg_docs_retrieved=avg_docs,
    )


def determine_winner(summaries: dict[str, AgentSummary]) -> tuple[str | None, str]:
    """Determine winner based on metrics."""
    if len(summaries) < 2:
        return None, "Need at least 2 agents to compare"

    agents = list(summaries.keys())
    s1, s2 = summaries[agents[0]], summaries[agents[1]]

    # Scoring: higher is better for schema pass rate, lower is better for latency/tokens
    score = {agents[0]: 0, agents[1]: 0}
    reasons = []

    # Schema pass rate (most important)
    if s1.schema_pass_rate > s2.schema_pass_rate:
        score[agents[0]] += 2
        reasons.append(f"{agents[0]} has higher schema pass rate ({s1.schema_pass_rate:.0%} vs {s2.schema_pass_rate:.0%})")
    elif s2.schema_pass_rate > s1.schema_pass_rate:
        score[agents[1]] += 2
        reasons.append(f"{agents[1]} has higher schema pass rate ({s2.schema_pass_rate:.0%} vs {s1.schema_pass_rate:.0%})")

    # Latency
    if s1.avg_latency_ms < s2.avg_latency_ms * 0.9:  # 10% threshold
        score[agents[0]] += 1
        reasons.append(f"{agents[0]} is faster ({s1.avg_latency_ms:.0f}ms vs {s2.avg_latency_ms:.0f}ms)")
    elif s2.avg_latency_ms < s1.avg_latency_ms * 0.9:
        score[agents[1]] += 1
        reasons.append(f"{agents[1]} is faster ({s2.avg_latency_ms:.0f}ms vs {s1.avg_latency_ms:.0f}ms)")

    # Token usage (cost)
    if s1.avg_tokens < s2.avg_tokens * 0.9:
        score[agents[0]] += 1
        reasons.append(f"{agents[0]} uses fewer tokens ({s1.avg_tokens:.0f} vs {s2.avg_tokens:.0f})")
    elif s2.avg_tokens < s1.avg_tokens * 0.9:
        score[agents[1]] += 1
        reasons.append(f"{agents[1]} uses fewer tokens ({s2.avg_tokens:.0f} vs {s1.avg_tokens:.0f})")

    winner = max(score, key=score.get) if score[agents[0]] != score[agents[1]] else None
    recommendation = "; ".join(reasons) if reasons else "Agents performed similarly"

    return winner, recommendation


def run_comparative_eval(
    agents: list[AgentType] = ["langgraph", "dspy_react"],
    cases: list[GoldenCase] | None = None,
    verbose: bool = True,
) -> ComparativeReport:
    """
    Run comparative evaluation across multiple agents.

    Args:
        agents: List of agent types to compare
        cases: Golden cases to evaluate (defaults to all)
        verbose: Print progress

    Returns:
        ComparativeReport with full comparison data
    """
    cases = cases or get_all_golden_cases()
    all_results: list[AgentCaseResult] = []

    if verbose:
        print("=" * 60)
        print("COMPARATIVE EVALUATION")
        print("=" * 60)
        print(f"Agents: {', '.join(agents)}")
        print(f"Cases: {len(cases)}")
        print("=" * 60)

    for agent_type in agents:
        if verbose:
            print(f"\n>>> Running {agent_type} agent...")

        for case in cases:
            if verbose:
                print(f"  {case.id}...", end=" ", flush=True)

            result = run_agent_on_case(case, agent_type)
            all_results.append(result)

            if verbose:
                status = "✓" if result.success and result.schema_passed else "✗"
                latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
                print(f"{status} ({latency})")

    # Calculate summaries
    summaries = {}
    for agent_type in agents:
        agent_results = [r for r in all_results if r.agent_type == agent_type]
        summaries[agent_type] = calculate_summary(agent_type, agent_results)

    # Determine winner
    winner, recommendation = determine_winner(summaries)

    report = ComparativeReport(
        timestamp=datetime.now().isoformat(),
        cases_evaluated=len(cases),
        agents=agents,
        summaries=summaries,
        case_results=all_results,
        winner=winner,
        recommendation=recommendation,
    )

    if verbose:
        print_comparative_report(report)

    return report


def print_comparative_report(report: ComparativeReport) -> None:
    """Print formatted comparative report."""
    print("\n" + "=" * 60)
    print("COMPARATIVE RESULTS")
    print("=" * 60)

    # Summary table
    print(f"\n{'Metric':<25} ", end="")
    for agent in report.agents:
        print(f"{agent:>15} ", end="")
    print()
    print("-" * (25 + 16 * len(report.agents)))

    metrics = [
        ("Success Rate", lambda s: f"{s.successful_cases}/{s.total_cases}"),
        ("Schema Pass Rate", lambda s: f"{s.schema_pass_rate:.0%}"),
        ("Avg Latency", lambda s: f"{s.avg_latency_ms:.0f}ms"),
        ("P95 Latency", lambda s: f"{s.p95_latency_ms:.0f}ms"),
        ("Avg Tokens", lambda s: f"{s.avg_tokens:.0f}"),
        ("Total Tokens", lambda s: f"{s.total_tokens}"),
    ]

    for metric_name, metric_fn in metrics:
        print(f"{metric_name:<25} ", end="")
        for agent in report.agents:
            print(f"{metric_fn(report.summaries[agent]):>15} ", end="")
        print()

    # Agent-specific metrics
    if "langgraph" in report.summaries:
        s = report.summaries["langgraph"]
        if s.avg_docs_retrieved is not None:
            print(f"{'Avg Docs Retrieved':<25} {s.avg_docs_retrieved:>15.1f}")

    if "dspy_react" in report.summaries:
        s = report.summaries["dspy_react"]
        if s.avg_tools_used is not None:
            print(f"{'Avg Tools Used':<25} {'':>15} {s.avg_tools_used:>15.1f}")

    # Winner
    print("\n" + "-" * 60)
    if report.winner:
        print(f"WINNER: {report.winner}")
    else:
        print("WINNER: Tie / No clear winner")
    print(f"Recommendation: {report.recommendation}")
    print("=" * 60)


def main():
    """CLI entry point."""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run comparative agent evaluation")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["langgraph", "dspy_react"],
        help="Agents to compare",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of formatted text",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    report = run_comparative_eval(
        agents=args.agents,
        verbose=not args.quiet,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
