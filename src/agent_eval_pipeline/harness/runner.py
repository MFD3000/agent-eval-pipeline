"""
Phase 8: Unified Eval Harness

This orchestrates all eval gates into a single pipeline.
It's what CI calls to run the complete evaluation.

ORCHESTRATION ORDER:
--------------------
1. Schema Eval (fast, deterministic) - fail fast on structure
2. Retrieval Eval (no LLM calls) - check RAG quality
3. Judge Eval (expensive) - semantic quality
4. Perf Eval (already ran agent) - latency/cost

This order is intentional:
- Fast, cheap checks first
- Expensive LLM-as-judge only if structure passes
- Perf eval uses cached results where possible

REPORTING:
----------
The harness produces a structured report that includes:
- Pass/fail for each gate
- Detailed results per case
- Aggregate metrics
- Actionable failure messages

CI INTEGRATION:
---------------
- Exit code 0 = all gates passed
- Exit code 1 = one or more gates failed
- Structured JSON output for CI parsing

The eval harness runs gates in order of cost: schema validation first
because it's instant, then retrieval, then LLM-as-judge which is expensive.
If early gates fail, we skip later ones to save compute. The harness produces
structured reports that CI can parse to show inline PR comments with
exactly what failed and why.
"""

import json
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any

from agent_eval_pipeline.golden_sets import get_all_golden_cases
from agent_eval_pipeline.observability import (
    init_phoenix,
    get_tracer,
    EVAL_HARNESS_RUN_ID,
    EVAL_GATE_NAME,
    EVAL_GATE_STATUS,
    eval_gate_attributes,
)
from agent_eval_pipeline.evals.schema_eval import run_schema_eval, SchemaEvalReport
from agent_eval_pipeline.evals.retrieval_eval import run_retrieval_eval, RetrievalEvalReport
from agent_eval_pipeline.evals.judge_eval import run_judge_eval, JudgeEvalReport
from agent_eval_pipeline.evals.perf_eval import run_perf_eval, PerfEvalResult
from agent_eval_pipeline.harness.context import (
    AgentRunContext,
    EvalRunContext,
    build_agent_contexts,
)


class GateStatus(Enum):
    """Status of an eval gate."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class GateResult:
    """Result of running a single gate."""
    name: str
    status: GateStatus
    duration_ms: float
    summary: str
    details: dict[str, Any] | None = None


@dataclass
class HarnessReport:
    """Complete evaluation report."""
    timestamp: str
    total_duration_ms: float
    all_passed: bool
    gates: list[GateResult]
    summary: str

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_duration_ms": self.total_duration_ms,
            "all_passed": self.all_passed,
            "gates": [
                {
                    "name": g.name,
                    "status": g.status.value,
                    "duration_ms": g.duration_ms,
                    "summary": g.summary,
                }
                for g in self.gates
            ],
            "summary": self.summary,
        }


def run_gate(
    name: str,
    runner: callable,
    skip: bool = False,
    verbose: bool = False,
) -> tuple[GateResult, Any]:
    """
    Run a single eval gate with timing and error handling.

    Returns:
        Tuple of (GateResult, raw_report)
    """
    tracer = get_tracer()

    if skip:
        return GateResult(
            name=name,
            status=GateStatus.SKIPPED,
            duration_ms=0,
            summary="Skipped due to prior failure",
        ), None

    if verbose:
        print(f"\n{'='*60}")
        print(f"RUNNING: {name}")
        print(f"{'='*60}")

    start = time.time()

    # Create span for this gate
    gate_attrs = {EVAL_GATE_NAME: name}
    with tracer.start_span(f"eval_gate.{name.lower().replace(' ', '_')}", attributes=gate_attrs) as span:
        try:
            report = runner()
            duration_ms = (time.time() - start) * 1000

            # Determine pass/fail based on report type
            if hasattr(report, 'all_passed'):
                passed = report.all_passed
            elif hasattr(report, 'passed'):
                passed = report.passed
            else:
                passed = True  # Default to passed if unknown

            status = GateStatus.PASSED if passed else GateStatus.FAILED

            # Generate summary
            if hasattr(report, 'pass_rate'):
                summary = f"Pass rate: {report.pass_rate:.1%}"
                span.set_attribute("eval.gate.pass_rate", report.pass_rate)
            elif hasattr(report, 'avg_score'):
                summary = f"Avg score: {report.avg_score:.2f}/5"
                span.set_attribute("eval.gate.avg_score", report.avg_score)
            elif hasattr(report, 'avg_f1'):
                summary = f"Avg F1: {report.avg_f1:.2f}"
                span.set_attribute("eval.gate.avg_f1", report.avg_f1)
            else:
                summary = "Completed"

            # Update span with results
            span.set_attribute(EVAL_GATE_STATUS, status.value)
            span.set_status("ok" if passed else "error", summary)

            if verbose:
                print(f"\n>>> {name}: {'PASSED' if passed else 'FAILED'} ({summary})")

            return GateResult(
                name=name,
                status=status,
                duration_ms=duration_ms,
                summary=summary,
            ), report

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            span.set_attribute(EVAL_GATE_STATUS, "error")
            span.record_exception(e)
            span.set_status("error", str(e))

            if verbose:
                print(f"\n>>> {name}: ERROR - {e}")

            return GateResult(
                name=name,
                status=GateStatus.ERROR,
                duration_ms=duration_ms,
                summary=f"Error: {str(e)[:100]}",
            ), None


def run_all_evals(
    fail_fast: bool = True,
    skip_expensive: bool = False,
    verbose: bool = True,
    contexts: list[AgentRunContext] | None = None,
) -> HarnessReport:
    """
    Run all evaluation gates.

    Args:
        fail_fast: Stop after first failure
        skip_expensive: Skip LLM-as-judge (for quick local testing)
        verbose: Print progress
        contexts: Pre-computed agent results. If None, runs agents once upfront.

    Returns:
        HarnessReport with all results

    CONTEXT SHARING:
    ----------------
    This harness runs agents ONCE and shares results across all evaluators.
    This provides:
    - 4-5x reduction in LLM API calls
    - Consistent evaluation (all gates score the same output)
    - Real retrieval validation (not simulated)
    """
    tracer = get_tracer()
    run_id = str(uuid.uuid4())

    # Create root span for entire eval run
    root_attrs = {
        EVAL_HARNESS_RUN_ID: run_id,
        "eval.harness.fail_fast": fail_fast,
        "eval.harness.skip_expensive": skip_expensive,
    }

    with tracer.start_span("eval_harness_run", attributes=root_attrs) as root_span:
        start_time = time.time()
        gates: list[GateResult] = []
        has_failure = False

        # Run agents ONCE if contexts not provided
        if contexts is None:
            cases = get_all_golden_cases()
            if verbose:
                print("\n" + "=" * 60)
                print("RUNNING AGENTS (once, results shared across all gates)")
                print("=" * 60)
            contexts = build_agent_contexts(cases, verbose=verbose)

            # Log agent run stats
            success_count = sum(1 for ctx in contexts if ctx.success)
            root_span.set_attribute("eval.harness.agent_runs", len(contexts))
            root_span.set_attribute("eval.harness.agent_success_count", success_count)

            if verbose:
                print(f"\nAgent runs: {success_count}/{len(contexts)} successful")

        # Gate 1: Schema Eval (uses contexts)
        schema_result, _ = run_gate(
            name="Schema Validation",
            runner=lambda: run_schema_eval(contexts=contexts),
            skip=False,
            verbose=verbose,
        )
        gates.append(schema_result)
        if schema_result.status != GateStatus.PASSED:
            has_failure = True

        # Gate 2: Retrieval Eval (uses contexts for actual retrieval validation)
        retrieval_result, _ = run_gate(
            name="Retrieval Quality",
            runner=lambda: run_retrieval_eval(contexts=contexts, use_actual_retrieval=True),
            skip=fail_fast and has_failure,
            verbose=verbose,
        )
        gates.append(retrieval_result)
        if retrieval_result.status == GateStatus.FAILED:
            has_failure = True

        # Gate 3: Judge Eval (expensive, uses contexts)
        judge_result, _ = run_gate(
            name="LLM-as-Judge",
            runner=lambda: run_judge_eval(contexts=contexts),
            skip=(fail_fast and has_failure) or skip_expensive,
            verbose=verbose,
        )
        gates.append(judge_result)
        if judge_result.status == GateStatus.FAILED:
            has_failure = True

        # Gate 4: Performance Eval (uses contexts)
        perf_result, _ = run_gate(
            name="Performance Regression",
            runner=lambda: run_perf_eval(contexts=contexts),
            skip=fail_fast and has_failure,
            verbose=verbose,
        )
        gates.append(perf_result)
        if perf_result.status == GateStatus.FAILED:
            has_failure = True

        # Calculate totals
        total_duration = (time.time() - start_time) * 1000
        all_passed = all(
            g.status in (GateStatus.PASSED, GateStatus.SKIPPED)
            for g in gates
        )

        # Generate summary
        passed_count = sum(1 for g in gates if g.status == GateStatus.PASSED)
        failed_count = sum(1 for g in gates if g.status == GateStatus.FAILED)
        summary = f"{passed_count} passed, {failed_count} failed"

        # Update root span with final results
        root_span.set_attribute("eval.harness.all_passed", all_passed)
        root_span.set_attribute("eval.harness.passed_count", passed_count)
        root_span.set_attribute("eval.harness.failed_count", failed_count)
        root_span.set_attribute("eval.harness.duration_ms", total_duration)
        root_span.set_status("ok" if all_passed else "error", summary)

        return HarnessReport(
            timestamp=datetime.now().isoformat(),
            total_duration_ms=total_duration,
            all_passed=all_passed,
            gates=gates,
            summary=summary,
        )


def print_report(report: HarnessReport) -> None:
    """Print a formatted report to stdout."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    for gate in report.gates:
        status_icon = {
            GateStatus.PASSED: "[PASS]",
            GateStatus.FAILED: "[FAIL]",
            GateStatus.SKIPPED: "[SKIP]",
            GateStatus.ERROR: "[ERR!]",
        }[gate.status]

        print(f"  {status_icon} {gate.name}: {gate.summary} ({gate.duration_ms:.0f}ms)")

    print("-" * 60)
    print(f"Total: {report.summary}")
    print(f"Duration: {report.total_duration_ms:.0f}ms")

    if report.all_passed:
        print("\n" + "=" * 60)
        print(">>> ALL EVAL GATES PASSED <<<")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(">>> EVAL GATES FAILED <<<")
        print("=" * 60)


def main():
    """CLI entry point."""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize Phoenix observability (if enabled)
    init_phoenix()

    parser = argparse.ArgumentParser(description="Run evaluation gates")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=True,
        help="Stop after first failure (default: true)",
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Run all gates regardless of failures",
    )
    parser.add_argument(
        "--skip-expensive",
        action="store_true",
        help="Skip LLM-as-judge eval (for quick testing)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report instead of formatted text",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    fail_fast = not args.no_fail_fast
    verbose = not args.quiet

    report = run_all_evals(
        fail_fast=fail_fast,
        skip_expensive=args.skip_expensive,
        verbose=verbose,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)

    # Exit code for CI
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
