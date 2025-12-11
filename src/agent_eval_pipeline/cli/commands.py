"""
CLI commands - entry points for evaluation gates.

Each command follows a consistent pattern:
1. Parse arguments
2. Load environment
3. Run evaluation
4. Print results
5. Return exit code

INTERVIEW TALKING POINT:
------------------------
"CLI commands are thin wrappers around the evaluation functions.
They handle argument parsing and output formatting, but delegate
the actual work to the elevated modules. This keeps the CLI simple
and the business logic testable."
"""

from __future__ import annotations

import argparse
import json
import sys


def _load_env() -> None:
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # dotenv is optional


def run_schema_cli() -> int:
    """CLI entry point for schema evaluation."""
    from agent_eval_pipeline.evals.schema_eval import run_schema_eval

    _load_env()

    parser = argparse.ArgumentParser(description="Run schema validation eval")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    print("=" * 60)
    print("SCHEMA VALIDATION EVAL")
    print("=" * 60)

    report = run_schema_eval()

    if not args.quiet:
        for result in report.results:
            status = "PASS" if result.is_valid else "FAIL"
            print(f"  [{status}] {result.case_id}")
            if not result.is_valid:
                for error in result.errors[:3]:
                    print(f"        Error: {error}")

    print(f"\nPass rate: {report.pass_rate:.1%}")
    print(f"Total: {report.passed}/{report.total}")

    if report.all_passed:
        print("\n>>> SCHEMA EVAL GATE: PASSED <<<")
        return 0
    else:
        print("\n>>> SCHEMA EVAL GATE: FAILED <<<")
        return 1


def run_retrieval_cli() -> int:
    """CLI entry point for retrieval evaluation."""
    from agent_eval_pipeline.evals.retrieval_eval import run_retrieval_eval

    _load_env()

    parser = argparse.ArgumentParser(description="Run retrieval quality eval")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    print("=" * 60)
    print("RETRIEVAL QUALITY EVAL")
    print("=" * 60)

    report = run_retrieval_eval()

    if not args.quiet:
        for result in report.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.query} (F1: {result.f1:.2f})")

    print(f"\nAverage F1: {report.avg_f1:.2f}")
    print(f"Threshold: {report.threshold}")
    print(f"Passed: {report.passed}/{report.total}")

    if report.all_passed:
        print("\n>>> RETRIEVAL EVAL GATE: PASSED <<<")
        return 0
    else:
        print("\n>>> RETRIEVAL EVAL GATE: FAILED <<<")
        return 1


def run_judge_cli() -> int:
    """CLI entry point for LLM-as-judge evaluation."""
    from agent_eval_pipeline.evals.judge_eval import run_judge_eval_cli

    _load_env()
    return run_judge_eval_cli()


def run_perf_cli() -> int:
    """CLI entry point for performance evaluation."""
    from agent_eval_pipeline.evals.perf_eval import run_perf_eval_cli

    _load_env()

    parser = argparse.ArgumentParser(description="Run performance regression eval")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline with current results",
    )
    args, _ = parser.parse_known_args()

    return run_perf_eval_cli(update_baseline=args.update_baseline)


def run_all_cli() -> int:
    """CLI entry point for full eval harness."""
    from agent_eval_pipeline.harness.runner import main as harness_main

    _load_env()
    harness_main()
    return 0  # harness_main calls sys.exit


def main() -> int:
    """
    Main CLI entry point with subcommands.

    Usage:
        agent-eval schema    # Run schema validation
        agent-eval retrieval # Run retrieval quality
        agent-eval judge     # Run LLM-as-judge
        agent-eval perf      # Run performance eval
        agent-eval all       # Run all gates
    """
    _load_env()

    parser = argparse.ArgumentParser(
        description="Agent evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  schema      Run schema validation eval (fast, deterministic)
  retrieval   Run retrieval quality eval (no LLM calls)
  judge       Run LLM-as-judge semantic eval
  perf        Run performance regression eval
  all         Run all evaluation gates

Examples:
  agent-eval schema           # Quick structural check
  agent-eval perf --update-baseline  # Update perf baseline
  agent-eval all --skip-expensive    # Skip judge eval
        """,
    )

    parser.add_argument(
        "command",
        choices=["schema", "retrieval", "judge", "perf", "all"],
        help="Evaluation to run",
    )

    # Parse just the command first
    args, remaining = parser.parse_known_args()

    # Dispatch to appropriate handler
    commands = {
        "schema": run_schema_cli,
        "retrieval": run_retrieval_cli,
        "judge": run_judge_cli,
        "perf": run_perf_cli,
        "all": run_all_cli,
    }

    # Re-inject remaining args for the subcommand
    sys.argv = [sys.argv[0]] + remaining

    try:
        return commands[args.command]()
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
