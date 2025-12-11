"""
CLI module - unified command-line interface.

Provides entry points for:
- Running individual evaluation gates
- Running the full eval harness
- Managing baselines

INTERVIEW TALKING POINT:
------------------------
"The CLI module provides a clean interface for CI and local development.
Each command maps to a specific evaluation gate. The unified 'eval-all'
command runs the entire pipeline with configurable fail-fast behavior."
"""

from agent_eval_pipeline.cli.commands import (
    main,
    run_schema_cli,
    run_retrieval_cli,
    run_judge_cli,
    run_perf_cli,
    run_all_cli,
)

__all__ = [
    "main",
    "run_schema_cli",
    "run_retrieval_cli",
    "run_judge_cli",
    "run_perf_cli",
    "run_all_cli",
]
