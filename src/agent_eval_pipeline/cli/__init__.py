"""
CLI module - unified command-line interface.

Provides entry points for:
- Running individual evaluation gates
- Running the full eval harness
- Managing baselines
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
