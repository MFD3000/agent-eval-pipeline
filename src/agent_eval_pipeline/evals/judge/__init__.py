"""
Judge evaluation module - LLM-as-judge for semantic quality.

ELEVATED ARCHITECTURE:
----------------------
- prompts.py: Externalized prompts (testable, versionable)
- schemas.py: Data models for judge I/O
- evaluator.py: Core evaluation logic with DI
"""

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
from agent_eval_pipeline.evals.judge.evaluator import (
    run_judge,
    calculate_weighted_score,
    run_judge_eval,
)

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
]
