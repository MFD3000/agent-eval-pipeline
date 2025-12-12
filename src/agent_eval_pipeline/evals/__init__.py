"""
Evaluation gates module - comprehensive eval pipeline.

ELEVATED ARCHITECTURE:
----------------------
The evals module provides multiple evaluation gates:
- schema_eval: Schema validation (fast, deterministic)
- retrieval_eval: Retrieval quality (precision, recall)
- judge_eval: LLM-as-judge semantic evaluation
- perf_eval: Performance regression detection
- deepeval: DeepEval G-Eval metrics with pytest integration
- ragas: RAGAS RAG-specialized metrics

Each elevated module follows the gold standard pattern:
- Protocol defines interface
- Production implementation
- Test double for fast testing
- Factory function for convenience

MULTI-FRAMEWORK EVALUATION:
---------------------------
We support multiple evaluation frameworks to catch different types of issues:
- DeepEval: Pytest-native, custom G-Eval metrics, CI/CD ready
- RAGAS: RAG-specialized metrics (faithfulness, context precision/recall)
- Custom: Domain-specific LLM-as-judge with healthcare rubrics
- DSPy: Optimizable judge that can be tuned with examples
"""

# Judge evaluation
from agent_eval_pipeline.evals.judge import (
    DimensionScore,
    JudgeOutput,
    JudgeEvalResult,
    JudgeEvalReport,
    WEIGHTS,
    JUDGE_SYSTEM_PROMPT,
    format_judge_user_prompt,
    run_judge,
    calculate_weighted_score,
    run_judge_eval,
)

# Performance evaluation
from agent_eval_pipeline.evals.perf import (
    BaselineStore,
    PerformanceBaseline,
    FileBaselineStore,
    InMemoryBaselineStore,
    get_baseline_store,
    MODEL_PRICING,
    estimate_cost,
    CasePerformance,
    PerformanceMetrics,
    RegressionCheck,
    PerfEvalResult,
    run_perf_eval,
)

# CLI entry points (from backward compat modules)
from agent_eval_pipeline.evals.judge_eval import run_judge_eval_cli
from agent_eval_pipeline.evals.perf_eval import run_perf_eval_cli

# DeepEval integration
from agent_eval_pipeline.evals.deepeval import (
    golden_case_to_llm_test_case,
    golden_cases_to_dataset,
    get_clinical_correctness,
    get_safety_compliance,
    get_completeness,
    get_answer_clarity,
    get_healthcare_metrics,
    run_deepeval_evaluation,
    DeepEvalResult,
    DeepEvalReport,
)

# RAGAS integration
from agent_eval_pipeline.evals.ragas import (
    golden_case_to_ragas_sample,
    create_ragas_dataset,
    get_ragas_metrics,
    run_ragas_evaluation,
    RagasResult,
    RagasReport,
)

__all__ = [
    # Judge
    "DimensionScore",
    "JudgeOutput",
    "JudgeEvalResult",
    "JudgeEvalReport",
    "WEIGHTS",
    "JUDGE_SYSTEM_PROMPT",
    "format_judge_user_prompt",
    "run_judge",
    "calculate_weighted_score",
    "run_judge_eval",
    "run_judge_eval_cli",
    # Perf
    "BaselineStore",
    "PerformanceBaseline",
    "FileBaselineStore",
    "InMemoryBaselineStore",
    "get_baseline_store",
    "MODEL_PRICING",
    "estimate_cost",
    "CasePerformance",
    "PerformanceMetrics",
    "RegressionCheck",
    "PerfEvalResult",
    "run_perf_eval",
    "run_perf_eval_cli",
    # DeepEval
    "golden_case_to_llm_test_case",
    "golden_cases_to_dataset",
    "get_clinical_correctness",
    "get_safety_compliance",
    "get_completeness",
    "get_answer_clarity",
    "get_healthcare_metrics",
    "run_deepeval_evaluation",
    "DeepEvalResult",
    "DeepEvalReport",
    # RAGAS
    "golden_case_to_ragas_sample",
    "create_ragas_dataset",
    "get_ragas_metrics",
    "run_ragas_evaluation",
    "RagasResult",
    "RagasReport",
]
