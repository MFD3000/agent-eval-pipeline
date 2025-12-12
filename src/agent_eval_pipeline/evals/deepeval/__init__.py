"""
DeepEval Integration Module

DeepEval is an open-source LLM evaluation framework that provides:
- LLMTestCase structure for standardized test cases
- Built-in RAG metrics (Faithfulness, Hallucination, Context P/R)
- G-Eval for custom LLM-as-judge metrics
- Native pytest integration with assert_test


WHY DEEPEVAL:
-------------
1. Pytest integration - runs with `deepeval test run` or regular pytest
2. G-Eval - custom metrics using LLM-as-judge with evaluation steps
3. RAG metrics - faithfulness, hallucination, context precision/recall
4. Parallel execution - `deepeval test run -n 4` for speed
5. CI/CD ready - exit codes, JSON output, thresholds
"""

from agent_eval_pipeline.evals.deepeval.adapters import (
    golden_case_to_llm_test_case,
    golden_cases_to_dataset,
    agent_result_to_test_case,
)

from agent_eval_pipeline.evals.deepeval.metrics import (
    get_clinical_correctness,
    get_safety_compliance,
    get_completeness,
    get_answer_clarity,
    get_healthcare_metrics,
    get_rag_metrics,
    get_all_metrics,
)

from agent_eval_pipeline.evals.deepeval.evaluator import (
    run_deepeval_evaluation,
    DeepEvalResult,
    DeepEvalReport,
)

__all__ = [
    # Adapters
    "golden_case_to_llm_test_case",
    "golden_cases_to_dataset",
    "agent_result_to_test_case",
    # Metrics (lazy-loaded via functions)
    "get_clinical_correctness",
    "get_safety_compliance",
    "get_completeness",
    "get_answer_clarity",
    "get_healthcare_metrics",
    "get_rag_metrics",
    "get_all_metrics",
    # Evaluator
    "run_deepeval_evaluation",
    "DeepEvalResult",
    "DeepEvalReport",
]
