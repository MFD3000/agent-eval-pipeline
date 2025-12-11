"""
RAGAS Integration Module

RAGAS (Retrieval Augmented Generation Assessment) is a framework specialized
for evaluating RAG pipelines. It excels at measuring:
- Faithfulness: Does the response stick to retrieved context?
- Context Precision: Are retrieved documents relevant to the question?
- Context Recall: Are all necessary documents retrieved?
- Answer Relevancy: Does the answer address the question?

INTERVIEW TALKING POINT:
------------------------
"RAGAS is specifically designed for RAG evaluation. While DeepEval gives us
general LLM testing, RAGAS focuses on the retrieval-generation interplay.
Its faithfulness metric catches when the LLM 'hallucinates' beyond the
retrieved context. Context precision/recall tells us if our vector search
is working properly."

WHY RAGAS:
----------
1. RAG-specific - metrics designed for retrieval + generation
2. Reference-free options - can evaluate without ground truth
3. Claim-level analysis - breaks response into verifiable claims
4. Non-LLM metrics - faster evaluation for some metrics
5. Dataset-based - efficient batch evaluation

RAGAS vs DEEPEVAL for RAG:
--------------------------
- RAGAS: More sophisticated claim extraction, specialized for RAG
- DeepEval: Broader coverage, pytest integration, custom metrics
- Use both: RAGAS for deep RAG analysis, DeepEval for CI gates
"""

from agent_eval_pipeline.evals.ragas.adapters import (
    golden_case_to_ragas_sample,
    create_ragas_dataset,
    agent_result_to_ragas_sample,
)

from agent_eval_pipeline.evals.ragas.metrics import (
    get_ragas_metrics,
    get_faithfulness_metric,
    get_context_metrics,
    RAGAS_METRIC_NAMES,
)

from agent_eval_pipeline.evals.ragas.evaluator import (
    run_ragas_evaluation,
    RagasResult,
    RagasReport,
)

__all__ = [
    # Adapters
    "golden_case_to_ragas_sample",
    "create_ragas_dataset",
    "agent_result_to_ragas_sample",
    # Metrics
    "get_ragas_metrics",
    "get_faithfulness_metric",
    "get_context_metrics",
    "RAGAS_METRIC_NAMES",
    # Evaluator
    "run_ragas_evaluation",
    "RagasResult",
    "RagasReport",
]
