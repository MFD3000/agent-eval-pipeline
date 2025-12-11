"""
Performance evaluation module - latency and cost regression detection.

ELEVATED ARCHITECTURE:
----------------------
- baseline.py: BaselineStore protocol + implementations (File, InMemory)
- pricing.py: Cost estimation logic
- metrics.py: Data models for performance metrics
- evaluator.py: Core evaluation logic with DI

INTERVIEW TALKING POINT:
------------------------
"The perf eval module uses the adapter pattern for baseline storage.
In production, we use FileBaselineStore to persist between runs.
In tests, InMemoryBaselineStore lets me verify regression detection
logic without touching the filesystem. The evaluator accepts the
store via constructor injection."
"""

from agent_eval_pipeline.evals.perf.baseline import (
    BaselineStore,
    PerformanceBaseline,
    FileBaselineStore,
    InMemoryBaselineStore,
    get_baseline_store,
)
from agent_eval_pipeline.evals.perf.pricing import (
    MODEL_PRICING,
    estimate_cost,
)
from agent_eval_pipeline.evals.perf.metrics import (
    CasePerformance,
    PerformanceMetrics,
    RegressionCheck,
    PerfEvalResult,
)
from agent_eval_pipeline.evals.perf.evaluator import (
    run_perf_eval,
)

__all__ = [
    # Baseline
    "BaselineStore",
    "PerformanceBaseline",
    "FileBaselineStore",
    "InMemoryBaselineStore",
    "get_baseline_store",
    # Pricing
    "MODEL_PRICING",
    "estimate_cost",
    # Metrics
    "CasePerformance",
    "PerformanceMetrics",
    "RegressionCheck",
    "PerfEvalResult",
    # Evaluator
    "run_perf_eval",
]
