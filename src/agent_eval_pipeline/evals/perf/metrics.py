"""
Performance metrics - data models for performance evaluation.

These dataclasses provide a clear schema for:
1. Per-case performance data
2. Aggregate metrics
3. Regression detection results

INTERVIEW TALKING POINT:
------------------------
"Metrics are plain dataclasses - no behavior, just data.
This makes them easy to serialize for reports, compare
across runs, and use in assertions. The RegressionCheck
dataclass captures everything needed to understand why
a regression was detected."
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# CASE-LEVEL METRICS
# ---------------------------------------------------------------------------


@dataclass
class CasePerformance:
    """Performance metrics for a single case.

    Captures latency, token usage, and model for one evaluation run.
    """

    case_id: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# AGGREGATE METRICS
# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics across all cases.

    Provides percentile latencies (p50, p95) and average token usage.
    """

    p50_latency_ms: float
    p95_latency_ms: float
    avg_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    total_cost_estimate: float
    models_used: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# REGRESSION DETECTION
# ---------------------------------------------------------------------------


@dataclass
class RegressionCheck:
    """Result of checking for regression against baseline.

    Captures the comparison details for debugging regressions.
    """

    metric: str
    baseline_value: float
    current_value: float
    change_percent: float
    threshold_percent: float
    is_regression: bool


# ---------------------------------------------------------------------------
# EVALUATION RESULT
# ---------------------------------------------------------------------------


@dataclass
class PerfEvalResult:
    """Result of performance evaluation.

    Contains everything needed for CI reporting:
    - Pass/fail status
    - Current metrics
    - Baseline comparison
    - Per-case details
    """

    passed: bool
    metrics: PerformanceMetrics
    baseline: "PerformanceBaseline | None"  # Forward ref to avoid circular import
    regression_checks: list[RegressionCheck]
    case_results: list[CasePerformance]
    failure_reason: str | None = None


# Import for type checking only (avoid circular import)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eval_pipeline.evals.perf.baseline import PerformanceBaseline
