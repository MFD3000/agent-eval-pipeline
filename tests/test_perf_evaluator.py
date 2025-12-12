"""
Unit Tests for Performance Evaluator

Tests the performance evaluation logic with dependency injection.
Uses mock baseline stores to test regression detection.

STAFF ENGINEER PATTERNS:
------------------------
1. Inject InMemoryBaselineStore to test regression logic
2. Mock run_agent to avoid actual LLM calls
3. Test edge cases (no successful cases, no baseline)
4. Verify threshold logic
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from agent_eval_pipeline.evals.perf.evaluator import (
    _calculate_metrics,
    _check_regressions,
    DEFAULT_LATENCY_THRESHOLD,
    DEFAULT_TOKEN_THRESHOLD,
)
from agent_eval_pipeline.evals.perf.metrics import (
    CasePerformance,
    PerformanceMetrics,
    RegressionCheck,
    PerfEvalResult,
)
from agent_eval_pipeline.evals.perf.baseline import (
    PerformanceBaseline,
    InMemoryBaselineStore,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_case_performances():
    """Create sample case performance results."""
    return [
        CasePerformance(
            case_id="case-001",
            latency_ms=1000.0,
            input_tokens=400,
            output_tokens=200,
            total_tokens=600,
            model="gpt-4o-mini",
            success=True,
        ),
        CasePerformance(
            case_id="case-002",
            latency_ms=1500.0,
            input_tokens=500,
            output_tokens=250,
            total_tokens=750,
            model="gpt-4o-mini",
            success=True,
        ),
        CasePerformance(
            case_id="case-003",
            latency_ms=2000.0,
            input_tokens=600,
            output_tokens=300,
            total_tokens=900,
            model="gpt-4o-mini",
            success=True,
        ),
    ]


@pytest.fixture
def sample_baseline():
    """Create a sample baseline."""
    return PerformanceBaseline(
        p50_latency_ms=1500.0,
        p95_latency_ms=2000.0,
        avg_input_tokens=500.0,
        avg_output_tokens=250.0,
        avg_total_tokens=750.0,
        expected_model="gpt-4o-mini",
        run_count=5,
    )


# ---------------------------------------------------------------------------
# CALCULATE METRICS TESTS
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    """Test the _calculate_metrics function."""

    def test_calculate_metrics_basic(self, sample_case_performances):
        """Should calculate correct aggregate metrics."""
        metrics = _calculate_metrics(sample_case_performances)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.p50_latency_ms == 1500.0  # Middle value
        assert metrics.p95_latency_ms == 2000.0  # 95th percentile
        assert metrics.avg_input_tokens == 500.0  # (400+500+600)/3
        assert metrics.avg_output_tokens == 250.0  # (200+250+300)/3
        assert metrics.avg_total_tokens == 750.0  # (600+750+900)/3

    def test_calculate_metrics_single_case(self):
        """Should handle single case correctly."""
        single_case = [
            CasePerformance(
                case_id="case-001",
                latency_ms=1000.0,
                input_tokens=500,
                output_tokens=200,
                total_tokens=700,
                model="gpt-4o-mini",
                success=True,
            )
        ]

        metrics = _calculate_metrics(single_case)

        assert metrics.p50_latency_ms == 1000.0
        assert metrics.p95_latency_ms == 1000.0
        assert metrics.avg_total_tokens == 700.0

    def test_calculate_metrics_tracks_models(self, sample_case_performances):
        """Should track which models were used."""
        metrics = _calculate_metrics(sample_case_performances)

        assert "gpt-4o-mini" in metrics.models_used


# ---------------------------------------------------------------------------
# CHECK REGRESSIONS TESTS
# ---------------------------------------------------------------------------


class TestCheckRegressions:
    """Test the _check_regressions function."""

    def test_no_regression_when_no_baseline(self):
        """Should return no regressions when baseline is None."""
        metrics = PerformanceMetrics(
            p50_latency_ms=1500.0,
            p95_latency_ms=2000.0,
            avg_latency_ms=1500.0,
            avg_input_tokens=500.0,
            avg_output_tokens=250.0,
            avg_total_tokens=750.0,
            total_cost_estimate=0.01,
        )

        checks, is_regression = _check_regressions(
            metrics=metrics,
            baseline=None,
            latency_threshold=0.15,
            token_threshold=0.20,
            primary_model="gpt-4o-mini",
        )

        assert checks == []
        assert is_regression is False

    def test_no_regression_when_within_threshold(self, sample_baseline):
        """Should pass when metrics are within threshold."""
        metrics = PerformanceMetrics(
            p50_latency_ms=1600.0,  # 6.7% increase, under 15%
            p95_latency_ms=2100.0,  # 5% increase, under 15%
            avg_latency_ms=1550.0,
            avg_input_tokens=520.0,
            avg_output_tokens=260.0,
            avg_total_tokens=780.0,  # 4% increase, under 20%
            total_cost_estimate=0.01,
        )

        checks, is_regression = _check_regressions(
            metrics=metrics,
            baseline=sample_baseline,
            latency_threshold=0.15,
            token_threshold=0.20,
            primary_model="gpt-4o-mini",
        )

        assert is_regression is False
        assert len(checks) == 2  # latency and tokens checked
        assert all(not c.is_regression for c in checks)

    def test_latency_regression_detected(self, sample_baseline):
        """Should detect latency regression when over threshold."""
        metrics = PerformanceMetrics(
            p50_latency_ms=1500.0,
            p95_latency_ms=2500.0,  # 25% increase, over 15%
            avg_latency_ms=2000.0,
            avg_input_tokens=500.0,
            avg_output_tokens=250.0,
            avg_total_tokens=750.0,
            total_cost_estimate=0.01,
        )

        checks, is_regression = _check_regressions(
            metrics=metrics,
            baseline=sample_baseline,
            latency_threshold=0.15,
            token_threshold=0.20,
            primary_model="gpt-4o-mini",
        )

        assert is_regression is True
        latency_check = next(c for c in checks if c.metric == "p95_latency_ms")
        assert latency_check.is_regression is True
        assert latency_check.change_percent == 25.0

    def test_token_regression_detected(self, sample_baseline):
        """Should detect token regression when over threshold."""
        metrics = PerformanceMetrics(
            p50_latency_ms=1500.0,
            p95_latency_ms=2000.0,
            avg_latency_ms=1750.0,
            avg_input_tokens=650.0,
            avg_output_tokens=350.0,
            avg_total_tokens=1000.0,  # 33% increase, over 20%
            total_cost_estimate=0.01,
        )

        checks, is_regression = _check_regressions(
            metrics=metrics,
            baseline=sample_baseline,
            latency_threshold=0.15,
            token_threshold=0.20,
            primary_model="gpt-4o-mini",
        )

        assert is_regression is True
        token_check = next(c for c in checks if c.metric == "avg_total_tokens")
        assert token_check.is_regression is True

    def test_model_change_regression(self, sample_baseline):
        """Should detect model change as regression."""
        metrics = PerformanceMetrics(
            p50_latency_ms=1500.0,
            p95_latency_ms=2000.0,
            avg_latency_ms=1750.0,
            avg_input_tokens=500.0,
            avg_output_tokens=250.0,
            avg_total_tokens=750.0,
            total_cost_estimate=0.01,
        )

        checks, is_regression = _check_regressions(
            metrics=metrics,
            baseline=sample_baseline,
            latency_threshold=0.15,
            token_threshold=0.20,
            primary_model="gpt-4",  # Different model!
        )

        assert is_regression is True
        model_check = next(c for c in checks if c.metric == "model")
        assert model_check.is_regression is True


# ---------------------------------------------------------------------------
# BASELINE STORE TESTS
# ---------------------------------------------------------------------------


class TestInMemoryBaselineStore:
    """Test the InMemoryBaselineStore."""

    def test_save_and_load(self, sample_baseline):
        """Should save and load baseline."""
        store = InMemoryBaselineStore()

        store.save(sample_baseline)
        loaded = store.load()

        assert loaded is not None
        assert loaded.p50_latency_ms == sample_baseline.p50_latency_ms
        assert loaded.run_count == sample_baseline.run_count

    def test_load_returns_none_when_empty(self):
        """Should return None when no baseline saved."""
        store = InMemoryBaselineStore()
        assert store.load() is None


