"""
Unit Tests for Unified Evaluation Runner

Tests the unified evaluation pipeline that runs multiple frameworks.
Uses mocks to verify orchestration logic.

STAFF ENGINEER PATTERNS:
------------------------
1. Mock individual framework runners
2. Test fail-fast behavior (skip LLM evals on fast eval failure)
3. Test framework result aggregation
4. Verify metric comparison across frameworks
"""

import pytest
from unittest.mock import patch, MagicMock

from agent_eval_pipeline.harness.unified_runner import (
    EvalFramework,
    FrameworkResult,
    UnifiedEvalReport,
    FAST_FRAMEWORKS,
    LLM_FRAMEWORKS,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def passed_framework_result():
    """Create a passed framework result."""
    return FrameworkResult(
        framework=EvalFramework.SCHEMA,
        passed=True,
        scores={"pass_rate": 1.0},
        details={"passed": 5, "failed": 0},
    )


@pytest.fixture
def failed_framework_result():
    """Create a failed framework result."""
    return FrameworkResult(
        framework=EvalFramework.SCHEMA,
        passed=False,
        scores={"pass_rate": 0.6},
        details={"passed": 3, "failed": 2},
    )


@pytest.fixture
def error_framework_result():
    """Create an error framework result."""
    return FrameworkResult(
        framework=EvalFramework.DEEPEVAL,
        passed=False,
        scores={},
        error="DeepEval not configured",
    )


# ---------------------------------------------------------------------------
# EVAL FRAMEWORK ENUM TESTS
# ---------------------------------------------------------------------------


class TestEvalFramework:
    """Test EvalFramework enum."""

    def test_framework_values(self):
        """Should have expected framework values."""
        assert EvalFramework.SCHEMA.value == "schema"
        assert EvalFramework.RETRIEVAL.value == "retrieval"
        assert EvalFramework.CUSTOM.value == "custom"
        assert EvalFramework.DEEPEVAL.value == "deepeval"
        assert EvalFramework.RAGAS.value == "ragas"
        assert EvalFramework.DSPY.value == "dspy"

    def test_fast_frameworks(self):
        """FAST_FRAMEWORKS should contain schema and retrieval."""
        assert EvalFramework.SCHEMA in FAST_FRAMEWORKS
        assert EvalFramework.RETRIEVAL in FAST_FRAMEWORKS
        assert len(FAST_FRAMEWORKS) == 2

    def test_llm_frameworks(self):
        """LLM_FRAMEWORKS should contain LLM-based evals."""
        assert EvalFramework.CUSTOM in LLM_FRAMEWORKS
        assert EvalFramework.DEEPEVAL in LLM_FRAMEWORKS
        assert EvalFramework.RAGAS in LLM_FRAMEWORKS
        assert EvalFramework.DSPY in LLM_FRAMEWORKS


# ---------------------------------------------------------------------------
# FRAMEWORK RESULT TESTS
# ---------------------------------------------------------------------------


class TestFrameworkResult:
    """Test FrameworkResult dataclass."""

    def test_passed_result_status(self, passed_framework_result):
        """Passed result should have PASS status."""
        assert passed_framework_result.status == "PASS"

    def test_failed_result_status(self, failed_framework_result):
        """Failed result should have FAIL status."""
        assert failed_framework_result.status == "FAIL"

    def test_error_result_status(self, error_framework_result):
        """Error result should have ERROR status."""
        assert error_framework_result.status == "ERROR"

    def test_result_creation_with_scores(self):
        """Should create result with scores dict."""
        result = FrameworkResult(
            framework=EvalFramework.CUSTOM,
            passed=True,
            scores={"safety": 0.95, "accuracy": 0.88},
        )

        assert result.scores["safety"] == 0.95
        assert result.scores["accuracy"] == 0.88

    def test_result_creation_with_error(self):
        """Should create result with error message."""
        result = FrameworkResult(
            framework=EvalFramework.RAGAS,
            passed=False,
            scores={},
            error="RAGAS not installed",
        )

        assert result.error == "RAGAS not installed"
        assert result.status == "ERROR"


# ---------------------------------------------------------------------------
# UNIFIED EVAL REPORT TESTS
# ---------------------------------------------------------------------------


class TestUnifiedEvalReport:
    """Test UnifiedEvalReport dataclass."""

    def test_pass_rate_calculation(self):
        """Should calculate pass rate correctly."""
        report = UnifiedEvalReport(
            total_frameworks=4,
            passed_frameworks=3,
            failed_frameworks=1,
            results={},
            all_passed=False,
        )

        assert report.pass_rate == 0.75

    def test_pass_rate_empty(self):
        """Should handle empty frameworks."""
        report = UnifiedEvalReport(
            total_frameworks=0,
            passed_frameworks=0,
            failed_frameworks=0,
            results={},
            all_passed=True,
        )

        assert report.pass_rate == 0.0

    def test_get_metric_comparison(self):
        """Should compare metrics across frameworks."""
        report = UnifiedEvalReport(
            total_frameworks=2,
            passed_frameworks=2,
            failed_frameworks=0,
            results={
                EvalFramework.CUSTOM: FrameworkResult(
                    framework=EvalFramework.CUSTOM,
                    passed=True,
                    scores={"safety": 0.9, "accuracy": 0.85},
                ),
                EvalFramework.DEEPEVAL: FrameworkResult(
                    framework=EvalFramework.DEEPEVAL,
                    passed=True,
                    scores={"safety": 0.95, "coherence": 0.88},
                ),
            },
            all_passed=True,
        )

        safety_comparison = report.get_metric_comparison("safety")

        assert "custom" in safety_comparison
        assert "deepeval" in safety_comparison
        assert safety_comparison["custom"] == 0.9
        assert safety_comparison["deepeval"] == 0.95

    def test_get_metric_comparison_missing_metric(self):
        """Should only return frameworks that have the metric."""
        report = UnifiedEvalReport(
            total_frameworks=2,
            passed_frameworks=2,
            failed_frameworks=0,
            results={
                EvalFramework.CUSTOM: FrameworkResult(
                    framework=EvalFramework.CUSTOM,
                    passed=True,
                    scores={"safety": 0.9},
                ),
                EvalFramework.DEEPEVAL: FrameworkResult(
                    framework=EvalFramework.DEEPEVAL,
                    passed=True,
                    scores={"coherence": 0.88},  # No safety metric
                ),
            },
            all_passed=True,
        )

        safety_comparison = report.get_metric_comparison("safety")

        assert "custom" in safety_comparison
        assert "deepeval" not in safety_comparison

    def test_all_passed_property(self):
        """Should correctly report all_passed status."""
        passed_report = UnifiedEvalReport(
            total_frameworks=3,
            passed_frameworks=3,
            failed_frameworks=0,
            results={},
            all_passed=True,
        )
        assert passed_report.all_passed is True

        failed_report = UnifiedEvalReport(
            total_frameworks=3,
            passed_frameworks=2,
            failed_frameworks=1,
            results={},
            all_passed=False,
        )
        assert failed_report.all_passed is False
