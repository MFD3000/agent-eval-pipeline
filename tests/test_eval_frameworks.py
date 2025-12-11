"""
Tests for DeepEval and RAGAS Integration

These tests verify the integration modules work correctly WITHOUT making
actual LLM calls. They test:
- Adapter functions (schema conversion)
- Metric configuration
- Module structure
- Data flow

INTERVIEW TALKING POINT:
------------------------
"I test eval framework integrations at two levels. Unit tests verify adapters
and data conversion without API calls - these run in milliseconds. Integration
tests with real LLMs are marked to skip without an API key, so CI can run
the fast tests while we run full eval suites manually or in nightly builds."
"""

import os
import pytest


# ---------------------------------------------------------------------------
# DEEPEVAL ADAPTER TESTS
# ---------------------------------------------------------------------------


class TestDeepEvalAdapters:
    """Test DeepEval adapter functions."""

    def test_format_labs_for_input(self):
        """Test lab value formatting."""
        from agent_eval_pipeline.evals.deepeval.adapters import format_labs_for_input
        from agent_eval_pipeline.golden_sets.thyroid_cases import LabValue

        labs = [
            LabValue(
                date="2025-01-01",
                marker="TSH",
                value=5.5,
                unit="mIU/L",
                ref_low=0.4,
                ref_high=4.0,
            ),
        ]

        result = format_labs_for_input(labs)

        assert "TSH" in result
        assert "5.5" in result
        assert "mIU/L" in result
        assert "0.4-4.0" in result

    def test_format_history_for_input(self):
        """Test history formatting."""
        from agent_eval_pipeline.evals.deepeval.adapters import format_history_for_input
        from agent_eval_pipeline.golden_sets.thyroid_cases import LabValue

        history = [
            LabValue(date="2025-01-01", marker="TSH", value=4.2, unit="mIU/L"),
        ]

        result = format_history_for_input(history)

        assert "2025-01-01" in result
        assert "TSH" in result
        assert "4.2" in result

    def test_format_empty_history(self):
        """Test empty history handling."""
        from agent_eval_pipeline.evals.deepeval.adapters import format_history_for_input

        result = format_history_for_input([])

        assert "No historical data" in result

    def test_golden_cases_to_dataset(self):
        """Test dataset creation from golden cases."""
        from agent_eval_pipeline.evals.deepeval.adapters import golden_cases_to_dataset
        from agent_eval_pipeline.golden_sets.thyroid_cases import get_all_golden_cases

        cases = get_all_golden_cases()[:2]  # Just test with 2
        dataset = golden_cases_to_dataset(cases)

        assert len(dataset.goldens) == 2
        assert dataset.goldens[0].input is not None
        assert dataset.goldens[0].expected_output is not None


class TestDeepEvalMetrics:
    """Test DeepEval metric configuration (requires API key)."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_healthcare_metrics_defined(self):
        """Verify all healthcare metrics are defined."""
        from agent_eval_pipeline.evals.deepeval.metrics import (
            get_clinical_correctness,
            get_safety_compliance,
            get_completeness,
            get_answer_clarity,
            get_healthcare_metrics,
        )

        metrics = get_healthcare_metrics()
        assert len(metrics) == 4

        clinical = get_clinical_correctness()
        assert clinical is not None
        assert clinical.name == "Clinical Correctness"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_safety_has_high_threshold(self):
        """Safety metric should have high threshold."""
        from agent_eval_pipeline.evals.deepeval.metrics import get_safety_compliance

        safety = get_safety_compliance()
        assert safety.threshold >= 0.9


class TestDeepEvalResultSchemas:
    """Test DeepEval result dataclasses."""

    def test_deepeval_result_creation(self):
        """Test creating DeepEvalResult."""
        from agent_eval_pipeline.evals.deepeval.evaluator import (
            DeepEvalResult,
            DeepEvalMetricResult,
        )

        result = DeepEvalResult(
            case_id="test-001",
            passed=True,
            metrics=[
                DeepEvalMetricResult(
                    name="Safety",
                    score=0.95,
                    passed=True,
                    threshold=0.9,
                )
            ],
            overall_score=0.95,
        )

        assert result.case_id == "test-001"
        assert result.passed is True
        assert len(result.failed_metrics) == 0

    def test_deepeval_report_creation(self):
        """Test creating DeepEvalReport."""
        from agent_eval_pipeline.evals.deepeval.evaluator import (
            DeepEvalReport,
            DeepEvalResult,
        )

        report = DeepEvalReport(
            total_cases=5,
            passed_cases=4,
            failed_cases=1,
            results=[],
            metric_averages={"safety": 0.92},
        )

        assert report.pass_rate == 0.8
        assert not report.all_passed


# ---------------------------------------------------------------------------
# RAGAS ADAPTER TESTS
# ---------------------------------------------------------------------------


class TestRagasAdapters:
    """Test RAGAS adapter functions."""

    def test_format_labs_text(self):
        """Test lab text formatting for RAGAS."""
        from agent_eval_pipeline.evals.ragas.adapters import format_labs_text
        from agent_eval_pipeline.golden_sets.thyroid_cases import LabValue

        labs = [
            LabValue(
                date="2025-01-01",
                marker="TSH",
                value=5.5,
                unit="mIU/L",
                ref_low=0.4,
                ref_high=4.0,
            ),
        ]

        result = format_labs_text(labs)

        assert "TSH: 5.5 mIU/L" in result
        assert "0.4-4.0" in result

    def test_format_user_input(self):
        """Test user input formatting."""
        from agent_eval_pipeline.evals.ragas.adapters import format_user_input
        from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id

        case = get_case_by_id("thyroid-001")
        result = format_user_input(case)

        assert case.query in result
        assert "TSH" in result
        assert "fatigue" in result.lower()


class TestRagasMetrics:
    """Test RAGAS metric configuration."""

    def test_metric_names_defined(self):
        """Verify metric names are defined."""
        from agent_eval_pipeline.evals.ragas.metrics import RAGAS_METRIC_NAMES

        assert "faithfulness" in RAGAS_METRIC_NAMES
        assert "context_precision" in RAGAS_METRIC_NAMES
        assert "context_recall" in RAGAS_METRIC_NAMES

    def test_default_thresholds_defined(self):
        """Verify default thresholds are set."""
        from agent_eval_pipeline.evals.ragas.metrics import DEFAULT_THRESHOLDS

        assert DEFAULT_THRESHOLDS["faithfulness"] >= 0.7
        assert DEFAULT_THRESHOLDS["context_precision"] >= 0.5

    def test_check_thresholds_pass(self):
        """Test threshold checking with passing scores."""
        from agent_eval_pipeline.evals.ragas.metrics import check_thresholds

        scores = {
            "faithfulness": 0.9,
            "context_precision": 0.8,
        }

        passed, failed = check_thresholds(scores)

        assert passed is True
        assert len(failed) == 0

    def test_check_thresholds_fail(self):
        """Test threshold checking with failing scores."""
        from agent_eval_pipeline.evals.ragas.metrics import check_thresholds

        scores = {
            "faithfulness": 0.5,  # Below default 0.8
            "context_precision": 0.8,
        }

        passed, failed = check_thresholds(scores)

        assert passed is False
        assert "faithfulness" in failed


class TestRagasResultSchemas:
    """Test RAGAS result dataclasses."""

    def test_ragas_result_creation(self):
        """Test creating RagasResult."""
        from agent_eval_pipeline.evals.ragas.evaluator import RagasResult

        result = RagasResult(
            case_id="test-001",
            scores={"faithfulness": 0.9},
            passed=True,
            failed_metrics=[],
        )

        assert result.case_id == "test-001"
        assert result.passed is True

    def test_ragas_report_creation(self):
        """Test creating RagasReport."""
        from agent_eval_pipeline.evals.ragas.evaluator import RagasReport

        report = RagasReport(
            total_cases=5,
            evaluated_cases=4,
            passed_cases=3,
            failed_cases=1,
            skipped_cases=1,
            metric_averages={"faithfulness": 0.85},
            results=[],
        )

        assert report.pass_rate == 0.75  # 3/4 evaluated
        assert not report.all_passed


# ---------------------------------------------------------------------------
# UNIFIED RUNNER TESTS
# ---------------------------------------------------------------------------


class TestUnifiedRunner:
    """Test unified evaluation runner."""

    def test_eval_framework_enum(self):
        """Test framework enum values."""
        from agent_eval_pipeline.harness.unified_runner import EvalFramework

        assert EvalFramework.SCHEMA.value == "schema"
        assert EvalFramework.DEEPEVAL.value == "deepeval"
        assert EvalFramework.RAGAS.value == "ragas"

    def test_framework_result_status(self):
        """Test framework result status property."""
        from agent_eval_pipeline.harness.unified_runner import (
            FrameworkResult,
            EvalFramework,
        )

        passed_result = FrameworkResult(
            framework=EvalFramework.SCHEMA,
            passed=True,
            scores={"pass_rate": 1.0},
        )
        assert passed_result.status == "PASS"

        failed_result = FrameworkResult(
            framework=EvalFramework.SCHEMA,
            passed=False,
            scores={},
        )
        assert failed_result.status == "FAIL"

        error_result = FrameworkResult(
            framework=EvalFramework.SCHEMA,
            passed=False,
            scores={},
            error="Test error",
        )
        assert error_result.status == "ERROR"

    def test_unified_report_pass_rate(self):
        """Test unified report pass rate calculation."""
        from agent_eval_pipeline.harness.unified_runner import (
            UnifiedEvalReport,
            FrameworkResult,
            EvalFramework,
        )

        results = {
            EvalFramework.SCHEMA: FrameworkResult(
                framework=EvalFramework.SCHEMA,
                passed=True,
                scores={},
            ),
            EvalFramework.DEEPEVAL: FrameworkResult(
                framework=EvalFramework.DEEPEVAL,
                passed=False,
                scores={},
            ),
        }

        report = UnifiedEvalReport(
            total_frameworks=2,
            passed_frameworks=1,
            failed_frameworks=1,
            results=results,
        )

        assert report.pass_rate == 0.5
        assert not report.all_passed


# ---------------------------------------------------------------------------
# IMPORT TESTS
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify all modules import correctly."""

    def test_deepeval_module_imports(self):
        """Test DeepEval module imports."""
        from agent_eval_pipeline.evals.deepeval import (
            golden_case_to_llm_test_case,
            golden_cases_to_dataset,
            get_clinical_correctness,
            get_safety_compliance,
            get_healthcare_metrics,
            run_deepeval_evaluation,
        )

        assert golden_case_to_llm_test_case is not None
        assert golden_cases_to_dataset is not None
        assert get_clinical_correctness is not None
        assert get_safety_compliance is not None
        assert get_healthcare_metrics is not None
        assert run_deepeval_evaluation is not None

    def test_ragas_module_imports(self):
        """Test RAGAS module imports."""
        from agent_eval_pipeline.evals.ragas import (
            golden_case_to_ragas_sample,
            create_ragas_dataset,
            get_ragas_metrics,
            run_ragas_evaluation,
        )

        assert golden_case_to_ragas_sample is not None
        assert create_ragas_dataset is not None
        assert get_ragas_metrics is not None
        assert run_ragas_evaluation is not None

    def test_evals_module_exports(self):
        """Test main evals module exports new frameworks."""
        from agent_eval_pipeline.evals import (
            # DeepEval
            golden_case_to_llm_test_case,
            get_clinical_correctness,
            get_safety_compliance,
            run_deepeval_evaluation,
            # RAGAS
            golden_case_to_ragas_sample,
            get_ragas_metrics,
            run_ragas_evaluation,
        )

        # All should be importable
        assert golden_case_to_llm_test_case is not None
        assert get_clinical_correctness is not None
        assert run_deepeval_evaluation is not None
        assert golden_case_to_ragas_sample is not None
        assert run_ragas_evaluation is not None
