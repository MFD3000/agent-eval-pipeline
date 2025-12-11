"""
DSPy Integration Tests

These tests demonstrate DSPy patterns and verify our DSPy ReAct agent works correctly.
They're designed to run WITHOUT API calls where possible.

INTERVIEW TALKING POINT:
------------------------
"I wrote tests for our DSPy ReAct agent that verify tool functions work,
module structure is correct, and the agent produces expected outputs.
Integration tests with real LLMs are marked to skip without an API key."

TEST PATTERNS DEMONSTRATED:
--------------------------
1. Testing tool functions in isolation
2. Testing module composition
3. Testing DSPy judge signatures
4. Integration tests with real LLM (skipped without API key)
"""

import pytest
import dspy


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_labs():
    """Sample lab data for testing."""
    return [
        {
            "marker": "TSH",
            "value": 5.5,
            "unit": "mIU/L",
            "ref_low": 0.4,
            "ref_high": 4.0,
        },
        {
            "marker": "Free T4",
            "value": 0.8,
            "unit": "ng/dL",
            "ref_low": 0.8,
            "ref_high": 1.8,
        },
    ]


@pytest.fixture
def sample_history():
    """Sample historical data."""
    return [
        {
            "marker": "TSH",
            "value": 4.2,
            "unit": "mIU/L",
            "date": "2024-06-01",
        },
    ]


# ---------------------------------------------------------------------------
# SIGNATURE TESTS
# ---------------------------------------------------------------------------


class TestSignatures:
    """Test DSPy signature definitions."""

    def test_react_signature_fields(self):
        """Verify LabAnalysisSignature has correct fields."""
        from agent_eval_pipeline.agent.dspy_react_agent import LabAnalysisSignature

        sig = LabAnalysisSignature

        # Check input fields
        assert "query" in sig.input_fields
        assert "labs" in sig.input_fields
        assert "medications" in sig.input_fields
        assert "symptoms" in sig.input_fields

        # Check output fields
        assert "analysis" in sig.output_fields

    def test_judge_signature_fields(self):
        """Verify judge signatures have score outputs."""
        from agent_eval_pipeline.evals.judge.dspy_judge import (
            EvaluateClinicalCorrectness,
            EvaluateSafetyCompliance,
        )

        # Clinical correctness
        assert "score" in EvaluateClinicalCorrectness.output_fields
        assert "reasoning" in EvaluateClinicalCorrectness.output_fields

        # Safety compliance
        assert "score" in EvaluateSafetyCompliance.output_fields
        assert "critical_issues" in EvaluateSafetyCompliance.output_fields


# ---------------------------------------------------------------------------
# MODULE STRUCTURE TESTS
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """Test DSPy module composition."""

    def test_dspy_judge_module_has_evaluators(self):
        """Verify DSPyJudge has all dimension evaluators."""
        from agent_eval_pipeline.evals.judge.dspy_judge import DSPyJudge

        judge = DSPyJudge()

        # Check all four dimensions
        assert hasattr(judge, "clinical")
        assert hasattr(judge, "safety")
        assert hasattr(judge, "completeness")
        assert hasattr(judge, "clarity")

        # All should be ChainOfThought
        assert isinstance(judge.clinical, dspy.ChainOfThought)
        assert isinstance(judge.safety, dspy.ChainOfThought)


# ---------------------------------------------------------------------------
# DATA CONVERSION TESTS
# ---------------------------------------------------------------------------


class TestDataConversion:
    """Test conversion between DSPy and our schemas."""

    def test_judge_output_conversion(self):
        """Test converting DSPy judge prediction to JudgeOutput."""
        from agent_eval_pipeline.evals.judge.dspy_judge import (
            dspy_prediction_to_judge_output,
        )

        prediction = dspy.Prediction(
            scores={
                "clinical_correctness": 4.5,
                "safety_compliance": 5.0,
                "completeness": 4.0,
                "clarity": 4.5,
            },
            weighted_score=4.55,
            reasoning={
                "clinical_correctness": "Accurate interpretation",
                "safety_compliance": "Appropriate disclaimers",
                "completeness": "Covered all markers",
                "clarity": "Clear language",
            },
            critical_issues=[],
        )

        output = dspy_prediction_to_judge_output(prediction)

        assert output.clinical_correctness.score == 4.5
        assert output.safety_compliance.score == 5.0
        assert output.clinical_correctness.reasoning == "Accurate interpretation"
        assert len(output.critical_issues) == 0


# ---------------------------------------------------------------------------
# OPTIMIZATION INFRASTRUCTURE TESTS
# ---------------------------------------------------------------------------


class TestOptimizationInfrastructure:
    """Test the optimization pipeline setup."""

    def test_judge_metric_perfect_match(self):
        """Test judge metric with perfect score match."""
        from agent_eval_pipeline.evals.judge.dspy_judge import create_judge_metric

        metric = create_judge_metric()

        # Create example with expected scores
        example = dspy.Example(
            expected_scores={
                "clinical_correctness": 4.0,
                "safety_compliance": 5.0,
                "completeness": 4.0,
                "clarity": 4.0,
            }
        )

        # Prediction matches exactly
        prediction = dspy.Prediction(
            scores={
                "clinical_correctness": 4.0,
                "safety_compliance": 5.0,
                "completeness": 4.0,
                "clarity": 4.0,
            }
        )

        score = metric(example, prediction)
        assert score == 1.0  # Perfect match

    def test_judge_metric_with_errors(self):
        """Test judge metric penalizes errors."""
        from agent_eval_pipeline.evals.judge.dspy_judge import create_judge_metric

        metric = create_judge_metric()

        example = dspy.Example(
            expected_scores={
                "clinical_correctness": 5.0,
                "safety_compliance": 5.0,
                "completeness": 5.0,
                "clarity": 5.0,
            }
        )

        # Prediction is off by 1 on each dimension
        prediction = dspy.Prediction(
            scores={
                "clinical_correctness": 4.0,
                "safety_compliance": 4.0,
                "completeness": 4.0,
                "clarity": 4.0,
            }
        )

        score = metric(example, prediction)
        # Total error = 4, max error = 16, score = 1 - 4/16 = 0.75
        assert score == 0.75

    def test_judge_metric_no_ground_truth(self):
        """Test metric handles examples without ground truth."""
        from agent_eval_pipeline.evals.judge.dspy_judge import create_judge_metric

        metric = create_judge_metric()

        # Example without expected_scores
        example = dspy.Example(query="test")
        prediction = dspy.Prediction(scores={})

        score = metric(example, prediction)
        assert score == 1.0  # Default when no ground truth


# ---------------------------------------------------------------------------
# REACT AGENT TESTS
# ---------------------------------------------------------------------------


class TestReActAgent:
    """Test ReAct agent structure."""

    def test_react_agent_has_tools(self):
        """Verify ReAct agent has expected tools."""
        from agent_eval_pipeline.agent.dspy_react_agent import LabReActAgent

        agent = LabReActAgent()

        # Should have react module
        assert hasattr(agent, "react")
        assert isinstance(agent.react, dspy.ReAct)

    def test_tool_functions_work(self):
        """Test that tool functions return expected results."""
        from agent_eval_pipeline.agent.dspy_react_agent import (
            lookup_reference_range,
            check_medication_interaction,
            search_medical_context,
        )

        # Test lookup
        tsh_ref = lookup_reference_range("TSH")
        assert "0.4-4.0" in tsh_ref

        # Test medication interaction
        interaction = check_medication_interaction("TSH", "biotin")
        assert "interfere" in interaction.lower() or "assay" in interaction.lower()

        # Test context search
        context = search_medical_context("thyroid")
        assert "TSH" in context

    def test_unknown_marker_handled(self):
        """Test graceful handling of unknown markers."""
        from agent_eval_pipeline.agent.dspy_react_agent import lookup_reference_range

        result = lookup_reference_range("UnknownMarkerXYZ")
        assert "not found" in result.lower()

    def test_format_labs_helper(self, sample_labs):
        """Test lab formatting in ReAct agent."""
        from agent_eval_pipeline.agent.dspy_react_agent import LabReActAgent

        agent = LabReActAgent()
        formatted = agent._format_labs(sample_labs)

        assert "TSH" in formatted
        assert "5.5" in formatted


# ---------------------------------------------------------------------------
# INTEGRATION TESTS (SKIPPED WITHOUT API KEY)
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests requiring API key."""

    @pytest.mark.skipif(
        not pytest.importorskip("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_react_agent_produces_output(self, sample_labs):
        """Test ReAct agent produces valid output."""
        from agent_eval_pipeline.agent.dspy_react_agent import run_react_agent

        result = run_react_agent(
            query="What do my thyroid results mean?",
            labs=sample_labs,
            symptoms=["fatigue"],
        )

        assert result.analysis is not None
        assert len(result.analysis) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_dspy_judge_produces_scores(self):
        """Test DSPy judge produces valid scores."""
        from agent_eval_pipeline.evals.judge.dspy_judge import run_dspy_judge
        from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id
        from agent_eval_pipeline.agent import run_agent

        case = get_case_by_id("thyroid-001")
        agent_result = run_agent(case)

        if hasattr(agent_result, "output"):
            judge_result = run_dspy_judge(case, agent_result.output)

            assert judge_result.clinical_correctness.score >= 1.0
            assert judge_result.clinical_correctness.score <= 5.0
            assert len(judge_result.clinical_correctness.reasoning) > 0
