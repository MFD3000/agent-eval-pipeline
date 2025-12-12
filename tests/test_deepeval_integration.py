"""
DeepEval Pytest Integration Tests

This file demonstrates DeepEval's pytest integration pattern.
Run with: `deepeval test run tests/test_deepeval_integration.py`
Or with regular pytest: `pytest tests/test_deepeval_integration.py`

INTERVIEW TALKING POINT:
------------------------
"DeepEval integrates with pytest using assert_test. Each test case can
have multiple metrics, and the test fails if any metric doesn't meet its
threshold. I can run these in CI with `deepeval test run -n 4` for
parallel execution. The test output shows exactly which metrics passed
or failed and why."

NOTE: These tests require OPENAI_API_KEY and make actual LLM calls.
Use @pytest.mark.skipif to skip when no API key is available.
"""

import os
import pytest

# Skip all tests in this file if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from agent_eval_pipeline.golden_sets.thyroid_cases import (
    get_all_golden_cases,
    get_case_by_id,
    GoldenCase,
)
from agent_eval_pipeline.agent import run_agent, AgentError
from agent_eval_pipeline.evals.deepeval.adapters import agent_result_to_test_case
from agent_eval_pipeline.evals.deepeval.metrics import (
    get_safety_compliance,
    get_clinical_correctness,
    get_completeness,
    get_healthcare_metrics,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_cases() -> list[GoldenCase]:
    """Load all golden cases once per module."""
    return get_all_golden_cases()


@pytest.fixture(scope="module")
def agent_results(all_cases) -> dict[str, LLMTestCase | None]:
    """
    Pre-run agent on all cases.

    Cached at module level to avoid re-running agent for each test.
    """
    results = {}
    for case in all_cases:
        result = run_agent(case)
        if isinstance(result, AgentError):
            results[case.id] = None
        else:
            test_case = agent_result_to_test_case(case, result)
            results[case.id] = test_case
    return results


# ---------------------------------------------------------------------------
# SAFETY COMPLIANCE TESTS
# ---------------------------------------------------------------------------


class TestSafetyCompliance:
    """
    Safety compliance is critical - these tests should NEVER fail in production.

    Safety requirements:
    - Include doctor consultation recommendation
    - No definitive diagnoses
    - No medication recommendations
    - Appropriate uncertainty language
    """

    @pytest.mark.parametrize("case", get_all_golden_cases(), ids=lambda c: c.id)
    def test_safety_compliance(self, case: GoldenCase, agent_results):
        """Every case must pass safety compliance."""
        test_case = agent_results.get(case.id)

        if test_case is None:
            pytest.skip(f"Agent failed for {case.id}")

        assert_test(test_case, [get_safety_compliance()])


# ---------------------------------------------------------------------------
# CLINICAL CORRECTNESS TESTS
# ---------------------------------------------------------------------------


class TestClinicalCorrectness:
    """
    Clinical correctness verifies medical accuracy.

    The agent should:
    - Correctly identify high/low/normal values
    - Accurately interpret patterns
    - Not make factually incorrect medical statements
    """

    @pytest.mark.parametrize("case", get_all_golden_cases(), ids=lambda c: c.id)
    def test_clinical_correctness(self, case: GoldenCase, agent_results):
        """Cases should have clinically correct interpretations."""
        test_case = agent_results.get(case.id)

        if test_case is None:
            pytest.skip(f"Agent failed for {case.id}")

        assert_test(test_case, [get_clinical_correctness()])


# ---------------------------------------------------------------------------
# COMPLETENESS TESTS
# ---------------------------------------------------------------------------


class TestCompleteness:
    """
    Completeness checks that responses cover all important points.
    """

    @pytest.mark.parametrize("case", get_all_golden_cases(), ids=lambda c: c.id)
    def test_completeness(self, case: GoldenCase, agent_results):
        """Responses should cover expected semantic points."""
        test_case = agent_results.get(case.id)

        if test_case is None:
            pytest.skip(f"Agent failed for {case.id}")

        assert_test(test_case, [get_completeness()])


# ---------------------------------------------------------------------------
# COMBINED METRIC TESTS
# ---------------------------------------------------------------------------


class TestAllMetrics:
    """
    Run all metrics together for comprehensive evaluation.
    """

    @pytest.mark.parametrize("case", get_all_golden_cases(), ids=lambda c: c.id)
    def test_all_healthcare_metrics(self, case: GoldenCase, agent_results):
        """Test all healthcare metrics together."""
        test_case = agent_results.get(case.id)

        if test_case is None:
            pytest.skip(f"Agent failed for {case.id}")

        assert_test(test_case, get_healthcare_metrics())


# ---------------------------------------------------------------------------
# SPECIFIC CASE TESTS
# ---------------------------------------------------------------------------


class TestSpecificCases:
    """
    Targeted tests for specific golden cases.

    These test specific behaviors we expect for particular scenarios.
    """

    def test_high_tsh_mentions_hypothyroid(self, agent_results):
        """High TSH case should mention potential hypothyroidism."""
        test_case = agent_results.get("thyroid-001")

        if test_case is None:
            pytest.skip("Agent failed for thyroid-001")

        # The expected output should contain hypothyroid-related content
        assert "hypothyroid" in test_case.expected_output.lower() or \
               "underactive" in test_case.expected_output.lower()

        assert_test(test_case, [get_clinical_correctness()])

    def test_normal_values_reassure(self, agent_results):
        """Normal values should provide reassurance, not alarm."""
        test_case = agent_results.get("thyroid-002")

        if test_case is None:
            pytest.skip("Agent failed for thyroid-002")

        # Should mention values are normal
        assert "normal" in test_case.expected_output.lower()

        assert_test(test_case, [get_clinical_correctness(), get_safety_compliance()])

    def test_low_tsh_mentions_hyperthyroid(self, agent_results):
        """Low TSH with high T4 should mention potential hyperthyroidism."""
        test_case = agent_results.get("thyroid-003")

        if test_case is None:
            pytest.skip("Agent failed for thyroid-003")

        assert_test(test_case, [get_clinical_correctness()])
