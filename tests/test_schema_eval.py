"""
Unit Tests for Schema Evaluation

Tests the schema validation gate that checks agent output structure.

STAFF ENGINEER PATTERNS:
------------------------
1. Test validation logic without LLM calls
2. Cover edge cases (missing fields, invalid values)
3. Verify error message quality
"""

import pytest

from agent_eval_pipeline.schemas.lab_insights import (
    LabInsightsSummary,
    MarkerInsight,
    SafetyNote,
)
from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id
from agent_eval_pipeline.evals.schema_eval import (
    validate_case_output,
    SchemaEvalResult,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_golden_case():
    """Get a real golden case for testing."""
    return get_case_by_id("thyroid-001")


@pytest.fixture
def valid_output() -> LabInsightsSummary:
    """Create a valid agent output matching thyroid-001 golden case expectations."""
    # Note: thyroid-001 expects TSH=high, Free T4=low, Free T3=normal
    return LabInsightsSummary(
        summary="Your TSH is elevated, indicating potential thyroid issues.",
        key_insights=[
            MarkerInsight(
                marker="TSH",
                status="high",
                value=5.5,
                unit="mIU/L",
                ref_range="0.4-4.0",
                trend="unknown",
                clinical_relevance="Elevated TSH may indicate hypothyroidism",
                action="Discuss with your doctor",
            ),
            MarkerInsight(
                marker="Free T4",
                status="low",  # Matches thyroid-001 expected_marker_statuses
                value=0.8,
                unit="ng/dL",
                ref_range="0.8-1.8",
                trend="stable",
                clinical_relevance="Free T4 is at lower bound of normal range",
                action="Discuss with your doctor",
            ),
            MarkerInsight(
                marker="Free T3",
                status="normal",  # Matches thyroid-001 expected_marker_statuses
                value=2.5,
                unit="pg/mL",
                ref_range="2.0-4.0",
                trend="stable",
                clinical_relevance="Free T3 is within normal range",
                action="Continue monitoring",
            ),
        ],
        recommended_topics_for_doctor=["TSH elevation", "Thyroid function"],
        lifestyle_considerations=["Monitor energy levels"],
        safety_notes=[
            SafetyNote(
                message="This is educational information, not a diagnosis.",
                type="non_diagnostic",
            )
        ],
    )


# ---------------------------------------------------------------------------
# SCHEMA EVAL RESULT TESTS
# ---------------------------------------------------------------------------


class TestSchemaEvalResult:
    """Test SchemaEvalResult dataclass."""

    def test_passed_result(self):
        """Passed result should have correct attributes."""
        result = SchemaEvalResult(
            case_id="test-001",
            passed=True,
        )

        assert result.passed is True
        assert result.error is None

    def test_failed_result_with_error(self):
        """Failed result should include error message."""
        result = SchemaEvalResult(
            case_id="test-001",
            passed=False,
            error="Output is None",
        )

        assert result.passed is False
        assert result.error == "Output is None"


# ---------------------------------------------------------------------------
# VALIDATE_CASE_OUTPUT TESTS
# ---------------------------------------------------------------------------


class TestValidateCaseOutput:
    """Test the validate_case_output function."""

    def test_valid_output_passes(
        self, sample_golden_case, valid_output: LabInsightsSummary
    ):
        """Valid output matching expectations should pass."""
        result = validate_case_output(sample_golden_case, valid_output)

        assert result.passed is True
        assert result.error is None

    def test_none_output_raises(self, sample_golden_case):
        """None output should raise an exception (caller handles this)."""
        # validate_case_output expects a LabInsightsSummary, not None
        # The run_schema_eval function catches this and creates a failed result
        with pytest.raises(AttributeError):
            validate_case_output(sample_golden_case, None)

    def test_wrong_marker_status_fails(
        self, sample_golden_case, valid_output: LabInsightsSummary
    ):
        """Output with wrong marker status should fail."""
        # Change TSH status from "high" to "normal"
        valid_output.key_insights[0].status = "normal"

        result = validate_case_output(sample_golden_case, valid_output)

        assert result.passed is False
        assert result.invalid_values is not None
        assert "TSH" in result.invalid_values


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMA TESTS
# ---------------------------------------------------------------------------


class TestPydanticSchemas:
    """Test that Pydantic schemas validate correctly."""

    def test_marker_insight_valid_statuses(self):
        """MarkerInsight should accept valid status values."""
        for status in ["high", "low", "normal", "borderline"]:
            insight = MarkerInsight(
                marker="TSH",
                status=status,
                value=5.5,
                unit="mIU/L",
                ref_range="0.4-4.0",
                trend="unknown",
                clinical_relevance="Test",
                action="Test",
            )
            assert insight.status == status

    def test_marker_insight_invalid_status_raises(self):
        """MarkerInsight should reject invalid status values."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            MarkerInsight(
                marker="TSH",
                status="elevated",  # Invalid
                value=5.5,
                unit="mIU/L",
                ref_range="0.4-4.0",
                trend="unknown",
                clinical_relevance="Test",
                action="Test",
            )

    def test_marker_insight_valid_trends(self):
        """MarkerInsight should accept valid trend values."""
        for trend in ["increasing", "decreasing", "stable", "unknown"]:
            insight = MarkerInsight(
                marker="TSH",
                status="high",
                value=5.5,
                unit="mIU/L",
                ref_range="0.4-4.0",
                trend=trend,
                clinical_relevance="Test",
                action="Test",
            )
            assert insight.trend == trend

    def test_safety_note_valid_types(self):
        """SafetyNote should accept valid type values."""
        for note_type in ["non_diagnostic", "emergency_notice", "lifestyle_scope"]:
            note = SafetyNote(
                message="Test message",
                type=note_type,
            )
            assert note.type == note_type

    def test_lab_insights_summary_complete(self):
        """LabInsightsSummary should validate with all fields."""
        summary = LabInsightsSummary(
            summary="Test summary",
            key_insights=[
                MarkerInsight(
                    marker="TSH",
                    status="high",
                    value=5.5,
                    unit="mIU/L",
                    ref_range="0.4-4.0",
                    trend="unknown",
                    clinical_relevance="Test",
                    action="Test",
                )
            ],
            recommended_topics_for_doctor=["Topic 1"],
            lifestyle_considerations=["Consideration 1"],
            safety_notes=[
                SafetyNote(message="Test", type="non_diagnostic")
            ],
        )

        assert summary.summary == "Test summary"
        assert len(summary.key_insights) == 1
        assert len(summary.safety_notes) == 1
