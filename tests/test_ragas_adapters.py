"""
Unit Tests for RAGAS Adapters

Tests the conversion between project schemas and RAGAS samples.
No actual RAGAS evaluation - just schema conversion.

STAFF ENGINEER PATTERNS:
------------------------
1. Test format functions independently
2. Verify RAGAS sample creation
3. Test dataset creation for batch evaluation
4. Handle edge cases (missing fields, errors)
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from agent_eval_pipeline.evals.ragas.adapters import (
    format_labs_text,
    format_user_input,
    format_reference,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@dataclass
class MockLabValue:
    """Mock lab value for testing."""
    marker: str
    value: float
    unit: str
    ref_low: float | None = None
    ref_high: float | None = None


@pytest.fixture
def mock_lab_values():
    """Create mock lab values."""
    return [
        MockLabValue(marker="TSH", value=5.5, unit="mIU/L", ref_low=0.4, ref_high=4.0),
        MockLabValue(marker="Free T4", value=0.9, unit="ng/dL", ref_low=0.8, ref_high=1.8),
    ]


@pytest.fixture
def mock_golden_case(mock_lab_values):
    """Create a mock golden case."""
    case = MagicMock()
    case.id = "thyroid-001"
    case.query = "What do my thyroid results mean?"
    case.labs = mock_lab_values
    case.symptoms = ["fatigue", "weight gain"]
    case.expected_semantic_points = [
        "Elevated TSH indicates potential hypothyroidism",
        "Recommend discussing with doctor",
    ]
    return case




# ---------------------------------------------------------------------------
# FORMAT LABS TEXT TESTS
# ---------------------------------------------------------------------------


class TestFormatLabsText:
    """Test format_labs_text function."""

    def test_formats_labs_with_reference(self, mock_lab_values):
        """Should format labs with reference ranges."""
        result = format_labs_text(mock_lab_values)

        assert "TSH: 5.5 mIU/L (reference: 0.4-4.0)" in result
        assert "Free T4: 0.9 ng/dL (reference: 0.8-1.8)" in result

    def test_formats_labs_without_reference(self):
        """Should format labs without reference when missing."""
        labs = [MockLabValue(marker="TSH", value=5.5, unit="mIU/L")]

        result = format_labs_text(labs)

        assert "TSH: 5.5 mIU/L" in result
        assert "reference" not in result

    def test_handles_empty_labs(self):
        """Should handle empty lab list."""
        result = format_labs_text([])
        assert result == ""


# ---------------------------------------------------------------------------
# FORMAT USER INPUT TESTS
# ---------------------------------------------------------------------------


class TestFormatUserInput:
    """Test format_user_input function."""

    def test_includes_query(self, mock_golden_case):
        """Should include the query."""
        result = format_user_input(mock_golden_case)
        assert "What do my thyroid results mean?" in result

    def test_includes_labs(self, mock_golden_case):
        """Should include lab values."""
        result = format_user_input(mock_golden_case)
        assert "TSH: 5.5 mIU/L" in result

    def test_includes_symptoms(self, mock_golden_case):
        """Should include symptoms."""
        result = format_user_input(mock_golden_case)
        assert "fatigue" in result
        assert "weight gain" in result

    def test_handles_no_symptoms(self, mock_golden_case):
        """Should handle case with no symptoms."""
        mock_golden_case.symptoms = []
        result = format_user_input(mock_golden_case)
        assert "Symptoms: None" in result


# ---------------------------------------------------------------------------
# FORMAT REFERENCE TESTS
# ---------------------------------------------------------------------------


class TestFormatReference:
    """Test format_reference function."""

    def test_formats_semantic_points(self, mock_golden_case):
        """Should format expected semantic points."""
        result = format_reference(mock_golden_case)

        assert "Elevated TSH" in result
        assert "Recommend discussing" in result
        assert result.startswith("- ")  # Bullet format
