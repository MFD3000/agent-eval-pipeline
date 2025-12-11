"""
Tests for the eval pipeline.

These tests validate that the eval harness works correctly,
NOT that the agent produces good outputs (that's what the evals do).

WHAT TO TEST:
- Schema models validate correctly
- Golden sets are properly structured
- Eval functions handle edge cases
- Metrics calculations are correct

INTERVIEW TALKING POINT:
"We have two layers of testing: pytest tests that validate the eval
infrastructure itself, and the eval gates that validate the agent.
This is like testing your test framework - you need confidence that
your evaluation system works before trusting its results."
"""

import pytest
from pydantic import ValidationError

from agent_eval_pipeline.schemas.lab_insights import (
    MarkerInsight,
    SafetyNote,
    LabInsightsSummary,
)
from agent_eval_pipeline.golden_sets.thyroid_cases import (
    get_all_golden_cases,
    get_case_by_id,
    GoldenCase,
)
from agent_eval_pipeline.evals.retrieval_eval import (
    calculate_retrieval_metrics,
)


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_marker_insight_valid(self):
        """Valid MarkerInsight should parse correctly."""
        insight = MarkerInsight(
            marker="TSH",
            status="high",
            value=5.5,
            unit="mIU/L",
            ref_range="0.4-4.0",
            trend="increasing",
            clinical_relevance="May indicate underactive thyroid",
            action="Discuss with doctor",
        )
        assert insight.marker == "TSH"
        assert insight.status == "high"

    def test_marker_insight_invalid_status(self):
        """Invalid status should raise ValidationError."""
        with pytest.raises(ValidationError):
            MarkerInsight(
                marker="TSH",
                status="elevated",  # Invalid - not in Literal
                value=5.5,
                unit="mIU/L",
                ref_range="0.4-4.0",
                trend="increasing",
                clinical_relevance="High",
                action="See doctor",
            )

    def test_marker_insight_invalid_trend(self):
        """Invalid trend should raise ValidationError."""
        with pytest.raises(ValidationError):
            MarkerInsight(
                marker="TSH",
                status="high",
                value=5.5,
                unit="mIU/L",
                ref_range="0.4-4.0",
                trend="rising",  # Invalid - should be "increasing"
                clinical_relevance="High",
                action="See doctor",
            )

    def test_safety_note_valid(self):
        """Valid SafetyNote should parse correctly."""
        note = SafetyNote(
            message="This is not a diagnosis",
            type="non_diagnostic",
        )
        assert note.type == "non_diagnostic"

    def test_lab_insights_summary_complete(self):
        """Complete LabInsightsSummary should validate."""
        summary = LabInsightsSummary(
            summary="Your TSH is elevated.",
            key_insights=[
                MarkerInsight(
                    marker="TSH",
                    status="high",
                    value=5.5,
                    unit="mIU/L",
                    ref_range="0.4-4.0",
                    trend="increasing",
                    clinical_relevance="May indicate underactive thyroid",
                    action="Discuss with doctor",
                )
            ],
            recommended_topics_for_doctor=["TSH trend", "Fatigue"],
            lifestyle_considerations=["Monitor energy levels"],
            safety_notes=[
                SafetyNote(
                    message="This is not a diagnosis",
                    type="non_diagnostic",
                )
            ],
        )
        assert len(summary.key_insights) == 1
        assert summary.key_insights[0].marker == "TSH"


class TestGoldenSets:
    """Test golden set fixtures."""

    def test_get_all_golden_cases(self):
        """Should return multiple cases."""
        cases = get_all_golden_cases()
        assert len(cases) >= 5  # We defined 5 thyroid cases

    def test_get_case_by_id_exists(self):
        """Should find existing case."""
        case = get_case_by_id("thyroid-001")
        assert case is not None
        assert case.id == "thyroid-001"

    def test_get_case_by_id_not_exists(self):
        """Should return None for missing case."""
        case = get_case_by_id("nonexistent-999")
        assert case is None

    def test_golden_case_structure(self):
        """Golden cases should have required fields."""
        cases = get_all_golden_cases()
        for case in cases:
            assert case.id
            assert case.member_id
            assert case.query
            assert len(case.labs) > 0
            # Semantic points should be defined
            assert len(case.expected_semantic_points) > 0


class TestRetrievalMetrics:
    """Test retrieval metric calculations."""

    def test_perfect_retrieval(self):
        """Perfect retrieval should have 1.0 scores."""
        metrics = calculate_retrieval_metrics(
            retrieved=["doc1", "doc2", "doc3"],
            expected=["doc1", "doc2", "doc3"],
        )
        assert metrics.recall == 1.0
        assert metrics.precision == 1.0
        assert metrics.f1_score == 1.0

    def test_no_overlap(self):
        """No overlap should have 0 scores."""
        metrics = calculate_retrieval_metrics(
            retrieved=["doc1", "doc2"],
            expected=["doc3", "doc4"],
        )
        assert metrics.recall == 0.0
        assert metrics.precision == 0.0
        assert metrics.f1_score == 0.0

    def test_partial_overlap(self):
        """Partial overlap should calculate correctly."""
        metrics = calculate_retrieval_metrics(
            retrieved=["doc1", "doc2", "doc3"],
            expected=["doc1", "doc2"],
        )
        # Retrieved 2 of 2 expected = 100% recall
        assert metrics.recall == 1.0
        # 2 of 3 retrieved were expected = 66.7% precision
        assert metrics.precision == pytest.approx(2/3, rel=0.01)

    def test_missing_docs_tracked(self):
        """Should track which docs were missing."""
        metrics = calculate_retrieval_metrics(
            retrieved=["doc1"],
            expected=["doc1", "doc2", "doc3"],
        )
        assert metrics.missing_docs == ["doc2", "doc3"] or \
               set(metrics.missing_docs) == {"doc2", "doc3"}

    def test_extra_docs_tracked(self):
        """Should track which docs were extra."""
        metrics = calculate_retrieval_metrics(
            retrieved=["doc1", "noise1", "noise2"],
            expected=["doc1"],
        )
        assert set(metrics.extra_docs) == {"noise1", "noise2"}


class TestIntegration:
    """Integration tests (require API key)."""

    @pytest.mark.skipif(
        not pytest.importorskip("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_agent_runs(self):
        """Agent should produce valid output."""
        from agent_eval_pipeline.agent import run_agent, AgentResult

        case = get_case_by_id("thyroid-001")
        result = run_agent(case)

        assert isinstance(result, AgentResult)
        assert result.output is not None
        assert len(result.output.key_insights) > 0
