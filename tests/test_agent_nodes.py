"""
Unit Tests for Elevated Agent Nodes

These tests validate the individual node functions in isolation,
using mocks/fakes to avoid API calls.

STAFF ENGINEER PATTERNS:
------------------------
1. Each node tested in isolation with dependency injection
2. No API calls - uses mock embeddings and mock LLM
3. Tests document expected behavior through assertions
4. Edge cases covered (empty results, missing fields)
"""

import pytest
from unittest.mock import MagicMock, patch

from agent_eval_pipeline.agent.state import AgentState, create_initial_state
from agent_eval_pipeline.agent.nodes import (
    create_retrieve_node,
    create_analyze_node,
    apply_safety,
)
from agent_eval_pipeline.schemas.lab_insights import (
    LabInsightsSummary,
    MarkerInsight,
    SafetyNote,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample agent state for testing."""
    return create_initial_state(
        query="What do my thyroid results mean?",
        labs=[
            {
                "marker": "TSH",
                "value": 5.5,
                "unit": "mIU/L",
                "ref_low": 0.4,
                "ref_high": 4.0,
            },
            {
                "marker": "Free T4",
                "value": 0.9,
                "unit": "ng/dL",
                "ref_low": 0.8,
                "ref_high": 1.8,
            },
        ],
        history=[],
        symptoms=["fatigue"],
    )


@pytest.fixture
def sample_analysis() -> LabInsightsSummary:
    """Create a sample analysis output."""
    return LabInsightsSummary(
        summary="Your TSH is elevated at 5.5 mIU/L, above the normal range of 0.4-4.0.",
        key_insights=[
            MarkerInsight(
                marker="TSH",
                status="high",
                value=5.5,
                unit="mIU/L",
                ref_range="0.4-4.0",
                trend="unknown",
                clinical_relevance="Elevated TSH may indicate underactive thyroid",
                action="Discuss with your doctor",
            )
        ],
        recommended_topics_for_doctor=["Elevated TSH levels", "Fatigue symptoms"],
        lifestyle_considerations=["Monitor energy levels"],
        safety_notes=[],  # Empty - safety node will add these
    )


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    store = MagicMock()
    # The retrieve node uses search_by_markers, not search
    store.search_by_markers.return_value = [
        MagicMock(
            id="thyroid-tsh-001",
            title="TSH Overview",
            content="TSH info for testing",
            markers=["TSH"],
            score=0.9,
        ),
        MagicMock(
            id="thyroid-t4-001",
            title="Free T4 Overview",
            content="Free T4 info for testing",
            markers=["Free T4"],
            score=0.85,
        ),
    ]
    return store


# ---------------------------------------------------------------------------
# RETRIEVE NODE TESTS
# ---------------------------------------------------------------------------


class TestRetrieveNode:
    """Test the retrieval node function."""

    def test_retrieve_returns_docs(self, sample_state: AgentState, mock_vector_store):
        """Retrieve node should return documents from the store."""
        retrieve = create_retrieve_node(mock_vector_store)

        result = retrieve(sample_state)

        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) > 0
        mock_vector_store.search_by_markers.assert_called_once()

    def test_retrieve_passes_markers_to_store(self, sample_state: AgentState, mock_vector_store):
        """Retrieve node should pass the markers to the vector store."""
        retrieve = create_retrieve_node(mock_vector_store)

        retrieve(sample_state)

        # Verify search_by_markers was called with marker names
        call_args = mock_vector_store.search_by_markers.call_args
        # First arg should be list of markers
        markers = call_args[0][0]
        assert "TSH" in markers
        assert "Free T4" in markers

    def test_retrieve_handles_empty_results(self, sample_state: AgentState):
        """Retrieve node should handle empty search results."""
        empty_store = MagicMock()
        empty_store.search_by_markers.return_value = []

        retrieve = create_retrieve_node(empty_store)
        result = retrieve(sample_state)

        assert result["retrieved_docs"] == []


# ---------------------------------------------------------------------------
# ANALYZE NODE TESTS
# ---------------------------------------------------------------------------


class TestAnalyzeNode:
    """Test the analysis node function."""

    def test_analyze_produces_valid_schema(self, sample_state: AgentState):
        """Analyze node should produce valid LabInsightsSummary."""
        # The analyze node expects retrieved_docs with title, content, markers, score
        sample_state["retrieved_docs"] = [
            {
                "id": "doc1",
                "title": "TSH Overview",
                "content": "TSH info for thyroid function",
                "markers": ["TSH"],
                "score": 0.9,
            }
        ]

        with patch(
            "agent_eval_pipeline.agent.nodes.analyze.ChatOpenAI"
        ) as mock_chat:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_llm
            mock_llm.invoke.return_value = LabInsightsSummary(
                summary="Test summary",
                key_insights=[
                    MarkerInsight(
                        marker="TSH",
                        status="high",
                        value=5.5,
                        unit="mIU/L",
                        ref_range="0.4-4.0",
                        trend="unknown",
                        clinical_relevance="Test relevance",
                        action="Test action",
                    )
                ],
                recommended_topics_for_doctor=["Test topic"],
                lifestyle_considerations=["Test consideration"],
                safety_notes=[],
            )
            mock_chat.return_value = mock_llm

            analyze = create_analyze_node()
            result = analyze(sample_state)

            assert "raw_analysis" in result
            assert result["raw_analysis"]["summary"] == "Test summary"


# ---------------------------------------------------------------------------
# SAFETY NODE TESTS
# ---------------------------------------------------------------------------


class TestSafetyNode:
    """Test the safety node function."""

    def test_safety_adds_non_diagnostic_note(
        self, sample_state: AgentState, sample_analysis: LabInsightsSummary
    ):
        """Safety node should add non-diagnostic disclaimer."""
        # Safety node expects raw_analysis as a dict, returns final_output
        sample_state["raw_analysis"] = sample_analysis.model_dump()

        result = apply_safety(sample_state)

        assert "final_output" in result
        output = result["final_output"]

        # Should have at least one safety note
        assert len(output.safety_notes) > 0

        # Should include non-diagnostic disclaimer
        note_types = [note.type for note in output.safety_notes]
        assert "non_diagnostic" in note_types

    def test_safety_preserves_existing_notes(
        self, sample_state: AgentState, sample_analysis: LabInsightsSummary
    ):
        """Safety node should preserve any existing safety notes."""
        existing_note = SafetyNote(
            message="Existing note",
            type="lifestyle_scope",
        )
        sample_analysis.safety_notes = [existing_note]
        sample_state["raw_analysis"] = sample_analysis.model_dump()

        result = apply_safety(sample_state)

        output = result["final_output"]
        # Should have both existing and new notes
        assert len(output.safety_notes) >= 2
        assert any(note.message == "Existing note" for note in output.safety_notes)


# ---------------------------------------------------------------------------
# STATE CREATION TESTS
# ---------------------------------------------------------------------------


class TestStateCreation:
    """Test AgentState creation and validation."""

    def test_create_initial_state_minimal(self):
        """Should create state with minimal required fields."""
        state = create_initial_state(
            query="test query",
            labs=[{"marker": "TSH", "value": 1.0, "unit": "mIU/L"}],
        )

        assert state["query"] == "test query"
        assert len(state["labs"]) == 1

    def test_create_initial_state_with_optionals(self):
        """Should create state with optional fields."""
        state = create_initial_state(
            query="test query",
            labs=[{"marker": "TSH", "value": 1.0, "unit": "mIU/L"}],
            history=[{"marker": "TSH", "value": 0.9, "date": "2024-01-01"}],
            symptoms=["fatigue", "weight gain"],
        )

        assert len(state["history"]) == 1
        assert state["symptoms"] == ["fatigue", "weight gain"]

    def test_state_has_expected_keys(self):
        """State should have all expected keys for graph execution."""
        state = create_initial_state(
            query="test",
            labs=[],
        )

        # Core keys that should exist
        assert "query" in state
        assert "labs" in state
