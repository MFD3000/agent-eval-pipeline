"""
Unit Tests for AgentRunContext Sharing

Tests the context sharing mechanism that enables running agents ONCE
and sharing results across all evaluators.

WHAT'S BEING TESTED:
-------------------
1. AgentRunContext creation and properties
2. EvalRunContext container operations
3. Evaluators accepting contexts (schema, judge, perf, retrieval)
4. Context building from agent runs
5. Actual retrieval validation mode
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from agent_eval_pipeline.harness.context import (
    AgentRunContext,
    EvalRunContext,
    build_agent_contexts,
    build_eval_run_context,
)
from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase, LabValue
from agent_eval_pipeline.schemas.lab_insights import (
    LabInsightsSummary,
    MarkerInsight,
    SafetyNote,
)
from agent_eval_pipeline.agent import AgentResult, AgentError


# ---------------------------------------------------------------------------
# TEST FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_case() -> GoldenCase:
    """Create a mock golden case for testing."""
    return GoldenCase(
        id="test-001",
        description="Test case",
        member_id="member-123",
        query="What do my TSH results mean?",
        labs=[
            LabValue(
                date="2024-01-15",
                marker="TSH",
                value=8.5,
                unit="mIU/L",
                ref_low=0.4,
                ref_high=4.0,
            ),
        ],
        expected_marker_statuses={"TSH": "high"},
        expected_doc_ids=["doc_thyroid_101", "doc_tsh_interpretation"],
    )


@pytest.fixture
def mock_output() -> LabInsightsSummary:
    """Create a mock agent output."""
    return LabInsightsSummary(
        key_insights=[
            MarkerInsight(
                marker="TSH",
                value=8.5,
                unit="mIU/L",
                status="high",
                ref_range="0.4-4.0",
                trend="stable",
                clinical_relevance="TSH controls thyroid function. High TSH may indicate underactive thyroid.",
                action="Discuss with doctor",
            )
        ],
        summary="Elevated TSH detected.",
        recommended_topics_for_doctor=["thyroid function"],
        lifestyle_considerations=["Regular exercise", "Healthy diet"],
        safety_notes=[
            SafetyNote(
                type="non_diagnostic",
                message="Consult your doctor for proper evaluation.",
            )
        ],
    )


@pytest.fixture
def mock_agent_result(mock_output) -> AgentResult:
    """Create a mock successful agent result."""
    return AgentResult(
        output=mock_output,
        latency_ms=1500.0,
        total_tokens=1000,
        input_tokens=600,
        output_tokens=400,
        model="gpt-4o-mini",
        agent_type="langgraph",
        retrieved_docs=[
            {"id": "doc_thyroid_101", "title": "Thyroid Guide"},
            {"id": "doc_tsh_interpretation", "title": "TSH Interpretation"},
        ],
    )


@pytest.fixture
def mock_agent_error() -> AgentError:
    """Create a mock agent error."""
    return AgentError(
        error_type="API Error",
        error_message="Rate limit exceeded",
    )


# ---------------------------------------------------------------------------
# AGENT RUN CONTEXT TESTS
# ---------------------------------------------------------------------------


class TestAgentRunContext:
    """Test AgentRunContext dataclass."""

    def test_successful_context_properties(self, mock_case, mock_agent_result):
        """Successful context should expose result properties."""
        ctx = AgentRunContext(case=mock_case, result=mock_agent_result)

        assert ctx.case_id == "test-001"
        assert ctx.success is True
        assert ctx.output is not None
        assert ctx.output.summary == "Elevated TSH detected."
        assert ctx.latency_ms == 1500.0
        assert ctx.total_tokens == 1000
        assert ctx.input_tokens == 600
        assert ctx.output_tokens == 400
        assert ctx.model == "gpt-4o-mini"
        assert ctx.error_message is None

    def test_failed_context_properties(self, mock_case, mock_agent_error):
        """Failed context should handle error gracefully."""
        ctx = AgentRunContext(case=mock_case, result=mock_agent_error)

        assert ctx.case_id == "test-001"
        assert ctx.success is False
        assert ctx.output is None
        assert ctx.latency_ms is None
        assert ctx.total_tokens is None
        assert ctx.model is None
        assert "Rate limit exceeded" in ctx.error_message

    def test_retrieved_docs_success(self, mock_case, mock_agent_result):
        """Successful context should expose retrieved docs."""
        ctx = AgentRunContext(case=mock_case, result=mock_agent_result)

        assert len(ctx.retrieved_docs) == 2
        assert ctx.retrieved_docs[0]["id"] == "doc_thyroid_101"

    def test_retrieved_docs_failure(self, mock_case, mock_agent_error):
        """Failed context should return empty retrieved docs."""
        ctx = AgentRunContext(case=mock_case, result=mock_agent_error)

        assert ctx.retrieved_docs == []

    def test_timestamp_auto_populated(self, mock_case, mock_agent_result):
        """Context should auto-populate timestamp."""
        ctx = AgentRunContext(case=mock_case, result=mock_agent_result)

        assert ctx.timestamp is not None
        assert len(ctx.timestamp) > 0  # ISO format


# ---------------------------------------------------------------------------
# EVAL RUN CONTEXT TESTS
# ---------------------------------------------------------------------------


class TestEvalRunContext:
    """Test EvalRunContext container."""

    def test_add_and_get_context(self, mock_case, mock_agent_result):
        """Should add and retrieve contexts by case_id."""
        eval_ctx = EvalRunContext()
        agent_ctx = AgentRunContext(case=mock_case, result=mock_agent_result)

        eval_ctx.add_context(agent_ctx)

        retrieved = eval_ctx.get_context("test-001")
        assert retrieved is not None
        assert retrieved.case_id == "test-001"

    def test_get_nonexistent_context(self):
        """Should return None for nonexistent case_id."""
        eval_ctx = EvalRunContext()

        assert eval_ctx.get_context("nonexistent") is None

    def test_contexts_list(self, mock_case, mock_agent_result):
        """Should provide contexts as list."""
        eval_ctx = EvalRunContext()
        agent_ctx = AgentRunContext(case=mock_case, result=mock_agent_result)
        eval_ctx.add_context(agent_ctx)

        contexts = eval_ctx.contexts
        assert len(contexts) == 1
        assert contexts[0].case_id == "test-001"

    def test_successful_and_failed_contexts(self, mock_case, mock_agent_result, mock_agent_error):
        """Should separate successful and failed contexts."""
        eval_ctx = EvalRunContext()

        # Add successful context
        success_ctx = AgentRunContext(case=mock_case, result=mock_agent_result)
        eval_ctx.add_context(success_ctx)

        # Add failed context with different case
        failed_case = GoldenCase(
            id="test-002",
            description="Failed test",
            member_id="member-456",
            query="Test query",
            labs=[LabValue(date="2024-01-15", marker="TSH", value=5.0, unit="mIU/L")],
        )
        failed_ctx = AgentRunContext(case=failed_case, result=mock_agent_error)
        eval_ctx.add_context(failed_ctx)

        assert len(eval_ctx.successful_contexts) == 1
        assert len(eval_ctx.failed_contexts) == 1
        assert eval_ctx.success_rate == 0.5

    def test_run_id_auto_generated(self):
        """Should auto-generate unique run_id."""
        ctx1 = EvalRunContext()
        ctx2 = EvalRunContext()

        assert ctx1.run_id != ctx2.run_id
        assert len(ctx1.run_id) > 0


# ---------------------------------------------------------------------------
# SCHEMA EVAL WITH CONTEXT TESTS
# ---------------------------------------------------------------------------


class TestSchemaEvalWithContext:
    """Test schema_eval accepts contexts."""

    def test_run_schema_eval_with_contexts(self, mock_case, mock_agent_result):
        """Schema eval should use pre-computed contexts."""
        from agent_eval_pipeline.evals.schema_eval import run_schema_eval

        ctx = AgentRunContext(case=mock_case, result=mock_agent_result)

        # Run with contexts - should NOT call run_agent
        report = run_schema_eval(contexts=[ctx])

        assert report.total_cases == 1
        # Should pass because output matches expected marker status
        assert report.passed_cases == 1

    def test_run_schema_eval_with_failed_context(self, mock_case, mock_agent_error):
        """Schema eval should handle failed contexts."""
        from agent_eval_pipeline.evals.schema_eval import run_schema_eval

        ctx = AgentRunContext(case=mock_case, result=mock_agent_error)

        report = run_schema_eval(contexts=[ctx])

        assert report.total_cases == 1
        assert report.failed_cases == 1
        assert "Rate limit exceeded" in report.results[0].error


# ---------------------------------------------------------------------------
# PERF EVAL WITH CONTEXT TESTS
# ---------------------------------------------------------------------------


class TestPerfEvalWithContext:
    """Test perf_eval accepts contexts."""

    def test_run_perf_eval_with_contexts(self, mock_case, mock_agent_result):
        """Perf eval should use pre-computed contexts."""
        from agent_eval_pipeline.evals.perf.evaluator import run_perf_eval
        from agent_eval_pipeline.evals.perf.baseline import InMemoryBaselineStore

        ctx = AgentRunContext(case=mock_case, result=mock_agent_result)
        store = InMemoryBaselineStore()

        # Run with contexts - should NOT call run_agent
        result = run_perf_eval(contexts=[ctx], baseline_store=store)

        assert len(result.case_results) == 1
        assert result.case_results[0].success is True
        assert result.case_results[0].latency_ms == 1500.0
        assert result.case_results[0].total_tokens == 1000

    def test_run_perf_eval_with_failed_context(self, mock_case, mock_agent_error):
        """Perf eval should handle failed contexts."""
        from agent_eval_pipeline.evals.perf.evaluator import run_perf_eval
        from agent_eval_pipeline.evals.perf.baseline import InMemoryBaselineStore

        ctx = AgentRunContext(case=mock_case, result=mock_agent_error)
        store = InMemoryBaselineStore()

        result = run_perf_eval(contexts=[ctx], baseline_store=store)

        assert len(result.case_results) == 1
        assert result.case_results[0].success is False


# ---------------------------------------------------------------------------
# RETRIEVAL EVAL WITH ACTUAL RETRIEVAL TESTS
# ---------------------------------------------------------------------------


class TestRetrievalEvalActualRetrieval:
    """Test retrieval_eval with actual retrieval mode."""

    def test_actual_retrieval_with_matching_docs(self, mock_case, mock_agent_result):
        """Actual retrieval should calculate metrics from real retrieved docs."""
        from agent_eval_pipeline.evals.retrieval_eval import run_retrieval_eval

        ctx = AgentRunContext(case=mock_case, result=mock_agent_result)

        # Run with actual retrieval mode
        report = run_retrieval_eval(
            contexts=[ctx],
            use_actual_retrieval=True,
            threshold=0.5,
        )

        assert report.total_cases == 1
        # Should pass because retrieved docs match expected
        assert report.passed_cases == 1
        assert report.avg_f1 > 0.5

    def test_actual_retrieval_falls_back_to_simulated(self, mock_case):
        """Without contexts, should use simulated retrieval."""
        from agent_eval_pipeline.evals.retrieval_eval import run_retrieval_eval

        # Run without contexts - uses simulated
        report = run_retrieval_eval(
            cases=[mock_case],
            use_actual_retrieval=False,
            threshold=0.5,
            seed=42,
        )

        assert report.total_cases == 1

    def test_actual_retrieval_skips_failed_contexts(self, mock_case, mock_agent_error):
        """Actual retrieval should skip failed agent runs."""
        from agent_eval_pipeline.evals.retrieval_eval import run_retrieval_eval

        ctx = AgentRunContext(case=mock_case, result=mock_agent_error)

        report = run_retrieval_eval(
            contexts=[ctx],
            use_actual_retrieval=True,
        )

        # Should skip the failed context
        assert report.total_cases == 0


# ---------------------------------------------------------------------------
# BUILD CONTEXTS TESTS
# ---------------------------------------------------------------------------


class TestBuildContexts:
    """Test context building functions."""

    @patch("agent_eval_pipeline.agent.run_agent")
    def test_build_agent_contexts(self, mock_run_agent, mock_case, mock_agent_result):
        """build_agent_contexts should create contexts from agent runs."""
        mock_run_agent.return_value = mock_agent_result

        contexts = build_agent_contexts([mock_case])

        assert len(contexts) == 1
        assert contexts[0].case_id == "test-001"
        assert contexts[0].success is True
        mock_run_agent.assert_called_once_with(mock_case)

    @patch("agent_eval_pipeline.agent.run_agent")
    def test_build_eval_run_context(self, mock_run_agent, mock_case, mock_agent_result):
        """build_eval_run_context should create EvalRunContext with all results."""
        mock_run_agent.return_value = mock_agent_result

        eval_ctx = build_eval_run_context([mock_case])

        assert len(eval_ctx.contexts) == 1
        assert eval_ctx.get_context("test-001") is not None
        assert eval_ctx.success_rate == 1.0


# ---------------------------------------------------------------------------
# INTEGRATION TEST
# ---------------------------------------------------------------------------


class TestContextSharingIntegration:
    """Integration test for full context sharing flow."""

    def test_contexts_shared_across_evaluators(self, mock_case, mock_agent_result):
        """Same context should be usable across multiple evaluators."""
        from agent_eval_pipeline.evals.schema_eval import run_schema_eval
        from agent_eval_pipeline.evals.perf.evaluator import run_perf_eval
        from agent_eval_pipeline.evals.perf.baseline import InMemoryBaselineStore

        # Create context once
        ctx = AgentRunContext(case=mock_case, result=mock_agent_result)
        contexts = [ctx]

        # Use in schema eval
        schema_report = run_schema_eval(contexts=contexts)
        assert schema_report.total_cases == 1

        # Use same contexts in perf eval
        perf_result = run_perf_eval(
            contexts=contexts,
            baseline_store=InMemoryBaselineStore(),
        )
        assert len(perf_result.case_results) == 1

        # Both should see the same case_id
        assert schema_report.results[0].case_id == perf_result.case_results[0].case_id
