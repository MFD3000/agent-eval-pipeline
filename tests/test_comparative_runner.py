"""
Unit Tests for Comparative Runner

Tests the agent comparison logic without making actual agent calls.
Uses mocks to verify comparison and scoring logic.

STAFF ENGINEER PATTERNS:
------------------------
1. Mock run_agent to avoid LLM calls
2. Test summary calculation logic
3. Test winner determination
4. Verify observability integration
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from agent_eval_pipeline.harness.comparative_runner import (
    AgentCaseResult,
    AgentSummary,
    ComparativeReport,
    run_agent_on_case,
    calculate_summary,
    determine_winner,
    run_comparative_eval,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_golden_case():
    """Create a mock golden case."""
    case = MagicMock()
    case.id = "thyroid-001"
    case.expected_marker_statuses = {"TSH": "high"}
    case.must_include_doctor_recommendation = True
    case.must_not_diagnose = True
    return case


@pytest.fixture
def mock_agent_result():
    """Create a mock successful agent result."""
    from agent_eval_pipeline.schemas.lab_insights import (
        LabInsightsSummary,
        MarkerInsight,
        SafetyNote,
    )

    output = LabInsightsSummary(
        summary="Your TSH is elevated.",
        key_insights=[
            MarkerInsight(
                marker="TSH",
                status="high",
                value=5.5,
                unit="mIU/L",
                ref_range="0.4-4.0",
                trend="unknown",
                clinical_relevance="Elevated TSH",
                action="Consult doctor",
            )
        ],
        recommended_topics_for_doctor=["Thyroid function"],
        lifestyle_considerations=["Monitor energy"],
        safety_notes=[SafetyNote(message="Not a diagnosis", type="non_diagnostic")],
    )

    result = MagicMock()
    result.output = output
    result.latency_ms = 1500.0
    result.input_tokens = 500
    result.output_tokens = 200
    result.total_tokens = 700
    result.tools_used = ["search_thyroid_info"]
    result.reasoning_steps = 3
    result.retrieved_docs = [{"id": "doc-1", "content": "TSH info"}]
    return result


@pytest.fixture
def mock_agent_error():
    """Create a mock agent error."""
    from agent_eval_pipeline.agent import AgentError
    return AgentError(
        error_type="TestError",
        error_message="Test error message",
    )


@pytest.fixture
def sample_case_results():
    """Create sample case results for both agents."""
    return [
        # LangGraph results
        AgentCaseResult(
            case_id="thyroid-001",
            agent_type="langgraph",
            success=True,
            latency_ms=1000.0,
            input_tokens=400,
            output_tokens=200,
            total_tokens=600,
            schema_passed=True,
            retrieved_docs=[{"id": "doc-1"}],
        ),
        AgentCaseResult(
            case_id="thyroid-002",
            agent_type="langgraph",
            success=True,
            latency_ms=1200.0,
            input_tokens=450,
            output_tokens=220,
            total_tokens=670,
            schema_passed=True,
            retrieved_docs=[{"id": "doc-2"}, {"id": "doc-3"}],
        ),
        # DSPy ReAct results
        AgentCaseResult(
            case_id="thyroid-001",
            agent_type="dspy_react",
            success=True,
            latency_ms=1500.0,
            input_tokens=600,
            output_tokens=300,
            total_tokens=900,
            schema_passed=True,
            tools_used=["search", "calculate"],
        ),
        AgentCaseResult(
            case_id="thyroid-002",
            agent_type="dspy_react",
            success=True,
            latency_ms=1800.0,
            input_tokens=650,
            output_tokens=320,
            total_tokens=970,
            schema_passed=False,
            tools_used=["search"],
        ),
    ]


# ---------------------------------------------------------------------------
# AGENT CASE RESULT TESTS
# ---------------------------------------------------------------------------


class TestAgentCaseResult:
    """Test AgentCaseResult dataclass."""

    def test_successful_result(self):
        """Should create successful result."""
        result = AgentCaseResult(
            case_id="test-001",
            agent_type="langgraph",
            success=True,
            latency_ms=1500.0,
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            schema_passed=True,
        )

        assert result.success is True
        assert result.error is None
        assert result.schema_passed is True

    def test_failed_result(self):
        """Should create failed result with error."""
        result = AgentCaseResult(
            case_id="test-001",
            agent_type="dspy_react",
            success=False,
            error="Agent timeout",
        )

        assert result.success is False
        assert result.error == "Agent timeout"


# ---------------------------------------------------------------------------
# CALCULATE SUMMARY TESTS
# ---------------------------------------------------------------------------


class TestCalculateSummary:
    """Test the calculate_summary function."""

    def test_calculate_langgraph_summary(self, sample_case_results):
        """Should calculate correct summary for LangGraph."""
        langgraph_results = [r for r in sample_case_results if r.agent_type == "langgraph"]

        summary = calculate_summary("langgraph", langgraph_results)

        assert summary.agent_type == "langgraph"
        assert summary.total_cases == 2
        assert summary.successful_cases == 2
        assert summary.failed_cases == 0
        assert summary.schema_pass_rate == 1.0
        assert summary.avg_docs_retrieved is not None  # LangGraph-specific

    def test_calculate_dspy_summary(self, sample_case_results):
        """Should calculate correct summary for DSPy ReAct."""
        dspy_results = [r for r in sample_case_results if r.agent_type == "dspy_react"]

        summary = calculate_summary("dspy_react", dspy_results)

        assert summary.agent_type == "dspy_react"
        assert summary.total_cases == 2
        assert summary.schema_pass_rate == 0.5  # 1 of 2 passed
        assert summary.avg_tools_used is not None  # DSPy-specific

    def test_calculate_latency_percentiles(self, sample_case_results):
        """Should calculate latency percentiles correctly."""
        langgraph_results = [r for r in sample_case_results if r.agent_type == "langgraph"]

        summary = calculate_summary("langgraph", langgraph_results)

        # With 2 values (1000, 1200), p50 should be 1000 or 1200
        assert summary.p50_latency_ms in [1000.0, 1200.0]
        assert summary.avg_latency_ms == 1100.0  # (1000+1200)/2

    def test_handle_empty_results(self):
        """Should handle empty results gracefully."""
        summary = calculate_summary("langgraph", [])

        assert summary.total_cases == 0
        assert summary.successful_cases == 0
        assert summary.schema_pass_rate == 0.0
        assert summary.avg_latency_ms == 0.0


# ---------------------------------------------------------------------------
# DETERMINE WINNER TESTS
# ---------------------------------------------------------------------------


class TestDetermineWinner:
    """Test the determine_winner function."""

    def test_winner_by_schema_pass_rate(self):
        """Should favor agent with higher schema pass rate."""
        summaries = {
            "langgraph": AgentSummary(
                agent_type="langgraph",
                total_cases=5,
                successful_cases=5,
                failed_cases=0,
                schema_pass_rate=1.0,  # Winner
                avg_latency_ms=1500.0,
                p50_latency_ms=1400.0,
                p95_latency_ms=1800.0,
                avg_tokens=700.0,
                total_tokens=3500,
            ),
            "dspy_react": AgentSummary(
                agent_type="dspy_react",
                total_cases=5,
                successful_cases=5,
                failed_cases=0,
                schema_pass_rate=0.6,  # Lower
                avg_latency_ms=1500.0,  # Same latency (to not give bonus points)
                p50_latency_ms=1400.0,
                p95_latency_ms=1800.0,
                avg_tokens=700.0,  # Same tokens (to not give bonus points)
                total_tokens=3500,
            ),
        }

        winner, recommendation = determine_winner(summaries)

        assert winner == "langgraph"
        assert "schema pass rate" in recommendation.lower()

    def test_winner_by_latency(self):
        """Should favor faster agent when schema rates equal."""
        summaries = {
            "langgraph": AgentSummary(
                agent_type="langgraph",
                total_cases=5,
                successful_cases=5,
                failed_cases=0,
                schema_pass_rate=0.8,
                avg_latency_ms=2000.0,  # Slower
                p50_latency_ms=1900.0,
                p95_latency_ms=2200.0,
                avg_tokens=700.0,
                total_tokens=3500,
            ),
            "dspy_react": AgentSummary(
                agent_type="dspy_react",
                total_cases=5,
                successful_cases=5,
                failed_cases=0,
                schema_pass_rate=0.8,  # Same
                avg_latency_ms=1000.0,  # Faster (>10% difference)
                p50_latency_ms=900.0,
                p95_latency_ms=1200.0,
                avg_tokens=700.0,
                total_tokens=3500,
            ),
        }

        winner, recommendation = determine_winner(summaries)

        assert winner == "dspy_react"
        assert "faster" in recommendation.lower()

    def test_tie_when_similar(self):
        """Should return tie when agents are similar."""
        summaries = {
            "langgraph": AgentSummary(
                agent_type="langgraph",
                total_cases=5,
                successful_cases=5,
                failed_cases=0,
                schema_pass_rate=0.8,
                avg_latency_ms=1000.0,
                p50_latency_ms=950.0,
                p95_latency_ms=1100.0,
                avg_tokens=700.0,
                total_tokens=3500,
            ),
            "dspy_react": AgentSummary(
                agent_type="dspy_react",
                total_cases=5,
                successful_cases=5,
                failed_cases=0,
                schema_pass_rate=0.8,
                avg_latency_ms=1050.0,  # Within 10%
                p50_latency_ms=1000.0,
                p95_latency_ms=1150.0,
                avg_tokens=720.0,  # Within 10%
                total_tokens=3600,
            ),
        }

        winner, recommendation = determine_winner(summaries)

        assert winner is None
        assert "similar" in recommendation.lower()

    def test_need_two_agents(self):
        """Should require at least 2 agents."""
        summaries = {
            "langgraph": AgentSummary(
                agent_type="langgraph",
                total_cases=5,
                successful_cases=5,
                failed_cases=0,
                schema_pass_rate=1.0,
                avg_latency_ms=1000.0,
                p50_latency_ms=950.0,
                p95_latency_ms=1100.0,
                avg_tokens=700.0,
                total_tokens=3500,
            ),
        }

        winner, recommendation = determine_winner(summaries)

        assert winner is None
        assert "2 agents" in recommendation


# ---------------------------------------------------------------------------
# COMPARATIVE REPORT TESTS
# ---------------------------------------------------------------------------


class TestComparativeReport:
    """Test ComparativeReport dataclass."""

    def test_report_to_dict(self):
        """Should serialize to dict correctly."""
        summary = AgentSummary(
            agent_type="langgraph",
            total_cases=5,
            successful_cases=5,
            failed_cases=0,
            schema_pass_rate=1.0,
            avg_latency_ms=1000.0,
            p50_latency_ms=950.0,
            p95_latency_ms=1100.0,
            avg_tokens=700.0,
            total_tokens=3500,
        )

        report = ComparativeReport(
            timestamp="2024-01-01T12:00:00",
            cases_evaluated=5,
            agents=["langgraph"],
            summaries={"langgraph": summary},
            case_results=[],
            winner="langgraph",
            recommendation="LangGraph performed better",
        )

        d = report.to_dict()

        assert d["timestamp"] == "2024-01-01T12:00:00"
        assert d["cases_evaluated"] == 5
        assert d["winner"] == "langgraph"
        assert "langgraph" in d["summaries"]


# ---------------------------------------------------------------------------
# RUN AGENT ON CASE TESTS
# ---------------------------------------------------------------------------


class TestRunAgentOnCase:
    """Test run_agent_on_case function."""

    def test_successful_run(self, mock_golden_case, mock_agent_result):
        """Should return successful result."""
        with patch("agent_eval_pipeline.harness.comparative_runner.run_agent") as mock_run:
            mock_run.return_value = mock_agent_result
            with patch("agent_eval_pipeline.harness.comparative_runner.get_tracer") as mock_tracer:
                mock_tracer.return_value = MagicMock()
                mock_tracer.return_value.start_span.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_tracer.return_value.start_span.return_value.__exit__ = MagicMock(return_value=False)

                result = run_agent_on_case(mock_golden_case, "langgraph")

        assert result.success is True
        assert result.case_id == "thyroid-001"
        assert result.agent_type == "langgraph"
        assert result.latency_ms == 1500.0

    def test_error_run(self, mock_golden_case, mock_agent_error):
        """Should handle agent error."""
        with patch("agent_eval_pipeline.harness.comparative_runner.run_agent") as mock_run:
            mock_run.return_value = mock_agent_error
            with patch("agent_eval_pipeline.harness.comparative_runner.get_tracer") as mock_tracer:
                mock_tracer.return_value = MagicMock()
                mock_tracer.return_value.start_span.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_tracer.return_value.start_span.return_value.__exit__ = MagicMock(return_value=False)

                result = run_agent_on_case(mock_golden_case, "langgraph")

        assert result.success is False
        assert "TestError" in result.error


# ---------------------------------------------------------------------------
# RUN COMPARATIVE EVAL TESTS
# ---------------------------------------------------------------------------


class TestRunComparativeEval:
    """Test run_comparative_eval function."""

    def test_runs_both_agents(self, mock_golden_case, mock_agent_result):
        """Should run both agents on all cases."""
        with patch("agent_eval_pipeline.harness.comparative_runner.get_all_golden_cases") as mock_cases:
            mock_cases.return_value = [mock_golden_case]
            with patch("agent_eval_pipeline.harness.comparative_runner.run_agent_on_case") as mock_run:
                mock_run.return_value = AgentCaseResult(
                    case_id="thyroid-001",
                    agent_type="langgraph",
                    success=True,
                    latency_ms=1000.0,
                    input_tokens=400,
                    output_tokens=200,
                    total_tokens=600,
                    schema_passed=True,
                )
                with patch("agent_eval_pipeline.harness.comparative_runner.get_tracer") as mock_tracer:
                    mock_tracer.return_value = MagicMock()
                    mock_tracer.return_value.start_span.return_value.__enter__ = MagicMock(return_value=MagicMock())
                    mock_tracer.return_value.start_span.return_value.__exit__ = MagicMock(return_value=False)

                    report = run_comparative_eval(
                        agents=["langgraph", "dspy_react"],
                        verbose=False,
                    )

        # Should have called run_agent_on_case for each agent on each case
        assert mock_run.call_count == 2  # 2 agents * 1 case
        assert report.cases_evaluated == 1
        assert len(report.agents) == 2
