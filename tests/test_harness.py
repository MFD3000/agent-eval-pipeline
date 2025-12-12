"""
Unit Tests for Eval Harness

Tests the evaluation orchestration without making actual LLM calls.
Uses mocks to verify the harness logic in isolation.

STAFF ENGINEER PATTERNS:
------------------------
1. Mock the expensive eval gates to test orchestration logic
2. Verify fail-fast behavior
3. Verify gate ordering
4. Test report generation
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from agent_eval_pipeline.harness.runner import (
    run_gate,
    GateStatus,
    GateResult,
    HarnessReport,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@dataclass
class MockEvalReport:
    """Mock eval report for testing."""

    all_passed: bool = True
    pass_rate: float = 1.0


# ---------------------------------------------------------------------------
# RUN_GATE TESTS
# ---------------------------------------------------------------------------


class TestRunGate:
    """Test the run_gate function."""

    def test_run_gate_success(self):
        """Gate should return PASSED when runner succeeds."""

        def mock_runner():
            return MockEvalReport(all_passed=True, pass_rate=1.0)

        result, report = run_gate("Test Gate", mock_runner)

        assert result.name == "Test Gate"
        assert result.status == GateStatus.PASSED
        assert result.duration_ms >= 0

    def test_run_gate_failure(self):
        """Gate should return FAILED when runner reports failure."""

        def mock_runner():
            return MockEvalReport(all_passed=False, pass_rate=0.6)

        result, report = run_gate("Test Gate", mock_runner)

        assert result.status == GateStatus.FAILED

    def test_run_gate_error(self):
        """Gate should return ERROR when runner raises exception."""

        def mock_runner():
            raise ValueError("Something went wrong")

        result, report = run_gate("Test Gate", mock_runner)

        assert result.status == GateStatus.ERROR
        assert "Something went wrong" in result.summary
        assert report is None

    def test_run_gate_skip(self):
        """Gate should return SKIPPED when skip=True."""

        def mock_runner():
            return MockEvalReport()

        result, report = run_gate("Test Gate", mock_runner, skip=True)

        assert result.status == GateStatus.SKIPPED
        assert result.duration_ms == 0
        assert report is None


# ---------------------------------------------------------------------------
# HARNESS REPORT TESTS
# ---------------------------------------------------------------------------


class TestHarnessReport:
    """Test HarnessReport serialization."""

    def test_to_dict_basic(self):
        """Report should serialize to dict correctly."""
        report = HarnessReport(
            timestamp="2024-01-01T12:00:00",
            total_duration_ms=5000.0,
            all_passed=True,
            gates=[
                GateResult(
                    name="Schema",
                    status=GateStatus.PASSED,
                    duration_ms=100.0,
                    summary="Pass rate: 100%",
                )
            ],
            summary="1 passed, 0 failed",
        )

        d = report.to_dict()

        assert d["timestamp"] == "2024-01-01T12:00:00"
        assert d["total_duration_ms"] == 5000.0
        assert d["all_passed"] is True
        assert len(d["gates"]) == 1
        assert d["gates"][0]["name"] == "Schema"
        assert d["gates"][0]["status"] == "passed"

    def test_to_dict_with_multiple_gates(self):
        """Report with multiple gates should serialize correctly."""
        report = HarnessReport(
            timestamp="2024-01-01T12:00:00",
            total_duration_ms=10000.0,
            all_passed=False,
            gates=[
                GateResult("Schema", GateStatus.PASSED, 100.0, "Pass rate: 100%"),
                GateResult("Retrieval", GateStatus.FAILED, 200.0, "Avg F1: 0.50"),
                GateResult("Judge", GateStatus.SKIPPED, 0.0, "Skipped"),
            ],
            summary="1 passed, 1 failed",
        )

        d = report.to_dict()

        assert len(d["gates"]) == 3
        assert d["gates"][0]["status"] == "passed"
        assert d["gates"][1]["status"] == "failed"
        assert d["gates"][2]["status"] == "skipped"


# ---------------------------------------------------------------------------
# GATE STATUS TESTS
# ---------------------------------------------------------------------------


class TestGateStatus:
    """Minimal regression tests for GateStatus."""

    def test_gate_status_used_in_gate_result(self):
        """Verify GateStatus integrates with GateResult behaviorally."""
        result = GateResult(
            name="schema",
            status=GateStatus.PASSED,
            duration_ms=1.0,
            summary="ok",
        )

        report = HarnessReport(
            timestamp="2024-01-01T00:00:00",
            total_duration_ms=2.0,
            all_passed=True,
            gates=[result],
            summary="1 passed",
        )

        serialized = report.to_dict()

        assert serialized["gates"][0]["status"] == GateStatus.PASSED.value
