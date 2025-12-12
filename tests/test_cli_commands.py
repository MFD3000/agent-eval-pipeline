"""
Unit Tests for CLI Commands

Tests the CLI entry points without making actual eval runs.
Uses mocks to verify the CLI orchestration logic.

STAFF ENGINEER PATTERNS:
------------------------
1. Mock the expensive eval functions
2. Test CLI argument parsing
3. Verify exit codes
4. Test error handling
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


# ---------------------------------------------------------------------------
# LOAD_ENV TESTS
# ---------------------------------------------------------------------------


class TestLoadEnv:
    """Test environment loading."""

    def test_load_env_does_not_raise(self):
        """Should not raise even if dotenv missing."""
        from agent_eval_pipeline.cli.commands import _load_env
        # Should not raise
        _load_env()


# ---------------------------------------------------------------------------
# MAIN CLI DISPATCH TESTS
# ---------------------------------------------------------------------------


class TestMainCliDispatch:
    """Test main CLI dispatches to correct handlers."""

    def test_main_dispatches_to_schema(self):
        """Main should dispatch 'schema' to run_schema_cli."""
        from agent_eval_pipeline.cli import commands

        with patch.object(commands, "run_schema_cli") as mock_schema:
            mock_schema.return_value = 0
            with patch("sys.argv", ["agent-eval", "schema"]):
                result = commands.main()

            mock_schema.assert_called_once()
            assert result == 0

    def test_main_dispatches_to_retrieval(self):
        """Main should dispatch 'retrieval' to run_retrieval_cli."""
        from agent_eval_pipeline.cli import commands

        with patch.object(commands, "run_retrieval_cli") as mock_retrieval:
            mock_retrieval.return_value = 0
            with patch("sys.argv", ["agent-eval", "retrieval"]):
                result = commands.main()

            mock_retrieval.assert_called_once()
            assert result == 0

    def test_main_dispatches_to_judge(self):
        """Main should dispatch 'judge' to run_judge_cli."""
        from agent_eval_pipeline.cli import commands

        with patch.object(commands, "run_judge_cli") as mock_judge:
            mock_judge.return_value = 0
            with patch("sys.argv", ["agent-eval", "judge"]):
                result = commands.main()

            mock_judge.assert_called_once()
            assert result == 0

    def test_main_dispatches_to_perf(self):
        """Main should dispatch 'perf' to run_perf_cli."""
        from agent_eval_pipeline.cli import commands

        with patch.object(commands, "run_perf_cli") as mock_perf:
            mock_perf.return_value = 0
            with patch("sys.argv", ["agent-eval", "perf"]):
                result = commands.main()

            mock_perf.assert_called_once()
            assert result == 0

    def test_main_dispatches_to_all(self):
        """Main should dispatch 'all' to run_all_cli."""
        from agent_eval_pipeline.cli import commands

        with patch.object(commands, "run_all_cli") as mock_all:
            mock_all.return_value = 0
            with patch("sys.argv", ["agent-eval", "all"]):
                result = commands.main()

            mock_all.assert_called_once()
            assert result == 0

    def test_main_handles_keyboard_interrupt(self):
        """Main should return 130 on KeyboardInterrupt."""
        from agent_eval_pipeline.cli import commands

        with patch.object(commands, "run_schema_cli") as mock_schema:
            mock_schema.side_effect = KeyboardInterrupt()
            with patch("sys.argv", ["agent-eval", "schema"]):
                result = commands.main()

            assert result == 130
