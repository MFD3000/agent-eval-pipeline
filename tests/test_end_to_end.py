"""
End-to-End Smoke Tests

Validates the full pipeline integration: LangGraph agent + eval harness + InMemory store.

These tests ensure:
1. The harness runs agents once and shares contexts across evaluators
2. All eval gates can execute without errors
3. The system works with mock embeddings (no real API calls for retrieval)
"""

import pytest
import os


class TestLangGraphPipelineSmoke:
    """Smoke tests for the full LangGraph pipeline."""

    @pytest.fixture(autouse=True)
    def setup_mock_embeddings(self, monkeypatch):
        """Use mock embeddings to avoid real API calls for retrieval."""
        monkeypatch.setenv("USE_MOCK_EMBEDDINGS", "true")

    def test_langgraph_pipeline_smoke(self):
        """
        End-to-end smoke test: LangGraph agent + eval harness.

        This test validates:
        1. Harness builds AgentRunContexts correctly
        2. Schema eval receives and processes contexts
        3. Retrieval eval receives and processes contexts
        4. No duplicate agent runs occur

        Uses --skip-expensive equivalent by only running fast evals.
        """
        from agent_eval_pipeline.golden_sets import get_all_golden_cases
        from agent_eval_pipeline.harness.context import build_agent_contexts
        from agent_eval_pipeline.evals.schema_eval import run_schema_eval
        from agent_eval_pipeline.evals.retrieval_eval import run_retrieval_eval

        # Get a subset of cases for speed
        cases = get_all_golden_cases()[:2]

        # Build contexts once (this is what the harness does)
        contexts = build_agent_contexts(cases, verbose=False)

        # Verify contexts were built
        assert len(contexts) == 2
        for ctx in contexts:
            assert ctx.case is not None
            assert ctx.result is not None

        # Run schema eval with contexts (no re-running agents)
        schema_report = run_schema_eval(contexts=contexts, verbose=False)
        assert schema_report is not None
        assert schema_report.total_cases == 2

        # Run retrieval eval with contexts
        retrieval_report = run_retrieval_eval(contexts=contexts, verbose=False)
        assert retrieval_report is not None

    def test_context_sharing_prevents_duplicate_runs(self):
        """
        Verify that passing contexts prevents duplicate agent runs.

        This is the key benefit of AgentRunContext - evaluators should
        NOT call run_agent() internally when contexts are provided.
        """
        from unittest.mock import patch
        from agent_eval_pipeline.golden_sets import get_all_golden_cases
        from agent_eval_pipeline.harness.context import build_agent_contexts

        # Build real contexts (runs agents once)
        cases = get_all_golden_cases()[:1]
        contexts = build_agent_contexts(cases, verbose=False)

        # Patch run_agent to track if it's called again
        with patch("agent_eval_pipeline.evals.schema_eval.run_agent") as mock_run:
            from agent_eval_pipeline.evals.schema_eval import run_schema_eval

            # Run with contexts - should NOT call run_agent
            run_schema_eval(contexts=contexts, verbose=False)

            # Verify run_agent was never called (contexts were used instead)
            mock_run.assert_not_called()

    def test_harness_runner_integration(self):
        """
        Test the actual harness runner with skip-expensive mode.

        This validates the full integration path that CI would use.
        """
        from agent_eval_pipeline.harness.runner import run_all_evals

        # Run with skip_expensive=True (skips LLM-as-judge)
        # Note: run_all_evals uses golden cases internally
        report = run_all_evals(
            fail_fast=False,
            skip_expensive=True,
            verbose=False,
        )

        # Should have run schema and retrieval at minimum
        assert report is not None
        # Report has gates list
        assert hasattr(report, "gates")
        assert len(report.gates) > 0


class TestEvaluatorContextAcceptance:
    """Test that all evaluators accept and use AgentRunContext."""

    @pytest.fixture(scope="class")
    def real_contexts(self):
        """Build real contexts by running agents once (cached for class)."""
        from agent_eval_pipeline.golden_sets import get_all_golden_cases
        from agent_eval_pipeline.harness.context import build_agent_contexts

        cases = get_all_golden_cases()[:1]
        return build_agent_contexts(cases, verbose=False)

    def test_schema_eval_accepts_contexts(self, real_contexts):
        """Schema eval should accept contexts parameter."""
        from agent_eval_pipeline.evals.schema_eval import run_schema_eval

        report = run_schema_eval(contexts=real_contexts, verbose=False)
        assert report is not None
        assert report.total_cases == 1

    def test_retrieval_eval_accepts_contexts(self, real_contexts):
        """Retrieval eval should accept contexts parameter."""
        from agent_eval_pipeline.evals.retrieval_eval import run_retrieval_eval

        report = run_retrieval_eval(contexts=real_contexts, verbose=False)
        assert report is not None

    def test_perf_eval_accepts_contexts(self, real_contexts):
        """Perf eval should accept contexts parameter."""
        from agent_eval_pipeline.evals.perf.evaluator import run_perf_eval

        report = run_perf_eval(contexts=real_contexts, verbose=False)
        assert report is not None

    def test_ragas_eval_accepts_contexts_signature(self):
        """RAGAS eval should have contexts in its signature."""
        import inspect
        from agent_eval_pipeline.evals.ragas.evaluator import run_ragas_evaluation

        sig = inspect.signature(run_ragas_evaluation)
        assert "contexts" in sig.parameters

    def test_deepeval_eval_accepts_contexts_signature(self):
        """DeepEval eval should have contexts in its signature."""
        import inspect
        from agent_eval_pipeline.evals.deepeval.evaluator import run_deepeval_evaluation

        sig = inspect.signature(run_deepeval_evaluation)
        assert "contexts" in sig.parameters
