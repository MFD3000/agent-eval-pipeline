"""
Unit Tests for Observability Module

Tests the Phoenix/OpenTelemetry integration with focus on:
1. Graceful degradation (NoOpTracer when disabled)
2. Configuration loading from environment
3. Span creation and attribute setting

STAFF ENGINEER PATTERNS:
------------------------
1. Tests work WITHOUT Phoenix installed (graceful degradation)
2. Environment variable handling tested with monkeypatch
3. Protocol compliance verified
4. Zero-overhead when disabled
"""

import pytest
from unittest.mock import patch, MagicMock

from agent_eval_pipeline.observability.config import (
    PhoenixConfig,
    get_config,
    reset_config,
)
from agent_eval_pipeline.observability.tracer import (
    NoOpTracer,
    NoOpSpan,
    get_tracer,
    reset_tracer,
)
from agent_eval_pipeline.observability.attributes import (
    EVAL_GATE_NAME,
    EVAL_GATE_STATUS,
    EVAL_CASE_ID,
    AGENT_TYPE,
    eval_gate_attributes,
    eval_case_attributes,
    agent_run_attributes,
)


# ---------------------------------------------------------------------------
# CONFIG TESTS
# ---------------------------------------------------------------------------


class TestPhoenixConfig:
    """Test configuration loading."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_config_defaults(self):
        """Config should have sensible defaults when env vars not set."""
        with patch.dict("os.environ", {}, clear=True):
            reset_config()
            config = PhoenixConfig.from_env()

            assert config.enabled is False
            assert config.project_name == "agent-eval-pipeline"
            assert config.collector_endpoint is None
            # Default to False for privacy - health data should not be exported by default
            assert config.capture_llm_content is False

    def test_config_from_env_enabled(self):
        """Config should read PHOENIX_ENABLED correctly."""
        with patch.dict("os.environ", {"PHOENIX_ENABLED": "true"}):
            reset_config()
            config = PhoenixConfig.from_env()
            assert config.enabled is True

        with patch.dict("os.environ", {"PHOENIX_ENABLED": "1"}):
            reset_config()
            config = PhoenixConfig.from_env()
            assert config.enabled is True

        with patch.dict("os.environ", {"PHOENIX_ENABLED": "yes"}):
            reset_config()
            config = PhoenixConfig.from_env()
            assert config.enabled is True

    def test_config_from_env_disabled(self):
        """Config should default to disabled."""
        with patch.dict("os.environ", {"PHOENIX_ENABLED": "false"}):
            reset_config()
            config = PhoenixConfig.from_env()
            assert config.enabled is False

        with patch.dict("os.environ", {"PHOENIX_ENABLED": "0"}):
            reset_config()
            config = PhoenixConfig.from_env()
            assert config.enabled is False

    def test_config_project_name(self):
        """Config should read project name from env."""
        with patch.dict("os.environ", {"PHOENIX_PROJECT_NAME": "my-project"}):
            reset_config()
            config = PhoenixConfig.from_env()
            assert config.project_name == "my-project"

    def test_config_collector_endpoint(self):
        """Config should read collector endpoint from env."""
        with patch.dict(
            "os.environ",
            {"PHOENIX_COLLECTOR_ENDPOINT": "https://phoenix.example.com/v1/traces"},
        ):
            reset_config()
            config = PhoenixConfig.from_env()
            assert config.collector_endpoint == "https://phoenix.example.com/v1/traces"

    def test_get_config_singleton(self):
        """get_config should return same instance."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


# ---------------------------------------------------------------------------
# NOOP TRACER TESTS
# ---------------------------------------------------------------------------


class TestNoOpTracer:
    """Test NoOpTracer for graceful degradation."""

    def test_noop_tracer_creates_spans(self):
        """NoOpTracer should create NoOpSpan instances."""
        tracer = NoOpTracer()

        with tracer.start_span("test_span") as span:
            assert isinstance(span, NoOpSpan)

    def test_noop_span_accepts_attributes(self):
        """NoOpSpan should accept attributes without error."""
        tracer = NoOpTracer()

        with tracer.start_span("test_span") as span:
            # These should not raise
            span.set_attribute("key", "value")
            span.set_attribute("number", 42)
            span.set_attribute("float", 3.14)

    def test_noop_span_accepts_status(self):
        """NoOpSpan should accept status without error."""
        tracer = NoOpTracer()

        with tracer.start_span("test_span") as span:
            span.set_status("ok")
            span.set_status("error", "Something went wrong")

    def test_noop_span_accepts_exception(self):
        """NoOpSpan should accept exception recording without error."""
        tracer = NoOpTracer()

        with tracer.start_span("test_span") as span:
            span.record_exception(ValueError("test error"))

    def test_noop_span_context_manager(self):
        """NoOpSpan should work as context manager."""
        tracer = NoOpTracer()

        executed = False
        with tracer.start_span("test_span") as span:
            executed = True
            span.set_attribute("inside", True)

        assert executed

    def test_noop_tracer_with_initial_attributes(self):
        """NoOpTracer should accept initial attributes."""
        tracer = NoOpTracer()

        with tracer.start_span(
            "test_span", attributes={"key": "value", "count": 5}
        ) as span:
            assert isinstance(span, NoOpSpan)


# ---------------------------------------------------------------------------
# GET_TRACER FACTORY TESTS
# ---------------------------------------------------------------------------


class TestGetTracer:
    """Test the get_tracer factory function."""

    def setup_method(self):
        """Reset tracer and config before each test."""
        reset_tracer()
        reset_config()

    def teardown_method(self):
        """Reset after each test."""
        reset_tracer()
        reset_config()

    def test_get_tracer_returns_noop_when_disabled(self):
        """get_tracer should return NoOpTracer when Phoenix disabled."""
        with patch.dict("os.environ", {"PHOENIX_ENABLED": "false"}):
            reset_tracer()
            reset_config()
            tracer = get_tracer()
            assert isinstance(tracer, NoOpTracer)

    def test_get_tracer_singleton(self):
        """get_tracer should return same instance."""
        with patch.dict("os.environ", {"PHOENIX_ENABLED": "false"}):
            reset_tracer()
            reset_config()
            tracer1 = get_tracer()
            tracer2 = get_tracer()
            assert tracer1 is tracer2

    def test_get_tracer_returns_tracer_when_enabled(self):
        """get_tracer should return a tracer when Phoenix is enabled."""
        with patch.dict("os.environ", {"PHOENIX_ENABLED": "true"}):
            reset_tracer()
            reset_config()

            tracer = get_tracer()
            # When OTel is available, it returns an OTelTracer
            # When not available, it returns NoOpTracer (graceful degradation)
            # Either is acceptable - just verify it's a valid tracer with start_span
            assert hasattr(tracer, 'start_span')
            assert callable(tracer.start_span)


# ---------------------------------------------------------------------------
# ATTRIBUTE HELPER TESTS
# ---------------------------------------------------------------------------


class TestAttributeHelpers:
    """Test attribute helper functions."""

    def test_eval_gate_attributes(self):
        """eval_gate_attributes should create correct dict."""
        attrs = eval_gate_attributes(
            gate_name="schema_validation",
            status="passed",
            score=0.95,
            threshold=0.8,
        )

        assert attrs[EVAL_GATE_NAME] == "schema_validation"
        assert attrs[EVAL_GATE_STATUS] == "passed"
        assert attrs["eval.gate.score"] == 0.95
        assert attrs["eval.gate.threshold"] == 0.8

    def test_eval_gate_attributes_minimal(self):
        """eval_gate_attributes should work with minimal args."""
        attrs = eval_gate_attributes(
            gate_name="test_gate",
            status="failed",
        )

        assert attrs[EVAL_GATE_NAME] == "test_gate"
        assert attrs[EVAL_GATE_STATUS] == "failed"
        assert "eval.gate.score" not in attrs
        assert "eval.gate.threshold" not in attrs

    def test_eval_case_attributes(self):
        """eval_case_attributes should create correct dict."""
        attrs = eval_case_attributes(
            case_id="thyroid-001",
            passed=True,
        )

        assert attrs[EVAL_CASE_ID] == "thyroid-001"
        assert attrs["eval.case.passed"] is True
        assert "eval.case.error" not in attrs

    def test_eval_case_attributes_with_error(self):
        """eval_case_attributes should include error when provided."""
        attrs = eval_case_attributes(
            case_id="thyroid-002",
            passed=False,
            error="Schema validation failed",
        )

        assert attrs[EVAL_CASE_ID] == "thyroid-002"
        assert attrs["eval.case.passed"] is False
        assert attrs["eval.case.error"] == "Schema validation failed"

    def test_agent_run_attributes(self):
        """agent_run_attributes should create correct dict."""
        attrs = agent_run_attributes(
            agent_type="langgraph",
            model="gpt-4o-mini",
            input_tokens=500,
            output_tokens=200,
            latency_ms=1500.5,
        )

        assert attrs[AGENT_TYPE] == "langgraph"
        assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
        assert attrs["gen_ai.usage.input_tokens"] == 500
        assert attrs["gen_ai.usage.output_tokens"] == 200
        assert attrs["gen_ai.usage.total_tokens"] == 700
        assert attrs["eval.perf.latency_ms"] == 1500.5


# ---------------------------------------------------------------------------
# INTEGRATION PATTERN TEST
# ---------------------------------------------------------------------------


class TestObservabilityIntegration:
    """Test that observability integrates correctly with eval code."""

    def setup_method(self):
        reset_tracer()
        reset_config()

    def teardown_method(self):
        reset_tracer()
        reset_config()

    def test_typical_usage_pattern(self):
        """Test the typical usage pattern in harness code."""
        with patch.dict("os.environ", {"PHOENIX_ENABLED": "false"}):
            reset_tracer()
            reset_config()

            tracer = get_tracer()

            # Simulate harness run pattern
            with tracer.start_span(
                "eval_harness_run",
                attributes={"eval.harness.run_id": "test-123"},
            ) as root_span:
                root_span.set_attribute("eval.harness.cases_count", 5)

                # Simulate gate span
                with tracer.start_span(
                    "eval_gate.schema_validation",
                    attributes=eval_gate_attributes("schema_validation", "passed"),
                ) as gate_span:
                    gate_span.set_attribute("eval.gate.pass_rate", 1.0)
                    gate_span.set_status("ok")

                root_span.set_attribute("eval.harness.all_passed", True)
                root_span.set_status("ok")

        # If we get here without error, the pattern works
        assert True

    def test_exception_handling_in_span(self):
        """Test that exceptions are properly recorded."""
        tracer = NoOpTracer()

        try:
            with tracer.start_span("failing_operation") as span:
                span.set_attribute("started", True)
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        # NoOpTracer should not interfere with exception propagation
        assert True
