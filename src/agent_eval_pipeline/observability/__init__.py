"""
Observability Module - Phoenix + OpenTelemetry Integration

Provides LLM-native observability for agent executions and eval runs using
Arize Phoenix with OpenInference auto-instrumentation.

USAGE:
------
# At application startup:
from agent_eval_pipeline.observability import init_phoenix

init_phoenix()  # Starts local Phoenix UI if PHOENIX_ENABLED=true

# In code that needs tracing:
from agent_eval_pipeline.observability import get_tracer

tracer = get_tracer()
with tracer.start_span("my_operation", attributes={"key": "value"}) as span:
    # ... do work ...
    span.set_attribute("result", "success")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_eval_pipeline.observability.config import (
    PhoenixConfig,
    get_config,
    reset_config,
)
from agent_eval_pipeline.observability.tracer import (
    TracerProtocol,
    SpanProtocol,
    NoOpTracer,
    NoOpSpan,
    get_tracer,
    reset_tracer,
)
from agent_eval_pipeline.observability.attributes import (
    # GenAI
    GEN_AI_SYSTEM,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    # Eval
    EVAL_HARNESS_RUN_ID,
    EVAL_GATE_NAME,
    EVAL_GATE_STATUS,
    EVAL_CASE_ID,
    EVAL_CASE_PASSED,
    EVAL_JUDGE_DIMENSION,
    EVAL_JUDGE_SCORE,
    # Agent
    AGENT_TYPE,
    AGENT_NODE_NAME,
    AGENT_RETRIEVED_DOC_COUNT,
    AGENT_TOOLS_USED,
    # Helpers
    eval_gate_attributes,
    eval_case_attributes,
    agent_run_attributes,
)

logger = logging.getLogger(__name__)

_phoenix_initialized = False


def init_phoenix(config: PhoenixConfig | None = None) -> bool:
    """
    Initialize Phoenix observability.

    This should be called once at application startup.
    Sets up OpenTelemetry tracer provider and registers auto-instrumentors.

    Args:
        config: Optional config (uses env vars if not provided)

    Returns:
        True if Phoenix was initialized, False if disabled or failed
    """
    global _phoenix_initialized
    if _phoenix_initialized:
        return True

    config = config or get_config()

    if not config.enabled:
        logger.debug("Phoenix observability disabled")
        return False

    try:
        import phoenix as px
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Start Phoenix - local UI if no endpoint, otherwise connect to remote
        if config.collector_endpoint:
            # Remote Phoenix instance
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=config.collector_endpoint)
            logger.info(f"Phoenix connecting to remote: {config.collector_endpoint}")
        else:
            # Local Phoenix - launch app and get exporter
            session = px.launch_app()
            exporter = px.otel.SimpleSpanProcessor.exporter()
            logger.info(f"Phoenix UI available at: {session.url}")

        # Set up tracer provider
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Register auto-instrumentors
        from agent_eval_pipeline.observability.instrumentation import register_instrumentors
        register_instrumentors()

        _phoenix_initialized = True
        return True

    except ImportError as e:
        logger.warning(f"Phoenix not installed, observability disabled: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Phoenix: {e}")
        return False


def shutdown_phoenix() -> None:
    """Shutdown Phoenix and cleanup resources."""
    global _phoenix_initialized

    if not _phoenix_initialized:
        return

    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception as e:
        logger.warning(f"Error shutting down Phoenix: {e}")

    reset_tracer()
    reset_config()
    _phoenix_initialized = False


__all__ = [
    # Initialization
    "init_phoenix",
    "shutdown_phoenix",
    # Config
    "PhoenixConfig",
    "get_config",
    "reset_config",
    # Tracer
    "TracerProtocol",
    "SpanProtocol",
    "NoOpTracer",
    "NoOpSpan",
    "get_tracer",
    "reset_tracer",
    # Attributes - GenAI
    "GEN_AI_SYSTEM",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "GEN_AI_USAGE_TOTAL_TOKENS",
    # Attributes - Eval
    "EVAL_HARNESS_RUN_ID",
    "EVAL_GATE_NAME",
    "EVAL_GATE_STATUS",
    "EVAL_CASE_ID",
    "EVAL_CASE_PASSED",
    "EVAL_JUDGE_DIMENSION",
    "EVAL_JUDGE_SCORE",
    # Attributes - Agent
    "AGENT_TYPE",
    "AGENT_NODE_NAME",
    "AGENT_RETRIEVED_DOC_COUNT",
    "AGENT_TOOLS_USED",
    # Helpers
    "eval_gate_attributes",
    "eval_case_attributes",
    "agent_run_attributes",
]
