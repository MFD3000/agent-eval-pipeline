"""
Tracer Factory and NoOp Implementations

Provides get_tracer() factory that returns either a real OTel tracer
or a NoOpTracer for graceful degradation.
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import Any, Iterator, Protocol


# ---------------------------------------------------------------------------
# PROTOCOLS
# ---------------------------------------------------------------------------


class SpanProtocol(Protocol):
    """Protocol for span operations."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        ...

    def set_status(self, status: str, description: str | None = None) -> None:
        """Set span status (ok, error)."""
        ...

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        ...

    def end(self) -> None:
        """End the span."""
        ...


class TracerProtocol(Protocol):
    """Protocol for tracer operations."""

    @contextmanager
    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[SpanProtocol]:
        """Start a new span as context manager."""
        ...


# ---------------------------------------------------------------------------
# NOOP IMPLEMENTATIONS (for graceful degradation)
# ---------------------------------------------------------------------------


class NoOpSpan:
    """No-op span that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: str, description: str | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def end(self) -> None:
        pass


class NoOpTracer:
    """No-op tracer that creates no-op spans."""

    @contextmanager
    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[NoOpSpan]:
        yield NoOpSpan()


# ---------------------------------------------------------------------------
# REAL OTEL TRACER (wrapped for our protocol)
# ---------------------------------------------------------------------------


class OTelSpan:
    """Wrapper around OTel span to match our protocol."""

    def __init__(self, span: Any):
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def set_status(self, status: str, description: str | None = None) -> None:
        from opentelemetry.trace import StatusCode
        code = StatusCode.OK if status == "ok" else StatusCode.ERROR
        self._span.set_status(code, description)

    def record_exception(self, exception: Exception) -> None:
        self._span.record_exception(exception)

    def end(self) -> None:
        self._span.end()


class OTelTracer:
    """Wrapper around OTel tracer to match our protocol."""

    def __init__(self, tracer: Any):
        self._tracer = tracer

    @contextmanager
    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[OTelSpan]:
        with self._tracer.start_as_current_span(name, attributes=attributes) as span:
            yield OTelSpan(span)


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------


_tracer: TracerProtocol | None = None


def get_tracer(service_name: str = "agent-eval-pipeline") -> TracerProtocol:
    """
    Get the global tracer instance.

    Returns OTelTracer if Phoenix is enabled and OTel is available,
    otherwise returns NoOpTracer for zero overhead.

    Args:
        service_name: Service name for the tracer (used on first call only)

    Returns:
        TracerProtocol implementation
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    from agent_eval_pipeline.observability.config import get_config

    config = get_config()

    if not config.enabled:
        _tracer = NoOpTracer()
        return _tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        # Get or create tracer provider
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            # Provider not set up yet - return NoOp
            # (init_phoenix should have been called first)
            _tracer = NoOpTracer()
            return _tracer

        otel_tracer = trace.get_tracer(service_name)
        _tracer = OTelTracer(otel_tracer)
        return _tracer

    except ImportError:
        # OTel not installed - graceful degradation
        _tracer = NoOpTracer()
        return _tracer


def reset_tracer() -> None:
    """Reset tracer (useful for testing)."""
    global _tracer
    _tracer = None
