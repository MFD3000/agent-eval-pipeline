"""
OpenInference Auto-Instrumentation

Registers auto-instrumentors for OpenAI, LangChain, and DSPy.
All LLM calls are automatically traced without code changes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_instrumented = False


def register_instrumentors() -> bool:
    """
    Register OpenInference auto-instrumentors.

    This should be called once at startup, before any LLM calls.
    Instruments: OpenAI, LangChain, DSPy

    Returns:
        True if any instrumentors were registered, False otherwise
    """
    global _instrumented
    if _instrumented:
        return True

    registered = []

    # OpenAI
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
        registered.append("openai")
    except ImportError:
        logger.debug("OpenAI instrumentor not available")
    except Exception as e:
        logger.warning(f"Failed to instrument OpenAI: {e}")

    # LangChain
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
        registered.append("langchain")
    except ImportError:
        logger.debug("LangChain instrumentor not available")
    except Exception as e:
        logger.warning(f"Failed to instrument LangChain: {e}")

    # DSPy
    try:
        from openinference.instrumentation.dspy import DSPyInstrumentor
        DSPyInstrumentor().instrument()
        registered.append("dspy")
    except ImportError:
        logger.debug("DSPy instrumentor not available")
    except Exception as e:
        logger.warning(f"Failed to instrument DSPy: {e}")

    # LiteLLM (used by DSPy under the hood)
    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        LiteLLMInstrumentor().instrument()
        registered.append("litellm")
    except ImportError:
        logger.debug("LiteLLM instrumentor not available")
    except Exception as e:
        logger.warning(f"Failed to instrument LiteLLM: {e}")

    if registered:
        logger.info(f"Registered instrumentors: {', '.join(registered)}")
        _instrumented = True
        return True

    return False


def uninstrument() -> None:
    """Remove all instrumentors (useful for testing)."""
    global _instrumented

    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from openinference.instrumentation.dspy import DSPyInstrumentor
        DSPyInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        LiteLLMInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    _instrumented = False
