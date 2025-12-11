"""
Core module - shared protocols and types for the entire system.

This module provides the foundational contracts that enable:
- Dependency injection throughout the codebase
- Easy testing with mock implementations
- Clear separation of concerns

USAGE:
------
from agent_eval_pipeline.core import VectorStore, EmbeddingProvider

class MyVectorStore:
    '''Implements VectorStore protocol.'''
    ...
"""

from agent_eval_pipeline.core.protocols import (
    # Protocols
    EmbeddingProvider,
    VectorStore,
    BaselineStore,
    AgentRunner,
    Evaluator,
    # Data classes
    DocumentResult,
    PerformanceBaseline,
    AgentResult,
    AgentError,
    EvalResult,
    EvalReport,
)

__all__ = [
    # Protocols
    "EmbeddingProvider",
    "VectorStore",
    "BaselineStore",
    "AgentRunner",
    "Evaluator",
    # Data classes
    "DocumentResult",
    "PerformanceBaseline",
    "AgentResult",
    "AgentError",
    "EvalResult",
    "EvalReport",
]
