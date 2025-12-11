"""
Core protocols defining contracts for the entire system.

All infrastructure components implement these protocols,
enabling dependency injection and easy testing.

PATTERN: This follows the same structure as embeddings/openai_embeddings.py
- Protocol defines the contract
- Multiple implementations possible
- Factory functions for instantiation
- Test doubles for fast unit tests

INTERVIEW TALKING POINT:
------------------------
"Every infrastructure component follows the same pattern: a Protocol defines
the contract, concrete classes implement it, mock versions enable fast tests,
and factory functions handle instantiation. This is hexagonal architecture
applied consistently across the codebase."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
    from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase


# ---------------------------------------------------------------------------
# EMBEDDING PROVIDER PROTOCOL
# ---------------------------------------------------------------------------
# Re-exported from embeddings module for consistency

@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Contract for embedding generation.

    Implementations:
    - OpenAIEmbeddings (production)
    - MockEmbeddings (testing)
    """

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        ...


# ---------------------------------------------------------------------------
# VECTOR STORE PROTOCOL
# ---------------------------------------------------------------------------

@dataclass
class DocumentResult:
    """A retrieved document with similarity score."""
    id: str
    title: str
    content: str
    markers: list[str]
    score: float | None = None


@runtime_checkable
class VectorStore(Protocol):
    """
    Contract for vector similarity search.

    Implementations:
    - PgVectorStore (production with PostgreSQL)
    - InMemoryVectorStore (testing/development)
    """

    def connect(self) -> None:
        """Establish connection to the store."""
        ...

    def close(self) -> None:
        """Close connection to the store."""
        ...

    def search(
        self,
        query: str,
        limit: int = 5,
        marker_filter: list[str] | None = None,
    ) -> list[DocumentResult]:
        """Search for similar documents by query text."""
        ...

    def search_by_markers(
        self,
        markers: list[str],
        limit: int = 5,
    ) -> list[DocumentResult]:
        """Search for documents related to specific lab markers."""
        ...


# ---------------------------------------------------------------------------
# BASELINE STORE PROTOCOL
# ---------------------------------------------------------------------------

@dataclass
class PerformanceBaseline:
    """
    Stored performance baseline from previous evaluation runs.

    Used to detect regressions in latency and token usage.
    """
    p50_latency_ms: float
    p95_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    expected_model: str
    run_count: int = 1


@runtime_checkable
class BaselineStore(Protocol):
    """
    Contract for performance baseline persistence.

    Implementations:
    - FileBaselineStore (production)
    - InMemoryBaselineStore (testing)
    """

    def load(self) -> PerformanceBaseline | None:
        """Load baseline from storage. Returns None if not found."""
        ...

    def save(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to storage."""
        ...


# ---------------------------------------------------------------------------
# AGENT PROTOCOL
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """
    Successful result from running an agent.

    Contains both the output and performance metrics for evaluation.
    """
    output: Any  # LabInsightsSummary, but avoiding circular import
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    # LangGraph-specific (optional)
    retrieved_docs: list[dict] | None = None
    retrieval_latency_ms: float | None = None


@dataclass
class AgentError:
    """Error result when agent execution fails."""
    error_type: str
    error_message: str


@runtime_checkable
class AgentRunner(Protocol):
    """
    Contract for running agents on golden cases.

    Implementations:
    - LangGraphAgentRunner
    - LegacyAgentRunner
    """

    def run(self, case: Any) -> AgentResult | AgentError:  # GoldenCase
        """Run the agent on a golden case."""
        ...


# ---------------------------------------------------------------------------
# EVALUATOR PROTOCOL
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluating a single case."""
    case_id: str
    passed: bool
    score: float | None = None
    details: dict[str, Any] | None = None


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    total_cases: int
    passed_cases: int
    failed_cases: int
    results: list[EvalResult]

    @property
    def all_passed(self) -> bool:
        return self.failed_cases == 0

    @property
    def pass_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases


@runtime_checkable
class Evaluator(Protocol):
    """
    Contract for evaluation gates.

    Each eval gate (schema, retrieval, judge, perf) implements this protocol.
    """

    def evaluate(
        self,
        cases: list[Any],  # list[GoldenCase]
        verbose: bool = False,
    ) -> EvalReport:
        """Run evaluation on golden cases."""
        ...
