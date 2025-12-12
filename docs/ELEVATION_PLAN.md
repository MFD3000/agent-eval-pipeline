# Code Elevation Plan: Agent Eval Pipeline

## Executive Summary

This plan transforms the agent-eval-pipeline from a working prototype into **production-grade, interview-ready architecture** by applying the patterns already proven in `embeddings/openai_embeddings.py` across the entire codebase.

**Philosophy:** The embeddings module IS our reference implementation. We extend its patterns, not invent new ones.

---

## The Gold Standard Pattern

`embeddings/openai_embeddings.py` (117 lines) demonstrates perfect elevation:

```
┌─────────────────────────────────────────────────────────────┐
│                    GOLD STANDARD PATTERN                     │
├─────────────────────────────────────────────────────────────┤
│  1. Protocol (interface)      → EmbeddingProvider           │
│  2. Production implementation → OpenAIEmbeddings            │
│  3. Test double               → MockEmbeddings              │
│  4. Factory function          → get_embedding_provider()    │
└─────────────────────────────────────────────────────────────┘
```

**Interview Quote:** "Every infrastructure component follows the same pattern: a Protocol defines the contract, concrete classes implement it, mock versions enable fast tests, and factory functions handle instantiation. This is hexagonal architecture applied consistently."

---

## Target Architecture

```
src/agent_eval_pipeline/
├── core/                          # NEW: Shared protocols & types
│   ├── __init__.py
│   ├── protocols.py               # All Protocol definitions
│   └── types.py                   # Shared dataclasses
│
├── retrieval/                     # ELEVATED
│   ├── __init__.py                # Re-exports
│   ├── document.py                # Document model
│   ├── store.py                   # VectorStore protocol + implementations
│   └── seeds/                     # NEW: Externalized data
│       ├── __init__.py
│       └── medical_knowledge.py   # Seed documents
│
├── agent/                         # ELEVATED
│   ├── __init__.py                # Factory + unified interface
│   ├── state.py                   # AgentState TypedDict
│   ├── nodes/                     # NEW: Extracted nodes
│   │   ├── __init__.py
│   │   ├── retrieve.py            # Pure retrieval node
│   │   ├── analyze.py             # Pure analysis node
│   │   └── safety.py              # Pure safety node
│   ├── graph.py                   # Graph construction (DI-enabled)
│   ├── langgraph_runner.py        # LangGraph entry point
│   └── legacy_agent.py            # Original OpenAI agent
│
├── evals/                         # ELEVATED
│   ├── __init__.py
│   ├── core/                      # NEW: Shared eval infrastructure
│   │   ├── __init__.py
│   │   ├── protocols.py           # Evaluator protocol
│   │   └── runner.py              # Generic gate runner
│   ├── schema/                    # ELEVATED
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── retrieval/                 # ELEVATED
│   │   ├── __init__.py
│   │   ├── corpus.py              # Document corpus (externalized)
│   │   ├── metrics.py             # Metric calculations (pure)
│   │   └── evaluator.py
│   ├── judge/                     # ELEVATED
│   │   ├── __init__.py
│   │   ├── prompts.py             # Externalized prompts
│   │   ├── schemas.py             # Judge output schemas
│   │   └── evaluator.py
│   └── perf/                      # ELEVATED
│       ├── __init__.py
│       ├── baseline.py            # BaselineStore protocol + impls
│       ├── pricing.py             # Cost estimation (pure)
│       ├── metrics.py             # Performance metrics (pure)
│       └── evaluator.py
│
├── harness/                       # ELEVATED
│   ├── __init__.py
│   └── runner.py                  # Orchestration (mostly unchanged)
│
├── embeddings/                    # ALREADY ELEVATED (reference)
│   ├── __init__.py
│   └── openai_embeddings.py
│
├── schemas/                       # ALREADY GOOD
│   ├── __init__.py
│   └── lab_insights.py
│
├── golden_sets/                   # MINOR REFINEMENT
│   ├── __init__.py
│   └── thyroid_cases.py
│
├── cli/                           # NEW: CLI extracted from modules
│   ├── __init__.py
│   ├── run_evals.py               # Main harness CLI
│   ├── run_judge.py               # Judge-only CLI
│   └── run_perf.py                # Perf-only CLI
│
└── config/                        # NEW: Externalized configuration
    ├── __init__.py
    ├── models.py                  # Model pricing, defaults
    └── settings.py                # Pydantic settings
```

---

## Phase 1: Foundation Layer (Core Protocols)

**Goal:** Establish shared contracts that all modules implement.

### 1.1 Create `core/protocols.py`

```python
"""
Core protocols defining contracts for the entire system.

All infrastructure components implement these protocols,
enabling dependency injection and easy testing.
"""

from typing import Protocol, runtime_checkable
import numpy as np

from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase


# Already exists in embeddings - re-export for consistency
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract for embedding generation."""
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Contract for vector similarity search."""
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def search(self, query: str, limit: int = 5) -> list["Document"]: ...
    def search_by_markers(self, markers: list[str], limit: int = 5) -> list["Document"]: ...


@runtime_checkable
class BaselineStore(Protocol):
    """Contract for performance baseline persistence."""
    def load(self) -> "PerformanceBaseline | None": ...
    def save(self, baseline: "PerformanceBaseline") -> None: ...


@runtime_checkable
class Evaluator(Protocol):
    """Contract for evaluation gates."""
    def evaluate(self, cases: list[GoldenCase]) -> "EvalReport": ...


@runtime_checkable
class AgentRunner(Protocol):
    """Contract for running agents."""
    def run(self, case: GoldenCase) -> "AgentResult | AgentError": ...
```

**Why:** Single source of truth for all interfaces. Import from one place.

---

## Phase 2: Retrieval Layer Elevation

**Goal:** Transform `pgvector_store.py` (548 lines) into focused modules.

### 2.1 Create `retrieval/document.py` (~30 lines)

```python
"""Document model for retrieval."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Document:
    """A document with embedding for retrieval."""
    id: str
    title: str
    content: str
    markers: list[str]
    embedding: np.ndarray | None = None
    score: float | None = None
```

### 2.2 Create `retrieval/store.py` (~200 lines)

```python
"""
Vector store implementations following the gold standard pattern.

Pattern: Protocol → Production impl → Test double → Factory
"""

from typing import Protocol
from dataclasses import dataclass
import numpy as np

from agent_eval_pipeline.core.protocols import VectorStore, EmbeddingProvider
from agent_eval_pipeline.retrieval.document import Document


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""
    connection_string: str = "postgresql://localhost/eval_pipeline"
    embedding_dim: int = 1536
    table_name: str = "medical_documents"


class PgVectorStore:
    """
    PostgreSQL vector store using pgvector.

    Dependencies are INJECTED, not created internally.
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        embeddings: EmbeddingProvider,  # INJECTED
    ):
        self.config = config
        self._embeddings = embeddings  # No global state!
        self._conn = None

    # ... implementation (cleaned, ~120 lines)


class InMemoryVectorStore:
    """
    In-memory vector store for testing.

    Implements same interface, no database required.
    """

    def __init__(self, embeddings: EmbeddingProvider):
        self._embeddings = embeddings
        self._documents: dict[str, Document] = {}

    # ... implementation (~60 lines)


def get_vector_store(
    use_postgres: bool = False,
    embeddings: EmbeddingProvider | None = None,
) -> VectorStore:
    """Factory function following gold standard pattern."""
    if embeddings is None:
        from agent_eval_pipeline.embeddings import get_embedding_provider
        embeddings = get_embedding_provider(use_mock=not use_postgres)

    if use_postgres:
        return PgVectorStore(VectorStoreConfig(), embeddings)
    return InMemoryVectorStore(embeddings)
```

### 2.3 Create `retrieval/seeds/medical_knowledge.py` (~150 lines)

Move seed data out of the store module. Data is data, not infrastructure.

**Interview Quote:** "Seed data is externalized so content teams can update medical knowledge without touching infrastructure code. The store module has one job: persist and query vectors."

---

## Phase 3: Agent Layer Elevation

**Goal:** Transform `langgraph_agent.py` (461 lines) into testable, injectable nodes.

### 3.1 Create `agent/state.py` (~40 lines)

```python
"""
Agent state definition - the data flowing through the graph.

Separated because state schema changes for different reasons
than node logic or graph structure.
"""

from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary


class AgentState(TypedDict):
    """State flowing through the LangGraph."""
    # Input
    query: str
    labs: list[dict]
    history: list[dict]
    symptoms: list[str]

    # LLM context
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Intermediate
    retrieved_docs: list[dict]
    raw_analysis: dict | None

    # Output
    final_output: LabInsightsSummary | None

    # Metrics
    retrieval_latency_ms: float
    analysis_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
```

### 3.2 Create `agent/nodes/retrieve.py` (~50 lines)

```python
"""
Retrieval node - pure function, dependencies injected.

This node is TESTABLE IN ISOLATION because:
1. VectorStore is injected, not global
2. No side effects beyond state updates
3. Deterministic given same inputs
"""

import time
from agent_eval_pipeline.core.protocols import VectorStore
from agent_eval_pipeline.agent.state import AgentState


def create_retrieve_node(store: VectorStore):
    """
    Factory that creates a retrieval node with injected store.

    Why a factory? So tests can inject MockVectorStore.
    """

    def retrieve_context(state: AgentState) -> dict:
        """Retrieve relevant medical documents."""
        start = time.time()

        markers = [lab["marker"] for lab in state["labs"]]
        docs = store.search_by_markers(markers, limit=5)

        retrieved = [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "markers": doc.markers,
                "score": doc.score,
            }
            for doc in docs
        ]

        return {
            "retrieved_docs": retrieved,
            "retrieval_latency_ms": (time.time() - start) * 1000,
        }

    return retrieve_context
```

### 3.3 Create `agent/nodes/analyze.py` (~80 lines)

```python
"""
Analysis node - LLM interaction isolated.

Dependencies:
- LLM client (injectable for testing)
- Prompt template (externalizable)
"""

import time
from langchain_openai import ChatOpenAI
from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
from agent_eval_pipeline.agent.state import AgentState


def create_analysis_prompt(state: AgentState) -> str:
    """Build the analysis prompt from state. PURE FUNCTION."""
    # ... prompt construction (testable separately)


def create_analyze_node(model: ChatOpenAI | None = None):
    """Factory that creates analysis node with injectable model."""

    def analyze_labs(state: AgentState) -> dict:
        start = time.time()

        llm = model or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        structured_llm = llm.with_structured_output(LabInsightsSummary)

        prompt = create_analysis_prompt(state)
        response = structured_llm.invoke(prompt)

        return {
            "raw_analysis": response.model_dump() if response else None,
            "analysis_latency_ms": (time.time() - start) * 1000,
            # ... tokens
        }

    return analyze_labs
```

### 3.4 Create `agent/graph.py` (~60 lines)

```python
"""
Graph construction with dependency injection.

The graph is just WIRING - all logic lives in nodes.
"""

from langgraph.graph import StateGraph, END
from agent_eval_pipeline.core.protocols import VectorStore
from agent_eval_pipeline.agent.state import AgentState
from agent_eval_pipeline.agent.nodes.retrieve import create_retrieve_node
from agent_eval_pipeline.agent.nodes.analyze import create_analyze_node
from agent_eval_pipeline.agent.nodes.safety import apply_safety


def build_agent_graph(
    store: VectorStore,
    model: ChatOpenAI | None = None,
) -> StateGraph:
    """
    Build the agent graph with injected dependencies.

    Why inject? So tests can use MockVectorStore and MockLLM.
    """
    workflow = StateGraph(AgentState)

    # Nodes created with dependencies
    workflow.add_node("retrieve_context", create_retrieve_node(store))
    workflow.add_node("analyze_labs", create_analyze_node(model))
    workflow.add_node("apply_safety", apply_safety)  # Pure, no deps

    # Edges
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "analyze_labs")
    workflow.add_edge("analyze_labs", "apply_safety")
    workflow.add_edge("apply_safety", END)

    return workflow.compile()
```

**Interview Quote:** "The graph is just wiring. Each node is a pure function created by a factory that injects its dependencies. This means I can test the retrieval node with a mock store in under 1ms, without touching the database or LLM."

---

## Phase 4: Eval Layer Elevation

**Goal:** Extract prompts, externalize config, enable isolated testing.

### 4.1 Create `evals/judge/prompts.py` (~50 lines)

```python
"""
Judge prompts externalized for versioning and A/B testing.

Could also be loaded from files for easier non-code updates.
"""

JUDGE_SYSTEM_PROMPT_V1 = """You are an expert evaluator...
[current prompt content]
"""

# Version tracking enables prompt regression testing
CURRENT_JUDGE_PROMPT = JUDGE_SYSTEM_PROMPT_V1

def get_judge_prompt(version: str = "v1") -> str:
    """Get judge prompt by version."""
    prompts = {"v1": JUDGE_SYSTEM_PROMPT_V1}
    return prompts.get(version, CURRENT_JUDGE_PROMPT)
```

### 4.2 Create `evals/perf/baseline.py` (~60 lines)

```python
"""
Baseline storage following gold standard pattern.

Protocol → FileBaselineStore → InMemoryBaselineStore → Factory
"""

from typing import Protocol
from pathlib import Path
import json
from dataclasses import dataclass


@dataclass
class PerformanceBaseline:
    """Stored performance baseline."""
    p50_latency_ms: float
    p95_latency_ms: float
    avg_total_tokens: float
    expected_model: str


class BaselineStore(Protocol):
    """Contract for baseline persistence."""
    def load(self) -> PerformanceBaseline | None: ...
    def save(self, baseline: PerformanceBaseline) -> None: ...


class FileBaselineStore:
    """File-based baseline storage."""

    def __init__(self, path: Path):
        self._path = path

    def load(self) -> PerformanceBaseline | None:
        if not self._path.exists():
            return None
        data = json.loads(self._path.read_text())
        return PerformanceBaseline(**data)

    def save(self, baseline: PerformanceBaseline) -> None:
        self._path.write_text(json.dumps(asdict(baseline), indent=2))


class InMemoryBaselineStore:
    """In-memory baseline for testing."""

    def __init__(self, initial: PerformanceBaseline | None = None):
        self._baseline = initial

    def load(self) -> PerformanceBaseline | None:
        return self._baseline

    def save(self, baseline: PerformanceBaseline) -> None:
        self._baseline = baseline


def get_baseline_store(path: Path | None = None) -> BaselineStore:
    """Factory function."""
    if path:
        return FileBaselineStore(path)
    return FileBaselineStore(Path(".perf_baseline.json"))
```

---

## Phase 5: CLI Extraction

**Goal:** Separate CLI concerns from core logic.

### 5.1 Create `cli/run_evals.py`

Move CLI logic from individual eval files to dedicated CLI module:

```python
"""CLI for running evaluation gates."""

import argparse
import sys
from dotenv import load_dotenv

from agent_eval_pipeline.harness.runner import run_all_evals, print_report


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run evaluation gates")
    # ... argument parsing

    report = run_all_evals(
        fail_fast=not args.no_fail_fast,
        skip_expensive=args.skip_expensive,
        verbose=not args.quiet,
    )

    print_report(report)
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
```

---

## Testing Strategy Post-Elevation

### Unit Tests (<1ms each, no I/O)

```python
# tests/unit/test_nodes.py

def test_retrieve_node_extracts_markers():
    """Test retrieval node in isolation."""
    mock_store = InMemoryVectorStore(MockEmbeddings())
    mock_store.insert_document(Document(
        id="test", title="Test", content="TSH info", markers=["TSH"]
    ))

    retrieve = create_retrieve_node(mock_store)

    state = {"labs": [{"marker": "TSH", "value": 5.0}]}
    result = retrieve(state)

    assert len(result["retrieved_docs"]) == 1
    assert result["retrieval_latency_ms"] < 10  # Fast!


def test_safety_node_adds_disclaimer():
    """Safety node is pure - no mocking needed."""
    state = {
        "raw_analysis": {
            "summary": "Test",
            "key_insights": [],
            "safety_notes": [],
            # ...
        }
    }

    result = apply_safety(state)

    assert any(
        n.type == "non_diagnostic"
        for n in result["final_output"].safety_notes
    )
```

### Integration Tests (~100ms, real wiring)

```python
# tests/integration/test_graph.py

def test_full_graph_with_mock_llm():
    """Test graph wiring with mock components."""
    store = InMemoryVectorStore(MockEmbeddings())
    seed_vector_store(store)

    mock_llm = MockChatOpenAI(responses=[...])

    graph = build_agent_graph(store=store, model=mock_llm)
    # ... test full flow
```

---

## File Changes Summary

| Action | File | Lines Before | Lines After |
|--------|------|--------------|-------------|
| CREATE | `core/protocols.py` | 0 | ~50 |
| CREATE | `core/types.py` | 0 | ~30 |
| SPLIT | `retrieval/pgvector_store.py` | 548 | - |
| CREATE | `retrieval/document.py` | 0 | ~30 |
| CREATE | `retrieval/store.py` | 0 | ~200 |
| CREATE | `retrieval/seeds/medical_knowledge.py` | 0 | ~150 |
| SPLIT | `agent/langgraph_agent.py` | 461 | - |
| CREATE | `agent/state.py` | 0 | ~40 |
| CREATE | `agent/nodes/retrieve.py` | 0 | ~50 |
| CREATE | `agent/nodes/analyze.py` | 0 | ~80 |
| CREATE | `agent/nodes/safety.py` | 0 | ~40 |
| CREATE | `agent/graph.py` | 0 | ~60 |
| CREATE | `agent/langgraph_runner.py` | 0 | ~80 |
| SPLIT | `evals/judge_eval.py` | 455 | - |
| CREATE | `evals/judge/prompts.py` | 0 | ~50 |
| CREATE | `evals/judge/schemas.py` | 0 | ~40 |
| CREATE | `evals/judge/evaluator.py` | 0 | ~150 |
| SPLIT | `evals/perf_eval.py` | 428 | - |
| CREATE | `evals/perf/baseline.py` | 0 | ~60 |
| CREATE | `evals/perf/pricing.py` | 0 | ~30 |
| CREATE | `evals/perf/metrics.py` | 0 | ~50 |
| CREATE | `evals/perf/evaluator.py` | 0 | ~150 |
| REFACTOR | `harness/runner.py` | 346 | ~200 |

**Total:** ~1,800 lines of focused, testable code replacing ~1,800 lines of mixed concerns.

---

## Interview Impact

### Before Elevation

"We have a LangGraph agent with pgvector retrieval and eval gates."

### After Elevation

"Every infrastructure component follows hexagonal architecture:

1. **Protocols define contracts** - `VectorStore`, `BaselineStore`, `Evaluator`
2. **Production implementations** - `PgVectorStore`, `FileBaselineStore`
3. **Test doubles** - `InMemoryVectorStore`, `InMemoryBaselineStore`
4. **Factory functions** - `get_vector_store()`, `get_baseline_store()`

Each LangGraph node is created by a factory that injects dependencies. I can test the retrieval node with a mock store in under 1ms. The graph is just wiring - all logic is in pure, testable node functions.

The eval system has externalized prompts for A/B testing and versioning. Baseline storage follows the same Protocol pattern, so I can test regression detection without touching the filesystem."

---

## Execution Order

1. **Phase 1: Core** - Create protocols (enables all other phases)
2. **Phase 2: Retrieval** - Highest coupling score, most impact
3. **Phase 3: Agent** - Eliminates global state, enables node testing
4. **Phase 4: Evals** - Externalize prompts, inject baselines
5. **Phase 5: CLI** - Clean separation, last priority

**Estimated Effort:** Each phase is independent and can be done incrementally.

---

## Validation Checklist

After elevation, verify:

- [ ] All existing tests pass
- [ ] No global mutable state
- [ ] Each class <100 lines
- [ ] Unit tests run in <100ms total
- [ ] Every Protocol has at least 2 implementations
- [ ] Factory functions for all infrastructure
- [ ] CLI separated from core logic
- [ ] Interview talking points updated

---

## Ready for Approval

This plan transforms the codebase by consistently applying the pattern already proven in `embeddings/openai_embeddings.py`. No new patterns to learn - just systematic application of what works.
