# Code Elevation Assessment: Agent Eval Pipeline

**Assessment Date:** 2025-01-29
**Assessor:** Claude Code (using code-elevation skill methodology)

---

## Executive Summary

The agent-eval-pipeline demonstrates solid architecture for an interview-prep project with good separation of concerns at the module level. However, several modules have grown into **god classes** with 5+ responsibilities, impacting testability and maintainability.

| Severity | Count | Files |
|----------|-------|-------|
| RED FLAG | 4 | pgvector_store.py, langgraph_agent.py, judge_eval.py, perf_eval.py |
| YELLOW FLAG | 2 | retrieval_eval.py, runner.py |
| GREEN FLAG | 4 | lab_insights_agent.py, agent/__init__.py, schemas/, tests/ |

**Recommended Priority:** Start with `pgvector_store.py` (highest impact, clearest extraction path)

---

## Module Analysis

### 1. pgvector_store.py (548 lines) - RED FLAG

**Responsibilities Identified: 6**

| # | Responsibility | Lines | Reason to Change |
|---|----------------|-------|------------------|
| 1 | VectorStoreConfig | 49-56 | Config schema changes |
| 2 | Document model | 62-71 | Data structure changes |
| 3 | Embedding coordination | 84-93 | Provider switching |
| 4 | PgVectorStore (DB ops) | 99-286 | Database changes |
| 5 | SimulatedVectorStore | 292-377 | Testing requirements |
| 6 | Seed data | 406-549 | Content updates |

**SOLID Violations:**

- **SRP**: 6 distinct responsibilities = 6 reasons to change
- **DIP**: Hard-coded `_get_embeddings()` global function, not injectable
- **OCP**: Adding new store types requires modifying factory function

**Coupling Score:** 6 responsibilities × 3 dependencies = **18** (high)

**Pain Points:**
- Testing requires either Postgres or uses SimulatedVectorStore (no interface)
- Seed data embedded in module (should be separate)
- Global mutable state: `_embedding_provider`

**Recommendation:** Extract into:
```
retrieval/
├── config.py           # VectorStoreConfig
├── document.py         # Document model
├── store_interface.py  # VectorStore protocol
├── pgvector_store.py   # PgVectorStore only (~100 lines)
├── memory_store.py     # SimulatedVectorStore
├── factory.py          # get_vector_store()
└── seed_data.py        # Medical documents
```

---

### 2. langgraph_agent.py (461 lines) - RED FLAG

**Responsibilities Identified: 5**

| # | Responsibility | Lines | Reason to Change |
|---|----------------|-------|------------------|
| 1 | AgentState definition | 55-84 | State schema changes |
| 2 | Vector store coordination | 91-113 | Retrieval changes |
| 3 | Node functions (retrieve/analyze/safety) | 115-276 | Business logic |
| 4 | Graph construction | 282-314 | Workflow changes |
| 5 | Public API + result types | 321-462 | Interface changes |

**SOLID Violations:**

- **SRP**: State, nodes, graph, and API all in one file
- **DIP**: Global mutable state (`_vector_store`, `_store_seeded`, `_agent_graph`)
- **OCP**: Adding nodes requires modifying graph construction

**Global State Issues:**
```python
# Lines 91-93, 307-308 - Mutable globals make testing difficult
_vector_store = None
_store_seeded = False
_agent_graph = None
```

**Pain Points:**
- Cannot test nodes in isolation without module-level side effects
- Global state causes test pollution
- `_get_vector_store()` seeds on first use (hidden side effect)

**Recommendation:** Extract into:
```
agent/
├── state.py           # AgentState TypedDict
├── nodes/
│   ├── retrieve.py    # retrieve_context node
│   ├── analyze.py     # analyze_labs node
│   └── safety.py      # apply_safety node
├── graph.py           # Graph construction (injectable deps)
├── runner.py          # Public API with DI
└── langgraph_agent.py # Thin orchestrator
```

---

### 3. judge_eval.py (455 lines) - RED FLAG

**Responsibilities Identified: 4**

| # | Responsibility | Lines |
|---|----------------|-------|
| 1 | Output schemas (DimensionScore, JudgeOutput) | 74-92 |
| 2 | Judge prompt | 99-137 |
| 3 | Evaluation logic | 203-400 |
| 4 | CLI interface | 403-455 |

**SOLID Violations:**

- **SRP**: Schema + prompt + logic + CLI mixed
- **OCP**: Changing scoring dimensions requires modifying multiple places

**Pain Points:**
- WEIGHTS dict at line 233 is disconnected from JudgeOutput fields
- Prompt text embedded in code (hard to version/A-B test)
- CLI mixed with core logic

**Recommendation:**
```
evals/
├── judge/
│   ├── schemas.py      # DimensionScore, JudgeOutput
│   ├── prompts.py      # JUDGE_SYSTEM_PROMPT (or load from file)
│   ├── evaluator.py    # run_judge(), calculate_weighted_score()
│   └── cli.py          # CLI entry point
└── judge_eval.py       # Thin re-export for backward compat
```

---

### 4. perf_eval.py (428 lines) - RED FLAG

**Responsibilities Identified: 4**

| # | Responsibility | Lines |
|---|----------------|-------|
| 1 | Baseline I/O | 64-104 |
| 2 | Metrics models | 111-157 |
| 3 | Cost estimation | 164-181 |
| 4 | Evaluation + CLI | 187-428 |

**SOLID Violations:**

- **SRP**: File I/O, metrics, pricing, evaluation all mixed
- **DIP**: Hard-coded `BASELINE_FILE` path (line 64)

```python
# Line 64 - Hard-coded path, not injectable
BASELINE_FILE = Path(__file__).parent.parent.parent.parent / ".perf_baseline.json"
```

**Pain Points:**
- Can't test baseline logic without real file system
- MODEL_PRICING will need updates (OpenAI changes prices)
- CLI mixed with evaluation logic

**Recommendation:**
```
evals/
├── perf/
│   ├── baseline.py     # BaselineStore protocol + FileBaselineStore
│   ├── metrics.py      # CasePerformance, PerformanceMetrics
│   ├── pricing.py      # MODEL_PRICING, estimate_cost()
│   └── evaluator.py    # run_perf_eval() with injected deps
└── perf_eval.py        # Thin facade
```

---

### 5. runner.py (346 lines) - YELLOW FLAG

**Assessment:** Mostly orchestration, which is acceptable. The `run_gate()` function is well-designed.

**Minor Issues:**
- CLI argument parsing mixed with core logic
- Could benefit from extracting `GateRunner` class

**Verdict:** OK for now, revisit if it grows beyond 400 lines.

---

### 6. lab_insights_agent.py (269 lines) - GREEN FLAG

**Assessment:** Well-structured single-responsibility module.

- `SYSTEM_PROMPT` is a constant (appropriate)
- `format_*` helpers are pure functions
- `run_agent()` is clean with clear I/O

**Verdict:** Good example of proper separation. Use as template.

---

### 7. agent/__init__.py (131 lines) - GREEN FLAG

**Assessment:** Excellent factory pattern implementation.

- `Agent` Protocol defines clean interface
- `run_agent()` factory with environment-based switching
- Result types properly unified

**Verdict:** Reference implementation for other modules.

---

## Test Analysis

**Current Test Coverage:**
- `test_evals.py`: 220 lines with good unit tests for schemas and metrics
- Integration test requires API key (acceptable)

**Testing Pain Points:**

| Module | Testability Issue |
|--------|-------------------|
| pgvector_store.py | Global `_embedding_provider` state |
| langgraph_agent.py | Three global variables, auto-seeding |
| perf_eval.py | Hard-coded file path for baseline |
| judge_eval.py | No mock for OpenAI client |

**Estimated Test Speed Improvement After Elevation:**
- Current: Unknown (likely >1s due to potential I/O)
- Target: <100ms for all unit tests (no I/O)

---

## Coupling Metrics

| Module | Dependencies | Responsibilities | Coupling Score | Risk |
|--------|-------------|------------------|----------------|------|
| pgvector_store.py | 3 | 6 | 18 | HIGH |
| langgraph_agent.py | 5 | 5 | 25 | HIGH |
| judge_eval.py | 4 | 4 | 16 | MEDIUM |
| perf_eval.py | 3 | 4 | 12 | MEDIUM |
| lab_insights_agent.py | 3 | 1 | 3 | LOW |

**Formula:** `dependencies × responsibilities = coupling score`

---

## Recommended Elevation Order

### Phase 1: pgvector_store.py (Highest ROI)
**Why First:**
- Most responsibilities (6)
- Affects both agents and tests
- Clearest extraction path
- Enables faster tests immediately

### Phase 2: langgraph_agent.py
**Why Second:**
- Core interview piece (needs to be explainable)
- Global state is an anti-pattern
- Node extraction enables unit testing

### Phase 3: judge_eval.py + perf_eval.py
**Why Together:**
- Similar patterns
- Both need prompt/config externalization
- Shared learnings

---

## Quick Wins (No Structural Changes)

1. **Extract seed data to JSON file**
   - `get_medical_documents()` → load from `data/medical_docs.json`
   - 140 lines removed from pgvector_store.py

2. **Move MODEL_PRICING to config**
   - Create `config/pricing.json`
   - Update via environment or file

3. **Create VectorStore Protocol**
   - Both stores already share methods
   - Just add the interface

```python
# retrieval/protocols.py
from typing import Protocol

class VectorStore(Protocol):
    def connect(self) -> None: ...
    def search(self, query: str, limit: int = 5) -> list[Document]: ...
    def search_by_markers(self, markers: list[str], limit: int = 5) -> list[Document]: ...
```

---

## Interview Readiness Impact

| Current State | After Elevation |
|--------------|-----------------|
| "We have a LangGraph agent" | "Each node is independently testable" |
| "We use pgvector for RAG" | "Repository pattern with swappable stores" |
| "We have eval gates" | "Clean separation: schema/prompt/logic/CLI" |
| "Global state for singletons" | "Dependency injection throughout" |

**Key Interview Talking Points After Elevation:**

1. "I refactored the retrieval layer using Repository pattern - the agent doesn't know if it's hitting Postgres or an in-memory mock"

2. "Each LangGraph node is a pure function that can be tested in isolation in <1ms"

3. "The eval system follows hexagonal architecture - prompts are externalized, scoring is configurable"

---

## Success Criteria

After elevation, the codebase should achieve:

- [ ] Each class <100 lines with single responsibility
- [ ] Unit tests run in <100ms (no I/O)
- [ ] No global mutable state
- [ ] All infrastructure injected via constructors
- [ ] Can explain each module's purpose in one sentence
- [ ] Test coverage >80% for business logic

---

## Next Steps

1. **Read the code-elevation skill examples:**
   - `.claude/skills/code-elevation/references/examples/direct-extractor/`

2. **Start with pgvector_store.py extraction:**
   - Create Protocol first
   - Extract Document and Config
   - Separate stores to their own files
   - Add dependency injection

3. **Validate with tests:**
   - Write characterization tests before refactoring
   - Add unit tests for extracted components
   - Ensure integration tests still pass
