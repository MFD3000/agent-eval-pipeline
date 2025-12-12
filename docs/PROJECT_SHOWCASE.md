# Project Showcase: Technologies & Architectural Decisions

A quick-reference guide to the engineering decisions demonstrated in this codebase - designed for technical interviews and portfolio review.

---

## Technologies Demonstrated

### LLM Frameworks

- **LangGraph** - State machine-based agent with explicit node graph for retrieval → analysis → safety flow
- **DSPy** - Declarative prompt programming with ReAct agent, tool use, and optimizable signatures
- **OpenAI Structured Outputs** - Pydantic models as output contracts, guaranteeing valid JSON schema

### Evaluation Frameworks

- **DeepEval** - Pytest-native LLM testing with custom G-Eval metrics for healthcare domain
- **RAGAS** - RAG-specialized evaluation (faithfulness, context precision/recall, answer relevancy)
- **LLM-as-Judge** - Custom rubric-based evaluation with weighted dimensions and critical issue detection
- **DSPy Judge** - Optimizable judge using DSPy signatures for potential prompt tuning

### Observability

- **Phoenix/Arize** - LLM-native tracing with auto-instrumentation for OpenAI, LangChain, DSPy
- **OpenTelemetry** - Span-based tracing with custom attributes for eval gates, agent runs, token usage
- **Graceful Degradation** - NoOpTracer pattern when observability is disabled (zero overhead)

### Infrastructure

- **Pydantic v2** - Schema validation, serialization, and structured output contracts
- **pytest** - Test framework with fixtures, parametrization, and custom markers
- **Python Protocols** - Structural subtyping for dependency injection without ABC complexity

---

## Architectural Decisions

### 1. Protocol-First Design

```
Decision: Use Python Protocols instead of ABCs for all injectable dependencies
Why: Enables duck typing with type safety, simpler test doubles, no inheritance coupling
Where: core/protocols.py defines EmbeddingProvider, VectorStore, BaselineStore, etc.
```

### 2. Evaluation Gate Pipeline

```
Decision: Order eval gates by cost - fast/cheap checks first, expensive LLM calls last
Why: Fail fast on structure before spending $$ on semantic evaluation
Order: Schema (0ms) → Retrieval (0ms) → Judge (2-5s) → Perf (0ms)
```

### 3. Context Sharing (AgentRunContext)

```
Decision: Run agents ONCE, share results across all evaluators
Why: 4-5x reduction in LLM API calls, consistent evaluation (same output scored by all gates)
Bonus: Enables real retrieval validation against actual retrieved_docs
```

### 4. Dependency Injection via Factory Functions

```
Decision: Inject dependencies through factory functions, not global state
Why: Testability without mocking frameworks, runtime flexibility, explicit dependencies
Example: get_vector_store(use_postgres=False, embeddings=mock) for testing
```

### 5. Elevated Module Architecture

```
Decision: Split complex modules into focused sub-packages with backward-compat re-exports
Why: Separation of concerns (prompts, schemas, evaluator logic), maintains old import paths
Example: evals/judge_eval.py → evals/judge/{schemas,prompts,evaluator}.py
```

### 6. Healthcare Safety Layers

```
Decision: Multiple redundant safety checks at schema, agent, and eval levels
Why: Healthcare AI requires defense in depth - no single point of failure for safety
Layers: Schema requires SafetyNote[] → Agent injects disclaimers → Judge scores safety compliance
```

### 7. Golden Case Testing

```
Decision: Define expected outputs per test case, not just inputs
Why: Enables deterministic validation of LLM outputs against known-good expectations
Fields: expected_marker_statuses, expected_doc_ids, must_not_diagnose, etc.
```

### 8. Test Double Strategy

```
Decision: Every protocol has an in-memory/mock implementation for testing
Why: Fast tests without I/O, no API keys needed for most test runs
Examples: InMemoryVectorStore, InMemoryBaselineStore, MockEmbeddings
```

### 9. Graceful Degradation Pattern

```
Decision: Features degrade gracefully when optional dependencies unavailable
Why: Core functionality works without Phoenix, Postgres, etc.
Implementation: NoOpTracer, fallback to InMemoryVectorStore, optional imports
```

### 10. Unified Agent Interface

```
Decision: Single run_agent() entry point dispatches to implementation by type
Why: Evaluators don't care which agent implementation - same interface, same output schema
Supports: langgraph, dspy_react (extensible to new implementations)
```

### 11. Baseline-Driven Performance Regression

```
Decision: Compare current metrics against persisted baseline, flag regressions
Why: Catch latency/token increases before they hit production
Thresholds: 15% latency regression, 20% token regression, model change detection
```

### 12. Multi-Framework Evaluation

```
Decision: Run multiple eval frameworks (DeepEval, RAGAS, custom) on same outputs
Why: Each framework catches different issues - cross-validate for confidence
Comparison: Unified report shows same metrics across frameworks
```

---

## Code Patterns Worth Discussing

### Pattern: Result Type Union
```python
def run_agent(case) -> AgentResult | AgentError:
    # Explicit success/error types instead of exceptions for expected failures
```

### Pattern: Dataclass + Properties for Computed Fields
```python
@dataclass
class EvalReport:
    passed_cases: int
    total_cases: int

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases else 0.0
```

### Pattern: Context Manager for Spans
```python
with tracer.start_span("eval_gate", attributes={...}) as span:
    result = run_evaluation()
    span.set_attribute("passed", result.passed)
```

### Pattern: Lazy Imports for Optional Dependencies
```python
def run_deepeval(...):
    from deepeval import evaluate  # Only imported when function called
```

### Pattern: Seeded Randomness for Reproducibility
```python
def run_retrieval_eval(seed: int = 42):
    rng = random.Random(seed)  # Deterministic for CI, overridable for exploration
```

---

## Interview Talking Points

1. **"Walk me through the architecture"**
   → Start with eval gate pipeline diagram, explain cost-ordered execution, context sharing

2. **"How do you test LLM-based code?"**
   → Protocol-based mocks, golden cases with expected outputs, LLM-as-judge for semantics

3. **"Why multiple evaluation frameworks?"**
   → Each catches different issues - DeepEval for CI, RAGAS for RAG quality, custom for domain rules

4. **"How do you handle safety in healthcare AI?"**
   → Defense in depth: schema requires safety notes, agent injects disclaimers, judge scores compliance

5. **"What would you do differently?"**
   → Could extract base EvalReport class, consolidate framework adapters, add async agent execution

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Source Files | 60 |
| Test Files | 16 |
| Source LOC | ~10,800 |
| Test LOC | ~4,600 |
| Test Coverage | 203 passing tests |
| Agent Implementations | 2 (LangGraph, DSPy ReAct) |
| Eval Frameworks | 4 (Schema, Custom Judge, DeepEval, RAGAS) |
| Protocols Defined | 5 |

---

## File Quick Reference

| What | Where |
|------|-------|
| Agent entry point | `agent/__init__.py` → `run_agent()` |
| Eval harness | `harness/runner.py` → `run_all_evals()` |
| Context sharing | `harness/context.py` → `AgentRunContext` |
| Golden cases | `golden_sets/thyroid_cases.py` |
| Output schema | `schemas/lab_insights.py` → `LabInsightsSummary` |
| LLM-as-Judge | `evals/judge/evaluator.py` |
| Performance baseline | `evals/perf/baseline.py` |
| Protocols | `core/protocols.py` |
| Observability | `observability/{config,tracer,attributes}.py` |
