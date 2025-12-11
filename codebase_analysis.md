# Agent Eval Pipeline - Comprehensive Codebase Analysis

## 1. Project Overview

### Project Type
**AI Agent Evaluation System** - A production-style evaluation gates system for AI agents that analyze healthcare lab results.

### Tech Stack
| Category | Technologies |
|----------|--------------|
| Language | Python 3.11+ |
| LLM Orchestration | LangGraph, DSPy |
| LLM Provider | OpenAI (GPT-4o, GPT-4o-mini) |
| Evaluation Frameworks | DeepEval, RAGAS, Custom LLM-as-Judge |
| Vector Store | PostgreSQL + pgvector (production), In-memory (testing) |
| Embeddings | OpenAI text-embedding-3-small |
| Schema Validation | Pydantic v2 |
| Testing | pytest, pytest-asyncio |
| Build System | Hatch |
| CI/CD | GitHub Actions |

### Architecture Pattern
**Hexagonal Architecture (Ports & Adapters)** with:
- Protocols as ports (interfaces)
- Multiple implementations (production + test doubles)
- Factory functions for dependency injection
- Clear separation of domain logic from infrastructure

### Project Statistics
- **Python Files**: 59
- **Project Size**: ~1.3MB
- **Lines of Code**: ~6,500+ (estimated)
- **Test Files**: 4
- **Golden Cases**: 5 thyroid panel scenarios

---

## 2. Directory Structure Analysis

```
agent-eval-pipeline/
├── .claude/                    # Claude Code skills and configuration
│   └── skills/
│       ├── code-elevation/     # Code quality elevation skill
│       └── healthcare-eval/    # Healthcare AI evaluation skill
├── .github/workflows/          # CI/CD pipeline
│   └── eval-gates.yml          # Evaluation gates workflow
├── src/agent_eval_pipeline/    # Main source code
│   ├── agent/                  # Agent implementations
│   │   ├── nodes/              # LangGraph node functions
│   │   ├── langgraph_agent.py  # LangGraph orchestration
│   │   ├── dspy_agent.py       # DSPy declarative agent
│   │   └── dspy_react_agent.py # DSPy ReAct tool-using agent
│   ├── cli/                    # Command-line interface
│   ├── core/                   # Protocols and core types
│   ├── embeddings/             # Text embedding providers
│   ├── evals/                  # Evaluation gates
│   │   ├── deepeval/           # DeepEval integration
│   │   ├── ragas/              # RAGAS integration
│   │   ├── judge/              # LLM-as-judge evaluation
│   │   └── perf/               # Performance regression
│   ├── golden_sets/            # Test cases (source of truth)
│   ├── harness/                # Evaluation orchestration
│   ├── retrieval/              # Vector store implementations
│   │   └── seeds/              # Medical knowledge base
│   └── schemas/                # Pydantic output schemas
└── tests/                      # Test suite
```

### Key Directories Explained

#### `agent/` - Agent Implementations
Contains multiple agent paradigms for the same use case:
- **LangGraph**: Imperative state machine with explicit nodes/edges
- **DSPy**: Declarative signatures with automatic prompt optimization
- **DSPy ReAct**: Tool-using agent with reasoning traces

#### `evals/` - Evaluation Gates
Multi-layered evaluation system:
- **schema_eval.py**: Pydantic validation (fast, deterministic)
- **retrieval_eval.py**: RAG quality metrics (no LLM calls)
- **judge/**: LLM-as-judge semantic evaluation
- **deepeval/**: DeepEval G-Eval metrics integration
- **ragas/**: RAGAS RAG-specialized metrics
- **perf/**: Latency and token usage regression detection

#### `harness/` - Evaluation Orchestration
- **runner.py**: Original 4-gate evaluation pipeline
- **unified_runner.py**: Multi-framework comparison runner

---

## 3. File-by-File Breakdown

### Core Application Files

| File | Purpose | Key Components |
|------|---------|----------------|
| `core/protocols.py` | Interface contracts | `EmbeddingProvider`, `VectorStore`, `BaselineStore`, `AgentRunner`, `Evaluator` |
| `schemas/lab_insights.py` | Output schema | `MarkerInsight`, `SafetyNote`, `LabInsightsSummary` |
| `agent/state.py` | LangGraph state | `AgentState` TypedDict |
| `agent/graph.py` | Graph construction | `build_agent_graph()`, dependency injection |
| `agent/langgraph_agent.py` | LangGraph facade | Re-exports from elevated modules |
| `agent/dspy_agent.py` | DSPy agent | `LabInsightsModule`, `AnalyzeLabs` signature |
| `agent/dspy_react_agent.py` | ReAct agent | Tool-using agent with medical tools |

### Evaluation Files

| File | Purpose | Key Components |
|------|---------|----------------|
| `evals/schema_eval.py` | Schema validation | Pydantic validation of agent output |
| `evals/retrieval_eval.py` | RAG quality | Precision, Recall, F1 metrics |
| `evals/judge/evaluator.py` | LLM-as-judge | Multi-dimensional scoring |
| `evals/judge/dspy_judge.py` | DSPy judge | Optimizable evaluation |
| `evals/deepeval/metrics.py` | G-Eval metrics | Clinical correctness, safety compliance |
| `evals/deepeval/adapters.py` | Format conversion | `GoldenCase` → `LLMTestCase` |
| `evals/ragas/metrics.py` | RAGAS config | Faithfulness, context metrics |
| `evals/ragas/adapters.py` | Format conversion | `GoldenCase` → `SingleTurnSample` |
| `evals/perf/evaluator.py` | Performance | Latency/token regression detection |

### Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, entry points |
| `.github/workflows/eval-gates.yml` | CI pipeline with eval gates |
| `.env.example` | Environment variable template |

### Test Files

| File | Test Count | Purpose |
|------|------------|---------|
| `tests/test_evals.py` | ~15 | Schema validation, golden sets, metrics |
| `tests/test_dspy.py` | ~20 | DSPy modules, signatures, tools |
| `tests/test_eval_frameworks.py` | ~22 | DeepEval/RAGAS integration |
| `tests/test_deepeval_integration.py` | ~5 | Pytest-native DeepEval tests |

---

## 4. Architecture Deep Dive

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            EVALUATION HARNESS                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Schema  │→ │ Retrieval│→ │   Judge  │→ │ DeepEval │→ │  RAGAS   │  │
│  │   Eval   │  │   Eval   │  │   Eval   │  │   Eval   │  │   Eval   │  │
│  │ (fast)   │  │ (no LLM) │  │ (custom) │  │ (G-Eval) │  │  (RAG)   │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             GOLDEN CASES                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ case_id, query, labs[], history[], expected_semantic_points[]   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT IMPLEMENTATIONS                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │    LangGraph    │  │      DSPy       │  │   DSPy ReAct    │         │
│  │  State Machine  │  │   Declarative   │  │   Tool-Using    │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┼────────────────────┘                   │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      RETRIEVAL LAYER                             │   │
│  │  ┌─────────────┐              ┌─────────────┐                   │   │
│  │  │ PgVectorStore│ (prod)     │InMemoryStore │ (test)           │   │
│  │  └─────────────┘              └─────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT SCHEMA                                   │
│  LabInsightsSummary { summary, key_insights[], doctor_topics[],         │
│                       lifestyle[], safety_notes[] }                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### LangGraph Agent Flow

```
┌─────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  START  │ ──► │ retrieve_context │ ──► │   analyze_labs   │ ──► │ apply_safety │ ──► END
└─────────┘     └──────────────────┘     └──────────────────┘     └──────────────┘
                      │                         │                       │
                      ▼                         ▼                       ▼
               VectorStore.search()      ChatOpenAI.invoke()      Add disclaimers
               (injected dependency)     (structured output)      (pure function)
```

### Dependency Injection Pattern

```python
# Protocol defines contract
class VectorStore(Protocol):
    def search(self, query: str, limit: int) -> list[DocumentResult]: ...

# Production implementation
class PgVectorStore:
    def __init__(self, config: Config, embeddings: EmbeddingProvider): ...

# Test implementation
class InMemoryVectorStore:
    def __init__(self, embeddings: EmbeddingProvider): ...

# Factory handles instantiation
def get_vector_store(use_postgres: bool) -> VectorStore: ...

# Node accepts injected dependency
def create_retrieve_node(store: VectorStore) -> Callable:
    def retrieve(state: AgentState) -> dict:
        results = store.search(...)
        return {"retrieved_docs": results}
    return retrieve
```

---

## 5. Evaluation Framework Comparison

| Aspect | Schema | Retrieval | Custom Judge | DeepEval | RAGAS |
|--------|--------|-----------|--------------|----------|-------|
| Speed | <10ms | <100ms | ~2-5s | ~2-5s | ~2-5s |
| LLM Calls | No | No | Yes | Yes | Yes |
| Focus | Structure | RAG quality | Domain rubrics | General LLM | RAG-specific |
| Key Metrics | Validation | P/R/F1 | 4 dimensions | G-Eval | Faithfulness |
| CI Gate | Hard | Hard | Soft | Soft | Soft |
| Customization | None | Low | High | Medium | Low |

### Evaluation Order Strategy

1. **Fast gates first** - Schema, retrieval (fail fast, save compute)
2. **LLM gates if fast pass** - Judge, DeepEval, RAGAS
3. **Compare across frameworks** - Build confidence
4. **Gate on critical metrics** - Safety must pass

---

## 6. Key Design Patterns

### 1. Protocol-First Design
Every infrastructure component starts with a Protocol:
```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...
```

### 2. Factory Functions
All instantiation goes through factories:
```python
def get_embedding_provider(use_mock: bool = True) -> EmbeddingProvider:
    return MockEmbeddings() if use_mock else OpenAIEmbeddings()
```

### 3. Test Doubles
Every production class has a test double:
- `OpenAIEmbeddings` → `MockEmbeddings`
- `PgVectorStore` → `InMemoryVectorStore`
- `FileBaselineStore` → `InMemoryBaselineStore`

### 4. DSPy Signatures for Structured Output
```python
class AnalyzeLabs(dspy.Signature):
    """Analyze lab results."""
    query: str = dspy.InputField()
    labs: str = dspy.InputField()
    summary: str = dspy.OutputField()
```

### 5. Lazy Loading for Optional Dependencies
```python
# Metrics only instantiate when called (avoids API key requirement at import)
def get_clinical_correctness() -> GEval:
    return _get_metric("clinical_correctness")
```

### 6. Golden Cases as Source of Truth
```python
@dataclass
class GoldenCase:
    id: str
    query: str
    labs: list[LabValue]
    expected_semantic_points: list[str]  # What MUST be true
    must_include_doctor_recommendation: bool = True
    must_not_diagnose: bool = True
```

---

## 7. Environment & Setup

### Required Environment Variables
```bash
OPENAI_API_KEY=sk-...          # Required for LLM calls
JUDGE_MODEL=gpt-4o             # Model for LLM-as-judge
AGENT_MODEL=gpt-4o-mini        # Model for agent
USE_POSTGRES=false             # Use PgVector vs InMemory
USE_MOCK_EMBEDDINGS=true       # Use mock embeddings for testing
DATABASE_URL=postgresql://...  # If USE_POSTGRES=true
```

### Installation
```bash
pip install -e ".[dev]"
cp .env.example .env
# Add OPENAI_API_KEY to .env
```

### Running Tests
```bash
PYTHONPATH=src pytest tests/ -v              # All tests
PYTHONPATH=src pytest tests/test_dspy.py -v  # DSPy tests (no API key)
```

### Running Evaluations
```bash
# Full pipeline
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner

# Skip expensive LLM calls
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --skip-expensive

# Unified multi-framework
PYTHONPATH=src python -m agent_eval_pipeline.harness.unified_runner -v
```

---

## 8. CI/CD Integration

### GitHub Actions Workflow
```yaml
name: Eval Gates
on:
  pull_request:
    paths: ['src/**', 'tests/**', 'pyproject.toml']

jobs:
  eval-gates:
    steps:
      - Install dependencies
      - Run eval harness → eval_report.json
      - Upload artifacts
      - Comment on PR with results
```

### PR Comment Format
```
## ✅ Eval Gates: All Gates Passed

| Gate | Status | Summary |
|------|--------|---------|
| Schema Validation | ✅ passed | Pass rate: 100.0% |
| Retrieval Quality | ✅ passed | Avg F1: 0.85 |
| LLM-as-Judge | ✅ passed | Avg score: 4.2/5 |
| Performance | ✅ passed | Within baseline |
```

---

## 9. Visual Architecture Diagram

```
                                 ┌─────────────────────────────────────────┐
                                 │           GitHub Actions CI              │
                                 │  ┌─────────────────────────────────────┐│
                                 │  │    eval-gates.yml                   ││
                                 │  │    - Run harness                    ││
                                 │  │    - Upload report                  ││
                                 │  │    - Comment on PR                  ││
                                 │  └─────────────────────────────────────┘│
                                 └──────────────────┬──────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                    HARNESS LAYER                                       │
│                                                                                       │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│   │    Schema    │───▶│  Retrieval   │───▶│    Judge     │───▶│    Perf      │       │
│   │     Eval     │    │     Eval     │    │     Eval     │    │    Eval      │       │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘       │
│          │                   │                   │                   │               │
│          │                   │                   │                   │               │
│   ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐          │
│   │  Pydantic   │    │  P/R/F1     │    │  4 Dims     │    │  Baseline   │          │
│   │  Validate   │    │  Metrics    │    │  Scoring    │    │  Compare    │          │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                                       │
│   ┌──────────────────────────────────────────────────────────────────────────────┐   │
│   │                       UNIFIED RUNNER (multi-framework)                        │   │
│   │   DeepEval (G-Eval)  │  RAGAS (Faithfulness)  │  DSPy Judge (Optimizable)    │   │
│   └──────────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                   GOLDEN CASES                                         │
│                                                                                       │
│   ┌──────────────────────────────────────────────────────────────────────────────┐   │
│   │  thyroid-001: High TSH with rising trend - clear hypothyroid signal          │   │
│   │  thyroid-002: Normal thyroid panel - should reassure user                    │   │
│   │  thyroid-003: Low TSH - potential hyperthyroid pattern                       │   │
│   │  thyroid-004: Borderline TSH - requires nuanced response                     │   │
│   │  thyroid-005: Vague query with mixed signals                                 │   │
│   └──────────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                   AGENT LAYER                                          │
│                                                                                       │
│   ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────┐  │
│   │       LangGraph         │  │         DSPy            │  │     DSPy ReAct      │  │
│   │    ┌──────────────┐    │  │    ┌──────────────┐    │  │  ┌──────────────┐   │  │
│   │    │   retrieve   │    │  │    │AnalyzeLabs   │    │  │  │   ReAct      │   │  │
│   │    └──────┬───────┘    │  │    │  Signature   │    │  │  │   Module     │   │  │
│   │           ▼            │  │    └──────────────┘    │  │  └──────────────┘   │  │
│   │    ┌──────────────┐    │  │    ┌──────────────┐    │  │  Tools:            │  │
│   │    │   analyze    │    │  │    │ChainOfThought│    │  │  - lookup_ref_range│  │
│   │    └──────┬───────┘    │  │    └──────────────┘    │  │  - check_med_int   │  │
│   │           ▼            │  │    ┌──────────────┐    │  │  - search_context  │  │
│   │    ┌──────────────┐    │  │    │ SafetyCheck  │    │  │                    │  │
│   │    │apply_safety  │    │  │    └──────────────┘    │  └────────────────────┘  │
│   │    └──────────────┘    │  └────────────────────────┘                          │
│   └────────────────────────┘                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                 INFRASTRUCTURE LAYER                                   │
│                                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────────┐    │
│   │                           VECTOR STORE                                       │    │
│   │   ┌─────────────────────┐           ┌─────────────────────┐                 │    │
│   │   │    PgVectorStore    │           │  InMemoryVectorStore │                │    │
│   │   │    (production)     │           │     (testing)        │                │    │
│   │   │   PostgreSQL+HNSW   │           │   Cosine similarity  │                │    │
│   │   └─────────────────────┘           └─────────────────────┘                 │    │
│   └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────────┐    │
│   │                           EMBEDDINGS                                         │    │
│   │   ┌─────────────────────┐           ┌─────────────────────┐                 │    │
│   │   │  OpenAIEmbeddings   │           │    MockEmbeddings   │                 │    │
│   │   │ text-embedding-3-sm │           │   Random vectors    │                 │    │
│   │   └─────────────────────┘           └─────────────────────┘                 │    │
│   └─────────────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                   OUTPUT SCHEMA                                        │
│                                                                                       │
│   LabInsightsSummary                                                                  │
│   ├── summary: str                      # 2-3 sentence overview                       │
│   ├── key_insights: list[MarkerInsight] # Detailed per-marker analysis               │
│   │   ├── marker: str                   # "TSH"                                       │
│   │   ├── status: Literal[high|low|normal|borderline]                                 │
│   │   ├── value: float                  # 5.5                                         │
│   │   ├── trend: Literal[increasing|decreasing|stable|unknown]                        │
│   │   └── action: str                   # "Discuss with doctor"                       │
│   ├── recommended_topics_for_doctor: list[str]                                        │
│   ├── lifestyle_considerations: list[str]                                             │
│   └── safety_notes: list[SafetyNote]    # Required disclaimers                        │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Key Insights & Recommendations

### Strengths

1. **Multi-Framework Evaluation** - Using DeepEval, RAGAS, and custom judges catches different issue types
2. **Protocol-Based Design** - Easy to swap implementations for testing or production
3. **Layered Evaluation** - Fast checks first, expensive LLM evals only if needed
4. **Multiple Agent Paradigms** - LangGraph and DSPy demonstrate different approaches
5. **Comprehensive Documentation** - Interview talking points embedded throughout
6. **CI Integration** - Eval gates block PRs with quality issues

### Potential Improvements

1. **Async Support** - Add async versions of agent and eval functions for better throughput
2. **Caching** - Cache LLM responses during development to reduce API costs
3. **More Golden Cases** - Expand from 5 to 50+ cases covering edge cases
4. **Baseline Tracking** - Store historical baselines for trend analysis
5. **Metric Dashboard** - Visualize evaluation trends over time

### Security Considerations

1. **API Keys** - Currently loaded from environment; consider secrets manager
2. **Input Validation** - Lab values should be validated before processing
3. **Output Sanitization** - Healthcare responses should be reviewed for safety
4. **Rate Limiting** - Add rate limiting for API calls in production

### Performance Optimization Opportunities

1. **Batch Embeddings** - Already implemented, but could be expanded
2. **Parallel Evaluation** - Run independent evals concurrently
3. **Model Selection** - Use smaller models for simple tasks (GPT-4o-mini for schema)
4. **Connection Pooling** - For PostgreSQL in production

---

## 11. Interview Talking Points Summary

1. **"Why multiple agent implementations?"**
   > LangGraph is imperative (explicit state), DSPy is declarative (optimizable prompts). Having both demonstrates understanding of both paradigms.

2. **"How do you test LLM-based code?"**
   > Multiple layers: Mocks for unit tests, golden cases for integration, LLM-as-judge for semantic quality. InMemoryVectorStore enables full flow testing without databases.

3. **"Why multiple evaluation frameworks?"**
   > Each catches different issues. DeepEval for custom G-Eval metrics, RAGAS for RAG quality, custom judge for domain rubrics. Running all three builds confidence.

4. **"What's the eval pipeline for?"**
   > CI for prompts. Every PR runs through schema validation, retrieval quality, LLM-as-judge, and performance regression. Fast checks first, expensive checks only if earlier ones pass.

5. **"How does DSPy optimization work?"**
   > DSPy separates WHAT (signatures) from HOW (prompts). Define a metric, let optimizers find effective prompts. Meta-learning for the evaluator itself.

---

*Generated by Claude Code - Comprehensive Codebase Analysis*
*Last Updated: 2025-12-08*
