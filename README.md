# Agent Eval Pipeline

A production-style evaluation gates system for AI agents, demonstrating **LangGraph**, **DSPy**, **DeepEval**, and **RAGAS**.

## Overview

This project implements a health insights agent that analyzes lab results and provides educational information. It showcases:

- **Multiple Agent Implementations**: LangGraph (imperative) and DSPy (declarative)
- **Multi-Framework Evaluation**: DeepEval, RAGAS, custom LLM-as-Judge, DSPy Judge
- **Comprehensive Eval Gates**: Schema, Retrieval, Faithfulness, Safety, Performance
- **Production Patterns**: Dependency Injection, Protocols, Test doubles, Factory functions
- **Observability**: Phoenix/OpenTelemetry tracing with auto-instrumentation

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set up environment
cp .env.example .env  # Add your OPENAI_API_KEY

# Run tests (no API key needed for most)
pytest tests/ -v

# Run the eval pipeline (requires OPENAI_API_KEY)
python -m agent_eval_pipeline.harness.runner
```

## Running Tests

```bash
# Run all unit tests (no API key required)
pytest tests/ -v

# Exclude tests that require LLM calls
pytest tests/ -v \
  --ignore=tests/test_deepeval_integration.py \
  --ignore=tests/test_ragas_integration.py \
  --ignore=tests/test_llm_judge.py

# Run with coverage
pytest tests/ --cov=agent_eval_pipeline --cov-report=term-missing

# Run specific test files
pytest tests/test_schema_eval.py -v
pytest tests/test_harness.py -v
pytest tests/test_dspy.py -v
```

## Running the Eval Pipeline

### Quick Validation (No LLM Judge)

```bash
# Run schema + retrieval + performance gates (fast, minimal API cost)
python -m agent_eval_pipeline.harness.runner --skip-expensive
```

### Full Pipeline

```bash
# Run all gates: Schema → Retrieval → LLM-as-Judge → Performance
python -m agent_eval_pipeline.harness.runner

# Continue running all gates even if one fails
python -m agent_eval_pipeline.harness.runner --no-fail-fast

# JSON output for CI/CD
python -m agent_eval_pipeline.harness.runner --json --quiet
```

### Individual Gates

```bash
# Schema validation (fast, deterministic)
python -m agent_eval_pipeline.evals.schema_eval

# Retrieval quality (no LLM calls)
python -m agent_eval_pipeline.evals.retrieval_eval

# LLM-as-judge (uses GPT-4o)
python -m agent_eval_pipeline.evals.judge_eval

# Performance regression
python -m agent_eval_pipeline.evals.perf_eval
python -m agent_eval_pipeline.evals.perf_eval --update-baseline
```

### DeepEval and RAGAS

```bash
# Run DeepEval with pytest (requires OPENAI_API_KEY)
deepeval test run tests/test_deepeval_integration.py

# Run RAGAS evaluation
python -c "
from agent_eval_pipeline.evals.ragas import run_ragas_evaluation
report = run_ragas_evaluation(verbose=True)
print(f'Faithfulness: {report.metric_averages.get(\"faithfulness\", 0):.2f}')
"

# Run unified evaluation (all frameworks)
python -m agent_eval_pipeline.harness.unified_runner -v
```

## Architecture Overview

### Evaluation Gate Pipeline

The harness orchestrates evaluation gates in order of cost, enabling fast failure detection:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Eval Gate Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
│  │   Schema    │ -> │  Retrieval  │ -> │  LLM-as-    │ -> │   Perf    │ │
│  │ Validation  │    │   Quality   │    │   Judge     │    │  Regress  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘ │
│       ~0ms              ~0ms              ~2-5s              ~0ms       │
│   (Pydantic)       (F1 metrics)       (GPT-4 eval)        (baseline)   │
│                                                                          │
│  ▪ Fast, cheap checks first                                             │
│  ▪ Expensive LLM-as-judge only if structure passes                      │
│  ▪ Fail-fast mode skips downstream gates on failure                     │
│  ▪ Context sharing: agents run ONCE, results shared across all gates    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Context Sharing**: The harness runs agents once and shares `AgentRunContext` across all evaluators, reducing LLM API calls by 4-5x and ensuring all gates score the same output.

### Safety Guardrails (Healthcare Domain)

Multiple layers ensure safe outputs:

1. **Schema-Level**: Required `SafetyNote` fields with `non_diagnostic` type
2. **Agent-Level**: Safety node injects mandatory disclaimers
3. **Eval-Level**: `must_not_diagnose` criterion, `must_not_contain` prohibited phrases
4. **Golden Cases**: Explicit assertions for doctor recommendations and diagnostic language

### Agent Implementations

**LangGraph Agent** (default): Explicit state machine with RAG retrieval
```bash
python -c "
from agent_eval_pipeline.agent import run_agent
from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id

case = get_case_by_id('thyroid-001')
result = run_agent(case, agent_type='langgraph')
print('Summary:', result.output.summary)
"
```

**DSPy ReAct Agent**: Tool-using agent with explicit reasoning traces
```bash
python -c "
from agent_eval_pipeline.agent import run_agent
from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id

case = get_case_by_id('thyroid-001')
result = run_agent(case, agent_type='dspy_react')
print('Summary:', result.output.summary)
print('Tools used:', result.tools_used)
"
```

## Directory Structure

```
agent_eval_pipeline/
├── core/                   # Protocols (interfaces)
├── embeddings/             # Text embeddings
├── retrieval/              # Vector store + RAG
├── agent/                  # Agent implementations
│   ├── state.py            # AgentState TypedDict
│   ├── nodes/              # LangGraph node functions
│   ├── graph.py            # LangGraph graph builder
│   └── dspy_react_agent.py # DSPy ReAct with tools
├── evals/                  # Evaluation gates
│   ├── schema_eval.py      # Pydantic validation
│   ├── retrieval_eval.py   # RAG quality (P/R/F1)
│   ├── judge/              # LLM-as-judge
│   ├── deepeval/           # DeepEval integration
│   ├── ragas/              # RAGAS integration
│   └── perf/               # Performance regression
├── harness/                # Evaluation orchestration
│   ├── runner.py           # Eval gates runner
│   ├── context.py          # AgentRunContext for result sharing
│   └── comparative_runner.py
├── golden_sets/            # Test cases with expectations
└── observability/          # Phoenix/OpenTelemetry tracing
```

## Observability with Phoenix

Enable tracing to visualize the full execution flow:

```bash
# Start Phoenix and run eval
PHOENIX_ENABLED=true python -m agent_eval_pipeline.harness.runner
```

Phoenix UI opens at `http://localhost:6006` showing:
- Full trace hierarchy (harness → gates → agent → LLM calls)
- Token usage and latency per span
- Agent type (`langgraph` vs `dspy_react`) in span attributes

> **Privacy**: `PHOENIX_CAPTURE_LLM_CONTENT` defaults to `false`. Set to `true` only when debugging prompts in a compliant environment.

## Environment Variables

```bash
OPENAI_API_KEY=sk-...          # Required for LLM calls
JUDGE_MODEL=gpt-4o             # Model for LLM-as-judge (default: gpt-4o)
AGENT_MODEL=gpt-4o-mini        # Model for agent (default: gpt-4o-mini)
USE_POSTGRES=false             # Use PgVector vs InMemory store
AGENT_TYPE=langgraph           # Default agent type (langgraph or dspy_react)

# Phoenix Observability
PHOENIX_ENABLED=false
PHOENIX_PROJECT_NAME=agent-eval-pipeline
PHOENIX_COLLECTOR_ENDPOINT=    # Leave empty for local Phoenix UI
PHOENIX_CAPTURE_LLM_CONTENT=false
```

## Key Patterns

### Protocol-First Design
```python
class VectorStore(Protocol):
    def search(self, query: str, limit: int) -> list[DocumentResult]: ...

# Multiple implementations: PgVectorStore (prod), InMemoryVectorStore (test)
```

### Dependency Injection
```python
def create_retrieve_node(store: VectorStore):  # Inject, don't create
    def retrieve(state: AgentState) -> dict:
        results = store.search(...)
        return {"retrieved_docs": results}
    return retrieve
```

### Golden Case Assertions
```python
@dataclass
class GoldenCase:
    must_include_doctor_recommendation: bool = True
    must_not_diagnose: bool = True
    must_not_contain: list[str]  # Prohibited phrases
```

## CI/CD Integration

```yaml
# GitHub Actions example
- name: Run eval pipeline
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    python -m agent_eval_pipeline.harness.runner --json > eval-report.json

- name: Check results
  run: |
    jq '.all_passed' eval-report.json
```

**Exit codes:**
- `0` - All gates passed
- `1` - One or more gates failed

## Troubleshooting

**Schema Validation Failures**: Check `expected_marker_statuses` in golden cases match LLM output interpretation.

**Retrieval Quality Failures**: Verify `expected_doc_ids` in golden cases and vector store seeding.

**LLM-as-Judge Low Scores**: Review judge output for specific dimension failures; check safety disclaimers.

**Performance Regression**: Run `--update-baseline` if regression is acceptable after review.

## License

MIT
