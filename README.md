# Agent Eval Pipeline

A production-style evaluation gates system for AI agents, demonstrating **LangGraph**, **DSPy**, **DeepEval**, and **RAGAS** - the exact tech stack used at Function Health.

## Overview

This project implements a health insights agent that analyzes lab results and provides educational information. It showcases:

- **Multiple Agent Implementations**: LangGraph (imperative) and DSPy (declarative)
- **Multi-Framework Evaluation**: DeepEval, RAGAS, custom LLM-as-Judge, DSPy Judge
- **Comprehensive Eval Gates**: Schema, Retrieval, Faithfulness, Safety, Performance
- **Production Patterns**: DI, Protocols, Test doubles, Factory functions
- **Interview-Ready Architecture**: Documented patterns and talking points

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set up environment
cp .env.example .env  # Add your OPENAI_API_KEY

# Run tests (no API key needed for most)
PYTHONPATH=src pytest tests/ -v

# Run the eval pipeline
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner
```

## Architecture

```
agent_eval_pipeline/
├── core/                   # Protocols (interfaces)
│   └── protocols.py        # EmbeddingProvider, VectorStore, etc.
├── embeddings/             # Text embeddings (gold standard pattern)
│   └── openai_embeddings.py
├── retrieval/              # Vector store + RAG
│   ├── store.py            # PgVector + InMemory implementations
│   └── seeds/              # Medical knowledge base
├── agent/                  # Agent implementations (2 active)
│   ├── state.py            # AgentState TypedDict
│   ├── nodes/              # LangGraph node functions
│   ├── graph.py            # LangGraph graph builder
│   ├── langgraph_runner.py # LangGraph execution
│   └── dspy_react_agent.py # DSPy ReAct with tools
├── evals/                  # Evaluation gates
│   ├── schema_eval.py      # Pydantic validation
│   ├── retrieval_eval.py   # RAG quality (P/R/F1)
│   ├── judge/              # LLM-as-judge
│   │   ├── dspy_judge.py   # DSPy-optimizable judge
│   │   └── evaluator.py    # Traditional judge
│   ├── deepeval/           # DeepEval integration
│   ├── ragas/              # RAGAS integration
│   └── perf/               # Performance regression
├── harness/                # Evaluation orchestration
│   ├── runner.py           # Eval gates runner
│   └── comparative_runner.py # Agent comparison runner
├── observability/          # Phoenix/OpenTelemetry tracing
└── cli/                    # Command-line interface
```

## Running the Agents

The pipeline supports two agent implementations, selectable via `AGENT_TYPE` env var or `agent_type` parameter.

### LangGraph Agent (State Machine) - Default

The LangGraph agent uses an explicit state machine with nodes for retrieval, analysis, and safety checks.

```bash
# Run via unified interface
PYTHONPATH=src python -c "
from agent_eval_pipeline.agent import run_agent
from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id

case = get_case_by_id('thyroid-001')
result = run_agent(case, agent_type='langgraph')

print('Summary:', result.output.summary)
print('Latency:', result.latency_ms, 'ms')
print('Retrieved docs:', len(result.retrieved_docs or []))
"
```

**When to use LangGraph:**
- Complex multi-step workflows with RAG retrieval
- Need explicit state management and checkpointing
- Want to trace execution through discrete nodes
- Human-in-the-loop requirements

### DSPy ReAct Agent (Tool-Using)

The ReAct agent uses DSPy's declarative approach with explicit tool calls for reasoning.

```bash
# Run the ReAct agent with tools
PYTHONPATH=src python -c "
from agent_eval_pipeline.agent import run_agent
from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id

case = get_case_by_id('thyroid-001')
result = run_agent(case, agent_type='dspy_react')

print('Summary:', result.output.summary)
print('Tools used:', result.tools_used)
print('Reasoning steps:', result.reasoning_steps)
"
```

**Available tools:**
- `lookup_reference_range` - Get standard ranges for markers
- `check_medication_interaction` - Check if meds affect lab values
- `search_medical_context` - Search medical knowledge base

**When to use DSPy ReAct:**
- Want explicit reasoning traces
- Need tool-based information gathering
- Prefer declarative prompt specifications
- Experimenting with different approaches

## Running Evaluations

### Individual Gates

```bash
# Schema validation (fast, deterministic)
PYTHONPATH=src python -m agent_eval_pipeline.evals.schema_eval

# Retrieval quality (no LLM calls)
PYTHONPATH=src python -m agent_eval_pipeline.evals.retrieval_eval

# LLM-as-judge (uses GPT-4o)
PYTHONPATH=src python -m agent_eval_pipeline.evals.judge_eval

# Performance regression
PYTHONPATH=src python -m agent_eval_pipeline.evals.perf_eval
PYTHONPATH=src python -m agent_eval_pipeline.evals.perf_eval --update-baseline
```

### Full Pipeline

```bash
# Run all gates
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner

# Skip expensive LLM-as-judge (for quick testing)
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --skip-expensive

# JSON output for CI
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --json
```

### DeepEval Evaluation

DeepEval provides pytest-native testing with custom G-Eval metrics:

```bash
# Run DeepEval with pytest
PYTHONPATH=src deepeval test run tests/test_deepeval_integration.py

# Or run standalone
PYTHONPATH=src python -c "
from agent_eval_pipeline.evals.deepeval import run_deepeval_evaluation

report = run_deepeval_evaluation(verbose=True)
print(f'Pass rate: {report.pass_rate:.1%}')
print(f'Safety avg: {report.metric_averages.get(\"Safety Compliance\", 0):.2f}')
"
```

**DeepEval Metrics:**
- `clinical_correctness` - Medically accurate interpretations
- `safety_compliance` - No diagnoses, includes disclaimers (threshold: 0.9)
- `completeness` - Covers all important points
- `answer_clarity` - Accessible language

### RAGAS Evaluation

RAGAS specializes in RAG evaluation with faithfulness and context metrics:

```bash
# Run RAGAS evaluation
PYTHONPATH=src python -c "
from agent_eval_pipeline.evals.ragas import run_ragas_evaluation

report = run_ragas_evaluation(verbose=True)
print(f'Faithfulness: {report.metric_averages.get(\"faithfulness\", 0):.2f}')
print(f'Context Precision: {report.metric_averages.get(\"context_precision\", 0):.2f}')
"
```

**RAGAS Metrics:**
- `faithfulness` - Response grounded in retrieved context
- `context_precision` - Retrieved docs are relevant
- `context_recall` - All needed docs retrieved
- `answer_relevancy` - Response addresses the question

### Unified Evaluation

Run all frameworks together for comprehensive evaluation:

```bash
# Run unified evaluation
PYTHONPATH=src python -m agent_eval_pipeline.harness.unified_runner -v

# Select specific frameworks
PYTHONPATH=src python -m agent_eval_pipeline.harness.unified_runner -f deepeval ragas

# JSON output for CI
PYTHONPATH=src python -m agent_eval_pipeline.harness.unified_runner --json
```

### DSPy vs Traditional Judge

```bash
# Compare judge implementations
PYTHONPATH=src python -c "
from agent_eval_pipeline.evals.judge.dspy_judge import compare_judges
from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id
from agent_eval_pipeline.agent import run_agent

case = get_case_by_id('thyroid-001')
result = run_agent(case)
compare_judges(case, result.output)
"
```

## Testing

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run only DSPy tests (no API key needed)
PYTHONPATH=src pytest tests/test_dspy.py -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=agent_eval_pipeline
```

## Key Patterns

### 1. Protocol-First Design
```python
class VectorStore(Protocol):
    def search(self, query: str, limit: int) -> list[DocumentResult]: ...

# Multiple implementations
class PgVectorStore:  # Production
class InMemoryVectorStore:  # Testing
```

### 2. Dependency Injection
```python
def create_retrieve_node(store: VectorStore):  # Inject, don't create
    def retrieve(state: AgentState) -> dict:
        results = store.search(...)
        return {"retrieved_docs": results}
    return retrieve
```

### 3. DSPy Signatures
```python
class AnalyzeLabs(dspy.Signature):
    """Analyze lab results."""  # Becomes the prompt instruction
    query: str = dspy.InputField(desc="User's question")
    labs: str = dspy.InputField(desc="Lab values")
    summary: str = dspy.OutputField(desc="Analysis summary")
```

### 4. LangGraph State Machine
```python
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("analyze", analyze_node)
workflow.add_edge("retrieve", "analyze")
```

## LangGraph vs DSPy ReAct Comparison

| Aspect | LangGraph | DSPy ReAct |
|--------|-----------|------------|
| **Paradigm** | Imperative state machine | Declarative with tools |
| **Retrieval** | RAG via vector store | Tool-based lookup |
| **State** | Explicit TypedDict | Implicit in ReAct loop |
| **Reasoning** | Node-by-node | Thought-Action-Observation |
| **Best for** | Complex workflows | Explicit reasoning traces |
| **Debugging** | State inspection | Tool call history |

## Evaluation Framework Comparison

| Aspect | DeepEval | RAGAS | Custom Judge | DSPy Judge |
|--------|----------|-------|--------------|------------|
| **Focus** | General LLM testing | RAG evaluation | Domain-specific | Optimizable |
| **Pytest** | Native `assert_test` | Custom setup | Custom setup | Custom setup |
| **Key Metrics** | G-Eval, Hallucination | Faithfulness, Context P/R | Rubric-based | Signature-based |
| **Customization** | G-Eval criteria | Limited | Full control | DSPy modules |
| **Speed** | Moderate | Moderate | Fast (cached) | Moderate |
| **Best For** | CI/CD gates | RAG quality | Healthcare domain | Auto-tuning |

## Interview Talking Points

1. **"Why two agent implementations?"**
   > "LangGraph and DSPy ReAct represent different paradigms. LangGraph is a state machine with RAG retrieval - I control the exact flow through nodes. DSPy ReAct uses tool-based reasoning with explicit thought traces. Both produce the same LabInsightsSummary schema, so I can fairly compare them using the comparative eval runner."

2. **"How do you test LLM-based code?"**
   > "Multiple layers: Protocol-based mocks for unit tests, golden cases for integration tests, LLM-as-judge for semantic quality. The InMemoryVectorStore and MockEmbeddings let me test the full flow without API calls."

3. **"Why multiple evaluation frameworks?"**
   > "Each framework catches different issues. DeepEval gives us pytest-native testing with custom G-Eval metrics - perfect for CI gates. RAGAS specializes in RAG evaluation with faithfulness and context precision. Our custom judge uses domain-specific healthcare rubrics. Running all three means we catch different types of issues."

4. **"What's the eval pipeline for?"**
   > "Multiple gates that run in CI: schema validation catches structural issues instantly, retrieval eval checks RAG quality without LLM calls, then DeepEval/RAGAS/custom judge assess semantic quality in parallel. Fast checks first, expensive checks only if earlier ones pass."

5. **"How does DSPy optimization work?"**
   > "DSPy separates what you want (signatures) from how to get it (prompts). Optimizers like BootstrapFewShot find effective prompts by trying examples. MIPROv2 even optimizes the instructions. I define a metric, and DSPy maximizes it."

6. **"What's the difference between DeepEval and RAGAS?"**
   > "DeepEval is general-purpose LLM testing with pytest integration and custom G-Eval metrics. RAGAS is specialized for RAG - it extracts claims from responses and verifies each against the retrieved context. DeepEval for CI gates, RAGAS for deep RAG analysis."

## Environment Variables

```bash
OPENAI_API_KEY=sk-...          # Required for LLM calls
JUDGE_MODEL=gpt-4o             # Model for LLM-as-judge (default: gpt-4o)
AGENT_MODEL=gpt-4o-mini        # Model for agent (default: gpt-4o-mini)
USE_POSTGRES=false             # Use PgVector vs InMemory store
AGENT_TYPE=langgraph           # Default agent type (langgraph or dspy_react)

# Phoenix Observability (optional)
PHOENIX_ENABLED=false              # Set to true to enable tracing
PHOENIX_PROJECT_NAME=agent-eval-pipeline
PHOENIX_COLLECTOR_ENDPOINT=        # Leave empty for local Phoenix UI
PHOENIX_CAPTURE_LLM_CONTENT=true   # Log prompts/responses
```

## Operator Instructions

This section covers how to run, monitor, and troubleshoot the eval pipeline in production.

### Installation

```bash
# Basic installation
pip install -e .

# With observability (Phoenix tracing)
pip install -e ".[observability]"

# With dev dependencies
pip install -e ".[dev]"

# All extras
pip install -e ".[dev,observability]"
```

### Running the Eval Pipeline

#### Standard Eval Run (All Gates)

```bash
# Run all 4 gates: Schema → Retrieval → Judge → Performance
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner

# Skip expensive LLM-as-judge (for quick testing)
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --skip-expensive

# Continue running all gates even if one fails
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --no-fail-fast

# JSON output for CI/CD parsing
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --json --quiet
```

**Exit codes:**
- `0` - All gates passed
- `1` - One or more gates failed

#### Comparative Eval (LangGraph vs DSPy ReAct)

```bash
# Compare both agents side-by-side
PYTHONPATH=src python -m agent_eval_pipeline.harness.comparative_runner

# Compare specific agents
PYTHONPATH=src python -m agent_eval_pipeline.harness.comparative_runner --agents langgraph dspy_react

# JSON output
PYTHONPATH=src python -m agent_eval_pipeline.harness.comparative_runner --json
```

**Output includes:**
- Schema pass rate per agent
- Latency metrics (avg, p50, p95)
- Token usage (cost proxy)
- Winner recommendation

### Observability with Phoenix

Phoenix provides LLM-native tracing with auto-instrumentation for OpenAI, LangChain, and DSPy calls.

#### Local Development (Phoenix UI)

```bash
# Install observability deps
pip install -e ".[observability]"

# Enable Phoenix and run eval
PHOENIX_ENABLED=true PYTHONPATH=src python -m agent_eval_pipeline.harness.runner
```

Phoenix UI opens automatically at `http://localhost:6006` showing:
- Full trace hierarchy (harness → gates → agent → LLM calls)
- Token usage per span
- Latency breakdown
- LLM prompts and responses (if `PHOENIX_CAPTURE_LLM_CONTENT=true`)

#### Production (Remote Phoenix/OTLP)

```bash
# Connect to remote Phoenix instance
export PHOENIX_ENABLED=true
export PHOENIX_COLLECTOR_ENDPOINT=https://your-phoenix-instance.com/v1/traces
export PHOENIX_PROJECT_NAME=prod-eval-pipeline

PYTHONPATH=src python -m agent_eval_pipeline.harness.runner
```

#### Trace Hierarchy

```
eval_harness_run                    # Root span
├── eval_gate.schema_validation     # Gate spans with pass/fail
├── eval_gate.retrieval_quality
├── eval_gate.llm-as-judge
│   └── gen_ai.chat                 # Auto-instrumented LLM calls
└── eval_gate.performance_regression
```

**Key attributes on spans:**
- `eval.harness.run_id` - UUID to correlate all spans in a run
- `eval.gate.name` - Gate name (e.g., "Schema Validation")
- `eval.gate.status` - "passed", "failed", or "error"
- `agent.type` - "langgraph" or "dspy_react"
- `gen_ai.usage.total_tokens` - Token count per LLM call

### CI/CD Integration

#### GitHub Actions Example

```yaml
name: Eval Gates
on: [pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run eval pipeline
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --json > eval-report.json

      - name: Upload eval report
        uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: eval-report.json
```

#### Parsing JSON Output

```bash
# Check if all gates passed
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --json --quiet | jq '.all_passed'

# Get gate summaries
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --json --quiet | jq '.gates[] | {name, status, summary}'
```

### Troubleshooting

#### Common Issues

**1. Schema Validation Failures**
```
[FAIL] Schema Validation: Pass rate: 60.0%
```
- Check `expected_marker_statuses` in golden cases match LLM output
- Review the agent's interpretation of borderline values
- Update golden cases if LLM interpretation is reasonable

**2. Retrieval Quality Failures**
```
[FAIL] Retrieval Quality: Avg F1: 0.51
```
- Check `expected_doc_ids` in golden cases are comprehensive
- Review vector store seeding in `retrieval/seeds/`
- Verify embeddings are working correctly

**3. LLM-as-Judge Low Scores**
```
[FAIL] LLM-as-Judge: Avg score: 3.20/5
```
- Review judge output for specific dimension failures
- Check if agent output includes required safety disclaimers
- Verify clinical accuracy of responses

**4. Performance Regression**
```
[FAIL] Performance Regression: p95 latency exceeded baseline
```
- Check for API latency issues
- Review token usage (more tokens = slower)
- Run `--update-baseline` if regression is acceptable

#### Debug Mode

```bash
# Verbose output with full traces
PYTHONPATH=src python -m agent_eval_pipeline.harness.runner 2>&1 | tee eval.log

# Run single gate for debugging
PYTHONPATH=src python -m agent_eval_pipeline.evals.schema_eval
PYTHONPATH=src python -m agent_eval_pipeline.evals.retrieval_eval
PYTHONPATH=src python -m agent_eval_pipeline.evals.judge_eval
PYTHONPATH=src python -m agent_eval_pipeline.evals.perf_eval
```

#### Phoenix Debugging

With Phoenix enabled, you can:
1. View exact prompts sent to LLMs
2. See token counts per call
3. Identify slow spans in the trace
4. Debug agent reasoning steps

```bash
# Enable Phoenix with full content capture
PHOENIX_ENABLED=true PHOENIX_CAPTURE_LLM_CONTENT=true \
  PYTHONPATH=src python -m agent_eval_pipeline.harness.runner
```

### Monitoring Recommendations

1. **Track eval pass rates over time** - Detect regressions early
2. **Monitor p95 latency** - Catch performance degradation
3. **Alert on safety score drops** - Critical for healthcare domain
4. **Track token usage** - Cost monitoring
5. **Compare agent versions** - Use comparative runner for A/B testing

### Scaling Considerations

- **Parallel gate execution**: Currently sequential; can parallelize non-dependent gates
- **Caching**: Agent results are not cached between gates (room for optimization)
- **Rate limiting**: No built-in rate limiting; rely on OpenAI client defaults
- **Batch processing**: Golden cases run sequentially; parallelizable with async

## Project Structure Details

### Elevated Modules (Production-Ready)

These modules follow the "gold standard" pattern with Protocol + Implementation + Test Double + Factory:

- `embeddings/` - Text embedding generation
- `retrieval/` - Vector store operations
- `agent/nodes/` - LangGraph node functions
- `evals/judge/` - LLM-as-judge evaluation
- `evals/perf/` - Performance regression detection

### Entry Points

```toml
[project.scripts]
agent-eval = "agent_eval_pipeline.cli:main"
agent-eval-schema = "agent_eval_pipeline.cli:run_schema_cli"
agent-eval-judge = "agent_eval_pipeline.cli:run_judge_cli"
agent-eval-perf = "agent_eval_pipeline.cli:run_perf_cli"
agent-eval-all = "agent_eval_pipeline.cli:run_all_cli"
```

## License

MIT
