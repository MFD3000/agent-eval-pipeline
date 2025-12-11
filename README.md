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
├── agent/                  # Agent implementations
│   ├── langgraph_agent.py  # LangGraph state machine
│   ├── dspy_agent.py       # DSPy declarative agent
│   └── dspy_react_agent.py # DSPy ReAct with tools
├── evals/                  # Evaluation gates
│   ├── schema_eval.py      # Pydantic validation
│   ├── retrieval_eval.py   # RAG quality (P/R/F1)
│   ├── judge/              # LLM-as-judge
│   │   ├── dspy_judge.py   # DSPy-optimizable judge
│   │   └── evaluator.py    # Traditional judge
│   ├── deepeval/           # DeepEval integration
│   │   ├── adapters.py     # GoldenCase → LLMTestCase
│   │   ├── metrics.py      # Custom G-Eval healthcare metrics
│   │   └── evaluator.py    # DeepEval runner
│   ├── ragas/              # RAGAS integration
│   │   ├── adapters.py     # GoldenCase → SingleTurnSample
│   │   ├── metrics.py      # Faithfulness, Context P/R
│   │   └── evaluator.py    # RAGAS runner
│   └── perf/               # Performance regression
├── harness/                # Evaluation orchestration
│   ├── runner.py           # Original eval runner
│   └── unified_runner.py   # Multi-framework runner
└── cli/                    # Command-line interface
```

## Running the Agents

### LangGraph Agent (State Machine)

The LangGraph agent uses an explicit state machine with nodes for retrieval, analysis, and safety checks.

```bash
# Run via Python
PYTHONPATH=src python -c "
from agent_eval_pipeline.agent import run_agent
from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id

case = get_case_by_id('thyroid-001')
result = run_agent(case, use_langgraph=True)

print('Summary:', result.output.summary)
print('Latency:', result.latency_ms, 'ms')
"
```

**When to use LangGraph:**
- Complex multi-step workflows
- Human-in-the-loop requirements
- Explicit state management needed
- Checkpointing/resumption required

### DSPy Agent (Declarative)

The DSPy agent uses signatures (I/O specs) and lets DSPy handle prompting.

```bash
# Run the basic DSPy agent
PYTHONPATH=src python -c "
from agent_eval_pipeline.agent.dspy_agent import run_dspy_agent

result = run_dspy_agent(
    query='What do my thyroid results indicate?',
    labs=[
        {'marker': 'TSH', 'value': 5.5, 'unit': 'mIU/L', 'ref_low': 0.4, 'ref_high': 4.0},
        {'marker': 'Free T4', 'value': 0.9, 'unit': 'ng/dL', 'ref_low': 0.8, 'ref_high': 1.8},
    ],
    symptoms=['fatigue', 'weight gain']
)

print('Summary:', result.output.summary)
print('Reasoning:', result.reasoning[:200], '...')
"
```

**When to use DSPy:**
- Prompt optimization is important
- Structured output requirements
- Want automatic few-shot example selection
- Experimenting with different approaches

### DSPy ReAct Agent (Tool-Using)

The ReAct agent can call tools to gather information before responding.

```bash
# Run the ReAct agent with tools
PYTHONPATH=src python -c "
from agent_eval_pipeline.agent.dspy_react_agent import run_react_agent

result = run_react_agent(
    query='I take levothyroxine and biotin. My TSH is low - should I worry?',
    labs=[
        {'marker': 'TSH', 'value': 0.2, 'unit': 'mIU/L', 'ref_low': 0.4, 'ref_high': 4.0},
        {'marker': 'Free T4', 'value': 1.1, 'unit': 'ng/dL', 'ref_low': 0.8, 'ref_high': 1.8},
    ],
    medications=['levothyroxine 75mcg', 'biotin 5000mcg'],
    symptoms=['anxiety']
)

print('Tools used:', result.tools_used)
print('Analysis:', result.analysis)
"
```

**Available tools:**
- `lookup_reference_range` - Get standard ranges for markers
- `check_medication_interaction` - Check if meds affect lab values
- `search_medical_context` - Search medical knowledge base

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

## DSPy Optimization

DSPy can automatically optimize prompts using examples:

```python
from agent_eval_pipeline.agent.dspy_agent import (
    create_dspy_agent,
    optimize_agent,
    golden_cases_to_trainset,
)
from agent_eval_pipeline.golden_sets.thyroid_cases import get_all_golden_cases

# Create agent and training set
agent = create_dspy_agent()
trainset = golden_cases_to_trainset(get_all_golden_cases())

# Optimize (finds better prompts automatically)
optimized_agent = optimize_agent(
    agent,
    trainset=trainset,
    optimizer_type="bootstrap",  # or "mipro" for more thorough
)

# Use the optimized agent
result = optimized_agent(query="...", labs=[...])
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

## LangGraph vs DSPy Comparison

| Aspect | LangGraph | DSPy |
|--------|-----------|------|
| **Paradigm** | Imperative (how) | Declarative (what) |
| **Prompts** | Hand-written | Auto-optimized |
| **State** | Explicit TypedDict | Implicit |
| **Testing** | Node isolation | Signature introspection |
| **Best for** | Complex workflows | Prompt optimization |
| **Debugging** | State inspection | Reasoning traces |

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

1. **"Why multiple agent implementations?"**
   > "LangGraph and DSPy represent different paradigms. LangGraph is imperative - I control the exact flow with explicit state. DSPy is declarative - I specify what I want and it figures out how. Having both lets me choose the right tool for each situation."

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
AGENT_TYPE=langgraph           # Default agent type (langgraph or legacy)
```

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
