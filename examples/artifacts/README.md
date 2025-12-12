# Example Artifacts

Real output from running the evaluation pipeline. These demonstrate what each component produces.

## Files

| File | Description |
|------|-------------|
| `pipeline_output.txt` | Full evaluation harness run showing all gates (Schema, Retrieval, Judge, Perf) |
| `deepeval_output.txt` | DeepEval pytest integration with G-Eval metrics (safety, clinical correctness) |
| `ragas_evaluation_output.txt` | RAGAS RAG metrics (faithfulness, context recall, answer relevancy) |
| `ragas_output.txt` | RAGAS adapter unit tests |
| `unit_tests_output.txt` | Full pytest run (189 tests) excluding slow LLM integration tests |
| `comparative_eval_output.json` | Side-by-side comparison of LangGraph vs DSPy ReAct agents |
| `otel_phoenix_output.txt` | OpenTelemetry tracing with Phoenix observability |

## Key Observations

### Pipeline (`pipeline_output.txt`)
- Agents run **once**, results shared across all eval gates
- Gates run in order: Schema -> Retrieval -> Judge -> Perf
- Failed gate stops pipeline (fail-fast)

### DeepEval (`deepeval_output.txt`)
- Safety compliance metric catches definitive diagnostic language
- Shows LLM-as-judge providing actionable feedback
- Real failures demonstrate eval catching issues

### RAGAS (`ragas_evaluation_output.txt`)
- RAG-specific metrics: faithfulness, context precision/recall
- Thresholds configurable per metric
- Progress bar shows async evaluation

### Comparative (`comparative_eval_output.json`)
- LangGraph: RAG-based, retrieves 5 docs/case, fewer tokens
- DSPy ReAct: Tool-based, uses 1.8 tools/case, higher schema pass rate
- Automated winner determination with reasoning

### OTEL/Phoenix (`otel_phoenix_output.txt`)
- Span hierarchy for debugging
- Token usage tracking per operation
- Latency attribution across pipeline stages

## Regenerating

```bash
# Full pipeline
python -m agent_eval_pipeline.harness.runner

# DeepEval tests
python -m pytest tests/test_deepeval_integration.py -v

# RAGAS evaluation
python -c "from agent_eval_pipeline.evals.ragas import run_ragas_evaluation; ..."

# Comparative eval
python -m agent_eval_pipeline.harness.comparative_runner --json

# Unit tests
python -m pytest tests/ -v --ignore=tests/test_deepeval_integration.py
```
