# Evaluation Tooling Overview

This project layers several evaluation/observability tools, each covering a different aspect of reliability for the lab-analysis agents.

## DeepEval (Pytest-native semantic checks)
- **Purpose**: Run behavioral/NLP-focused tests inside pytest. Uses G-Eval prompts so we can encode healthcare-specific rubrics (clinical correctness, safety compliance, completeness, clarity) and gate on those metrics in CI.
- **When it runs**: `PYTHONPATH=src pytest tests/test_deepeval_integration.py` or via `run_deepeval_evaluation()` inside the harness/unified runner.
- **Value**: Catch semantic regressions early. Because it plugs into pytest, it’s ideal for PR-level gates where we want a quick pass/fail signal without spinning up the full harness.

## RAGAS (Retrieval-grounding analytics)
- **Purpose**: Exercise the RAG portions of the system with metrics tailored to grounding: faithfulness, context precision/recall, answer relevancy, factual correctness.
- **When it runs**: `PYTHONPATH=src python -m agent_eval_pipeline.evals.ragas` or through `run_ragas_eval()` in the unified runner.
- **Value**: Detect hallucinations or retrieval mismatches that schema/DeepEval/judge might miss. Because we feed actual `retrieved_docs` into the adapters, RAGAS tells us whether responses are truly grounded in the retrieved context.

## OTEL + Phoenix (Observability / tracing)
- **Purpose**: Instrument LLM calls, eval gates, and agent nodes via OpenTelemetry. Phoenix provides an LLM-friendly UI for inspecting spans, prompts, and metrics. In this repo we default to metadata-only tracing (`PHOENIX_CAPTURE_LLM_CONTENT=false`) but developers can opt in when debugging in a compliant environment.
- **When it runs**: `PHOENIX_ENABLED=true PYTHONPATH=src python -m agent_eval_pipeline.harness.runner` starts tracing; data stays local unless `PHOENIX_COLLECTOR_ENDPOINT` points elsewhere.
- **Value**: Gives us end-to-end visibility—per-case spans, gate duration, token usage—so regressions can be diagnosed quickly. Especially useful during manual reviews or perf tuning.

In short: **DeepEval** keeps CI honest about semantics, **RAGAS** validates grounding, and **OTEL/Phoenix** lets us observe the system in motion.
