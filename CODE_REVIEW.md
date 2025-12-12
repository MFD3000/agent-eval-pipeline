# Code Review Findings

## High-Level Summary
Architecture is modular and thoughtfully structured, but multiple public entry points are broken (CLI commands, adapters, harness). Evaluation tooling cannot run in its current state despite the solid intent.

## Critical Issues
1. **Schema CLI crashes** (src/agent_eval_pipeline/cli/commands.py:49-57) – references nonexistent fields (`is_valid`, `errors`, `report.passed`, `report.total`).
2. **Retrieval CLI crashes** (commands.py:82-95) – prints nonexistent fields (`result.query`, `result.f1`, `report.passed`, `report.total`).
3. **RAGAS adapter broken** (evals/ragas/adapters.py:65-158) – uses wrong schema field names and treats retrieved docs as objects leading to AttributeErrors.
4. **DeepEval adapter broken** (evals/deepeval/adapters.py:125-161) – same schema/retrieved-doc issues as RAGAS.
5. **Unified runner never runs real schema validation** (harness/unified_runner.py:112-140) – imports nonexistent `run_schema_validation` and only checks if an agent returned output.

## Reliability & Correctness
- Graph cache reuses stale `VectorStore` instances because cache key ignores store identity (agent/graph.py:112-134).
- Mock embeddings emit wrong vector sizes (embeddings/openai_embeddings.py:95-101) leading to store errors.
- Analysis node hardcodes `output_tokens = 500`, hiding true regressions (agent/nodes/analyze.py:150-158).
- Retrieval eval injects randomness without seeding (evals/retrieval_eval.py:101-118) causing flaky gates.

## Performance
- Vector store seeding reruns on every cache miss (agent/graph.py:120-130).
- DSPy LM instantiated per call (agent/dspy_react_agent.py:404-451).

## Security & Privacy
- Phoenix observability captures prompts/responses by default (observability/config.py:12-35), risking PHI leakage when exporting traces. Should default to disabled/masked content.

## Testing & Observability
- CLI tests mock away formatting, so broken attributes slipped through; add integration-style assertions.
- No adapter tests for DeepEval/RAGAS—add fixtures to ensure schema alignment.
- Unified runner swallows schema validation; emit structured telemetry/errors.

## Code Quality & Maintainability
- Divergent field names cause drift across modules; import schemas instead of duplicating strings.
- CLI formatting should be centralized to prevent divergence.
- Hidden graph cache makes dependency behavior hard to reason about.

## ML-Specific Issues
- RAGAS/DeepEval adapters currently crash, so faithfulness/safety metrics never run.
- Retrieved context is passed verbatim, risking context-window overflow; consider trimming.

## Refactoring Opportunities
- Unify evaluation adapters around shared serialization helpers.
- Rework `get_agent_graph` to avoid hidden singleton caches when explicit stores are passed.
- Replace random retrieval simulator with deterministic checks against the actual store.

## Nice-to-Haves
- Document Phoenix data-capture implications in README.
- Add CLI usage examples for DSPy agent.
- Emit JSON summaries of gate metrics for CI parsing.

## Overall Score
Score: **3/10** – Reject until critical issues and evaluation path are fixed.
