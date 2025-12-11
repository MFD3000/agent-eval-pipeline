"""
Semantic Conventions for Span Attributes

Defines attribute keys following OpenTelemetry GenAI conventions
plus custom namespaces for eval and agent metrics.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

# ---------------------------------------------------------------------------
# GENAI NAMESPACE (OTel standard)
# ---------------------------------------------------------------------------

# System
GEN_AI_SYSTEM = "gen_ai.system"  # "openai", "anthropic", etc.
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"  # "gpt-4o-mini"

# Usage
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

# Request/Response (optional, controlled by PHOENIX_CAPTURE_LLM_CONTENT)
GEN_AI_PROMPT = "gen_ai.prompt"
GEN_AI_COMPLETION = "gen_ai.completion"


# ---------------------------------------------------------------------------
# EVAL NAMESPACE (custom)
# ---------------------------------------------------------------------------

# Harness level
EVAL_HARNESS_RUN_ID = "eval.harness.run_id"  # UUID for correlation
EVAL_HARNESS_AGENT_TYPE = "eval.harness.agent_type"  # "langgraph", "dspy_react"
EVAL_HARNESS_CASES_COUNT = "eval.harness.cases_count"

# Gate level
EVAL_GATE_NAME = "eval.gate.name"  # "schema_validation", "retrieval_quality", etc.
EVAL_GATE_STATUS = "eval.gate.status"  # "passed", "failed"
EVAL_GATE_THRESHOLD = "eval.gate.threshold"
EVAL_GATE_SCORE = "eval.gate.score"

# Case level
EVAL_CASE_ID = "eval.case.id"  # "thyroid-001"
EVAL_CASE_PASSED = "eval.case.passed"  # bool
EVAL_CASE_ERROR = "eval.case.error"  # error message if failed

# Judge specific
EVAL_JUDGE_DIMENSION = "eval.judge.dimension"  # "clinical_correctness", etc.
EVAL_JUDGE_SCORE = "eval.judge.score"  # 1-5
EVAL_JUDGE_REASONING = "eval.judge.reasoning"
EVAL_JUDGE_WEIGHTED_SCORE = "eval.judge.weighted_score"

# Retrieval specific
EVAL_RETRIEVAL_PRECISION = "eval.retrieval.precision"
EVAL_RETRIEVAL_RECALL = "eval.retrieval.recall"
EVAL_RETRIEVAL_F1 = "eval.retrieval.f1"
EVAL_RETRIEVAL_MISSING_DOCS = "eval.retrieval.missing_docs"

# Performance specific
EVAL_PERF_LATENCY_MS = "eval.perf.latency_ms"
EVAL_PERF_P50_LATENCY_MS = "eval.perf.p50_latency_ms"
EVAL_PERF_P95_LATENCY_MS = "eval.perf.p95_latency_ms"
EVAL_PERF_BASELINE_P95_MS = "eval.perf.baseline_p95_ms"
EVAL_PERF_REGRESSION = "eval.perf.regression"  # bool


# ---------------------------------------------------------------------------
# AGENT NAMESPACE (custom)
# ---------------------------------------------------------------------------

AGENT_TYPE = "agent.type"  # "langgraph", "dspy_react"
AGENT_NODE_NAME = "agent.node.name"  # "retrieve", "analyze", "safety"

# LangGraph specific
AGENT_RETRIEVED_DOC_COUNT = "agent.retrieved_doc_count"
AGENT_RETRIEVED_DOC_IDS = "agent.retrieved_doc_ids"
AGENT_RETRIEVAL_LATENCY_MS = "agent.retrieval_latency_ms"

# DSPy ReAct specific
AGENT_TOOLS_USED = "agent.tools_used"  # list of tool names
AGENT_REASONING_STEPS = "agent.reasoning_steps"  # count


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def eval_gate_attributes(
    gate_name: str,
    status: str,
    score: float | None = None,
    threshold: float | None = None,
) -> dict:
    """Create attributes dict for an eval gate span."""
    attrs = {
        EVAL_GATE_NAME: gate_name,
        EVAL_GATE_STATUS: status,
    }
    if score is not None:
        attrs[EVAL_GATE_SCORE] = score
    if threshold is not None:
        attrs[EVAL_GATE_THRESHOLD] = threshold
    return attrs


def eval_case_attributes(
    case_id: str,
    passed: bool,
    error: str | None = None,
) -> dict:
    """Create attributes dict for an eval case span."""
    attrs = {
        EVAL_CASE_ID: case_id,
        EVAL_CASE_PASSED: passed,
    }
    if error:
        attrs[EVAL_CASE_ERROR] = error
    return attrs


def agent_run_attributes(
    agent_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> dict:
    """Create attributes dict for an agent run span."""
    return {
        AGENT_TYPE: agent_type,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        GEN_AI_USAGE_TOTAL_TOKENS: input_tokens + output_tokens,
        EVAL_PERF_LATENCY_MS: latency_ms,
    }
