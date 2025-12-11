# Interview Preparation: Function Health - Senior AI Engineer

This document contains 30 interview questions with detailed answers based on the agent-eval-pipeline project and the Function Health job requirements.

---

## Section 1: Agentic Systems & Orchestration

### Q1: Describe your experience building stateful, graph-based agent workflows. What orchestration frameworks have you used?

**Answer:**

I've built agentic systems using both **LangGraph** and **DSPy**, which represent two different paradigms:

**LangGraph (Imperative/State Machine):**
In my lab insights agent project, I implemented a LangGraph workflow with explicit state management:

```python
# From agent/graph.py
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_context", create_retrieve_node(store))
workflow.add_node("analyze_labs", create_analyze_node(model))
workflow.add_node("apply_safety", apply_safety)
workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "analyze_labs")
workflow.add_edge("analyze_labs", "apply_safety")
```

The key insight is that LangGraph gives you **explicit control over flow** - you see exactly how state transforms through each node. This is valuable when you need:
- Checkpointing for long-running workflows
- Human-in-the-loop approval steps
- Complex branching logic

**DSPy (Declarative):**
I also implemented the same agent using DSPy's declarative approach:

```python
# From agent/dspy_agent.py
class LabInsightsModule(dspy.Module):
    def __init__(self):
        self.extract_markers = dspy.Predict(ExtractMarkers)
        self.analyze = dspy.ChainOfThought(AnalyzeLabs)  # Adds reasoning
        self.safety_check = dspy.Predict(SafetyCheck)
```

DSPy's advantage is **automatic prompt optimization** - I define signatures (what I want) and let optimizers find effective prompts.

**When to use each:**
- LangGraph: Complex workflows, explicit state, human-in-loop
- DSPy: Prompt optimization, structured output, rapid experimentation

---

### Q2: How do you implement tool use and function calling in your agents?

**Answer:**

I've implemented tool use in two ways:

**1. LangGraph with explicit tool nodes:**
Each tool becomes a node in the graph. The retrieve node, for example, calls the vector store:

```python
def create_retrieve_node(store: VectorStore):
    def retrieve_context(state: AgentState) -> dict:
        markers = [lab["marker"] for lab in state["labs"]]
        docs = store.search_by_markers(markers, limit=5)
        return {"retrieved_docs": docs, "retrieval_latency_ms": latency}
    return retrieve_context
```

**2. DSPy ReAct for dynamic tool selection:**
I implemented a ReAct agent that decides which tools to call based on the query:

```python
# From agent/dspy_react_agent.py
tools = [
    lookup_reference_range,      # Look up standard ranges
    check_medication_interaction, # Check drug/lab interactions
    search_medical_context,       # Search knowledge base
]
self.react = dspy.ReAct(signature=LabAnalysisSignature, tools=tools, max_iters=5)
```

The ReAct agent reasons explicitly before each action: "The patient mentions biotin supplements and has low TSH. I should check for biotin/TSH interaction." This creates an auditable reasoning trace.

**Key considerations for tool calling:**
- **Timeouts and circuit breakers** - Tools can fail; wrap with retry logic
- **Structured I/O** - Use Pydantic models for tool inputs/outputs
- **Sandboxing** - Tools shouldn't have unbounded scope

---

### Q3: How do you handle state management in multi-turn agent conversations?

**Answer:**

In LangGraph, state is managed through a **TypedDict** that flows through the graph:

```python
# From agent/state.py
class AgentState(TypedDict):
    query: str
    labs: list[dict]
    history: list[dict]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_docs: list[dict]
    raw_analysis: dict | None
    final_output: LabInsightsSummary | None
    # Metrics
    retrieval_latency_ms: float
    input_tokens: int
    output_tokens: int
```

**Key patterns:**

1. **Immutable updates** - Nodes return dicts that get merged, not mutated
2. **Message accumulation** - The `add_messages` annotation appends rather than replaces
3. **Checkpointing** - LangGraph can serialize state for pause/resume

For multi-turn conversations in production, I'd add:
- **Session storage** (Redis) for active conversations
- **Long-term memory** (PostgreSQL) for user context
- **Token budget management** - Summarize old messages when approaching limits

---

### Q4: Explain the difference between DSPy signatures and traditional prompt engineering.

**Answer:**

Traditional prompt engineering is **imperative** - you write exact instructions:

```python
prompt = """You are a health assistant. Given the lab values below, provide insights.
Be sure to include disclaimers. Don't diagnose conditions.
Labs: {labs}
Question: {query}
..."""
```

DSPy signatures are **declarative** - you specify structure, not instructions:

```python
class AnalyzeLabs(dspy.Signature):
    """Analyze lab results and provide health insights."""
    query: str = dspy.InputField(desc="User's question")
    labs: str = dspy.InputField(desc="Lab values")
    summary: str = dspy.OutputField(desc="2-3 sentence summary")
    insights_json: str = dspy.OutputField(desc="JSON array of insights")
```

**The key difference is optimization:**

With traditional prompts, I manually iterate:
1. Write prompt → Test → See failures → Rewrite → Repeat

With DSPy, I define a metric and let optimizers find good prompts:

```python
optimizer = dspy.BootstrapFewShot(metric=clinical_accuracy, max_bootstrapped_demos=3)
optimized = optimizer.compile(agent, trainset=golden_cases)
```

**When DSPy shines:**
- Structured output requirements
- You have labeled examples
- Prompt needs to work across diverse inputs

**When traditional prompts are better:**
- Simple, one-off tasks
- You need exact control over wording
- No training data available

---

## Section 2: Evaluation Pipelines & Quality Gates

### Q5: Walk me through how you would set up an evaluation pipeline for an AI agent.

**Answer:**

I built a four-gate evaluation pipeline that runs in CI:

**Gate 1: Schema Validation (Fast, Deterministic)**
```python
# Validates agent output matches Pydantic schema
result = run_schema_eval()  # < 100ms, no LLM calls
```
Catches: Missing fields, wrong types, invalid enum values

**Gate 2: Retrieval Quality (No LLM)**
```python
# Measures precision/recall/F1 against expected documents
metrics = calculate_retrieval_metrics(retrieved, expected)
```
Catches: RAG regressions, embedding drift

**Gate 3: LLM-as-Judge (Expensive, Semantic)**
```python
# GPT-4o evaluates against weighted rubric
# Clinical correctness (40%), Safety (30%), Completeness (20%), Clarity (10%)
judge_output = run_judge(case, agent_output)
weighted_score = calculate_weighted_score(judge_output)
```
Catches: Clinical errors, safety violations, incomplete answers

**Gate 4: Performance Regression**
```python
# Compare p95 latency and token usage against baseline
if latency_increase > 15% or token_increase > 20%:
    fail("Regression detected")
```
Catches: Prompt bloat, model routing errors, retrieval slowdowns

**Execution order matters:**
- Cheap/fast checks first (fail fast)
- Expensive LLM-as-judge only if structure passes
- Results feed into CI with structured JSON output

---

### Q6: How do you design an LLM-as-judge system? What are the pitfalls?

**Answer:**

My judge system uses a **multi-dimensional rubric** with weighted scoring:

```python
WEIGHTS = {
    "clinical_correctness": 0.40,  # Most important in healthcare
    "safety_compliance": 0.30,     # Never diagnose, always disclaim
    "completeness": 0.20,          # Cover all markers
    "clarity": 0.10,               # Accessible language
}
```

**Design decisions:**

1. **Separate evaluators per dimension** - In my DSPy implementation, each dimension has its own module:
```python
self.clinical = dspy.ChainOfThought(EvaluateClinicalCorrectness)
self.safety = dspy.ChainOfThought(EvaluateSafetyCompliance)
```
This allows independent optimization of each aspect.

2. **Critical issues as automatic failures:**
```python
CRITICAL_ISSUES = [
    "Diagnosing a condition",
    "Recommending specific medications",
    "Dismissing concerning values",
]
```

3. **Structured output** - The judge returns Pydantic models, not free text:
```python
class JudgeOutput(BaseModel):
    clinical_correctness: DimensionScore
    safety_compliance: DimensionScore
    critical_issues: list[str]
```

**Pitfalls to avoid:**

- **Self-evaluation bias** - Don't use the same model to generate and judge
- **Prompt sensitivity** - Judge prompts need careful calibration
- **Score drift** - Monitor judge consistency over time
- **Gaming** - Agents can learn to produce "judge-friendly" but unhelpful outputs

**Mitigation:** I implemented meta-optimization for the judge itself:
```python
# Optimize judge prompts to correlate with human ratings
metric = create_judge_metric()  # Measures agreement with expert labels
optimized_judge = optimizer.compile(judge, calibration_set)
```

---

### Q7: How do you create and maintain golden test sets for AI evaluation?

**Answer:**

My golden sets are structured with both inputs and expected outputs:

```python
# From golden_sets/thyroid_cases.py
@dataclass
class GoldenCase:
    id: str
    description: str
    query: str
    labs: list[LabValue]
    history: list[LabValue]
    symptoms: list[str]
    expected_semantic_points: list[str]  # What should be covered
```

**Key principles:**

1. **Representative coverage** - Cases cover:
   - Normal results (verify no false alarms)
   - Clearly abnormal (verify detection)
   - Borderline (verify appropriate hedging)
   - Trending patterns (verify trend analysis)
   - Multiple abnormalities (verify completeness)

2. **Expected semantic points, not exact text:**
```python
expected_semantic_points = [
    "TSH is elevated above reference range",
    "This pattern may suggest hypothyroidism",
    "Recommend discussing with healthcare provider",
]
```
The judge checks if these concepts appear, not exact wording.

3. **Version control** - Golden sets are in git, changes require review

4. **Continuous expansion** - When we find failures in production, add as test cases

**Maintenance:**
- Review quarterly with domain experts
- Add edge cases discovered in production
- Remove outdated cases when product changes
- Track coverage metrics

---

### Q8: How do you integrate AI evaluations into CI/CD?

**Answer:**

My harness runs in GitHub Actions with structured output:

```yaml
# .github/workflows/eval.yml
- name: Run Eval Pipeline
  run: |
    PYTHONPATH=src python -m agent_eval_pipeline.harness.runner --json
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

**Gate structure:**

```python
# From harness/runner.py
def run_all_evals(fail_fast=True, skip_expensive=False):
    # Fast checks first
    schema_result = run_gate("Schema Validation", run_schema_eval)
    if fail_fast and not schema_result.passed:
        return early_failure()

    # Then retrieval (no LLM)
    retrieval_result = run_gate("Retrieval Quality", run_retrieval_eval)

    # Expensive LLM-as-judge only if earlier gates pass
    if not skip_expensive:
        judge_result = run_gate("LLM-as-Judge", run_judge_eval)

    # Performance regression
    perf_result = run_gate("Performance", run_perf_eval)
```

**PR workflow:**
1. **On PR open:** Run schema + retrieval (fast, cheap)
2. **On approval:** Run full suite including judge
3. **On merge to main:** Update performance baseline

**Cost control:**
- Cache LLM responses for identical inputs
- Use smaller model for PR checks, full model for merge
- Skip expensive evals for docs-only changes

---

## Section 3: Retrieval & RAG

### Q9: How do you implement and evaluate retrieval for RAG systems?

**Answer:**

**Implementation:**

I built a vector store abstraction with dependency injection:

```python
# Protocol defines the interface
class VectorStore(Protocol):
    def search(self, query: str, limit: int) -> list[DocumentResult]: ...
    def search_by_markers(self, markers: list[str], limit: int) -> list[DocumentResult]: ...

# Production implementation
class PgVectorStore:
    def __init__(self, config: VectorStoreConfig, embeddings: EmbeddingProvider):
        self._embeddings = embeddings  # Injected, not created

# Test implementation
class InMemoryVectorStore:
    def __init__(self, embeddings: EmbeddingProvider):
        self._docs = []
```

**Evaluation:**

I measure precision, recall, and F1 against expected documents:

```python
def calculate_retrieval_metrics(retrieved: list[str], expected: list[str]):
    retrieved_set = set(retrieved)
    expected_set = set(expected)

    true_positives = len(retrieved_set & expected_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(expected_set) if expected_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        missing_docs=list(expected_set - retrieved_set),
        extra_docs=list(retrieved_set - expected_set),
    )
```

**Key insight:** I track `missing_docs` and `extra_docs` explicitly - this tells me if the problem is recall (missing relevant docs) or precision (too much noise).

---

### Q10: What strategies would you use to improve retrieval quality?

**Answer:**

Several strategies, ordered by implementation complexity:

**1. Hybrid search (keyword + semantic):**
```sql
-- Combine BM25 and vector similarity
SELECT *,
    (0.7 * vector_similarity + 0.3 * bm25_score) as combined_score
FROM documents
ORDER BY combined_score DESC
```

**2. Query expansion:**
```python
# In my ReAct agent
markers = extract_markers(query)  # "TSH", "T4"
context = search_medical_context("thyroid")  # Expand to related concepts
```

**3. Reranking:**
```python
# First pass: fast retrieval (top 20)
candidates = vector_store.search(query, limit=20)
# Second pass: cross-encoder reranking (top 5)
reranked = reranker.rerank(query, candidates)[:5]
```

**4. Marker-specific retrieval:**
In my implementation, I search by lab markers, not just query text:
```python
def search_by_markers(self, markers: list[str], limit: int):
    # Filter documents that discuss these specific markers
    # More precise than general semantic search
```

**5. Chunk optimization:**
- Semantic chunking (by topic, not character count)
- Overlap between chunks for context continuity
- Metadata filtering (date, source, marker type)

---

### Q11: How do you handle the embedding/retrieval layer for testability?

**Answer:**

I follow the **gold standard pattern** from my embeddings module:

```python
# 1. Protocol (interface)
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...

# 2. Production implementation
class OpenAIEmbeddings:
    def __init__(self, model: str = "text-embedding-3-small"):
        self._client = OpenAI()

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(input=text, model=self.model)
        return np.array(response.data[0].embedding)

# 3. Test double
class MockEmbeddings:
    def embed(self, text: str) -> np.ndarray:
        # Deterministic hash-based embedding for testing
        h = hashlib.sha256(text.encode()).digest()
        return np.frombuffer(h * 48, dtype=np.float32)[:1536]

# 4. Factory function
def get_embedding_provider(use_mock: bool = False) -> EmbeddingProvider:
    return MockEmbeddings() if use_mock else OpenAIEmbeddings()
```

**Testing the full RAG flow without API calls:**

```python
def test_retrieval_flow():
    # Inject mock embeddings
    embeddings = MockEmbeddings()
    store = InMemoryVectorStore(embeddings)
    store.connect()
    seed_vector_store(store)  # Add test documents

    # Test retrieval
    results = store.search_by_markers(["TSH"], limit=3)
    assert len(results) == 3
    assert "thyroid" in results[0].content.lower()
```

This test runs in ~10ms with no external dependencies.

---

## Section 4: Production Readiness & Observability

### Q12: How do you ensure production readiness for AI systems?

**Answer:**

I focus on several areas:

**1. Structured output validation:**
```python
# Every agent output must match Pydantic schema
class LabInsightsSummary(BaseModel):
    summary: str
    key_insights: list[MarkerInsight]
    safety_notes: list[SafetyNote]  # Required!
```

**2. Safety guardrails:**
```python
# From agent/nodes/safety.py - pure function, always runs
def apply_safety(state: AgentState) -> dict:
    output = state["raw_analysis"]

    # Ensure non-diagnostic disclaimer exists
    if not any(n.type == "non_diagnostic" for n in output.safety_notes):
        output.safety_notes.append(SafetyNote(
            message="This is for educational purposes only...",
            type="non_diagnostic"
        ))

    # Ensure doctor recommendation exists
    if not output.recommended_topics_for_doctor:
        output.recommended_topics_for_doctor = [
            "Review these results with your healthcare provider"
        ]

    return {"final_output": output}
```

**3. Error handling with typed errors:**
```python
@dataclass
class AgentError:
    error_type: str  # "ValidationError", "TimeoutError", "RateLimitError"
    error_message: str

def run_agent(case) -> AgentResult | AgentError:
    try:
        ...
    except ValidationError as e:
        return AgentError("ValidationError", str(e))
    except TimeoutError:
        return AgentError("TimeoutError", "Agent timed out after 30s")
```

**4. Latency tracking:**
```python
# Built into state
class AgentState(TypedDict):
    retrieval_latency_ms: float
    analysis_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
```

**5. Regression testing:**
```python
# Performance baseline comparison
if p95_latency > baseline.p95_latency * 1.15:
    fail("Latency regression: 15% increase detected")
```

---

### Q13: What observability would you implement for an AI agent system?

**Answer:**

**Metrics (Prometheus/Grafana):**
```python
# Counters
agent_requests_total{model, status}
agent_errors_total{error_type}

# Histograms
agent_latency_seconds{quantile="0.5|0.95|0.99"}
agent_tokens_used{type="input|output"}

# Gauges
agent_cost_dollars_per_hour
```

**Traces (OpenTelemetry):**
```python
with tracer.start_span("agent.run") as span:
    span.set_attribute("case_id", case.id)
    span.set_attribute("model", model_name)

    with tracer.start_span("retrieval"):
        docs = retrieve(query)
        span.set_attribute("docs_retrieved", len(docs))

    with tracer.start_span("analysis"):
        result = analyze(docs)
        span.set_attribute("tokens.input", result.input_tokens)
```

**Structured logs:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "event": "agent.completed",
  "case_id": "thyroid-001",
  "latency_ms": 1250,
  "tokens": {"input": 1500, "output": 450},
  "model": "gpt-4o-mini",
  "judge_score": 4.3,
  "retrieval_f1": 0.85
}
```

**Dashboards I'd build:**
1. **Real-time health:** Request rate, error rate, p95 latency
2. **Cost tracking:** Tokens/hour, $/day by model
3. **Quality trends:** Judge scores over time, regression alerts
4. **Retrieval quality:** F1 scores, cache hit rates

---

### Q14: How do you handle rate limiting and circuit breakers for LLM APIs?

**Answer:**

**Rate limiting:**
```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=60, period=60)  # 60 RPM
def call_openai(messages):
    return client.chat.completions.create(...)
```

**Circuit breaker pattern:**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failures = 0
        self.last_failure = None
        self.state = "CLOSED"  # CLOSED -> OPEN -> HALF_OPEN

    def call(self, func, *args):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError("Circuit breaker is open")

        try:
            result = func(*args)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise
```

**Fallback strategies:**
```python
def get_analysis(query, labs):
    try:
        return primary_model.analyze(query, labs)
    except (RateLimitError, CircuitOpenError):
        # Fallback to cheaper/faster model
        return fallback_model.analyze(query, labs)
    except Exception:
        # Return cached similar response or graceful degradation
        return get_cached_similar(query) or default_response()
```

---

## Section 5: Cost & Latency Optimization

### Q15: How do you optimize cost and latency for LLM-based systems?

**Answer:**

My performance eval tracks these metrics and fails on regressions:

**1. Model routing:**
```python
def select_model(complexity_score: float) -> str:
    if complexity_score < 0.3:
        return "gpt-4o-mini"  # $0.15/1M input
    elif complexity_score < 0.7:
        return "gpt-4o"       # $2.50/1M input
    else:
        return "gpt-4-turbo"  # Complex cases only
```

**2. Prompt optimization:**
```python
# Track token usage in eval
def estimate_cost(input_tokens, output_tokens, model):
    pricing = MODEL_PRICING[model]
    return (input_tokens / 1M) * pricing["input"] + \
           (output_tokens / 1M) * pricing["output"]

# Fail if tokens increase by >20%
if avg_tokens > baseline.avg_tokens * 1.20:
    fail("Token regression detected")
```

**3. Caching strategies:**
```python
# Semantic cache for similar queries
cache_key = hash(normalize(query) + str(sorted(markers)))
if cache_key in semantic_cache:
    return semantic_cache[cache_key]
```

**4. KV cache optimization:**
- Use consistent system prompts (cacheable prefix)
- Batch similar requests
- Prompt prefix sharing across requests

**5. Retrieval optimization:**
```python
# Reduce context by filtering retrieved docs
def smart_context_selection(docs, max_tokens=2000):
    selected = []
    token_count = 0
    for doc in sorted(docs, key=lambda d: d.score, reverse=True):
        doc_tokens = count_tokens(doc.content)
        if token_count + doc_tokens > max_tokens:
            break
        selected.append(doc)
        token_count += doc_tokens
    return selected
```

---

### Q16: How would you implement model routing based on query complexity?

**Answer:**

**Complexity classification:**
```python
class QueryComplexityClassifier(dspy.Signature):
    """Classify query complexity for model routing."""
    query: str = dspy.InputField()
    lab_count: int = dspy.InputField()
    has_history: bool = dspy.InputField()
    has_symptoms: bool = dspy.InputField()

    complexity: Literal["simple", "moderate", "complex"] = dspy.OutputField()
    reasoning: str = dspy.OutputField()

def route_to_model(query, labs, history, symptoms):
    # Fast heuristics first
    if len(labs) == 1 and not history and not symptoms:
        return "gpt-4o-mini"  # Simple single-marker query

    if len(labs) > 5 or any(is_critical(lab) for lab in labs):
        return "gpt-4o"  # Complex or critical

    # Use classifier for edge cases
    classifier = dspy.Predict(QueryComplexityClassifier)
    result = classifier(
        query=query,
        lab_count=len(labs),
        has_history=bool(history),
        has_symptoms=bool(symptoms)
    )

    model_map = {
        "simple": "gpt-4o-mini",
        "moderate": "gpt-4o-mini",
        "complex": "gpt-4o"
    }
    return model_map[result.complexity]
```

**Cost tracking:**
```python
# From evals/perf/pricing.py
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# Track routing decisions
routing_metrics.labels(
    query_complexity=complexity,
    model_selected=model,
    actual_cost=cost
).inc()
```

---

## Section 6: System Design

### Q17: Design a health insights agent that handles 10,000 requests per minute.

**Answer:**

**Architecture:**

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │ API Pod │          │ API Pod │          │ API Pod │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │  Redis  │          │pgvector │          │  LLM    │
   │ (cache) │          │  (RAG)  │          │  APIs   │
   └─────────┘          └─────────┘          └─────────┘
```

**Key components:**

1. **Stateless API pods** (K8s deployment, HPA on CPU/latency)
2. **Redis for:**
   - Semantic cache (similar query results)
   - Rate limiting state
   - Session data for multi-turn
3. **pgvector for RAG:**
   - Read replicas for query scaling
   - Connection pooling (PgBouncer)
4. **LLM API calls:**
   - Multiple provider fallback
   - Request batching where possible
   - Circuit breakers per provider

**Scaling math:**
- 10,000 RPM = 167 RPS
- Assume 1.5s average latency
- Need ~250 concurrent connections
- With 10 pods, 25 concurrent per pod

**Optimizations:**
```python
# Semantic caching
cache_key = semantic_hash(query, markers)
if cached := redis.get(cache_key):
    return cached  # ~5ms

# Otherwise compute
result = agent.run(query, labs)
redis.setex(cache_key, ttl=3600, value=result)  # Cache 1hr
```

---

### Q18: How would you design the data model for a lab insights system?

**Answer:**

**Core entities:**

```sql
-- Users and sessions
CREATE TABLE users (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    started_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP
);

-- Lab data (normalized)
CREATE TABLE lab_results (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    marker VARCHAR(50) NOT NULL,
    value DECIMAL NOT NULL,
    unit VARCHAR(20) NOT NULL,
    ref_low DECIMAL,
    ref_high DECIMAL,
    collected_at TIMESTAMP NOT NULL,
    source VARCHAR(100),  -- "Quest", "LabCorp", etc.
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_lab_results_user_marker ON lab_results(user_id, marker, collected_at DESC);

-- Agent interactions
CREATE TABLE agent_interactions (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    query TEXT NOT NULL,
    lab_result_ids UUID[],  -- Labs included in this query
    response JSONB NOT NULL,  -- LabInsightsSummary
    model VARCHAR(50),
    latency_ms INT,
    input_tokens INT,
    output_tokens INT,
    judge_score DECIMAL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector store for RAG
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    title VARCHAR(200),
    content TEXT NOT NULL,
    markers VARCHAR(50)[],  -- Related markers
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_documents_embedding ON documents
    USING ivfflat (embedding vector_cosine_ops);
```

**Key design decisions:**
- **Denormalized response storage** (JSONB) for fast retrieval
- **Array type for markers** enables efficient filtering
- **IVFFlat index** for vector similarity (good balance of speed/accuracy)
- **Separate interaction table** for analytics and debugging

---

## Section 7: Safety & Healthcare AI

### Q19: How do you ensure safety in a healthcare AI application?

**Answer:**

Safety is built into multiple layers:

**1. Schema-level constraints:**
```python
class LabInsightsSummary(BaseModel):
    # Required safety field - can't skip
    safety_notes: list[SafetyNote]

class SafetyNote(BaseModel):
    message: str
    type: Literal["non_diagnostic", "seek_care", "disclaimer"]
```

**2. Dedicated safety node (always runs):**
```python
# From agent/nodes/safety.py
def apply_safety(state: AgentState) -> dict:
    # Ensure non-diagnostic disclaimer
    if not has_disclaimer(output):
        add_disclaimer(output)

    # Ensure doctor recommendation
    if not output.recommended_topics_for_doctor:
        output.recommended_topics_for_doctor = [
            "Review these results with your healthcare provider"
        ]

    return {"final_output": output}
```

**3. LLM-as-judge safety scoring:**
```python
# 30% weight on safety compliance
WEIGHTS = {
    "safety_compliance": 0.30,  # Second highest
    ...
}

# Critical issues = automatic failure
CRITICAL_ISSUES = [
    "Diagnosing a condition",
    "Recommending specific medications",
    "Dismissing concerning values",
]
```

**4. DSPy assertions:**
```python
dspy.Assert(
    safety.is_safe,
    "Safety check failed: content must be revised"
)
```

**5. Content filtering:**
- Block diagnostic language ("you have X")
- Block medication recommendations
- Flag concerning values for escalation

---

### Q20: What are the unique challenges of AI in healthcare vs other domains?

**Answer:**

**1. Higher stakes for errors:**
- Wrong financial advice: lose money
- Wrong health advice: potential harm
- Our safety bar must be much higher

**2. Regulatory requirements:**
- HIPAA for data handling
- Can't provide medical advice without licensing
- Must maintain clear disclaimers
- Audit trails required

**3. Domain expertise validation:**
```python
# Our judge rubric weights clinical correctness at 40%
# In other domains, might weight differently
WEIGHTS = {
    "clinical_correctness": 0.40,  # Healthcare: highest
    "safety_compliance": 0.30,
    ...
}
```

**4. User trust calibration:**
- Users may over-trust AI health advice
- Must actively encourage professional consultation
- Can't be overly confident even when correct

**5. Liability concerns:**
- Every output must be educational, not diagnostic
- Clear scope limitations
- Cannot recommend treatments

**How I address these:**
```python
# Forced disclaimers in schema
class LabInsightsSummary(BaseModel):
    safety_notes: list[SafetyNote]  # Required
    recommended_topics_for_doctor: list[str]  # Required
```

---

## Section 8: Behavioral & Leadership

### Q21: Tell me about a time you shipped an AI feature that moved a core product KPI.

**Answer:**

"In this project, I built a lab insights agent with a comprehensive evaluation pipeline. Here's how I'd frame this for production impact:

**The problem:** Users were receiving lab results but didn't understand what they meant. They'd either ignore concerning values or unnecessarily worry about normal results.

**My approach:**
1. Built an agent that explains lab results in accessible language
2. Implemented safety guardrails to avoid diagnostic language
3. Created an eval pipeline to ensure quality:
   - Schema validation (structural correctness)
   - LLM-as-judge (semantic quality)
   - Performance regression (latency/cost)

**Measurable quality targets:**
```python
# From my eval pipeline
threshold = 4.2  # Minimum judge score out of 5
latency_p95 < 2000  # Sub-2-second response
safety_violations = 0  # Zero tolerance
```

**Impact I'd expect:**
- Increased user engagement with lab results
- Reduced support tickets asking "what does this mean?"
- Higher user confidence in the platform
- Zero safety incidents from AI advice"

---

### Q22: How do you approach technical decisions when there are multiple valid options?

**Answer:**

"I faced this exact situation choosing between LangGraph and DSPy for my agent implementation. Here's my framework:

**1. Clarify the decision criteria:**
- What matters most? (testability, performance, maintainability)
- What are the constraints? (team expertise, timeline, scale)

**2. Prototype both approaches:**
I built the same agent in both frameworks:
```python
# LangGraph - explicit state machine
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("analyze", analyze_node)

# DSPy - declarative signatures
self.analyze = dspy.ChainOfThought(AnalyzeLabs)
```

**3. Evaluate against criteria:**

| Criteria | LangGraph | DSPy |
|----------|-----------|------|
| Testability | Node isolation | Signature introspection |
| Prompt optimization | Manual | Automatic |
| State management | Explicit | Implicit |
| Debugging | State inspection | Reasoning traces |

**4. Make a reversible decision:**
I implemented both, with a shared interface:
```python
def run_agent(case, use_langgraph=True):
    if use_langgraph:
        return run_langgraph_agent(case)
    else:
        return run_dspy_agent(case)
```

**5. Document the decision:**
The code includes comments explaining when to use each approach.

This approach lets us A/B test in production and change our minds based on data."

---

### Q23: How do you mentor engineers on AI systems?

**Answer:**

"I focus on three areas:

**1. Code as documentation:**
My code includes extensive 'interview talking points':
```python
# INTERVIEW TALKING POINT:
# "The safety node is pure - no external dependencies, no side effects.
# Given the same raw_analysis, it always produces the same output.
# I can test this in microseconds."
```

These explain not just what but why.

**2. Test-driven understanding:**
I write tests that demonstrate patterns:
```python
class TestModuleStructure:
    def test_lab_insights_module_has_predictors(self):
        '''Verify LabInsightsModule has expected predictors.'''
        module = LabInsightsModule()
        assert hasattr(module, 'analyze')
        assert isinstance(module.analyze, dspy.ChainOfThought)
```

Tests serve as executable documentation.

**3. Design patterns they can reuse:**
I established the 'gold standard' pattern:
```
Protocol → Production Implementation → Test Double → Factory Function
```

Every elevated module follows this pattern, making it easy for others to add new modules consistently.

**4. Architecture decisions are recorded:**
The `ELEVATION_PLAN.md` and `CODE_ELEVATION_ASSESSMENT.md` document the reasoning behind architectural choices."

---

### Q24: Describe a situation where you had to push back on a product request.

**Answer:**

"In healthcare AI, I'd push back on anything that crosses the diagnostic line.

**Scenario:** Product wants the agent to say 'Your TSH indicates hypothyroidism' instead of 'Your TSH is elevated above the reference range.'

**My pushback:**

1. **Explain the risk:**
   - Diagnosis requires clinical judgment, not just lab values
   - We could be wrong and cause harm
   - Regulatory/legal exposure

2. **Show the alternative:**
   ```python
   # Instead of: 'You have hypothyroidism'
   # We say:
   summary = 'Your TSH of 6.5 is above the typical range (0.4-4.0). '\
             'This pattern is sometimes associated with thyroid conditions. '\
             'We recommend discussing these results with your healthcare provider.'
   ```

3. **Demonstrate with evals:**
   ```python
   # Our judge has 'safety_compliance' at 30% weight
   # Diagnostic language triggers critical failure
   CRITICAL_ISSUES = ['Diagnosing a condition']
   ```

4. **Propose compromise:**
   - More educational content about what TSH means
   - Better 'questions to ask your doctor' section
   - Clearer visualization of trends over time

**Outcome:** We can be helpful without being diagnostic. The safety guardrails I built enforce this automatically."

---

### Q25: How do you balance moving fast with maintaining quality?

**Answer:**

"The eval pipeline is my answer to this tension.

**How it enables speed:**
1. **Automated quality gates** - I don't manually review every change
2. **Fast feedback** - Schema eval runs in <100ms
3. **Confidence to iterate** - If tests pass, ship it

**How it maintains quality:**
1. **No skipping gates** - CI blocks merge on failure
2. **Regression detection** - Performance baseline comparison
3. **Semantic validation** - LLM-as-judge catches subtle issues

**Practical example:**
```bash
# Fast iteration cycle
git checkout -b feature/improve-prompt
# Make changes
pytest tests/  # Fast local validation
git push  # CI runs full eval
# If green, merge with confidence
```

**The key insight:**
> 'Move fast' and 'maintain quality' aren't opposites when you have good automated checks. The eval pipeline lets me ship multiple times per day while catching regressions automatically.

**What I'd add at scale:**
- Shadow traffic testing (run new version on real queries, compare)
- Canary releases (1% traffic first)
- Feature flags (instant rollback)
- A/B testing (measure actual impact)"

---

## Section 9: Technical Deep Dives

### Q26: Explain how you would implement semantic caching for an LLM system.

**Answer:**

```python
import hashlib
import numpy as np
from redis import Redis

class SemanticCache:
    def __init__(self, embeddings: EmbeddingProvider, redis: Redis, threshold: float = 0.92):
        self._embeddings = embeddings
        self._redis = redis
        self._threshold = threshold

    def _get_embedding_key(self, query: str, context: str) -> str:
        # Normalize and hash for exact match cache key
        normalized = query.lower().strip() + "|" + context
        return f"cache:exact:{hashlib.sha256(normalized.encode()).hexdigest()}"

    def get(self, query: str, context: str) -> dict | None:
        # Try exact match first (fast)
        exact_key = self._get_embedding_key(query, context)
        if cached := self._redis.get(exact_key):
            return json.loads(cached)

        # Try semantic match (slower but catches paraphrases)
        query_embedding = self._embeddings.embed(query)

        # Search cached embeddings
        for key in self._redis.scan_iter("cache:semantic:*"):
            cached_data = json.loads(self._redis.get(key))
            cached_embedding = np.array(cached_data["embedding"])

            similarity = np.dot(query_embedding, cached_embedding) / \
                        (np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding))

            if similarity > self._threshold:
                return cached_data["response"]

        return None

    def set(self, query: str, context: str, response: dict, ttl: int = 3600):
        # Store exact match
        exact_key = self._get_embedding_key(query, context)
        self._redis.setex(exact_key, ttl, json.dumps(response))

        # Store semantic match
        embedding = self._embeddings.embed(query)
        semantic_key = f"cache:semantic:{exact_key}"
        self._redis.setex(semantic_key, ttl, json.dumps({
            "embedding": embedding.tolist(),
            "response": response
        }))
```

**Optimizations:**
- Use approximate nearest neighbor (FAISS/Annoy) for large cache
- Shard by query type/marker
- Different TTLs for different query types
- Track cache hit rates for tuning threshold

---

### Q27: How would you implement A/B testing for prompt changes?

**Answer:**

```python
import hashlib
from dataclasses import dataclass

@dataclass
class Experiment:
    name: str
    control_prompt: str
    treatment_prompt: str
    traffic_percent: float  # 0.0 to 1.0

class PromptExperiment:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.metrics = defaultdict(list)

    def get_variant(self, user_id: str) -> str:
        # Deterministic assignment based on user_id
        hash_val = int(hashlib.md5(
            f"{self.experiment.name}:{user_id}".encode()
        ).hexdigest(), 16)

        if (hash_val % 100) / 100 < self.experiment.traffic_percent:
            return "treatment"
        return "control"

    def get_prompt(self, user_id: str) -> str:
        variant = self.get_variant(user_id)
        if variant == "treatment":
            return self.experiment.treatment_prompt
        return self.experiment.control_prompt

    def record_metric(self, user_id: str, metric_name: str, value: float):
        variant = self.get_variant(user_id)
        self.metrics[f"{variant}:{metric_name}"].append(value)

    def get_results(self) -> dict:
        return {
            "control": {
                "judge_score": np.mean(self.metrics["control:judge_score"]),
                "latency_p50": np.percentile(self.metrics["control:latency"], 50),
                "n": len(self.metrics["control:judge_score"]),
            },
            "treatment": {
                "judge_score": np.mean(self.metrics["treatment:judge_score"]),
                "latency_p50": np.percentile(self.metrics["treatment:latency"], 50),
                "n": len(self.metrics["treatment:judge_score"]),
            },
            "significance": calculate_significance(
                self.metrics["control:judge_score"],
                self.metrics["treatment:judge_score"]
            )
        }
```

**Usage in agent:**
```python
experiment = PromptExperiment(Experiment(
    name="cot_vs_direct",
    control_prompt="Analyze these labs...",
    treatment_prompt="Think step by step. Analyze these labs...",
    traffic_percent=0.1,  # 10% get treatment
))

prompt = experiment.get_prompt(user_id)
result = run_agent(prompt, labs)
experiment.record_metric(user_id, "judge_score", judge_result.score)
```

---

### Q28: How do you handle versioning of prompts and models in production?

**Answer:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class PromptVersion:
    version: str  # "v1.2.3"
    prompt_template: str
    model: str
    created_at: datetime
    created_by: str
    description: str
    eval_scores: dict  # Results from eval pipeline
    is_active: bool = False

class PromptRegistry:
    def __init__(self, db):
        self._db = db

    def register(self, version: PromptVersion):
        # Store in database
        self._db.prompts.insert(version)

        # Version must pass evals before activation
        if not self._passes_evals(version):
            raise ValueError("Version failed eval gates")

    def activate(self, version_id: str):
        # Deactivate current
        self._db.prompts.update({"is_active": True}, {"is_active": False})
        # Activate new
        self._db.prompts.update({"version": version_id}, {"is_active": True})

    def get_active(self) -> PromptVersion:
        return self._db.prompts.find_one({"is_active": True})

    def rollback(self, to_version: str):
        # Instant rollback without deployment
        self.activate(to_version)
        log.warning(f"Rolled back to {to_version}")

    def _passes_evals(self, version: PromptVersion) -> bool:
        # Run eval pipeline
        report = run_all_evals(prompt=version.prompt_template, model=version.model)
        version.eval_scores = {
            "judge_score": report.judge_result.avg_score,
            "latency_p95": report.perf_result.metrics.p95_latency_ms,
        }
        return report.all_passed

# Usage
registry = PromptRegistry(db)

# Deploy new version
new_version = PromptVersion(
    version="v1.3.0",
    prompt_template="...",
    model="gpt-4o-mini",
    created_at=datetime.now(),
    created_by="engineer@function.health",
    description="Added CoT reasoning for complex cases",
    eval_scores={},
)
registry.register(new_version)  # Runs evals
registry.activate("v1.3.0")  # Goes live

# Quick rollback if issues
registry.rollback("v1.2.0")
```

---

### Q29: What's your approach to debugging a failing AI agent in production?

**Answer:**

**Step 1: Gather context**
```python
# Structured logs tell me what happened
{
    "event": "agent.error",
    "case_id": "abc-123",
    "error_type": "ValidationError",
    "error_message": "safety_notes is required",
    "model": "gpt-4o-mini",
    "prompt_version": "v1.2.3",
    "input_tokens": 1500,
    "raw_output": "...",  # What the LLM actually returned
}
```

**Step 2: Reproduce locally**
```python
# Load the exact inputs
case = load_case_from_logs("abc-123")

# Run with same configuration
result = run_agent(
    case,
    model="gpt-4o-mini",
    prompt_version="v1.2.3",
)

# Compare outputs
```

**Step 3: Check eval metrics**
```python
# Did this pass our golden sets?
judge_score = run_judge(case, result.output)
# If score is low, which dimension failed?
print(judge_score.reasoning)
```

**Step 4: Trace the execution**
```python
# LangGraph: inspect state at each node
for node, state in graph.stream(initial_state):
    print(f"{node}: {state}")

# DSPy: inspect reasoning
print(result.reasoning)  # ChainOfThought reasoning
```

**Step 5: Root cause categories**
1. **Model output parsing failed** → Improve output schema/validation
2. **Retrieval returned wrong docs** → Check embedding quality, add to golden set
3. **Safety check false positive** → Adjust safety thresholds
4. **Latency spike** → Check token count, model routing
5. **New edge case** → Add to golden set, improve prompt

**Step 6: Fix and prevent regression**
```python
# Add failing case to golden set
golden_cases.append(GoldenCase(
    id="prod-failure-001",
    query=case.query,
    labs=case.labs,
    expected_semantic_points=[...],
))

# Verify fix passes
pytest tests/test_evals.py -k "prod-failure-001"
```

---

### Q30: How would you approach building a personalized health agent with long-term memory?

**Answer:**

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Context Assembly                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Current    │  │  User       │  │  Historical         │  │
│  │  Labs       │  │  Profile    │  │  Interactions       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent Processing                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Retrieve   │  │  Analyze    │  │  Personalize        │  │
│  │  Context    │──▶  Labs      │──▶  Response           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Memory Update                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Store interaction, update user profile, log trends │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Data model:**

```python
@dataclass
class UserProfile:
    user_id: str
    # Static profile
    age: int
    sex: str
    conditions: list[str]  # ["hypothyroidism", "diabetes"]
    medications: list[str]

    # Dynamic memory
    lab_history: list[LabResult]  # All historical labs
    interaction_summary: str  # LLM-generated summary of past conversations
    preferences: dict  # {"detail_level": "high", "tone": "reassuring"}

    # Computed
    marker_baselines: dict[str, float]  # Personal normal ranges

class MemoryManager:
    def __init__(self, db, embeddings):
        self._db = db
        self._embeddings = embeddings

    def get_relevant_history(self, user_id: str, current_markers: list[str]) -> list[dict]:
        # Get historical values for current markers
        return self._db.query('''
            SELECT marker, value, unit, collected_at
            FROM lab_results
            WHERE user_id = %s AND marker = ANY(%s)
            ORDER BY collected_at DESC
            LIMIT 10
        ''', [user_id, current_markers])

    def get_interaction_context(self, user_id: str, query: str) -> str:
        # Semantic search over past interactions
        query_embedding = self._embeddings.embed(query)
        similar = self._db.query('''
            SELECT query, response_summary
            FROM agent_interactions
            WHERE user_id = %s
            ORDER BY embedding <-> %s
            LIMIT 3
        ''', [user_id, query_embedding])
        return format_as_context(similar)

    def update_after_interaction(self, user_id: str, interaction: dict):
        # Store the interaction
        self._db.insert('agent_interactions', interaction)

        # Update user profile summary periodically
        if self._should_update_summary(user_id):
            self._regenerate_summary(user_id)
```

**Personalization in the agent:**

```python
class PersonalizedLabInsightsModule(dspy.Module):
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.analyze = dspy.ChainOfThought(PersonalizedAnalysis)

    def forward(self, user_id: str, query: str, labs: list[dict]):
        # Gather personalized context
        profile = self.memory.get_profile(user_id)
        history = self.memory.get_relevant_history(user_id, [l['marker'] for l in labs])
        past_context = self.memory.get_interaction_context(user_id, query)

        # Analyze with personalization
        result = self.analyze(
            query=query,
            labs=format_labs(labs),
            user_profile=format_profile(profile),
            historical_labs=format_history(history),
            past_interactions=past_context,
        )

        # Update memory
        self.memory.update_after_interaction(user_id, {
            'query': query,
            'response_summary': result.summary,
            'markers_discussed': [l['marker'] for l in labs],
        })

        return result
```

---

## Quick Reference Card

### Key Numbers to Remember
- Judge threshold: **4.2/5**
- Safety weight: **30%**
- Clinical correctness weight: **40%**
- Latency regression threshold: **15%**
- Token regression threshold: **20%**
- p95 target: **<2000ms**

### Key Patterns
1. **Protocol → Implementation → Test Double → Factory**
2. **Signatures (DSPy) vs Nodes (LangGraph)**
3. **Eval gates: Schema → Retrieval → Judge → Perf**
4. **Safety: Schema + Node + Judge + Assertions**

### One-Liners for Each Topic
- **LangGraph**: "Explicit state machine - I control the flow"
- **DSPy**: "Declarative signatures - optimizer finds the prompts"
- **ReAct**: "Reasoning traces before each tool call"
- **Eval pipeline**: "Fast checks first, expensive only if passed"
- **Judge**: "Multi-dimensional rubric, weighted for healthcare"
- **Safety**: "Multiple layers - schema, node, judge, assertions"
- **Cost**: "Model routing + semantic cache + token regression tests"
