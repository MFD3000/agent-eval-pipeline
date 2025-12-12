# Task: Implement Semantic Caching for Agent Responses

## Priority: P2 (Performance Optimization)
## Estimated Effort: 5-6 hours
## Status: Planned

---

## Problem Statement

Current agent latency is ~7-10 seconds per request, with ~95% of time spent in LLM calls. For repeated or similar queries, we're making redundant API calls that could be cached.

## Goals

1. Reduce latency for repeated/similar queries to <100ms (cache hit)
2. Reduce LLM API costs for common queries
3. Maintain correctness - **no false cache hits** (critical for healthcare)

---

## Design Principles

### Cache Key Requirements (CRITICAL)

The cache key MUST include ALL inputs that affect the output:

```python
cache_key = deterministic_hash(
    query,              # User's question
    sorted(labs),       # Lab values (sorted for consistency)
    sorted(history),    # Historical values
    sorted(symptoms),   # Reported symptoms
    model_name,         # e.g., "gpt-4o-mini"
    prompt_version,     # Hash of prompt template
    schema_version,     # Hash of output schema
)
```

### What NOT to Cache

- Queries with real-time data requirements
- Queries that reference "today", "current", etc.
- Any query where staleness could cause harm

### Cache Policy

| Setting | Value | Rationale |
|---------|-------|-----------|
| TTL | 24 hours | Lab interpretations don't change daily |
| Max entries | 10,000 | Prevent unbounded memory growth |
| Eviction | LRU | Keep frequently accessed entries |

---

## Implementation Plan

### Phase 1: Cache Key Builder (1 hour)

```python
# src/agent_eval_pipeline/cache/keys.py

import hashlib
import json
from typing import Any

PROMPT_VERSION = "v1.0"  # Bump when prompt changes
SCHEMA_VERSION = "v1.0"  # Bump when output schema changes

def build_cache_key(
    query: str,
    labs: list[dict],
    history: list[dict],
    symptoms: list[str],
    model: str,
) -> str:
    """
    Build a deterministic cache key from all inputs.

    CRITICAL: Any change to inputs that affects output
    MUST be reflected in the cache key.
    """
    # Sort for determinism
    sorted_labs = sorted(labs, key=lambda x: (x.get('marker', ''), x.get('date', '')))
    sorted_history = sorted(history, key=lambda x: (x.get('marker', ''), x.get('date', '')))
    sorted_symptoms = sorted(symptoms)

    key_data = {
        "query": query.strip().lower(),
        "labs": sorted_labs,
        "history": sorted_history,
        "symptoms": sorted_symptoms,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
    }

    # Deterministic JSON serialization
    key_json = json.dumps(key_data, sort_keys=True, separators=(',', ':'))

    # SHA-256 for fixed-length key
    return hashlib.sha256(key_json.encode()).hexdigest()
```

### Phase 2: Cache Protocol & Implementations (1 hour)

```python
# src/agent_eval_pipeline/cache/protocol.py

from typing import Protocol, Any
from datetime import timedelta

class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: str) -> Any | None:
        """Get value from cache. Returns None if not found or expired."""
        ...

    def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        """Set value in cache with optional TTL."""
        ...

    def delete(self, key: str) -> None:
        """Delete a key from cache."""
        ...

    def clear(self) -> None:
        """Clear all entries."""
        ...


# src/agent_eval_pipeline/cache/memory.py

from datetime import datetime, timedelta
from collections import OrderedDict

class InMemoryCache:
    """LRU cache with TTL support. For development/testing."""

    def __init__(self, max_size: int = 10000, default_ttl: timedelta = timedelta(hours=24)):
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None

        value, expires_at = self._cache[key]

        if datetime.now() > expires_at:
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        ttl = ttl or self._default_ttl
        expires_at = datetime.now() + ttl

        # Evict if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (value, expires_at)
        self._cache.move_to_end(key)
```

### Phase 3: Redis Implementation (1 hour)

```python
# src/agent_eval_pipeline/cache/redis.py

import json
from datetime import timedelta
from typing import Any

class RedisCache:
    """Redis-backed cache for production."""

    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "agent:"):
        import redis
        self._client = redis.from_url(url)
        self._prefix = prefix

    def get(self, key: str) -> Any | None:
        data = self._client.get(self._prefix + key)
        if data is None:
            return None
        return json.loads(data)

    def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        data = json.dumps(value, default=str)
        if ttl:
            self._client.setex(self._prefix + key, ttl, data)
        else:
            self._client.set(self._prefix + key, data)
```

### Phase 4: Integration into Agent (30 min)

```python
# Modify agent/__init__.py

def run_agent(
    case: GoldenCase,
    agent_type: AgentType | None = None,
    use_cache: bool | None = None,  # NEW
) -> AgentResult | AgentError:
    """
    Run agent with optional caching.

    Cache behavior controlled by:
    - use_cache parameter (explicit)
    - CACHE_ENABLED env var (default: false)
    """
    # Check cache
    if _should_use_cache(use_cache):
        cache = _get_cache()
        cache_key = build_cache_key(...)

        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Cache HIT for {case.id}")
            return AgentResult(**cached, from_cache=True)

        logger.info(f"Cache MISS for {case.id}")

    # Run agent
    result = _run_agent_impl(case, agent_type)

    # Store in cache
    if _should_use_cache(use_cache) and isinstance(result, AgentResult):
        cache.set(cache_key, result.model_dump())

    return result
```

### Phase 5: Observability (30 min)

Add cache metrics to traces:

```python
span.set_attribute("cache.enabled", use_cache)
span.set_attribute("cache.hit", cached is not None)
span.set_attribute("cache.key", cache_key[:16] + "...")  # Truncated for display
```

### Phase 6: Testing (1-2 hours)

```python
# tests/test_cache.py

class TestCacheKeyDeterminism:
    """Cache keys must be deterministic."""

    def test_same_inputs_same_key(self):
        key1 = build_cache_key(query="test", labs=[...], ...)
        key2 = build_cache_key(query="test", labs=[...], ...)
        assert key1 == key2

    def test_different_query_different_key(self):
        key1 = build_cache_key(query="TSH results", ...)
        key2 = build_cache_key(query="thyroid results", ...)
        assert key1 != key2

    def test_different_labs_different_key(self):
        """CRITICAL: Different lab values must never share cache."""
        key1 = build_cache_key(labs=[{"marker": "TSH", "value": 5.5}], ...)
        key2 = build_cache_key(labs=[{"marker": "TSH", "value": 55}], ...)  # 10x different!
        assert key1 != key2

    def test_lab_order_irrelevant(self):
        """Order shouldn't matter - sorted internally."""
        key1 = build_cache_key(labs=[{"marker": "TSH"}, {"marker": "T4"}], ...)
        key2 = build_cache_key(labs=[{"marker": "T4"}, {"marker": "TSH"}], ...)
        assert key1 == key2


class TestCacheCorrectness:
    """Ensure no false cache hits."""

    def test_model_change_invalidates(self):
        """Different models must not share cache."""
        ...

    def test_prompt_version_invalidates(self):
        """Prompt changes must invalidate cache."""
        ...
```

---

## Configuration

```bash
# Environment variables
CACHE_ENABLED=false          # Default OFF for safety
CACHE_BACKEND=memory         # "memory" or "redis"
CACHE_TTL_HOURS=24           # Time to live
CACHE_MAX_SIZE=10000         # Max entries (memory only)
REDIS_URL=redis://localhost:6379
```

---

## Rollout Plan

1. **Development**: Test with `CACHE_ENABLED=true`, `CACHE_BACKEND=memory`
2. **Staging**: Enable with Redis, monitor hit rates
3. **Production**: Gradual rollout with feature flag
4. **Monitoring**: Track cache hit rate, latency improvement, any correctness issues

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Cache hit rate | >30% (for repeated queries) |
| Hit latency | <100ms |
| Miss latency | No regression from current |
| False positive rate | 0% (CRITICAL) |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Stale cached data | User sees outdated info | 24h TTL, version in key |
| Cache key collision | Wrong response returned | SHA-256, comprehensive tests |
| Memory pressure | OOM in production | LRU eviction, max size limit |
| Redis unavailable | Cache miss, fallback to LLM | Graceful degradation |

---

## Future Enhancements

1. **Semantic similarity caching** (with caution for healthcare)
2. **Cache warming** for common queries
3. **Per-user cache** for personalized responses
4. **Cache analytics dashboard**
