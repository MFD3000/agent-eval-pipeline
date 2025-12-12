"""
Baseline storage - Protocol and implementations for storing performance baselines.

Following the gold standard pattern:
1. Protocol defines the interface
2. FileBaselineStore for production (persistent)
3. InMemoryBaselineStore for testing (fast, no I/O)
4. Factory function for convenience
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# BASELINE DATA MODEL
# ---------------------------------------------------------------------------


@dataclass
class PerformanceBaseline:
    """Stored performance baseline from previous runs.

    These values represent the "expected" performance.
    Deviations beyond thresholds trigger regression failures.
    """

    p50_latency_ms: float
    p95_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    expected_model: str
    run_count: int = 1


# ---------------------------------------------------------------------------
# BASELINE STORE PROTOCOL
# ---------------------------------------------------------------------------


@runtime_checkable
class BaselineStore(Protocol):
    """Protocol for baseline storage implementations."""

    def load(self) -> PerformanceBaseline | None:
        """Load baseline, returns None if not found."""
        ...

    def save(self, baseline: PerformanceBaseline) -> None:
        """Save baseline."""
        ...


# ---------------------------------------------------------------------------
# FILE-BASED IMPLEMENTATION (Production)
# ---------------------------------------------------------------------------


class FileBaselineStore:
    """Production baseline store using JSON file.

    Persists baseline between CI runs to detect regressions
    across different commits.
    """

    def __init__(self, file_path: Path | str | None = None):
        if file_path is None:
            # Default to project root
            file_path = Path(__file__).parent.parent.parent.parent.parent / ".perf_baseline.json"
        self._path = Path(file_path)

    @property
    def path(self) -> Path:
        """Get the baseline file path."""
        return self._path

    def load(self) -> PerformanceBaseline | None:
        """Load baseline from JSON file."""
        if not self._path.exists():
            return None

        try:
            with open(self._path) as f:
                data = json.load(f)
            return PerformanceBaseline(**data)
        except Exception as e:
            print(f"Warning: Could not load baseline: {e}")
            return None

    def save(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to JSON file."""
        with open(self._path, "w") as f:
            json.dump(
                {
                    "p50_latency_ms": baseline.p50_latency_ms,
                    "p95_latency_ms": baseline.p95_latency_ms,
                    "avg_input_tokens": baseline.avg_input_tokens,
                    "avg_output_tokens": baseline.avg_output_tokens,
                    "avg_total_tokens": baseline.avg_total_tokens,
                    "expected_model": baseline.expected_model,
                    "run_count": baseline.run_count,
                },
                f,
                indent=2,
            )


# ---------------------------------------------------------------------------
# IN-MEMORY IMPLEMENTATION (Testing)
# ---------------------------------------------------------------------------


class InMemoryBaselineStore:
    """Test baseline store - no file I/O.

    Perfect for unit tests that need to verify regression
    detection logic without touching the filesystem.
    """

    def __init__(self, initial_baseline: PerformanceBaseline | None = None):
        self._baseline = initial_baseline
        self._save_called = False

    @property
    def save_called(self) -> bool:
        """Check if save was called (for test assertions)."""
        return self._save_called

    @property
    def current_baseline(self) -> PerformanceBaseline | None:
        """Get current baseline (for test assertions)."""
        return self._baseline

    def load(self) -> PerformanceBaseline | None:
        """Return stored baseline."""
        return self._baseline

    def save(self, baseline: PerformanceBaseline) -> None:
        """Store baseline in memory."""
        self._baseline = baseline
        self._save_called = True


# ---------------------------------------------------------------------------
# FACTORY FUNCTION
# ---------------------------------------------------------------------------


def get_baseline_store(
    use_file: bool = True,
    file_path: Path | str | None = None,
    initial_baseline: PerformanceBaseline | None = None,
) -> BaselineStore:
    """
    Factory function for baseline stores.

    Args:
        use_file: If True, use FileBaselineStore. If False, use InMemoryBaselineStore.
        file_path: Custom path for FileBaselineStore.
        initial_baseline: Initial baseline for InMemoryBaselineStore.

    Returns:
        BaselineStore implementation.

    Example:
        # Production
        store = get_baseline_store()

        # Testing
        store = get_baseline_store(
            use_file=False,
            initial_baseline=PerformanceBaseline(...)
        )
    """
    if use_file:
        return FileBaselineStore(file_path)
    else:
        return InMemoryBaselineStore(initial_baseline)
