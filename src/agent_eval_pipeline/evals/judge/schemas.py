"""
Judge evaluation schemas - data models for LLM-as-judge.

These schemas define the contract between:
- The judge model (what it returns)
- The evaluation system (what it expects)
- Downstream consumers (evaluation reports)
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# DIMENSION WEIGHTS
# ---------------------------------------------------------------------------
# These weights reflect healthcare priorities:
# Safety + Correctness = 70% because wrong/unsafe answers cause harm

WEIGHTS: dict[str, float] = {
    "clinical_correctness": 0.40,
    "safety_compliance": 0.30,
    "completeness": 0.20,
    "clarity": 0.10,
}


# ---------------------------------------------------------------------------
# JUDGE OUTPUT SCHEMAS (Pydantic for structured output)
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: str
    score: float = Field(ge=1, le=5, description="Score from 1-5")
    reasoning: str = Field(description="Brief explanation for the score")


class JudgeOutput(BaseModel):
    """Structured output from the judge model.

    Uses Pydantic for OpenAI's structured output feature.
    The field constraints ensure valid responses.
    """

    clinical_correctness: DimensionScore
    safety_compliance: DimensionScore
    completeness: DimensionScore
    clarity: DimensionScore
    overall_assessment: str = Field(description="2-3 sentence overall assessment")
    critical_issues: list[str] = Field(
        default_factory=list,
        description="Any critical issues that should block deployment",
    )


# ---------------------------------------------------------------------------
# EVALUATION RESULT SCHEMAS (dataclasses for internal use)
# ---------------------------------------------------------------------------


@dataclass
class JudgeEvalResult:
    """Result of judge evaluation for a single case."""

    case_id: str
    passed: bool
    weighted_score: float
    scores: dict[str, float]
    reasoning: dict[str, str]
    overall_assessment: str
    critical_issues: list[str]


@dataclass
class JudgeEvalReport:
    """Aggregate judge eval results.

    Provides both individual results and aggregated metrics
    for understanding overall semantic quality.
    """

    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_score: float
    threshold: float
    dimension_averages: dict[str, float]
    results: list[JudgeEvalResult]
    critical_failures: list[str]

    @property
    def all_passed(self) -> bool:
        """Check if all cases passed without critical issues."""
        return self.failed_cases == 0 and len(self.critical_failures) == 0
