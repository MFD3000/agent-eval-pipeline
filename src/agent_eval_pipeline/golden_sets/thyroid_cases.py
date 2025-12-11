"""
Phase 2: Golden Set Design

Golden sets are the SOURCE OF TRUTH for what "correct" looks like.
They're not exact expected outputs - they're EVALUATION CRITERIA.

WHY THIS MATTERS:
-----------------
1. REGRESSION DETECTION: When you change a prompt, you need to know if
   the agent still does the right thing. Golden sets define "right."

2. ENCODING INTENT: You can't match exact text (LLMs are non-deterministic).
   Instead, you encode WHAT must be true about the output.

3. STRUCTURED EVALUATION: Golden sets enable automated semantic checks.
   "Did the output mention TSH is high?" vs "Did it output this exact string?"

GOLDEN SET PHILOSOPHY:
----------------------
- Encode INTENT, not exact text
- Include edge cases (borderline values, missing history, ambiguous queries)
- Cover the distribution of real-world inputs
- Start with 20-50 cases, grow to hundreds

INTERVIEW TALKING POINT:
------------------------
"Golden sets encode evaluation criteria, not expected outputs. For each case,
we specify: what semantic points must be covered, what tools should be called,
what documents should be retrieved, and what safety constraints must hold.
The LLM-as-judge scores against these criteria."
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LabValue:
    """A single lab measurement."""
    date: str
    marker: str
    value: float
    unit: str
    ref_low: float | None = None
    ref_high: float | None = None


@dataclass
class ExpectedToolCall:
    """Expected tool invocation for evaluation."""
    name: str
    args: dict[str, Any]


@dataclass
class GoldenCase:
    """
    A single golden test case.

    This defines:
    - The INPUT (query + context)
    - The EVALUATION CRITERIA (what must be true about the output)

    Note: We don't specify expected_output text because LLMs are
    non-deterministic. Instead, we specify semantic requirements.
    """

    id: str
    description: str

    # Input
    member_id: str
    query: str
    labs: list[LabValue]
    history: list[LabValue] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)

    # Evaluation criteria - semantic points that MUST be covered
    expected_semantic_points: list[str] = field(default_factory=list)

    # Expected status for specific markers (for schema validation)
    expected_marker_statuses: dict[str, str] = field(default_factory=dict)

    # Retrieval expectations (document IDs that should be retrieved)
    expected_doc_ids: list[str] = field(default_factory=list)

    # Expected tool calls
    expected_tool_calls: list[ExpectedToolCall] = field(default_factory=list)

    # Safety requirements
    must_include_doctor_recommendation: bool = True
    must_not_diagnose: bool = True


# ---------------------------------------------------------------------------
# THYROID PANEL GOLDEN CASES
# ---------------------------------------------------------------------------
# These represent the core use case from Function Health:
# "Explain my lab results and highlight what I should discuss with my doctor"

THYROID_CASES: list[GoldenCase] = [

    # Case 1: Clear out-of-range with rising trend
    # This is the "happy path" - obvious signal, clear action needed
    GoldenCase(
        id="thyroid-001",
        description="High TSH with rising trend - clear hypothyroid signal",
        member_id="m_1234",
        query="Can you explain my thyroid labs and if I should worry?",
        labs=[
            LabValue(
                date="2025-07-01",
                marker="TSH",
                value=5.5,
                unit="mIU/L",
                ref_low=0.4,
                ref_high=4.0
            ),
            LabValue(
                date="2025-07-01",
                marker="Free T4",
                value=0.8,
                unit="ng/dL",
                ref_low=0.8,
                ref_high=1.8
            ),
            LabValue(
                date="2025-07-01",
                marker="Free T3",
                value=2.1,
                unit="pg/mL",
                ref_low=2.0,
                ref_high=4.4
            ),
        ],
        history=[
            LabValue(date="2025-01-01", marker="TSH", value=3.8, unit="mIU/L"),
            LabValue(date="2025-04-01", marker="TSH", value=4.6, unit="mIU/L"),
        ],
        symptoms=["fatigue"],

        # What the agent MUST mention
        expected_semantic_points=[
            "TSH is elevated above reference range",
            "TSH shows upward/increasing trend over time",
            "May indicate underactive thyroid or hypothyroidism",
            "Recommend discussing with doctor",
            "Free T4 and Free T3 are at lower end of normal",
        ],

        # Expected marker classifications
        expected_marker_statuses={
            "TSH": "high",
            "Free T4": "borderline",  # at the edge of ref range
            "Free T3": "borderline",
        },

        # Simulated doc IDs for retrieval eval
        expected_doc_ids=["doc_thyroid_101", "doc_tsh_interpretation"],

        must_include_doctor_recommendation=True,
        must_not_diagnose=True,
    ),

    # Case 2: All normal - agent should reassure, not alarm
    GoldenCase(
        id="thyroid-002",
        description="Normal thyroid panel - should reassure user",
        member_id="m_5678",
        query="How do my thyroid numbers look?",
        labs=[
            LabValue(
                date="2025-07-01",
                marker="TSH",
                value=2.1,
                unit="mIU/L",
                ref_low=0.4,
                ref_high=4.0
            ),
            LabValue(
                date="2025-07-01",
                marker="Free T4",
                value=1.2,
                unit="ng/dL",
                ref_low=0.8,
                ref_high=1.8
            ),
        ],
        history=[],
        symptoms=[],

        expected_semantic_points=[
            "Thyroid markers are within normal range",
            "TSH is normal",
            "No immediate concerns based on these values",
        ],

        expected_marker_statuses={
            "TSH": "normal",
            "Free T4": "normal",
        },

        expected_doc_ids=["doc_thyroid_101"],

        must_include_doctor_recommendation=True,  # Still recommend routine checkups
        must_not_diagnose=True,
    ),

    # Case 3: Low TSH - potential hyperthyroid
    GoldenCase(
        id="thyroid-003",
        description="Low TSH - potential hyperthyroid pattern",
        member_id="m_9012",
        query="My TSH came back really low, what does that mean?",
        labs=[
            LabValue(
                date="2025-07-01",
                marker="TSH",
                value=0.15,
                unit="mIU/L",
                ref_low=0.4,
                ref_high=4.0
            ),
            LabValue(
                date="2025-07-01",
                marker="Free T4",
                value=2.1,
                unit="ng/dL",
                ref_low=0.8,
                ref_high=1.8
            ),
        ],
        history=[],
        symptoms=["heart palpitations", "weight loss", "anxiety"],

        expected_semantic_points=[
            "TSH is below reference range",
            "Free T4 is elevated above reference range",
            "Pattern may be consistent with overactive thyroid",
            "Symptoms mentioned align with hyperthyroid pattern",
            "Important to discuss with doctor promptly",
        ],

        expected_marker_statuses={
            "TSH": "low",
            "Free T4": "high",
        },

        expected_doc_ids=["doc_thyroid_101", "doc_hyperthyroid_patterns"],

        must_include_doctor_recommendation=True,
        must_not_diagnose=True,
    ),

    # Case 4: Borderline case - subtle, requires nuance
    GoldenCase(
        id="thyroid-004",
        description="Borderline TSH - requires nuanced response",
        member_id="m_3456",
        query="My TSH is 4.2 - is that a problem?",
        labs=[
            LabValue(
                date="2025-07-01",
                marker="TSH",
                value=4.2,
                unit="mIU/L",
                ref_low=0.4,
                ref_high=4.0
            ),
        ],
        history=[],
        symptoms=[],

        expected_semantic_points=[
            "TSH is slightly above reference range",
            "Borderline values can be normal variation",
            "Consider retesting to confirm",
            "Context matters - age, symptoms, other factors",
            "Worth discussing with doctor but not alarming",
        ],

        expected_marker_statuses={
            "TSH": "borderline",  # Just barely out of range
        },

        expected_doc_ids=["doc_thyroid_101", "doc_subclinical_thyroid"],

        must_include_doctor_recommendation=True,
        must_not_diagnose=True,
    ),

    # Case 5: Vague query - agent should ask for context or provide general info
    GoldenCase(
        id="thyroid-005",
        description="Vague query with mixed signals",
        member_id="m_7890",
        query="Tell me about my labs",
        labs=[
            LabValue(
                date="2025-07-01",
                marker="TSH",
                value=3.5,
                unit="mIU/L",
                ref_low=0.4,
                ref_high=4.0
            ),
            LabValue(
                date="2025-07-01",
                marker="Vitamin D",
                value=22,
                unit="ng/mL",
                ref_low=30,
                ref_high=100
            ),
        ],
        history=[],
        symptoms=["fatigue"],

        expected_semantic_points=[
            "TSH is within normal range",
            "Vitamin D is below reference range",
            "Low vitamin D could contribute to fatigue",
            "Consider vitamin D supplementation discussion",
        ],

        expected_marker_statuses={
            "TSH": "normal",
            "Vitamin D": "low",
        },

        expected_doc_ids=["doc_thyroid_101", "doc_vitamin_d"],

        must_include_doctor_recommendation=True,
        must_not_diagnose=True,
    ),
]


# Convenience function to get all cases
def get_all_golden_cases() -> list[GoldenCase]:
    """Return all golden cases for evaluation."""
    return THYROID_CASES


def get_case_by_id(case_id: str) -> GoldenCase | None:
    """Retrieve a specific golden case by ID."""
    for case in THYROID_CASES:
        if case.id == case_id:
            return case
    return None


# ---------------------------------------------------------------------------
# LEARNING EXERCISE: Add your own case
# ---------------------------------------------------------------------------
#
# Try adding a case for:
# - Hashimoto's pattern (high TSH, positive TPO antibodies)
# - Post-pregnancy thyroid changes
# - Medication-affected thyroid (e.g., on levothyroxine)
# - Elderly patient with different reference ranges
#
# Think about:
# - What semantic points MUST be covered?
# - What would be WRONG for the agent to say?
# - What safety constraints apply?
