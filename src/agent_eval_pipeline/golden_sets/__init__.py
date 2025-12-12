"""
Golden Sets Package

Provides test cases with evaluation criteria for agent testing.
All golden case types and utilities are re-exported here for convenience.

EXTENSIBILITY:
--------------
To add a new biomarker panel (e.g., lipid panel):

1. Create `lipid_cases.py` following the `thyroid_cases.py` pattern
2. Add import and extend `get_all_golden_cases()` below
3. All evaluators automatically pick up the new cases

Example:
    from agent_eval_pipeline.golden_sets import (
        GoldenCase,
        LabValue,
        get_all_golden_cases,
        get_case_by_id,
    )
"""

# Re-export data classes
from agent_eval_pipeline.golden_sets.thyroid_cases import (
    LabValue,
    ExpectedToolCall,
    GoldenCase,
)

# Re-export thyroid-specific access (for backwards compatibility)
from agent_eval_pipeline.golden_sets.thyroid_cases import (
    THYROID_CASES,
    get_all_golden_cases as get_thyroid_cases,
    get_case_by_id as get_thyroid_case_by_id,
)


def get_all_golden_cases() -> list[GoldenCase]:
    """
    Get all golden cases from all biomarker panels.

    This is the primary entry point for evaluators.
    Add new panels here as they're created.

    Returns:
        Combined list of golden cases from all panels
    """
    cases: list[GoldenCase] = []

    # Thyroid panel
    cases.extend(THYROID_CASES)

    # Future panels go here:
    # from agent_eval_pipeline.golden_sets.lipid_cases import LIPID_CASES
    # cases.extend(LIPID_CASES)

    return cases


def get_case_by_id(case_id: str) -> GoldenCase | None:
    """
    Get a specific golden case by ID from any panel.

    Args:
        case_id: The case ID (e.g., "thyroid-001", "lipid-001")

    Returns:
        The matching GoldenCase or None if not found
    """
    for case in get_all_golden_cases():
        if case.id == case_id:
            return case
    return None


__all__ = [
    # Data classes
    "LabValue",
    "ExpectedToolCall",
    "GoldenCase",
    # Access functions
    "get_all_golden_cases",
    "get_case_by_id",
    # Panel-specific (for direct access if needed)
    "THYROID_CASES",
    "get_thyroid_cases",
    "get_thyroid_case_by_id",
]
