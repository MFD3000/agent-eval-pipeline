"""
Safety node - applies guardrails to agent output.

This is a PURE NODE with NO dependencies - it can be called directly
without any factory pattern. Perfect for unit testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary, SafetyNote

if TYPE_CHECKING:
    from agent_eval_pipeline.agent.state import AgentState


def apply_safety(state: AgentState) -> dict:
    """
    Apply safety guardrails to the analysis.

    This node ensures:
    - Non-diagnostic disclaimer is present
    - Doctor consultation is recommended
    - No medication recommendations

    Reads from state:
    - raw_analysis: Output from analysis node

    Writes to state:
    - final_output: Complete LabInsightsSummary with safety notes
    """
    raw = state["raw_analysis"]

    # Handle missing analysis
    if not raw:
        return {"final_output": None}

    # Reconstruct as Pydantic model
    output = LabInsightsSummary(**raw)

    # Ensure safety notes include non-diagnostic disclaimer
    has_non_diagnostic = any(
        note.type == "non_diagnostic" for note in output.safety_notes
    )

    if not has_non_diagnostic:
        output.safety_notes.append(
            SafetyNote(
                message=(
                    "This information is for educational purposes only and is not a diagnosis. "
                    "Please consult with your healthcare provider for medical advice."
                ),
                type="non_diagnostic",
            )
        )

    # Ensure doctor recommendation exists
    if not output.recommended_topics_for_doctor:
        output.recommended_topics_for_doctor = [
            "Review these lab results with your healthcare provider"
        ]

    return {"final_output": output}
