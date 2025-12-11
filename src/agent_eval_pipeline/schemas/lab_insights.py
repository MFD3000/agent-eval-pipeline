"""
Phase 1: Structured Output Schemas

These Pydantic models define the OUTPUT CONTRACT for the LabInsightsAgent.
Every response from the agent MUST validate against these schemas.

WHY THIS MATTERS:
-----------------
1. PREDICTABILITY: Without schemas, LLM outputs are unpredictable strings.
   With schemas, you get guaranteed structure you can program against.

2. REGRESSION DETECTION: If a prompt change breaks the output format,
   schema validation catches it immediately - before it hits production.

3. TYPE SAFETY: Downstream code can rely on these types. No more
   `if "insights" in response and isinstance(response["insights"], list)`.

4. DOCUMENTATION: The schema IS the documentation. Anyone can look at
   these models and understand exactly what the agent produces.

INTERVIEW TALKING POINT:
------------------------
"We use Pydantic schemas as contracts between the LLM and the rest of the
system. The first eval gate validates every output against these schemas.
If a prompt change breaks the structure, the PR is blocked before merge."
"""

from typing import Literal
from pydantic import BaseModel, Field


class MarkerInsight(BaseModel):
    """
    Analysis of a single lab marker (e.g., TSH, Free T4).

    This is the atomic unit of insight - one marker, one analysis.
    The agent produces a list of these for all relevant markers.
    """

    marker: str = Field(
        description="Name of the lab marker (e.g., 'TSH', 'Free T4')"
    )

    status: Literal["high", "low", "normal", "borderline"] = Field(
        description="Whether the value is out of reference range"
    )
    # WHY Literal instead of str?
    # - Prevents hallucinated statuses like "slightly elevated" or "concerning"
    # - Makes downstream logic simple: if status == "high": ...
    # - Eval gate catches if model outputs invalid status

    value: float = Field(
        description="The measured value"
    )

    unit: str = Field(
        description="Unit of measurement (e.g., 'mIU/L', 'ng/dL')"
    )

    ref_range: str = Field(
        description="Reference range as string (e.g., '0.4-4.0')"
    )

    trend: Literal["increasing", "decreasing", "stable", "unknown"] = Field(
        description="Direction of change compared to historical values"
    )
    # WHY include trend?
    # - A single out-of-range value is different from a consistently rising trend
    # - This is key clinical context for "should I worry?" questions

    clinical_relevance: str = Field(
        description="Brief explanation of what this marker means clinically"
    )
    # This is where the LLM adds value - translating medical jargon
    # Example: "TSH controls thyroid function. High TSH may indicate
    # your thyroid is underactive."

    action: str = Field(
        description="Recommended action (e.g., 'Discuss with doctor', 'Monitor')"
    )


class SafetyNote(BaseModel):
    """
    Guardrail messaging to keep the agent within safe boundaries.

    In healthcare AI, you MUST include disclaimers. This isn't just legal
    protection - it's about not causing harm by being misinterpreted as
    medical advice.
    """

    message: str = Field(
        description="The safety message content"
    )

    type: Literal["non_diagnostic", "emergency_notice", "lifestyle_scope"] = Field(
        description="Category of safety note"
    )
    # Types explained:
    # - non_diagnostic: "This is not a diagnosis. Consult your doctor."
    # - emergency_notice: "If experiencing X symptoms, seek immediate care."
    # - lifestyle_scope: "These suggestions are general wellness, not treatment."


class LabInsightsSummary(BaseModel):
    """
    The complete structured response from LabInsightsAgent.

    This is the TOP-LEVEL SCHEMA that the agent must produce.
    Every field here is intentional and serves the user experience.
    """

    summary: str = Field(
        description="Plain-language overview of the lab results (2-3 sentences)"
    )
    # The TL;DR - what a busy person reads first
    # Example: "Your thyroid markers show elevated TSH with a rising trend
    # over the past 6 months. This may indicate your thyroid is becoming
    # less active. Your other markers are within normal range."

    key_insights: list[MarkerInsight] = Field(
        description="Detailed analysis of each relevant marker"
    )
    # Only include markers that matter - don't dump every normal value
    # The agent should prioritize: out-of-range > trending > normal-but-relevant

    recommended_topics_for_doctor: list[str] = Field(
        description="Specific questions/topics to discuss with healthcare provider"
    )
    # Actionable output - this is what makes the agent useful
    # Example: ["Rising TSH trend over 6 months", "Fatigue symptoms",
    #           "Whether thyroid medication might be appropriate"]

    lifestyle_considerations: list[str] = Field(
        description="General wellness suggestions (not medical advice)"
    )
    # Safe, general suggestions
    # Example: ["Ensure adequate iodine intake through diet",
    #           "Monitor energy levels and sleep quality"]

    safety_notes: list[SafetyNote] = Field(
        description="Required guardrail messages"
    )
    # MUST include at least one non_diagnostic note
    # The eval can check for this

# This is what the schema eval does on every golden case.
