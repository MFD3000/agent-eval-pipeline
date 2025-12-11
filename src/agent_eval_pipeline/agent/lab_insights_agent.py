"""
Phase 3: The LabInsightsAgent

This is the agent that produces structured lab insights.
It uses OpenAI's structured output mode to guarantee schema compliance.

WHY STRUCTURED OUTPUTS MATTER:
------------------------------
1. GUARANTEED SCHEMA: OpenAI's response_format with JSON schema ensures
   the output ALWAYS matches your Pydantic model. No parsing errors.

2. SEPARATION OF CONCERNS: The agent focuses on WHAT to say, not HOW
   to format it. The schema handles structure.

3. TESTABILITY: With guaranteed structure, you can write reliable tests.
   No flaky tests due to format variations.

AGENT DESIGN PRINCIPLES:
------------------------
- System prompt sets role, constraints, and safety guardrails
- User message provides the specific query and context
- Output is always LabInsightsSummary

INTERVIEW TALKING POINT:
------------------------
"The agent uses structured outputs so we get guaranteed JSON that matches
our Pydantic schema. This means we can reliably test the agent's behavior
and catch regressions. The first eval gate validates this structure on
every golden case."
"""

import os
import json
from typing import Any
from dataclasses import dataclass
import time

from openai import OpenAI
from pydantic import ValidationError

from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase, LabValue


# System prompt - this is where the "personality" and constraints live
SYSTEM_PROMPT = """You are a health insights assistant that helps members understand their lab results.

ROLE:
- You explain lab values in plain, accessible language
- You identify patterns and trends across historical data
- You highlight what's worth discussing with a healthcare provider
- You provide general wellness context (not medical advice)

CONSTRAINTS (CRITICAL - NEVER VIOLATE):
1. NEVER diagnose conditions. Say "may be consistent with" not "you have"
2. NEVER recommend specific treatments or medications
3. ALWAYS recommend discussing findings with a healthcare provider
4. NEVER claim certainty about clinical implications
5. Focus on EDUCATION, not prescription

OUTPUT REQUIREMENTS:
- Provide a clear summary first (2-3 sentences)
- Analyze each relevant marker with status, trend, and clinical context
- Suggest specific topics to discuss with doctor
- Include appropriate safety disclaimers

TONE:
- Empathetic but not alarming
- Educational but accessible
- Empowering but not overstepping medical boundaries"""


def format_labs_for_prompt(labs: list[LabValue], history: list[LabValue]) -> str:
    """Format lab values for the LLM prompt."""
    lines = ["CURRENT LAB VALUES:"]
    for lab in labs:
        ref_range = ""
        if lab.ref_low is not None and lab.ref_high is not None:
            ref_range = f" (ref: {lab.ref_low}-{lab.ref_high})"
        lines.append(f"- {lab.marker}: {lab.value} {lab.unit}{ref_range} [{lab.date}]")

    if history:
        lines.append("\nHISTORICAL VALUES:")
        for lab in history:
            lines.append(f"- {lab.marker}: {lab.value} {lab.unit} [{lab.date}]")

    return "\n".join(lines)


def format_symptoms(symptoms: list[str]) -> str:
    """Format symptoms for the prompt."""
    if not symptoms:
        return "No symptoms reported."
    return "REPORTED SYMPTOMS: " + ", ".join(symptoms)


@dataclass
class AgentResult:
    """Result from running the agent, including metrics for eval."""
    output: LabInsightsSummary
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str


@dataclass
class AgentError:
    """Error result when agent fails."""
    error_type: str
    error_message: str
    raw_response: str | None = None


def run_agent(
    case: GoldenCase,
    model: str | None = None,
) -> AgentResult | AgentError:
    """
    Run the LabInsightsAgent on a golden case.

    Args:
        case: The golden case to run
        model: Override the model (default: AGENT_MODEL env var or gpt-4o-mini)

    Returns:
        AgentResult on success, AgentError on failure
    """
    # Get configuration
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return AgentError(
            error_type="ConfigurationError",
            error_message="OPENAI_API_KEY environment variable not set"
        )

    model = model or os.environ.get("AGENT_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    # Build the user message
    user_message = f"""MEMBER QUERY: {case.query}

{format_labs_for_prompt(case.labs, case.history)}

{format_symptoms(case.symptoms)}

Please analyze these lab results and provide insights."""

    # Call OpenAI with structured output
    start_time = time.time()
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=LabInsightsSummary,
        )
        latency_ms = (time.time() - start_time) * 1000

        # Extract the parsed output
        parsed_output = response.choices[0].message.parsed

        if parsed_output is None:
            return AgentError(
                error_type="ParseError",
                error_message="Model returned None for parsed output",
                raw_response=response.choices[0].message.content
            )

        return AgentResult(
            output=parsed_output,
            latency_ms=latency_ms,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            model=model,
        )

    except ValidationError as e:
        return AgentError(
            error_type="ValidationError",
            error_message=str(e),
        )
    except Exception as e:
        return AgentError(
            error_type=type(e).__name__,
            error_message=str(e),
        )


def run_agent_raw(
    query: str,
    labs: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
    symptoms: list[str] | None = None,
    model: str | None = None,
) -> AgentResult | AgentError:
    """
    Run the agent with raw inputs (for testing without GoldenCase).

    This is useful for:
    - Manual testing during development
    - Integration with external systems
    - One-off queries
    """
    # Convert dicts to LabValue objects
    lab_values = [
        LabValue(
            date=lab["date"],
            marker=lab["marker"],
            value=lab["value"],
            unit=lab["unit"],
            ref_low=lab.get("ref_low"),
            ref_high=lab.get("ref_high"),
        )
        for lab in labs
    ]

    history_values = [
        LabValue(
            date=h["date"],
            marker=h["marker"],
            value=h["value"],
            unit=h["unit"],
        )
        for h in (history or [])
    ]

    # Create a temporary GoldenCase
    temp_case = GoldenCase(
        id="manual-test",
        description="Manual test case",
        member_id="test",
        query=query,
        labs=lab_values,
        history=history_values,
        symptoms=symptoms or [],
    )

    return run_agent(temp_case, model=model)


# ---------------------------------------------------------------------------
# LEARNING EXERCISE: Try the agent
# ---------------------------------------------------------------------------
#
# Run this file directly to test the agent:
#
# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     load_dotenv()
#
#     from agent_eval_pipeline.golden_sets.thyroid_cases import get_case_by_id
#
#     case = get_case_by_id("thyroid-001")
#     result = run_agent(case)
#
#     if isinstance(result, AgentResult):
#         print("SUCCESS!")
#         print(f"Summary: {result.output.summary}")
#         print(f"Latency: {result.latency_ms:.0f}ms")
#         print(f"Tokens: {result.total_tokens}")
#         for insight in result.output.key_insights:
#             print(f"  - {insight.marker}: {insight.status} ({insight.trend})")
#     else:
#         print(f"ERROR: {result.error_type}: {result.error_message}")
