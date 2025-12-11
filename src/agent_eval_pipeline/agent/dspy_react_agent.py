"""
DSPy ReAct Agent - Tool-using agent pattern

This demonstrates DSPy's ReAct module, which implements the
"Reasoning and Acting" paradigm for tool-using agents.

REACT PATTERN:
--------------
1. THOUGHT: Reason about current situation
2. ACTION: Decide which tool to call
3. OBSERVATION: See tool result
4. REPEAT until task complete

WHY REACT:
----------
- Explicit reasoning traces make debugging easier
- Tool selection is principled, not random
- Can handle multi-step tasks that require gathering information
- The reasoning is optimizable with DSPy

INTERVIEW TALKING POINT:
------------------------
"I implemented a ReAct agent for lab analysis that can use tools -
looking up reference ranges, checking drug interactions, searching
medical literature. DSPy's ReAct module handles the reasoning loop
automatically. I just define the tools and signature, and it figures
out when to call what. The reasoning traces are great for debugging
and explainability."

USE CASE:
---------
The lab insights task sometimes needs external information:
- Look up reference ranges for unusual markers
- Check if medications affect lab values
- Search for condition-specific interpretations

A ReAct agent can decide when it needs more info vs when it can answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import dspy


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------
# Tools are functions the agent can call to gather information.
# DSPy automatically generates tool descriptions from docstrings.


def lookup_reference_range(marker: str) -> str:
    """
    Look up the standard reference range for a lab marker.

    Args:
        marker: The lab marker name (e.g., "TSH", "Hemoglobin")

    Returns:
        Reference range information including units and notes
    """
    # In production, this would query a medical database
    reference_db = {
        "TSH": "0.4-4.0 mIU/L. Higher in elderly. Lower in pregnancy.",
        "Free T4": "0.8-1.8 ng/dL. May vary by assay method.",
        "Free T3": "2.3-4.2 pg/mL. Less commonly tested than T4.",
        "TPO Antibodies": "<35 IU/mL. Elevated in autoimmune thyroid disease.",
        "Hemoglobin": "Men: 13.5-17.5 g/dL, Women: 12.0-16.0 g/dL",
        "Ferritin": "Men: 24-336 ng/mL, Women: 11-307 ng/mL. Lower in iron deficiency.",
        "Vitamin D": "30-100 ng/mL. <20 is deficient, 20-29 is insufficient.",
        "B12": "200-900 pg/mL. Symptoms may occur even in 'normal' range.",
        "HbA1c": "<5.7% normal, 5.7-6.4% prediabetes, >=6.5% diabetes.",
        "Cortisol (AM)": "6-23 mcg/dL. Should be tested in early morning.",
    }

    marker_upper = marker.upper().replace(" ", "")
    for key, value in reference_db.items():
        if key.upper().replace(" ", "") == marker_upper:
            return f"{key}: {value}"

    return f"Reference range for '{marker}' not found in database. Consult lab-specific ranges."


def check_medication_interaction(marker: str, medication: str) -> str:
    """
    Check if a medication can affect a lab marker value.

    Args:
        marker: The lab marker name
        medication: The medication name

    Returns:
        Information about how the medication may affect the marker
    """
    # In production, this would query a drug interaction database
    interactions = {
        ("tsh", "levothyroxine"): "Levothyroxine treats hypothyroidism. TSH should decrease with proper dosing. Test 6-8 weeks after dose change.",
        ("tsh", "biotin"): "Biotin (B7) can interfere with TSH assays, causing falsely low TSH readings. Stop biotin 2-3 days before testing.",
        ("tsh", "prednisone"): "Corticosteroids can suppress TSH. May see lower values during treatment.",
        ("cortisol", "prednisone"): "Exogenous steroids suppress natural cortisol production. Results may be misleading.",
        ("potassium", "lisinopril"): "ACE inhibitors can increase potassium. Monitor for hyperkalemia.",
        ("glucose", "metformin"): "Metformin lowers blood glucose. Fasting glucose should improve with treatment.",
        ("hba1c", "metformin"): "Metformin improves glycemic control. HbA1c should decrease over 2-3 months.",
    }

    key = (marker.lower(), medication.lower())
    if key in interactions:
        return interactions[key]

    return f"No known significant interaction between {medication} and {marker} levels in our database."


def search_medical_context(query: str) -> str:
    """
    Search for medical context related to lab interpretation.

    Args:
        query: Search query about lab values or conditions

    Returns:
        Relevant medical context and educational information
    """
    # In production, this would search a medical knowledge base or RAG system
    contexts = {
        "thyroid": """
Thyroid Function Overview:
- TSH is the primary screening test. High TSH suggests hypothyroidism, low suggests hyperthyroidism.
- Free T4 and Free T3 measure actual hormone levels.
- Pattern: High TSH + Low T4 = Primary hypothyroidism
- Pattern: Low TSH + High T4 = Hyperthyroidism
- Subclinical disease: Abnormal TSH with normal T4/T3
- TPO antibodies indicate autoimmune thyroid disease (Hashimoto's or Graves')
""",
        "fatigue": """
Common Lab Causes of Fatigue:
- Thyroid dysfunction (check TSH, Free T4)
- Anemia (check CBC, ferritin, iron studies)
- Vitamin deficiencies (B12, D, folate)
- Blood sugar issues (fasting glucose, HbA1c)
- Kidney or liver dysfunction (BMP, CMP)
- Consider cortisol if other tests normal
""",
        "diabetes": """
Diabetes Markers:
- Fasting glucose: Normal <100, Prediabetes 100-125, Diabetes >=126 mg/dL
- HbA1c: Normal <5.7%, Prediabetes 5.7-6.4%, Diabetes >=6.5%
- HbA1c reflects 2-3 month average glucose
- Target HbA1c for most diabetics: <7%
- Consider checking kidney function (eGFR, microalbumin) in diabetics
""",
        "anemia": """
Anemia Workup:
- Hemoglobin/Hematocrit: Low indicates anemia
- MCV: Low = microcytic (iron deficiency), High = macrocytic (B12/folate)
- Ferritin: Iron stores. Low = iron deficiency
- B12 and Folate: Check if MCV elevated
- Reticulocyte count: Low = production problem, High = destruction/loss
- Iron, TIBC: Full iron studies if ferritin equivocal
""",
    }

    query_lower = query.lower()
    for keyword, context in contexts.items():
        if keyword in query_lower:
            return context

    return "No specific context found. Consider consulting clinical references for interpretation."


# ---------------------------------------------------------------------------
# REACT SIGNATURE
# ---------------------------------------------------------------------------


class LabAnalysisSignature(dspy.Signature):
    """
    Analyze lab results and provide educational health insights.

    You have access to tools for looking up reference ranges, checking
    medication interactions, and searching medical context. Use them
    when you need additional information to provide accurate insights.

    Always recommend consulting a healthcare provider. Never diagnose
    conditions or recommend specific treatments.
    """

    query: str = dspy.InputField(desc="User's question about their lab results")
    labs: str = dspy.InputField(desc="Lab values with their results")
    medications: str = dspy.InputField(desc="Current medications, if any")
    symptoms: str = dspy.InputField(desc="Reported symptoms, if any")

    analysis: str = dspy.OutputField(
        desc="Comprehensive analysis including: summary, key findings, "
             "what to discuss with doctor, and lifestyle considerations. "
             "Include appropriate disclaimers."
    )


# ---------------------------------------------------------------------------
# REACT AGENT
# ---------------------------------------------------------------------------


class LabReActAgent(dspy.Module):
    """
    ReAct agent for lab analysis with tool use.

    This agent can:
    1. Look up reference ranges for unfamiliar markers
    2. Check medication interactions
    3. Search for relevant medical context
    4. Reason through complex multi-marker patterns

    INTERVIEW TALKING POINT:
    "The ReAct agent explicitly reasons before each action. When analyzing
    complex labs, it might first search for thyroid context, then check
    if the patient's levothyroxine affects TSH, then synthesize everything.
    Each reasoning step is logged, making it easy to debug and explain."
    """

    def __init__(self):
        super().__init__()

        # Define tools the agent can use
        tools = [
            lookup_reference_range,
            check_medication_interaction,
            search_medical_context,
        ]

        # Create ReAct module with our signature and tools
        self.react = dspy.ReAct(
            signature=LabAnalysisSignature,
            tools=tools,
            max_iters=5,  # Limit reasoning steps
        )

    def forward(
        self,
        query: str,
        labs: list[dict],
        medications: list[str] | None = None,
        symptoms: list[str] | None = None,
    ) -> dspy.Prediction:
        """
        Run the ReAct agent.

        Args:
            query: User's question
            labs: Lab values as list of dicts
            medications: Optional list of current medications
            symptoms: Optional list of symptoms

        Returns:
            Prediction with analysis and reasoning trace
        """
        # Format inputs
        labs_text = self._format_labs(labs)
        meds_text = ", ".join(medications) if medications else "None reported"
        symptoms_text = ", ".join(symptoms) if symptoms else "None reported"

        # Run ReAct
        result = self.react(
            query=query,
            labs=labs_text,
            medications=meds_text,
            symptoms=symptoms_text,
        )

        return result

    def _format_labs(self, labs: list[dict]) -> str:
        """Format lab values for the prompt."""
        lines = []
        for lab in labs:
            ref = ""
            if lab.get("ref_low") and lab.get("ref_high"):
                ref = f" (ref: {lab['ref_low']}-{lab['ref_high']})"
            lines.append(f"- {lab['marker']}: {lab['value']} {lab['unit']}{ref}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RESULT DATACLASS
# ---------------------------------------------------------------------------


@dataclass
class ReActAgentResult:
    """Result from ReAct agent including reasoning trace."""

    analysis: str
    tools_used: list[str]
    reasoning_steps: int

    def __str__(self):
        return f"""
ReAct Agent Result
==================
Tools Used: {', '.join(self.tools_used) or 'None'}
Reasoning Steps: {self.reasoning_steps}

Analysis:
{self.analysis}
"""


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def create_react_agent(model: str = "gpt-4o-mini") -> LabReActAgent:
    """
    Create a ReAct-based lab analysis agent.

    Args:
        model: OpenAI model to use

    Returns:
        Configured LabReActAgent
    """
    lm = dspy.LM(
        f"openai/{model}",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    dspy.configure(lm=lm)

    return LabReActAgent()


def run_react_agent(
    query: str,
    labs: list[dict],
    medications: list[str] | None = None,
    symptoms: list[str] | None = None,
    model: str = "gpt-4o-mini",
) -> ReActAgentResult:
    """
    Run the ReAct agent on lab analysis.

    Example:
        >>> result = run_react_agent(
        ...     query="I'm on levothyroxine. What do my thyroid results mean?",
        ...     labs=[
        ...         {"marker": "TSH", "value": 2.5, "unit": "mIU/L", "ref_low": 0.4, "ref_high": 4.0},
        ...         {"marker": "Free T4", "value": 1.2, "unit": "ng/dL", "ref_low": 0.8, "ref_high": 1.8},
        ...     ],
        ...     medications=["levothyroxine 50mcg"],
        ...     symptoms=["mild fatigue"]
        ... )
        >>> print(result)

    INTERVIEW TALKING POINT:
    "The ReAct agent decides what tools to use based on the query. If
    someone mentions a medication, it automatically checks for interactions.
    If there's an unusual marker, it looks up the reference range. The
    reasoning is explicit and traceable."
    """
    agent = create_react_agent(model=model)

    prediction = agent(
        query=query,
        labs=labs,
        medications=medications,
        symptoms=symptoms,
    )

    # Extract tools used from trajectory (if available)
    tools_used = []
    reasoning_steps = 0

    # DSPy ReAct stores trajectory in the prediction
    if hasattr(prediction, 'trajectory'):
        trajectory = prediction.trajectory
        # Count tool calls in trajectory
        for step in trajectory.split('\n'):
            if 'Tool:' in step or 'Action:' in step:
                reasoning_steps += 1
            for tool_name in ['lookup_reference_range', 'check_medication_interaction', 'search_medical_context']:
                if tool_name in step and tool_name not in tools_used:
                    tools_used.append(tool_name)

    return ReActAgentResult(
        analysis=prediction.analysis,
        tools_used=tools_used,
        reasoning_steps=reasoning_steps,
    )


# ---------------------------------------------------------------------------
# COMPARISON: SIMPLE VS REACT
# ---------------------------------------------------------------------------


def compare_simple_vs_react(
    query: str,
    labs: list[dict],
    medications: list[str] | None = None,
    symptoms: list[str] | None = None,
):
    """
    Compare simple DSPy agent vs ReAct agent.

    Useful for demonstrating when tool use adds value.
    """
    from agent_eval_pipeline.agent.dspy_agent import run_dspy_agent

    print("=" * 60)
    print("COMPARISON: Simple DSPy vs ReAct Agent")
    print("=" * 60)

    print(f"\nQuery: {query}")
    print(f"Labs: {labs}")
    print(f"Medications: {medications}")
    print(f"Symptoms: {symptoms}")

    # Simple agent
    print("\n" + "-" * 40)
    print("[Simple DSPy Agent]")
    print("-" * 40)
    try:
        simple_result = run_dspy_agent(
            query=query,
            labs=labs,
            symptoms=symptoms,
        )
        print(simple_result.output.summary)
    except Exception as e:
        print(f"Error: {e}")

    # ReAct agent
    print("\n" + "-" * 40)
    print("[ReAct Agent with Tools]")
    print("-" * 40)
    try:
        react_result = run_react_agent(
            query=query,
            labs=labs,
            medications=medications,
            symptoms=symptoms,
        )
        print(f"Tools used: {react_result.tools_used}")
        print(f"Analysis:\n{react_result.analysis[:500]}...")
    except Exception as e:
        print(f"Error: {e}")


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Example: Patient on thyroid medication with unusual marker
    result = run_react_agent(
        query="I'm taking levothyroxine and biotin supplements. My TSH came back low - should I be concerned?",
        labs=[
            {"marker": "TSH", "value": 0.2, "unit": "mIU/L", "ref_low": 0.4, "ref_high": 4.0},
            {"marker": "Free T4", "value": 1.1, "unit": "ng/dL", "ref_low": 0.8, "ref_high": 1.8},
        ],
        medications=["levothyroxine 75mcg", "biotin 5000mcg"],
        symptoms=["anxiety", "slight tremor"],
    )

    print(result)
