"""
DSPy-based Lab Insights Agent

This module reimplements the lab insights agent using DSPy patterns.
It demonstrates the key DSPy concepts:
1. Signatures - Declarative I/O specifications
2. Modules - Composable building blocks
3. ChainOfThought - Reasoning before answering
4. Assertions - Runtime constraints
5. Optimizers - Automatic prompt tuning

INTERVIEW TALKING POINT:
------------------------
"I reimplemented our lab insights agent using DSPy to understand the paradigm.
The key insight is that DSPy separates WHAT you want (signatures) from HOW
to get it (optimizers). Instead of hand-tuning prompts, I define the task
structure and let the optimizer find effective prompts from examples."

WHY DSPY FOR THIS USE CASE:
---------------------------
1. STRUCTURED OUTPUT: DSPy signatures naturally map to our Pydantic schemas
2. SAFETY CONSTRAINTS: dspy.Assert can enforce safety rules at runtime
3. OPTIMIZATION: We can auto-tune prompts using our golden cases
4. MODULARITY: Each step (retrieve, analyze, safety) is a composable module
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import dspy

from agent_eval_pipeline.schemas.lab_insights import (
    LabInsightsSummary,
    MarkerInsight,
    SafetyNote,
)


# ---------------------------------------------------------------------------
# DSPY SIGNATURES
# ---------------------------------------------------------------------------
# Signatures declare WHAT we want - input fields and output fields.
# DSPy handles HOW to prompt the LLM.


class AnalyzeLabs(dspy.Signature):
    """
    Analyze lab results and provide health insights.

    You are a health education assistant. Analyze the lab values and provide
    educational information. Do NOT diagnose conditions or recommend treatments.
    Always recommend consulting a healthcare provider.
    """

    # Input fields
    query: str = dspy.InputField(desc="User's question about their lab results")
    labs: str = dspy.InputField(desc="Current lab values with reference ranges")
    history: str = dspy.InputField(desc="Historical lab values for trend analysis")
    symptoms: str = dspy.InputField(desc="User-reported symptoms, if any")
    context: str = dspy.InputField(desc="Retrieved medical knowledge context")

    # Output fields
    summary: str = dspy.OutputField(desc="2-3 sentence summary of key findings")
    insights_json: str = dspy.OutputField(
        desc="JSON array of marker insights with fields: marker, status (normal/low/high/borderline), "
             "value, unit, ref_range, trend (stable/increasing/decreasing), clinical_relevance, action"
    )
    doctor_topics: str = dspy.OutputField(desc="Comma-separated topics to discuss with doctor")
    lifestyle: str = dspy.OutputField(desc="Comma-separated lifestyle considerations")


class SafetyCheck(dspy.Signature):
    """
    Review health content for safety compliance.

    Ensure the response:
    1. Does NOT diagnose any condition
    2. Does NOT recommend specific medications
    3. DOES recommend consulting a healthcare provider
    4. Includes appropriate disclaimers
    """

    content: str = dspy.InputField(desc="The health content to review")

    is_safe: bool = dspy.OutputField(desc="True if content passes all safety checks")
    issues: str = dspy.OutputField(desc="Any safety issues found, or 'None'")
    disclaimer: str = dspy.OutputField(
        desc="Appropriate disclaimer to add, e.g., 'This is for educational purposes only...'"
    )


class ExtractMarkers(dspy.Signature):
    """Extract marker names from lab values for retrieval."""

    labs: str = dspy.InputField(desc="Lab values text")
    markers: str = dspy.OutputField(desc="Comma-separated list of marker names (e.g., TSH, T4, T3)")


# ---------------------------------------------------------------------------
# DSPY MODULES
# ---------------------------------------------------------------------------
# Modules are composable building blocks that use signatures.


class LabInsightsModule(dspy.Module):
    """
    DSPy module for generating lab insights.

    This demonstrates DSPy's module composition pattern:
    - extract_markers: Simple Predict for marker extraction
    - analyze: ChainOfThought for reasoning through analysis
    - safety_check: Predict for safety validation

    INTERVIEW TALKING POINT:
    "The module composes three DSPy predictors. ChainOfThought on the
    analysis step means the LLM reasons through its interpretation before
    generating output. I can swap Predict for ChainOfThought on any step
    to add explicit reasoning."
    """

    def __init__(self, retriever=None):
        super().__init__()

        # Simple extraction (no reasoning needed)
        self.extract_markers = dspy.Predict(ExtractMarkers)

        # ChainOfThought for analysis - adds explicit reasoning
        self.analyze = dspy.ChainOfThought(AnalyzeLabs)

        # Safety validation
        self.safety_check = dspy.Predict(SafetyCheck)

        # Optional retriever for RAG
        self._retriever = retriever

    def forward(
        self,
        query: str,
        labs: list[dict],
        history: list[dict] | None = None,
        symptoms: list[str] | None = None,
    ) -> dspy.Prediction:
        """
        Run the lab insights pipeline.

        Args:
            query: User's question
            labs: List of lab value dicts
            history: Optional historical values
            symptoms: Optional symptom list

        Returns:
            dspy.Prediction with analysis results
        """
        # Format inputs
        labs_text = self._format_labs(labs)
        history_text = self._format_history(history or [])
        symptoms_text = ", ".join(symptoms) if symptoms else "None reported"

        # Step 1: Extract markers for retrieval
        markers_result = self.extract_markers(labs=labs_text)
        marker_list = [m.strip() for m in markers_result.markers.split(",")]

        # Step 2: Retrieve context (if retriever available)
        if self._retriever:
            context = self._retrieve_context(marker_list)
        else:
            context = "No additional context available."

        # Step 3: Analyze with ChainOfThought (includes reasoning)
        analysis = self.analyze(
            query=query,
            labs=labs_text,
            history=history_text,
            symptoms=symptoms_text,
            context=context,
        )

        # Step 4: Safety check
        content_to_check = f"{analysis.summary}\n{analysis.insights_json}"
        safety = self.safety_check(content=content_to_check)

        # DSPy Assertion: Enforce safety at runtime
        dspy.Assert(
            safety.is_safe,
            f"Safety check failed: {safety.issues}. Content must be revised."
        )

        return dspy.Prediction(
            summary=analysis.summary,
            insights_json=analysis.insights_json,
            doctor_topics=analysis.doctor_topics,
            lifestyle=analysis.lifestyle,
            reasoning=analysis.reasoning,  # From ChainOfThought
            safety_disclaimer=safety.disclaimer,
            markers_extracted=marker_list,
        )

    def _format_labs(self, labs: list[dict]) -> str:
        """Format lab values for the prompt."""
        lines = []
        for lab in labs:
            ref = f"(ref: {lab.get('ref_low', '?')}-{lab.get('ref_high', '?')})"
            lines.append(f"- {lab['marker']}: {lab['value']} {lab['unit']} {ref}")
        return "\n".join(lines)

    def _format_history(self, history: list[dict]) -> str:
        """Format historical values for the prompt."""
        if not history:
            return "No historical data available."
        lines = []
        for h in history:
            lines.append(f"- {h['marker']}: {h['value']} {h['unit']} [{h.get('date', 'unknown')}]")
        return "\n".join(lines)

    def _retrieve_context(self, markers: list[str]) -> str:
        """Retrieve context from vector store."""
        try:
            results = self._retriever.search_by_markers(markers, limit=3)
            return "\n\n".join(r.content for r in results)
        except Exception:
            return "No additional context available."


# ---------------------------------------------------------------------------
# RESULT CONVERSION
# ---------------------------------------------------------------------------


@dataclass
class DSPyAgentResult:
    """Result from the DSPy agent."""

    output: LabInsightsSummary
    reasoning: str
    markers_extracted: list[str]


def parse_dspy_result(prediction: dspy.Prediction) -> DSPyAgentResult:
    """
    Convert DSPy prediction to our schema.

    This bridges DSPy's flexible output with our strict Pydantic models.
    """
    import json

    # Parse insights JSON
    try:
        insights_data = json.loads(prediction.insights_json)
    except json.JSONDecodeError:
        insights_data = []

    # Convert to MarkerInsight objects
    key_insights = []
    for item in insights_data:
        try:
            insight = MarkerInsight(
                marker=item.get("marker", "Unknown"),
                status=item.get("status", "normal"),
                value=float(item.get("value", 0)),
                unit=item.get("unit", ""),
                ref_range=item.get("ref_range", ""),
                trend=item.get("trend", "stable"),
                clinical_relevance=item.get("clinical_relevance", ""),
                action=item.get("action", "Discuss with healthcare provider"),
            )
            key_insights.append(insight)
        except Exception:
            continue

    # Parse comma-separated fields
    doctor_topics = [t.strip() for t in prediction.doctor_topics.split(",") if t.strip()]
    lifestyle = [l.strip() for l in prediction.lifestyle.split(",") if l.strip()]

    # Build output
    output = LabInsightsSummary(
        summary=prediction.summary,
        key_insights=key_insights,
        recommended_topics_for_doctor=doctor_topics or ["Review these results with your doctor"],
        lifestyle_considerations=lifestyle or [],
        safety_notes=[
            SafetyNote(
                message=prediction.safety_disclaimer,
                type="non_diagnostic",
            )
        ],
    )

    return DSPyAgentResult(
        output=output,
        reasoning=prediction.reasoning,
        markers_extracted=prediction.markers_extracted,
    )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def create_dspy_agent(
    model: str = "gpt-4o-mini",
    retriever=None,
) -> LabInsightsModule:
    """
    Create a DSPy-based lab insights agent.

    Args:
        model: OpenAI model to use
        retriever: Optional VectorStore for RAG

    Returns:
        Configured LabInsightsModule

    Example:
        >>> agent = create_dspy_agent()
        >>> result = agent(
        ...     query="What do my thyroid results mean?",
        ...     labs=[{"marker": "TSH", "value": 5.5, "unit": "mIU/L", ...}]
        ... )
        >>> print(result.summary)
    """
    # Configure DSPy with OpenAI
    lm = dspy.LM(
        f"openai/{model}",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    dspy.configure(lm=lm)

    return LabInsightsModule(retriever=retriever)


def run_dspy_agent(
    query: str,
    labs: list[dict],
    history: list[dict] | None = None,
    symptoms: list[str] | None = None,
    model: str = "gpt-4o-mini",
    retriever=None,
) -> DSPyAgentResult:
    """
    Run the DSPy agent on inputs.

    This is a convenience function for one-off runs.
    For batch processing, create the agent once with create_dspy_agent().

    INTERVIEW TALKING POINT:
    "The DSPy agent uses the same interface as our LangGraph agent,
    but the implementation is declarative. I define signatures for
    what I want, and DSPy handles the prompting. If I want to optimize,
    I compile with examples and DSPy finds better prompts automatically."
    """
    agent = create_dspy_agent(model=model, retriever=retriever)

    prediction = agent(
        query=query,
        labs=labs,
        history=history,
        symptoms=symptoms,
    )

    return parse_dspy_result(prediction)


# ---------------------------------------------------------------------------
# OPTIMIZATION
# ---------------------------------------------------------------------------
# This is where DSPy really shines - automatic prompt optimization


def optimize_agent(
    agent: LabInsightsModule,
    trainset: list,
    metric=None,
    optimizer_type: Literal["bootstrap", "mipro"] = "bootstrap",
) -> LabInsightsModule:
    """
    Optimize the agent using DSPy optimizers.

    This automatically tunes prompts based on examples and a metric.

    Args:
        agent: The LabInsightsModule to optimize
        trainset: List of dspy.Example objects with inputs and expected outputs
        metric: Function(example, prediction) -> float score
        optimizer_type: "bootstrap" (fast) or "mipro" (thorough)

    Returns:
        Optimized agent with tuned prompts

    INTERVIEW TALKING POINT:
    "DSPy optimizers are the key differentiator. Instead of manually
    iterating on prompts, I define a metric - in our case, the judge
    score - and let the optimizer find prompts that maximize it.
    BootstrapFewShot adds examples to the prompt. MIPROv2 also
    optimizes the instructions themselves."
    """
    if metric is None:
        # Default metric: check if output is safe and has insights
        def metric(example, prediction, trace=None):
            try:
                result = parse_dspy_result(prediction)
                has_insights = len(result.output.key_insights) > 0
                has_summary = len(result.output.summary) > 20
                has_topics = len(result.output.recommended_topics_for_doctor) > 0
                return float(has_insights and has_summary and has_topics)
            except Exception:
                return 0.0

    if optimizer_type == "bootstrap":
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
        )
    else:
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto="light",  # "light", "medium", or "heavy"
            num_threads=4,
        )

    optimized = optimizer.compile(agent, trainset=trainset)
    return optimized


def golden_cases_to_trainset(cases) -> list:
    """
    Convert our golden cases to DSPy training examples.

    This bridges our evaluation infrastructure with DSPy's optimization.
    """
    trainset = []

    for case in cases:
        example = dspy.Example(
            query=case.query,
            labs=[
                {
                    "marker": lab.marker,
                    "value": lab.value,
                    "unit": lab.unit,
                    "ref_low": lab.ref_low,
                    "ref_high": lab.ref_high,
                }
                for lab in case.labs
            ],
            history=[
                {
                    "marker": h.marker,
                    "value": h.value,
                    "unit": h.unit,
                    "date": h.date,
                }
                for h in case.history
            ],
            symptoms=case.symptoms,
            # Expected semantic points become hints for optimization
            expected_points=case.expected_semantic_points,
        ).with_inputs("query", "labs", "history", "symptoms")

        trainset.append(example)

    return trainset
