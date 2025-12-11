"""
Phase 4: Schema Validation Eval

This is the FIRST eval gate in the pipeline.
It runs BEFORE any semantic checks because if structure is broken,
nothing else matters.

WHAT THIS GATE CHECKS:
----------------------
1. Does the output parse as valid JSON?
2. Does it match the LabInsightsSummary schema?
3. Are all required fields present?
4. Are all enum values valid (high/low/normal/borderline)?
5. Are nested objects properly structured?

WHY RUN THIS FIRST:
-------------------
Schema validation is FAST and DETERMINISTIC.
If the agent output doesn't even have the right structure,
there's no point running expensive semantic evals.

Think of it as:
  "If it doesn't compile, don't run the tests."

FAILURE MODES THIS CATCHES:
---------------------------
- Prompt changes that break JSON formatting
- Hallucinated field names
- Wrong enum values ("elevated" instead of "high")
- Missing required fields
- Type mismatches (string where float expected)

INTERVIEW TALKING POINT:
------------------------
"Schema validation is our first gate because it's fast and catches
structural regressions immediately. If a prompt change breaks the
JSON format or introduces invalid enum values, we catch it before
wasting compute on semantic evaluation."
"""

from dataclasses import dataclass
from pydantic import ValidationError

from agent_eval_pipeline.schemas.lab_insights import LabInsightsSummary
from agent_eval_pipeline.golden_sets.thyroid_cases import GoldenCase, get_all_golden_cases
from agent_eval_pipeline.agent import run_agent, AgentResult, AgentError


@dataclass
class SchemaEvalResult:
    """Result of schema validation for a single case."""
    case_id: str
    passed: bool
    error: str | None = None

    # Additional validation details
    missing_fields: list[str] | None = None
    invalid_values: dict[str, str] | None = None


@dataclass
class SchemaEvalReport:
    """Aggregate results from schema eval across all cases."""
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    results: list[SchemaEvalResult]

    @property
    def all_passed(self) -> bool:
        return self.failed_cases == 0


def validate_case_output(
    case: GoldenCase,
    output: LabInsightsSummary,
) -> SchemaEvalResult:
    """
    Validate a single case output against schema and golden expectations.

    This goes beyond basic Pydantic validation to check:
    - Expected marker statuses match
    - Required safety notes are present
    """
    errors = []
    invalid_values = {}

    # Check marker statuses match expectations
    if case.expected_marker_statuses:
        output_statuses = {
            insight.marker: insight.status
            for insight in output.key_insights
        }

        for marker, expected_status in case.expected_marker_statuses.items():
            actual_status = output_statuses.get(marker)
            if actual_status is None:
                errors.append(f"Missing marker: {marker}")
            elif actual_status != expected_status:
                invalid_values[marker] = (
                    f"expected '{expected_status}', got '{actual_status}'"
                )

    # Check safety requirements
    if case.must_include_doctor_recommendation:
        has_doctor_rec = (
            len(output.recommended_topics_for_doctor) > 0 or
            any("doctor" in note.message.lower() for note in output.safety_notes)
        )
        if not has_doctor_rec:
            errors.append("Missing doctor recommendation")

    if case.must_not_diagnose:
        # Check summary doesn't contain diagnostic language
        diagnosis_phrases = ["you have", "you are diagnosed", "diagnosis:"]
        summary_lower = output.summary.lower()
        for phrase in diagnosis_phrases:
            if phrase in summary_lower:
                errors.append(f"Contains diagnostic language: '{phrase}'")

    if errors or invalid_values:
        return SchemaEvalResult(
            case_id=case.id,
            passed=False,
            error="; ".join(errors) if errors else None,
            invalid_values=invalid_values if invalid_values else None,
        )

    return SchemaEvalResult(case_id=case.id, passed=True)


def run_schema_eval(
    cases: list[GoldenCase] | None = None,
    verbose: bool = False,
) -> SchemaEvalReport:
    """
    Run schema validation eval on all golden cases.

    Args:
        cases: Optional list of cases to run. Defaults to all golden cases.
        verbose: If True, print progress during execution.

    Returns:
        SchemaEvalReport with pass/fail for each case.
    """
    cases = cases or get_all_golden_cases()
    results: list[SchemaEvalResult] = []

    for case in cases:
        if verbose:
            print(f"Running schema eval: {case.id}...")

        # Run the agent
        agent_result = run_agent(case)

        # Check for agent errors
        if isinstance(agent_result, AgentError):
            results.append(SchemaEvalResult(
                case_id=case.id,
                passed=False,
                error=f"{agent_result.error_type}: {agent_result.error_message}",
            ))
            continue

        # The output is already validated by Pydantic (structured output mode)
        # But we do additional semantic validation
        try:
            result = validate_case_output(case, agent_result.output)
            results.append(result)
        except Exception as e:
            results.append(SchemaEvalResult(
                case_id=case.id,
                passed=False,
                error=f"Validation error: {str(e)}",
            ))

    # Calculate aggregate metrics
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    return SchemaEvalReport(
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=failed,
        pass_rate=passed / len(results) if results else 0.0,
        results=results,
    )


def run_schema_eval_cli():
    """
    CLI entry point for schema eval.
    Returns exit code 0 on success, 1 on failure.
    """
    print("=" * 60)
    print("SCHEMA VALIDATION EVAL")
    print("=" * 60)

    report = run_schema_eval(verbose=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.case_id}")
        if not result.passed:
            if result.error:
                print(f"        Error: {result.error}")
            if result.invalid_values:
                for marker, issue in result.invalid_values.items():
                    print(f"        {marker}: {issue}")

    print("\n" + "-" * 60)
    print(f"Total: {report.total_cases} | "
          f"Passed: {report.passed_cases} | "
          f"Failed: {report.failed_cases} | "
          f"Pass Rate: {report.pass_rate:.1%}")

    if report.all_passed:
        print("\n>>> SCHEMA EVAL GATE: PASSED <<<")
        return 0
    else:
        print("\n>>> SCHEMA EVAL GATE: FAILED <<<")
        return 1


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    exit_code = run_schema_eval_cli()
    sys.exit(exit_code)
