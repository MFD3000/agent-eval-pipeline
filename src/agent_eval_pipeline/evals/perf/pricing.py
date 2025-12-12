"""
Cost estimation - model pricing and cost calculation.

Pricing is externalized so it can be:
1. Updated independently when OpenAI changes prices
2. Extended for new models
3. Used for cost projections
"""

# ---------------------------------------------------------------------------
# MODEL PRICING
# ---------------------------------------------------------------------------
# Approximate pricing per 1M tokens (as of 2024)
# Source: https://openai.com/pricing

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

# Default fallback pricing (use cheapest model pricing)
DEFAULT_PRICING = MODEL_PRICING["gpt-4o-mini"]


# ---------------------------------------------------------------------------
# COST ESTIMATION
# ---------------------------------------------------------------------------


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """
    Estimate cost in USD for a given token usage.

    This is a PURE FUNCTION - no side effects.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name for pricing lookup

    Returns:
        Estimated cost in USD

    Example:
        >>> estimate_cost(1000, 500, "gpt-4o-mini")
        0.00045  # ($0.15/1M * 1000) + ($0.60/1M * 500)
    """
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def estimate_monthly_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    daily_requests: int,
) -> float:
    """
    Estimate monthly cost for a usage pattern.

    Useful for capacity planning and cost projections.

    Args:
        input_tokens: Tokens per request (input)
        output_tokens: Tokens per request (output)
        model: Model name
        daily_requests: Expected daily request volume

    Returns:
        Estimated monthly cost in USD (30 days)
    """
    per_request = estimate_cost(input_tokens, output_tokens, model)
    return per_request * daily_requests * 30
