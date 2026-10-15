"""Estimated USD costs from OpenAI's published standard token prices.

Checked 2026-10-13:
https://developers.openai.com/api/docs/models/gpt-6-luna
https://developers.openai.com/api/docs/models/text-embedding-3-small

These estimates use uncached, short-context Standard rates. Cached tokens,
long-context pricing, processing tiers, and regional premiums are not tracked.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional


LLM_PRICES_PER_MILLION = {
    "gpt-6-luna": (Decimal("0.10"), Decimal("0.50")),
}
EMBEDDING_INPUT_PRICES_PER_MILLION = {
    "text-embedding-3-small": Decimal("0.02"),
}

MILLION = Decimal("1000000")
COST_PRECISION = Decimal("0.00000001")  # ai_usage_events.estimated_cost is NUMERIC(12, 8)


def estimate_cost(
    *,
    event_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    request_success: bool,
) -> Optional[Decimal]:
    """Return a token-based estimate, or None when price/usage is unavailable."""
    if not request_success and input_tokens == 0 and output_tokens == 0:
        return None

    if event_type == "llm":
        prices = LLM_PRICES_PER_MILLION.get(model)
        if prices is None:
            return None
        amount = (
            Decimal(input_tokens) * prices[0]
            + Decimal(output_tokens) * prices[1]
        ) / MILLION
    elif event_type == "embedding":
        input_price = EMBEDDING_INPUT_PRICES_PER_MILLION.get(model)
        if input_price is None:
            return None
        amount = Decimal(input_tokens) * input_price / MILLION
    else:
        return None

    return amount.quantize(COST_PRECISION, rounding=ROUND_HALF_UP)
