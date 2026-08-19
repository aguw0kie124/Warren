"""What a model call costs, in one place.

Pulled forward from Module F because E1's gate has to price two graph shapes
against each other, and a gate that estimated dollars with its own private
constants would be the second copy of a number that has to stay right.

**An unpriced model returns None rather than a guess.** A made-up rate is worse
than a null: it produces a plausible figure that no one re-derives, and the
whole reason to record cost is to be able to trust it later.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    """USD per million tokens.

    `cache_write` is the 5-minute-TTL rate (1.25x input); the 1-hour TTL is 2x
    and is not used here. `cache_read` is 0.10x input.
    """

    input: float
    output: float
    cache_write: float
    cache_read: float


# Anthropic first-party API rates. Keyed by the exact model id `settings`
# carries, plus the alias, because the pinned snapshot and the alias resolve to
# the same model today and either may appear in a recorded row.
PRICING: dict[str, Price] = {
    "claude-haiku-4-5-20251001": Price(1.00, 5.00, 1.25, 0.10),
    "claude-haiku-4-5":          Price(1.00, 5.00, 1.25, 0.10),
    "claude-sonnet-5":           Price(3.00, 15.00, 3.75, 0.30),
    "claude-opus-5":             Price(5.00, 25.00, 6.25, 0.50),
}

_PER_MILLION = 1_000_000


def price_for(model: str) -> Price | None:
    price = PRICING.get(model)
    if price is None:
        logger.warning("no pricing for model %r — cost will be recorded as unknown", model)
    return price


def estimate_usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Dollars for one call's token counts, or None for an unpriced model.

    Cache-read and cache-write tokens are billed *instead of* input tokens, not
    on top of them, which is how `usage_metadata` reports them — so they are
    summed as separate terms rather than folded into `input_tokens`.
    """
    price = price_for(model)
    if price is None:
        return None
    return (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * price.cache_write
    ) / _PER_MILLION
