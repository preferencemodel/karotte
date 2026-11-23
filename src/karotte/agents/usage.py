from typing import Any

from karotte.schemas.transcript import TokenUsageEvent, UsageSource

MESSAGES_USAGE_BUCKETS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def messages_usage_event(
    usage: dict[str, Any], source: UsageSource = "agent"
) -> TokenUsageEvent:
    """Messages-API usage keeps uncached input, cache reads and cache writes in
    disjoint buckets; the transcript's ``input_tokens`` is the full prompt."""
    cache_read = int_or_zero(usage.get("cache_read_input_tokens"))
    cache_write = int_or_zero(usage.get("cache_creation_input_tokens"))
    return TokenUsageEvent(
        input_tokens=int_or_zero(usage.get("input_tokens")) + cache_read + cache_write,
        output_tokens=int_or_zero(usage.get("output_tokens")),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        source=source,
    )


def int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and value > 0 else 0
