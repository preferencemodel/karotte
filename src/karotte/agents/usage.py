from typing import Any

from karotte.schemas.transcript import TokenUsageEvent


def messages_usage_event(usage: dict[str, Any]) -> TokenUsageEvent:
    """Messages-API usage keeps uncached input, cache reads and cache writes in
    disjoint buckets; the transcript's ``input_tokens`` is the full prompt."""
    cache_read = _int(usage.get("cache_read_input_tokens"))
    cache_write = _int(usage.get("cache_creation_input_tokens"))
    return TokenUsageEvent(
        input_tokens=_int(usage.get("input_tokens")) + cache_read + cache_write,
        output_tokens=_int(usage.get("output_tokens")),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0
