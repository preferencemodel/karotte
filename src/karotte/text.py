from typing import Any


def escape_surrogates(text: str) -> str:
    """Replace lone surrogates — what `surrogateescape` produces for filesystem
    bytes that are not valid UTF-8 — with their backslash escapes, so the text
    survives JSON serialization."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def escape_surrogates_deep(value: Any) -> Any:
    """`escape_surrogates` applied to every string nested in `value`."""
    if isinstance(value, str):
        return escape_surrogates(value)
    if isinstance(value, dict):
        typed: dict[Any, Any] = value
        return {
            escape_surrogates_deep(k): escape_surrogates_deep(v)
            for k, v in typed.items()
        }
    if isinstance(value, (list, tuple)):
        items: list[Any] | tuple[Any, ...] = value
        return [escape_surrogates_deep(item) for item in items]
    return value
