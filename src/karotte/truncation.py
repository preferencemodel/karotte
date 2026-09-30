import json
from typing import Any

from karotte.text import escape_surrogates_deep

MB = 1024 * 1024

MAX_METADATA_VALUE_CHARS = 1 * MB
MAX_METADATA_TOTAL_CHARS = 4 * MB
_TAIL_KEEP = 8192


def truncate_middle(text: str, max_chars: int) -> str:
    """Cap text to roughly max_chars, keeping head and tail around a notice."""
    if len(text) <= max_chars:
        return text
    tail_keep = min(_TAIL_KEEP, max_chars // 4)
    notice = f"\n[... truncated: exceeded {max_chars} characters ...]\n"
    tail = text[len(text) - tail_keep :] if tail_keep else ""
    return text[: max_chars - tail_keep] + notice + tail


def json_encoded_len(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=False).encode())


def head_within_json_bytes(text: str, max_bytes: int) -> str:
    """A prefix of text whose JSON encoding fits max_bytes."""
    encoded = json_encoded_len(text)
    if encoded <= max_bytes:
        return text
    keep = len(text) * max_bytes // encoded
    while keep > 0 and json_encoded_len(text[:keep]) > max_bytes:
        keep = keep * 9 // 10
    return text[:keep]


def sanitize_metadata(metadata: dict[str, Any]) -> None:
    """Escape surrogates and truncate oversized metadata values in place,
    bounding each value and the total."""
    remaining = MAX_METADATA_TOTAL_CHARS
    for key, value in metadata.items():
        value = metadata[key] = escape_surrogates_deep(value)
        text = value if isinstance(value, str) else str(value)
        budget = min(MAX_METADATA_VALUE_CHARS, remaining)
        if len(text) > budget:
            metadata[key] = truncate_middle(text, budget)
            remaining -= budget
        else:
            remaining -= len(text)
