"""Keep the provider's last raw stream chunks for a blocked turn.

litellm turns every provider block into `content_filter` and throws away the
provider's own reason. To keep it, we wrap the litellm parsers that see the raw
chunks and hold on to the last few of each turn. The wrapper only reads. A parser
that litellm has moved is skipped with a warning.
"""

import copy
import functools
from collections import deque
from contextvars import ContextVar
from importlib import import_module
from typing import Any

from loguru import logger

# litellm classes whose `chunk_parser` receives the provider's raw chunks.
PROVIDER_CHUNK_PARSERS = (
    # Gemini on Vertex AI: block reason, message and safety ratings
    (
        "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini",
        "ModelResponseIterator",
    ),
)

# The last chunk has the block. The two before it give some context.
MAX_KEPT_CHUNKS = 3

_current_turn_chunks: ContextVar[deque[dict[str, Any]] | None] = ContextVar(
    "provider_chunks", default=None
)


def tap_provider_chunks() -> deque[dict[str, Any]]:
    """Start a new turn. The returned deque fills up as the stream is parsed."""
    chunks: deque[dict[str, Any]] = deque(maxlen=MAX_KEPT_CHUNKS)
    _current_turn_chunks.set(chunks)
    return chunks


def _tap_parser(module: str, cls: str) -> bool:
    try:
        iterator = getattr(import_module(module), cls)
        original = iterator.chunk_parser
    except (ImportError, AttributeError) as e:
        logger.warning(
            "Cannot tap provider chunks from {module}.{cls}: {e}",
            module=module,
            cls=cls,
            e=e,
        )
        return False
    if getattr(original, "_taps_provider_chunks", False):
        return True

    # Pass litellm's arguments through as they are.
    @functools.wraps(original)
    def chunk_parser(*args: Any, **kwargs: Any) -> Any:
        chunks = _current_turn_chunks.get()
        chunk = kwargs.get("chunk", args[1] if len(args) > 1 else None)
        if chunks is not None and isinstance(chunk, dict):
            try:
                # Copy it: the parser may change the dict.
                chunks.append(copy.deepcopy(chunk))
            except Exception as e:  # never fail the model call over this
                logger.debug("Could not keep a provider chunk: {e}", e=e)
        return original(*args, **kwargs)

    chunk_parser._taps_provider_chunks = True  # pyright: ignore[reportAttributeAccessIssue]
    iterator.chunk_parser = chunk_parser
    return True


def install_provider_chunk_taps() -> None:
    for module, cls in PROVIDER_CHUNK_PARSERS:
        _ = _tap_parser(module, cls)
