"""Adapter from Mistral Vibe's ``--output streaming`` NDJSON to transcript events.

Each stdout line is one conversation message (Vibe's ``LLMMessage`` dumped as
JSON): ``role`` is ``system``/``user``/``assistant``/``tool``, assistant
messages carry ``tool_calls`` (``{id, index, function: {name, arguments},
type}``), and a ``role: "tool"`` line carries the result for ``tool_call_id``.
Vibe never writes token usage to stdout (its ``LLMUsage`` stays internal), so
usage is estimated from message content and marked as such.

Parsing is pure and lenient: no I/O, malformed lines are logged and skipped.
The native log is saved as an artifact elsewhere and remains the source of
truth, so the events can always be re-derived from it.
"""

import json
from collections.abc import Iterable, Iterator
from typing import Any

from litellm import token_counter
from loguru import logger
from mcp.types import CallToolResult, TextContent

from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

_ROLES = frozenset({"system", "user", "assistant", "tool"})

# Native fields that map onto `Message`; the rest of a line goes into the
# event's `raw` passthrough. `message_id`/`reasoning_message_id` are Vibe
# bookkeeping ids present on every line and carry no content, so they are
# not preserved in the normalized transcript.
_MAPPED_FIELDS = frozenset(
    {
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "reasoning_content",
        "message_id",
        "reasoning_message_id",
    }
)


def parse_stream(lines: Iterable[str], model: str = "") -> Iterator[Event]:
    """Convert Vibe NDJSON lines into normalized transcript events.

    Yields, per line: a `MessageAddedEvent` for every message; after each
    assistant message an estimated `TokenUsageEvent` and one
    `ToolCallStartedEvent` per tool call; before each tool-result message the
    matching `ToolCallCompletedEvent`. `model` selects the tokenizer for the
    usage estimate.
    """
    input_tokens = 0
    synthetic_id_count = 0

    for i, line in enumerate(lines):
        record = _parse_line(line, i)
        if record is None:
            continue

        role = record.get("role")
        if role not in _ROLES:
            logger.warning("Skipping vibe NDJSON line {} with role {!r}.", i, role)
            continue

        tool_calls: list[ChatCompletionMessageToolCall] = []
        for element in record.get("tool_calls") or []:
            tool_call = _parse_tool_call(element, i)
            if tool_call is None:
                continue
            if not tool_call.id:
                synthetic_id_count += 1
                tool_call.id = f"vibe-tool-call-{synthetic_id_count}"
            tool_calls.append(tool_call)

        message = Message(
            role=role,
            content=record.get("content"),
            tool_calls=tool_calls or None,
            reasoning_content=record.get("reasoning_content") or None,
            tool_call_id=record.get("tool_call_id"),
        )
        raw = {
            key: value
            for key, value in record.items()
            if key not in _MAPPED_FIELDS and value is not None and value is not False
        }

        if role == "tool":
            tool_call_id = record.get("tool_call_id")
            if tool_call_id:
                yield ToolCallCompletedEvent(
                    tool_call_id=tool_call_id,
                    result=CallToolResult(
                        content=[TextContent(type="text", text=message.content or "")]
                        if isinstance(message.content, str)
                        else []
                    ),
                )
            else:
                logger.warning("Vibe tool result on line {} has no tool_call_id.", i)

        yield MessageAddedEvent(message=message, raw=raw or None)

        message_tokens = _count_tokens(model, _message_text(record))
        if role == "assistant":
            yield TokenUsageEvent(
                input_tokens=input_tokens,
                output_tokens=message_tokens,
                estimated=True,
            )
        input_tokens += message_tokens

        for tool_call in tool_calls:
            yield ToolCallStartedEvent(tool_call=tool_call)


def _parse_line(line: str, i: int) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Skipping unparseable vibe NDJSON line {}: {!r}", i, line[:200])
        return None
    if not isinstance(record, dict):
        logger.warning("Skipping non-object vibe NDJSON line {}.", i)
        return None
    return record


def _parse_tool_call(element: Any, i: int) -> ChatCompletionMessageToolCall | None:
    if not isinstance(element, dict):
        logger.warning("Skipping malformed tool call on vibe NDJSON line {}.", i)
        return None
    function = element.get("function")
    if not isinstance(function, dict):
        function = {}
    return ChatCompletionMessageToolCall(
        id=element.get("id") or "",
        function=Function(
            name=function.get("name"),
            arguments=function.get("arguments") or "",
        ),
        type=element.get("type") or "function",
    )


def _message_text(record: dict[str, Any]) -> str:
    """All model-produced text of a message, for token estimation."""
    parts: list[str] = []
    for key in ("content", "reasoning_content"):
        value = record.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    for element in record.get("tool_calls") or []:
        if isinstance(element, dict) and isinstance(element.get("function"), dict):
            function = element["function"]
            parts.append(str(function.get("name") or ""))
            parts.append(str(function.get("arguments") or ""))
    return "\n".join(parts)


def _count_tokens(model: str, text: str) -> int:
    if not text:
        return 0
    try:
        return token_counter(model=model, text=text)
    except Exception:
        # litellm falls back to a default tokenizer for unknown models but can
        # still fail (e.g. empty model string); estimate from length instead.
        return max(1, len(text) // 4)
