"""Adapter from Grok Build's ``--output-format streaming-messages-json`` NDJSON
to transcript events.

Each stdout line is one object tagged by ``type``: a ``system`` preamble
(``subtype: "init"``), ``assistant`` lines whose ``message.content[]`` holds
``text``/``thinking``/``tool_use`` blocks plus the response's ``usage``,
``user`` lines carrying ``tool_result`` blocks, and a terminal ``result``.
Unlike Vibe, Grok reports real token usage per model response.

Lines with a non-null ``parent_tool_use_id`` come from subagents; their tool
calls are not part of the main conversation, so they are left out of the
transcript (the native log, saved as an artifact, keeps them).

Parsing is pure and lenient: no I/O, malformed lines are logged and skipped.
"""

import json
from collections.abc import Iterable, Iterator
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, ImageContent, TextContent

from karotte.agents.tool_results import tool_result_events
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)


def parse_stream(lines: Iterable[str]) -> Iterator[Event]:
    """Convert Grok NDJSON lines into normalized transcript events.

    Lines are independent, so a caller reading Grok's stdout as it arrives can
    feed each one to :func:`parse_line` instead.
    """
    for i, line in enumerate(lines):
        yield from parse_line(line, i)


def parse_line(line: str, i: int) -> Iterator[Event]:
    """Events for line ``i`` of Grok's NDJSON output.

    An ``assistant`` line yields a `MessageAddedEvent`, a `TokenUsageEvent`
    and one `ToolCallStartedEvent` per tool call. Each ``tool_result`` block of
    a ``user`` line yields a `ToolCallCompletedEvent` and the tool
    `MessageAddedEvent`. ``system`` and ``result`` lines carry only metadata
    and yield nothing.
    """
    record = _parse_line(line, i)
    if record is None:
        return
    if record.get("parent_tool_use_id") is not None:
        return

    kind = record.get("type")
    message = record.get("message")
    if kind not in ("assistant", "user"):
        return
    if not isinstance(message, dict):
        logger.warning("Skipping grok {} line {} without a message.", kind, i)
        return
    blocks = message.get("content")
    if isinstance(blocks, str):
        blocks = [{"type": "text", "text": blocks}]
    if not isinstance(blocks, list):
        blocks = []

    if kind == "assistant":
        yield from _assistant_events(blocks, message.get("usage"), i)
    else:
        yield from _user_events(blocks)


def _assistant_events(blocks: list[Any], usage: Any, i: int) -> Iterator[Event]:
    texts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[ChatCompletionMessageToolCall] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        match block.get("type"):
            case "text":
                texts.append(str(block.get("text") or ""))
            case "thinking":
                thoughts.append(str(block.get("thinking") or ""))
            case "tool_use":
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=block.get("id") or f"grok-tool-call-{i}-{len(tool_calls)}",
                        function=Function(
                            name=block.get("name"),
                            arguments=json.dumps(block.get("input") or {}),
                        ),
                        type="function",
                    )
                )
            case _:
                pass

    yield MessageAddedEvent(
        message=Message(
            role="assistant",
            content="".join(texts) or None,
            reasoning_content="".join(thoughts) or None,
            tool_calls=tool_calls or None,
        )
    )
    if isinstance(usage, dict):
        yield _usage_event(usage)
    for tool_call in tool_calls:
        yield ToolCallStartedEvent(tool_call=tool_call)


def _user_events(blocks: list[Any]) -> Iterator[Event]:
    texts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(str(block.get("text") or ""))
            continue
        if block.get("type") != "tool_result":
            continue
        tool_call_id = block.get("tool_use_id") or ""
        text = _result_text(block.get("content"))
        is_error = bool(block.get("is_error"))
        if (image := _read_file_image(text)) is not None:
            yield from tool_result_events(
                tool_call_id, CallToolResult(content=[image], isError=is_error)
            )
            continue
        yield ToolCallCompletedEvent(
            tool_call_id=tool_call_id,
            result=CallToolResult(
                content=[TextContent(type="text", text=text)],
                isError=is_error,
            ),
        )
        yield MessageAddedEvent(
            message=Message(role="tool", content=text, tool_call_id=tool_call_id)
        )
    if texts:
        yield MessageAddedEvent(message=Message(role="user", content="".join(texts)))


def _usage_event(usage: dict[str, Any]) -> TokenUsageEvent:
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


def _read_file_image(text: str) -> ImageContent | None:
    """The image in Grok's JSON-encoded ``read_file`` result, if it read one."""
    if not text.startswith("{"):
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("type") != "ReadFile":
        return None
    image = record.get("ImageContent")
    if not isinstance(image, dict):
        return None
    data, mime_type = image.get("data"), image.get("mime_type")
    if not isinstance(data, str) or not isinstance(mime_type, str):
        return None
    return ImageContent(type="image", data=data, mimeType=mime_type)


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _parse_line(line: str, i: int) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Skipping unparseable grok NDJSON line {}: {!r}", i, line[:200])
        return None
    if not isinstance(record, dict):
        logger.warning("Skipping non-object grok NDJSON line {}.", i)
        return None
    return record
