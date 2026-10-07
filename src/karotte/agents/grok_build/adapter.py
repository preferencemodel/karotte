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
from mcp.types import CallToolResult, TextContent

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

    Yields, per ``assistant`` line: a `MessageAddedEvent`, a `TokenUsageEvent`
    and one `ToolCallStartedEvent` per tool call. Per ``tool_result`` block of
    a ``user`` line: a `ToolCallCompletedEvent` and the tool `MessageAddedEvent`.
    ``system`` and ``result`` lines carry only metadata and yield nothing.
    """
    for i, line in enumerate(lines):
        record = _parse_line(line, i)
        if record is None:
            continue
        if record.get("parent_tool_use_id") is not None:
            continue

        kind = record.get("type")
        message = record.get("message")
        if kind not in ("assistant", "user"):
            continue
        if not isinstance(message, dict):
            logger.warning("Skipping grok {} line {} without a message.", kind, i)
            continue
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
        yield ToolCallCompletedEvent(
            tool_call_id=tool_call_id,
            result=CallToolResult(
                content=[TextContent(type="text", text=text)],
                isError=bool(block.get("is_error")),
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


def _result_text(content: Any) -> str:
    if isinstance(content, list):
        content = "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if not isinstance(content, str):
        return ""
    return _readable_result(content)


# Keys, in order of preference, under which Grok puts the text it shows the
# model for a tool result.
_PROMPT_TEXT_KEYS = (
    "output_for_prompt",
    "tool_output_for_prompt",
    "summary_for_prompt",
    "content",
    "output",
    "OkayOutput",
    "ErrorOutput",
)


def _readable_result(text: str) -> str:
    """The readable part of a Grok tool result.

    Grok serializes each tool's result as JSON tagged by ``type``, e.g.
    ``{"type":"Bash","output":[<bytes>],"output_for_prompt":"exit: 0\\n..."}``
    or ``{"type":"ReadFile","FileContent":{"content":...}}``, with some output
    as arrays of byte values. Pull out the text Grok shows the model, falling
    back to the raw string for shapes this doesn't know. The NDJSON artifact
    keeps the full result.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, list):
        # ACP content blocks, as on tool-argument errors.
        texts = [
            str(block["content"].get("text") or "")
            for block in parsed
            if isinstance(block, dict) and isinstance(block.get("content"), dict)
        ]
        return "".join(texts) if texts else text
    if not isinstance(parsed, dict):
        return text

    found = _prompt_text(parsed)
    if found is None:
        # One level down: tagged variants wrap their payload, e.g.
        # {"type":"ReadFile","FileContent":{...}} or {"FileNotFound":"Error: ..."}.
        # Variant names are CamelCase; other keys (tool_name, command, ...) are
        # metadata, not output.
        for key, value in parsed.items():
            if not key[:1].isupper():
                continue
            if isinstance(value, str):
                found = value
            elif isinstance(value, dict):
                found = _prompt_text(value)
            if found is not None:
                break
    if found is not None:
        return found

    # Grep reports raw stdout/stderr bytes and nothing else readable.
    streams = [_decode_bytes(parsed.get(key)) for key in ("stdout", "stderr")]
    if any(s is not None for s in streams):
        return "".join(s for s in streams if s)
    return text


def _prompt_text(record: dict[str, Any]) -> str | None:
    for key in _PROMPT_TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = _prompt_text(value)
            if nested is not None:
                return nested
    return None


def _decode_bytes(value: Any) -> str | None:
    if not isinstance(value, list) or not all(
        isinstance(b, int) and 0 <= b < 256 for b in value
    ):
        return None
    return bytes(value).decode("utf-8", errors="replace")


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
