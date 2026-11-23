"""Adapter from Grok Build's ``--output-format streaming-messages-json`` NDJSON
to transcript events.

Each stdout line is one object tagged by ``type``: a ``system`` preamble
(``subtype: "init"``), ``assistant`` lines whose ``message.content[]`` holds
``text``/``thinking``/``tool_use`` blocks plus the response's ``usage``,
``user`` lines carrying ``tool_result`` blocks, and a terminal ``result``.
Unlike Vibe, Grok reports real token usage per model response.

Subagent lines leave ``parent_tool_use_id`` null, so only lines between a
foreground ``spawn_subagent`` call and its result are known to be subagents'
(only their usage is kept); background subagents' lines stay in the transcript,
as do those of a foreground one Grok moves to the background for want of a slot.
The step's ``result`` total covers usage lost when parallel subagents' lines
merge; session titles and compaction summaries appear nowhere.

Parsing is lenient: malformed lines are logged and skipped.
"""

import json
from collections.abc import Iterable, Iterator
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, ImageContent, TextContent

from karotte.agents.ndjson import parse_json_line
from karotte.agents.tool_results import tool_result_events
from karotte.agents.usage import (
    MESSAGES_USAGE_BUCKETS,
    int_or_zero,
    messages_usage_event,
)
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    ContextCompactedEvent,
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    UsageSource,
)


def parse_stream(lines: Iterable[str]) -> Iterator[Event]:
    """Convert one step's Grok NDJSON lines into normalized transcript events."""
    parse = StreamParser()
    for i, line in enumerate(lines):
        yield from parse(line, i)


class StreamParser:
    """:func:`parse_stream` fed one line at a time, as Grok writes them."""

    def __init__(self) -> None:
        self._foreground: set[str] = set()
        self._pending: set[str] = set()
        self._spawned: bool = False
        self._counted: dict[str, int] = dict.fromkeys(MESSAGES_USAGE_BUCKETS, 0)

    def __call__(self, line: str, i: int) -> Iterator[Event]:
        """Events for line ``i``: an ``assistant`` line yields a
        `MessageAddedEvent`, a `TokenUsageEvent` and one `ToolCallStartedEvent`
        per tool call; each ``tool_result`` block of a ``user`` line yields a
        `ToolCallCompletedEvent` and the tool `MessageAddedEvent`."""
        record = parse_json_line(line, i, "grok")
        if record is None:
            return
        kind = record.get("type")
        if kind == "result":
            yield from self._unreported(record.get("usage"))
            return
        if kind == "system" and record.get("subtype") == "compact_boundary":
            if not self._foreground:
                yield ContextCompactedEvent()
            return
        if kind not in ("assistant", "user"):
            return
        message = record.get("message")
        if not isinstance(message, dict):
            logger.warning("Skipping grok {} line {} without a message.", kind, i)
            return
        blocks = message.get("content")
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        if not isinstance(blocks, list):
            blocks = []
        usage = message.get("usage")

        if record.get("parent_tool_use_id") is not None or self._foreground:
            if kind == "assistant" and isinstance(usage, dict):
                yield self._usage_event(usage, "subagent")
            elif kind == "user" and record.get("parent_tool_use_id") is None:
                yield from _user_events(self._foreground_results(blocks))
            return
        if kind == "user":
            for block in blocks:
                if isinstance(block, dict):
                    self._pending.discard(_nonempty_str(block.get("tool_use_id")) or "")
            yield from _user_events(blocks)
            return
        self._track_calls(blocks)
        usage_event = (
            self._usage_event(usage, "agent") if isinstance(usage, dict) else None
        )
        yield from _assistant_events(blocks, usage_event, i)

    def _track_calls(self, blocks: list[Any]) -> None:
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            call_id = _nonempty_str(block.get("id"))
            if call_id is not None:
                self._pending.add(call_id)
            if block.get("name") != "spawn_subagent":
                continue
            self._spawned = True
            arguments = block.get("input")
            background = isinstance(arguments, dict) and arguments.get("background")
            if call_id is not None and background is False:
                self._foreground.add(call_id)

    def _foreground_results(self, blocks: list[Any]) -> list[Any]:
        """The results of the main agent's pending calls among a ``user`` line's
        blocks; the rest are the subagents'."""
        results: list[Any] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            call_id = _nonempty_str(block.get("tool_use_id"))
            if call_id is not None and call_id in self._pending:
                self._pending.discard(call_id)
                self._foreground.discard(call_id)
                results.append(block)
        return results

    def _usage_event(
        self, usage: dict[str, Any], source: UsageSource
    ) -> TokenUsageEvent:
        for bucket in MESSAGES_USAGE_BUCKETS:
            self._counted[bucket] += int_or_zero(usage.get(bucket))
        return messages_usage_event(usage, source)

    def _unreported(self, usage: Any) -> Iterator[Event]:
        """What the step's ``result`` total has beyond the counted usage."""
        if not isinstance(usage, dict):
            return
        missing = {
            bucket: max(0, int_or_zero(usage.get(bucket)) - self._counted[bucket])
            for bucket in MESSAGES_USAGE_BUCKETS
        }
        if any(missing.values()):
            yield self._usage_event(missing, "subagent" if self._spawned else "agent")


def _assistant_events(
    blocks: list[Any], usage: TokenUsageEvent | None, i: int
) -> Iterator[Event]:
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
                        id=_nonempty_str(block.get("id"))
                        or f"grok-tool-call-{i}-{len(tool_calls)}",
                        function=Function(
                            name=_nonempty_str(block.get("name")),
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
    if usage is not None:
        yield usage
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
        tool_call_id = _nonempty_str(block.get("tool_use_id")) or ""
        text = _result_text(block.get("content"))
        is_error = bool(block.get("is_error"))
        if (image := _read_file_image(text)) is not None:
            yield from tool_result_events(
                tool_call_id, CallToolResult(content=[image], is_error=is_error)
            )
            continue
        yield ToolCallCompletedEvent(
            tool_call_id=tool_call_id,
            result=CallToolResult(
                content=[TextContent(type="text", text=text)],
                is_error=is_error,
            ),
        )
        yield MessageAddedEvent(
            message=Message(role="tool", content=text, tool_call_id=tool_call_id)
        )
    if texts:
        yield MessageAddedEvent(message=Message(role="user", content="".join(texts)))


def _nonempty_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _read_file_image(text: str) -> ImageContent | None:
    """The image in Grok's JSON-encoded ``read_file`` result, if it read one."""
    if not text.startswith("{"):
        return None
    try:
        record = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(record, dict) or record.get("type") != "ReadFile":
        return None
    image = record.get("ImageContent")
    if not isinstance(image, dict):
        return None
    data, mime_type = image.get("data"), image.get("mime_type")
    if not isinstance(data, str) or not isinstance(mime_type, str):
        return None
    return ImageContent(type="image", data=data, mime_type=mime_type)


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
