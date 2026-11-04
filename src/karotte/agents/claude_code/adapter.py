"""Adapter from Claude Code's ``--output-format stream-json`` (with
``--include-partial-messages``) to transcript events.

Claude Code writes one ``assistant`` line per finished content block, so one
model response spans several lines sharing a message id. The response's final
usage arrives only in its ``message_delta`` stream event; ``message_stop`` ends
it. Claude Code runs a tool as soon as its call has streamed, so a result can
arrive before the response that asked for it ends; such results are held until
then. Lines with a non-null ``parent_tool_use_id`` come from subagents and are
left out (the native log, saved as an artifact, keeps them).

With ``--include-hook-events``, the context a tool hook hands the model (the
time-left note) arrives just before the tool's result, and is kept with it, at
the end of the tool message, where the builtin agent puts its own note.
"""

import json
from collections.abc import Iterable, Iterator
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, ContentBlock, ImageContent, TextContent

from karotte.agents.message_loop import append_text_content
from karotte.agents.ndjson import parse_json_line
from karotte.agents.tool_results import tool_result_events
from karotte.agents.usage import messages_usage_event
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    ToolCallStartedEvent,
)

# Claude Code runs `PostToolUse` hooks after a tool call that succeeded and
# `PostToolUseFailure` ones after one that failed, a shell command that exited
# non-zero included. What they add reaches the model next to the result.
TOOL_RESULT_HOOK_EVENTS = ("PostToolUse", "PostToolUseFailure")


def parse_stream(lines: Iterable[str]) -> Iterator[Event]:
    """Convert Claude Code stream-json lines into normalized transcript events."""
    parse = StreamParser()
    for i, line in enumerate(lines):
        yield from parse(line, i)


class StreamParser:
    """:func:`parse_stream` fed one line at a time, as Claude Code writes them."""

    def __init__(self) -> None:
        self._blocks: list[dict[str, Any]] = []
        self._usage: dict[str, Any] = {}
        self._in_response: bool = False
        self._held_results: list[Event] = []
        self._hook_notes: list[str] = []
        """What tool hooks added, for the tool result that follows them."""
        self.error: str | None = None
        """What Claude Code's ``result`` line reported, if the run ended in an error."""

    def __call__(self, line: str, i: int) -> Iterator[Event]:
        record = parse_json_line(line, i, "claude")
        if record is None or record.get("parent_tool_use_id") is not None:
            return
        match record.get("type"):
            case "stream_event":
                yield from self._stream_event(record.get("event"))
            case "assistant":
                content = _message(record).get("content")
                if isinstance(content, list):
                    self._blocks.extend(b for b in content if isinstance(b, dict))
            case "system" if record.get("subtype") == "hook_response":
                if (note := _tool_hook_note(record)) is not None:
                    self._hook_notes.append(note)
            case "user":
                events = list(_user_events(_message(record).get("content")))
                if self._hook_notes and _add_to_last_tool_message(
                    events, self._hook_notes
                ):
                    self._hook_notes = []
                if self._in_response:
                    self._held_results.extend(events)
                else:
                    yield from events
            case "result" if record.get("is_error"):
                self.error = str(record.get("result"))
                logger.warning("claude reported an error: {}", self.error)
            case _:
                pass

    def _stream_event(self, event: Any) -> Iterator[Event]:
        if not isinstance(event, dict):
            return
        match event.get("type"):
            case "message_start":
                self._in_response = True
                self._blocks = []
                message = event.get("message")
                usage = message.get("usage") if isinstance(message, dict) else None
                self._usage = dict(usage) if isinstance(usage, dict) else {}
            case "message_delta":
                if isinstance(usage := event.get("usage"), dict):
                    self._usage.update(usage)
            case "message_stop":
                yield from self.flush()
            case _:
                pass

    def flush(self) -> Iterator[Event]:
        """The response in progress, with the results held for it."""
        if self._in_response and self._blocks:
            yield from _response_events(self._blocks, self._usage)
        elif self._in_response:
            yield messages_usage_event(self._usage)
        yield from self._held_results
        self._blocks, self._usage, self._held_results = [], {}, []
        self._in_response = False


def _response_events(
    blocks: list[dict[str, Any]], usage: dict[str, Any]
) -> Iterator[Event]:
    texts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[ChatCompletionMessageToolCall] = []
    for block in blocks:
        match block.get("type"):
            case "text":
                texts.append(str(block.get("text") or ""))
            case "thinking":
                thoughts.append(str(block.get("thinking") or ""))
            case "tool_use":
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=block.get("id") or "",
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
    yield messages_usage_event(usage)
    for tool_call in tool_calls:
        yield ToolCallStartedEvent(tool_call=tool_call)


def _user_events(content: Any) -> Iterator[Event]:
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_result":
            result = CallToolResult(
                content=_result_content(block.get("content")),
                isError=bool(block.get("is_error")),
            )
            yield from tool_result_events(str(block.get("tool_use_id") or ""), result)
    if texts:
        yield MessageAddedEvent(message=Message(role="user", content="".join(texts)))


def _tool_hook_note(record: dict[str, Any]) -> str | None:
    """The context a tool hook's output hands the model, if any."""
    if record.get("hook_event") not in TOOL_RESULT_HOOK_EVENTS:
        return None
    try:
        output = json.loads(str(record.get("stdout") or ""))
    except json.JSONDecodeError:
        return None
    specific = output.get("hookSpecificOutput") if isinstance(output, dict) else None
    context = specific.get("additionalContext") if isinstance(specific, dict) else None
    return context if isinstance(context, str) and context else None


def _add_to_last_tool_message(events: list[Event], notes: list[str]) -> bool:
    """Append `notes` to the last tool message in `events`, as the builtin agent
    appends its own; False if there is none."""
    for event in reversed(events):
        if isinstance(event, MessageAddedEvent) and event.message.role == "tool":
            for note in notes:
                append_text_content(event.message, note)
            return True
    return False


def _result_content(content: Any) -> list[ContentBlock]:
    if isinstance(content, str):
        return [TextContent(type="text", text=content)]
    if not isinstance(content, list):
        return []
    blocks: list[ContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            blocks.append(TextContent(type="text", text=str(block.get("text") or "")))
        elif block.get("type") == "image":
            source = block.get("source")
            if isinstance(source, dict) and source.get("type") == "base64":
                blocks.append(
                    ImageContent(
                        type="image",
                        data=str(source.get("data") or ""),
                        mimeType=str(source.get("media_type") or ""),
                    )
                )
    return blocks


def _message(record: dict[str, Any]) -> dict[str, Any]:
    message = record.get("message")
    return message if isinstance(message, dict) else {}
