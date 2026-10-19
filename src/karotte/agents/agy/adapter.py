"""Adapter from ``agy --output-format stream-json`` to transcript events; each
model response becomes one assistant message with the tool calls it made."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, TextContent

from karotte.agents.ndjson import parse_json_line
from karotte.agents.tool_results import tool_result_events
from karotte.agents.usage import int_or_zero
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallStartedEvent,
)


@dataclass
class Session:
    """What carries over between the steps of one agy conversation."""

    conversation_id: str | None = None


def parse_stream(
    lines: Iterable[str], session: Session | None = None, step: int = 0
) -> Iterator[Event]:
    """Convert ``agy`` stream-json lines into normalized transcript events."""
    parse = StreamParser(session or Session(), step)
    for i, line in enumerate(lines):
        yield from parse(line, i)
    yield from parse.flush()


class StreamParser:
    """:func:`parse_stream` fed one line at a time, as agy writes them."""

    def __init__(self, session: Session, step: int) -> None:
        self._session: Session = session
        # agy numbers steps per conversation; a new conversation starts over.
        self._id_prefix: str = f"step{step}_"
        self._response: int | None = None
        self._texts: list[str] = []
        self._usage: dict[str, Any] = {}
        self._calls: dict[str, ChatCompletionMessageToolCall] = {}
        self._results: list[Event] = []

    def __call__(self, line: str, i: int) -> Iterator[Event]:
        record = parse_json_line(line, i, "agy")
        if record is None:
            return
        match record.get("event"):
            case "init":
                if isinstance(conversation := record.get("conversation_id"), str):
                    self._session.conversation_id = conversation
            case "step_update":
                update = record.get("step_update")
                if isinstance(update, dict):
                    yield from self._step(update)
            case "result":
                yield from self.flush()
                result = record.get("result")
                if isinstance(result, dict) and result.get("status") != "SUCCESS":
                    logger.warning("agy reported an error: {}", result.get("error"))
            case _:
                pass

    def _step(self, update: dict[str, Any]) -> Iterator[Event]:
        index = update.get("step_index")
        match update.get("step_type"):
            case "agent_response":
                if index != self._response:
                    yield from self.flush()
                    self._response = index if isinstance(index, int) else None
                if isinstance(delta := update.get("text_delta"), str):
                    self._texts.append(delta)
                if isinstance(usage := update.get("usage"), dict):
                    self._usage = usage
            case "tool":
                info = update.get("tool_info")
                info = info if isinstance(info, dict) else {}
                call_id = f"{self._id_prefix}{index}"
                if call_id not in self._calls:
                    self._calls[call_id] = _tool_call(info, call_id)
                if (state := update.get("state")) in ("DONE", "ERROR"):
                    self._results.extend(
                        tool_result_events(call_id, _result(info, state == "ERROR"))
                    )
            case kind:
                logger.debug("Skipping agy {} step.", kind)

    def flush(self) -> Iterator[Event]:
        """The response in progress, with its calls and the results they got."""
        calls = list(self._calls.values())
        if self._texts or calls:
            yield MessageAddedEvent(
                message=Message(
                    role="assistant",
                    content="".join(self._texts) or None,
                    tool_calls=calls or None,
                )
            )
        if self._usage:
            yield _usage_event(self._usage)
        for tool_call in calls:
            yield ToolCallStartedEvent(tool_call=tool_call)
        yield from self._results
        self._response, self._texts, self._usage = None, [], {}
        self._calls, self._results = {}, []


def _usage_event(usage: dict[str, Any]) -> TokenUsageEvent:
    """agy's input tokens leave out cache reads; its output tokens include thinking."""
    cache_read = int_or_zero(usage.get("cache_read_tokens"))
    return TokenUsageEvent(
        input_tokens=int_or_zero(usage.get("input_tokens")) + cache_read,
        output_tokens=int_or_zero(usage.get("output_tokens")),
        cache_read_tokens=cache_read,
        cache_write_tokens=0,
    )


def _tool_call(info: dict[str, Any], call_id: str) -> ChatCompletionMessageToolCall:
    name = str(info.get("name") or "")
    arguments = info.get("parameters")
    arguments = arguments if isinstance(arguments, dict) else {}
    if name == "call_mcp_tool":
        name = f"mcp__{arguments.get('ServerName')}__{arguments.get('ToolName')}"
        arguments = arguments.get("Arguments")
        arguments = arguments if isinstance(arguments, dict) else {}
    return ChatCompletionMessageToolCall(
        id=call_id,
        function=Function(name=name, arguments=json.dumps(arguments)),
        type="function",
    )


def _result(info: dict[str, Any], failed: bool) -> CallToolResult:
    if failed:
        error = info.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        return CallToolResult(
            content=[TextContent(type="text", text=str(message or "tool failed"))],
            isError=True,
        )
    output = info.get("output")
    if not isinstance(output, str) or not output:
        return CallToolResult(content=[])
    return CallToolResult(content=[TextContent(type="text", text=output)])
