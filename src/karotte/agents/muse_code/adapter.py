"""Adapter from Muse Code's session log to transcript events; its ``exec
--json`` output has no tool arguments or per-call usage."""

import json
from collections.abc import Iterator
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, ContentBlock, ImageContent, TextContent

from karotte.agents.ndjson import parse_json_line
from karotte.agents.tool_results import tool_result_events
from karotte.agents.usage import int_or_zero
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    ContextCompactedEvent,
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallStartedEvent,
)


class StreamParser:
    """Turns session log lines into events. A response's tool calls are sent as
    soon as Muse commits them; their results wait for any images that follow."""

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._thoughts: list[str] = []
        self._usage: list[Event] = []
        self._results: list[tuple[str, str]] = []
        self._images: dict[str, list[ImageContent]] = {}
        self._failed: set[str] = set()
        self.error: str | None = None
        """Why the step's turn ended other than completed, if it did."""

    def __call__(self, line: str, i: int) -> Iterator[Event]:
        record = parse_json_line(line, i, "muse-code")
        if record is None:
            return
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return
        if record.get("payload_type") == "tool_batch.effect.terminal":
            self._effect_ended(_dict(payload.get("record")))
            return
        if (
            record.get("payload_type") != "runtime.session"
            or payload.get("kind") != "run"
        ):
            return
        event = _dict(payload.get("event"))
        match event.get("kind"):
            case "model_response_created":
                yield from self.flush()
            case "reasoning_committed":
                if text := event.get("text"):
                    self._thoughts.append(str(text))
            case "assistant_message_committed":
                if text := event.get("text"):
                    self._texts.append(str(text))
            case "assistant_tool_calls_committed":
                calls = [_tool_call(_dict(c)) for c in event.get("tool_calls") or []]
                yield from self._send_message(calls)
            case "tool_result_batch_committed":
                for result in event.get("results") or []:
                    result = _dict(result)
                    self._results.append(
                        (str(result.get("tool_call_id")), str(result.get("text") or ""))
                    )
            case "tool_result_model_visible_content":
                images = [_image(_dict(c)) for c in event.get("content") or []]
                self._images[str(event.get("call_id"))] = [i for i in images if i]
            case "goal_usage_attribution":
                if (usage := _usage_event(_dict(event.get("record")))) is not None:
                    self._usage.append(usage)
            case "context_compaction_installed":
                yield from self.flush()
                yield ContextCompactedEvent()
            case "terminal":
                yield from self.flush()
                if event.get("terminal") != "completed":
                    self.error = str(event.get("reason") or event.get("terminal"))
                    logger.warning("muse-code ended its turn: {}", event)
            case _:
                pass

    def flush(self) -> Iterator[Event]:
        """What the step hasn't sent: held results, then text and reasoning
        that no tool call followed."""
        for call_id, text in self._results:
            content: list[ContentBlock] = [TextContent(type="text", text=text)]
            content.extend(self._images.pop(call_id, []))
            yield from tool_result_events(
                call_id,
                CallToolResult(content=content, is_error=call_id in self._failed),
            )
        self._results, self._images, self._failed = [], {}, set()
        if self._texts or self._thoughts:
            yield from self._send_message([])
        yield from self._usage
        self._usage = []

    def _send_message(
        self, calls: list[ChatCompletionMessageToolCall]
    ) -> Iterator[Event]:
        yield MessageAddedEvent(
            message=Message(
                role="assistant",
                content="\n\n".join(self._texts) or None,
                reasoning_content="\n\n".join(self._thoughts) or None,
                tool_calls=calls or None,
            )
        )
        self._texts, self._thoughts = [], []
        for call in calls:
            yield ToolCallStartedEvent(tool_call=call)
        yield from self._usage
        self._usage = []

    def _effect_ended(self, record: dict[str, Any]) -> None:
        outcome = _dict(record.get("outcome"))
        terminal = _dict(_dict(outcome.get("task_completion")).get("terminal"))
        if outcome.get("kind") == "failed" or terminal.get("kind") == "failed":
            self._failed.add(str(record.get("call_id")))


def _tool_call(call: dict[str, Any]) -> ChatCompletionMessageToolCall:
    args = call.get("args")
    return ChatCompletionMessageToolCall(
        id=str(call.get("call_id")),
        function=Function(
            name=str(call.get("name")),
            arguments=args if isinstance(args, str) else json.dumps(args or {}),
        ),
        type="function",
    )


def _image(content: dict[str, Any]) -> ImageContent | None:
    if content.get("kind") != "image":
        return None
    return ImageContent(
        type="image",
        data=str(content.get("base64_data") or ""),
        mime_type=str(content.get("media_type") or ""),
    )


def _usage_event(record: dict[str, Any]) -> TokenUsageEvent | None:
    """One model call's usage, the main agent's or a helper's; tool records
    report none."""
    quantity = _dict(record.get("quantity"))
    if quantity.get("reported") is not True:
        return None
    return TokenUsageEvent(
        input_tokens=int_or_zero(quantity.get("input_tokens")),
        output_tokens=int_or_zero(quantity.get("output_tokens")),
        cache_read_tokens=int_or_zero(quantity.get("cached_tokens")),
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
