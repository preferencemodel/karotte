"""Adapter from Mistral Vibe's ``--output streaming`` NDJSON to transcript events.

Each stdout line is one conversation message (Vibe's ``LLMMessage`` dumped as
JSON): ``role`` is ``system``/``user``/``assistant``/``tool``, assistant
messages carry ``tool_calls`` (``{id, index, function: {name, arguments},
type}``), and a ``role: "tool"`` line carries the result for ``tool_call_id``.
Vibe never writes token usage to stdout (its ``LLMUsage`` stays internal), so
usage is estimated from message content and marked as such. Its session logs
record the real totals, including subagents and compaction summaries that
stdout never shows; :class:`UsageLedger` turns them into corrections.

Parsing is pure and lenient: no I/O, malformed lines are logged and skipped.
The native log is saved as an artifact elsewhere and remains the source of
truth, so the events can always be re-derived from it.
"""

from collections.abc import Iterable, Iterator
from typing import Any

from litellm import token_counter
from loguru import logger
from mcp.types import CallToolResult, TextContent

from karotte.agents.ndjson import parse_json_line
from karotte.agents.usage import int_or_zero
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    UsageSource,
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
    parse = StreamParser(model)
    for i, line in enumerate(lines):
        yield from parse(line, i)


class StreamParser:
    """:func:`parse_stream` fed one line at a time, as Vibe writes them."""

    def __init__(self, model: str = "") -> None:
        self._model: str = model
        self._input_tokens: int = 0
        self._synthetic_id_count: int = 0

    def __call__(self, line: str, i: int) -> Iterator[Event]:
        record = parse_json_line(line, i, "vibe")
        if record is None:
            return

        role = record.get("role")
        if role not in _ROLES:
            logger.warning("Skipping vibe NDJSON line {} with role {!r}.", i, role)
            return

        tool_calls: list[ChatCompletionMessageToolCall] = []
        for element in record.get("tool_calls") or []:
            tool_call = _parse_tool_call(element, i)
            if tool_call is None:
                continue
            if not tool_call.id:
                self._synthetic_id_count += 1
                tool_call.id = f"vibe-tool-call-{self._synthetic_id_count}"
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

        message_tokens = _count_tokens(self._model, _message_text(record))
        if role == "assistant":
            yield TokenUsageEvent(
                input_tokens=self._input_tokens,
                output_tokens=message_tokens,
                estimated=True,
            )
        self._input_tokens += message_tokens

        for tool_call in tool_calls:
            yield ToolCallStartedEvent(tool_call=tool_call)


class UsageLedger:
    """Tokens the events of every step have reported so far, by source."""

    def __init__(self) -> None:
        self.counted: dict[UsageSource, tuple[int, int]] = {
            "agent": (0, 0),
            "subagent": (0, 0),
            "helper": (0, 0),
        }

    def add(self, event: TokenUsageEvent) -> None:
        tokens_in, tokens_out = self.counted[event.source]
        self.counted[event.source] = (
            tokens_in + event.input_tokens,
            tokens_out + event.output_tokens,
        )

    def corrections(
        self, sessions: Iterable[dict[str, Any]], subagents: Iterable[dict[str, Any]]
    ) -> Iterator[TokenUsageEvent]:
        """What the session logs' totals have beyond the counted usage. The main
        sessions' share includes their compaction summaries, which the logs
        don't break out."""
        yield from self._missing("agent", logged_total(sessions), estimated=True)
        yield from self._missing("subagent", logged_total(subagents), estimated=False)

    def _missing(
        self, source: UsageSource, logged: tuple[int, int], estimated: bool
    ) -> Iterator[TokenUsageEvent]:
        counted_in, counted_out = self.counted[source]
        event = TokenUsageEvent(
            input_tokens=max(0, logged[0] - counted_in),
            output_tokens=max(0, logged[1] - counted_out),
            estimated=estimated,
            source=source,
        )
        if event.input_tokens or event.output_tokens:
            self.add(event)
            yield event


def logged_total(metas: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """Input and output tokens of the session logs' ``meta.json`` contents. A
    compaction carries a session's stats over into a new log naming it as
    parent, so only the last log of each chain counts."""
    metas = list(metas)
    parents = {str(meta.get("parent_session_id")) for meta in metas}
    stats = [
        meta.get("stats")
        for meta in metas
        if str(meta.get("session_id")) not in parents
    ]
    stats = [s for s in stats if isinstance(s, dict)]
    return (
        sum(int_or_zero(s.get("session_prompt_tokens")) for s in stats),
        sum(int_or_zero(s.get("session_completion_tokens")) for s in stats),
    )


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
