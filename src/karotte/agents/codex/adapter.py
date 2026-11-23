"""Adapter from ``codex exec --json`` to transcript events; tool calls that run
together share one assistant message with the text and reasoning before them.

Codex's usage totals cover only the main thread; its rollout files, when readable,
tell compaction calls apart and hold the subagents' calls.
"""

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, ContentBlock, ImageContent, TextContent

from karotte.agents.ndjson import parse_json_line
from karotte.agents.student_files import (
    ScanBudget,
    student_file_first_line,
    student_file_lines,
)
from karotte.agents.tool_results import tool_result_events
from karotte.agents.usage import int_or_zero
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallStartedEvent,
    UsageSource,
)

_TOOL_ITEMS = frozenset({"command_execution", "mcp_tool_call", "file_change"})
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
)


@dataclass
class Session:
    """What carries over between the steps of one Codex session."""

    thread_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    counted_responses: set[str] = field(default_factory=set)
    """Responses from the rollout files already counted as helper or subagent usage."""


type RolloutFiles = Callable[[ScanBudget], Iterable[Path]]
"""Every rollout file in Codex's home, as far as the budget goes."""


def parse_stream(
    lines: Iterable[str], session: Session | None = None, step: int = 0
) -> Iterator[Event]:
    """Convert ``codex exec --json`` lines into normalized transcript events."""
    parse = StreamParser(session or Session(), step)
    for i, line in enumerate(lines):
        yield from parse(line, i)


class StreamParser:
    """:func:`parse_stream` fed one line at a time, as Codex writes them."""

    def __init__(
        self, session: Session, step: int, rollout_files: RolloutFiles | None = None
    ) -> None:
        self._session: Session = session
        self._rollout_files: RolloutFiles | None = rollout_files
        # Codex numbers items from item_0 in every process, so each step needs its own ids.
        self._id_prefix: str = f"step{step}_"
        self._texts: list[str] = []
        self._thoughts: list[str] = []
        self._calls: dict[str, ChatCompletionMessageToolCall] = {}
        self._running: set[str] = set()
        self._results: list[Event] = []
        self._unnamed: int = 0

    def __call__(self, line: str, i: int) -> Iterator[Event]:
        record = parse_json_line(line, i, "codex")
        if record is None:
            return
        match record.get("type"):
            case "thread.started":
                thread_id = record.get("thread_id")
                if isinstance(thread_id, str) and thread_id != self._session.thread_id:
                    self._session.thread_id = thread_id
                    self._session.usage = {}
            case "item.started":
                self._item_started(_item(record))
            case "item.completed":
                yield from self._item_completed(_item(record))
            case "turn.completed":
                yield from self.flush()
                compactions, subagents = self._rollout_usage()
                if isinstance(usage := record.get("usage"), dict):
                    yield from self._turn_usage(usage, compactions)
                yield from self._subagent_usage(subagents)
            case "turn.failed":
                yield from self.flush()
                logger.warning("codex reported an error: {}", record)
            case "error":
                # Codex retries some errors within the turn, so this doesn't end it.
                logger.warning("codex reported an error: {}", record)
            case _:
                pass

    def _item_started(self, item: dict[str, Any]) -> None:
        # Without an id the completion can't be matched, so it records the call.
        if item.get("type") in _TOOL_ITEMS and _item_id(item) is not None:
            call_id = self._call_id(item)
            self._calls[call_id] = _tool_call(item, call_id)
            self._running.add(call_id)

    def _item_completed(self, item: dict[str, Any]) -> Iterator[Event]:
        match item.get("type"):
            case "agent_message":
                self._texts.append(str(item.get("text") or ""))
            case "reasoning":
                self._thoughts.append(str(item.get("text") or ""))
            case kind if kind in _TOOL_ITEMS:
                call_id = self._call_id(item)
                if call_id not in self._calls:
                    self._calls[call_id] = _tool_call(item, call_id)
                self._running.discard(call_id)
                self._results.extend(tool_result_events(call_id, _result(item)))
                if not self._running:
                    yield from self._send_calls()
            case "error":
                logger.warning("codex: {}", item.get("message"))
            case kind:
                logger.debug("Skipping codex {} item.", kind)

    def _call_id(self, item: dict[str, Any]) -> str:
        if (item_id := _item_id(item)) is None:
            self._unnamed += 1
            item_id = f"unnamed_{self._unnamed}"
        return self._id_prefix + item_id

    def _send_calls(self) -> Iterator[Event]:
        if not self._calls:
            return
        calls = list(self._calls.values())
        yield self._assistant_message(calls)
        for tool_call in calls:
            yield ToolCallStartedEvent(tool_call=tool_call)
        yield from self._results
        self._calls, self._running, self._results = {}, set(), []

    def flush(self) -> Iterator[Event]:
        """What the step hasn't sent: running calls with the results they got,
        then text and reasoning that no call followed."""
        yield from self._send_calls()
        if self._texts or self._thoughts:
            yield self._assistant_message([])

    def finish(self) -> Iterator[Event]:
        """:meth:`flush`, then subagent calls logged since the turn completed."""
        yield from self.flush()
        yield from self._subagent_usage(self._rollout_usage()[1])

    def _assistant_message(
        self, tool_calls: list[ChatCompletionMessageToolCall]
    ) -> MessageAddedEvent:
        message = Message(
            role="assistant",
            content="\n\n".join(self._texts) or None,
            reasoning_content="\n\n".join(self._thoughts) or None,
            tool_calls=tool_calls or None,
        )
        self._texts, self._thoughts = [], []
        return MessageAddedEvent(message=message)

    def _turn_usage(
        self, totals: dict[str, Any], compactions: dict[str, dict[str, Any]]
    ) -> Iterator[TokenUsageEvent]:
        """This step's share of the thread totals Codex reports, less the
        compaction calls in it. A turn that got no usage update reports zeros,
        which mustn't lower the running totals."""
        previous = self._session.usage
        current = {
            key: max(int_or_zero(totals.get(key)), previous.get(key, 0))
            for key in _USAGE_KEYS
        }
        step = {key: current[key] - previous.get(key, 0) for key in _USAGE_KEYS}
        self._session.usage = current
        helper = dict.fromkeys(_USAGE_KEYS, 0)
        for response_id, usage in compactions.items():
            if response_id not in self._session.counted_responses:
                self._session.counted_responses.add(response_id)
                for key in _USAGE_KEYS:
                    helper[key] += int_or_zero(usage.get(key))
        helper = {key: min(helper[key], step[key]) for key in _USAGE_KEYS}
        yield _usage_event({key: step[key] - helper[key] for key in _USAGE_KEYS})
        if any(helper.values()):
            yield _usage_event(helper, "helper")

    def _subagent_usage(
        self, subagents: dict[str, tuple[dict[str, Any], UsageSource]]
    ) -> Iterator[TokenUsageEvent]:
        for response_id, (usage, source) in subagents.items():
            if response_id not in self._session.counted_responses:
                self._session.counted_responses.add(response_id)
                yield _usage_event(usage, source)

    def _rollout_usage(
        self,
    ) -> tuple[
        dict[str, dict[str, Any]], dict[str, tuple[dict[str, Any], UsageSource]]
    ]:
        """By response id, from the rollout files: the thread's compaction calls,
        and its subagents' calls with their compactions as helper calls."""
        thread = self._session.thread_id
        if self._rollout_files is None or thread is None:
            return {}, {}
        budget = ScanBudget()
        threads = {
            path: _rollout_thread(path, budget) for path in self._rollout_files(budget)
        }
        parents = dict(t for t in threads.values() if t is not None)
        compactions: dict[str, dict[str, Any]] = {}
        subagents: dict[str, tuple[dict[str, Any], UsageSource]] = {}
        for path, ids in threads.items():
            if ids is None:
                continue
            records, compacted = _usage_records(path, ids[0], budget)
            if ids[0] == thread:
                compactions.update((r, records[r]) for r in compacted if r in records)
            elif _descends_from(ids[0], thread, parents):
                subagents.update(
                    (r, (usage, "helper" if r in compacted else "subagent"))
                    for r, usage in records.items()
                )
        return compactions, subagents


def _usage_event(
    usage: dict[str, Any], source: UsageSource = "agent"
) -> TokenUsageEvent:
    return TokenUsageEvent(
        input_tokens=int_or_zero(usage.get("input_tokens")),
        output_tokens=int_or_zero(usage.get("output_tokens")),
        cache_read_tokens=int_or_zero(usage.get("cached_input_tokens")),
        cache_write_tokens=int_or_zero(usage.get("cache_write_input_tokens")),
        source=source,
    )


def _rollout_thread(path: Path, budget: ScanBudget) -> tuple[str, str | None] | None:
    """The thread a rollout file logs and its parent thread, from its first
    line; a forked thread's file repeats its parent's ``session_meta`` later."""
    first = student_file_first_line(path, budget)
    record = parse_json_line(first, 0, "codex rollout") if first else None
    payload = record.get("payload") if record else None
    if (
        not record
        or record.get("type") != "session_meta"
        or not isinstance(payload, dict)
    ):
        return None
    thread_id, parent = payload.get("id"), payload.get("parent_thread_id")
    if not isinstance(thread_id, str):
        return None
    return thread_id, parent if isinstance(parent, str) else None


def _usage_records(
    path: Path, thread_id: str, budget: ScanBudget
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """The thread's usage by response id, and the responses that were compactions."""
    records: dict[str, dict[str, Any]] = {}
    compacted: set[str] = set()
    for record in _rollout_records(path, budget):
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record.get("type") == "compacted":
            if isinstance(response_id := payload.get("compaction_response_id"), str):
                compacted.add(response_id)
        elif record.get("type") == "token_usage_record":
            response_id, usage = payload.get("response_id"), payload.get("usage")
            if (
                payload.get("thread_id") == thread_id
                and isinstance(response_id, str)
                and isinstance(usage, dict)
            ):
                records[response_id] = usage
    return records, compacted


def _rollout_records(path: Path, budget: ScanBudget) -> Iterator[dict[str, Any]]:
    """The file's lines that may be usage or compaction records, parsed."""
    for i, line in enumerate(student_file_lines(path, budget=budget)):
        if '"token_usage_record"' not in line and '"compacted"' not in line:
            continue
        if (record := parse_json_line(line, i, "codex rollout")) is not None:
            yield record


def _descends_from(
    thread_id: str, ancestor: str, parents: dict[str, str | None]
) -> bool:
    seen: set[str] = set()
    parent = parents.get(thread_id)
    while parent is not None and parent not in seen:
        if parent == ancestor:
            return True
        seen.add(parent)
        parent = parents.get(parent)
    return False


def _tool_call(item: dict[str, Any], call_id: str) -> ChatCompletionMessageToolCall:
    name, arguments = _name_and_arguments(item)
    return ChatCompletionMessageToolCall(
        id=call_id,
        function=Function(name=name, arguments=json.dumps(arguments)),
        type="function",
    )


def _name_and_arguments(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    match item.get("type"):
        case "command_execution":
            return "exec_command", {"command": item.get("command")}
        case "mcp_tool_call":
            arguments = item.get("arguments")
            return (
                f"mcp__{item.get('server')}__{item.get('tool')}",
                arguments if isinstance(arguments, dict) else {},
            )
        case _:
            return "apply_patch", {"changes": item.get("changes") or []}


def _result(item: dict[str, Any]) -> CallToolResult:
    match item.get("type"):
        case "command_execution":
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=str(item.get("aggregated_output") or "")
                    )
                ],
                is_error=item.get("exit_code") != 0,
            )
        case "mcp_tool_call":
            if (error := item.get("error")) is not None:
                message = error.get("message") if isinstance(error, dict) else error
                return CallToolResult(
                    content=[TextContent(type="text", text=str(message))], is_error=True
                )
            result = item.get("result")
            content = result.get("content") if isinstance(result, dict) else None
            return CallToolResult(content=_mcp_content(content))
        case _:
            status = str(item.get("status") or "")
            return CallToolResult(
                content=[TextContent(type="text", text=status)],
                is_error=status != "completed",
            )


def _mcp_content(content: Any) -> list[ContentBlock]:
    if not isinstance(content, list):
        return []
    blocks: list[ContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            blocks.append(TextContent(type="text", text=str(block.get("text") or "")))
        elif block.get("type") == "image":
            blocks.append(
                ImageContent(
                    type="image",
                    data=str(block.get("data") or ""),
                    mime_type=str(
                        block.get("mimeType") or block.get("mime_type") or ""
                    ),
                )
            )
    return blocks


def _item_id(item: dict[str, Any]) -> str | None:
    item_id = item.get("id")
    return item_id if isinstance(item_id, str) and item_id else None


def _item(record: dict[str, Any]) -> dict[str, Any]:
    item = record.get("item")
    return item if isinstance(item, dict) else {}
