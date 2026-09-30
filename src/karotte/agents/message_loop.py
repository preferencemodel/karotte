import json
import time
from collections.abc import AsyncGenerator, Sequence
from typing import Any, ClassVar, Literal

from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from litellm.types.utils import Message as LiteLlmMessage
from loguru import logger
from mcp.types import CallToolResult, ContentBlock, ImageContent, TextContent

from karotte.agents.agent import (
    EmptyTurnLimitReachedError,
    RunContext,
    StepContextWindowLimitReachedError,
    StepTimeLimitReachedError,
    TurnLimitReachedError,
)
from karotte.agents.message_source import MessageSource
from karotte.schemas.chat import ChatCompletionMessageToolCall, Message
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)


def _monotonic() -> float:
    """Wall-clock reference for step time limits; indirected for testability."""
    return time.monotonic()


MAX_CONSECUTIVE_EMPTY_TURNS = 3

EMPTY_TURN_NUDGE = (
    "Your last turn produced no message or tool call. Pick up "
    "where you left off. Do not apologize or recap. Break the remaining work "
    "into smaller pieces if necessary."
)


class MessageLoopAgent:
    """Agent that drives turns from a `MessageSource` and executes the tool
    calls each message contains against the MCP client.

    Builtin and external agents differ only in their source; both reuse this
    loop. Events are yielded raw -- the runner applies transcript / run-state
    bookkeeping.
    """

    allows_student_mcp_access: ClassVar[bool] = False
    native_tool_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, source: MessageSource) -> None:
        self._source: MessageSource = source
        self._turn_count: int = 0
        self._ctx: RunContext

    async def start(self, ctx: RunContext) -> None:
        self._ctx = ctx

    async def run_step(
        self,
        instructions: str,
        time_limit_seconds: float | None = None,
        on_time_limit: Literal["error", "score"] = "error",
        context_window_limit: int | None = None,
        on_context_window_limit: Literal["error", "score"] = "error",
    ) -> AsyncGenerator[Event]:
        deadline = (
            _monotonic() + time_limit_seconds
            if time_limit_seconds is not None
            else None
        )
        # Context-window length of the most recent turn (its reported input
        # tokens); 0 until the first turn reports usage.
        context_length = 0

        instructions_message = LiteLlmMessage(role="user", content=instructions)
        yield MessageAddedEvent(message=Message(**instructions_message.model_dump()))

        empty_turns = 0

        while True:
            self._turn_count += 1
            turn_limit = self._ctx.config.turn_limit
            if turn_limit is not None and self._turn_count > turn_limit:
                msg = f"Turn limit of {turn_limit} reached."
                raise TurnLimitReachedError(msg)

            # Time can only be observed between turns; a turn already generating
            # is allowed to finish, so a step may overrun by up to one turn.
            if deadline is not None and _monotonic() >= deadline:
                if on_time_limit == "error":
                    msg = f"Time limit of {time_limit_seconds}s reached."
                    raise StepTimeLimitReachedError(msg)
                return

            # The context window is measured from the previous turn's usage, so
            # like the time limit a step may overrun by up to one turn.
            if (
                context_window_limit is not None
                and context_length >= context_window_limit
            ):
                if on_context_window_limit == "error":
                    msg = f"Context window limit of {context_window_limit} reached."
                    raise StepContextWindowLimitReachedError(msg)
                return

            message_event = None
            async for event in self._source.collect(
                self._ctx.transcript.messages, self._ctx.tools
            ):
                yield event
                if isinstance(event, MessageAddedEvent):
                    message_event = event
                elif isinstance(event, TokenUsageEvent):
                    context_length = event.input_tokens

            assert message_event is not None
            message = message_event.message

            if message_event.finish_reason == "length":
                logger.warning(
                    "Turn hit the output token limit: text={has_text}, tool_calls={n_tool_calls}",
                    has_text=_has_text(message),
                    n_tool_calls=len(message.tool_calls or []),
                )

            if message_event.finish_reason == "content_filter":
                break

            if not message.tool_calls:
                truncated = message_event.finish_reason == "length" or bool(
                    (message.reasoning_content or "").strip()
                )
                if _has_text(message) or not truncated:
                    break

                empty_turns += 1
                if empty_turns > MAX_CONSECUTIVE_EMPTY_TURNS:
                    msg = (
                        f"Model produced neither text nor tool calls "
                        f"{empty_turns} turns in a row."
                    )
                    raise EmptyTurnLimitReachedError(msg)

                logger.warning(
                    "Empty turn ({n} of {max} allowed in a row); nudging the model.",
                    n=empty_turns,
                    max=MAX_CONSECUTIVE_EMPTY_TURNS,
                )
                yield MessageAddedEvent(
                    message=Message(role="user", content=EMPTY_TURN_NUDGE)
                )
                continue

            empty_turns = 0

            # Buffer the turn's tool events so the remaining counters can be
            # folded into its last tool result before any event is yielded.
            tool_events = [
                event
                async for event in _execute_tool_calls(
                    message.tool_calls, self._ctx.mcp_client
                )
            ]
            if deadline is not None and self._ctx.config.inject_time_remaining_counter:
                remaining = max(0.0, deadline - _monotonic())
                _append_time_remaining(tool_events, remaining)
            if (
                context_window_limit is not None
                and self._ctx.config.inject_context_remaining_counter
            ):
                remaining_context = max(0, context_window_limit - context_length)
                _append_context_remaining(tool_events, remaining_context)
            for event in tool_events:
                yield event

    async def stop(self) -> None:
        pass


def _has_text(message: Message) -> bool:
    """Whether the message carries any non-blank text for the reader.

    Reasoning doesn't count: a turn is just as stalled with it as without, and
    healthy tool-calling turns carry it too.
    """
    content = message.content
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            part.get("type") == "text" and str(part.get("text") or "").strip()
            for part in content
        )
    return False


def _append_time_remaining(events: list[Event], remaining_seconds: float) -> None:
    """Fold a remaining-time note into the last tool-result message of a turn."""
    note = f"Time remaining: {int(remaining_seconds)} seconds"
    for event in reversed(events):
        if isinstance(event, MessageAddedEvent):
            _append_text_content(event.message, note)
            return


def _append_context_remaining(events: list[Event], remaining_tokens: int) -> None:
    """Fold a remaining-context note into the last tool-result message of a turn."""
    note = f"Context remaining: {remaining_tokens}"
    for event in reversed(events):
        if isinstance(event, MessageAddedEvent):
            _append_text_content(event.message, note)
            return


def _append_text_content(message: Message, text: str) -> None:
    content = message.content
    if content is None:
        message.content = text
    elif isinstance(content, str):
        message.content = f"{content}\n\n{text}"
    else:
        content.append({"type": "text", "text": text})


async def _execute_tool_calls(
    tool_calls: Sequence[ChatCompletionMessageToolCall],
    mcp_client: Client[StreamableHttpTransport],
) -> AsyncGenerator[Event]:
    for tool_call in tool_calls:
        yield ToolCallStartedEvent(tool_call=tool_call)

        assert tool_call.function.name

        try:
            result = await mcp_client.call_tool_mcp(
                tool_call.function.name,
                json.loads(tool_call.function.arguments or "{}"),
            )
        except json.JSONDecodeError:
            # Sanitize the tool call arguments to prevent downstream API errors.
            # Some providers (e.g., TogetherAI) validate JSON in tool_call arguments
            # and reject requests containing malformed JSON.
            tool_call.function.arguments = json.dumps(
                {"_error": "malformed JSON in original arguments"}
            )
            result = _build_call_tool_result(
                result="Error: Invalid JSON in tool arguments",
                is_error=True,
            )

        yield ToolCallCompletedEvent(
            tool_call_id=tool_call.id,
            result=result,
        )

        result_dict = {}
        result_dict["role"] = "tool"
        result_dict["tool_call_id"] = tool_call.id

        if len(result.content) == 0:
            result_dict["content"] = []
        else:
            result_dict["content"] = [_to_chat_content_part(result.content[0])]

        yield MessageAddedEvent(message=Message(**result_dict))


def _to_chat_content_part(block: ContentBlock) -> dict[str, Any]:
    """Project an MCP content block onto a chat-completions content part.

    MCP blocks carry protocol-level fields (`annotations`, `_meta`) that are
    addressed to the host, not the model. Strict OpenAI-compatible endpoints
    (e.g. Fireworks) forbid unknown keys in content parts and reject the whole
    request, so this whitelists the fields a provider understands rather than
    dumping the model.
    """
    if isinstance(block, ImageContent):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
        }

    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}

    # Audio, resource links and embedded resources have no chat equivalent;
    # hand the model the block's own JSON instead of dropping it.
    return {
        "type": "text",
        "text": block.model_dump_json(by_alias=True, exclude_none=True),
    }


def _build_call_tool_result(result: str, is_error: bool) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=result)],
        structuredContent={"result": result},
        isError=is_error,
    )
