from collections.abc import AsyncGenerator, Sequence
from typing import Any, final

import pytest
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from pydantic import AnyUrl

from karotte.agents.agent import EmptyTurnLimitReachedError, RunContext
from karotte.agents.message_loop import (
    EMPTY_TURN_NUDGE,
    MAX_CONSECUTIVE_EMPTY_TURNS,
    MessageLoopAgent,
    _to_chat_content_part,  # pyright: ignore[reportPrivateUsage]
)
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    ToolCallStartedEvent,
    Transcript,
)

# Fields MCP addresses to the host. Strict OpenAI-compatible endpoints (e.g.
# Fireworks) forbid unknown keys in content parts and reject the whole request,
# so none of them may reach a provider.
_MCP_ONLY_KEYS = {"annotations", "meta", "_meta"}


def test_text_content_carries_no_mcp_only_keys():
    block = TextContent(type="text", text='{"stdout":"Hello!\\n","exit_code":0}')

    part = _to_chat_content_part(block)

    assert part == {"type": "text", "text": '{"stdout":"Hello!\\n","exit_code":0}'}


def test_annotated_text_content_carries_no_mcp_only_keys():
    block = TextContent(
        type="text",
        text="hello",
        annotations={"audience": ["assistant"], "priority": 1.0},  # pyright: ignore[reportArgumentType]
        _meta={"trace": "abc"},
    )

    part = _to_chat_content_part(block)

    assert part == {"type": "text", "text": "hello"}


def test_image_content_becomes_a_data_url():
    block = ImageContent(type="image", data="Ym9ndXM=", mimeType="image/png")

    part = _to_chat_content_part(block)

    assert part == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,Ym9ndXM="},
    }


def test_audio_content_falls_back_to_json_text():
    block = AudioContent(type="audio", data="Ym9ndXM=", mimeType="audio/wav")

    part = _to_chat_content_part(block)

    assert part["type"] == "text"
    assert "audio/wav" in part["text"]


def test_embedded_resource_falls_back_to_json_text():
    block = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(uri=AnyUrl("file:///tmp/x.bin"), blob="Ym9ndXM="),
    )

    part = _to_chat_content_part(block)

    assert part["type"] == "text"
    assert "file:///tmp/x.bin" in part["text"]


def test_no_content_block_type_leaks_mcp_only_keys():
    blocks = [
        TextContent(type="text", text="hello"),
        ImageContent(type="image", data="Ym9ndXM=", mimeType="image/png"),
        AudioContent(type="audio", data="Ym9ndXM=", mimeType="audio/wav"),
    ]

    for block in blocks:
        assert not _MCP_ONLY_KEYS & _to_chat_content_part(block).keys()


# =============================================================================
# Empty-turn nudging
#
# A turn carrying neither text nor tool calls leaves the loop nothing to do.
# It happens when the model spends its whole output budget thinking, which the
# provider reports as finish_reason="length" -- and sometimes, wrongly, as
# "stop". Ending the step there scores the run on work the model never got to
# submit, so the loop nudges instead.
# =============================================================================


def _assistant(
    content: str | list[dict[str, Any]] | None = None,
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
    reasoning: str | None = None,
) -> Message:
    return Message(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning,
    )


def _tool_call(call_id: str = "call_1") -> list[ChatCompletionMessageToolCall]:
    return [
        ChatCompletionMessageToolCall(
            id=call_id,
            type="function",
            function=Function(name="bash", arguments='{"command": "echo hi"}'),
        )
    ]


@final
class _ScriptedSource:
    """Yields one scripted turn per call, with its finish_reason."""

    def __init__(self, turns: Sequence[tuple[Message, str | None]]) -> None:
        self._turns = turns
        self.calls = 0
        self.histories: list[list[Message]] = []

    async def collect(
        self,
        messages: list[Message],
        tools: list[Any],  # pyright: ignore[reportUnusedParameter]
    ) -> AsyncGenerator[Event]:
        if self.calls >= len(self._turns):
            raise AssertionError("loop asked for more turns than the script has")
        self.histories.append(list(messages))
        message, finish_reason = self._turns[self.calls]
        self.calls += 1
        yield MessageAddedEvent(message=message, finish_reason=finish_reason)


@final
class _StubMcpClient:
    """Stands in for the MCP client; every tool call succeeds with "ok"."""

    async def call_tool_mcp(
        self,
        name: str,  # pyright: ignore[reportUnusedParameter]
        arguments: dict[str, Any],  # pyright: ignore[reportUnusedParameter]
    ) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text="ok")])


async def _run(
    turns: Sequence[tuple[Message, str | None]],
    config: EvaluationRunConfig,
) -> tuple[list[Event], _ScriptedSource]:
    """Drive one step over `turns`, mirroring the runner's transcript upkeep."""
    source = _ScriptedSource(turns)
    agent = MessageLoopAgent(source)
    transcript = Transcript(run_id=config.run_id)
    await agent.start(
        RunContext(
            config=config,
            mcp_client=_StubMcpClient(),  # pyright: ignore[reportArgumentType]
            tools=[],
            transcript=transcript,
        )
    )

    events: list[Event] = []
    async for event in agent.run_step("Do the thing"):
        events.append(event)
        transcript.events.append(event)
    return events, source


def _nudges(events: list[Event]) -> list[str]:
    return [
        e.message.content
        for e in events
        if isinstance(e, MessageAddedEvent)
        and e.message.role == "user"
        and isinstance(e.message.content, str)
        and e.message.content == EMPTY_TURN_NUDGE
    ]


@pytest.mark.asyncio
async def test_turn_with_neither_text_nor_tool_calls_is_nudged(
    sample_config: EvaluationRunConfig,
):
    """The production shape: a full budget of thinking, nothing emitted."""
    events, source = await _run(
        [
            (_assistant(content="", reasoning="a" * 5000), "length"),
            (_assistant(content="Done."), "stop"),
        ],
        sample_config,
    )

    assert len(_nudges(events)) == 1
    assert source.calls == 2


@pytest.mark.asyncio
async def test_mislabelled_truncation_is_nudged_too(
    sample_config: EvaluationRunConfig,
):
    """Some truncated turns come back as finish_reason="stop"; the turn's shape
    is what decides, not the label."""
    events, _ = await _run(
        [
            (_assistant(content=None, reasoning="a" * 5000), "stop"),
            (_assistant(content="Done."), "stop"),
        ],
        sample_config,
    )

    assert len(_nudges(events)) == 1


@pytest.mark.asyncio
async def test_empty_message_without_truncation_ends_the_step(
    sample_config: EvaluationRunConfig,
):
    """Env-authored fake models end a step with an empty assistant message and no
    finish_reason. Nothing was cut off there, so it ends the step as it always
    did rather than nudging a source that has no further messages to give."""
    events, source = await _run([(_assistant(content=""), None)], sample_config)

    assert _nudges(events) == []
    assert source.calls == 1


@pytest.mark.asyncio
async def test_a_refusal_ends_the_step(sample_config: EvaluationRunConfig):
    """A declined request comes back empty; nudging only asks a model that
    deliberately refused to refuse again."""
    events, source = await _run(
        [(_assistant(content="", reasoning="a" * 100), "content_filter")],
        sample_config,
    )

    assert _nudges(events) == []
    assert source.calls == 1


@pytest.mark.asyncio
async def test_a_refusal_drops_the_tool_calls_it_came_with(
    sample_config: EvaluationRunConfig,
):
    """A declined turn ends the step even when it carries tool calls -- running
    them would act on the request the provider just refused."""
    events, source = await _run(
        [(_assistant(content="", tool_calls=_tool_call()), "content_filter")],
        sample_config,
    )

    assert [e for e in events if isinstance(e, ToolCallStartedEvent)] == []
    assert source.calls == 1


@pytest.mark.asyncio
async def test_text_without_tool_calls_ends_the_step(
    sample_config: EvaluationRunConfig,
):
    """The normal way a step ends -- nudging here would never let it finish."""
    events, source = await _run(
        [(_assistant(content="All done."), "stop")],
        sample_config,
    )

    assert _nudges(events) == []
    assert source.calls == 1


@pytest.mark.asyncio
async def test_tool_calls_without_text_are_not_nudged(
    sample_config: EvaluationRunConfig,
):
    """Text-free tool-calling turns are the single most common healthy shape."""
    events, _ = await _run(
        [
            (
                _assistant(content="", tool_calls=_tool_call(), reasoning="x"),
                "tool_calls",
            ),
            (_assistant(content="Done."), "stop"),
        ],
        sample_config,
    )

    assert _nudges(events) == []


@pytest.mark.asyncio
async def test_list_content_carrying_text_ends_the_step(
    sample_config: EvaluationRunConfig,
):
    events, _ = await _run(
        [(_assistant(content=[{"type": "text", "text": "All done."}]), "stop")],
        sample_config,
    )

    assert _nudges(events) == []


@pytest.mark.asyncio
async def test_list_content_carrying_no_text_is_nudged(
    sample_config: EvaluationRunConfig,
):
    events, _ = await _run(
        [
            (_assistant(content=[{"type": "text", "text": "   "}]), "length"),
            (_assistant(content="Done."), "stop"),
        ],
        sample_config,
    )

    assert len(_nudges(events)) == 1


@pytest.mark.asyncio
async def test_nudging_gives_up_after_a_bounded_number_of_empty_turns(
    sample_config: EvaluationRunConfig,
):
    empty = (_assistant(content="", reasoning="a" * 100), "length")

    with pytest.raises(EmptyTurnLimitReachedError):
        await _run([empty] * (MAX_CONSECUTIVE_EMPTY_TURNS + 1), sample_config)


@pytest.mark.asyncio
async def test_a_productive_turn_resets_the_empty_turn_budget(
    sample_config: EvaluationRunConfig,
):
    """Only *consecutive* empty turns count, so a model that recovers and later
    stalls again gets the full allowance a second time."""
    empty = (_assistant(content="", reasoning="a" * 100), "length")
    productive = (_assistant(content="", tool_calls=_tool_call()), "tool_calls")

    events, _ = await _run(
        [*[empty] * MAX_CONSECUTIVE_EMPTY_TURNS, productive]
        + [
            *[empty] * MAX_CONSECUTIVE_EMPTY_TURNS,
            (_assistant(content="Done."), "stop"),
        ],
        sample_config,
    )

    assert len(_nudges(events)) == MAX_CONSECUTIVE_EMPTY_TURNS * 2


@pytest.mark.asyncio
async def test_nudge_is_addressed_to_the_model_as_a_user_message(
    sample_config: EvaluationRunConfig,
):
    """The nudge has to reach the model as conversation, not as a bare log line."""
    events, _ = await _run(
        [
            (_assistant(content="", reasoning="a" * 100), "length"),
            (_assistant(content="Done."), "stop"),
        ],
        sample_config,
    )

    nudge = next(
        e
        for e in events
        if isinstance(e, MessageAddedEvent)
        and e.message.role == "user"
        and e.message.content != "Do the thing"
    )
    assert isinstance(nudge.message.content, str)
    assert nudge.message.content.strip()


@pytest.mark.asyncio
async def test_nudge_reaches_the_model_on_the_following_turn(
    sample_config: EvaluationRunConfig,
):
    """Yielding the nudge is only half of it -- it has to land in the history the
    source is handed next, otherwise the retry repeats the stalled turn."""
    _, source = await _run(
        [
            (_assistant(content="", reasoning="a" * 100), "length"),
            (_assistant(content="Done."), "stop"),
        ],
        sample_config,
    )

    assert source.histories[0][-1].content == "Do the thing"
    assert source.histories[1][-1].role == "user"
    assert source.histories[1][-1].content == EMPTY_TURN_NUDGE
