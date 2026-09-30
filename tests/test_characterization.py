"""Characterization tests that pin the runner's behavior across refactors.

They drive `EvaluationRunner.run()` with a scripted model against a real MCP
server subprocess and use only its public surface.

Golden files live in tests/resources/golden/. To regenerate them after an
intentional behavior change, run:

    UPDATE_GOLDEN=1 pytest tests/test_characterization.py

and review the diff of the golden files carefully.
"""

import copy
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, final, override

import pytest
from litellm import CustomStreamWrapper
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
from litellm.types.utils import Delta as LiteLlmDelta
from litellm.types.utils import Function as LiteLlmFunction

from karotte import Step, Task
from karotte.evaluation_runner import EvaluationRunner
from karotte.fake_model import setup_fake_model
from karotte.judges.always_pass_judge import AlwaysPassJudge
from karotte.judges.regex_judge import RegexJudge
from karotte.mcp_servers.http_mcp_server import HttpMcpServer
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.schemas.transcript import (
    ErrorEvent,
    Event,
    MessageAddedEvent,
    TaskCompletedEvent,
    ToolCallCompletedEvent,
)

_RESOURCE_DIR = Path(__file__).parent / "resources"
_GOLDEN_DIR = _RESOURCE_DIR / "golden"


# =============================================================================
# Golden-file helpers
# =============================================================================


def _normalize_events(events: list[Event]) -> list[dict[str, Any]]:
    """Serialize events for comparison, dropping the wall-clock timestamp."""
    normalized: list[dict[str, Any]] = []
    for event in events:
        dumped = event.model_dump(mode="json")
        dumped.pop("timestamp", None)
        normalized.append(dumped)
    return normalized


def _assert_matches_golden(name: str, data: Any) -> None:
    path = _GOLDEN_DIR / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
    assert path.exists(), (
        f"Golden file {path} is missing. Generate it with UPDATE_GOLDEN=1."
    )
    golden = json.loads(path.read_text())
    assert data == golden, (
        f"Behavior differs from golden file {name}. If the change is intentional, "
        "regenerate with UPDATE_GOLDEN=1 and review the golden diff."
    )


# =============================================================================
# Task fixtures: deterministic tools served by a real MCP server subprocess
# =============================================================================


class _StepA(Step):
    @property
    @override
    def instructions(self) -> str:
        return "Step A: echo something and state that step A is done."

    @property
    @override
    def judge(self) -> RegexJudge:
        return RegexJudge([re.compile(r"Step A done\.")])


class _StepB(Step):
    @property
    @override
    def instructions(self) -> str:
        return "Step B: exercise the remaining tools and state that step B is done."

    @property
    @override
    def judge(self) -> RegexJudge:
        return RegexJudge([re.compile(r"Step B done\.")])


@final
class CharacterizationTask(Task):
    id = "characterization-task"

    @property
    @override
    def system_prompt(self) -> str:
        return "You are being evaluated. Follow the step instructions."

    @property
    @override
    def tools(self) -> list[str]:
        return ["echo", "get_image", "empty_result"]

    @property
    @override
    def steps(self) -> list[Step]:
        return [_StepA(config=self.config), _StepB(config=self.config)]


class _SingleStep(Step):
    @property
    @override
    def instructions(self) -> str:
        return "Single step: echo, then show an image."

    @property
    @override
    def judge(self) -> AlwaysPassJudge:
        return AlwaysPassJudge()


@final
class SingleStepTask(Task):
    id = "characterization-single-step-task"

    @property
    @override
    def system_prompt(self) -> str:
        return "You are being evaluated. Follow the step instructions."

    @property
    @override
    def tools(self) -> list[str]:
        return ["echo", "get_image"]

    @property
    @override
    def steps(self) -> list[Step]:
        return [_SingleStep(config=self.config)]


def _make_config(
    mcp_server: HttpMcpServer, tmp_path: Path, **overrides: Any
) -> EvaluationRunConfig:
    kwargs: dict[str, Any] = {
        "run_id": "characterization-run",
        "task_id": "characterization-task",
        "model": "test_model",
        "model_api_key": "test_key",
        "mcp_server_config": HttpMcpServerConfig(port=mcp_server.config.port),
        "transcript_file": (tmp_path / "transcript.json").as_posix(),
        **overrides,
    }
    return EvaluationRunConfig(**kwargs)


def _install_fake_messages(
    monkeypatch: pytest.MonkeyPatch, messages: list[Message]
) -> None:
    """Provide `environment.fake_model.get_messages` the way a real env does."""

    def get_messages(_config: EvaluationRunConfig) -> list[Message]:
        return messages

    module = ModuleType("environment.fake_model")
    module.get_messages = get_messages  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "environment.fake_model", module)


def _tool_call(
    call_id: str, name: str, arguments: str
) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


async def _run_to_completion(
    config: EvaluationRunConfig,
    task: Task,
    monkeypatch: pytest.MonkeyPatch,
    messages: list[Message],
) -> list[Event]:
    _install_fake_messages(monkeypatch, messages)
    runner = EvaluationRunner(config, task)
    setup_fake_model(runner, config)
    return [event async for event in runner.run()]


# =============================================================================
# Golden test 1: full multi-step run through the fake-model message source
# =============================================================================


@pytest.mark.asyncio
async def test_golden_event_sequence_full_run(
    characterization_mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Pins the complete ordered event stream of a two-step run covering:
    multiple tool calls per turn, image tool results, empty tool results,
    and malformed tool-call JSON."""
    config = _make_config(characterization_mcp_server, tmp_path, use_fake_model=True)
    messages = [
        # Step A, turn 1: two tool calls in one message.
        Message(
            role="assistant",
            content="Echoing twice.",
            tool_calls=[
                _tool_call("call_1", "echo", json.dumps({"text": "hello"})),
                _tool_call("call_2", "echo", json.dumps({"text": "world"})),
            ],
        ),
        # Step A, turn 2: no tool calls ends the step.
        Message(role="assistant", content="Step A done."),
        # Step B, turn 1: image result shaping.
        Message(
            role="assistant",
            content="Fetching an image.",
            tool_calls=[_tool_call("call_3", "get_image", "{}")],
        ),
        # Step B, turn 2: empty result and malformed JSON arguments.
        Message(
            role="assistant",
            content="Covering the remaining edge cases.",
            tool_calls=[
                _tool_call("call_4", "empty_result", "{}"),
                _tool_call("call_5", "echo", '{"text": broken'),
            ],
        ),
        Message(role="assistant", content="Step B done."),
    ]

    events = await _run_to_completion(
        config, CharacterizationTask(config), monkeypatch, messages
    )

    _assert_matches_golden("full_run_events.json", _normalize_events(events))


# =============================================================================
# Golden test 2: litellm boundary — streaming, message assembly, token usage,
# and the completion-params serialization round-trip
# =============================================================================


@final
class _ScriptedStream(CustomStreamWrapper):
    """Passes the runner's isinstance check while yielding scripted chunks."""

    def __init__(self, chunks: list[ModelResponseStream]):  # pyright: ignore[reportMissingSuperCall]
        self._chunks = chunks

    @override
    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


def _stream_chunk(
    delta: LiteLlmDelta,
    finish_reason: str | None = None,
    usage: Usage | None = None,
) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-characterization",
        created=1735689600,
        model="test_model",
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )


def _tool_call_turn(
    call_id: str, name: str, arguments_parts: list[str], usage: Usage
) -> list[ModelResponseStream]:
    """A scripted turn: role/content chunks, a tool call whose arguments are
    split across chunks, and a final usage chunk."""
    chunks = [
        _stream_chunk(LiteLlmDelta(role="assistant", content="Calling ")),
        _stream_chunk(LiteLlmDelta(content=f"{name}.")),
        _stream_chunk(
            LiteLlmDelta(
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id=call_id,
                        type="function",
                        index=0,
                        function=LiteLlmFunction(
                            name=name, arguments=arguments_parts[0]
                        ),
                    )
                ]
            )
        ),
    ]
    for part in arguments_parts[1:]:
        chunks.append(
            _stream_chunk(
                LiteLlmDelta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=None,
                            type=None,
                            index=0,
                            function=LiteLlmFunction(name=None, arguments=part),
                        )
                    ]
                )
            )
        )
    chunks.append(
        _stream_chunk(LiteLlmDelta(), finish_reason="tool_calls", usage=usage)
    )
    return chunks


@pytest.mark.asyncio
async def test_golden_llm_stream_and_completion_params(
    characterization_mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Drives a full run through the real litellm code path (streaming, chunk
    assembly, token usage) with `litellm.acompletion` scripted at the library
    boundary. Pins both the emitted event stream and the completion params of
    the final call, which characterizes the message serialization round-trip
    (tool schema conversion, tool-result messages, image content, and inline
    cache_control placement)."""
    # The builtin path routes non-Anthropic models through KAROTTE_PROXY_URL when
    # set; keep this golden independent of the ambient environment.
    monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
    config = _make_config(
        characterization_mcp_server,
        tmp_path,
        task_id="characterization-single-step-task",
    )

    turns = [
        _tool_call_turn(
            "call_echo",
            "echo",
            ['{"text": ', '"hello"}'],
            usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        ),
        _tool_call_turn(
            "call_image",
            "get_image",
            ["{}"],
            usage=Usage(prompt_tokens=23, completion_tokens=5, total_tokens=28),
        ),
        [
            _stream_chunk(LiteLlmDelta(role="assistant", content="All ")),
            _stream_chunk(LiteLlmDelta(content="done.")),
            _stream_chunk(
                LiteLlmDelta(),
                finish_reason="stop",
                usage=Usage(prompt_tokens=31, completion_tokens=3, total_tokens=34),
            ),
        ],
    ]
    turn_iter = iter(turns)
    captured_params: list[dict[str, Any]] = []

    async def scripted_acompletion(**kwargs: Any) -> _ScriptedStream:
        captured_params.append(copy.deepcopy(kwargs))
        return _ScriptedStream(next(turn_iter))

    monkeypatch.setattr("litellm.acompletion", scripted_acompletion)

    runner = EvaluationRunner(config, SingleStepTask(config))
    events = [event async for event in runner.run()]

    assert len(captured_params) == 3
    _assert_matches_golden("llm_stream_events.json", _normalize_events(events))
    _assert_matches_golden("llm_completion_params.json", captured_params[-1])


# =============================================================================
# Behavioral characterization of individual seams
# =============================================================================


def _events_of_type[T](events: list[Event], event_type: type[T]) -> list[T]:
    return [e for e in events if isinstance(e, event_type)]


@pytest.mark.asyncio
async def test_run_fails_when_judge_does_not_match(
    characterization_mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A step whose judge finds no matching model output fails the task."""
    config = _make_config(characterization_mcp_server, tmp_path, use_fake_model=True)
    messages = [
        Message(role="assistant", content="Nothing the judge would accept."),
    ]

    events = await _run_to_completion(
        config, CharacterizationTask(config), monkeypatch, messages
    )

    completed = _events_of_type(events, TaskCompletedEvent)
    assert len(completed) == 1
    assert completed[0].status == "failed"


@pytest.mark.asyncio
async def test_malformed_json_sanitizes_arguments_in_transcript(
    characterization_mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Malformed tool-call JSON produces an error result and rewrites the
    tool call's arguments in place so later provider requests stay valid."""
    config = _make_config(characterization_mcp_server, tmp_path, use_fake_model=True)
    malformed_call = _tool_call("call_1", "echo", '{"text": broken')
    messages = [
        Message(
            role="assistant",
            content="Calling echo with malformed JSON.",
            tool_calls=[malformed_call],
        ),
        Message(role="assistant", content="Done."),
        Message(role="assistant", content="Done."),
    ]

    events = await _run_to_completion(
        config, CharacterizationTask(config), monkeypatch, messages
    )

    tool_results = _events_of_type(events, ToolCallCompletedEvent)
    assert len(tool_results) == 1
    assert tool_results[0].result.isError
    assert malformed_call.function.arguments == json.dumps(
        {"_error": "malformed JSON in original arguments"}
    )


@pytest.mark.asyncio
async def test_turn_limit_ends_run_with_error(
    characterization_mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _make_config(
        characterization_mcp_server, tmp_path, use_fake_model=True, turn_limit=1
    )
    messages = [
        Message(
            role="assistant",
            content="First turn.",
            tool_calls=[_tool_call("call_1", "echo", '{"text": "hi"}')],
        ),
        Message(role="assistant", content="Second turn never happens."),
    ]

    events = await _run_to_completion(
        config, CharacterizationTask(config), monkeypatch, messages
    )

    errors = _events_of_type(events, ErrorEvent)
    assert len(errors) == 1
    assert errors[0].exception_type == "TurnLimitReachedError"
    assert "Turn limit of 1 reached." in errors[0].message
    completed = _events_of_type(events, TaskCompletedEvent)
    assert len(completed) == 1
    assert completed[0].status == "error"


@pytest.mark.asyncio
async def test_image_and_empty_tool_results_shape_messages(
    characterization_mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Image tool results become image_url data URLs; empty results become
    empty content lists."""
    config = _make_config(characterization_mcp_server, tmp_path, use_fake_model=True)
    messages = [
        Message(
            role="assistant",
            content="Image and empty results.",
            tool_calls=[
                _tool_call("call_1", "get_image", "{}"),
                _tool_call("call_2", "empty_result", "{}"),
            ],
        ),
        Message(role="assistant", content="Done."),
        Message(role="assistant", content="Done."),
    ]

    events = await _run_to_completion(
        config, CharacterizationTask(config), monkeypatch, messages
    )

    tool_messages = [
        e.message
        for e in _events_of_type(events, MessageAddedEvent)
        if e.message.role == "tool"
    ]
    assert len(tool_messages) == 2

    image_message, empty_message = tool_messages
    assert image_message.tool_call_id == "call_1"
    assert isinstance(image_message.content, list)
    assert image_message.content[0]["type"] == "image_url"
    assert image_message.content[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )

    assert empty_message.tool_call_id == "call_2"
    assert empty_message.content == []


@pytest.mark.asyncio
async def test_run_passes_when_judges_match(
    characterization_mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _make_config(characterization_mcp_server, tmp_path, use_fake_model=True)
    messages = [
        Message(role="assistant", content="Step A done."),
        Message(role="assistant", content="Step B done."),
    ]

    events = await _run_to_completion(
        config, CharacterizationTask(config), monkeypatch, messages
    )

    completed = _events_of_type(events, TaskCompletedEvent)
    assert completed[0].status == "passed"
