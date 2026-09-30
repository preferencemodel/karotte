import json
import os
from pathlib import Path
from typing import Any, final
from unittest.mock import AsyncMock, Mock, patch

import httpx
import litellm
import litellm.exceptions
import pytest
from litellm import Choices, ModelResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.utils import Message as LiteLlmMessage
from loguru import logger
from mcp.types import TextContent

from karotte import Step, Task
from karotte.agents import BackendSource, BuiltinSource
from karotte.agents.backend_source import EXTERNAL_MESSAGE_TIMEOUT_S
from karotte.agents.builtin_source import (
    LLM_RETRY_AFTER_MAX_S,
    LLM_RETRY_WAIT_MIN_S,
    _retry_after,  # pyright: ignore[reportPrivateUsage]
    _retry_wait,  # pyright: ignore[reportPrivateUsage]
)
from karotte.agents.message_loop import (
    _execute_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from karotte.confinement import Contract
from karotte.evaluation_runner import (
    EvaluationRunner,
    _log_confinement,  # pyright: ignore[reportPrivateUsage]
    _save_extra_artifacts,  # pyright: ignore[reportPrivateUsage]
)
from karotte.mcp_servers.http_mcp_server import HttpMcpServer
from karotte.model_spec import spec_for
from karotte.providers import (
    SERVICE_TIER_ENV,
    AnthropicProvider,
    _fix_xai_arguments,  # pyright: ignore[reportPrivateUsage]
    _strip_deepseek_tokens,  # pyright: ignore[reportPrivateUsage]
    provider_for,
    repair_tool_calls,
)
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.schemas.transcript import (
    ErrorEvent,
    MessageAddedEvent,
    MessageChunkEvent,
    MessageChunkResetEvent,
    ScoringEvent,
    StepCompletedEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskPreHookCompletedEvent,
    TaskStartedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    Transcript,
)


def _make_test_task(config: EvaluationRunConfig | None = None) -> Task:
    """Create a TestTask for tests that don't exercise the task."""
    from tests.conftest import TestTask

    if config is None:
        config = EvaluationRunConfig(
            run_id="test_run",
            task_id="test-task",
            model="test_model",
            model_api_key="test_key",
            mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
            transcript_file="",
        )
    return TestTask(config)


def _collect(runner: EvaluationRunner):
    """Drive the builtin source's turn using a runner's config/transcript/tools."""
    return BuiltinSource(runner.config).collect(
        runner.transcript.messages, runner.tools
    )


def _completion_params(runner: EvaluationRunner):
    """Build the builtin source's completion params from a runner's config/transcript/tools."""
    return BuiltinSource(runner.config).get_completion_params(
        runner.transcript.messages, runner.tools
    )


def test_raises_error_if_task_not_found(sample_config: EvaluationRunConfig):
    from karotte.load_tasks import load_task

    sample_config.task_id = "nonexistent-task"

    with pytest.raises(ValueError, match="nonexistent-task"):
        load_task(sample_config)


def test_builtin_agent_disallows_student_mcp_access(
    sample_config: EvaluationRunConfig,
):
    runner = EvaluationRunner(sample_config, _make_test_task(sample_config))
    assert runner.allows_student_mcp_access is False


def test_external_agent_disallows_student_mcp_access(
    sample_config: EvaluationRunConfig,
):
    config = sample_config.model_copy(
        update={"model": "pt/foo", "backend_uri": "http://backend"}
    )
    runner = EvaluationRunner(config, _make_test_task(config))
    assert runner.allows_student_mcp_access is False


def test_cli_agent_allows_student_mcp_access(sample_config: EvaluationRunConfig):
    config = sample_config.model_copy(update={"agent": "mistral-vibe"})
    runner = EvaluationRunner(config, _make_test_task(config))
    assert runner.allows_student_mcp_access is True


def test_save_extra_artifacts_noop_when_absent(sample_config: EvaluationRunConfig):
    with patch("karotte.evaluation_runner.save_artifact") as mock_save:
        _save_extra_artifacts(sample_config)
        _save_extra_artifacts(sample_config.model_copy(update={"extra_config": {}}))
    mock_save.assert_not_called()


def test_save_extra_artifacts_saves_existing_string_path(
    sample_config: EvaluationRunConfig, tmp_path: Path
):
    summary = tmp_path / "reasoning_summary.md"
    summary.write_text("my reasoning")
    config = sample_config.model_copy(
        update={"extra_config": {"extra_artifact_paths": str(summary)}}
    )

    with patch("karotte.evaluation_runner.save_artifact") as mock_save:
        _save_extra_artifacts(config)

    mock_save.assert_called_once_with(config, summary)


def test_save_extra_artifacts_skips_missing_and_saves_existing(
    sample_config: EvaluationRunConfig, tmp_path: Path
):
    present = tmp_path / "present.md"
    present.write_text("here")
    missing = tmp_path / "missing.md"
    config = sample_config.model_copy(
        update={"extra_config": {"extra_artifact_paths": [str(present), str(missing)]}}
    )

    with patch("karotte.evaluation_runner.save_artifact") as mock_save:
        _save_extra_artifacts(config)

    # Only the existing path is uploaded; the missing one is skipped, not fatal.
    mock_save.assert_called_once_with(config, present)


@pytest.mark.asyncio
async def test_no_crash_on_empty_steps(
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    @final
    class EmptyTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "empty-task"

        @property
        def steps(self):
            return []

        @property
        def tools(self):
            return []

    sample_config.task_id = "empty-task"
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, EmptyTask(sample_config))

    event = None
    async for event in runner.run():
        pass

    assert event
    assert isinstance(event, TaskCompletedEvent)
    assert event.status == "passed"


@final
class _BashTask(Task):
    @property
    def system_prompt(self) -> str | None:
        return None

    id = "bash-task"

    @property
    def steps(self):
        return []

    @property
    def tools(self):
        return ["bash"]


def _loaded_tool_names(runner: EvaluationRunner) -> list[str]:
    return [t["function"]["name"] for t in runner.tools]


@pytest.mark.asyncio
async def test_builtin_agent_registers_task_bash_tool(
    sample_config: EvaluationRunConfig, mcp_server: HttpMcpServer
):
    sample_config.task_id = "bash-task"
    sample_config.mcp_server_config.port = mcp_server.config.port
    runner = EvaluationRunner(sample_config, _BashTask(sample_config))

    async for _ in runner.prepare():
        pass

    assert "bash" in _loaded_tool_names(runner)


@pytest.mark.asyncio
async def test_registering_tools_logs_no_schema_warning(
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    caplog: pytest.LogCaptureFixture,
):
    sample_config.task_id = "bash-task"
    sample_config.mcp_server_config.port = mcp_server.config.port
    runner = EvaluationRunner(sample_config, _BashTask(sample_config))

    async for _ in runner.prepare():
        pass

    assert "not listed by server" not in caplog.text


@pytest.mark.asyncio
async def test_prepare_logs_the_confinement_in_force_once(
    sample_config: EvaluationRunConfig, mcp_server: HttpMcpServer
):
    sample_config.task_id = "bash-task"
    sample_config.mcp_server_config.port = mcp_server.config.port
    runner = EvaluationRunner(sample_config, _BashTask(sample_config))
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        async for _ in runner.prepare():
            pass
    finally:
        logger.remove(handler)

    assert len([m for m in messages if "Confinement: " in m]) == 1


@pytest.mark.parametrize(
    ("contract", "level"), [(Contract.PREVENTED, "INFO"), (Contract.REAPED, "WARNING")]
)
def test_confinement_line_warns_only_when_something_is_off(
    monkeypatch: pytest.MonkeyPatch, contract: Contract, level: str
):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.setenv("KAROTTE_SANDBOX", "runc")
    monkeypatch.setattr(
        "karotte.evaluation_runner.ipc_namespace_available", lambda: True
    )
    records: list[tuple[str, str]] = []
    handler = logger.add(
        lambda m: records.append((m.record["level"].name, m.record["message"])),
        level="INFO",
    )
    try:
        _log_confinement(
            {"memory": contract, "processes": contract, "files": contract}, True
        )
    finally:
        logger.remove(handler)

    assert [lvl for lvl, msg in records if msg.startswith("Confinement: ")] == [level]


@final
class _SystemPromptTask(Task):
    id = "bash-task"

    @property
    def system_prompt(self) -> str:
        return "karotte system prompt"

    @property
    def steps(self):
        return []

    @property
    def tools(self):
        return []


def _system_messages(events: list[Any]) -> list[Message]:
    return [
        e.message
        for e in events
        if isinstance(e, MessageAddedEvent) and e.message.role == "system"
    ]


@pytest.mark.asyncio
async def test_builtin_agent_emits_task_system_prompt(
    sample_config: EvaluationRunConfig, mcp_server: HttpMcpServer
):
    sample_config.task_id = "bash-task"
    sample_config.mcp_server_config.port = mcp_server.config.port
    runner = EvaluationRunner(sample_config, _SystemPromptTask(sample_config))

    events = [e async for e in runner.prepare()]

    messages = _system_messages(events)
    assert len(messages) == 1
    assert messages[0].content == "karotte system prompt"


@pytest.mark.asyncio
async def test_cli_agent_omits_task_system_prompt(
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """CLI agents send their own system prompt; karotte's must not be emitted."""
    monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
    config = sample_config.model_copy(update={"agent": "mistral-vibe"})
    config.task_id = "bash-task"
    config.mcp_server_config.port = mcp_server.config.port
    runner = EvaluationRunner(config, _SystemPromptTask(config))

    events = [e async for e in runner.prepare()]

    assert _system_messages(events) == []


@pytest.mark.asyncio
async def test_vibe_agent_omits_its_native_bash_tool(
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Vibe supplies its own bash, so karotte must not register a competing one."""
    monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
    config = sample_config.model_copy(update={"agent": "mistral-vibe"})
    config.task_id = "bash-task"
    config.mcp_server_config.port = mcp_server.config.port
    runner = EvaluationRunner(config, _BashTask(config))

    async for _ in runner.prepare():
        pass

    assert "bash" not in _loaded_tool_names(runner)


@pytest.mark.asyncio
async def test_task_pre_hook_gets_executed(
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    tmp_path: Path,
):
    @final
    class PreHookTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "pre-hook-task"

        @property
        def steps(self):
            return []

        @property
        def tools(self):
            return []

        def pre_hook(self):
            return {"task_id": self.id}

    transcript_file = tmp_path / "transcript.json"

    sample_config.task_id = "pre-hook-task"
    sample_config.mcp_server_config.port = mcp_server.config.port
    sample_config.transcript_file = transcript_file.as_posix()

    runner = EvaluationRunner(sample_config, PreHookTask(sample_config))

    passed = False
    async for event in runner.run():
        if isinstance(event, TaskPreHookCompletedEvent) and event.metadata == {
            "task_id": "pre-hook-task"
        }:
            passed = True

    assert passed is True


@pytest.mark.asyncio
async def test_exception_before_transcript_initialized(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    test_task: Task,
):
    def failing_configure_tools(self):  # pyright: ignore[reportMissingParameterType,reportUnusedParameter]
        raise ValueError("configure_tools failed")

    monkeypatch.setattr(type(test_task), "configure_tools", failing_configure_tools)

    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, test_task)

    events = []
    async for event in runner.run():
        events.append(event)

    assert len(events) == 2

    error_event = events[0]
    assert isinstance(error_event, ErrorEvent)
    assert error_event.exception_type == "ValueError"
    assert error_event.message == "configure_tools failed"
    assert error_event.traceback is not None
    assert "ValueError: configure_tools failed" in error_event.traceback
    assert "Traceback (most recent call last):" in error_event.traceback

    task_completed_event = events[1]
    assert isinstance(task_completed_event, TaskCompletedEvent)
    assert task_completed_event.status == "error"


@pytest.mark.asyncio
async def test_exception_after_transcript_initialized(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    tmp_path: Path,
    test_task: Task,
):
    async def mock_execute_step(self, step, step_index):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        raise RuntimeError("Step execution failed")
        yield  # pyright: ignore[reportUnreachable]

    monkeypatch.setattr(EvaluationRunner, "_execute_step", mock_execute_step)

    transcript_file = tmp_path / "transcript.json"

    sample_config.mcp_server_config.port = mcp_server.config.port
    sample_config.transcript_file = transcript_file.as_posix()

    runner = EvaluationRunner(sample_config, test_task)

    events = []
    async for event in runner.run():
        events.append(event)

    error_event = None
    task_completed_event = None
    for event in events:
        if isinstance(event, ErrorEvent):
            error_event = event
        if isinstance(event, TaskCompletedEvent):
            task_completed_event = event

    assert error_event is not None
    assert error_event.exception_type == "RuntimeError"
    assert error_event.message == "Step execution failed"
    assert error_event.traceback is not None
    assert "RuntimeError: Step execution failed" in error_event.traceback
    assert "Traceback (most recent call last):" in error_event.traceback

    assert task_completed_event is not None
    assert task_completed_event.status == "error"

    assert transcript_file.is_file()
    transcript = Transcript.model_validate_json(transcript_file.read_text())

    error_events_in_transcript = [
        e for e in transcript.events if isinstance(e, ErrorEvent)
    ]
    assert len(error_events_in_transcript) == 1
    assert error_events_in_transcript[0].exception_type == "RuntimeError"
    assert error_events_in_transcript[0].message == "Step execution failed"
    assert error_events_in_transcript[0].traceback is not None
    assert (
        "RuntimeError: Step execution failed" in error_events_in_transcript[0].traceback
    )


@pytest.mark.asyncio
async def test_keyboard_interrupt_handling(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    tmp_path: Path,
    test_task: Task,
):
    async def mock_execute_step(self, step, step_index):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        raise KeyboardInterrupt()
        yield  # pyright: ignore[reportUnreachable]

    monkeypatch.setattr(EvaluationRunner, "_execute_step", mock_execute_step)

    transcript_file = tmp_path / "transcript.json"

    sample_config.mcp_server_config.port = mcp_server.config.port
    sample_config.transcript_file = transcript_file.as_posix()

    runner = EvaluationRunner(sample_config, test_task)

    events = []
    async for event in runner.run():
        events.append(event)

    error_event = None
    task_completed_event = None
    for event in events:
        if isinstance(event, ErrorEvent):
            error_event = event
        if isinstance(event, TaskCompletedEvent):
            task_completed_event = event

    assert error_event is not None
    assert error_event.exception_type == "KeyboardInterrupt"
    assert error_event.message == ""

    assert task_completed_event is not None
    assert task_completed_event.status == "error"

    assert transcript_file.is_file()
    transcript = Transcript.model_validate_json(transcript_file.read_text())

    error_events_in_transcript = [
        e for e in transcript.events if isinstance(e, ErrorEvent)
    ]
    assert len(error_events_in_transcript) == 1
    assert error_events_in_transcript[0].exception_type == "KeyboardInterrupt"


@pytest.mark.asyncio
async def test_transcript_not_saved_when_error_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    tmp_path: Path,
    test_task: Task,
):
    def failing_configure_tools(self):  # pyright: ignore[reportMissingParameterType,reportUnusedParameter]
        raise ValueError("configure_tools failed")

    monkeypatch.setattr(type(test_task), "configure_tools", failing_configure_tools)

    transcript_file = tmp_path / "transcript.json"

    sample_config.mcp_server_config.port = mcp_server.config.port
    sample_config.transcript_file = transcript_file.as_posix()

    runner = EvaluationRunner(sample_config, test_task)

    events = []
    async for event in runner.run():
        events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], ErrorEvent)
    assert isinstance(events[1], TaskCompletedEvent)

    assert not transcript_file.exists()


@pytest.mark.asyncio
async def test_extra_task_instructions_applied_per_step(
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    test_task: Task,
):
    sample_config.mcp_server_config.port = mcp_server.config.port
    sample_config.extra_config = {
        "extra_task_instructions": "extra overlay",
    }

    runner = EvaluationRunner(sample_config, test_task)

    user_messages = [
        event.message.content
        async for event in runner.run()
        if isinstance(event, MessageAddedEvent) and event.message.role == "user"
    ]

    assert user_messages[0] == "Step 42.\n\nextra overlay"


@pytest.mark.asyncio
async def test_step_counting_starts_at_zero(
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    test_task: Task,
):
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, test_task)

    passed = False

    async for event in runner.run():
        if isinstance(event, StepStartedEvent):
            assert event.step >= 0
            passed = True

    assert passed, "Step counting did not start at zero"


@pytest.mark.asyncio
async def test_json_decode_error_handling_in_execute_tool_calls():
    """Test that invalid JSON in tool call arguments is handled gracefully."""

    # Create a tool call with invalid JSON arguments
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments='{"command": "cat > /wor',  # Invalid JSON - unclosed string
        ),
        type="function",
    )

    # Create a mock MCP client
    mock_mcp_client = Mock()
    mock_mcp_client.call_tool_mcp = AsyncMock()

    events = []
    async for event in _execute_tool_calls([tool_call], mock_mcp_client):
        events.append(event)

    # Should have 3 events: ToolCallStartedEvent, ToolCallCompletedEvent, and MessageAddedEvent
    assert len(events) == 3

    # Verify the completed event contains an error result
    completed_event = events[1]
    assert isinstance(completed_event, ToolCallCompletedEvent)
    assert completed_event.result.isError is True
    assert isinstance(completed_event.result.content[0], TextContent)
    assert (
        completed_event.result.content[0].text
        == "Error: Invalid JSON in tool arguments"
    )

    # Verify that call_tool_mcp was NOT called due to JSON error
    mock_mcp_client.call_tool_mcp.assert_not_called()


@pytest.mark.asyncio
async def test_guard_case_malformed_json_cat_redirect():
    """Test the specific guard case: '{"command": "cat > /wor' triggers proper error handling."""

    # This is the exact guard case that should trigger the error handling
    tool_call = ChatCompletionMessageToolCall(
        id="guard_test_id",
        function=Function(
            name="bash",
            arguments='{"command": "cat > /wor',  # Unclosed string - the guard case
        ),
        type="function",
    )

    # Create a mock MCP client
    mock_mcp_client = Mock()
    mock_mcp_client.call_tool_mcp = AsyncMock()

    events = []
    async for event in _execute_tool_calls([tool_call], mock_mcp_client):
        events.append(event)

    # Verify the error was caught and handled
    assert len(events) == 3

    # Check the ToolCallCompletedEvent has the error
    completed_event = events[1]
    assert isinstance(completed_event, ToolCallCompletedEvent)
    assert completed_event.result.isError is True
    assert isinstance(completed_event.result.content[0], TextContent)
    assert (
        completed_event.result.content[0].text
        == "Error: Invalid JSON in tool arguments"
    )

    # Verify MCP client was never called due to JSON parsing failure
    mock_mcp_client.call_tool_mcp.assert_not_called()


@pytest.mark.asyncio
async def test_json_decode_error_sanitizes_tool_call_arguments():
    """Test that tool_call.function.arguments is sanitized after JSONDecodeError.

    This prevents downstream API errors when providers like TogetherAI validate
    the JSON in tool_call arguments.
    """

    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments='{"command": "cat > /wor',  # Invalid JSON
        ),
        type="function",
    )

    original_args = tool_call.function.arguments

    mock_mcp_client = Mock()
    mock_mcp_client.call_tool_mcp = AsyncMock()

    events = []
    async for event in _execute_tool_calls([tool_call], mock_mcp_client):
        events.append(event)

    # Verify the arguments were sanitized to valid JSON
    assert tool_call.function.arguments != original_args
    # Should be valid JSON now
    import json

    parsed = json.loads(tool_call.function.arguments)
    assert "_error" in parsed


# =============================================================================
# Tests for message serialization and cache control
# =============================================================================


def _prepare_messages(runner: EvaluationRunner) -> list[dict[str, Any]]:
    """Helper that reproduces the old _prepare_messages behavior for tests.

    Serializes messages and adds inline cache_control for non-vertex models.
    """
    source = BuiltinSource(runner.config)
    messages = source._serialize_messages(runner.transcript.messages)  # pyright: ignore[reportPrivateUsage]
    if not runner.config.model.startswith("vertex_ai/"):
        AnthropicProvider().apply_prompt_caching(runner.config.run_id, messages, {})
    return messages


def _create_runner_with_transcript(
    messages: list[Message], model: str = "test_model"
) -> EvaluationRunner:
    """Helper to create a runner with a transcript containing the given messages."""
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model=model,
        model_api_key="test_key" if not model.startswith("vertex_ai/") else None,
        mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
        transcript_file="",
    )
    runner = EvaluationRunner(config, _make_test_task())
    runner.transcript = Transcript(run_id="test_run")
    for message in messages:
        runner.transcript.events.append(MessageAddedEvent(message=message))
    return runner


def test_prepare_messages_empty_transcript():
    """Test _prepare_messages with empty transcript."""
    runner = _create_runner_with_transcript([])

    result = _prepare_messages(runner)

    assert result == []


def test_prepare_messages_normalizes_string_content_to_list():
    """Test that string content is normalized to list format."""
    runner = _create_runner_with_transcript(
        [
            Message(role="user", content="Hello"),
        ]
    )

    result = _prepare_messages(runner)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == [
        {
            "type": "text",
            "text": "Hello",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]


def test_prepare_messages_preserves_list_content():
    """Test that list content is preserved as-is (except for cache_control addition)."""
    runner = _create_runner_with_transcript(
        [
            Message(
                role="user",
                content=[{"type": "text", "text": "Hello"}],
            ),
        ]
    )

    result = _prepare_messages(runner)

    assert len(result) == 1
    assert result[0]["content"] == [
        {
            "type": "text",
            "text": "Hello",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]


def test_prepare_messages_adds_cache_control_to_last_message():
    """Test that cache_control is added only to the last message."""
    runner = _create_runner_with_transcript(
        [
            Message(role="user", content="First message"),
            Message(role="assistant", content="Second message"),
            Message(role="user", content="Third message"),
        ]
    )

    result = _prepare_messages(runner)

    # First two messages are not normalized and don't have cache_control
    assert result[0]["content"] == "First message"
    assert result[1]["content"] == "Second message"

    # Last message is normalized and has cache_control
    assert result[2]["content"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_prepare_messages_adds_cache_control_to_last_content_item():
    """Test that cache_control is added to the last content item when there are multiple."""
    runner = _create_runner_with_transcript(
        [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ],
            ),
        ]
    )

    result = _prepare_messages(runner)

    # First content item should NOT have cache_control
    assert "cache_control" not in result[0]["content"][0]

    # Last content item should have cache_control
    assert result[0]["content"][1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_prepare_messages_handles_none_content():
    """Test that None content is handled gracefully."""
    runner = _create_runner_with_transcript(
        [
            Message(role="assistant", content=None),
        ]
    )

    result = _prepare_messages(runner)

    assert len(result) == 1
    assert result[0]["content"] is None


def test_prepare_messages_handles_empty_list_content():
    """Test that empty list content is handled gracefully."""
    runner = _create_runner_with_transcript(
        [
            Message(role="user", content=[]),
        ]
    )

    result = _prepare_messages(runner)

    assert len(result) == 1
    assert result[0]["content"] == []


def test_prepare_messages_with_tool_calls():
    """Test that tool_calls are preserved."""
    runner = _create_runner_with_transcript(
        [
            Message(
                role="assistant",
                content="Let me run that command",
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="tool_1",
                        function=Function(name="bash", arguments='{"command": "ls"}'),
                        type="function",
                    )
                ],
            ),
        ]
    )

    result = _prepare_messages(runner)

    assert len(result) == 1
    assert result[0]["tool_calls"] is not None
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["function"]["name"] == "bash"


def test_prepare_messages_with_tool_response():
    """Test that tool responses are handled correctly."""
    runner = _create_runner_with_transcript(
        [
            Message(
                role="tool",
                content=[{"type": "text", "text": "file1.txt\nfile2.txt"}],
                tool_call_id="tool_1",
            ),
        ]
    )

    result = _prepare_messages(runner)

    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "tool_1"
    assert result[0]["content"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_prepare_messages_mixed_content_types():
    """Test handling of messages with different content types in sequence."""
    runner = _create_runner_with_transcript(
        [
            Message(role="user", content="String content"),
            Message(
                role="assistant", content=[{"type": "text", "text": "List content"}]
            ),
            Message(role="user", content="Another string"),
        ]
    )

    result = _prepare_messages(runner)

    # Only the last message with content gets normalized to list format
    assert result[0]["content"] == "String content"  # Not normalized
    assert isinstance(result[1]["content"], list)  # Already a list
    assert isinstance(result[2]["content"], list)  # Normalized for cache_control

    # Only the last message should have cache_control
    assert "cache_control" not in result[1]["content"][0]
    assert "cache_control" in result[2]["content"][0]


def test_prepare_messages_with_image_content():
    """Test that image content blocks are preserved."""
    runner = _create_runner_with_transcript(
        [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,ABC123"},
                    },
                ],
            ),
        ]
    )

    result = _prepare_messages(runner)

    assert len(result) == 1
    assert len(result[0]["content"]) == 2
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][1]["type"] == "image_url"
    # Cache control should be on the last content item (the image)
    assert result[0]["content"][1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


# =============================================================================
# Tests for litellm cache_control on image_url in tool results
# =============================================================================


@pytest.mark.asyncio
async def test_cache_control_preserved_on_image_in_anthropic_api_request(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that cache_control on image_url content reaches the Anthropic API.

    This is an integration test that verifies the full flow:
    EvaluationRunner → LiteLLM → Anthropic API

    The test mocks litellm's HTTP handler to capture the actual request body
    sent to Anthropic, verifying that cache_control is preserved on images
    after litellm's message transformation.

    Upstream PR: https://github.com/BerriAI/litellm/pull/18674
    """
    import json

    from litellm.llms.anthropic.chat import handler as anthropic_handler

    # Track the request body sent to Anthropic
    captured_request_body = None

    async def mock_make_call(
        client,  # pyright: ignore[reportUnusedParameter,reportMissingParameterType]
        api_base,  # pyright: ignore[reportUnusedParameter,reportMissingParameterType]
        headers,  # pyright: ignore[reportUnusedParameter,reportMissingParameterType]
        data,  # pyright: ignore[reportMissingParameterType]
        model,  # pyright: ignore[reportUnusedParameter,reportMissingParameterType]
        messages,  # pyright: ignore[reportUnusedParameter,reportMissingParameterType]
        logging_obj,  # pyright: ignore[reportUnusedParameter,reportMissingParameterType]
        timeout,  # pyright: ignore[reportUnusedParameter,reportMissingParameterType]
        **kwargs,  # pyright: ignore[reportUnusedParameter,reportMissingParameterType]
    ):
        nonlocal captured_request_body
        captured_request_body = json.loads(data)
        raise Exception("Mock: stopping after capturing request")

    monkeypatch.setattr(anthropic_handler, "make_call", mock_make_call)

    # Create runner with messages including a tool result with image and cache_control
    runner = EvaluationRunner(
        EvaluationRunConfig(
            run_id="test_run",
            task_id="test_task",
            model="claude-sonnet-4-20250514",
            model_api_key="test_key",
            mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
            transcript_file="",
        ),
        _make_test_task(),
    )
    runner.transcript = Transcript(run_id="test_run")
    runner.tools = []

    # Add messages to transcript: user message, assistant with tool_call, tool result with image
    runner.transcript.events.append(
        MessageAddedEvent(message=Message(role="user", content="Show me an image"))
    )
    runner.transcript.events.append(
        MessageAddedEvent(
            message=Message(
                role="assistant",
                content="Here's the image",
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="toolu_123",
                        function=Function(name="view_image", arguments="{}"),
                        type="function",
                    )
                ],
            )
        )
    )
    # Tool result with image_url and cache_control
    runner.transcript.events.append(
        MessageAddedEvent(
            message=Message(
                role="tool",
                tool_call_id="toolu_123",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            )
        )
    )

    # Call _collect_model_response which triggers the API call
    with patch("karotte.agents.builtin_source.LLM_RETRY_MAX_ATTEMPTS", 1):
        with pytest.raises(Exception, match="Mock: stopping after capturing request"):
            async for _ in _collect(runner):
                pass

    # Verify cache_control was preserved in the request to Anthropic
    assert captured_request_body is not None, "No request was captured"
    messages = captured_request_body.get("messages", [])

    # Find the tool result with image content (Anthropic format)
    # In Anthropic format, tool results are nested in user messages as type="tool_result"
    image_found = False
    for msg in messages:  # pyright: ignore[reportGeneralTypeIssues]
        if msg.get("role") == "user":
            for content in msg.get("content", []):
                if content.get("type") == "tool_result":
                    for item in content.get("content", []):
                        if item.get("type") == "image":
                            image_found = True
                            assert "cache_control" in item, (
                                "cache_control was dropped when converting image_url to "
                                "Anthropic's image format. This is the bug that the litellm "
                                "patch should fix."
                            )
                            assert item["cache_control"]["type"] == "ephemeral"
                            assert item["cache_control"]["ttl"] == "1h"

    assert image_found, "Image in tool result not found in Anthropic request"


# =============================================================================
# Tests for Vertex AI cache_control skip behavior
# =============================================================================


def test_prepare_messages_vertex_ai_skips_cache_control():
    """Test that Vertex AI models do not get cache_control added to messages.

    Vertex AI has implicit caching enabled by default, so explicit cache_control
    is unnecessary and can trigger permission errors for cachedContents.create.
    """
    runner = _create_runner_with_transcript(
        [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?"),
        ],
        model="vertex_ai/gemini-2.5-pro",
    )

    result = _prepare_messages(runner)

    # Verify no cache_control was added to any message
    for msg in result:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                assert "cache_control" not in item, (
                    "cache_control should not be added for Vertex AI models"
                )
        # String content should remain as string (not normalized)
        elif isinstance(content, str):
            pass  # OK, string content without cache_control


def test_prepare_messages_vertex_ai_preserves_content_structure():
    """Test that Vertex AI models preserve original content structure."""
    runner = _create_runner_with_transcript(
        [
            Message(role="user", content="String content"),
            Message(
                role="assistant",
                content=[{"type": "text", "text": "List content"}],
            ),
        ],
        model="vertex_ai/gemini-1.5-pro",
    )

    result = _prepare_messages(runner)

    # String content should remain as string
    assert result[0]["content"] == "String content"

    # List content should remain as list without cache_control
    assert result[1]["content"] == [{"type": "text", "text": "List content"}]


def test_prepare_messages_non_vertex_ai_still_adds_cache_control():
    """Test that non-Vertex AI models still get cache_control added."""
    runner = _create_runner_with_transcript(
        [
            Message(role="user", content="Hello"),
        ],
        model="claude-sonnet-4-20250514",
    )

    result = _prepare_messages(runner)

    # Non-Vertex AI models should have cache_control added
    assert result[0]["content"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


# =============================================================================
# Tests for per-provider prompt-caching enablement
# =============================================================================


def _cache_test_completion_params(
    model: str, monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None = None
) -> dict[str, Any]:
    """Build completion params for a model with caching-relevant env cleared.

    ``env`` entries are applied after the clearing, so tests can opt back in.
    """
    monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
    for var, value in (env or {}).items():
        monkeypatch.setenv(var, value)
    config = EvaluationRunConfig(
        run_id="cache_run_id",
        task_id="test_task",
        model=model,
        model_api_key="test_key" if not model.startswith("vertex_ai/") else None,
        mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
        transcript_file="",
    )
    source = BuiltinSource(config)
    return source.get_completion_params([Message(role="user", content="Hello")], [])


def test_prompt_caching_xai_sets_conversation_header(
    monkeypatch: pytest.MonkeyPatch,
):
    """xAI cache entries are per-server; x-grok-conv-id pins sticky routing."""
    params = _cache_test_completion_params("xai/grok-4.6", monkeypatch)

    assert params["extra_headers"]["x-grok-conv-id"] == "cache_run_id"
    assert "prompt_cache_key" not in params
    assert "cache_control" not in json.dumps(params["messages"])


def test_prompt_caching_openai_sets_prompt_cache_key(
    monkeypatch: pytest.MonkeyPatch,
):
    """GPT-5.6 requires prompt_cache_key for its prefix matching. It rides in
    extra_body because litellm drops the top-level kwarg on the wire."""
    params = _cache_test_completion_params("openai/gpt-5.6", monkeypatch)

    assert params["extra_body"]["prompt_cache_key"] == "cache_run_id"
    assert "prompt_cache_key" not in params
    assert "extra_headers" not in params
    assert "cache_control" not in json.dumps(params["messages"])


def test_prompt_caching_anthropic_adds_inline_cache_control(
    monkeypatch: pytest.MonkeyPatch,
):
    """Anthropic models get the explicit cache_control marker, no cache key."""
    params = _cache_test_completion_params("claude-sonnet-4-6", monkeypatch)

    assert params["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    assert "prompt_cache_key" not in params
    assert "extra_headers" not in params


def test_prompt_caching_anthropic_marker_survives_proxy(
    monkeypatch: pytest.MonkeyPatch,
):
    """A configured proxy must not disable Anthropic's cache_control marker."""
    params = _cache_test_completion_params(
        "claude-sonnet-4-6",
        monkeypatch,
        env={"KAROTTE_PROXY_URL": "https://proxy.example.com"},
    )

    assert params["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_prompt_caching_implicit_providers_get_no_annotations(
    monkeypatch: pytest.MonkeyPatch,
):
    """Together/MiniMax/Meta cache automatically; no marker, header, or key."""
    for model in (
        "together_ai/moonshotai/Kimi-K3",
        "minimax/MiniMax-M3",
        "meta/muse-spark-1.2",
    ):
        params = _cache_test_completion_params(model, monkeypatch)

        assert "prompt_cache_key" not in params, model
        assert "extra_headers" not in params, model
        assert "cache_control" not in json.dumps(params["messages"]), model


def test_prompt_caching_vertex_keeps_priority_headers_no_cache_control(
    monkeypatch: pytest.MonkeyPatch,
):
    """Gemini caches implicitly; explicit cache_control can trigger
    cachedContents.create permission errors, and the Priority PayGo headers
    must survive untouched."""
    params = _cache_test_completion_params(
        "vertex_ai/gemini-3.1-pro-preview",
        monkeypatch,
        env={SERVICE_TIER_ENV: "priority"},
    )

    assert "cache_control" not in json.dumps(params["messages"])
    assert "prompt_cache_key" not in params
    assert params["extra_headers"]["X-Vertex-AI-LLM-Request-Type"] == "shared"
    assert "x-grok-conv-id" not in params["extra_headers"]


# =============================================================================
# Tests for transcript file saving with parent directory creation
# =============================================================================


@pytest.mark.asyncio
async def test_transcript_saved_to_nested_directory(
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    tmp_path: Path,
):
    """Test that transcript is saved even when parent directories don't exist.

    This test verifies the fix that creates parent directories before saving
    the transcript file, preventing FileNotFoundError when the transcript_file
    path includes directories that don't exist yet.
    """

    @final
    class NestedDirTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "nested-dir-task"

        @property
        def steps(self):
            return []

        @property
        def tools(self):
            return []

    # Create a transcript path with multiple nested directories that don't exist
    transcript_file = tmp_path / "deeply" / "nested" / "path" / "transcript.json"

    # Verify the parent directories don't exist before the test
    assert not transcript_file.parent.exists()

    sample_config.task_id = "nested-dir-task"
    sample_config.mcp_server_config.port = mcp_server.config.port
    sample_config.transcript_file = transcript_file.as_posix()

    runner = EvaluationRunner(sample_config, NestedDirTask(sample_config))

    async for _ in runner.run():
        pass

    # Verify the transcript file was created along with its parent directories
    assert transcript_file.exists()
    assert transcript_file.is_file()

    # Verify the transcript content is valid JSON
    transcript = Transcript.model_validate_json(transcript_file.read_text())
    assert transcript.run_id == sample_config.run_id


# =============================================================================
# Tests for step post_hook execution
# =============================================================================


async def _mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
    """Mock source turn that returns a simple message with no tool calls."""
    yield MessageAddedEvent(
        message=Message(role="assistant", content="Mocked response")
    )


@pytest.mark.asyncio
async def test_step_post_hook_gets_executed(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """Test that step post_hook is called after step completion."""
    from karotte.judges.regex_judge import RegexJudge

    # Mock the model response to avoid actual API call
    monkeypatch.setattr(BuiltinSource, "collect", _mock_collect)

    # Track whether post_hook was called
    post_hook_called = []

    class PostHookStep(Step):
        @property
        def instructions(self):
            return "Test step"

        @property
        def judge(self):
            return RegexJudge([])

        def post_hook(self):
            post_hook_called.append(True)

    @final
    class TaskWithPostHook(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "post-hook-task"

        @property
        def tools(self):
            return []

        @property
        def steps(self):
            return [
                PostHookStep(config=sample_config),
            ]

    sample_config.task_id = "post-hook-task"
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, TaskWithPostHook(sample_config))

    async for _ in runner.run():
        pass

    assert len(post_hook_called) == 1, "post_hook should have been called once"


@pytest.mark.asyncio
async def test_extra_artifacts_saved_before_pre_scoring_hook_scrub(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    tmp_path: Path,
):
    """Extra artifacts are saved before pre_scoring_hook, not after.

    Many envs scrub the student workdir down to the official answer file inside
    pre_scoring_hook. If extra-artifact capture ran after that hook, a model-
    written file named via extra_config (e.g. an approach_hint.txt) would be
    gone before it could be persisted. Saving first keeps it.
    """
    from karotte.judges.regex_judge import RegexJudge

    monkeypatch.setattr(BuiltinSource, "collect", _mock_collect)

    # The file the model "wrote"; the env's pre_scoring_hook scrubs it away.
    hint = tmp_path / "approach_hint.txt"
    hint.write_text("my approach")

    saved: list[Path] = []
    monkeypatch.setattr(
        "karotte.evaluation_runner.save_artifact",
        lambda config, path: saved.append(Path(path)),  # pyright: ignore[reportUnknownLambdaType]
    )

    class ScrubbingStep(Step):
        @property
        def instructions(self):
            return "Test step"

        @property
        def judge(self):
            return RegexJudge([])

        def pre_scoring_hook(self):
            # Emulate an env scrubbing the student workdir to its answer file.
            hint.unlink(missing_ok=True)

    @final
    class TaskWithScrub(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "scrub-task"

        @property
        def tools(self):
            return []

        @property
        def steps(self):
            return [ScrubbingStep(config=sample_config)]

    sample_config.task_id = "scrub-task"
    sample_config.mcp_server_config.port = mcp_server.config.port
    sample_config.extra_config = {"extra_artifact_paths": str(hint)}

    runner = EvaluationRunner(sample_config, TaskWithScrub(sample_config))
    async for _ in runner.run():
        pass

    assert saved == [hint], (
        "extra artifact should be persisted before pre_scoring_hook scrubs it"
    )


@pytest.mark.asyncio
async def test_step_post_hook_called_for_each_step(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """Test that post_hook is called for each step that has one."""
    from karotte.judges.regex_judge import RegexJudge

    # Mock the model response to avoid actual API call
    monkeypatch.setattr(BuiltinSource, "collect", _mock_collect)

    # Track which steps had their post_hook called
    hooks_called = []

    def make_post_hook_step(step_name: str, has_hook: bool = True):
        class PostHookStep(Step):
            @property
            def instructions(self):
                return f"Step {step_name}"

            @property
            def judge(self):
                return RegexJudge([])

            def post_hook(self):
                if has_hook:
                    hooks_called.append(step_name)

        return PostHookStep(sample_config)

    @final
    class MultiStepTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "multi-step-post-hook-task"

        @property
        def tools(self):
            return []

        @property
        def steps(self):
            return [
                make_post_hook_step("step1", has_hook=True),
                make_post_hook_step("step2", has_hook=True),
                make_post_hook_step("step3", has_hook=False),
            ]

    sample_config.task_id = "multi-step-post-hook-task"
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, MultiStepTask(sample_config))

    async for _ in runner.run():
        pass

    assert hooks_called == ["step1", "step2"], (
        f"Expected hooks for step1 and step2 to be called, got {hooks_called}"
    )


@pytest.mark.asyncio
async def test_step_post_hook_not_called_when_none(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """Test that no error occurs when step has no post_hook (None)."""
    from karotte.judges.regex_judge import RegexJudge

    # Mock the model response to avoid actual API call
    monkeypatch.setattr(BuiltinSource, "collect", _mock_collect)

    class NoPostHookStep(Step):
        @property
        def instructions(self):
            return "Step without post_hook"

        @property
        def judge(self):
            return RegexJudge([])

        # post_hook defaults to returning None

    @final
    class NoPostHookTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "no-post-hook-task"

        @property
        def tools(self):
            return []

        @property
        def steps(self):
            return [NoPostHookStep(sample_config)]

    sample_config.task_id = "no-post-hook-task"
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, NoPostHookTask(sample_config))

    # Should complete without error
    events = []
    async for event in runner.run():
        events.append(event)

    # Verify task completed successfully
    assert any(
        isinstance(e, TaskCompletedEvent) and e.status == "passed" for e in events
    )


@pytest.mark.asyncio
async def test_step_post_hook_called_after_scoring(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """Test that post_hook is called after the step's scoring event."""
    from karotte.judges.regex_judge import RegexJudge

    # Mock the model response to avoid actual API call
    monkeypatch.setattr(BuiltinSource, "collect", _mock_collect)

    # Track the order of events
    event_order = []

    class OrderTestStep(Step):
        @property
        def instructions(self):
            return "Test step"

        @property
        def judge(self):
            return RegexJudge([])

        def post_hook(self):
            event_order.append("post_hook")

    @final
    class OrderTestTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "order-test-task"

        @property
        def tools(self):
            return []

        @property
        def steps(self):
            return [OrderTestStep(sample_config)]

    sample_config.task_id = "order-test-task"
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, OrderTestTask(sample_config))

    async for event in runner.run():
        if isinstance(event, ScoringEvent):
            event_order.append("scoring")
        elif isinstance(event, StepCompletedEvent):
            event_order.append("step_completed")

    # post_hook should be called after scoring but before step_completed
    assert event_order == ["scoring", "post_hook", "step_completed"], (
        f"Expected ['scoring', 'post_hook', 'step_completed'], got {event_order}"
    )


@pytest.mark.asyncio
async def test_step_post_hook_can_write_files(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    tmp_path: Path,
):
    """Test that post_hook can perform side effects like writing files."""
    from karotte.judges.regex_judge import RegexJudge

    # Mock the model response to avoid actual API call
    monkeypatch.setattr(BuiltinSource, "collect", _mock_collect)

    artifact_file = tmp_path / "artifact.txt"

    class FileWriteStep(Step):
        @property
        def instructions(self):
            return "Test step"

        @property
        def judge(self):
            return RegexJudge([])

        def post_hook(self):
            artifact_file.write_text("artifact content")

    @final
    class FileWriteTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "file-write-task"

        @property
        def tools(self):
            return []

        @property
        def steps(self):
            return [FileWriteStep(sample_config)]

    sample_config.task_id = "file-write-task"
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, FileWriteTask(sample_config))

    async for _ in runner.run():
        pass

    assert artifact_file.exists()
    assert artifact_file.read_text() == "artifact content"


# =============================================================================
# Tests for step pre_scoring_hook execution
# =============================================================================


@pytest.mark.asyncio
async def test_step_pre_scoring_hook_gets_executed(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """Test that step pre_scoring_hook is called before scoring."""
    from karotte.judges.regex_judge import RegexJudge

    monkeypatch.setattr(BuiltinSource, "collect", _mock_collect)

    pre_scoring_hook_called = []

    class PreScoringHookStep(Step):
        @property
        def instructions(self):
            return "Test step"

        @property
        def judge(self):
            return RegexJudge([])

        def pre_scoring_hook(self):
            pre_scoring_hook_called.append(True)

    @final
    class TaskWithPreScoringHook(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "pre-scoring-hook-task"

        @property
        def tools(self):
            return []

        @property
        def steps(self):
            return [PreScoringHookStep(config=sample_config)]

    sample_config.task_id = "pre-scoring-hook-task"
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, TaskWithPreScoringHook(sample_config))

    async for _ in runner.run():
        pass

    assert len(pre_scoring_hook_called) == 1, (
        "pre_scoring_hook should have been called once"
    )


@pytest.mark.asyncio
async def test_step_pre_scoring_hook_called_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """Test that pre_scoring_hook is called before the step's scoring event."""
    from karotte.judges.regex_judge import RegexJudge

    monkeypatch.setattr(BuiltinSource, "collect", _mock_collect)

    event_order = []

    class OrderTestStep(Step):
        @property
        def instructions(self):
            return "Test step"

        @property
        def judge(self):
            return RegexJudge([])

        def pre_scoring_hook(self):
            event_order.append("pre_scoring_hook")

        def post_hook(self):
            event_order.append("post_hook")

    @final
    class OrderTestTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "pre-scoring-order-task"

        @property
        def tools(self):
            return []

        @property
        def steps(self):
            return [OrderTestStep(sample_config)]

    sample_config.task_id = "pre-scoring-order-task"
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, OrderTestTask(sample_config))

    async for event in runner.run():
        if isinstance(event, ScoringEvent):
            event_order.append("scoring")
        elif isinstance(event, StepCompletedEvent):
            event_order.append("step_completed")

    assert event_order == [
        "pre_scoring_hook",
        "scoring",
        "post_hook",
        "step_completed",
    ], (
        f"Expected ['pre_scoring_hook', 'scoring', 'post_hook', 'step_completed'], got {event_order}"
    )


@pytest.mark.asyncio
async def test_step_pre_scoring_hook_called_for_each_step(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """Test that pre_scoring_hook is called for each step."""
    from karotte.judges.regex_judge import RegexJudge

    monkeypatch.setattr(BuiltinSource, "collect", _mock_collect)

    hooks_called = []

    def make_step(step_name: str):
        class HookStep(Step):
            @property
            def instructions(self):
                return f"Step {step_name}"

            @property
            def judge(self):
                return RegexJudge([])

            def pre_scoring_hook(self):
                hooks_called.append(step_name)

        return HookStep(sample_config)

    @final
    class MultiStepTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "multi-step-pre-scoring-task"

        @property
        def tools(self):
            return []

        @property
        def steps(self):
            return [make_step("step1"), make_step("step2"), make_step("step3")]

    sample_config.task_id = "multi-step-pre-scoring-task"
    sample_config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(sample_config, MultiStepTask(sample_config))

    async for _ in runner.run():
        pass

    assert hooks_called == ["step1", "step2", "step3"]


# =============================================================================
# Tests for DeepSeek special token sanitization
# =============================================================================


class TestMaxReasoningEffort:
    """The ceiling a run asking for "max" gets. Update this table when a new
    gpt-5.X family ships with a level its predecessor lacked."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # gpt-5.6 family — supports "max", a level above xhigh
            ("openai/gpt-5.6", "max"),
            ("openai/gpt-5.6-sol", "max"),
            ("openai/gpt-5.6-terra", "max"),
            ("openai/gpt-5.6-luna", "max"),
            # gpt-5.5 family — supports xhigh
            ("openai/gpt-5.5", "xhigh"),
            ("openai/gpt-5.5-pro", "xhigh"),
            ("openai/gpt-5.5-2026-04-23", "xhigh"),
            ("gpt-5.5", "xhigh"),  # no provider prefix
            # gpt-5.4 family — supports xhigh
            ("openai/gpt-5.4", "xhigh"),
            ("openai/gpt-5.4-mini", "xhigh"),
            ("openai/gpt-5.4-pro", "xhigh"),
            # gpt-5.2 family — supports xhigh
            ("openai/gpt-5.2", "xhigh"),
            ("openai/gpt-5.2-codex", "xhigh"),
            # gpt-5.3 — only -chat-latest and -codex variants exist; max is "high"
            ("openai/gpt-5.3-codex", "high"),
            ("openai/gpt-5.3-chat-latest", "high"),
            # gpt-5 / gpt-5.1 — no xhigh, so "high" is the top
            ("openai/gpt-5", "high"),
            ("openai/gpt-5.1", "high"),
            ("claude-opus-4-7", "max"),
            ("vertex_ai/gemini-3.1-pro-preview", "high"),
            # Kimi K2.x answers 400 to the parameter; K3 takes it.
            ("together_ai/moonshotai/Kimi-K2.6", None),
            ("together_ai/moonshotai/Kimi-K3", "max"),
        ],
    )
    def test_max_reasoning_effort(self, model: str, expected: str | None):
        assert spec_for(model).max_reasoning_effort == expected


class TestProviderReasoning:
    """Which models need reasoning switched on to stream it back."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("together_ai/moonshotai/Kimi-K2.6", True),
            ("together_ai/moonshotai/Kimi-K2.5", True),
            ("together_ai/moonshotai/Kimi-K3", True),
            # Non-thinking Together models and other providers are untouched.
            ("together_ai/deepseek-ai/DeepSeek-V3.1", False),
            ("together_ai/zai-org/GLM-5.1", False),
            ("claude-opus-4-7", False),
            ("openai/gpt-5.5", False),
        ],
    )
    def test_enable_provider_reasoning(self, model: str, expected: bool):
        spec = spec_for(model)
        params: dict[str, Any] = {}
        provider_for(spec).apply_reasoning(spec, None, params)
        assert (
            params.get("extra_body") == {"reasoning": {"enabled": True}}
        ) == expected


class TestMaxOutputTokens:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-5", 128000),
            ("openai/gpt-5.6", 128000),
            ("openai/gpt-5.6-sol", 128000),
            ("openai/gpt-5.6-terra", 128000),
            ("openai/gpt-5.6-luna", 128000),
            # Every gpt-5.6 snapshot has the same ceiling as the family.
            ("openai/gpt-5.6-2026-04-01", 128000),
            ("meta/muse-spark-1.3", 128000),
            # Unrecognized models fall back to the conservative default.
            ("openai/gpt-5.5", 64000),
            ("claude-opus-4-1", 64000),
        ],
    )
    def test_max_output_tokens(self, model: str, expected: int):
        assert spec_for(model).max_output_tokens == expected


class TestAdaptiveThinking:
    """Every Claude model that supports adaptive thinking asks for it with a
    summarized display. On 4.6/4.7/4.8 that switches thinking on; on the 5th
    generation it only surfaces the summary."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-fable-5", True),
            ("claude-sonnet-5", True),
            ("claude-opus-5", True),
            ("claude-opus-4-8", True),
            ("claude-opus-4-7", True),
            ("claude-opus-4-6", True),
            ("claude-sonnet-4-6", True),
            ("anthropic/claude-fable-5", True),  # provider-prefixed
            ("vertex_ai/claude-sonnet-5", True),
            # The 4.5 generation takes budget_tokens instead and 400s on this.
            ("claude-opus-4-5-20251101", False),
            ("claude-sonnet-4-5-20250929", False),
            ("claude-haiku-4-5-20251001", False),
            ("claude-sonnet-4-20250514", False),
            # Other providers are untouched.
            ("openai/gpt-5.5", False),
            ("vertex_ai/gemini-3.1-pro-preview", False),
            ("together_ai/moonshotai/Kimi-K2.6", False),
        ],
    )
    def test_adaptive_thinking(self, model: str, expected: bool):
        spec = spec_for(model)
        params: dict[str, Any] = {}
        provider_for(spec).apply_reasoning(spec, None, params)
        expected_thinking = {"type": "adaptive", "display": "summarized"}
        assert (params.get("thinking") == expected_thinking) == expected


class TestStripDeepseekTokens:
    """Tests for _strip_deepseek_tokens function."""

    def test_strips_tool_call_end_token(self):
        text = '{"command": "python test.py"}<｜tool▁call▁end｜>'
        result = _strip_deepseek_tokens(text)
        assert result == '{"command": "python test.py"}'

    def test_strips_tool_call_begin_token(self):
        text = '<｜tool▁call▁begin｜>{"command": "ls"}'
        result = _strip_deepseek_tokens(text)
        assert result == '{"command": "ls"}'

    def test_strips_multiple_tokens(self):
        text = '<｜tool▁call▁begin｜>{"cmd": "test"}<｜tool▁call▁end｜>'
        result = _strip_deepseek_tokens(text)
        assert result == '{"cmd": "test"}'

    def test_strips_tool_sep_token(self):
        text = '{"a": 1}<｜tool▁sep｜>{"b": 2}'
        result = _strip_deepseek_tokens(text)
        assert result == '{"a": 1}{"b": 2}'

    def test_strips_tool_outputs_tokens(self):
        text = "<｜tool▁outputs▁begin｜>output<｜tool▁outputs▁end｜>"
        result = _strip_deepseek_tokens(text)
        assert result == "output"

    def test_preserves_normal_text(self):
        text = '{"command": "echo hello"}'
        result = _strip_deepseek_tokens(text)
        assert result == '{"command": "echo hello"}'

    def test_preserves_similar_looking_text(self):
        # Should not strip text that looks similar but isn't an exact match
        text = '{"msg": "<|not_a_real_token|>"}'
        result = _strip_deepseek_tokens(text)
        assert result == '{"msg": "<|not_a_real_token|>"}'

    def test_handles_empty_string(self):
        result = _strip_deepseek_tokens("")
        assert result == ""


def _repair_deepseek(
    tool_calls: list[ChatCompletionMessageToolCall] | None,
) -> list[ChatCompletionMessageToolCall] | None:
    return repair_tool_calls(tool_calls, _strip_deepseek_tokens)


class TestRepairToolCalls:
    """Tests for repair_tool_calls, driven by the DeepSeek repair."""

    def test_returns_none_for_none_input(self):
        result = _repair_deepseek(None)
        assert result is None

    def test_returns_empty_list_for_empty_input(self):
        result = _repair_deepseek([])
        assert result == []

    def test_sanitizes_tool_call_arguments(self):
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="bash",
                arguments='{"command": "ls"}<｜tool▁call▁end｜>',
            ),
        )
        result = _repair_deepseek([tool_call])

        assert result is not None
        assert len(result) == 1
        assert result[0].function.arguments == '{"command": "ls"}'
        assert result[0].id == "call_123"
        assert result[0].function.name == "bash"

    def test_preserves_clean_arguments(self):
        tool_call = ChatCompletionMessageToolCall(
            id="call_456",
            type="function",
            function=Function(
                name="bash",
                arguments='{"command": "echo hello"}',
            ),
        )
        result = _repair_deepseek([tool_call])

        assert result is not None
        assert len(result) == 1
        assert result[0].function.arguments == '{"command": "echo hello"}'

    def test_handles_multiple_tool_calls(self):
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="bash",
                    arguments='{"cmd": "a"}<｜tool▁call▁end｜>',
                ),
            ),
            ChatCompletionMessageToolCall(
                id="call_2",
                type="function",
                function=Function(
                    name="bash",
                    arguments='{"cmd": "b"}',
                ),
            ),
        ]
        result = _repair_deepseek(tool_calls)

        assert result is not None
        assert len(result) == 2
        assert result[0].function.arguments == '{"cmd": "a"}'
        assert result[1].function.arguments == '{"cmd": "b"}'

    def test_handles_empty_arguments(self):
        """Test that empty arguments (default) are preserved."""
        tool_call = ChatCompletionMessageToolCall(
            id="call_789",
            type="function",
            function=Function(
                name="bash",
                # arguments defaults to "" when not provided
            ),
        )
        result = _repair_deepseek([tool_call])

        assert result is not None
        assert len(result) == 1
        # Empty string arguments should be preserved as-is (no sanitization needed)
        assert result[0].function.arguments == ""


class TestFixXaiArguments:
    """Tests for _fix_xai_arguments function.

    xAI emits every *string* tool-call argument as a JSON-encoded string, and
    often misplaces the quotes doing it. Payloads below have the shapes grok
    models produce.
    """

    def test_unwraps_grok_shaped_arguments(self):
        text = (
            '{"file_path": "\\"/workdir/solution.py\\"", '
            '"old": "\\"old text\\"", "from_line": 1}'
        )
        result = _fix_xai_arguments(text)
        assert json.loads(result) == {
            "file_path": "/workdir/solution.py",
            "old": "old text",
            "from_line": 1,
        }

    def test_unwraps_empty_string_argument(self):
        result = _fix_xai_arguments('{"old": "\\"\\""}')
        assert json.loads(result) == {"old": ""}

    @pytest.mark.parametrize(
        ("mangled", "path"),
        [
            ('";/tmp/small_03.jpg', "/tmp/small_03.jpg"),
            ('"; /tmp/r0.png', "/tmp/r0.png"),
            ('",/workdir/data/images/01.png', "/workdir/data/images/01.png"),
            ('", /tmp/overlay.jpg', "/tmp/overlay.jpg"),
            ('" /workdir/shared/image_0.jpg', "/workdir/shared/image_0.jpg"),
            ('"/workdir/shared/image_0.jpg', "/workdir/shared/image_0.jpg"),
            ('"/"/workdir/assets/img_00.png', "/workdir/assets/img_00.png"),
            (" /workdir/data/model.py", "/workdir/data/model.py"),
        ],
    )
    def test_strips_stray_quote_prefix(self, mangled: str, path: str):
        result = _fix_xai_arguments(json.dumps({"file_path": mangled}))
        assert json.loads(result) == {"file_path": path}

    def test_strips_stray_quote_prefix_beside_other_arguments(self):
        text = json.dumps(
            {"file_path": '";/workdir/shared/config/topology.json', "from_line": 1}
        )
        assert json.loads(_fix_xai_arguments(text)) == {
            "file_path": "/workdir/shared/config/topology.json",
            "from_line": 1,
        }

    @pytest.mark.parametrize(
        "unrecoverable",
        [
            '"',
            '",',
            '", ',
            ", ",
            ": 300, ",
            '", "from_line": 1, "to_line": 50}',
            '", "file_path": "/tmp/g0.jpg"}',
            '"; echo "===="; python3 -c "\nimport json\nprint(1)\n"',
            "https://placeholder",
            "data/images/00_cat.jpg",
        ],
    )
    def test_preserves_values_with_no_path_to_recover(self, unrecoverable: str):
        text = json.dumps({"file_path": unrecoverable})
        assert _fix_xai_arguments(text) == text

    def test_preserves_a_bash_command_that_is_a_bare_absolute_path(self):
        text = '{"command": "/workdir/.venv/bin/python"}'
        assert _fix_xai_arguments(text) == text

    def test_preserves_clean_arguments(self):
        text = '{"file_path": "/workdir/solution.py", "from_line": 1}'
        assert _fix_xai_arguments(text) == text

    def test_preserves_partially_quoted_arguments(self):
        text = '{"file_path": "/workdir/main.go", "old": "\\"hello\\"", "new": "hi"}'
        assert _fix_xai_arguments(text) == text

    def test_preserves_non_string_json(self):
        # "123" decodes to a number, not a string, so it is not double-encoded.
        text = '{"file_path": "\\"/tmp/a\\"", "extra": "123"}'
        assert _fix_xai_arguments(text) == text

    def test_preserves_arguments_with_no_strings(self):
        text = '{"from_line": 1, "to_line": 50}'
        assert _fix_xai_arguments(text) == text

    def test_preserves_malformed_json(self):
        assert _fix_xai_arguments('{"file_path": ') == '{"file_path": '

    def test_preserves_non_object_json(self):
        assert _fix_xai_arguments("[1, 2]") == "[1, 2]"

    def test_handles_empty_arguments(self):
        assert _fix_xai_arguments("{}") == "{}"


# =============================================================================
# Tests for LLM retry with exponential backoff
# =============================================================================


def _create_runner_with_tools_and_transcript(
    model: str = "test_model",
) -> EvaluationRunner:
    """Helper to create a runner with transcript and tools initialized for retry tests."""
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model=model,
        model_api_key="test_key",
        mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
        transcript_file="",
    )
    runner = EvaluationRunner(config, _make_test_task())
    runner.transcript = Transcript(run_id="test_run")
    runner.transcript.events.append(
        MessageAddedEvent(message=Message(role="user", content="Hello"))
    )
    runner.tools = []
    return runner


def _make_litellm_error(
    exc_class: type, message: str = "test error"
) -> litellm.exceptions.APIError:
    """Helper to create litellm exception instances."""
    return exc_class(message=message, model="test", llm_provider="anthropic")


def _make_mock_stream(
    chunks: list[dict[str, str | None]],
    fail_after: int | None = None,
    fail_with: Exception | None = None,
):
    """Create a mock object that passes isinstance(response, CustomStreamWrapper)
    and yields mock streaming chunks.

    Args:
        chunks: List of delta dicts (e.g. [{"content": "Hello", "role": None, "tool_calls": None}])
        fail_after: If set, raise fail_with after yielding this many chunks.
        fail_with: Exception to raise mid-stream.
    """
    from litellm import CustomStreamWrapper

    class MockStream(CustomStreamWrapper):
        def __init__(self):  # pyright: ignore[reportMissingSuperCall]
            pass  # Skip CustomStreamWrapper.__init__

        def __aiter__(self):
            return self._iter_chunks()

        async def _iter_chunks(self):
            from litellm.types.utils import ModelResponseStream, StreamingChoices

            for i, delta_dict in enumerate(chunks):
                if fail_after is not None and i >= fail_after and fail_with is not None:
                    raise fail_with
                chunk = Mock(spec=ModelResponseStream)
                choice = Mock(spec=StreamingChoices)
                choice.delta = Mock()
                choice.delta.model_dump = Mock(return_value=delta_dict)
                chunk.choices = [choice]
                yield chunk

    return MockStream()


def _make_mock_model_response(
    content: str = "response",
    tool_calls: list[Any] | None = None,
    usage: dict[str, int] | None = None,
    finish_reason: str = "stop",
) -> ModelResponse:
    """Create a mock ModelResponse that passes isinstance checks.

    Args:
        content: The assistant message content.
        tool_calls: The tool_calls value for the message.
        usage: If provided, dict with prompt_tokens and completion_tokens.
        finish_reason: The choice's finish_reason.
    """
    response = ModelResponse()
    choice = Choices(
        finish_reason=finish_reason,
        index=0,
        message=LiteLlmMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
    )
    response.choices = [choice]
    if usage:
        response.usage = Mock(  # pyright: ignore[reportAttributeAccessIssue]
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            # Explicit None so Mock's auto-attribute doesn't fabricate
            # prompt_tokens_details.cached_tokens as a truthy Mock.
            prompt_tokens_details=None,
        )
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_class",
    [
        litellm.exceptions.InternalServerError,
        litellm.exceptions.RateLimitError,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.Timeout,
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.BadGatewayError,
        litellm.InternalServerError,  # pyright: ignore[reportPrivateImportUsage]
        litellm.RateLimitError,  # pyright: ignore[reportPrivateImportUsage]
        litellm.ServiceUnavailableError,  # pyright: ignore[reportPrivateImportUsage]
        litellm.Timeout,  # pyright: ignore[reportPrivateImportUsage]
    ],
    ids=[
        "exceptions.InternalServerError",
        "exceptions.RateLimitError",
        "exceptions.ServiceUnavailableError",
        "exceptions.Timeout",
        "exceptions.APIConnectionError",
        "exceptions.BadGatewayError",
        "litellm.InternalServerError",
        "litellm.RateLimitError",
        "litellm.ServiceUnavailableError",
        "litellm.Timeout",
    ],
)
async def test_retry_recovers_from_transient_errors(
    exception_class: type,
):
    """Test that _collect_model_response retries on each transient error type
    and succeeds when the next attempt works."""
    runner = _create_runner_with_tools_and_transcript()

    call_count = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_litellm_error(exception_class, "transient failure")
        return _make_mock_stream([])

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response("Success after retry"),
        ),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        events = [event async for event in _collect(runner)]

    assert call_count == 2
    reset_events = [e for e in events if isinstance(e, MessageChunkResetEvent)]
    assert len(reset_events) == 1
    message_events = [e for e in events if isinstance(e, MessageAddedEvent)]
    assert len(message_events) == 1
    assert message_events[0].message.content == "Success after retry"


_FIREWORKS_GLM = "fireworks_ai/accounts/fireworks/models/glm-5p3"


@pytest.mark.asyncio
async def test_fireworks_load_shedding_moves_the_rest_of_the_run_to_priority(
    monkeypatch: pytest.MonkeyPatch,
):
    """Under auto, a 503 escalates the retry and every later turn, and usage records
    the tier; a 429 does not escalate."""
    monkeypatch.setenv(SERVICE_TIER_ENV, "auto")
    runner = _create_runner_with_tools_and_transcript(_FIREWORKS_GLM)
    source = BuiltinSource(runner.config)
    errors = [
        _make_litellm_error(litellm.exceptions.RateLimitError),
        _make_litellm_error(litellm.exceptions.ServiceUnavailableError),
    ]
    sent: list[dict[str, Any]] = []

    async def mock_acompletion(**kwargs):  # pyright: ignore[reportMissingParameterType]
        sent.append(kwargs)
        if errors:
            raise errors.pop(0)
        return _make_mock_stream([])

    async def turn() -> list[Any]:
        return [
            e async for e in source.collect(runner.transcript.messages, runner.tools)
        ]

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response(
                usage={"prompt_tokens": 10, "completion_tokens": 1}
            ),
        ),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        first = await turn()
        second = await turn()

    tiers = [p.get("extra_body", {}).get("service_tier") for p in sent]
    assert tiers == [None, None, "priority", "priority"]
    assert all(p["extra_headers"]["x-session-affinity"] == "test_run" for p in sent)
    usage = [e for e in first + second if isinstance(e, TokenUsageEvent)]
    assert [u.service_tier for u in usage] == ["priority", "priority"]


@pytest.mark.asyncio
async def test_fireworks_stays_on_the_default_tier_when_unset():
    """Unset, even a 503 keeps the run on the provider's default tier."""
    runner = _create_runner_with_tools_and_transcript(_FIREWORKS_GLM)
    errors = [_make_litellm_error(litellm.exceptions.ServiceUnavailableError)]
    sent: list[dict[str, Any]] = []

    async def mock_acompletion(**kwargs):  # pyright: ignore[reportMissingParameterType]
        sent.append(kwargs)
        if errors:
            raise errors.pop(0)
        return _make_mock_stream([])

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response(
                usage={"prompt_tokens": 10, "completion_tokens": 1}
            ),
        ),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        events = [event async for event in _collect(runner)]

    assert [p.get("extra_body", {}).get("service_tier") for p in sent] == [None, None]
    usage = [e for e in events if isinstance(e, TokenUsageEvent)]
    assert usage[0].service_tier is None


@pytest.mark.asyncio
async def test_vertex_gemini_resource_exhausted_moves_the_run_to_priority_under_auto(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(SERVICE_TIER_ENV, "auto")
    runner = _create_runner_with_tools_and_transcript(
        "vertex_ai/gemini-3.1-pro-preview"
    )
    errors = [_make_litellm_error(litellm.exceptions.RateLimitError)]
    sent: list[dict[str, Any]] = []

    async def mock_acompletion(**kwargs):  # pyright: ignore[reportMissingParameterType]
        sent.append(kwargs)
        if errors:
            raise errors.pop(0)
        return _make_mock_stream([])

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response(
                usage={"prompt_tokens": 10, "completion_tokens": 1}
            ),
        ),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        events = [event async for event in _collect(runner)]

    shared = [
        p.get("extra_headers", {}).get("X-Vertex-AI-LLM-Shared-Request-Type")
        for p in sent
    ]
    assert shared == [None, "priority"]
    usage = [e for e in events if isinstance(e, TokenUsageEvent)]
    assert usage[0].service_tier == "priority"


def _vertex_sse(traffic_type: str | None) -> bytes:
    usage: dict[str, Any] = {
        "promptTokenCount": 5,
        "candidatesTokenCount": 2,
        "totalTokenCount": 7,
    }
    if traffic_type:
        usage["trafficType"] = traffic_type
    chunks = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}]},
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "lo"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": usage,
        },
    ]
    return b"".join(f"data: {json.dumps(c)}\r\n\r\n".encode() for c in chunks)


async def _collect_vertex_gemini(
    traffic_type: str | None,
) -> tuple[list[Any], list[httpx.Request]]:
    """One turn through litellm's real Vertex streaming path over a fake transport."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=_vertex_sse(traffic_type),
            headers={"content-type": "text/event-stream"},
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    real_acompletion = litellm.acompletion

    async def acompletion(**kwargs: Any) -> Any:
        return await real_acompletion(**kwargs, client=client)

    runner = _create_runner_with_tools_and_transcript(
        "vertex_ai/gemini-3.1-pro-preview"
    )
    with (
        patch("litellm.acompletion", acompletion),
        patch.object(
            VertexBase, "_ensure_access_token_async", return_value=("tok", "proj")
        ),
    ):
        events = [event async for event in _collect(runner)]
    return events, requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "traffic_type", "sent_priority", "recorded"),
    [
        ("priority", "ON_DEMAND_PRIORITY", True, "priority"),
        ("priority", "ON_DEMAND", True, None),
        ("priority", None, True, "priority"),
        (None, "ON_DEMAND", False, None),
        (None, None, False, None),
    ],
)
async def test_vertex_usage_records_the_tier_that_served_the_request(
    monkeypatch: pytest.MonkeyPatch,
    mode: str | None,
    traffic_type: str | None,
    sent_priority: bool,
    recorded: str | None,
):
    if mode:
        monkeypatch.setenv(SERVICE_TIER_ENV, mode)

    events, requests = await _collect_vertex_gemini(traffic_type)

    assert len(requests) == 1
    shared_type = requests[0].headers.get("X-Vertex-AI-LLM-Shared-Request-Type")
    assert (shared_type == "priority") is sent_priority
    usage = [e for e in events if isinstance(e, TokenUsageEvent)]
    assert [(u.input_tokens, u.service_tier) for u in usage] == [(5, recorded)]


@pytest.mark.asyncio
async def test_collect_surfaces_refusal_finish_reason():
    """A Claude safety refusal (Anthropic stop_reason 'refusal', which litellm
    normalizes to 'content_filter') is surfaced on the MessageAddedEvent."""
    runner = _create_runner_with_tools_and_transcript()

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        return _make_mock_stream(
            [{"content": "", "role": "assistant", "tool_calls": None}]
        )

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response("", finish_reason="content_filter"),
        ),
    ):
        events = [event async for event in _collect(runner)]

    message_events = [e for e in events if isinstance(e, MessageAddedEvent)]
    assert len(message_events) == 1
    assert message_events[0].finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_collect_surfaces_normal_finish_reason():
    """Non-refusal turns carry their finish_reason on the MessageAddedEvent too."""
    runner = _create_runner_with_tools_and_transcript()

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        return _make_mock_stream(
            [{"content": "hi", "role": "assistant", "tool_calls": None}]
        )

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response("hi", finish_reason="stop"),
        ),
    ):
        events = [event async for event in _collect(runner)]

    message_events = [e for e in events if isinstance(e, MessageAddedEvent)]
    assert len(message_events) == 1
    assert message_events[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_retry_does_not_catch_non_transient_errors():
    """A 4xx that is neither a rate limit nor an auth error is not retried."""
    runner = _create_runner_with_tools_and_transcript()
    calls = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal calls
        calls += 1
        raise litellm.exceptions.BadRequestError(
            message="bad request", model="test", llm_provider="anthropic"
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        with pytest.raises(litellm.exceptions.BadRequestError):
            async for _ in _collect(runner):
                pass
    assert calls == 1


@pytest.mark.asyncio
async def test_auth_errors_get_a_few_attempts_then_reraise():
    """A 401 is retried AUTH_RETRY_MAX_ATTEMPTS times in total, since providers
    sometimes return spurious ones."""
    from karotte.agents.builtin_source import AUTH_RETRY_MAX_ATTEMPTS

    runner = _create_runner_with_tools_and_transcript()
    calls = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal calls
        calls += 1
        raise litellm.exceptions.AuthenticationError(
            message="The API key you provided is invalid.",
            model="test",
            llm_provider="fireworks_ai",
        )

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        with pytest.raises(litellm.exceptions.AuthenticationError):
            async for _ in _collect(runner):
                pass
    assert calls == AUTH_RETRY_MAX_ATTEMPTS == 3


@pytest.mark.asyncio
async def test_retry_exhaustion_reraises_original_error():
    """Test that after exhausting all retries, the original error is reraised."""
    runner = _create_runner_with_tools_and_transcript()

    call_count = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal call_count
        call_count += 1
        raise litellm.exceptions.InternalServerError(
            message="AnthropicError - Overloaded",
            model="test",
            llm_provider="anthropic",
        )

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        with pytest.raises(litellm.exceptions.InternalServerError, match="Overloaded"):
            async for _ in _collect(runner):
                pass

    assert call_count == 16


@pytest.mark.asyncio
async def test_retry_catches_midstream_overloaded_error():
    """Test that mid-stream 'Overloaded' errors (MidStreamFallbackError) are retried.

    This reproduces the real-world scenario from the traceback where Anthropic returns
    an Overloaded error mid-stream, which litellm wraps as MidStreamFallbackError
    (a subclass of ServiceUnavailableError).
    """
    from litellm.exceptions import MidStreamFallbackError

    runner = _create_runner_with_tools_and_transcript()

    call_count = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return _make_mock_stream(
                chunks=[
                    {"content": "partial", "role": None, "tool_calls": None},
                    {"content": "never seen", "role": None, "tool_calls": None},
                ],
                fail_after=1,
                fail_with=MidStreamFallbackError(
                    message="litellm.InternalServerError: AnthropicError - Overloaded",
                    model="test",
                    llm_provider="anthropic",
                ),
            )

        return _make_mock_stream([])

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response("Recovered after mid-stream error"),
        ),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        events = [event async for event in _collect(runner)]

    assert call_count == 2
    message_events = [e for e in events if isinstance(e, MessageAddedEvent)]
    assert len(message_events) == 1
    assert message_events[0].message.content == "Recovered after mid-stream error"
    # Partial chunks from failed attempt ARE yielded (streaming is live),
    # followed by a reset event, then chunks from the successful attempt
    chunk_events = [e for e in events if isinstance(e, MessageChunkEvent)]
    assert chunk_events[0].delta.content == "partial"
    reset_events = [e for e in events if isinstance(e, MessageChunkResetEvent)]
    assert len(reset_events) == 1


@pytest.mark.asyncio
async def test_midstream_error_does_not_pollute_transcript():
    """Test that a mid-stream error followed by successful retry leaves a clean transcript.

    Verifies that:
    - No partial MessageChunkEvents from the failed attempt appear in transcript
      (chunk events are not added to the transcript via _process_event)
    - Only the successful response's MessageAddedEvent is in the transcript
    - No duplicate or ghost events
    """
    runner = _create_runner_with_tools_and_transcript()
    call_count = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return _make_mock_stream(
                chunks=[
                    {"content": "Hello", "role": None, "tool_calls": None},
                    {"content": " world", "role": None, "tool_calls": None},
                    {"content": " this", "role": None, "tool_calls": None},
                    {"content": " never", "role": None, "tool_calls": None},
                ],
                fail_after=3,
                fail_with=litellm.exceptions.ServiceUnavailableError(
                    message="Overloaded", model="test", llm_provider="anthropic"
                ),
            )

        return _make_mock_stream(
            [
                {"content": "Clean", "role": None, "tool_calls": None},
                {"content": " response", "role": None, "tool_calls": None},
            ]
        )

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response("Clean response"),
        ),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        events = []
        async for event in _collect(runner):
            events.append(event)

    # The failed partial attempt is discarded (signaled by a
    # MessageChunkResetEvent), so the source emits exactly one final
    # MessageAddedEvent — the successful response. Only that message reaches the
    # transcript, since the runner never processes chunk events into it.
    message_added_events = [e for e in events if isinstance(e, MessageAddedEvent)]
    assert len(message_added_events) == 1
    assert message_added_events[0].message.role == "assistant"
    assert message_added_events[0].message.content == "Clean response"

    reset_events = [e for e in events if isinstance(e, MessageChunkResetEvent)]
    assert len(reset_events) == 1


def _map_anthropic_status(status_code: int, body: str = "upstream failure"):
    """Run a raw Anthropic HTTP status through litellm's real exception mapping."""
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type
    from litellm.llms.anthropic.common_utils import AnthropicError

    try:
        exception_type(
            model="claude-opus-5",
            original_exception=AnthropicError(status_code=status_code, message=body),
            custom_llm_provider="anthropic",
            completion_kwargs={},
            extra_kwargs={},
        )
    except Exception as mapped:
        return mapped
    raise AssertionError(f"litellm did not raise for status {status_code}")


@pytest.mark.asyncio
async def test_retry_recovers_from_cloudflare_520():
    """A Cloudflare 520 from api.anthropic.com is retried.

    litellm has no mapping branch for 520, so it falls through to a generic
    APIConnectionError rather than InternalServerError.
    """
    runner = _create_runner_with_tools_and_transcript()
    error = _map_anthropic_status(
        520,
        '{"type":"error","error":{"type":"api_error"},"retryable":true,"retry_after":60}',
    )
    assert isinstance(error, litellm.exceptions.APIConnectionError)

    call_count = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise error
        return _make_mock_stream([])

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response("Survived the 520"),
        ),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        events = [event async for event in _collect(runner)]

    assert call_count == 2
    message_events = [e for e in events if isinstance(e, MessageAddedEvent)]
    assert message_events[0].message.content == "Survived the 520"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 404, 413])
async def test_client_errors_are_not_retried(status_code: int):
    """4xx responses are the caller's fault and must fail fast. 401 is the
    exception (test_auth_errors_get_a_few_attempts_then_reraise): providers
    return false ones."""
    runner = _create_runner_with_tools_and_transcript()
    error = _map_anthropic_status(status_code)

    call_count = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal call_count
        call_count += 1
        raise error

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        with pytest.raises(type(error)):
            async for _ in _collect(runner):
                pass

    assert call_count == 1


def test_retry_after_read_from_response_headers():
    error = _make_litellm_error(litellm.exceptions.APIConnectionError)
    error.litellm_response_headers = {"retry-after": "60"}  # pyright: ignore[reportAttributeAccessIssue]

    assert _retry_after(error) == 60.0


def test_retry_after_read_from_error_body():
    """Anthropic puts retry_after in the JSON body, which litellm folds into the
    exception message rather than exposing as a field."""
    error = _map_anthropic_status(
        520, '{"type":"error","retryable":true,"retry_after":42}'
    )

    assert _retry_after(error) == 42.0


def test_retry_after_absent_falls_back_to_backoff():
    error = _make_litellm_error(litellm.exceptions.APIConnectionError)

    assert _retry_after(error) is None
    assert _retry_wait(1, retry_after=None) >= LLM_RETRY_WAIT_MIN_S


def test_retry_wait_honors_retry_after_over_backoff():
    """A provider hint wins over exponential backoff, in both directions."""
    assert _retry_wait(1, retry_after=60) == 60
    assert _retry_wait(16, retry_after=5) == 5


def test_retry_wait_caps_absurd_retry_after():
    assert _retry_wait(1, retry_after=99999) == LLM_RETRY_AFTER_MAX_S


@pytest.mark.asyncio
async def test_retry_sleeps_for_provider_requested_delay():
    """The 60s Anthropic asked for is actually slept, not the ~1s backoff."""
    runner = _create_runner_with_tools_and_transcript()
    error = _map_anthropic_status(
        520, '{"type":"error","retryable":true,"retry_after":60}'
    )

    call_count = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise error
        return _make_mock_stream([])

    sleep_mock = AsyncMock()
    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response("ok"),
        ),
        patch("asyncio.sleep", sleep_mock),
    ):
        _ = [event async for event in _collect(runner)]

    sleep_mock.assert_awaited_once_with(60.0)


@pytest.mark.asyncio
async def test_midstream_error_yields_partial_chunks_then_recovers():
    """Test that mid-stream chunks are yielded live (for real-time streaming),
    and that after a retry the successful response is also yielded.

    Consumers receive partial chunks from failed attempts — this is expected
    because streaming is now live. Consumers should handle this (e.g. by
    resetting on seeing a new MessageAddedEvent after partial chunks).
    """
    runner = _create_runner_with_tools_and_transcript()
    call_count = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return _make_mock_stream(
                chunks=[
                    {"content": "partial", "role": None, "tool_calls": None},
                    {"content": "more", "role": None, "tool_calls": None},
                ],
                fail_after=1,
                fail_with=litellm.exceptions.InternalServerError(
                    message="Overloaded", model="test", llm_provider="anthropic"
                ),
            )

        return _make_mock_stream(
            [
                {"content": "Good", "role": None, "tool_calls": None},
            ]
        )

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response("Good response"),
        ),
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
    ):
        yielded_events = []
        async for event in _collect(runner):
            yielded_events.append(event)

    # Partial chunks from failed attempt ARE yielded (live streaming),
    # followed by a reset event, then chunks from the successful attempt
    chunk_events = [e for e in yielded_events if isinstance(e, MessageChunkEvent)]
    assert len(chunk_events) == 2
    assert chunk_events[0].delta.content == "partial"
    assert chunk_events[1].delta.content == "Good"

    # Reset event appears between the partial and good chunks
    reset_events = [e for e in yielded_events if isinstance(e, MessageChunkResetEvent)]
    assert len(reset_events) == 1
    chunk_idx = yielded_events.index(chunk_events[0])
    reset_idx = yielded_events.index(reset_events[0])
    good_idx = yielded_events.index(chunk_events[1])
    assert chunk_idx < reset_idx < good_idx

    # The MessageAddedEvent should have the clean response from the successful attempt
    message_events = [e for e in yielded_events if isinstance(e, MessageAddedEvent)]
    assert len(message_events) == 1
    assert message_events[0].message.content == "Good response"


@pytest.mark.asyncio
async def test_successful_call_yields_all_chunks_and_events():
    """Test that when no retry is needed, all chunks and events are yielded normally."""
    runner = _create_runner_with_tools_and_transcript()

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        return _make_mock_stream(
            [
                {"content": "Hello", "role": None, "tool_calls": None},
                {"content": " world", "role": None, "tool_calls": None},
            ]
        )

    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response(
                "Hello world",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            ),
        ),
    ):
        yielded_events = []
        async for event in _collect(runner):
            yielded_events.append(event)

    chunk_events = [e for e in yielded_events if isinstance(e, MessageChunkEvent)]
    assert len(chunk_events) == 2
    assert chunk_events[0].delta.content == "Hello"
    assert chunk_events[1].delta.content == " world"

    message_events = [e for e in yielded_events if isinstance(e, MessageAddedEvent)]
    assert len(message_events) == 1
    assert message_events[0].message.content == "Hello world"

    token_events = [e for e in yielded_events if isinstance(e, TokenUsageEvent)]
    assert len(token_events) == 1
    assert token_events[0].input_tokens == 10
    assert token_events[0].output_tokens == 5


# =============================================================================
# Tests for _retry_wait
# =============================================================================


def test_retry_wait_never_exceeds_max():
    """Test that _retry_wait never returns a value exceeding LLM_RETRY_WAIT_MAX_S."""
    from karotte.agents.builtin_source import (
        LLM_RETRY_WAIT_MAX_S,
        _retry_wait,  # pyright: ignore[reportPrivateUsage]
    )

    for attempt in range(1, 20):
        for _ in range(50):
            wait = _retry_wait(attempt)
            assert wait <= LLM_RETRY_WAIT_MAX_S, (
                f"_retry_wait({attempt}) returned {wait}, exceeding max {LLM_RETRY_WAIT_MAX_S}"
            )


def test_retry_wait_never_below_min():
    """Test that _retry_wait never returns a value below LLM_RETRY_WAIT_MIN_S."""
    from karotte.agents.builtin_source import (
        LLM_RETRY_WAIT_MIN_S,
        _retry_wait,  # pyright: ignore[reportPrivateUsage]
    )

    for attempt in range(1, 20):
        wait = _retry_wait(attempt)
        assert wait >= LLM_RETRY_WAIT_MIN_S


# =============================================================================
# Tests for Vertex AI Priority PayGo headers
# =============================================================================


def _create_runner_for_completion_params(model: str) -> EvaluationRunner:
    """Helper to create a runner for testing _get_completion_params."""
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model=model,
        model_api_key="test_key" if not model.startswith("vertex_ai/") else None,
        mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
        transcript_file="",
    )
    runner = EvaluationRunner(config, _make_test_task())
    runner.transcript = Transcript(run_id="test_run")
    runner.transcript.events.append(
        MessageAddedEvent(message=Message(role="user", content="Hello"))
    )
    runner.tools = []
    return runner


def test_proxy_routes_non_anthropic_model_through_openai_endpoint():
    """A non-Anthropic model with KAROTTE_PROXY_URL set gets an OpenAI-compatible api_base."""
    runner = _create_runner_for_completion_params("mistral/mistral-medium-3-5")
    with patch.dict(
        os.environ, {"KAROTTE_PROXY_URL": "https://proxy.example"}, clear=True
    ):
        params = _completion_params(runner)
    assert params["api_base"] == "https://proxy.example/v1"


def test_proxy_not_applied_to_anthropic_model():
    """Anthropic keeps its native route (no api_base) and inline cache control."""
    runner = _create_runner_for_completion_params("claude-sonnet-4-20250514")
    with patch.dict(
        os.environ, {"KAROTTE_PROXY_URL": "https://proxy.example"}, clear=True
    ):
        params = _completion_params(runner)
    assert "api_base" not in params
    assert params["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_no_api_base_without_proxy():
    runner = _create_runner_for_completion_params("mistral/mistral-medium-3-5")
    with patch.dict(os.environ, {}, clear=True):
        params = _completion_params(runner)
    assert "api_base" not in params


@pytest.mark.parametrize(
    "model",
    [
        "vertex_ai/gemini-3.1-pro-preview",
        "vertex_ai/gemini-2.5-pro",
        "vertex_ai/gemini-2.0-flash",
    ],
)
def test_priority_paygo_headers_for_gemini_models(model: str):
    runner = _create_runner_for_completion_params(model)

    with patch.dict(os.environ, {SERVICE_TIER_ENV: "priority"}, clear=True):
        params = _completion_params(runner)

    assert params["extra_headers"] == {
        "X-Vertex-AI-LLM-Request-Type": "shared",
        "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
    }


def test_no_priority_paygo_headers_for_gemini_by_default():
    runner = _create_runner_for_completion_params("vertex_ai/gemini-3.1-pro-preview")

    with patch.dict(os.environ, {}, clear=True):
        params = _completion_params(runner)

    assert "extra_headers" not in params


def test_no_priority_paygo_headers_for_non_gemini_vertex_models():
    """Test that non-Gemini Vertex AI models don't get Priority PayGo headers."""
    runner = _create_runner_for_completion_params("vertex_ai/claude-sonnet-4")

    with patch.dict(os.environ, {SERVICE_TIER_ENV: "priority"}, clear=True):
        params = _completion_params(runner)

    assert "extra_headers" not in params


def test_no_priority_paygo_headers_for_non_vertex_models():
    """Test that non-Vertex AI models don't get Priority PayGo headers."""
    runner = _create_runner_for_completion_params("claude-sonnet-4-20250514")

    with patch.dict(os.environ, {SERVICE_TIER_ENV: "priority"}, clear=True):
        params = _completion_params(runner)

    assert "extra_headers" not in params


# =============================================================================
# Tests for claude-fable-5 LiteLLM routing workaround
# =============================================================================


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Every bare Claude id names its provider, so a release litellm has not
        # heard of routes the same way as one it has.
        ("claude-fable-5", "anthropic/claude-fable-5"),
        ("claude-sonnet-5", "anthropic/claude-sonnet-5"),
        ("claude-sonnet-4-20250514", "anthropic/claude-sonnet-4-20250514"),
        ("claude-brand-new-9", "anthropic/claude-brand-new-9"),
        # An id that already names its provider is left alone.
        ("anthropic/claude-opus-5", "anthropic/claude-opus-5"),
        ("vertex_ai/claude-opus-4-6", "vertex_ai/claude-opus-4-6"),
    ],
)
def test_bare_claude_ids_are_routed_via_the_anthropic_provider(
    model: str, expected: str
):
    runner = _create_runner_for_completion_params(model)

    with patch.dict(os.environ, {}, clear=True):
        params = _completion_params(runner)

    assert params["model"] == expected


# =============================================================================
# Tests for summarized thinking on adaptive Claude models
# =============================================================================


def test_claude_fable_5_requests_summarized_thinking():
    """claude-fable-5 defaults thinking.display to "omitted"; request "summarized"
    so the reasoning summary lands in reasoning_content instead of an empty field."""
    runner = _create_runner_for_completion_params("claude-fable-5")

    with patch.dict(os.environ, {}, clear=True):
        params = _completion_params(runner)

    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_claude_sonnet_5_requests_summarized_thinking():
    runner = _create_runner_for_completion_params("claude-sonnet-5")

    with patch.dict(os.environ, {}, clear=True):
        params = _completion_params(runner)

    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_claude_opus_4_8_requests_adaptive_thinking():
    """Opus 4.8 thinks only when asked: omitting the parameter runs it with no
    thinking at all, unlike the 5th generation."""
    runner = _create_runner_for_completion_params("claude-opus-4-8")

    with patch.dict(os.environ, {}, clear=True):
        params = _completion_params(runner)

    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_non_adaptive_models_do_not_request_thinking():
    runner = _create_runner_for_completion_params("claude-sonnet-4-20250514")

    with patch.dict(os.environ, {}, clear=True):
        params = _completion_params(runner)

    assert "thinking" not in params


# =============================================================================
# Tests for turn limit
# =============================================================================


@pytest.mark.asyncio
async def test_turn_limit_terminates_run_with_error(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """Test that exceeding the turn limit terminates the run with a TurnLimitReachedError."""
    from karotte.judges.regex_judge import RegexJudge

    turn_limit = 3
    sample_config.turn_limit = turn_limit
    sample_config.mcp_server_config.port = mcp_server.config.port

    call_count = 0

    async def mock_collect_with_tool_calls(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        """Mock that always returns a tool call, forcing an infinite loop."""
        nonlocal call_count
        call_count += 1
        yield MessageAddedEvent(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=f"call_{call_count}",
                        type="function",
                        function=Function(
                            name="bash", arguments='{"command": "echo hi"}'
                        ),
                    )
                ],
            )
        )

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect_with_tool_calls)

    class SimpleStep(Step):
        @property
        def instructions(self):
            return "Do something"

        @property
        def judge(self):
            return RegexJudge([])

    @final
    class TurnLimitTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "turn-limit-task"

        @property
        def tools(self):
            return ["bash"]

        @property
        def steps(self):
            return [SimpleStep(config=sample_config)]

    sample_config.task_id = "turn-limit-task"
    runner = EvaluationRunner(sample_config, TurnLimitTask(sample_config))

    events = []
    async for event in runner.run():
        events.append(event)

    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].exception_type == "TurnLimitReachedError"
    assert str(turn_limit) in error_events[0].message

    completed_events = [e for e in events if isinstance(e, TaskCompletedEvent)]
    assert len(completed_events) == 1
    assert completed_events[0].status == "error"

    # The model should have been called exactly turn_limit times
    assert call_count == turn_limit


def _tool_call_message(call_id: str) -> Message:
    return Message(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=call_id,
                type="function",
                function=Function(name="bash", arguments='{"command": "echo hi"}'),
            )
        ],
    )


def _time_limit_task(config: EvaluationRunConfig) -> Task:
    from karotte.judges.regex_judge import RegexJudge

    class SimpleStep(Step):
        @property
        def instructions(self):
            return "Do something"

        @property
        def judge(self):
            return RegexJudge([])

    @final
    class TimeLimitTask(Task):
        @property
        def system_prompt(self) -> str | None:
            return None

        id = "time-limit-task"

        @property
        def tools(self):
            return ["bash"]

        @property
        def steps(self):
            return [SimpleStep(config=config)]

    return TimeLimitTask(config)


def _tool_result_counter_notes(events: list[Any]) -> list[str]:
    """Every 'Time remaining:' note folded into a tool-result message."""
    notes = []
    for event in events:
        if not isinstance(event, MessageAddedEvent):
            continue
        message = event.message
        if message.role != "tool" or not isinstance(message.content, list):
            continue
        for part in message.content:
            text = part.get("text", "")
            if text.startswith("Time remaining:"):
                notes.append(text)
    return notes


@pytest.mark.asyncio
async def test_step_time_limit_error_terminates_run(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """A step exceeding its time limit aborts the run when behavior is 'error'."""
    sample_config.step_time_limit_seconds = 5
    sample_config.on_step_time_limit = "error"
    sample_config.mcp_server_config.port = mcp_server.config.port

    clock = {"t": 0.0}
    monkeypatch.setattr("karotte.agents.message_loop._monotonic", lambda: clock["t"])

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        yield MessageAddedEvent(message=_tool_call_message(f"call_{call_count}"))
        clock["t"] += 10  # push the next turn past the deadline

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].exception_type == "StepTimeLimitReachedError"
    assert "5" in error_events[0].message

    completed = [e for e in events if isinstance(e, TaskCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].status == "error"

    # One turn ran before the deadline was crossed on the next turn's check.
    assert call_count == 1


@pytest.mark.asyncio
async def test_step_time_limit_score_ends_step_and_scores(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """With behavior 'score', a timed-out step is scored instead of erroring."""
    sample_config.step_time_limit_seconds = 5
    sample_config.on_step_time_limit = "score"
    sample_config.mcp_server_config.port = mcp_server.config.port

    clock = {"t": 0.0}
    monkeypatch.setattr("karotte.agents.message_loop._monotonic", lambda: clock["t"])

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        yield MessageAddedEvent(message=_tool_call_message(f"call_{call_count}"))
        clock["t"] += 10

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    assert not [e for e in events if isinstance(e, ErrorEvent)]
    assert [e for e in events if isinstance(e, ScoringEvent)]
    assert [e for e in events if isinstance(e, StepCompletedEvent)]
    completed = [e for e in events if isinstance(e, TaskCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].status != "error"
    assert call_count == 1
    assert _tool_result_counter_notes(events) == ["Time remaining: 0 seconds"]


def _context_counter_notes(events: list[Any]) -> list[str]:
    """Every 'Context remaining:' note folded into a tool-result message."""
    notes = []
    for event in events:
        if not isinstance(event, MessageAddedEvent):
            continue
        message = event.message
        if message.role != "tool" or not isinstance(message.content, list):
            continue
        for part in message.content:
            text = part.get("text", "")
            if text.startswith("Context remaining:"):
                notes.append(text)
    return notes


@pytest.mark.asyncio
async def test_step_context_window_limit_error_terminates_run(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """A step exceeding its context-window limit aborts the run on 'error'."""
    sample_config.step_context_window_limit = 1000
    sample_config.on_step_context_window_limit = "error"
    sample_config.mcp_server_config.port = mcp_server.config.port

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        yield MessageAddedEvent(message=_tool_call_message(f"call_{call_count}"))
        # Report a context window already past the limit for the next turn's check.
        yield TokenUsageEvent(input_tokens=5000, output_tokens=1)

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].exception_type == "StepContextWindowLimitReachedError"
    assert "1000" in error_events[0].message

    completed = [e for e in events if isinstance(e, TaskCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].status == "error"

    # One turn ran before the limit was crossed on the next turn's check.
    assert call_count == 1


@pytest.mark.asyncio
async def test_step_context_window_limit_score_ends_step_and_scores(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """With behavior 'score', an over-context step is scored instead of erroring."""
    sample_config.step_context_window_limit = 1000
    sample_config.on_step_context_window_limit = "score"
    sample_config.mcp_server_config.port = mcp_server.config.port

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        yield MessageAddedEvent(message=_tool_call_message(f"call_{call_count}"))
        yield TokenUsageEvent(input_tokens=5000, output_tokens=1)

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    assert not [e for e in events if isinstance(e, ErrorEvent)]
    assert [e for e in events if isinstance(e, ScoringEvent)]
    assert [e for e in events if isinstance(e, StepCompletedEvent)]
    completed = [e for e in events if isinstance(e, TaskCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].status != "error"
    assert call_count == 1
    # The turn's counter reflects the limit already exhausted by its usage.
    assert _context_counter_notes(events) == ["Context remaining: 0"]


@pytest.mark.asyncio
async def test_remaining_context_counter_folded_into_tool_result(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """The remaining-context counter is appended to a turn's last tool result."""
    sample_config.step_context_window_limit = 10_000
    sample_config.mcp_server_config.port = mcp_server.config.port

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield MessageAddedEvent(message=_tool_call_message("call_1"))
            yield TokenUsageEvent(input_tokens=2000, output_tokens=1)
        else:
            yield MessageAddedEvent(
                message=Message(role="assistant", content="done", tool_calls=None)
            )

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    assert _context_counter_notes(events) == ["Context remaining: 8000"]


@pytest.mark.asyncio
async def test_no_remaining_context_counter_without_limit(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """No counter is injected when the step has no context-window limit."""
    sample_config.step_context_window_limit = None
    sample_config.mcp_server_config.port = mcp_server.config.port

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield MessageAddedEvent(message=_tool_call_message("call_1"))
            yield TokenUsageEvent(input_tokens=2000, output_tokens=1)
        else:
            yield MessageAddedEvent(
                message=Message(role="assistant", content="done", tool_calls=None)
            )

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    assert _context_counter_notes(events) == []


@pytest.mark.asyncio
async def test_context_counter_suppressed_when_injection_disabled(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """A limit is still enforced, but no counter is injected when disabled."""
    sample_config.step_context_window_limit = 10_000
    sample_config.inject_context_remaining_counter = False
    sample_config.mcp_server_config.port = mcp_server.config.port

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield MessageAddedEvent(message=_tool_call_message("call_1"))
            yield TokenUsageEvent(input_tokens=2000, output_tokens=1)
        else:
            yield MessageAddedEvent(
                message=Message(role="assistant", content="done", tool_calls=None)
            )

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    assert _context_counter_notes(events) == []


@pytest.mark.asyncio
async def test_remaining_time_counter_folded_into_tool_result(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """The remaining-time counter is appended to a turn's last tool result."""
    sample_config.step_time_limit_seconds = 1000
    sample_config.mcp_server_config.port = mcp_server.config.port

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield MessageAddedEvent(message=_tool_call_message("call_1"))
        else:
            yield MessageAddedEvent(
                message=Message(role="assistant", content="done", tool_calls=None)
            )

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    notes = _tool_result_counter_notes(events)
    assert len(notes) == 1
    assert notes[0].startswith("Time remaining:")


@pytest.mark.asyncio
async def test_no_remaining_time_counter_without_limit(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """No counter is injected when the step has no time limit."""
    sample_config.step_time_limit_seconds = None
    sample_config.mcp_server_config.port = mcp_server.config.port

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield MessageAddedEvent(message=_tool_call_message("call_1"))
        else:
            yield MessageAddedEvent(
                message=Message(role="assistant", content="done", tool_calls=None)
            )

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    assert _tool_result_counter_notes(events) == []


@pytest.mark.asyncio
async def test_counter_suppressed_when_injection_disabled(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
):
    """A time limit is still enforced, but no counter is injected when disabled."""
    sample_config.step_time_limit_seconds = 1000
    sample_config.inject_time_remaining_counter = False
    sample_config.mcp_server_config.port = mcp_server.config.port

    call_count = 0

    async def mock_collect(self, _messages, _tools):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield MessageAddedEvent(message=_tool_call_message("call_1"))
        else:
            yield MessageAddedEvent(
                message=Message(role="assistant", content="done", tool_calls=None)
            )

    monkeypatch.setattr(BuiltinSource, "collect", mock_collect)

    sample_config.task_id = "time-limit-task"
    runner = EvaluationRunner(sample_config, _time_limit_task(sample_config))

    events = [event async for event in runner.run()]

    assert _tool_result_counter_notes(events) == []


@pytest.mark.asyncio
async def test_get_external_message_times_out(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: EvaluationRunConfig,
):
    backend_client = AsyncMock()
    backend_client.get_message = AsyncMock(return_value=None)
    source = BackendSource(backend_client, sample_config.run_id)

    # Advance the clock past the timeout after the two initial readings.
    n = {"calls": 0}

    def fake_time() -> float:
        n["calls"] += 1
        if n["calls"] <= 2:
            return 0.0
        return EXTERNAL_MESSAGE_TIMEOUT_S + 1

    monkeypatch.setattr("karotte.agents.backend_source.time.time", fake_time)

    with pytest.raises(TimeoutError):
        async for _ in source.collect([], []):
            pass


@pytest.mark.asyncio
async def test_get_external_message_returns_message(
    sample_config: EvaluationRunConfig,
):
    message = Message(role="assistant", content="hello")
    msg_event = MessageAddedEvent(message=message)

    backend_client = AsyncMock()
    backend_client.get_message = AsyncMock(return_value=msg_event)
    backend_client.delete_message = AsyncMock()
    source = BackendSource(backend_client, sample_config.run_id)

    events = [event async for event in source.collect([], [])]

    assert events == [msg_event]
    backend_client.delete_message.assert_awaited_once_with(sample_config.run_id)


@final
class _NoStepsTask(Task):
    id = "empty-task"

    @property
    def system_prompt(self) -> str | None:
        return None

    @property
    def steps(self):
        return []

    @property
    def tools(self):
        return []


async def _task_started(
    config: EvaluationRunConfig, mcp_server: HttpMcpServer
) -> TaskStartedEvent:
    config.task_id = "empty-task"
    config.mcp_server_config.port = mcp_server.config.port
    runner = EvaluationRunner(config, _NoStepsTask(config))
    started = [e async for e in runner.run() if isinstance(e, TaskStartedEvent)]
    assert len(started) == 1
    return started[0]


async def _warnings_during(awaitable: Any) -> list[str]:
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await awaitable
    finally:
        logger.remove(handler)
    return messages


class TestReasoningEffortIsRecorded:
    """The transcript says what reached the provider, not what was asked for."""

    @pytest.mark.asyncio
    async def test_records_the_resolved_effort(
        self, sample_config: EvaluationRunConfig, mcp_server: HttpMcpServer
    ):
        config = sample_config.model_copy(
            update={"model": "claude-opus-5", "reasoning_effort": "max"}
        )
        assert (await _task_started(config, mcp_server)).reasoning_effort == "max"

    @pytest.mark.asyncio
    async def test_records_nothing_when_unset(
        self, sample_config: EvaluationRunConfig, mcp_server: HttpMcpServer
    ):
        config = sample_config.model_copy(update={"model": "claude-opus-5"})
        assert (await _task_started(config, mcp_server)).reasoning_effort is None

    @pytest.mark.asyncio
    async def test_records_nothing_for_an_agent_that_never_sends_it(
        self,
        sample_config: EvaluationRunConfig,
        mcp_server: HttpMcpServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        config = sample_config.model_copy(
            update={
                "model": "claude-opus-5",
                "agent": "mistral-vibe",
                "reasoning_effort": "max",
            }
        )
        assert (await _task_started(config, mcp_server)).reasoning_effort is None

    @pytest.mark.asyncio
    async def test_warns_when_the_requested_effort_is_dropped(
        self, sample_config: EvaluationRunConfig, mcp_server: HttpMcpServer
    ):
        config = sample_config.model_copy(
            update={
                "model": "together_ai/moonshotai/Kimi-K2.6",
                "reasoning_effort": "max",
            }
        )
        warnings = await _warnings_during(_task_started(config, mcp_server))
        assert any("reasoning_effort" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_does_not_warn_when_the_effort_is_applied(
        self, sample_config: EvaluationRunConfig, mcp_server: HttpMcpServer
    ):
        config = sample_config.model_copy(
            update={"model": "claude-opus-5", "reasoning_effort": "max"}
        )
        warnings = await _warnings_during(_task_started(config, mcp_server))
        assert not any("reasoning_effort" in w for w in warnings)
