from pathlib import Path

import pytest

from karotte.evaluation_runner import EvaluationRunner
from karotte.fake_model import setup_fake_model
from karotte.load_tasks import load_task
from karotte.mcp_servers.http_mcp_server import HttpMcpServer
from karotte.schemas.chat import (
    ChatCompletionMessageToolCall,
    Function,
    Message,
)
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.schemas.run_state import RunState
from karotte.schemas.transcript import (
    ErrorEvent,
    ScoringEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
)


@pytest.mark.asyncio
async def test_fake_model(
    mcp_server: HttpMcpServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    transcript_file = tmp_path / "transcript.json"

    config = EvaluationRunConfig(
        run_id="",
        task_id="test-task",
        model="fake-model",
        model_api_key="fake-api-key",
        mcp_server_config=HttpMcpServerConfig(port=mcp_server.config.port),
        transcript_file=transcript_file.as_posix(),
    )

    # Mock the environment module with test messages
    test_messages = [
        Message(role="assistant", content="Here is the answer: 42."),
        Message(role="assistant", content="And another answer: 43."),
    ]

    # Create a mock environment module
    from types import ModuleType

    mock_env_fake_model = ModuleType("environment.fake_model")
    mock_env_fake_model.get_messages = lambda _: test_messages  # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType]

    import sys

    sys.modules["environment.fake_model"] = mock_env_fake_model
    monkeypatch.setitem(sys.modules, "environment.fake_model", mock_env_fake_model)

    runner = EvaluationRunner(config, load_task(config))

    setup_fake_model(runner, config)

    run_state: RunState | None = None
    async for event in runner.run():
        if not run_state:
            assert isinstance(event, TaskStartedEvent)
            run_state = RunState(event)
        else:
            run_state.apply(event)

    assert run_state is not None
    assert run_state.score == 1.0


@pytest.mark.asyncio
async def test_step_ending_empty_message_is_not_treated_as_a_stalled_turn(
    mcp_server: HttpMcpServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Envs generated from older templates end each step with an empty assistant
    message; that has to keep ending the step rather than exhausting the source."""
    config = EvaluationRunConfig(
        run_id="",
        task_id="test-task",
        model="fake-model",
        model_api_key="fake-api-key",
        mcp_server_config=HttpMcpServerConfig(port=mcp_server.config.port),
        transcript_file=(tmp_path / "transcript.json").as_posix(),
    )

    import sys
    from types import ModuleType

    test_messages = [
        Message(role="assistant", content="Here is the answer: 42."),
        Message(
            role="assistant",
            content="And another answer: 43.",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(name="bash", arguments='{"command": "echo hi"}'),
                )
            ],
        ),
        Message(role="assistant", content=""),
    ]
    mock_env_fake_model = ModuleType("environment.fake_model")
    mock_env_fake_model.get_messages = lambda _: test_messages  # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType]
    monkeypatch.setitem(sys.modules, "environment.fake_model", mock_env_fake_model)

    runner = EvaluationRunner(config, load_task(config))
    setup_fake_model(runner, config)

    events = [event async for event in runner.run()]

    assert [e for e in events if isinstance(e, ErrorEvent)] == []
    completed = [e for e in events if isinstance(e, TaskCompletedEvent)]
    assert [e.status for e in completed] == ["passed"]
    assert len([e for e in events if isinstance(e, ScoringEvent)]) == 2


@pytest.mark.asyncio
async def test_raises_error_if_not_enough_messages_defined(
    mcp_server: HttpMcpServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    transcript_file = tmp_path / "transcript.json"

    config = EvaluationRunConfig(
        run_id="",
        task_id="test-task",
        model="fake-model",
        model_api_key="fake-api-key",
        mcp_server_config=HttpMcpServerConfig(port=mcp_server.config.port),
        transcript_file=transcript_file.as_posix(),
    )

    # Create a mock environment module with empty messages
    import sys
    from types import ModuleType

    mock_env_fake_model = ModuleType("environment.fake_model")
    mock_env_fake_model.get_messages = lambda _: []  # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType]
    monkeypatch.setitem(sys.modules, "environment.fake_model", mock_env_fake_model)

    runner = EvaluationRunner(config, load_task(config))

    setup_fake_model(runner, config)

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
    assert "Not enough messages defined for fake model." in error_event.message

    assert task_completed_event is not None
    assert task_completed_event.status == "error"
