from abc import ABC
from typing import final, override

import pytest

from karotte import Task
from karotte.evaluation_runner import EvaluationRunner
from karotte.mcp_servers.http_mcp_server import HttpMcpServer
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.schemas.transcript import MessageAddedEvent

_MCP_CONFIG = HttpMcpServerConfig(host="0.0.0.0", port=8080)


def _make_config(model: str = "claude-opus-4-8") -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="test_run",
        task_id="test-task",
        model=model,
        model_api_key="test_key",
        mcp_server_config=_MCP_CONFIG,
        transcript_file="",
    )


class TestTaskSystemPromptIsRequired:
    """A task without a `system_prompt` must not be instantiable."""

    def test_missing_system_prompt_raises_on_instantiation(self) -> None:
        class LegacyTask(Task):  # pyright: ignore[reportImplicitAbstractClass]
            id: str = "legacy-task"

            @property
            def steps(self):
                return []

            @property
            def tools(self):
                return []

        with pytest.raises(TypeError) as exc:
            LegacyTask(_make_config())  # pyright: ignore[reportAbstractUsage]

        message = str(exc.value)
        assert "LegacyTask" in message
        assert "system_prompt" in message
        assert "get_system_prompt" in message

    def test_intermediate_base_without_system_prompt_is_allowed(self) -> None:
        """Only instantiation fails, so shared bases can leave it to subclasses."""

        class Base(Task, ABC):
            @property
            def steps(self):
                return []

            @property
            def tools(self):
                return []

        @final
        class Concrete(Base):
            id = "concrete-task"

            @property
            @override
            def system_prompt(self) -> str:
                return "prompt"

        assert Concrete(_make_config()).system_prompt == "prompt"


class TestTaskSystemPromptOverride:
    """The value a task returns is what reaches the run."""

    def test_custom_system_prompt(self) -> None:
        config = _make_config()

        @final
        class CustomTask(Task):
            id = "custom-task"

            @property
            @override
            def system_prompt(self) -> str:
                return "You are a custom agent."

            @property
            def steps(self):
                return []

            @property
            def tools(self):
                return []

        assert CustomTask(config).system_prompt == "You are a custom agent."

    def test_none_system_prompt(self) -> None:
        config = _make_config()

        @final
        class NoPromptTask(Task):
            id = "no-prompt-task"

            @property
            @override
            def system_prompt(self) -> str | None:
                return None

            @property
            def steps(self):
                return []

            @property
            def tools(self):
                return []

        task = NoPromptTask(config)
        assert task.system_prompt is None


class TestEvaluationRunnerUsesTaskSystemPrompt:
    """EvaluationRunner should use task.system_prompt for the system message."""

    @pytest.mark.asyncio
    async def test_runner_uses_task_system_prompt(
        self,
        sample_config: EvaluationRunConfig,
        mcp_server: HttpMcpServer,
    ) -> None:
        @final
        class CustomPromptTask(Task):
            id = "custom-prompt-task"

            @property
            @override
            def system_prompt(self) -> str:
                return "Custom system prompt for testing."

            @property
            def steps(self):
                return []

            @property
            def tools(self):
                return []

        sample_config.task_id = "custom-prompt-task"
        sample_config.mcp_server_config.port = mcp_server.config.port

        runner = EvaluationRunner(sample_config, CustomPromptTask(sample_config))

        system_messages = []
        async for event in runner.run():
            if isinstance(event, MessageAddedEvent) and event.message.role == "system":
                system_messages.append(event.message)

        assert len(system_messages) == 1
        assert system_messages[0].content == "Custom system prompt for testing."

    @pytest.mark.asyncio
    async def test_runner_skips_system_message_when_none(
        self,
        sample_config: EvaluationRunConfig,
        mcp_server: HttpMcpServer,
    ) -> None:
        @final
        class NoPromptTask(Task):
            id = "no-prompt-task"

            @property
            @override
            def system_prompt(self) -> str | None:
                return None

            @property
            def steps(self):
                return []

            @property
            def tools(self):
                return []

        sample_config.task_id = "no-prompt-task"
        sample_config.mcp_server_config.port = mcp_server.config.port

        runner = EvaluationRunner(sample_config, NoPromptTask(sample_config))

        system_messages = []
        async for event in runner.run():
            if isinstance(event, MessageAddedEvent) and event.message.role == "system":
                system_messages.append(event.message)

        assert len(system_messages) == 0
