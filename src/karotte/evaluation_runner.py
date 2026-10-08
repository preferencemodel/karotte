import asyncio
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk
from loguru import logger
from mcp.types import Tool

from karotte import Step, Task, ToolConfigWriter
from karotte.agents import (
    Agent,
    BuiltinAgent,
    CliAgent,
    ExternalAgent,
    RunContext,
    get_cli_agent_type,
)
from karotte.backend_client import BackendClient
from karotte.confinement import (
    Contract,
    apply_default_limits,
    current_sandbox,
    describe_confinement,
)
from karotte.container import is_containerized
from karotte.durable_write import write_durably
from karotte.mcp_servers.resource_sampler import ResourceSampler
from karotte.protected_store import ProtectedStore
from karotte.save_artifact import save_artifact
from karotte.schemas import RunStatus
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.run_state import RunState
from karotte.schemas.transcript import (
    ErrorEvent,
    Event,
    MessageAddedEvent,
    MessageChunkEvent,
    MessageChunkResetEvent,
    MetadataEvent,
    ScoringEvent,
    StepCompletedEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskPreHookCompletedEvent,
    TaskStartedEvent,
    Transcript,
)
from karotte.subprocess import ipc_namespace_available
from karotte.text import escape_surrogates
from karotte.truncation import sanitize_metadata


class EvaluationRunner:
    def __init__(self, config: EvaluationRunConfig, task: Task):
        self.config: Final[EvaluationRunConfig] = config
        self.task: Task = task
        self.transcript: Transcript
        self.run_state: RunState
        self.tools: list[ChatCompletionToolParam]
        self.backend_client: BackendClient | None = None
        self.network_firewall: bool | None = None
        self._agent: Agent = self._make_agent()

    def _make_agent(self) -> Agent:
        match self.config.resolved_agent:
            case "builtin":
                return BuiltinAgent(self.config)
            case "external":
                assert self.config.backend_uri is not None, (
                    "backend_uri is required for training rollouts"
                )
                self.backend_client = BackendClient(self.config.backend_uri)
                return ExternalAgent(self.backend_client, self.config.run_id)
            case name:
                return get_cli_agent_type(name)(self.config)

    @property
    def allows_student_mcp_access(self) -> bool:
        """Whether the selected agent needs the student user to reach MCP."""
        return self._agent.allows_student_mcp_access

    async def run(self) -> AsyncGenerator[Event]:
        try:
            async with self._prepared_mcp_client() as mcp_client:
                try:
                    async for event in self._run(mcp_client):
                        yield event
                finally:
                    await self._agent.stop()
        except (Exception, KeyboardInterrupt) as e:
            error_event = ErrorEvent(
                exception_type=type(e).__name__,
                message=escape_surrogates(str(e)),
                traceback=escape_surrogates(traceback.format_exc()),
            )
            completed_event = TaskCompletedEvent(status="error")
            yield self._process_event(error_event)
            yield self._process_event(completed_event)
        finally:
            self._maybe_save_transcript_to_file()

    async def prepare(self) -> AsyncGenerator[Event]:
        """Set up the env like a real run, stopping just before the step loop.

        Yields the same events a run emits up to that point: tool loading,
        ``TaskStartedEvent``, ``pre_hook``, and the system-prompt message.
        Exceptions are deliberately not caught: a setup failure must surface to
        the caller so the process can exit non-zero. The transcript is built in
        memory but never saved to disk, and an existing transcript file from an
        earlier real run is left untouched.
        """
        async with self._prepared_mcp_client() as mcp_client:
            try:
                async for event in self._set_up_task(mcp_client):
                    yield event
            finally:
                await self._agent.stop()

    @asynccontextmanager
    async def _prepared_mcp_client(self):
        """Clear per-run state, configure the task's tools, connect to the MCP
        server, and register those tools."""
        ToolConfigWriter().clear()
        ProtectedStore().clear()
        self.task.configure_tools()
        tools = [t for t in self.task.tools if t not in self._agent.native_tool_names]
        async with self._connect_to_mcp_server() as mcp_client:
            # Caches register_tools' schema before it unlists itself; the client warns otherwise.
            _ = await mcp_client.list_tools()
            await mcp_client.call_tool("register_tools", {"tools": tools})
            yield mcp_client

    async def _set_up_task(
        self, mcp_client: Client[StreamableHttpTransport]
    ) -> AsyncGenerator[Event]:
        """Load tools, reset the transcript, emit TaskStartedEvent, run the
        task pre-hook, and add the system-prompt message."""
        self.tools = await self._load_tools(mcp_client)

        self.transcript = Transcript(run_id=self.config.run_id)

        effort = self.config.applied_reasoning_effort
        if self.config.reasoning_effort is not None and effort is None:
            logger.warning(
                "reasoning_effort={requested!r} is not applied for {model} on the {agent} agent; using the provider default.",
                requested=self.config.reasoning_effort,
                model=self.config.model,
                agent=self.config.resolved_agent,
            )

        yield self._process_event(
            TaskStartedEvent(
                run_id=self.config.run_id,
                task_id=self.task.id,
                model=self.config.model,
                reasoning_effort=effort,
                use_hints=self.config.use_hints,
                n_steps=len(self.task.steps)  # pyright: ignore[reportArgumentType]
                if hasattr(self.task.steps, "__len__")
                else -1,
            )
        )
        contracts = apply_default_limits(self.task.required_hardware)
        _log_confinement(contracts, self.network_firewall)
        metadata = self.task.pre_hook()
        yield self._process_event(TaskPreHookCompletedEvent(metadata=metadata))

        # CLI agents drive their own binary, which sends its own system prompt;
        # karotte's would never reach the model, so it is not emitted.
        system_prompt = (
            None if isinstance(self._agent, CliAgent) else self.task.system_prompt
        )
        if system_prompt:
            system_message = Message(
                content=system_prompt,
                role="system",
                tool_calls=None,
                reasoning_content=None,
                tool_call_id=None,
            )
            yield self._process_event(MessageAddedEvent(message=system_message))

        await self._agent.start(
            RunContext(
                config=self.config,
                mcp_client=mcp_client,
                tools=self.tools,
                transcript=self.transcript,
            )
        )

    async def _run(
        self, mcp_client: Client[StreamableHttpTransport]
    ) -> AsyncGenerator[Event]:
        self._delete_transcript()
        async for event in self._set_up_task(mcp_client):
            yield event

        run_status: RunStatus = "passed"

        for i_step, step in enumerate(self.task.steps):
            yield self._process_event(StepStartedEvent(step=i_step))
            event = None
            async for event in self._execute_step(step, i_step):
                yield event

            assert isinstance(event, ScoringEvent)
            yield self._process_event(StepCompletedEvent(step=i_step))

            if not event.scoring.continue_task:
                run_status = "failed"
                break

        yield self._process_event(TaskCompletedEvent(status=run_status))

    async def _execute_step(self, step: Step, step_index: int) -> AsyncGenerator[Event]:
        content = self.config.resolve_step_instructions(step.instructions, step_index)
        time_limit = self.config.resolve_step_time_limit(step_index)
        context_window_limit = self.config.resolve_step_context_window_limit(step_index)

        # The agent produces the turn events raw; the runner applies transcript /
        # run-state bookkeeping to everything except the transient streaming chunks.
        async for event in self._agent.run_step(
            content,
            time_limit_seconds=time_limit,
            on_time_limit=self.config.on_step_time_limit,
            context_window_limit=context_window_limit,
            on_context_window_limit=self.config.on_step_context_window_limit,
        ):
            if isinstance(event, (MessageChunkEvent, MessageChunkResetEvent)):
                yield event
            else:
                yield self._process_event(event)

        # Persist any extra artifacts named in extra_config BEFORE the env's
        # pre_scoring_hook and before scoring. Both can remove student files:
        # pre_scoring_hook often scrubs the student workdir down to the official
        # answer file, and judges may wipe files during evaluate(). Saving first
        # captures the file whenever it exists; runs after every step.
        _save_extra_artifacts(self.config)

        if self.config.mcp_server_config.profile_tool_calls:
            async with ResourceSampler() as sampler:
                scoring = await asyncio.to_thread(
                    step.score, self.transcript, step_index
                )
            resource_metrics = sampler.metrics
        else:
            scoring = await asyncio.to_thread(step.score, self.transcript, step_index)
            resource_metrics = None
        yield self._process_event(
            ScoringEvent(scoring=scoring, resource_metrics=resource_metrics)
        )

        step.post_hook()

    @asynccontextmanager
    async def _connect_to_mcp_server(self):
        transport = StreamableHttpTransport(self.config.mcp_server_config.client_url)
        async with Client[StreamableHttpTransport](transport) as client:
            yield client

    async def _load_tools(
        self, client: Client[StreamableHttpTransport]
    ) -> list[ChatCompletionToolParam]:
        """Retrieves the tools available for this task using the LiteLLM/OpenAI tool schema."""
        return [
            self._convert_mcp_tool_to_litellm_tool(tool)
            for tool in await client.list_tools()
        ]

    def _convert_mcp_tool_to_litellm_tool(self, tool: Tool) -> ChatCompletionToolParam:
        """The evaluation runner needs to use the MCP tool schema when talking to the MCP server
        but needs to expose tools to the model via the LiteLLM/OpenAI schema."""
        assert tool.description is not None
        return ChatCompletionToolParam(
            type="function",
            function=ChatCompletionToolParamFunctionChunk(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            ),
        )

    def _delete_transcript(self):
        if self.config.transcript_file:
            try:
                Path(self.config.transcript_file).unlink()
            except FileNotFoundError:
                pass

    def _process_event(self, event: Event):
        if isinstance(event, ScoringEvent):
            sanitize_metadata(event.scoring.metadata)
        elif isinstance(event, MetadataEvent):
            sanitize_metadata(event.metadata)
        if hasattr(self, "transcript"):
            self.transcript.events.append(event)
        if hasattr(self, "run_state"):
            self.run_state.apply(event)
        elif isinstance(event, TaskStartedEvent):
            self.run_state = RunState(event)

        return event

    def _maybe_save_transcript_to_file(self):
        if hasattr(self, "transcript") and self.config.transcript_file:
            write_durably(
                Path(self.config.transcript_file),
                self.transcript.model_dump_json(indent=2),
            )
            _ = print(f"\n📁 Transcript saved to: {self.config.transcript_file}")


def _log_confinement(contracts: dict[str, Contract], firewall: bool | None) -> None:
    """Log the confinement in force, as a warning when any of it is off."""
    if not is_containerized():
        logger.warning("Confinement: none (not in a container)")
        return
    line, degraded = describe_confinement(
        contracts,
        network_firewall=firewall,
        ipc_namespace=ipc_namespace_available(),
        sandbox=current_sandbox(),
    )
    (logger.warning if degraded else logger.info)(line)


def _save_extra_artifacts(config: EvaluationRunConfig) -> None:
    """Upload files named in `extra_config["extra_artifact_paths"]` as artifacts.

    Library-level hook so any environment can capture an extra output file (e.g.
    a model-written reasoning summary) for retrieval via the backend, without adding a
    `save_artifact` call to its own hooks. Useful when the file's existence is
    itself driven from `extra_config` (e.g. an injected system-prompt line
    telling the model to write it), so the env code never knows about the path.

    Values are absolute container paths -- the launch config knows where the
    model was told to write. Missing paths are skipped with a warning rather than
    failing the run (the model may not have written the file).
    """
    for p in config.extra_artifact_paths:
        path = Path(p)
        if not path.exists():
            logger.warning("extra_artifact_paths: {} does not exist, skipping", path)
            continue
        save_artifact(config, path)
