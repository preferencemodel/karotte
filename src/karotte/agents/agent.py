from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, final

from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from litellm import ChatCompletionToolParam

from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event, Transcript


class TurnLimitReachedError(Exception):
    """Raised when the evaluation run exceeds the configured turn limit."""


class StepTimeLimitReachedError(Exception):
    """Raised when a step exceeds its configured wall-clock time limit."""


class CliAgentExitedError(Exception):
    """Raised when a CLI agent's process exits non-zero, so the step ends as an
    error instead of being scored on whatever the agent left behind."""


class StepContextWindowLimitReachedError(Exception):
    """Raised when a step exceeds its configured context-window token limit."""


class EmptyTurnLimitReachedError(Exception):
    """Raised when the model produced neither text nor tool calls too many turns
    in a row, so nudging it to continue has stopped working."""


@final
@dataclass
class RunContext:
    """What an agent needs to solve steps within a run.

    The runner owns the transcript and applies event bookkeeping; the agent
    reads `transcript.messages` for conversation history and executes tool calls
    against `mcp_client`. One `RunContext` per run, passed to `Agent.start`.
    """

    config: EvaluationRunConfig
    mcp_client: Client[StreamableHttpTransport]
    tools: list[ChatCompletionToolParam]
    transcript: Transcript


class Agent(Protocol):
    """Produces a run's transcript, one step at a time.

    The runner keeps the task lifecycle (hooks, judging, artifacts, event
    bookkeeping); the agent decides how messages and tool calls get produced.
    An agent instance is stateful across steps within a run.
    """

    allows_student_mcp_access: ClassVar[bool]
    """Whether the student user may reach the MCP port.

    True only for agents that run in-container as the student and make native
    MCP tool calls (CLI agents). Builtin/external agents keep the port blocked:
    karotte executes their tool calls itself, so student access would only let
    task code bypass transcript recording.
    """

    native_tool_names: ClassVar[frozenset[str]]
    """Tools this agent supplies itself, and which the runner therefore
    omits when registering the task's tools. A CLI agent with its own bash/file
    tools lists them here so it uses its native ones instead of duplicates over
    MCP. Empty for agents that rely entirely on MCP tools.
    """

    async def start(self, ctx: RunContext) -> None:
        """Verify/configure the agent for the run."""
        ...

    def run_step(
        self,
        instructions: str,
        time_limit_seconds: float | None = None,
        on_time_limit: Literal["error", "score"] = "error",
        context_window_limit: int | None = None,
        on_context_window_limit: Literal["error", "score"] = "error",
    ) -> AsyncGenerator[Event]:
        """Solve one step, yielding raw transcript events.

        `time_limit_seconds` bounds the step's wall-clock duration; when it
        elapses the step either aborts (`on_time_limit="error"`) or ends so it
        can be scored (`on_time_limit="score"`). `None` disables the limit.

        `context_window_limit` bounds the step's context-window length (the input
        tokens of the most recent turn); when it is exceeded the step either
        aborts (`on_context_window_limit="error"`) or ends so it can be scored
        (`on_context_window_limit="score"`). `None` disables the limit.
        """
        ...

    async def stop(self) -> None:
        """Release any resources held for the run."""
        ...
