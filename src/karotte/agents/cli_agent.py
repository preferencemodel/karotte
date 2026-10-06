"""Base class and registry for agents that drive an external CLI coding tool.

A CLI agent runs the tool's binary in-container as the student user; the tool
talks to the MCP server over HTTP itself (hence ``allows_student_mcp_access``).
Each agent is the single source of truth for how its tool is installed at
``karotte build`` time (:meth:`install`, version-pinned via :attr:`version`) and
how it is invoked per step (:meth:`run_step`).
"""

import abc
import asyncio
import os
import signal
from collections.abc import AsyncGenerator
from typing import ClassVar, Literal

from karotte.agents.agent import RunContext
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event

# Root-owned, world-readable+executable directory the build installs CLI agents
# into, so the student user can run them but not tamper with them.
AGENTS_DIR = "/opt/agents"
AGENTS_BIN_DIR = f"{AGENTS_DIR}/bin"


class CliAgent(abc.ABC):
    name: ClassVar[str]
    """The value of ``EvaluationRunConfig.agent`` that selects this agent."""

    version: ClassVar[str]
    """The tool version this karotte release pins and installs."""

    allows_student_mcp_access: ClassVar[bool] = True
    native_tool_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: EvaluationRunConfig) -> None:
        self._config: EvaluationRunConfig = config
        self._ctx: RunContext

    @classmethod
    @abc.abstractmethod
    def install(cls) -> list[str]:
        """Shell commands, run as root at ``karotte build`` time, that install
        the pinned tool so the student user can execute it from
        :data:`AGENTS_BIN_DIR`."""

    async def start(self, ctx: RunContext) -> None:
        self._ctx = ctx

    @abc.abstractmethod
    def run_step(
        self,
        instructions: str,
        time_limit_seconds: float | None = None,
        on_time_limit: Literal["error", "score"] = "error",
        context_window_limit: int | None = None,
        on_context_window_limit: Literal["error", "score"] = "error",
    ) -> AsyncGenerator[Event]:
        """Solve one step by invoking the CLI, yielding raw transcript events.

        The time-limit and context-window parameters are part of the `Agent`
        protocol; a CLI agent runs the tool in its own process, so it cannot
        observe turns. The context-window limit is never enforced here (usage is
        only known after the run), though a subclass may still enforce the time
        limit by killing the process.
        """
        ...

    async def stop(self) -> None:
        return None

    @property
    def mcp_url(self) -> str:
        """Endpoint the tool's own MCP client connects to. The server binds
        0.0.0.0; a local client reaches it on loopback."""
        return self._config.mcp_server_config.client_url


def kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL a CLI agent process and its children.

    ``make_preexec`` puts the child in its own session, so its pid is a process
    group id; killing the group takes down any tools it spawned. Falls back to
    killing just the process when there is no group (e.g. no ``setsid`` outside
    a container) or it has already exited.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


_CLI_AGENTS: dict[str, type[CliAgent]] = {}


def register_cli_agent[C: CliAgent](cls: type[C]) -> type[C]:
    _CLI_AGENTS[cls.name] = cls
    return cls


def cli_agent_types() -> dict[str, type[CliAgent]]:
    """All registered CLI agents, keyed by ``name``."""
    return dict(_CLI_AGENTS)


def get_cli_agent_type(name: str) -> type[CliAgent]:
    try:
        return _CLI_AGENTS[name]
    except KeyError:
        known = ", ".join(sorted(_CLI_AGENTS)) or "(none)"
        raise ValueError(f"Unknown CLI agent {name!r}. Known agents: {known}.")
