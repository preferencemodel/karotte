"""Base class and registry for agents that drive an external CLI coding tool.

A CLI agent runs the tool's binary in-container as the student user, once per
step; the tool talks to the MCP server over HTTP itself (hence
``allows_student_mcp_access``). Each agent says how its tool is installed at
``karotte build`` time (:meth:`install`, version-pinned via :attr:`version`),
configured and invoked; the base class runs it.
"""

import abc
import asyncio
import os
import shutil
import signal
import tempfile
from collections.abc import AsyncGenerator, Callable, Iterable
from pathlib import Path
from typing import ClassVar, Literal

from loguru import logger

from karotte.agents.agent import (
    CliAgentExitedError,
    RunContext,
    StepTimeLimitReachedError,
)
from karotte.model_forwarder import ModelForwarder
from karotte.providers import PROXY_PLACEHOLDER_KEY
from karotte.save_artifact import save_artifact
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event, MessageAddedEvent
from karotte.subprocess import demoted_uid_gid, make_preexec, scrub_harness_secrets

# Root-owned, world-readable+executable directory the build installs CLI agents
# into, so the student user can run them but not tamper with them.
AGENTS_DIR = "/opt/agents"
AGENTS_BIN_DIR = f"{AGENTS_DIR}/bin"

# Chunks, not lines: one line can exceed StreamReader's line limit.
_READ_CHUNK_BYTES = 1 << 16
_DRAIN_TIMEOUT_SECONDS = 10.0
_EXIT_POLL_SECONDS = 0.05


class CliAgent(abc.ABC):
    name: ClassVar[str]
    """The value of ``EvaluationRunConfig.agent`` that selects this agent."""

    version: ClassVar[str]
    """The tool version this karotte release pins and installs."""

    allows_student_mcp_access: ClassVar[bool] = True
    native_tool_names: ClassVar[frozenset[str]] = frozenset()

    supports_temperature: ClassVar[bool] = False
    """Whether the tool sends the run config's ``temperature``."""

    applies_reasoning_effort: ClassVar[bool] = False
    """Whether the tool sends the run config's ``reasoning_effort``."""

    provider_url: ClassVar[str]
    """The provider API root, without ``/v1``, the tool calls when there is no
    proxy. It sits outside the student firewall, so the tool reaches it through
    a :class:`ModelForwarder`."""

    binary: ClassVar[str]

    home_dir: ClassVar[str]
    """Directory under the student workdir for the tool's config and state."""

    home_env: ClassVar[str]
    """Env var that points the tool at :attr:`home_dir`."""

    uses_prompt_file: ClassVar[bool] = False
    """Whether :meth:`_argv` reads the step's instructions from a file."""

    def __init__(self, config: EvaluationRunConfig) -> None:
        self._config: EvaluationRunConfig = config
        self._ctx: RunContext
        self._forwarder: ModelForwarder | None = None
        self._step_index: int = 0
        self._home: Path = Path(os.environ.get("KAROTTE_WORKDIR", ".")) / self.home_dir
        self._step_dir: Path | None = None

    @classmethod
    @abc.abstractmethod
    def install(cls) -> list[str]:
        """Shell commands, run as root at ``karotte build`` time, that install
        the pinned tool so the student user can execute it from
        :data:`AGENTS_BIN_DIR`."""

    @abc.abstractmethod
    def _config_files(self) -> dict[str, str]:
        """Contents of the tool's config files, by path relative to its home."""

    @abc.abstractmethod
    def _key_env(self) -> str:
        """Env var the tool reads its model API key from."""

    def _extra_env(self) -> dict[str, str]:
        return {}

    @abc.abstractmethod
    def _argv(self, instructions: str, prompt_file: Path | None) -> list[str]:
        """The command for one step; ``prompt_file`` holds ``instructions``
        when :attr:`uses_prompt_file` is set."""

    @abc.abstractmethod
    def _line_parser(self, instructions: str) -> Callable[[str, int], Iterable[Event]]:
        """Turns line ``i`` of one step's stdout into transcript events."""

    async def start(self, ctx: RunContext) -> None:
        self._ctx = ctx
        if not os.environ.get("KAROTTE_PROXY_URL") and self._config.model_api_key:
            self._forwarder = ModelForwarder(
                self.provider_url, self._config.model_api_key
            )
            _ = await self._forwarder.start()
        self._write_config()

    async def stop(self) -> None:
        if self._forwarder is not None:
            await self._forwarder.stop()
            self._forwarder = None
        if self._step_dir is not None:
            shutil.rmtree(self._step_dir, ignore_errors=True)
            self._step_dir = None

    def _step_files(self) -> Path:
        """Root's own directory for each step's prompt and log: the student can
        read a file in it by name, but can't plant links where root writes."""
        if self._step_dir is None:
            self._step_dir = Path(tempfile.mkdtemp(prefix=f"karotte-{self.name}-"))
            self._step_dir.chmod(0o711)
        return self._step_dir

    @property
    def model_url(self) -> str:
        """Where the tool sends model calls, without ``/v1``: the proxy, the
        forwarder to :attr:`provider_url`, or with no key to forward,
        :attr:`provider_url` itself."""
        if proxy := os.environ.get("KAROTTE_PROXY_URL"):
            return proxy.rstrip("/")
        if self._forwarder is not None:
            return self._forwarder.url
        return self.provider_url

    @property
    def model_api_key(self) -> str | None:
        """The key the tool sends. The forwarder adds the real one, so the tool
        gets a placeholder; a proxy that needs no key gets one too."""
        if self._forwarder is not None:
            return PROXY_PLACEHOLDER_KEY
        if self._config.model_api_key is None and os.environ.get("KAROTTE_PROXY_URL"):
            return PROXY_PLACEHOLDER_KEY
        return self._config.model_api_key

    @property
    def mcp_url(self) -> str:
        """Endpoint the tool's own MCP client connects to. The server binds
        0.0.0.0; a local client reaches it on loopback."""
        return self._config.mcp_server_config.client_url

    def _write_config(self) -> None:
        self._home.mkdir(parents=True, exist_ok=True)
        written = [self._home]
        for name, content in self._config_files().items():
            path = self._home / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            written.append(path)
        if (uid := demoted_uid_gid()) is not None:
            for path in written:
                os.chown(path, uid, uid)

    def _env(self) -> dict[str, str]:
        # Only what the student may see; the model key is added on purpose below.
        env = scrub_harness_secrets(os.environ)
        env["HOME"] = os.environ.get("KAROTTE_WORKDIR", str(Path.home()))
        env["USER"] = "student"
        env["LOGNAME"] = "student"
        env[self.home_env] = str(self._home)
        env.pop("PYTHONSAFEPATH", None)

        # sanitize_paths() strips the task venv's bin from PATH; the student needs it back.
        path_parts = [AGENTS_BIN_DIR]
        venv = env.get("VIRTUAL_ENV")
        if venv and (Path(venv) / "pyvenv.cfg").is_file():
            path_parts.insert(0, f"{venv}/bin")
        env["PATH"] = ":".join([*path_parts, env.get("PATH", "")])

        env.update(self._extra_env())
        if (api_key := self.model_api_key) is not None:
            env[self._key_env()] = api_key
        return env

    async def run_step(
        self,
        instructions: str,
        time_limit_seconds: float | None = None,
        on_time_limit: Literal["error", "score"] = "error",
        context_window_limit: int | None = None,  # pyright: ignore[reportUnusedParameter]
        on_context_window_limit: Literal["error", "score"] = "error",  # pyright: ignore[reportUnusedParameter]
    ) -> AsyncGenerator[Event]:
        """Run the tool once, yielding transcript events as it writes each line.

        Past `time_limit_seconds` the tool and its children are killed: `error`
        raises after yielding the output so far, `score` keeps it."""
        yield MessageAddedEvent(message=Message(role="user", content=instructions))

        step = self._step_index
        tool = Path(self.binary).name
        steps = self._step_files()
        prompt_path = None
        if self.uses_prompt_file:
            prompt_path = steps / f"step_{step}.prompt"
            prompt_path.write_text(instructions)
            prompt_path.chmod(0o644)
        argv = self._argv(instructions, prompt_path)
        parse = self._line_parser(instructions)
        self._step_index += 1
        log_path = steps / f"step_{step}.ndjson"

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env(),
            cwd=os.environ.get("KAROTTE_WORKDIR"),
            # The two pipes opened just above are handed to the demoted agent
            # along with the privilege drop, so it can reopen them by path.
            preexec_fn=make_preexec(1, 2),
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        stderr_task = asyncio.ensure_future(proc.stderr.read())

        loop = asyncio.get_running_loop()
        deadline = (
            None if time_limit_seconds is None else loop.time() + time_limit_seconds
        )

        def remaining() -> float | None:
            return None if deadline is None else max(0.0, deadline - loop.time())

        timed_out = False
        line_index = 0
        pending = b""
        try:
            with log_path.open("wb") as log:
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            proc.stdout.read(_READ_CHUNK_BYTES), remaining()
                        )
                    except TimeoutError:
                        timed_out = True
                        kill_process_tree(proc)
                        chunk = await _read_rest(proc.stdout)
                    log.write(chunk)
                    log.flush()
                    pending += chunk
                    *lines, pending = pending.split(b"\n")
                    if timed_out or not chunk:
                        lines.append(pending)
                    for line in lines:
                        for event in parse(
                            line.decode("utf-8", errors="replace"), line_index
                        ):
                            yield event
                        line_index += 1
                    if timed_out or not chunk:
                        break
            if not timed_out:
                try:
                    _ = await asyncio.wait_for(_exited(proc), remaining())
                except TimeoutError:
                    # wait_for times out on a past deadline even once the tool has exited.
                    if proc.returncode is None:
                        timed_out = True
                        kill_process_tree(proc)
                        _ = await _exited(proc)
        finally:
            # Also reached when the consumer stops early (e.g. the run is cancelled).
            if proc.returncode is None:
                kill_process_tree(proc)
                _ = await _exited(proc)
            try:
                stderr_b = await asyncio.wait_for(stderr_task, _DRAIN_TIMEOUT_SECONDS)
            except TimeoutError:
                stderr_b = b""
            save_artifact(self._config, log_path)

        if timed_out:
            logger.warning(
                "{} hit the {}s time limit on step {}", tool, time_limit_seconds, step
            )
            if on_time_limit == "error":
                msg = f"Time limit of {time_limit_seconds}s reached."
                raise StepTimeLimitReachedError(msg)
        elif proc.returncode != 0:
            stderr = stderr_b.decode("utf-8", errors="replace")[-2000:]
            msg = f"{tool} exited {proc.returncode} on step {step}: {stderr}"
            raise CliAgentExitedError(msg)


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


async def _exited(proc: asyncio.subprocess.Process) -> int:
    """The exit code, as soon as the process exits. ``proc.wait()`` can also
    wait for a child that outlives it to close an inherited pipe."""
    while proc.returncode is None:
        await asyncio.sleep(_EXIT_POLL_SECONDS)
    return proc.returncode


async def _read_rest(stream: asyncio.StreamReader) -> bytes:
    """What a killed process left in its stdout pipe. Bounded, in case a
    grandchild that escaped the kill still holds the pipe open."""
    try:
        return await asyncio.wait_for(stream.read(), _DRAIN_TIMEOUT_SECONDS)
    except TimeoutError:
        return b""


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
