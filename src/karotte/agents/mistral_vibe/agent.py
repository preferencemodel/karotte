"""Mistral Vibe CLI agent.

Runs the ``vibe`` binary once per step in headless mode, pointed at karotte's
MCP server. Vibe is a Python tool (``mistral-vibe`` on PyPI), so it installs as
an isolated ``uv tool`` into the shared agents dir at build time.

Invokes the binary, captures its NDJSON output as an artifact, and yields the
normalized transcript events derived from it by the adapter.
"""

import asyncio
import os
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import ClassVar, Literal, final

from loguru import logger

from karotte.agents.agent import (
    CliAgentExitedError,
    RunContext,
    StepTimeLimitReachedError,
)
from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    AGENTS_DIR,
    CliAgent,
    kill_process_tree,
    register_cli_agent,
)
from karotte.agents.mistral_vibe.adapter import parse_stream
from karotte.agents.models import resolve_model
from karotte.save_artifact import save_artifact
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event, MessageAddedEvent
from karotte.subprocess import demoted_uid_gid, make_preexec, scrub_harness_secrets

_VIBE_BIN = f"{AGENTS_BIN_DIR}/vibe"


@final
@register_cli_agent
class MistralVibeAgent(CliAgent):
    name: ClassVar[str] = "mistral-vibe"
    version: ClassVar[str] = "2.19.0"
    provider_url: ClassVar[str] = "https://api.mistral.ai"
    # Vibe ships its own shell and file read/edit tools; use those rather than
    # registering karotte's equivalents over MCP.
    native_tool_names: ClassVar[frozenset[str]] = frozenset(
        {"bash", "view_lines_in_file", "view_image_file", "replace_in_file"}
    )

    @classmethod
    def install(cls) -> list[str]:
        # `uv tool install` puts vibe in its own venv; the bin symlink goes to
        # the shared agents dir, and a world-read+execute chmod lets the student
        # run it. UV_TOOL_DIR keeps the tool venv out of the caller's HOME.
        return [
            f"mkdir -p {AGENTS_BIN_DIR}",
            (
                f"UV_TOOL_BIN_DIR={AGENTS_BIN_DIR} UV_TOOL_DIR={AGENTS_DIR}/tools "
                f"uv tool install --python 3.12 mistral-vibe=={cls.version}"
            ),
            f"chmod -R a+rX {AGENTS_DIR}",
        ]

    def __init__(self, config: EvaluationRunConfig) -> None:
        super().__init__(config)
        self._step_index: int = 0
        self._vibe_home: Path = Path(os.environ.get("KAROTTE_WORKDIR", ".")) / ".vibe"

    async def start(self, ctx: RunContext) -> None:
        await super().start(ctx)
        self._write_config()

    def _write_config(self) -> None:
        """Write Vibe's config (model, provider, MCP server) into a
        student-writable ``VIBE_HOME`` so the tool can also drop its own session
        state there.

        Model traffic goes to :attr:`model_url`: the proxy when
        ``KAROTTE_PROXY_URL`` is set, otherwise the forwarder to Mistral's API.
        Both expose an OpenAI-compatible route, so the provider is declared
        ``generic``/``openai``."""
        resolved = resolve_model(self._config.model, None)
        key_env = self._key_env()
        provider_name = "proxy" if os.environ.get("KAROTTE_PROXY_URL") else "mistral"
        api_base = f"{self.model_url}/v1"

        self._vibe_home.mkdir(parents=True, exist_ok=True)
        config = (
            f'active_model = "{resolved.model}"\n\n'
            "[session_logging]\n"
            "enabled = false\n\n"
            "[[providers]]\n"
            f'name = "{provider_name}"\n'
            f'api_base = "{api_base}"\n'
            f'api_key_env_var = "{key_env}"\n'
            'api_style = "openai"\n'
            'backend = "generic"\n\n'
            "[[models]]\n"
            f'name = "{resolved.model}"\n'
            f'provider = "{provider_name}"\n'
            f'alias = "{resolved.model}"\n\n'
            "[[mcp_servers]]\n"
            'name = "karotte"\n'
            'transport = "streamable-http"\n'
            f'url = "{self.mcp_url}"\n'
        )
        (self._vibe_home / "config.toml").write_text(config)

        uid = demoted_uid_gid()
        if uid is not None:
            # Vibe runs as the student and writes its own state under VIBE_HOME.
            for p in (self._vibe_home, self._vibe_home / "config.toml"):
                os.chown(p, uid, uid)

    def _env(self) -> dict[str, str]:
        # Runs as the student: start from what the student may see, not from
        # everything the harness was handed (see scrub_harness_secrets). The
        # agent's own model credentials are added below, on purpose.
        env = scrub_harness_secrets(os.environ)
        # The process runs as the student; point HOME at the student workdir so
        # the tool's own dotfiles/state don't land in (unreadable) /root.
        workdir = os.environ.get("KAROTTE_WORKDIR", str(Path.home()))
        env["HOME"] = workdir
        env["USER"] = "student"
        env["LOGNAME"] = "student"
        env["VIBE_HOME"] = str(self._vibe_home)
        env.pop("PYTHONSAFEPATH", None)

        # The Containerfile sets VIRTUAL_ENV and puts its bin on PATH, but
        # sanitize_paths() strips the student-writable venv bin from the (root)
        # runner's PATH before we launch (VIRTUAL_ENV itself survives). The agent
        # runs as the student, for whom that venv is the task's Python, so put
        # the bin back so its own shell/tools resolve it, mirroring the bash tool.
        path_parts = [AGENTS_BIN_DIR]
        venv = env.get("VIRTUAL_ENV")
        if venv and (Path(venv) / "pyvenv.cfg").is_file():
            path_parts.insert(0, f"{venv}/bin")
        env["PATH"] = ":".join([*path_parts, env.get("PATH", "")])

        if (api_key := self.model_api_key) is not None:
            env[self._key_env()] = api_key
        return env

    def _key_env(self) -> str:
        return resolve_model(self._config.model, None).key_env or "MISTRAL_API_KEY"

    def _argv(self, instructions: str) -> list[str]:
        # Passed to exec (no shell), so the instruction is a single safe argv
        # element -- no shell-escaping layers. Vibe has no prompt-file flag.
        argv = [
            _VIBE_BIN,
            "-p",
            instructions,
            "--output",
            "streaming",
            "--auto-approve",
            "--trust",
        ]
        if self._config.turn_limit is not None:
            argv += ["--max-turns", str(self._config.turn_limit)]
        return argv

    async def run_step(
        self,
        instructions: str,
        time_limit_seconds: float | None = None,
        on_time_limit: Literal["error", "score"] = "error",
        context_window_limit: int | None = None,
        on_context_window_limit: Literal["error", "score"] = "error",
    ) -> AsyncGenerator[Event]:
        """Run one Vibe invocation, enforcing `time_limit_seconds` by killing the
        process (and its children) if it overruns.

        Vibe has no wall-clock flag of its own, so the limit is imposed here:
        `error` re-raises after emitting whatever partial output was captured,
        `score` keeps that partial output and lets the step be scored. Unlike
        the builtin loop there is no per-turn counter -- Vibe runs the whole
        step in one opaque subprocess.

        `context_window_limit` is accepted for protocol compatibility but not
        enforced: Vibe only surfaces (estimated) usage after the step finishes,
        so there is no way to observe the context window between turns.
        """
        yield MessageAddedEvent(message=Message(role="user", content=instructions))

        log_path = self._vibe_home / f"step_{self._step_index}.ndjson"
        self._step_index += 1

        proc = await asyncio.create_subprocess_exec(
            *self._argv(instructions),
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
        # Drain both pipes concurrently so a chatty child can't deadlock on a
        # full pipe while we wait, and so we keep whatever it wrote before a kill.
        stdout_task = asyncio.ensure_future(proc.stdout.read())
        stderr_task = asyncio.ensure_future(proc.stderr.read())

        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), time_limit_seconds)
        except TimeoutError:
            timed_out = True
            kill_process_tree(proc)
            await proc.wait()

        stdout_b = await stdout_task
        stderr_b = await stderr_task
        stdout = stdout_b.decode("utf-8", errors="replace")
        lines = stdout.splitlines()

        log_path.write_text(stdout)
        save_artifact(self._config, log_path)

        if timed_out:
            logger.warning(
                "vibe hit the {}s time limit on step {}",
                time_limit_seconds,
                self._step_index - 1,
            )

        for event in _transcript_events(lines, instructions, self._config.model):
            yield event

        if timed_out and on_time_limit == "error":
            msg = f"Time limit of {time_limit_seconds}s reached."
            raise StepTimeLimitReachedError(msg)
        if not timed_out and proc.returncode != 0:
            stderr = stderr_b.decode("utf-8", errors="replace")[-2000:]
            msg = f"vibe exited {proc.returncode} on step {self._step_index - 1}: {stderr}"
            raise CliAgentExitedError(msg)


def _transcript_events(
    lines: list[str], instructions: str, model: str
) -> Iterator[Event]:
    """Normalized events for one step's NDJSON log.

    Vibe re-sends its own system prompt with every invocation and echoes the
    step instructions as its first user message; both are dropped here to avoid
    duplicating content the transcript already carries. The full log stays
    available as the artifact.
    """
    seen_user = False
    for event in parse_stream(lines, model=model):
        if isinstance(event, MessageAddedEvent):
            if event.message.role == "system":
                continue
            if not seen_user and event.message.role == "user":
                seen_user = True
                if event.message.content == instructions:
                    continue
        yield event
