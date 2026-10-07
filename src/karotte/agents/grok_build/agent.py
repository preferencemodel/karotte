"""Grok Build CLI agent.

Runs xAI's ``grok`` binary once per step in headless mode, pointed at
karotte's MCP server. Grok ships as a single static binary, so the build
downloads the pinned release for the container's architecture and checks it
against a pinned SHA-256 (xAI publishes no checksums).

Steps share one Grok session (created on the first step, resumed after), so a
later step sees the conversation of the earlier ones, as with the builtin loop.
"""

import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import ClassVar, Literal, final

from loguru import logger

from karotte.agents.agent import RunContext, StepTimeLimitReachedError
from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    CliAgent,
    kill_process_tree,
    register_cli_agent,
)
from karotte.agents.grok_build.adapter import parse_stream
from karotte.agents.models import resolve_model
from karotte.save_artifact import save_artifact
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event, MessageAddedEvent
from karotte.subprocess import demoted_uid_gid, make_preexec, scrub_harness_secrets

_GROK_BIN = f"{AGENTS_BIN_DIR}/grok"
_RELEASE_URL = "https://x.ai/cli/grok-{version}-linux-{arch}.gz"

# SHA-256 of the decompressed linux binary, per `uname -m` spelling xAI uses.
_SHA256: dict[str, str] = {
    "x86_64": "41626a53292324140b92556b9d42ff5542e3dcd04aff85eafb8689dd4adb44fc",
    "aarch64": "45b0943e736f00a249b9cf02af2be9e0749d97c09a6f55cfcf3029a1a836f23e",
}

# Grok tools that need xAI's backend or a human, which a task run has neither of.
_DISALLOWED_TOOLS = ("send_feedback", "image_edit", "ask_user_question")


@final
@register_cli_agent
class GrokBuildAgent(CliAgent):
    name: ClassVar[str] = "grok-build"
    version: ClassVar[str] = "1.0.46"
    provider_url: ClassVar[str] = "https://api.x.ai"
    # Grok ships its own shell and file read/edit tools (run_terminal_command,
    # read_file, search_replace); use those rather than karotte's over MCP.
    native_tool_names: ClassVar[frozenset[str]] = frozenset(
        {"bash", "view_lines_in_file", "replace_in_file"}
    )

    @classmethod
    def install(cls) -> list[str]:
        url = _RELEASE_URL.format(version=cls.version, arch="$arch")
        checks = " ".join(f"{arch}) sha={sha} ;;" for arch, sha in _SHA256.items())
        return [
            f"mkdir -p {AGENTS_BIN_DIR}",
            (
                "set -e; "
                'arch="$(uname -m)"; '
                'case "$arch" in amd64) arch=x86_64 ;; arm64) arch=aarch64 ;; esac; '
                f'case "$arch" in {checks} '
                '*) echo "grok-build: unsupported arch $arch" >&2; exit 1 ;; esac; '
                f'curl -fsSL "{url}" | gunzip > {_GROK_BIN}.tmp; '
                f'echo "$sha  {_GROK_BIN}.tmp" | sha256sum -c -; '
                f"mv {_GROK_BIN}.tmp {_GROK_BIN}"
            ),
            f"chmod 755 {_GROK_BIN}",
        ]

    def __init__(self, config: EvaluationRunConfig) -> None:
        super().__init__(config)
        self._step_index: int = 0
        self._session_id: str = str(uuid.uuid4())
        self._grok_home: Path = (
            Path(os.environ.get("KAROTTE_WORKDIR", ".")) / ".grok_home"
        )

    async def start(self, ctx: RunContext) -> None:
        await super().start(ctx)
        self._write_config()

    def _write_config(self) -> None:
        """Write Grok's config (model endpoint, MCP server, features) into a
        student-writable ``GROK_HOME``, where the tool also keeps its sessions.

        Model traffic goes through the proxy's OpenAI-compatible route when
        ``KAROTTE_PROXY_URL`` is set, so any provider works; without a proxy
        it goes through the forwarder to xAI, which only serves xAI models. Features that
        call xAI's backend on their own (telemetry, remote model catalog,
        backend web search, media generation) are turned off, and session
        titles use the run's model rather than Grok's default."""
        resolved = resolve_model(self._config.model, None)
        if not os.environ.get("KAROTTE_PROXY_URL") and resolved.provider != "xai":
            msg = (
                f"grok-build can only reach xAI models without a proxy; got "
                f"{self._config.model!r}. Set KAROTTE_PROXY_URL to use other providers."
            )
            raise ValueError(msg)

        model = _toml(resolved.model)
        config = (
            "[cli]\n"
            "auto_update = false\n\n"
            "[features]\n"
            'telemetry = "off"\n'
            "remote_fetch = false\n"
            "managed_config = false\n"
            "campaigns = false\n"
            "backend_tools = false\n"
            "feedback = false\n"
            "image_gen = false\n"
            "video_gen = false\n"
            "voice_mode = false\n"
            "turn_summary = false\n"
            "session_recap = false\n"
            "title_refresh = false\n"
            "non_git_warning = false\n\n"
            "[models]\n"
            f"session_summary = {model}\n\n"
            f"[model.{model}]\n"
            f"model = {model}\n"
            f"base_url = {_toml(f'{self.model_url}/v1')}\n"
            f"env_key = {_toml(self._key_env())}\n"
            'api_backend = "chat_completions"\n\n'
            "[mcp_servers.karotte]\n"
            f"url = {_toml(self.mcp_url)}\n"
        )
        self._grok_home.mkdir(parents=True, exist_ok=True)
        (self._grok_home / "config.toml").write_text(config)

        uid = demoted_uid_gid()
        if uid is not None:
            # Grok runs as the student and writes its own state under GROK_HOME.
            for p in (self._grok_home, self._grok_home / "config.toml"):
                os.chown(p, uid, uid)

    def _key_env(self) -> str:
        return resolve_model(self._config.model, None).key_env or "XAI_API_KEY"

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
        env["GROK_HOME"] = str(self._grok_home)
        env["GROK_DISABLE_AUTOUPDATER"] = "1"
        env["GROK_MEMORY"] = "0"
        env.pop("PYTHONSAFEPATH", None)

        # Same as the bash tool: sanitize_paths() strips the student-writable
        # venv bin from the runner's PATH, but for the student it is the task's
        # Python, so put it back for Grok's shell tool.
        path_parts = [AGENTS_BIN_DIR]
        venv = env.get("VIRTUAL_ENV")
        if venv and (Path(venv) / "pyvenv.cfg").is_file():
            path_parts.insert(0, f"{venv}/bin")
        env["PATH"] = ":".join([*path_parts, env.get("PATH", "")])

        if (api_key := self.model_api_key) is not None:
            env[self._key_env()] = api_key
        return env

    def _argv(self, instructions: str) -> list[str]:
        # Passed to exec (no shell), so the instruction is a single safe argv
        # element.
        resolved = resolve_model(self._config.model, None)
        session = (
            ["--session-id", self._session_id]
            if self._step_index == 0
            else ["--resume", self._session_id]
        )
        argv = [
            _GROK_BIN,
            "-p",
            instructions,
            "-m",
            resolved.model,
            *session,
            "--output-format",
            "streaming-messages-json",
            "--always-approve",
            "--disable-web-search",
            "--disallowed-tools",
            ",".join(_DISALLOWED_TOOLS),
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
        """Run one Grok invocation, enforcing `time_limit_seconds` by killing the
        process (and its children) if it overruns.

        `error` re-raises after emitting whatever partial output was captured,
        `score` keeps that partial output and lets the step be scored.
        `context_window_limit` is accepted for protocol compatibility but not
        enforced: usage is only read back after the step finishes.
        """
        yield MessageAddedEvent(message=Message(role="user", content=instructions))

        step = self._step_index
        argv = self._argv(instructions)
        self._step_index += 1
        log_path = self._grok_home / f"step_{step}.ndjson"

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

        stdout = (await stdout_task).decode("utf-8", errors="replace")
        stderr_b = await stderr_task

        log_path.write_text(stdout)
        save_artifact(self._config, log_path)

        if timed_out:
            logger.warning(
                "grok hit the {}s time limit on step {}", time_limit_seconds, step
            )
        elif proc.returncode != 0:
            logger.warning(
                "grok exited {} on step {}: {}",
                proc.returncode,
                step,
                stderr_b.decode("utf-8", errors="replace")[-2000:],
            )

        for event in parse_stream(stdout.splitlines()):
            yield event

        if timed_out and on_time_limit == "error":
            msg = f"Time limit of {time_limit_seconds}s reached."
            raise StepTimeLimitReachedError(msg)


def _toml(value: str) -> str:
    """A TOML basic string; JSON string escaping is valid TOML."""
    return json.dumps(value)
