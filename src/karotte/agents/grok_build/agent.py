"""Grok Build CLI agent.

Runs xAI's ``grok`` binary once per step in headless mode, pointed at
karotte's MCP server. Grok ships as a single static binary, so the build
downloads the pinned release for the container's architecture and checks it
against a pinned SHA-256 (xAI publishes no checksums).

Steps share one Grok session (created on the first step, resumed after), so a
later step sees the conversation of the earlier ones, as with the builtin loop.
"""

import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, Literal, final

from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    CliAgent,
    install_release,
    register_cli_agent,
    toml_string,
)
from karotte.agents.grok_build.adapter import parse_line
from karotte.agents.models import resolve_model
from karotte.model_spec import spec_for
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event

_GROK_BIN = f"{AGENTS_BIN_DIR}/grok"
_KEY_ENV = "XAI_API_KEY"
_RELEASE_URL = "https://x.ai/cli/grok-{version}-linux-{arch}.gz"

# SHA-256 of the decompressed linux binary, per `uname -m` spelling xAI uses.
_SHA256: dict[str, str] = {
    "x86_64": "f301b4e9a8ec691cdebc4ae8131f35833165d01b734aa14135a128afc6402a25",
    "aarch64": "475cbf18f8a1484b2d7c16bf47caed2113c460aa77bf407861dbeafcc6d926a1",
}

# Grok tools that need xAI's backend or a human, which a task run has neither of.
_DISALLOWED_TOOLS = ("send_feedback", "image_edit", "ask_user_question")


@final
@register_cli_agent
class GrokBuildAgent(CliAgent):
    name: ClassVar[str] = "grok-build"
    version: ClassVar[str] = "1.0.69"
    provider_url: ClassVar[str] = "https://api.x.ai"
    providers: ClassVar[tuple[str, ...]] = ("xai",)
    binary: ClassVar[str] = _GROK_BIN
    home_dir: ClassVar[str] = ".grok_home"
    home_env: ClassVar[str | None] = "GROK_HOME"
    supports_turn_limit: ClassVar[bool] = True
    supports_temperature: ClassVar[bool] = True
    applies_reasoning_effort: ClassVar[bool] = True
    prompt_input: ClassVar[Literal["argv", "file", "stdin"]] = "file"
    # Grok ships its own shell and file read/edit tools (run_terminal_command,
    # read_file, search_replace); use those rather than karotte's over MCP.
    native_tool_names: ClassVar[frozenset[str]] = frozenset(
        {"bash", "view_lines_in_file", "replace_in_file"}
    )

    @classmethod
    def install(cls) -> list[str]:
        return install_release(
            cls.name,
            _RELEASE_URL.format(version=cls.version, arch="$arch"),
            _SHA256,
            _GROK_BIN,
            unpack='gunzip -c "$tmp/download" > "$tmp/bin"',
            verify_unpacked=True,
        )

    def __init__(self, config: EvaluationRunConfig) -> None:
        super().__init__(config)
        self._session_id: str = str(uuid.uuid4())

    def _config_files(self) -> dict[str, str]:
        """Grok's config: model endpoint, MCP server, features.

        Model traffic goes through the proxy's OpenAI-compatible route when
        ``KAROTTE_PROXY_URL`` is set, else through the forwarder to xAI. It
        uses the Responses API: xAI carries reasoning over to the next turn
        only there (as encrypted content), and on Chat Completions it drops the
        reasoning the CLI sends back, so every turn starts without it.
        Features that call xAI's backend on their own (telemetry, remote model
        catalog, backend web search, media generation) are turned off, and
        session titles use the run's model rather than Grok's default."""
        resolved = resolve_model(self._config.model, None)
        model = toml_string(resolved.model)
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
            f"base_url = {toml_string(f'{self.model_url}/v1')}\n"
            f"env_key = {toml_string(_KEY_ENV)}\n"
            'api_backend = "responses"\n'
            f"{self._sampling_config()}\n"
            "[mcp_servers.karotte]\n"
            f"url = {toml_string(self.mcp_url)}\n"
        )
        return {"config.toml": config}

    def _sampling_config(self) -> str:
        """Temperature and reasoning-effort lines for the model's config table.

        Grok sends a model's ``temperature`` and the default entry of its
        ``reasoning_efforts`` menu with every request (its ``--reasoning-effort``
        flag is ignored for a model without a menu)."""
        spec = spec_for(self._config.model)
        lines: list[str] = []
        if self._config.temperature is not None:
            lines.append(f"temperature = {self._config.temperature}")
        effort = self._config.applied_reasoning_effort
        if effort is not None:
            options = [
                f"{{ value = {toml_string(level)}"
                + (", default = true" if level == effort else "")
                + " }"
                for level in spec.reasoning_effort_levels
            ]
            lines.append(f"reasoning_efforts = [{', '.join(options)}]")
        return "".join(f"{line}\n" for line in lines)

    def _key_env(self) -> str:
        return _KEY_ENV

    def _extra_env(self) -> dict[str, str]:
        return {"GROK_DISABLE_AUTOUPDATER": "1", "GROK_MEMORY": "0"}

    def _argv(self, instructions: str, prompt_file: Path | None) -> list[str]:
        # The prompt goes in a file, not argv: argv is visible to the student's
        # processes, so e.g. `pkill -f blender` would match a prompt that
        # mentions blender and kill the agent itself.
        assert prompt_file is not None
        resolved = resolve_model(self._config.model, None)
        session = (
            ["--session-id", self._session_id]
            if self._step_index == 0
            else ["--resume", self._session_id]
        )
        argv = [
            _GROK_BIN,
            "--prompt-file",
            str(prompt_file),
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

    def _line_parser(self, instructions: str) -> Callable[[str, int], Iterable[Event]]:
        return parse_line
