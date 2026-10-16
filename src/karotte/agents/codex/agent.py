"""Codex CLI agent: runs ``codex exec`` once per step, resuming one session."""

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
from karotte.agents.codex.adapter import Session, StreamParser
from karotte.agents.models import resolve_model
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event

_CODEX_BIN = f"{AGENTS_BIN_DIR}/codex"
_RELEASE_URL = (
    "https://github.com/openai/codex/releases/download/rust-v{version}/"
    "{binary}-$arch-unknown-linux-musl.tar.gz"
)

_SHA256: dict[str, dict[str, str]] = {
    "codex": {
        "x86_64": "86f268d81b898f3e144c5ecff2ad2fda2c5802e042d6195b72538fe0fa125057",
        "aarch64": "bf3ca2719f1143b1440fbd85c149379e696c782c50ad7991f8bfcec60b2600c1",
    },
    "codex-code-mode-host": {
        "x86_64": "458239ea45c1fc508d89695f180ab352f2834709a8a46929a4466f4f04399ada",
        "aarch64": "1395ce9367f8acd280674dff083edf25bf221d908b8351d5bd4f3fdde9181e9a",
    },
}


@final
@register_cli_agent
class CodexAgent(CliAgent):
    name: ClassVar[str] = "codex"
    version: ClassVar[str] = "0.162.1"
    provider_url: ClassVar[str] = "https://api.openai.com"
    binary: ClassVar[str] = _CODEX_BIN
    home_dir: ClassVar[str] = ".codex_home"
    home_env: ClassVar[str] = "CODEX_HOME"
    applies_reasoning_effort: ClassVar[bool] = True
    prompt_input: ClassVar[Literal["argv", "file", "stdin"]] = "stdin"
    native_tool_names: ClassVar[frozenset[str]] = frozenset(
        {"bash", "view_image_file", "replace_in_file"}
    )

    @classmethod
    def install(cls) -> list[str]:
        return [
            command
            for binary, digests in _SHA256.items()
            for command in install_release(
                cls.name,
                _RELEASE_URL.format(version=cls.version, binary=binary),
                digests,
                f"{AGENTS_BIN_DIR}/{binary}",
                unpack=(
                    'tar -xzf "$tmp/download" -C "$tmp"; '
                    f'mv "$tmp/{binary}-$arch-unknown-linux-musl" "$tmp/bin"'
                ),
            )
        ]

    def __init__(self, config: EvaluationRunConfig) -> None:
        super().__init__(config)
        self._session: Session = Session()
        self._parser: StreamParser | None = None

    def _config_files(self) -> dict[str, str]:
        """Codex's config: the model behind the proxy or forwarder, the MCP
        server, and nothing that calls OpenAI's backend on its own (telemetry,
        hosted web search, feedback)."""
        lines = [
            f"model = {toml_string(resolve_model(self._config.model, None).model)}",
            'model_provider = "karotte"',
            'model_reasoning_summary = "detailed"',
            'web_search = "disabled"',
            "check_for_update_on_startup = false",
        ]
        if (effort := self._config.applied_reasoning_effort) is not None:
            lines.append(f"model_reasoning_effort = {toml_string(effort)}")
        lines += [
            "",
            "[model_providers.karotte]",
            'name = "karotte"',
            f"base_url = {toml_string(f'{self.model_url}/v1')}",
            f"env_key = {toml_string(self._key_env())}",
            "",
            "[mcp_servers.karotte]",
            f"url = {toml_string(self.mcp_url)}",
            "",
            "[features]",
            # Retries an unreachable model endpoint forever otherwise.
            "unbounded_connection_retries = false",
            "",
            "[analytics]",
            "enabled = false",
            "",
            "[otel]",
            'metrics_exporter = "none"',
            "",
            "[feedback]",
            "enabled = false",
            "",
            "[history]",
            'persistence = "none"',
            "",
            "[shell_environment_policy]",
            "ignore_default_excludes = false",
        ]
        return {"config.toml": "\n".join(lines) + "\n"}

    def _key_env(self) -> str:
        return "OPENAI_API_KEY"

    def _argv(self, instructions: str, prompt_file: Path | None) -> list[str]:
        # Codex's sandbox needs user namespaces, which containers don't grant.
        argv = [
            _CODEX_BIN,
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if self._session.thread_id is not None:
            argv += ["resume", self._session.thread_id]
        return [*argv, "-"]

    def _line_parser(self, instructions: str) -> Callable[[str, int], Iterable[Event]]:
        self._parser = StreamParser(self._session, self._step_index)
        return self._parser

    def _output_ended(self) -> Iterable[Event]:
        return () if self._parser is None else self._parser.flush()
