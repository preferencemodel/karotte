"""Claude Code CLI agent.

Runs Anthropic's ``claude`` binary once per step in print mode, pointed at
karotte's MCP server. The build downloads the pinned native release for the
container's architecture and checks it against the SHA-256 in that release's
manifest. Steps share one session, created on the first step and resumed after.
"""

import json
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, Literal, final

from karotte.agents.claude_code.adapter import StreamParser
from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    CliAgent,
    register_cli_agent,
)
from karotte.agents.models import resolve_model
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event

_CLAUDE_BIN = f"{AGENTS_BIN_DIR}/claude"
_RELEASE_URL = (
    "https://downloads.claude.ai/claude-code-releases/{version}/linux-{arch}/claude"
)

# From the release's manifest.json, per platform name in its URL.
_SHA256: dict[str, str] = {
    "x64": "a967e7b1d8b4e47ee421d5433027880347952b0c0857abf880e2c942a4ec93b3",
    "arm64": "24caa9e6ff13bf227049a2626f1c816fc895023050f0ec3b12dbf14d897367e0",
}

# Claude Code tools that need Anthropic's backend or a human, which a task run
# has neither of.
_DISALLOWED_TOOLS = ("WebSearch", "WebFetch", "AskUserQuestion")

_MCP_CONFIG = "mcp.json"


@final
@register_cli_agent
class ClaudeCodeAgent(CliAgent):
    name: ClassVar[str] = "claude-code"
    version: ClassVar[str] = "2.1.292"
    provider_url: ClassVar[str] = "https://api.anthropic.com"
    provider: ClassVar[str] = "anthropic"
    binary: ClassVar[str] = _CLAUDE_BIN
    home_dir: ClassVar[str] = ".claude_home"
    home_env: ClassVar[str] = "CLAUDE_CONFIG_DIR"
    applies_reasoning_effort: ClassVar[bool] = True
    prompt_input: ClassVar[Literal["argv", "file", "stdin"]] = "stdin"
    native_tool_names: ClassVar[frozenset[str]] = frozenset(
        {"bash", "view_lines_in_file", "view_image_file", "replace_in_file"}
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
                'case "$arch" in x86_64|amd64) arch=x64 ;; aarch64) arch=arm64 ;; esac; '
                f'case "$arch" in {checks} '
                '*) echo "claude-code: unsupported arch $arch" >&2; exit 1 ;; esac; '
                f'curl -fsSL "{url}" -o {_CLAUDE_BIN}.tmp; '
                f'echo "$sha  {_CLAUDE_BIN}.tmp" | sha256sum -c -; '
                f"mv {_CLAUDE_BIN}.tmp {_CLAUDE_BIN}"
            ),
            f"chmod 755 {_CLAUDE_BIN}",
        ]

    def __init__(self, config: EvaluationRunConfig) -> None:
        super().__init__(config)
        self._session_id: str = str(uuid.uuid4())

    def _config_files(self) -> dict[str, str]:
        server = {"type": "http", "url": self.mcp_url}
        return {_MCP_CONFIG: json.dumps({"mcpServers": {"karotte": server}})}

    def _key_env(self) -> str:
        return "ANTHROPIC_API_KEY"

    def _extra_env(self) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": self.model_url,
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }

    def _argv(self, instructions: str, prompt_file: Path | None) -> list[str]:
        session = (
            ["--session-id", self._session_id]
            if self._step_index == 0
            else ["--resume", self._session_id]
        )
        argv = [
            _CLAUDE_BIN,
            "-p",
            "--model",
            resolve_model(self._config.model, None).model,
            *session,
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            # Claude 5 models return empty thinking text unless asked for summaries.
            "--thinking-display",
            "summarized",
            "--mcp-config",
            str(self._home / _MCP_CONFIG),
            "--strict-mcp-config",
            "--permission-mode",
            "bypassPermissions",
            "--disallowedTools",
            ",".join(_DISALLOWED_TOOLS),
        ]
        if (effort := self._config.applied_reasoning_effort) is not None:
            argv += ["--effort", effort]
        if self._config.turn_limit is not None:
            argv += ["--max-turns", str(self._config.turn_limit)]
        return argv

    def _line_parser(self, instructions: str) -> Callable[[str, int], Iterable[Event]]:
        return StreamParser()
