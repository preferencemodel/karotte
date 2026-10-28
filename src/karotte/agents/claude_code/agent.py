"""Claude Code CLI agent.

Runs Anthropic's ``claude`` binary once per step in print mode, pointed at
karotte's MCP server. The build downloads the pinned native release for the
container's architecture and checks it against the SHA-256 in that release's
manifest. Steps share one session, created on the first step and resumed after.
"""

import json
import os
import shlex
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, Literal, final

from karotte.agents.claude_code.adapter import TOOL_RESULT_HOOK_EVENTS, StreamParser
from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    CliAgent,
    install_release,
    register_cli_agent,
)
from karotte.agents.message_loop import TIME_REMAINING_NOTE
from karotte.agents.models import resolve_model
from karotte.model_forwarder import ModelForwarder
from karotte.model_spec import spec_for
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

# The env vars behind Claude Code's model aliases (`haiku`, `sonnet`, ...) and
# the models it picks for background calls and subagents.
_MODEL_ALIAS_ENVS = (
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)


@final
@register_cli_agent
class ClaudeCodeAgent(CliAgent):
    name: ClassVar[str] = "claude-code"
    version: ClassVar[str] = "2.1.292"
    provider_url: ClassVar[str] = "https://api.anthropic.com"
    # Claude Code only speaks Anthropic's Messages API (`/v1/messages`), so each
    # provider here must serve it. `hosted_vllm` is litellm's prefix for a
    # self-hosted server (vLLM, SGLang, ...), reached through KAROTTE_PROXY_URL.
    providers: ClassVar[tuple[str, ...]] = ("anthropic", "hosted_vllm")
    proxy_via_forwarder: ClassVar[bool] = True
    binary: ClassVar[str] = _CLAUDE_BIN
    home_dir: ClassVar[str] = ".claude_home"
    home_env: ClassVar[str | None] = "CLAUDE_CONFIG_DIR"
    applies_reasoning_effort: ClassVar[bool] = True
    supports_turn_limit: ClassVar[bool] = True
    prompt_input: ClassVar[Literal["argv", "file", "stdin"]] = "stdin"
    native_tool_names: ClassVar[frozenset[str]] = frozenset(
        {"bash", "view_lines_in_file", "view_image_file", "replace_in_file"}
    )

    @classmethod
    def install(cls) -> list[str]:
        return install_release(
            cls.name,
            _RELEASE_URL.format(version=cls.version, arch="$arch"),
            _SHA256,
            _CLAUDE_BIN,
            arch_names={"x86_64": "x64", "aarch64": "arm64"},
        )

    def __init__(self, config: EvaluationRunConfig) -> None:
        super().__init__(config)
        self._on_anthropic: bool = spec_for(config.model).provider == "anthropic"
        self._session_id: str = str(uuid.uuid4())
        self._parser: StreamParser | None = None

    def _make_forwarder(self) -> ModelForwarder | None:
        # Anthropic is the only provider with a fixed address; without a proxy,
        # any other model's key would be forwarded to Anthropic.
        if not self._on_anthropic and not os.environ.get("KAROTTE_PROXY_URL"):
            msg = (
                f"claude-code reaches {self._config.model} only through a proxy: "
                "set KAROTTE_PROXY_URL (`karotte run --proxy`) to its server"
            )
            raise ValueError(msg)
        return super()._make_forwarder()

    def _config_files(self) -> dict[str, str]:
        server = {"type": "http", "url": self.mcp_url}
        return {_MCP_CONFIG: json.dumps({"mcpServers": {"karotte": server}})}

    def _key_env(self) -> str:
        return "ANTHROPIC_API_KEY"

    def _extra_env(self) -> dict[str, str]:
        env = {
            "ANTHROPIC_BASE_URL": self.model_url,
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        if not self._on_anthropic:
            # The aliases name Claude models, which only Anthropic serves.
            model = resolve_model(self._config.model, None).model
            env |= dict.fromkeys(_MODEL_ALIAS_ENVS, model)
            # Claude Code asks for `thinking: {"type": "adaptive"}` on every
            # call, which SGLang refuses unless the server runs a reasoning
            # parser. With 0 it leaves `thinking` out, and the model's chat
            # template decides whether it thinks.
            env["MAX_THINKING_TOKENS"] = "0"
            if self._config.applied_reasoning_effort is None:
                # Likewise `output_config: {"effort": "high"}`, which SGLang
                # passes to the chat template, and Qwen3.8's refuses anything
                # but xhigh, medium or low. `unset` leaves it out, so the
                # template's default applies. It overrides `--effort`, so only
                # when the run config sets no effort.
                env["CLAUDE_CODE_EFFORT_LEVEL"] = "unset"
            if (limit := self._step_context_window_limit) is not None:
                # Claude Code assumes 200K for a model it doesn't know and
                # compacts there; this is the window it compacts at instead.
                env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(limit)
        return env

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
            # So the transcript keeps what tool hooks tell the model.
            "--include-hook-events",
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
        deadline = self._step_deadline
        if deadline is not None and self._config.inject_time_remaining_counter:
            argv += ["--settings", json.dumps(_time_remaining_settings(deadline))]
        return argv

    def _line_parser(self, instructions: str) -> Callable[[str, int], Iterable[Event]]:
        self._parser = StreamParser()
        return self._parser

    def _output_ended(self) -> Iterable[Event]:
        return () if self._parser is None else self._parser.flush()

    def _reported_error(self) -> str | None:
        # Claude Code's stderr can end in a warning such as
        # `[claude-code:unrecognized_model]`, printed for any non-Claude model.
        return None if self._parser is None else self._parser.error


def _time_remaining_settings(deadline: float) -> dict[str, object]:
    """`--settings` whose hooks tell the model how long its step has left after
    every tool call, the note the builtin agent adds after every turn. A hook's
    `additionalContext` reaches the model next to the tool's result."""
    return {
        "hooks": {
            event: [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": _time_remaining_hook(event, deadline),
                        }
                    ]
                }
            ]
            for event in TOOL_RESULT_HOOK_EVENTS
        }
    }


def _time_remaining_hook(event: str, deadline: float) -> str:
    """A POSIX shell command, since the task image may have no Python, that
    prints the hook output for `event` with the seconds left until `deadline`."""
    output = json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": TIME_REMAINING_NOTE.format(seconds="%s"),
            }
        }
    )
    return (
        f't=$(({int(deadline)} - $(date +%s))); [ "$t" -gt 0 ] || t=0; '
        f'printf {shlex.quote(output)} "$t"'
    )
