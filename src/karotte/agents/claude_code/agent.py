"""Claude Code CLI agent.

Runs Anthropic's ``claude`` binary once per step in print mode, pointed at
karotte's MCP server. The build downloads the pinned native release for the
container's architecture and checks it against the SHA-256 in that release's
manifest. Steps share one session, created on the first step and resumed after.
"""

import json
import os
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, Literal, final

from karotte.agents.claude_code.adapter import StreamParser
from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    CliAgent,
    install_release,
    register_cli_agent,
)
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
    "x64": "86dc1fcc69498159eaf15f446613e847010f9036bfbbdffd4c49350be8489ada",
    "arm64": "2e2496c9a8065df93acb0fca2d341f59a6b8777649b2e025ab8c9bcd5eef61f3",
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

# The output tokens Claude Code asks for per call on a model it doesn't know.
_UNKNOWN_MODEL_REPLY = 32000

# Claude Code compacts at its auto-compact window less the smaller of the reply
# and 20,000, less 13,000. It reads an auto-compact window only from 100,000 up.
_COMPACT_SUMMARY_TOKENS = 20000
_COMPACT_BUFFER_TOKENS = 13000
_MIN_COMPACT_WINDOW = 100000

# What compacting needs in the context past the compaction point, besides its
# reply: the turn that crossed the point, Claude Code's summary prompt, and a
# retry of a failed compaction, which can carry every tool's full definition.
COMPACTION_SLACK_TOKENS = 16384

# Claude Code waits at least this long on a stream that has gone quiet.
_MIN_STREAM_IDLE_TIMEOUT_SECONDS = 300


@final
@register_cli_agent
class ClaudeCodeAgent(CliAgent):
    name: ClassVar[str] = "claude-code"
    version: ClassVar[str] = "2.1.311"
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
        # Here rather than in `_extra_env`, so a bad option refuses the run
        # before anything starts.
        self._options_env: dict[str, str] = _options_env(config, self._on_anthropic)
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
            # Otherwise the task's MCP tools sit behind a ToolSearch call.
            "ENABLE_TOOL_SEARCH": "false",
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
        env |= self._options_env
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
        self._parser = StreamParser()
        return self._parser

    def _output_ended(self) -> Iterable[Event]:
        return () if self._parser is None else self._parser.flush()

    def _reported_error(self) -> str | None:
        # Claude Code's stderr can end in a warning such as
        # `[claude-code:unrecognized_model]`, printed for any non-Claude model.
        return None if self._parser is None else self._parser.error


def window_env(context_window: int, max_output_tokens: int) -> dict[str, str]:
    """Env for Claude Code on a model with ``context_window`` tokens of context,
    asking for ``max_output_tokens`` per call.

    Claude Code compacts at ``context_window - max_output_tokens -
    COMPACTION_SLACK_TOKENS``. That is late enough to use the context, and early
    enough that the compaction call, which carries the whole conversation and
    asks for a whole reply, still fits. A self-hosted server refuses a request
    whose prompt and ``max_tokens`` together exceed its context rather than
    shortening the reply, so compacting with less room ends the step.

    Raises ValueError when the reply leaves no compaction point Claude Code
    accepts.
    """
    compacts_at = context_window - max_output_tokens - COMPACTION_SLACK_TOKENS
    compact_window = (
        compacts_at
        + min(max_output_tokens, _COMPACT_SUMMARY_TOKENS)
        + _COMPACT_BUFFER_TOKENS
    )
    if compact_window < _MIN_COMPACT_WINDOW:
        msg = (
            f"a {max_output_tokens}-token reply leaves Claude Code no compaction "
            f"point in a {context_window}-token context (it would compact at "
            f"{compacts_at}); lower extra_config['max_output_tokens']"
        )
        raise ValueError(msg)
    return {
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context_window),
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(compact_window),
    }


def _options_env(config: EvaluationRunConfig, on_anthropic: bool) -> dict[str, str]:
    """Env for the run's ``extra_config`` options: ``max_output_tokens``,
    ``context_window`` and ``stream_idle_timeout_seconds``."""
    reply = _positive_int(config, "max_output_tokens")
    window = _positive_int(config, "context_window")
    idle = _positive_int(config, "stream_idle_timeout_seconds")
    env: dict[str, str] = {}
    if window is not None:
        if on_anthropic:
            msg = (
                "extra_config['context_window'] is for a model Claude Code doesn't "
                f"know; it reads {config.model}'s window from the model itself"
            )
            raise ValueError(msg)
        env |= window_env(window, reply or _UNKNOWN_MODEL_REPLY)
    elif reply is not None:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(reply)
    if idle is not None:
        if idle < _MIN_STREAM_IDLE_TIMEOUT_SECONDS:
            msg = (
                f"extra_config['stream_idle_timeout_seconds']={idle} is below the "
                f"{_MIN_STREAM_IDLE_TIMEOUT_SECONDS} Claude Code waits at least"
            )
            raise ValueError(msg)
        env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] = str(idle * 1000)
    return env


def _positive_int(config: EvaluationRunConfig, key: str) -> int | None:
    raw = (config.extra_config or {}).get(key)
    if raw is not None and (
        not isinstance(raw, int) or isinstance(raw, bool) or raw < 1
    ):
        msg = f"extra_config[{key!r}] must be a positive integer, got {raw!r}"
        raise ValueError(msg)
    return raw
