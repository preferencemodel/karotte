"""Mistral Vibe CLI agent.

Runs the ``vibe`` binary once per step in headless mode, pointed at karotte's
MCP server. Vibe is a Python tool (``mistral-vibe`` on PyPI), so it installs as
an isolated ``uv tool`` into the shared agents dir at build time.
"""

import os
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import ClassVar, final

from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    AGENTS_DIR,
    CliAgent,
    register_cli_agent,
)
from karotte.agents.mistral_vibe.adapter import StreamParser
from karotte.agents.models import resolve_model
from karotte.schemas.transcript import Event, MessageAddedEvent

_VIBE_BIN = f"{AGENTS_BIN_DIR}/vibe"


@final
@register_cli_agent
class MistralVibeAgent(CliAgent):
    name: ClassVar[str] = "mistral-vibe"
    version: ClassVar[str] = "2.19.0"
    provider_url: ClassVar[str] = "https://api.mistral.ai"
    provider: ClassVar[str] = "mistral"
    binary: ClassVar[str] = _VIBE_BIN
    home_dir: ClassVar[str] = ".vibe"
    home_env: ClassVar[str] = "VIBE_HOME"
    supports_turn_limit: ClassVar[bool] = True
    supports_temperature: ClassVar[bool] = True
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

    def _config_files(self) -> dict[str, str]:
        """Vibe's config: model, provider, MCP server.

        Model traffic goes to :attr:`model_url`: the proxy when
        ``KAROTTE_PROXY_URL`` is set, otherwise the forwarder to Mistral's API.
        Both expose an OpenAI-compatible route, so the provider is declared
        ``generic``/``openai``."""
        resolved = resolve_model(self._config.model, None)
        provider_name = "proxy" if os.environ.get("KAROTTE_PROXY_URL") else "mistral"
        api_base = f"{self.model_url}/v1"
        temperature = (
            f"temperature = {self._config.temperature}\n"
            if self._config.temperature is not None
            else ""
        )
        config = (
            f'active_model = "{resolved.model}"\n\n'
            "[session_logging]\n"
            "enabled = false\n\n"
            "[[providers]]\n"
            f'name = "{provider_name}"\n'
            f'api_base = "{api_base}"\n'
            f'api_key_env_var = "{self._key_env()}"\n'
            'api_style = "openai"\n'
            'backend = "generic"\n\n'
            "[[models]]\n"
            f'name = "{resolved.model}"\n'
            f'provider = "{provider_name}"\n'
            f'alias = "{resolved.model}"\n'
            f"{temperature}\n"
            "[[mcp_servers]]\n"
            'name = "karotte"\n'
            'transport = "streamable-http"\n'
            f'url = "{self.mcp_url}"\n'
        )
        return {"config.toml": config}

    def _key_env(self) -> str:
        return resolve_model(self._config.model, None).key_env or "MISTRAL_API_KEY"

    def _argv(self, instructions: str, prompt_file: Path | None) -> list[str]:
        # Vibe has no prompt-file flag.
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

    def _line_parser(self, instructions: str) -> Callable[[str, int], Iterable[Event]]:
        """Vibe re-sends its own system prompt with every invocation and echoes
        the step instructions as its first user message; both are dropped, as
        the transcript already carries them."""
        parse = StreamParser(self._config.model)
        seen_user = False

        def events(line: str, i: int) -> Iterator[Event]:
            nonlocal seen_user
            for event in parse(line, i):
                if isinstance(event, MessageAddedEvent):
                    if event.message.role == "system":
                        continue
                    if not seen_user and event.message.role == "user":
                        seen_user = True
                        if event.message.content == instructions:
                            continue
                yield event

        return events
