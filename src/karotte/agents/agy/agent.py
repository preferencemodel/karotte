"""Antigravity CLI agent: runs ``agy`` once per step in headless mode, resuming
one conversation, against the Gemini API or Vertex AI."""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, Literal, final

from karotte.agents.agy.adapter import Session, StreamParser
from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    CliAgent,
    install_release,
    register_cli_agent,
)
from karotte.agents.models import resolve_model
from karotte.model_forwarder import GeminiForwarder, ModelForwarder, vertex_forwarder
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event

_AGY_BIN = f"{AGENTS_BIN_DIR}/agy"
_RELEASE_URL = (
    "https://github.com/google-antigravity/antigravity-cli/releases/download/"
    "{version}/agy_cli_linux_$arch.tar.gz"
)

_SHA256: dict[str, str] = {
    "x64": "ed0526231e3916b6cc1b025cf356862e88cc036e27c3c5eae5a580495fc79773",
    "arm64": "107b7499ca90018ab2b6c1b636cdec4251dfd836656526845e354be1bfe05eba",
}


@final
@register_cli_agent
class AgyAgent(CliAgent):
    name: ClassVar[str] = "agy"
    version: ClassVar[str] = "1.4.10"
    provider_url: ClassVar[str] = "https://generativelanguage.googleapis.com"
    providers: ClassVar[tuple[str, ...]] = ("gemini", "vertex_ai")
    binary: ClassVar[str] = _AGY_BIN
    home_dir: ClassVar[str] = ".gemini"
    applies_reasoning_effort: ClassVar[bool] = True
    prompt_input: ClassVar[Literal["argv", "file", "stdin"]] = "stdin"
    native_tool_names: ClassVar[frozenset[str]] = frozenset(
        {"bash", "view_lines_in_file", "view_image_file", "replace_in_file"}
    )

    @classmethod
    def install(cls) -> list[str]:
        return install_release(
            cls.name,
            _RELEASE_URL.format(version=cls.version),
            _SHA256,
            _AGY_BIN,
            arch_names={"x86_64": "x64", "aarch64": "arm64"},
            unpack=(
                'tar -xzf "$tmp/download" -C "$tmp" antigravity; '
                'mv "$tmp/antigravity" "$tmp/bin"'
            ),
        )

    def __init__(self, config: EvaluationRunConfig) -> None:
        super().__init__(config)
        resolved = resolve_model(config.model, None)
        self._provider: str = resolved.provider
        self._model: str = resolved.model
        self._session: Session = Session()
        self._parser: StreamParser | None = None

    def _config_files(self) -> dict[str, str]:
        """API-key mode against the Gemini API, with the run's model declared
        as a custom one so agy sends its id unchanged."""
        settings = {
            "modelProvider": "gemini",
            "enableTelemetry": False,
            "customModelsConfig": {
                "customModels": {
                    self._model: {
                        "modelId": self._model,
                        "modelName": f"gemini-api://{self._model}",
                        "label": self._model,
                        "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    }
                }
            },
        }
        mcp = {"mcpServers": {"karotte": {"serverUrl": self.mcp_url}}}
        return {
            "antigravity-cli/settings.json": json.dumps(settings, indent=2),
            "config/mcp_config.json": json.dumps(mcp, indent=2),
        }

    def _key_env(self) -> str:
        return "GEMINI_API_KEY"

    def _extra_env(self) -> dict[str, str]:
        return {
            "HOME": str(self._home.parent.absolute()),
            "GOOGLE_GEMINI_BASE_URL": self.model_url,
            "AGY_CLI_DISABLE_AUTO_UPDATE": "1",
        }

    def _make_forwarder(self) -> ModelForwarder | None:
        if self._provider == "vertex_ai":
            return vertex_forwarder()
        if key := self._config.model_api_key:
            return GeminiForwarder(self.provider_url, key)
        return None

    def _prompt_text(self, instructions: str) -> str:
        return (
            json.dumps({"event": "user", "message": {"content": instructions}}) + "\n"
        )

    def _argv(self, instructions: str, prompt_file: Path | None) -> list[str]:
        argv = [
            _AGY_BIN,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--disable-slash-commands",
            # Otherwise agy may end a step before karotte's own time limit.
            "--print-timeout",
            "0",
            "--model",
            self._model,
        ]
        if (effort := self._config.applied_reasoning_effort) is not None:
            argv += ["--effort", effort]
        if self._session.conversation_id is not None:
            argv += ["--conversation", self._session.conversation_id]
        return argv

    def _line_parser(self, instructions: str) -> Callable[[str, int], Iterable[Event]]:
        self._parser = StreamParser(self._session, self._step_index)
        return self._parser

    def _output_ended(self) -> Iterable[Event]:
        return () if self._parser is None else self._parser.flush()
