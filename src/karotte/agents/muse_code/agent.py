"""Muse Code CLI agent: runs ``muse exec`` once per step in one session,
against Meta's API or a proxy, and reads the session log Muse writes."""

import json
import os
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, Literal, final

from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    CliAgent,
    install_release,
    register_cli_agent,
)
from karotte.agents.models import resolve_model
from karotte.agents.muse_code.adapter import StreamParser
from karotte.model_forwarder import ModelForwarder, MuseForwarder
from karotte.model_spec import spec_for
from karotte.providers import PROXY_PLACEHOLDER_KEY
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event

_MUSE_BIN = f"{AGENTS_BIN_DIR}/muse"
_RELEASE_URL = (
    "https://lookaside.facebook.com/lookaside/muse/download/?channel=muse"
    "&version={version}&file=muse-$arch-linux"
)

_SHA256: dict[str, str] = {
    "x86": "3685b023d45a1a07c8c559da00088bdf59564920b13a85adad11e13ad6f4d181",
    "aarch64": "7159d6431fc5ff4c9dea32c62d0b771f5089d14389ec672ae903755bd39374dc",
}


@final
@register_cli_agent
class MuseCodeAgent(CliAgent):
    name: ClassVar[str] = "muse-code"
    version: ClassVar[str] = "1.4.20-R8619.1"
    provider_url: ClassVar[str] = "https://api.meta.ai"
    providers: ClassVar[tuple[str, ...]] = ("meta",)
    proxy_via_forwarder: ClassVar[bool] = True
    binary: ClassVar[str] = _MUSE_BIN
    home_dir: ClassVar[str] = ".muse_home"
    applies_reasoning_effort: ClassVar[bool] = True
    prompt_input: ClassVar[Literal["argv", "file", "stdin"]] = "file"
    events_from: ClassVar[Literal["stdout", "file"]] = "file"
    native_tool_names: ClassVar[frozenset[str]] = frozenset(
        {"bash", "view_lines_in_file", "view_image_file", "replace_in_file"}
    )

    @classmethod
    def install(cls) -> list[str]:
        return install_release(
            cls.name,
            _RELEASE_URL.format(version=cls.version),
            _SHA256,
            _MUSE_BIN,
            arch_names={"x86_64": "x86", "aarch64": "aarch64"},
        )

    def __init__(self, config: EvaluationRunConfig) -> None:
        super().__init__(config)
        self._model: str = resolve_model(config.model, None).model
        self._session_id: str = str(uuid.uuid4())
        self._session_log: Path | None = None
        self._parser: StreamParser | None = None

    def _make_forwarder(self) -> ModelForwarder | None:
        # Always forward: only Meta's API serves Muse's model list.
        return MuseForwarder(
            os.environ.get("KAROTTE_PROXY_URL") or self.provider_url,
            self._config.model_api_key or PROXY_PLACEHOLDER_KEY,
            self._model,
            spec_for(self._config.model).max_output_tokens,
        )

    def _config_files(self) -> dict[str, str]:
        # Muse sends its key only to a base URL pinned here, not to `--base-url`.
        settings = {
            "schema_version": 1,
            "endpoint_transport": {
                "base_url": f"{self.model_url}/v1",
                "auth": "bearer",
            },
            "mcpServers": {"karotte": {"type": "streamable-http", "url": self.mcp_url}},
            "telemetry": {"enabled": False},
        }
        return {"config/muse/settings.json": json.dumps(settings, indent=2)}

    def _key_env(self) -> str:
        return "META_API_KEY"

    def _extra_env(self) -> dict[str, str]:
        return {
            "XDG_CONFIG_HOME": str(self._home / "config"),
            "XDG_DATA_HOME": str(self._home / "data"),
        }

    def _argv(self, instructions: str, prompt_file: Path | None) -> list[str]:
        assert prompt_file is not None
        argv = [
            _MUSE_BIN,
            "exec",
            "--json",
            # Muse's own sandbox needs bubblewrap; the student's sandbox applies.
            "--yolo",
            "--session-id",
            self._session_id,
            "--model",
            self._model,
            "--prompt-file",
            str(prompt_file),
        ]
        if (effort := self._config.applied_reasoning_effort) is not None:
            argv += ["--reasoning-effort", effort]
        return argv

    def _event_log(self) -> Path | None:
        # Kept once found, so a file the student plants later can't replace it.
        if self._session_log is None:
            sessions = self._home / "data" / "muse" / "sessions"
            found = sorted(sessions.glob(f"*/*/*/{self._session_id}/session.jsonl"))
            self._session_log = found[0] if found else None
        return self._session_log

    def _line_parser(self, instructions: str) -> Callable[[str, int], Iterable[Event]]:
        self._parser = StreamParser()
        return self._parser

    def _output_ended(self) -> Iterable[Event]:
        return () if self._parser is None else self._parser.flush()

    def _reported_error(self) -> str | None:
        return None if self._parser is None else self._parser.error
