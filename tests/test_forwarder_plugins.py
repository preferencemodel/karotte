"""Model forwarders from plugins."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from karotte import forwarder_plugins
from karotte.cli import app
from karotte.forwarder_plugins import plugin_forwarder, plugin_providers
from karotte.model_forwarder import ModelForwarder
from karotte.schemas.evaluation_run_config import EvaluationRunConfig

_CONFIG = EvaluationRunConfig(
    run_id="r", task_id="t", model="acme/m", model_api_key="k"
)


class _Plugin:
    agents: tuple[str, ...] = ("claude-code",)
    providers: tuple[str, ...] = ("acme",)

    def forwarder(
        self, config: EvaluationRunConfig, files: Path
    ) -> ModelForwarder | None:
        del files
        return ModelForwarder("https://acme.example", config.model_api_key or "")


@dataclass
class _EntryPoint:
    name: str
    plugin: object

    def load(self) -> object:
        if isinstance(self.plugin, Exception):
            raise self.plugin
        return self.plugin


@pytest.fixture(autouse=True)
def plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [_EntryPoint("b", _Plugin()), _EntryPoint("a", ImportError("broken"))]
    monkeypatch.setattr(forwarder_plugins, "entry_points", lambda group: entries)  # pyright: ignore[reportUnknownLambdaType]


def test_a_plugin_serves_only_its_agents_and_providers(tmp_path: Path):
    forwarder = plugin_forwarder("claude-code", "acme", _CONFIG, tmp_path)
    assert forwarder is not None
    assert forwarder.upstream == "https://acme.example"
    assert plugin_forwarder("codex", "acme", _CONFIG, tmp_path) is None
    assert plugin_forwarder("claude-code", "anthropic", _CONFIG, tmp_path) is None


def test_a_broken_plugin_is_skipped():
    assert plugin_providers("claude-code") == ("acme",)


def test_agents_list_shows_the_providers_plugins_add():
    result = CliRunner().invoke(app, ["agents", "list", "--json"])
    agents = {a["name"]: a for a in json.loads(result.stdout)}
    assert agents["claude-code"]["providers"] == ["anthropic", "hosted_vllm", "acme"]
    assert agents["codex"]["providers"] == ["openai"]
