"""The top-level command surface: the noun-verb commands."""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from karotte.cli import app
from karotte.model_catalog import ModelCatalog
from karotte.model_spec import (
    SPECIAL_TRAINING_MODEL_NAME,
    SPECIAL_TRAINING_MODEL_PREFIX,
)

runner = CliRunner()


class TestTasks:
    def test_list(self):
        result = runner.invoke(app, ["tasks", "list"])
        assert result.exit_code == 0
        assert "Available tasks:" in result.stdout

    def test_list_json(self):
        result = runner.invoke(app, ["tasks", "list", "--json"])
        assert result.exit_code == 0
        assert isinstance(json.loads(result.stdout), list)

    def test_old_name_is_gone(self):
        assert runner.invoke(app, ["list-tasks", "--json"]).exit_code != 0


class TestTemplates:
    def test_list(self):
        result = runner.invoke(app, ["templates", "list"])
        assert result.exit_code == 0
        assert "Available templates:" in result.stdout

    def test_list_json(self):
        result = runner.invoke(app, ["templates", "list", "--json"])
        assert result.exit_code == 0
        assert isinstance(json.loads(result.stdout), list)

    def test_old_name_is_gone(self):
        assert runner.invoke(app, ["list-templates", "--json"]).exit_code != 0


class TestCreateEnv:
    def test_prints_next_steps(self, tmp_path: Path):
        env = tmp_path / "my_env"
        result = runner.invoke(app, ["create-env", str(env), "--no-lock"])
        assert result.exit_code == 0
        lines = [line.strip() for line in result.stdout.splitlines()]
        start = lines.index("Next steps:")
        assert lines[start + 1 :] == [
            f"cd {env}",
            "uv sync --extra dev",
            "uv run setup_data.py",
            "uv run karotte create-run-config --model anthropic/claude-fable-5",
            "export ANTHROPIC_API_KEY=...",
            "uv run karotte run --config run_config.json",
        ]


class TestAgents:
    def test_list(self):
        result = runner.invoke(app, ["agents", "list"])
        assert result.exit_code == 0
        assert "mistral-vibe" in result.stdout
        assert "builtin" in result.stdout

    def test_install_is_hidden_from_help(self):
        result = runner.invoke(app, ["agents", "--help"])
        assert result.exit_code == 0
        assert " install " not in re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)

    def test_list_json(self):
        result = runner.invoke(app, ["agents", "list", "--json"])
        assert result.exit_code == 0
        agents = {a["name"]: a for a in json.loads(result.stdout)}

        assert agents["builtin"] == {
            "name": "builtin",
            "kind": "builtin",
            "version": None,
            "requires_install": False,
        }
        vibe = agents["mistral-vibe"]
        assert vibe["kind"] == "cli"
        assert vibe["requires_install"] is True
        assert isinstance(vibe["version"], str)

    def test_add_and_install_are_unchanged(self):
        """`agents install` is baked into every env's Containerfile."""
        for name in ("add", "remove", "install"):
            result = runner.invoke(app, ["agents", name, "--help"])
            assert result.exit_code == 0


class TestModels:
    def test_list(self):
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "claude-opus-5" in result.stdout
        assert "together_ai/" in result.stdout

    def test_list_hides_training_models(self):
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert SPECIAL_TRAINING_MODEL_NAME not in result.stdout
        assert SPECIAL_TRAINING_MODEL_PREFIX not in result.stdout

    def test_list_json(self):
        result = runner.invoke(app, ["models", "list", "--json"])
        assert result.exit_code == 0

        catalog = ModelCatalog.model_validate(json.loads(result.stdout))
        specs = {spec.model: spec for spec in catalog.models}
        assert specs["claude-opus-5"].max_output_tokens == 128000
        assert specs["openai/gpt-5.6"].max_reasoning_effort == "max"
        assert [f.prefix for f in catalog.families] == [
            "pt/",
            "together_ai/",
            "fireworks_ai/",
        ]

    def test_list_json_keeps_training_models(self):
        """The backend reads this output at build time for its model picker."""
        result = runner.invoke(app, ["models", "list", "--json"])
        catalog = ModelCatalog.model_validate(json.loads(result.stdout))
        assert SPECIAL_TRAINING_MODEL_NAME in {spec.model for spec in catalog.models}


def _help(command: str, columns: int) -> str:
    result = runner.invoke(app, [command, "--help"], env={"COLUMNS": str(columns)})
    assert result.exit_code == 0
    return re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)


@pytest.mark.parametrize("columns", [80, 120])
@pytest.mark.parametrize("command", ["run", "build"])
def test_help_is_not_truncated(command: str, columns: int):
    assert "…" not in _help(command, columns)


class TestRunHelp:
    def test_dev_names_the_folder_it_binds(self):
        assert "src/environment" in _help("run", 400)

    def test_keep_containers_example_is_runtime_neutral(self):
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "400"})
        assert result.exit_code == 0
        help_text = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        assert "podman cp" not in help_text
        assert "<runtime> cp" in help_text
