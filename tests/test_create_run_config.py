import json
from pathlib import Path

import pytest
import typer

import karotte.load_tasks
from karotte.cli.create_run_config import create_run_config
from karotte.task import Task


class TestCreateRunConfig:
    def test_writes_file_to_specified_path(self, tmp_path: Path):
        config_path = tmp_path / "my_config.json"

        create_run_config(config_path=str(config_path))

        assert config_path.exists()

    def test_written_file_is_valid_json(self, tmp_path: Path):
        config_path = tmp_path / "config.json"

        create_run_config(config_path=str(config_path))

        json.loads(config_path.read_text())  # raises if invalid

    def test_writes_a_fresh_run_id(self, tmp_path: Path):
        ids = []
        for name in ("a.json", "b.json"):
            create_run_config(config_path=str(tmp_path / name))
            ids.append(json.loads((tmp_path / name).read_text())["run_id"])

        assert all(ids)
        assert ids[0] != ids[1]

    def test_writes_no_key_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The run reads it from the provider's key variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-key")
        config_path = tmp_path / "config.json"

        create_run_config(config_path=str(config_path), model="openai/gpt-5.5")

        assert json.loads(config_path.read_text())["model_api_key"] is None
        assert "sk-ant-secret-key" not in config_path.read_text()

    def test_explicit_model_api_key_is_written_as_is(self, tmp_path: Path):
        config_path = tmp_path / "config.json"

        create_run_config(config_path=str(config_path), model_api_key="sk-ant-my-key")

        config = json.loads(config_path.read_text())
        assert config["model_api_key"] == "sk-ant-my-key"

    def test_explicit_env_var_reference_is_preserved(self, tmp_path: Path):
        """Passing a custom $OTHER_VAR reference should be stored verbatim."""
        config_path = tmp_path / "config.json"

        create_run_config(config_path=str(config_path), model_api_key="$MY_CUSTOM_KEY")

        config = json.loads(config_path.read_text())
        assert config["model_api_key"] == "$MY_CUSTOM_KEY"

    def test_default_model_is_set(self, tmp_path: Path):
        config_path = tmp_path / "config.json"

        create_run_config(config_path=str(config_path))

        config = json.loads(config_path.read_text())
        assert config["model"] == "anthropic/claude-opus-5-5"

    def test_custom_model_is_written(self, tmp_path: Path):
        config_path = tmp_path / "config.json"

        create_run_config(
            config_path=str(config_path), model="vertex_ai/gemini-3-pro-preview"
        )

        config = json.loads(config_path.read_text())
        assert config["model"] == "vertex_ai/gemini-3-pro-preview"

    def test_written_config_has_expected_defaults(self, tmp_path: Path):
        config_path = tmp_path / "config.json"

        create_run_config(config_path=str(config_path))

        config = json.loads(config_path.read_text())
        assert config["transcript_file"] == "out/transcript.json"
        assert config["use_hints"] is True

    def test_exits_cleanly_when_no_tasks_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """With every task filtered out, exit with a clear error, not IndexError."""

        def empty_loader() -> list[type[Task]]:
            return []

        monkeypatch.setattr(
            karotte.load_tasks, "_get_task_loader", lambda: empty_loader
        )
        config_path = tmp_path / "config.json"

        with pytest.raises(typer.Exit) as exc_info:
            create_run_config(config_path=str(config_path))

        assert exc_info.value.exit_code == 1
        assert not config_path.exists()


class TestTaskOption:
    @pytest.fixture(autouse=True)
    def _two_tasks(self, monkeypatch: pytest.MonkeyPatch):
        tasks = [type("A", (), {"id": "a-task"}), type("B", (), {"id": "b-task"})]
        monkeypatch.setattr(
            karotte.load_tasks, "_get_task_loader", lambda: lambda: tasks
        )

    def test_defaults_to_the_first_task(self, tmp_path: Path):
        config_path = tmp_path / "config.json"

        create_run_config(config_path=str(config_path))

        assert json.loads(config_path.read_text())["task_id"] == "a-task"

    def test_writes_the_chosen_task(self, tmp_path: Path):
        config_path = tmp_path / "config.json"

        create_run_config(config_path=str(config_path), task="b-task")

        assert json.loads(config_path.read_text())["task_id"] == "b-task"

    def test_unknown_task_lists_the_available_ones(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        config_path = tmp_path / "config.json"

        with pytest.raises(typer.Exit) as exc_info:
            create_run_config(config_path=str(config_path), task="nope")

        assert exc_info.value.exit_code == 1
        assert not config_path.exists()
        out = capsys.readouterr().out
        assert "'nope'" in out
        assert "a-task" in out
        assert "b-task" in out
