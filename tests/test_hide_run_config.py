import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from karotte.hide_run_config import (
    MODEL_API_KEY_PATH,
    RUN_CONFIG_PATH,
    hide_run_config_from_procfs,
)


def test_does_nothing_outside_container(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    monkeypatch.setattr("sys.argv", ["karotte", "run", "--config", '{"secret": "key"}'])

    with patch("karotte.hide_run_config.os.execv") as mock_execv:
        hide_run_config_from_procfs()

    mock_execv.assert_not_called()


def test_does_nothing_for_a_file_path_inside_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    config_file = tmp_path / "config.json"
    config_file.write_text('{"secret": "key"}')
    monkeypatch.setattr("sys.argv", ["karotte", "run", "--config", str(config_file)])

    with patch("karotte.hide_run_config.os.execv") as mock_execv:
        hide_run_config_from_procfs()

    mock_execv.assert_not_called()


def test_does_nothing_for_other_subcommands(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.setattr(
        "sys.argv", ["karotte", "build", "--config", '{"secret": "key"}']
    )

    with patch("karotte.hide_run_config.os.execv") as mock_execv:
        hide_run_config_from_procfs()

    mock_execv.assert_not_called()


def test_writes_root_only_file_and_reexecs_with_inline_json(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.setattr(
        "sys.argv",
        ["karotte", "run", "--no-containerized", "--config", '{"secret": "key"}'],
    )

    with (
        patch("karotte.hide_run_config.Path.write_text") as mock_write,
        patch("karotte.hide_run_config.os.chmod") as mock_chmod,
        patch("karotte.hide_run_config.os.execv") as mock_execv,
    ):
        hide_run_config_from_procfs()

    mock_write.assert_called_once_with('{"secret": "key"}')
    mock_chmod.assert_called_once_with(RUN_CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)
    new_argv = mock_execv.call_args[0][1]
    assert RUN_CONFIG_PATH in new_argv
    assert '{"secret": "key"}' not in new_argv
    assert "--no-containerized" in new_argv


def test_reexec_replaces_short_config_flag_with_file_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.setattr(
        "sys.argv", ["karotte", "run", "--no-containerized", "-c", '{"key": "val"}']
    )

    with (
        patch("karotte.hide_run_config.Path.write_text"),
        patch("karotte.hide_run_config.os.chmod"),
        patch("karotte.hide_run_config.os.execv") as mock_execv,
    ):
        hide_run_config_from_procfs()

    new_argv = mock_execv.call_args[0][1]
    assert "--config" in new_argv
    assert "-c" not in new_argv
    assert RUN_CONFIG_PATH in new_argv


def _hide(argv: list[str]) -> tuple[dict[str, str], list[str] | None]:
    """Run the hiding step on ``argv`` and return the files it wrote and the re-exec argv."""
    written: dict[str, str] = {}
    with (
        patch(
            "karotte.hide_run_config.Path.write_text",
            autospec=True,
            side_effect=lambda path, text: written.__setitem__(str(path), text),  # pyright: ignore[reportUnknownLambdaType]
        ),
        patch("karotte.hide_run_config.os.chmod"),
        patch("sys.argv", argv),
        patch("karotte.hide_run_config.os.execv") as mock_execv,
    ):
        hide_run_config_from_procfs()
    return written, mock_execv.call_args[0][1][1:] if mock_execv.called else None


@pytest.mark.usefixtures("in_container")
class TestModelApiKey:
    @pytest.fixture
    def in_container(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")

    @pytest.mark.parametrize(
        "flag", [["--model-api-key", "sk-secret"], ["--model-api-key=sk-secret"]]
    )
    def test_a_literal_key_moves_to_a_root_only_file(self, flag: list[str]):
        written, argv = _hide(["karotte", "run", "--task", "t", *flag])

        assert written == {MODEL_API_KEY_PATH: "sk-secret"}
        assert argv == [
            "karotte",
            "run",
            "--task",
            "t",
            "--model-api-key-file",
            MODEL_API_KEY_PATH,
        ]

    def test_a_variable_reference_stays(self):
        written, argv = _hide(["karotte", "run", "--model-api-key", "$MY_KEY"])

        assert written == {}
        assert argv is None

    def test_config_and_key_are_hidden_together(self):
        written, argv = _hide(
            ["karotte", "run", "-c", '{"k": 1}', "--model-api-key", "sk-secret"]
        )

        assert written == {RUN_CONFIG_PATH: '{"k": 1}', MODEL_API_KEY_PATH: "sk-secret"}
        assert argv == [
            "karotte",
            "run",
            "--config",
            RUN_CONFIG_PATH,
            "--model-api-key-file",
            MODEL_API_KEY_PATH,
        ]

    def test_an_inline_config_after_an_equals_sign_is_hidden(self):
        written, argv = _hide(["karotte", "run", '--config={"k": 1}'])

        assert written == {RUN_CONFIG_PATH: '{"k": 1}'}
        assert argv == ["karotte", "run", "--config", RUN_CONFIG_PATH]

    def test_other_subcommands_keep_their_key(self):
        _, argv = _hide(["karotte", "create-run-config", "--model-api-key", "sk-x"])

        assert argv is None
